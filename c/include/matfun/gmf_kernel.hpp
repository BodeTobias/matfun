// gmf_kernel.hpp -- matfun kernel for general (non-symmetric) real 3x3 matrices: F = f(A), J = dF/dA, H = d2F/dA2
// for f = exp, log (principal), A^alpha (principal), written once as a template over the number type T (double, or a
// GCC vector type of several matrices, smf_simd.hpp), branch-free.  Same method as python/src/matfun/_jaxgen.py
// (theory.md, "General matrices"):
//   * eigenvalues mu = m + x of A (scaled by a power of two sig), x of S = A - m I in closed form (trigonometric /
//     Cardano on the traceless deviator, normalized twice); eigenvalues far from m refined: Newton on det(A - mu I)
//     (elimination with pivoting), a pair from det A and the principal minors
//   * Newton-Hermite form over the eigenvalues (no eigenvectors, exact also for defective A):
//         F = sum_j f[N_j] w_j,  dF(E) = sum_jk f[N_j+N_k] w_j E w_k,  d2F(E1,E2) = sum_jkl f[N_j+N_k+N_l] (w_j E1 w_k E2 w_l + 1<->2)
//     w_0 = I, w_1 = S - x_1 I, w_2 = (S - x_1 I)(S - x_2 I), N_j = {x_1..x_j+1}; a complex pair through a real basis
//   * the 19 confluent divided differences f[x1^a x2^b x3^c] by residues: a group of close nodes (all three, or the
//     closest pair) by the Taylor series around its real centre (streamed over the series index, divided by the pole
//     of the node outside; the number of terms from a bound, the same for all lanes), single nodes by their Taylor
//     coefficients
// Output in the matrix layout of smf_eval_matrix: F[m] (9), J[9 m + c] (81), H[(9 m + c) 9 + e] (729), m = 3a + b.
#pragma once
#include "smf_kernel.hpp"

namespace SMF_NS {

// scalar helpers of the template (the vector versions are in smf_simd.hpp)
SMF_HD inline double m_sin(double x) { return std::sin(x); }
SMF_HD inline double m_cbrt(double x) { return std::cbrt(x); }
SMF_HD inline double m_atan2(double y, double x) { return std::atan2(y, x); }
SMF_HD inline double hmax(double x) { return x; }                  // largest lane (NaN ignored)
SMF_HD inline bool anyl(bool m) { return m; }                       // any lane set

namespace gmf {

constexpr double RHO = 0.7, R_EXP = 4.0, RHO_POLE = 0.6;   // group radius / distance to the singularity (exp: radius)
constexpr int KMAX = 200;                                   // series terms at most (ratio 0.7 needs ~ 112 for 1e-16)
constexpr double TWO_PI_3 = 2.0943951023931957;

// the divided differences f[x1^a x2^b x3^c] of the Newton form (a >= b >= c, a = order + 1), each one node more than
// its parent (-1: none) in this list: multiplicities, parent, node added; the first 3 / 9 / 19 for order 0 / 1 / 2
// (the tables TMS, TPAR, TNODE are local constants of eval: nvcc does not take namespace-scope arrays in device code)
constexpr int NDD[3] = {3, 9, 19};
// position in the list of f[N_j + N_k + ...] (multiplicities (#indices, #indices >= 1, #indices >= 2))
SMF_HD constexpr int tpos(int a, int b, int c) {
  return a == 1 ? (b + c) : (a == 2 ? (b == 0 ? 3 : (b == 1 ? 4 + c : 6 + c)) : (b == 0 ? 9 : (b == 1 ? 10 + c : (b == 2 ? 12 + c : 15 + c))));
}
SMF_HD constexpr int pos1(int j) { return tpos(1, j >= 1, j >= 2); }
SMF_HD constexpr int pos2(int j, int k) { return tpos(2, (j >= 1) + (k >= 1), (j >= 2) + (k >= 2)); }
SMF_HD constexpr int pos3(int j, int k, int l) {
  return tpos(3, (j >= 1) + (k >= 1) + (l >= 1), (j >= 2) + (k >= 2) + (l >= 2));
}

// series coefficients b_j of f(c (1 + u)) / scale: exp e^u: 1/j!; log: (-1)^(j+1)/j (b_0 apart); pow: binom(al, j)
inline void series_coef(int kind, double al, double *b) {
  b[0] = 1.0;
  for (int j = 1; j < KMAX; j++)
    b[j] = kind == EXP ? b[j - 1] / j : (kind == LOG ? (j % 2 ? 1.0 : -1.0) / j : b[j - 1] * (al - j + 1) / j);
}

// the table of series_coef, computed once: exp and log fixed, pow per thread for the last exponent used (host code)
struct SeriesTable {
  double b[KMAX];
};
inline const double *series_table(int kind, double al) {
  static const SeriesTable te = [] { SeriesTable t; series_coef(EXP, 0.0, t.b); return t; }();
  static const SeriesTable tl = [] { SeriesTable t; series_coef(LOG, 0.0, t.b); return t; }();
  if (kind == EXP) return te.b;
  if (kind == LOG) return tl.b;
  thread_local SeriesTable tp;
  thread_local double tp_al = -1.0;              // (al > 0 for pow)
  if (!(al == tp_al)) {
    series_coef(POW, al, tp.b);
    tp_al = al;
  }
  return tp.b;
}

// terms K of the group series: sum_k b_(n+k) h_k with |h_k| <= C(k+n, n) r^k (r: radius / centre), and the series of
// the pole (multiplicity <= 3: C(k+2, 2) q^k, q: radius / pole distance), until a bound < 1e-17 x the largest
SMF_HD inline int nterms(const double *b, int n, double r, double q) {
  int M = 1;
  if (r > 0) {
    double best = 0.0, c = 1.0, rk = 1.0;
    for (int k = 0; n + k < KMAX; k++) {
      double t = m_abs(b[n + k]) * c * rk;
      best = t > best ? t : best;
      M = k;
      if ((k > 2 && t <= 1e-17 * best) || !(rk > 0)) break;
      c = c * (k + n + 1) / (k + 1);
      rk *= r;
    }
  }
  if (q > 0) {
    double qk = 1.0;
    for (int k = 0; n + k < KMAX; k++) {
      if (k > M && (k + 1.0) * (k + 2.0) / 2.0 * qk <= 1e-17) break;
      M = k > M ? k : M;
      qk *= q;
    }
  }
  return n + M + 1 < KMAX ? n + M + 1 : KMAX;
}

// ------------------------------------------------------------------ complex numbers as (re, im)
template <class T>
struct cx {
  T re, im;
};
template <class T> SMF_INL cx<T> cadd(cx<T> a, cx<T> b) { return {a.re + b.re, a.im + b.im}; }
template <class T> SMF_INL cx<T> csub(cx<T> a, cx<T> b) { return {a.re - b.re, a.im - b.im}; }
template <class T> SMF_INL cx<T> cmul(cx<T> a, cx<T> b) { return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }
template <class T> SMF_INL cx<T> cscale(cx<T> a, T s) { return {a.re * s, a.im * s}; }
template <class T, class M> SMF_INL cx<T> csel(M m, cx<T> a, cx<T> b) { return {sel(m, a.re, b.re), sel(m, a.im, b.im)}; }
template <class T> SMF_INL T tmax(T a, T b) { return sel(a > b, a, b); }
template <class T> SMF_INL T tmin(T a, T b) { return sel(a < b, a, b); }
template <class T> SMF_INL T copysgn(T x, T y) { return sel(y < 0, -m_abs(x), m_abs(x)); }
template <class T>
SMF_INL T cabs(cx<T> a) {                                   // |a| without overflow / underflow of the squares
  T ar = m_abs(a.re), ai = m_abs(a.im), big = tmax(ar, ai), sml = tmin(ar, ai);
  T bs = sel(big > 0, big, 1.0), q = sml / bs;
  return sel(big > 0, big * m_sqrt(1.0 + q * q), 0.0);
}
template <class T>
SMF_INL cx<T> cinv(cx<T> b) {                               // 1 / b (Smith)
  auto sw = m_abs(b.im) > m_abs(b.re);
  T p = sel(sw, b.im, b.re), q = sel(sw, b.re, b.im);
  T r = q / sel(p != 0, p, 1.0), den = p + q * r;
  T re = 1.0 / den, im = -r / den;
  return {sel(sw, -im, re), sel(sw, -re, im)};
}
// a power of two within a factor 2 of x > 0 (1 for 0, subnormal, NaN, Inf)
template <class T>
SMF_INL T pow2safe(T x) {
  auto ok = (x > 1e-300) & (x < 1e300);
  return sel(ok, m_pow2floor(sel(ok, x, 1.0)), 1.0);
}

template <class T>
SMF_INL T det3(const T *M) {
  return M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) + M[2] * (M[3] * M[7] - M[4] * M[6]);
}

// det M by elimination with partial pivoting: backward stable, exact pivots for diagonal M
template <class T>
SMF_INL T det_lu(const T *M) {
  T a0 = m_abs(M[0]), a1 = m_abs(M[3]), a2 = m_abs(M[6]);
  auto p1 = (a1 > a0) & (a1 >= a2), p2 = (a2 > a0) & (a2 > a1);
  T P[3], R1[3], R2[3];
SMF_UNROLL(3)
  for (int c = 0; c < 3; c++) {                             // pivot row; (2, 0, 1) is an even permutation
    P[c] = sel(p1, M[3 + c], sel(p2, M[6 + c], M[c]));
    R1[c] = sel(p1 | p2, M[c], M[3 + c]);
    R2[c] = sel(p2, M[3 + c], M[6 + c]);
  }
  T sgn = sel(p1, -1.0, 1.0);
  T piv = sel(P[0] != 0, P[0], 1.0);
  T l1 = R1[0] / piv, l2 = R2[0] / piv;
  T b = R1[1] - l1 * P[1], c = R1[2] - l1 * P[2], d = R2[1] - l2 * P[1], e = R2[2] - l2 * P[2];
  auto sw = m_abs(d) > m_abs(b);
  T bb = sel(sw, d, b), cc = sel(sw, e, c), dd = sel(sw, b, d), ee = sel(sw, c, e);
  sgn = sel(sw, -sgn, sgn);
  T det = sgn * P[0] * bb * (ee - dd / sel(bb != 0, bb, 1.0) * cc);
  return sel((P[0] != 0) & (bb != 0), det, 0.0);
}

template <class T>
SMF_INL T minors2(const T *M) {
  return M[0] * M[4] - M[1] * M[3] + M[0] * M[8] - M[2] * M[6] + M[4] * M[8] - M[5] * M[7];
}

// eigenvalues of S: tau + those of the traceless S - tau I (tau = tr S / 3, the rounding residue of the shift),
// normalized by max |S| and by the size of the roots, from x^3 - P x - Q = 0: x[0] the real root farthest from the
// others (trigonometric formula) or the only real one (Cardano), x[1], x[2] from the deflated quadratic (a real pair
// or a conjugate pair)
template <class T>
SMF_INL void roots3(const T *S, cx<T> *x) {
  T tau = (S[0] + S[4] + S[8]) / 3.0;
  T Sp[9];
SMF_UNROLL(9)
  for (int k = 0; k < 9; k++) Sp[k] = k % 4 == 0 ? S[k] - tau : S[k];
  T s = m_abs(Sp[0]);
SMF_UNROLL(8)
  for (int k = 1; k < 9; k++) s = tmax(s, m_abs(Sp[k]));
  T t = pow2safe(s), it = 1.0 / t;
SMF_UNROLL(9)
  for (int k = 0; k < 9; k++) Sp[k] = Sp[k] * it;
  T P = 0.5 * (Sp[0] * Sp[0] + Sp[4] * Sp[4] + Sp[8] * Sp[8]) + Sp[1] * Sp[3] + Sp[2] * Sp[6] + Sp[5] * Sp[7];
  T Q = det3(Sp);
  T u = pow2safe(tmax(m_sqrt(m_abs(P)), m_cbrt(m_abs(Q)))), iu = 1.0 / u;
  P = P * iu * iu;
  Q = Q * iu * iu * iu;
  t = t * u;
  auto trig = (P > 0) & (4.0 * P * P * P >= 27.0 * Q * Q);
  T Ps = sel(P > 0, P, 1.0), p3 = Ps / 3.0;
  T r = 0.5 * Q / (p3 * m_sqrt(p3));
  r = sel(r > 1.0, 1.0, sel(r < -1.0, -1.0, r));
  T ph = m_acos(r) / 3.0;
  T s1t = 2.0 * m_sqrt(p3) * m_cos(sel(r >= 0, ph, ph + TWO_PI_3));
  T D = 0.25 * Q * Q - P * P * P / 27.0;
  T uc = m_cbrt(0.5 * Q + copysgn(m_sqrt(sel(D > 0, D, 0.0)), Q));
  T s1c = uc + sel(uc != 0, P / (3.0 * sel(uc != 0, uc, 1.0)), 0.0);
  T s1 = sel(trig, s1t, s1c);
  T d = P - 0.75 * s1 * s1;                                 // x^2 + s1 x + (s1^2 - P) = 0
  auto real = d >= 0;
  T sq = m_sqrt(m_abs(d));
  T sa = -0.5 * s1 - copysgn(sq, s1);
  T prod = sel(s1 != 0, Q / sel(s1 != 0, s1, 1.0), -P);
  T sb = sel(sa != 0, prod / sel(sa != 0, sa, 1.0), 0.0);
  T im = sel(real, 0.0, sq) * t;
  x[0] = {s1 * t + tau, 0.0 * t};
  x[1] = {sel(real, sa, -0.5 * s1) * t + tau, im};
  x[2] = {sel(real, sb, -0.5 * s1) * t + tau, -im};
}

// real node j: 3 Newton steps on det(A - mu I), derivative from the other nodes (where mask)
template <class T, class M>
SMF_INL void newton(const T *As, cx<T> *mu, int j, M mask) {
SMF_UNROLL(3)
  for (int it = 0; it < 3; it++) {
    cx<T> den = {1.0 + 0.0 * As[0], 0.0 * As[0]};
SMF_UNROLL(3)
    for (int k = 0; k < 3; k++)
      if (k != j) den = cmul(den, csub(mu[k], mu[j]));
    auto ok = mask & ((den.re != 0) | (den.im != 0));
    T B[9];
SMF_UNROLL(9)
    for (int k = 0; k < 9; k++) B[k] = k % 4 == 0 ? As[k] - mu[j].re : As[k];
    T step = det_lu(B) * cinv(den).re;                      // Re(det / den), det real
    mu[j].re = sel(ok, mu[j].re + step, mu[j].re);
  }
}

// f(mu) (k: order of the output; pow: times sig^(al - k)), f(t) = e^t (x: relative to the shift), log(sig t) (log1p
// of x/m near m), (sig t)^al
template <int KIND, class T>
SMF_INL cx<T> coef0(double al, T sig, T m, cx<T> x, cx<T> mu, int k) {
  if (KIND == EXP) {
    T e = m_exp(x.re);
    return {e * m_cos(x.im), e * m_sin(x.im)};
  }
  T th = m_atan2(mu.im, mu.re);
  if (KIND == LOG) {
    auto near = (m > 0) & (cabs(x) <= 0.5 * m);
    T ms = sel(m > 0, m, 1.0), zr = x.re / ms, zi = x.im / ms;
    T l1r = 0.5 * m_log1p(zr * (2.0 + zr) + zi * zi), l1i = m_atan2(zi, 1.0 + zr);
    return {sel(near, m_log(sig * ms) + l1r, m_log(sig * cabs(mu))), sel(near, l1i, th)};
  }
  T mag = pow_k(sig * cabs(mu), al, (double)k);
  T ph = (al - k) * th;
  cx<T> out = {mag * m_cos(ph), mag * m_sin(ph)};
  for (int i = 0; i < k; i++) out = cmul(out, mu);
  return out;
}

// ------------------------------------------------------------------ the kernel
template <int KIND, int ORDER, class T>
SMF_INL void eval(const T *a, T *F, T *J, T *H, double al, const double *bt) {
  constexpr int ND = NDD[ORDER];
  constexpr int TMS[19][3] = {{1, 0, 0}, {1, 1, 0}, {1, 1, 1}, {2, 0, 0}, {2, 1, 0}, {2, 1, 1}, {2, 2, 0},
                              {2, 2, 1}, {2, 2, 2}, {3, 0, 0}, {3, 1, 0}, {3, 1, 1}, {3, 2, 0}, {3, 2, 1},
                              {3, 2, 2}, {3, 3, 0}, {3, 3, 1}, {3, 3, 2}, {3, 3, 3}};
  constexpr int TPAR[19] = {-1, 0, 1, 0, 1, 2, 4, 5, 7, 3, 4, 5, 6, 7, 8, 12, 13, 14, 17};
  constexpr int TNODE[19] = {0, 1, 2, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2};
  const T zero = 0.0 * a[0], one = zero + 1.0;
  using Msk = decltype(zero > zero);
  // ---- scaling (exact power of two; exp: none), shift m, deviator S
  T amax = m_abs(a[0]);
SMF_UNROLL(8)
  for (int k = 1; k < 9; k++) amax = tmax(amax, m_abs(a[k]));
  const T sig = KIND == EXP ? one : pow2safe(amax), isig = 1.0 / sig;
  T As[9], S[9];
SMF_UNROLL(9)
  for (int k = 0; k < 9; k++) As[k] = a[k] * isig;
  const T m = (As[0] + As[4] + As[8]) / 3.0;
SMF_UNROLL(9)
  for (int k = 0; k < 9; k++) S[k] = k % 4 == 0 ? As[k] - m : As[k];
  cx<T> x[3], mu[3];
  roots3(S, x);
SMF_UNROLL(3)
  for (int i = 0; i < 3; i++) mu[i] = {m + x[i].re, x[i].im};

  // ---- the pair x[1], x[2] from its sum and product with A itself, after mu[0] by Newton, where the roots of S are
  //      not the more accurate ones (the pair loses eps |S|^2 / gap, the shift by m eps m): the pair small against
  //      mu[0], or the deviator not dominated by the shift and mu[0] separated from the pair
  T smax = m_abs(S[0]);
SMF_UNROLL(8)
  for (int k = 1; k < 9; k++) smax = tmax(smax, m_abs(S[k]));
  T mid = 0.5 * (mu[1].re + mu[2].re), Rp = 0.5 * cabs(csub(mu[1], mu[2]));
  auto small = m_abs(mid) + Rp <= 0.25 * m_abs(mu[0].re);
  auto aniso = (m_abs(m) <= smax) & (Rp <= 0.25 * m_abs(mid - mu[0].re));
  Msk pc = (mu[0].re != 0) & (small | aniso);
  T pcen = zero, ps = zero;
  Msk preal = zero == zero;
  if (anyl(pc)) {                                           // (skipped if no lane needs it)
    newton(As, mu, 0, pc);
    T m0 = sel(pc, mu[0].re, 1.0);
    auto big = m_abs(m0) >= m_abs(mid) + Rp;                // pi = det A / mu0, sigma = (c2 - pi) / mu0
    T pid = det_lu(As) / m0, c2 = minors2(As);
    T sgt = As[0] + As[4] + As[8] - m0;                     // else sigma = tr A - mu0, pi = c2 - mu0 sigma
    T sg = sel(big, (c2 - pid) / m0, sgt), pi = sel(big, pid, c2 - m0 * sgt);
    T disc = 0.25 * sg * sg - pi;
    pcen = 0.5 * sg;
    ps = m_sqrt(m_abs(disc));
    preal = disc >= 0;
    T r1 = pcen + copysgn(ps, pcen);
    T r2 = sel(r1 != 0, pi / sel(r1 != 0, r1, 1.0), pcen - copysgn(ps, pcen));
    mu[1] = csel(pc, cx<T>{sel(preal, r1, pcen), sel(preal, 0.0, ps)}, mu[1]);
    mu[2] = csel(pc, cx<T>{sel(preal, r2, pcen), sel(preal, 0.0, -ps)}, mu[2]);
  }
  Msk ab[3] = {pc, pc, pc};
  T og[3] = {zero, one, one + 1.0};
SMF_UNROLL(3)
  for (int i = 0; i < 3; i++) x[i] = csel(ab[i], cx<T>{mu[i].re - m, mu[i].im}, x[i]);

  // ---- node order: the node with the largest high derivatives last (exp: largest real part; log, pow: smallest
  //      modulus); a conjugate pair stays adjacent (real node first or last)
  auto key = [&](cx<T> v) { return KIND == EXP ? v.re : -cabs(v); };
  {
    cx<T> sx[3] = {x[0], x[1], x[2]}, sm[3] = {mu[0], mu[1], mu[2]};
    Msk sa[3] = {ab[0], ab[1], ab[2]};
    T so[3] = {og[0], og[1], og[2]};
    const int NET[3][2] = {{0, 1}, {1, 2}, {0, 1}};           // stable sorting network (real triple)
SMF_UNROLL(3)
    for (int s = 0; s < 3; s++) {
      const int i = NET[s][0], j = NET[s][1];
      auto sw = key(sm[i]) > key(sm[j]);
      cx<T> tx = sx[i], tm = sm[i];
      Msk ta = sa[i];
      T to = so[i];
      sx[i] = csel(sw, sx[j], sx[i]); sx[j] = csel(sw, tx, sx[j]);
      sm[i] = csel(sw, sm[j], sm[i]); sm[j] = csel(sw, tm, sm[j]);
      sa[i] = (sw & sa[j]) | (mnot(sw) & sa[i]); sa[j] = (sw & ta) | (mnot(sw) & sa[j]);
      so[i] = sel(sw, so[j], so[i]); so[j] = sel(sw, to, so[j]);
    }
    auto first = key(mu[0]) <= key(mu[1]);                    // complex pair: real node first or last
    auto cplx = x[1].im != 0;
    cx<T> nx[3], nm[3];
    Msk na[3];
    T no[3];
SMF_UNROLL(3)
    for (int k = 0; k < 3; k++) {
      const int r = (k + 1) % 3;
      nx[k] = csel(cplx, csel(first, x[k], x[r]), sx[k]);
      nm[k] = csel(cplx, csel(first, mu[k], mu[r]), sm[k]);
      na[k] = (cplx & ((first & ab[k]) | (mnot(first) & ab[r]))) | (mnot(cplx) & sa[k]);
      no[k] = sel(cplx, sel(first, og[k], og[r]), so[k]);
    }
SMF_UNROLL(3)
    for (int k = 0; k < 3; k++) { x[k] = nx[k]; mu[k] = nm[k]; ab[k] = na[k]; og[k] = no[k]; }
  }

  // ---- groups: all three, else the closest real / conjugate pair (with the pole of the third node), else none
  const double rho_f = KIND == POW && 3.0 / al < RHO ? 3.0 / al : RHO;
  auto ok_f = [&](T c, T R) -> Msk {
    if (KIND == EXP) return R <= R_EXP;
    return (m + c > 0) & (R <= rho_f * (m + c));
  };
  T cs = (x[0].re + x[1].re + x[2].re) / 3.0;
  T R3 = tmax(tmax(cabs(cx<T>{x[0].re - cs, x[0].im}), cabs(cx<T>{x[1].re - cs, x[1].im})),
              cabs(cx<T>{x[2].re - cs, x[2].im}));
  Msk triple = ok_f(cs, R3);
  Msk psel[3];                                              // pairs (0,1), (0,2), (1,2)
  T bestR = zero + INFINITY;
  const int PI_[3] = {0, 0, 1}, PJ_[3] = {1, 2, 2};
SMF_UNROLL(3)
  for (int p = 0; p < 3; p++) {
    const int i = PI_[p], j = PJ_[p], k = 3 - i - j;
    auto elig = ((x[i].im == 0) & (x[j].im == 0)) | ((x[i].re == x[j].re) & (x[i].im == -x[j].im));
    T c = 0.5 * (x[i].re + x[j].re), R = 0.5 * cabs(csub(x[i], x[j]));
    auto okp = elig & ok_f(c, R) & (R <= RHO_POLE * cabs(cx<T>{c - x[k].re, -x[k].im})) & mnot(triple);
    Msk better = okp & (R < bestR);
    bestR = sel(better, R, bestR);
    for (int q = 0; q < p; q++) psel[q] = psel[q] & mnot(better);
    psel[p] = better;
  }
  Msk inG[3] = {Msk(triple | psel[0] | psel[1]), Msk(triple | psel[0] | psel[2]), Msk(triple | psel[1] | psel[2])};
  Msk pk[3] = {psel[2], psel[1], psel[0]};                  // outside node of the pair group

  // ---- group centre and offsets; the refined pair as a group: symmetric offsets around its centre
  T gsum = sel(inG[0], 1.0, 0.0) + sel(inG[1], 1.0, 0.0) + sel(inG[2], 1.0, 0.0);
  T cxg = (sel(inG[0], x[0].re, 0.0) + sel(inG[1], x[1].re, 0.0) + sel(inG[2], x[2].re, 0.0)) / sel(gsum > 0, gsum, 1.0);
  T cmu = m + cxg;
  Msk is12[3];
SMF_UNROLL(3)
  for (int i = 0; i < 3; i++) is12[i] = og[i] > 0.5;
  Msk ovr = pc & (gsum == 2.0) & ((inG[0] & inG[1] & is12[0] & is12[1]) | (inG[0] & inG[2] & is12[0] & is12[2]) |
                                  (inG[1] & inG[2] & is12[1] & is12[2]));
  cxg = sel(ovr, pcen - m, cxg);
  cmu = sel(ovr, pcen, cmu);
  cx<T> dpair = {sel(preal, ps, 0.0), sel(preal, 0.0, ps)};
  cx<T> off[3];
SMF_UNROLL(3)
  for (int i = 0; i < 3; i++) {
    auto neg = (mu[i].re - pcen < 0) | (mu[i].im < 0);
    cx<T> di = csel(neg, cx<T>{-dpair.re, -dpair.im}, dpair);
    Msk ovi = ovr & inG[i];
    mu[i] = csel(ovi, cx<T>{pcen + di.re, di.im}, mu[i]);
    x[i] = csel(ovi, cx<T>{mu[i].re - m, mu[i].im}, x[i]);
    off[i] = csel(inG[i], csel(ovi, di, cx<T>{x[i].re - cxg, x[i].im}), cx<T>{zero, zero});
  }

  // ---- single real nodes far from m: Newton on det(A - mu I)
SMF_UNROLL(3)
  for (int i = 0; i < 3; i++) {
    Msk mask = mnot(inG[i]) & mnot(ab[i]) & (x[i].im == 0) & (m_abs(x[i].re) > 0.5 * m_abs(m));
    if (anyl(mask)) newton(As, mu, i, mask);
    ab[i] = ab[i] | mask;
    x[i] = csel(mask, cx<T>{mu[i].re - m, mu[i].im}, x[i]);
  }
  Msk absc = (inG[0] & ab[0]) | (inG[1] & ab[1]) | (inG[2] & ab[2]);
  T apole = zero;                                           // pole distance of the node outside a pair group
SMF_UNROLL(3)
  for (int k = 0; k < 3; k++) apole = sel(pk[k], sel(absc | ab[k], cmu - mu[k].re, cxg - x[k].re), apole);
  const T xshift = KIND == EXP ? tmax(tmax(x[0].re, x[1].re), x[2].re) : zero;
  Msk bad = KIND == EXP ? (zero != zero) : (((x[0].im == 0) & (mu[0].re <= 0)) | ((x[1].im == 0) & (mu[1].re <= 0)) |
                                           ((x[2].im == 0) & (mu[2].re <= 0)));

  // ---- divided differences: group part, the Taylor series of the group streamed over the series index j,
  //      sum_j G_j e_j with G = b / (a + u)^(multiplicity of the outside node), e^(ms) the dual vectors of the
  //      group nodes (e^(child)_j = e^(parent)_(j-1) + u e^(child)_(j-1) if its node is in the group)
  Msk hasg = inG[0] | inG[1] | inG[2];
  const T L = KIND == EXP ? one : sel(hasg & (cmu > 0), cmu, 1.0);     // normalization of the series variable
  const T iL = 1.0 / L;
  cx<T> u[3];
SMF_UNROLL(3)
  for (int i = 0; i < 3; i++) u[i] = cscale(off[i], iL);
  const T ap = sel(m_abs(apole) > 0, apole * iL, 1.0), iap = 1.0 / ap;
  // terms: the same for all lanes (largest radius ratio of the lanes with a group)
  T rr = tmax(cabs(u[0]), tmax(cabs(u[1]), cabs(u[2])));
  T rq = sel((pk[0] | pk[1] | pk[2]) & (rr * m_abs(iap) <= 1.0), rr * m_abs(iap), 0.0);
  rr = sel(hasg & (rr <= R_EXP + 0.5), rr, 0.0);           // (not finite: ignored, so that other lanes keep K)
  const int K = nterms(bt, 3 * (ORDER + 1) - 1, hmax(rr), hmax(rq));
  T b0 = one;
  if (KIND == LOG) b0 = coef0<LOG>(al, sig, m, cx<T>{cxg, zero}, cx<T>{cmu, zero}, 0).re;
  cx<T> Ep[ND], Ec[ND], acc[ND];
  const cx<T> czero = {zero, zero}, cone = {one, zero};
SMF_UNROLL(19)
  for (int c = 0; c < ND; c++) {
    Ep[c] = csel(inG[TNODE[c]], czero, TPAR[c] < 0 ? cone : Ep[TPAR[c]]);   // at j = -1
    acc[c] = czero;
  }
  T q1 = zero, q2 = zero, q3 = zero;                        // G^(r)_(j-1) of f / (a + u)^r
  for (int j = 0; j < K; j++) {
    const T bj = j == 0 ? b0 : zero + bt[j];
    const T g1 = (bj - q1) * iap, g2 = (g1 - q2) * iap, g3 = (g2 - q3) * iap;
    q1 = g1; q2 = g2; q3 = g3;
    const T q[4] = {bj, g1, g2, g3};
SMF_UNROLL(19)
    for (int c = 0; c < ND; c++) {
      const int par = TPAR[c], i = TNODE[c];
      cx<T> epp = par < 0 ? (j == 0 ? cone : czero) : Ep[par];      // parent at j - 1
      cx<T> epc = par < 0 ? czero : Ec[par];                       // parent at j
      Ec[c] = csel(inG[i], cadd(epp, cmul(u[i], Ep[c])), epc);
      T g = sel(pk[0], q[TMS[c][0]], sel(pk[1], q[TMS[c][1]], sel(pk[2], q[TMS[c][2]], q[0])));
      acc[c] = cadd(acc[c], cscale(Ec[c], g));
    }
SMF_UNROLL(19)
    for (int c = 0; c < ND; c++) Ep[c] = Ec[c];
  }
  cx<T> D[ND];
SMF_UNROLL(19)
  for (int c = 0; c < ND; c++) {
    const int n = TMS[c][0] + TMS[c][1] + TMS[c][2] - 1, k = TMS[c][0] - 1;
    T fac;
    if (KIND == EXP) {
      fac = m_exp(cxg - xshift);
    } else {
      T iLn = one;
      for (int r = 0; r < n - (KIND == POW ? k : 0); r++) iLn = iLn * iL;
      fac = KIND == LOG ? iLn : pow_k(sig * L, al, (double)k) * iLn;
    }
    D[c] = csel(hasg, cscale(acc[c], fac), czero);
  }
  // ---- residues of the single nodes i: coefficient of (t - mu_i)^(m_i - 1) in f(t) / prod_(z != i) (t - mu_z)^m_z
SMF_UNROLL(3)
  for (int i = 0; i < 3; i++) {
    Msk single = mnot(inG[i]);
    if (!anyl(single)) continue;                             // (all lanes: node i in the group)
    cx<T> xs = KIND == EXP ? cx<T>{x[i].re - xshift, x[i].im} : x[i];
    cx<T> f0[3];
    f0[0] = coef0<KIND>(al, sig, m, xs, mu[i], 0);
    if (KIND == POW && ORDER >= 1) f0[1] = coef0<KIND>(al, sig, m, xs, mu[i], 1);
    if (KIND == POW && ORDER >= 2) f0[2] = coef0<KIND>(al, sig, m, xs, mu[i], 2);
    cx<T> imu = cinv(mu[i]), w[3];
SMF_UNROLL(3)
    for (int z = 0; z < 3; z++)
      if (z != i) w[z] = cinv(csel(ab[i] | ab[z], csub(mu[i], mu[z]), csub(x[i], x[z])));
SMF_UNROLL(19)
    for (int c = 0; c < ND; c++) {
      const int mi = TMS[c][i];
      if (mi == 0) continue;
      const int k = TMS[c][0] - 1;
      cx<T> fb = KIND == POW ? f0[k] : f0[0], ser[3];
      if (KIND == EXP) {
        ser[0] = fb; ser[1] = fb; ser[2] = cscale(fb, zero + 0.5);
      } else if (KIND == LOG) {
        ser[0] = fb; ser[1] = imu; ser[2] = cscale(cmul(imu, imu), zero - 0.5);
      } else {
        ser[0] = fb;
        ser[1] = cscale(cmul(fb, imu), zero + al);
        ser[2] = cscale(cmul(ser[1], imu), zero + (al - 1.0) / 2.0);
      }
SMF_UNROLL(3)
      for (int z = 0; z < 3; z++) {                           // times the series of 1 / (d + tau)^m_z
        const int mz = TMS[c][z];
        if (z == i || mz == 0) continue;
        cx<T> wp = cone;
        for (int r = 0; r < mz; r++) wp = cmul(wp, w[z]);
        cx<T> p1 = cscale(cmul(wp, w[z]), zero - mz), p2 = cscale(cmul(cmul(wp, w[z]), w[z]), zero + mz * (mz + 1) / 2.0);
        cx<T> n0 = cmul(ser[0], wp), n1 = cadd(cmul(ser[1], wp), cmul(ser[0], p1));
        cx<T> n2 = cadd(cadd(cmul(ser[2], wp), cmul(ser[1], p1)), cmul(ser[0], p2));
        ser[0] = n0; ser[1] = n1; ser[2] = n2;
      }
      D[c] = csel(single, cadd(D[c], ser[mi - 1]), D[c]);
    }
  }

  // ---- real basis: w = T r, r_0 = I, r_1 = B_0, r_2 = B_0 B_1 - b0 b1 I, B_j = S - Re x_j I (or A - Re mu_j I),
  //      T = I + i N (N strictly lower), D' = Re(T^T D T ...)
  const T b0i = x[0].im, b1i = x[1].im;
  T B0[9], B1[9], r[3][9];
SMF_UNROLL(9)
  for (int k = 0; k < 9; k++) {
    const bool dg = k % 4 == 0;
    B0[k] = dg ? sel(ab[0], As[k] - mu[0].re, S[k] - x[0].re) : S[k];
    B1[k] = dg ? sel(ab[1], As[k] - mu[1].re, S[k] - x[1].re) : S[k];
    r[0][k] = dg ? one : zero;
    r[1][k] = B0[k];
  }
SMF_UNROLL(3)
  for (int i2 = 0; i2 < 3; i2++)
SMF_UNROLL(3)
    for (int j2 = 0; j2 < 3; j2++)
      r[2][3 * i2 + j2] = B0[3 * i2] * B1[j2] + B0[3 * i2 + 1] * B1[3 + j2] + B0[3 * i2 + 2] * B1[6 + j2] -
                          (i2 == j2 ? b0i * b1i : zero);
  const T a01 = sel(ab[0] | ab[1], mu[0].re - mu[1].re, x[0].re - x[1].re);
  const T N10 = -b0i, N20 = -b0i * a01, N21 = -(b0i + b1i);
  // (T^T v)_m = v_m + i (N^T v)_m on one index: (N^T v)_0 = N10 v_1 + N20 v_2, (N^T v)_1 = N21 v_2, (N^T v)_2 = 0
  auto nt = [&](const T *v, T *o, int s) { o[0] = N10 * v[s] + N20 * v[2 * s]; o[s] = N21 * v[2 * s]; o[2 * s] = zero; };
  const T nanf = sel(bad, zero + NAN, one);
  {
    T Dr[3], Di[3], Nr[3], Ni[3];
SMF_UNROLL(3)
    for (int j = 0; j < 3; j++) { Dr[j] = D[pos1(j)].re; Di[j] = D[pos1(j)].im; }
    nt(Dr, Nr, 1); nt(Di, Ni, 1);
    T e = KIND == EXP ? m_exp(m + xshift) : one;
SMF_UNROLL(9)
    for (int k = 0; k < 9; k++) F[k] = ((Dr[0] - Ni[0]) * r[0][k] + (Dr[1] - Ni[1]) * r[1][k] + (Dr[2] - Ni[2]) * r[2][k]) * e * nanf;
  }
  if (ORDER < 1) return;
  const T sc1 = (KIND == EXP ? m_exp(m + xshift) : (KIND == LOG ? isig : one)) * nanf;
  T D2[9];
  {
    T Dr[9], Di[9];
SMF_UNROLL(3)
    for (int j = 0; j < 3; j++)
SMF_UNROLL(3)
      for (int k = 0; k < 3; k++) { Dr[3 * j + k] = D[pos2(j, k)].re; Di[3 * j + k] = D[pos2(j, k)].im; }
    // apply T on index 0 (stride 3) and on index 1 (stride 1): (Dr, Di) <- (Dr - N^T Di, Di + N^T Dr)
SMF_UNROLL(3)
    for (int k = 0; k < 3; k++) {
      T vr[3] = {Dr[k], Dr[3 + k], Dr[6 + k]}, vi[3] = {Di[k], Di[3 + k], Di[6 + k]}, nr[3], ni[3];
      nt(vr, nr, 1); nt(vi, ni, 1);
SMF_UNROLL(3)
      for (int j = 0; j < 3; j++) { Dr[3 * j + k] = vr[j] - ni[j]; Di[3 * j + k] = vi[j] + nr[j]; }
    }
SMF_UNROLL(3)
    for (int j = 0; j < 3; j++) {
      T nr[3], ni[3];
      nt(Di + 3 * j, ni, 1);
SMF_UNROLL(3)
      for (int k = 0; k < 3; k++) D2[3 * j + k] = Dr[3 * j + k] - ni[k];
      (void)nr;
    }
  }
  // J_abcd = sum_mn D2_mn r_m[a][c] r_n[d][b]
  T Y[3][9];
SMF_UNROLL(3)
  for (int mm = 0; mm < 3; mm++)
SMF_UNROLL(9)
    for (int k = 0; k < 9; k++) Y[mm][k] = D2[3 * mm] * r[0][k] + D2[3 * mm + 1] * r[1][k] + D2[3 * mm + 2] * r[2][k];
  for (int aa = 0; aa < 3; aa++)
    for (int bb = 0; bb < 3; bb++)
      for (int cc = 0; cc < 3; cc++)
        for (int dd = 0; dd < 3; dd++)
          J[((3 * aa + bb) * 3 + cc) * 3 + dd] = (r[0][3 * aa + cc] * Y[0][3 * dd + bb] + r[1][3 * aa + cc] * Y[1][3 * dd + bb] +
                                                  r[2][3 * aa + cc] * Y[2][3 * dd + bb]) * sc1;
  if (ORDER < 2) return;
  const T sc2 = (KIND == EXP ? m_exp(m + xshift) : (KIND == LOG ? isig * isig : one)) * nanf;
  T D3[27];
  {
    T Dr[27], Di[27];
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++)
        for (int l = 0; l < 3; l++) {
          Dr[9 * j + 3 * k + l] = D[pos3(j, k, l)].re;
          Di[9 * j + 3 * k + l] = D[pos3(j, k, l)].im;
        }
    const int STR[3] = {9, 3, 1};
    for (int ax = 0; ax < 3; ax++) {                         // T on each index
      const int s = STR[ax];
      for (int base = 0; base < 27; base++) {
        if ((base / s) % 3 != 0) continue;                   // base: index ax = 0
        T vr[3] = {Dr[base], Dr[base + s], Dr[base + 2 * s]}, vi[3] = {Di[base], Di[base + s], Di[base + 2 * s]};
        T nr[3], ni[3];
        nt(vr, nr, 1); nt(vi, ni, 1);
        for (int j = 0; j < 3; j++) { Dr[base + j * s] = vr[j] - ni[j]; Di[base + j * s] = vi[j] + nr[j]; }
      }
    }
    for (int k = 0; k < 27; k++) D3[k] = Dr[k];
  }
  // T6_acdefb = sum_mnp D3_mnp r_m[a][c] r_n[d][e] r_p[f][b];  H_abcdef = T6_acdefb + T6_aefcdb
  T Z[9][9];                                                 // Z[mn][fb] = sum_p D3_mnp r_p[f][b]
  for (int mn = 0; mn < 9; mn++)
    for (int k = 0; k < 9; k++) Z[mn][k] = D3[3 * mn] * r[0][k] + D3[3 * mn + 1] * r[1][k] + D3[3 * mn + 2] * r[2][k];
  T Yh[3][81];                                               // Yh[m][(de) fb] = sum_n r_n[d][e] Z[mn][fb]
  for (int mm = 0; mm < 3; mm++)
    for (int de = 0; de < 9; de++)
      for (int fb = 0; fb < 9; fb++)
        Yh[mm][9 * de + fb] = r[0][de] * Z[3 * mm][fb] + r[1][de] * Z[3 * mm + 1][fb] + r[2][de] * Z[3 * mm + 2][fb];
  for (int aa = 0; aa < 3; aa++)
    for (int bb = 0; bb < 3; bb++)
      for (int cc = 0; cc < 3; cc++)
        for (int dd = 0; dd < 3; dd++)
          for (int ee = 0; ee < 3; ee++)
            for (int ff = 0; ff < 3; ff++) {
              const int i1 = 9 * (3 * dd + ee) + 3 * ff + bb, i2 = 9 * (3 * ff + cc) + 3 * dd + bb;
              T h = r[0][3 * aa + cc] * Yh[0][i1] + r[1][3 * aa + cc] * Yh[1][i1] + r[2][3 * aa + cc] * Yh[2][i1] +
                    r[0][3 * aa + ee] * Yh[0][i2] + r[1][3 * aa + ee] * Yh[1][i2] + r[2][3 * aa + ee] * Yh[2][i2];
              H[((((3 * aa + bb) * 3 + cc) * 3 + dd) * 3 + ee) * 3 + ff] = h * sc2;
            }
}

}  // namespace gmf
}  // namespace SMF_NS
