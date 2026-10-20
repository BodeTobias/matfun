// smf_kernel.hpp -- matfun 3x3 kernel (value and derivatives w.r.t. the Voigt components), written once as a
// template over the number type T:  T = double (one matrix)  or  a GCC vector type (several matrices at once,
// see smf_simd.hpp).  Plus the conversions for full-matrix input and output (from_matrix, to_matrix).
// Same algorithm as python/src/matfun/_jaxpure.py (_eig3, _f, _dd), but branch-free (selects instead of if), and the 2x2 Jacobi
// rotation takes cos/sin of theta = atan2(2 a12, a11 - a22) / 2 from half-angle formulas instead of atan2/cos/sin.
#pragma once
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define SMF_HD __host__ __device__
#else
#define SMF_HD
#endif

// loop unrolling: nvcc ignores "#pragma GCC unroll" (the small arrays V, P, Q, D2, Et then end up in local memory,
// ptxas: ~940 bytes stack frame per thread instead of <= 40), so CUDA gets its own pragma; plain "unroll" (full,
// all trip counts are constant), with "unroll n" nvcc 13.2 still left 704 bytes on the stack
#define SMF_PRAGMA_(x) _Pragma(#x)
#if defined(__CUDACC__)
#define SMF_UNROLL(n) _Pragma("unroll")
#elif defined(_MSC_VER) && !defined(__clang__)
#define SMF_UNROLL(n)
#else
#define SMF_UNROLL(n) SMF_PRAGMA_(GCC unroll n)
#endif

// namespace of the kernel; the runtime-dispatch build compiles it once per instruction set into separate
// namespaces (smf_scalar, smf_avx2, smf_avx512) so that inline functions built with different -march flags
// cannot replace each other at link time
#ifndef SMF_NS
#define SMF_NS smf
#endif

namespace SMF_NS {

enum { EXP = 0, LOG = 1, POW = 2, SQRT = 3 };

// ------------------------------------------------------------------ scalar operations
SMF_HD inline double sel(bool m, double a, double b) { return m ? a : b; }
SMF_HD inline bool mnot(bool m) { return !m; }
SMF_HD inline bool sgnbit(double x) { return std::signbit(x); }
SMF_HD inline double m_sqrt(double x) { return std::sqrt(x); }
SMF_HD inline double m_abs(double x) { return std::fabs(x); }
SMF_HD inline double m_acos(double x) { return std::acos(x); }
SMF_HD inline double m_cos(double x) { return std::cos(x); }
SMF_HD inline double m_exp(double x) { return std::exp(x); }
SMF_HD inline double m_log(double x) { return std::log(x); }
SMF_HD inline double m_atanh(double x) { return std::atanh(x); }
SMF_HD inline double m_expm1(double x) { return std::expm1(x); }
SMF_HD inline double m_log1p(double x) { return std::log1p(x); }
SMF_HD inline double m_pow(double x, double y) { return std::pow(x, y); }
// largest power of two <= x (x > 0, normal)
SMF_HD inline double m_pow2floor(double x) { return std::ldexp(1.0, std::ilogb(x)); }

// ------------------------------------------------------------------ the kernel
// 0^e: inf for e < 0, 1 for e = 0, 0 for e > 0
SMF_HD inline double zero_pow(double e) { return e < 0 ? INFINITY : (e == 0 ? 1.0 : 0.0); }

#if defined(_MSC_VER) && !defined(__clang__)
#define SMF_INL __forceinline
#else
#define SMF_INL SMF_HD inline __attribute__((always_inline))
#endif

// x^(al - k) for x > 0 with the exponent al - k = bh + bl exact (TwoSum): the rounding of al - k alone would cost
// |log x| times its error (1e-14 at x ~ 1e100); x^(bh + bl) = x^bh (1 + bl log x)
template <class T>
SMF_INL T pow_k(T x, double al, double k) {
  const double bh = al - k, bb = bh - al, bl = (al - (bh - bb)) + (-k - bb);
  T p = m_pow(x, bh + 0 * x);
  return bl == 0 ? p : p * (1.0 + bl * m_log(x));
}

template <int KIND, class T>
SMF_INL T fun(T x, double al) {
  if (KIND == EXP) return m_exp(x);
  if (KIND == LOG) return m_log(x);
  T xs = sel(x > 0, x, 1.0);
  T y = KIND == SQRT ? m_sqrt(xs) : m_pow(xs, al + 0 * xs);
  return sel(x > 0, y, 0.0);
}

// first divided difference f[a, b] (f[a, a] = f'(a)), numerically stable
template <int KIND, class T>
SMF_INL T divdiff(T a, T b, T fa, T fb, double al) {
  if (KIND == EXP) {                                    // e^hi (1 - e^-d) / d, d = hi - lo >= 0 (no overflow)
    T d = m_abs(a - b);
    auto pos = d > 0;
    T ds = sel(pos, d, 1.0);
    return sel(a > b, fa, fb) * sel(pos, -m_expm1(-ds) / ds, 1.0);
  }
  if (KIND == LOG) {                                    // 2 artanh(z) / (z (a+b)), z = (a-b)/(a+b)
    T s = 0.5 * (a + b), z = (a - b) / (a + b), az = m_abs(z), z2 = z * z;
    auto tiny = az < 1e-3, far = az >= 0.5;
    T zs = sel(tiny | far, 0.5, z);
    T mid = m_atanh(zs) / zs;
    T as = sel(far, a, 1.0), bs = sel(far, b, 2.0);
    T direct = (m_log(as) - m_log(bs)) / (as - bs);
    return sel(tiny, (1.0 + z2 / 3.0 + z2 * z2 / 5.0 + z2 * z2 * z2 / 7.0) / s, sel(far, direct, mid / s));
  }
  a = sel(a > 0, a, 0.0);                               // powers: PSD round-off clamped to 0
  b = sel(b > 0, b, 0.0);
  if (KIND == SQRT) return 1.0 / (m_sqrt(a) + m_sqrt(b));
  T s = 0.5 * (a + b), ss = sel(s > 0, s, 1.0);
  T z = (a - b) / (2.0 * ss), az = m_abs(z);
  auto zero = az == 0;
  auto near = mnot(zero) & (az < 0.5);
  T zs = sel(near, z, 0.25);
  T g = sel(zero, al + 0 * z, (m_expm1(al * m_log1p(zs)) - m_expm1(al * m_log1p(-zs))) / (2.0 * zs));
  T ps = pow_k(ss, al, 1.0);
  auto far = mnot(zero | near);
  T direct = (fa - fb) / sel(far, a - b, 1.0);         // fa = f(a) = a^al (0 for a <= 0)
  return sel(far, direct, sel(s > 0, ps, zero_pow(al - 1.0)) * g);
}

// f'(x) = f[x, x], same values as divdiff(x, x) but cheaper; fx = f(x)
template <int KIND, class T>
SMF_INL T deriv(T x, T fx, double al) {
  if (KIND == EXP) return fx;
  if (KIND == LOG) return 1.0 / x;
  x = sel(x > 0, x, 0.0);
  if (KIND == SQRT) return 1.0 / (2.0 * m_sqrt(x));
  T xs = sel(x > 0, x, 1.0);
  return sel(x > 0, al * pow_k(xs, al, 1.0), al * zero_pow(al - 1.0));
}

// ---- second divided difference f[x, y, z] (symmetric; f[x, x, x] = f''(x)/2), as _dd2 in _jaxpure.py:
//      all three points within a cluster around the middle one m -> Taylor series
//          f[x, y, z] = sum_k f^(k+2)(m)/(k+2)! h_k(lo - m, hi - m),  h_k(u, w) = sum_i u^i w^(k-i)
//      otherwise the difference formula over the most distant pair, from the first divided differences D
constexpr int NTERMS = 30;          // tau^30 * 31 < 1e-16 (exp: 1/32! * 31)
constexpr double TAU = 0.25;        // cluster radius relative to m (exp: absolute 1; pow: min(TAU, 3/al))

// series coefficients c[k] = f^(k+2)(m)/(k+2)! without the m-dependent factor (exp: e^m, log: 1/m^2, pow: m^(al-2))
inline void dd2_coef(int kind, double al, double *c) {
  if (kind == EXP) {
    double f = 0.5;
    for (int k = 0; k < NTERMS; k++) { c[k] = f; f /= (k + 3.0); }
  } else if (kind == LOG) {
    for (int k = 0; k < NTERMS; k++) c[k] = (k % 2 ? 1.0 : -1.0) / (k + 2.0);
  } else {
    double b = al * (al - 1.0) / 2.0;
    for (int k = 0; k < NTERMS; k++) { c[k] = b; b *= (al - k - 2.0) / (k + 3.0); }
  }
}

template <int KIND, class T>
SMF_INL T divdiff2(T li, T lj, T lk, T fi, T fj, T fk, T Dij, T Dik, T Djk, double al, const double *c) {
  if (KIND == POW || KIND == SQRT) {
    li = sel(li > 0, li, 0.0); lj = sel(lj > 0, lj, 0.0); lk = sel(lk > 0, lk, 0.0);
  }
  // difference formula f[x,y,z] = (f[y,z] - f[x,y]) / (z - x) with (x, z) the most distant pair
  T dA = lk - li, dB = lj - li, dC = lk - lj;
  T aA = m_abs(dA), aB = m_abs(dB), aC = m_abs(dC);
  auto useA = (aA >= aB) & (aA >= aC);
  auto useB = mnot(useA) & (aB >= aC);
  T num = sel(useA, Djk - Dij, sel(useB, Djk - Dik, Dik - Dij));
  T den = sel(useA, dA, sel(useB, dB, dC));
  T direct = num / sel(den != 0, den, 1.0);
  // Taylor series around the middle point
  T mn = sel(li < lj, li, lj), mx = sel(li < lj, lj, li);
  T lo = sel(mn < lk, mn, lk), hi = sel(mx > lk, mx, lk);
  T mid = sel(mn > sel(mx < lk, mx, lk), mn, sel(mx < lk, mx, lk));
  T fm = sel(mid == li, fi, sel(mid == lj, fj, fk));
  T u, w, scale;
  auto cluster = (hi - lo) <= 1.0;
  if (KIND == EXP) {
    u = lo - mid; w = hi - mid; scale = fm;
  } else {
    // pow: the series coefficients binom(al, k+2) grow with al, so the radius shrinks to 3/al for al > 12 (terms then
    // decay like 3^k/k!); for al <= 12 it stays TAU
    const double tau = KIND == POW && 3.0 / al < TAU ? 3.0 / al : TAU;
    cluster = (mid > 0) & (mid - lo <= tau * mid) & (hi - mid <= tau * mid);
    T ms = sel(cluster, mid, 1.0);
    u = (lo - mid) / ms; w = (hi - mid) / ms;
    scale = (KIND == LOG ? 1.0 + 0 * ms : sel(cluster, fm, 1.0)) / (ms * ms);   // pow: m^(al-2) = f(m) / m^2
  }
  T h = 1.0 + 0 * u, wk = h, s = c[0] + 0 * u;
SMF_UNROLL(30)
  for (int k = 1; k < NTERMS; k++) {
    wk = wk * w;
    h = u * h + wk;
    s = s + c[k] * h;
  }
  T out = sel(cluster, scale * s, direct);
  if (KIND == POW || KIND == SQRT) {         // all three points 0: f''(0)/2 = binom(al, 2) 0^(al-2)
    const double z2 = al == 1.0 ? 0.0 : al * (al - 1.0) / 2.0 * zero_pow(al - 2.0);
    out = sel(hi > 0, out, z2);
  }
  return out;
}

// Voigt layout (the result of eval3), i = Voigt index of F_ab:  F[i] (6);  J[6 i + k] = dF_i / dv_k (6x6, or
// nullptr);  H[(6 i + p) 6 + q] = d2F_i / dv_p dv_q (6x6x6, or nullptr; needs J and the coefficients c2 from dd2_coef)
// HPACK = 1: H gets only the 126 distinct entries, H[i * 21 + hpack_pq(p, q)] for p <= q (used by the CUDA kernel,
// which expands them with coalesced stores)
SMF_HD constexpr int vidx(int a, int b) { return a == b ? a : a + b + 2; }           // Voigt index of (a, b), a <= b
SMF_HD constexpr int hpack_pq(int p, int q) { return p * (11 - p) / 2 + q; }         // upper triangle of 6x6, p <= q
template <int KIND, class T, int HPACK = 0>
SMF_INL void eval3(const T *a_, T *F, T *J, double al, T *H = nullptr, const double *c2 = nullptr) {
  const double SQRT6 = 2.449489742783178, TWO_PI_3 = 2.0943951023931957;
  // ---- scale A by a power of two sig ~ max |A_ij| (exact): the products below (up to |A|^4 in the cross products)
  //      then neither overflow nor underflow; the eigenvalues are scaled back at the end, the eigenvectors are
  //      unaffected, and in the normal range all results are bitwise the same as without scaling
  const T b0 = a_[0], b1 = a_[1], b2 = a_[2], b3 = a_[3], b4 = a_[4], b5 = a_[5];
  T s01 = sel(m_abs(b0) > m_abs(b1), m_abs(b0), m_abs(b1)), s23 = sel(m_abs(b2) > m_abs(b3), m_abs(b2), m_abs(b3));
  T s45 = sel(m_abs(b4) > m_abs(b5), m_abs(b4), m_abs(b5)), s03 = sel(s01 > s23, s01, s23);
  T scale = sel(s03 > s45, s03, s45) + 1e-300;
  const T sig = m_pow2floor(scale), isig = 1.0 / sig;
  const T a0 = b0 * isig, a1 = b1 * isig, a2 = b2 * isig, a3 = b3 * isig, a4 = b4 * isig, a5 = b5 * isig;
  // ---- simple eigenvalue (trigonometric formula) and its eigenvector (cross products), from the deviator
  //      A - m I (exact where the diagonal is close to m, so accurate also near I), less its trace residue tr3 from
  //      the rounding of m and normalized by its own magnitude (a power of two): the formula needs a traceless
  //      matrix, and the eigenvector then does not depend on how small the deviator is
  T m = (a0 + a1 + a2) / 3.0, d0 = a0 - m, d1 = a1 - m, d2 = a2 - m;
  T tr3 = (d0 + d1 + d2) / 3.0;
  T e0 = d0 - tr3, e1 = d1 - tr3, e2 = d2 - tr3;
  T z01 = sel(m_abs(e0) > m_abs(e1), m_abs(e0), m_abs(e1)), z23 = sel(m_abs(e2) > m_abs(a3), m_abs(e2), m_abs(a3));
  T z45 = sel(m_abs(a4) > m_abs(a5), m_abs(a4), m_abs(a5)), z03 = sel(z01 > z23, z01, z23);
  T dmax = sel(z03 > z45, z03, z45);
  auto iso = dmax <= 1e-300;                           // A = m I
  T ids = 1.0 / sel(iso, 1.0, m_pow2floor(sel(iso, 1.0, dmax)));
  T g0 = e0 * ids, g1 = e1 * ids, g2 = e2 * ids, g3 = a3 * ids, g4 = a4 * ids, g5 = a5 * ids;
  T p = g0 * g0 + g1 * g1 + g2 * g2 + 2.0 * (g3 * g3 + g4 * g4 + g5 * g5);
  T q = g0 * (g1 * g2 - g5 * g5) - g3 * (g3 * g2 - g5 * g4) + g4 * (g3 * g5 - g1 * g4);
  T pg = sel(iso, 1.0, p), qg = sel(iso, 0.0, q);
  T r = 3.0 * SQRT6 * qg / (pg * m_sqrt(pg));
  r = sel(r > 1.0, 1.0, sel(r < -1.0, -1.0, r));
  T t = m_sqrt(2.0 * pg / 3.0), ph = m_acos(r) / 3.0;
  T sh = t * m_cos(sel(r >= 0, ph, ph + TWO_PI_3));
  T m00 = g0 - sh, m11 = g1 - sh, m22 = g2 - sh;     // rows M0 = (m00, g3, g4), M1 = (g3, m11, g5), M2 = (g4, g5, m22)
  T c0x = g3 * g5 - g4 * m11, c0y = g4 * g3 - m00 * g5, c0z = m00 * m11 - g3 * g3;
  T c1x = g3 * m22 - g4 * g5, c1y = g4 * g4 - m00 * m22, c1z = m00 * g5 - g3 * g4;
  T c2x = m11 * m22 - g5 * g5, c2y = g5 * g4 - g3 * m22, c2z = g3 * g5 - m11 * g4;
  T n0 = c0x * c0x + c0y * c0y + c0z * c0z, n1 = c1x * c1x + c1y * c1y + c1z * c1z, n2 = c2x * c2x + c2y * c2y + c2z * c2z;
  auto s1 = n1 > n0;
  T cx = sel(s1, c1x, c0x), cy = sel(s1, c1y, c0y), cz = sel(s1, c1z, c0z), nn = sel(s1, n1, n0);
  auto s2 = n2 > nn;
  cx = sel(s2, c2x, cx); cy = sel(s2, c2y, cy); cz = sel(s2, c2z, cz); nn = sel(s2, n2, nn);
  T nrm = m_sqrt(nn);
  auto ok = (nrm > 1e-300) & mnot(iso);
  T inv = 1.0 / sel(ok, nrm, 1.0);
  T vx = sel(ok, cx * inv, 1.0), vy = sel(ok, cy * inv, 0.0), vz = sel(ok, cz * inv, 0.0);
  // ---- orthonormal complement (u, w) of v, 2x2 Jacobi rotation in span(u, w)
  T ax = m_abs(vx), ay = m_abs(vy), az = m_abs(vz);
  auto k1 = ay < ax;
  auto k2 = az < sel(k1, ay, ax);
  T ex = sel(k1 | k2, 0.0, 1.0), ey = sel(k1 & mnot(k2), 1.0, 0.0), ez = sel(k2, 1.0, 0.0);
  T vk = ex * vx + ey * vy + ez * vz;
  T ux = ex - vk * vx, uy = ey - vk * vy, uz = ez - vk * vz;
  T nu = m_sqrt(ux * ux + uy * uy + uz * uz);
  auto oku = nu > 1e-300;
  T iu = 1.0 / sel(oku, nu, 1.0);
  ux = sel(oku, ux * iu, 0.0); uy = sel(oku, uy * iu, 0.0); uz = sel(oku, uz * iu, 0.0);
  T wx = vy * uz - vz * uy, wy = vz * ux - vx * uz, wz = vx * uy - vy * ux;
  // Rayleigh quotients of A (eigenvalues; accurate for spectra over orders of magnitude) and of the deviator A - m I
  // (x = lambda - m accurate relative to |A - m I|, for log near I)
  T Aux = a0 * ux + a3 * uy + a4 * uz, Auy = a3 * ux + a1 * uy + a5 * uz, Auz = a4 * ux + a5 * uy + a2 * uz;
  T Awx = a0 * wx + a3 * wy + a4 * wz, Awy = a3 * wx + a1 * wy + a5 * wz, Awz = a4 * wx + a5 * wy + a2 * wz;
  T Avx = a0 * vx + a3 * vy + a4 * vz, Avy = a3 * vx + a1 * vy + a5 * vz, Avz = a4 * vx + a5 * vy + a2 * vz;
  T a11 = ux * Aux + uy * Auy + uz * Auz, a12 = ux * Awx + uy * Awy + uz * Awz;
  T a22 = wx * Awx + wy * Awy + wz * Awz, lv = vx * Avx + vy * Avy + vz * Avz;
  T Dux = d0 * ux + a3 * uy + a4 * uz, Duy = a3 * ux + d1 * uy + a5 * uz, Duz = a4 * ux + a5 * uy + d2 * uz;
  T Dwx = d0 * wx + a3 * wy + a4 * wz, Dwy = a3 * wx + d1 * wy + a5 * wz, Dwz = a4 * wx + a5 * wy + d2 * wz;
  T b11 = ux * Dux + uy * Duy + uz * Duz, b12 = ux * Dwx + uy * Dwy + uz * Dwz, b22 = wx * Dwx + wy * Dwy + wz * Dwz;
  // 2x2 Jacobi rotation from the deviator (its angle; a11 - a22 of A would cancel near I)
  T jx = b11 - b22, jy = 2.0 * b12, jr = m_sqrt(jx * jx + jy * jy);
  auto jz = jr == 0, jpos = jx >= 0;
  T jrs = sel(jz, 1.0, jr);
  T big = m_sqrt((jrs + m_abs(jx)) / (2.0 * jrs));
  T sml = jy / (2.0 * jrs * big);
  T sgn = sel((jy > 0) | ((jy == 0) & mnot(sgnbit(jy))), 1.0, -1.0);
  T cs = sel(jz, 1.0, sel(jpos, big, sgn * sml));
  T sn = sel(jz, 0.0, sel(jpos, sml, sgn * big));
  T lam[3] = {lv * sig, (cs * cs * a11 + 2.0 * cs * sn * a12 + sn * sn * a22) * sig,
               (sn * sn * a11 - 2.0 * cs * sn * a12 + cs * cs * a22) * sig};
  T V[3][3] = {{vx, cs * ux + sn * wx, cs * wx - sn * ux},
               {vy, cs * uy + sn * wy, cs * wy - sn * uy},
               {vz, cs * uz + sn * wz, cs * wz - sn * uz}};
  // ---- F = V diag(f) V^T
  T fl[3] = {fun<KIND>(lam[0], al), fun<KIND>(lam[1], al), fun<KIND>(lam[2], al)};
  if (KIND == LOG) {               // near m: log(m sig) + log1p(x/m) with x from the deviator (the rounding of m cancels
                                   // against A - m I, accurate also near I)
    T bv = vx * (d0 * vx + a3 * vy + a4 * vz) + vy * (a3 * vx + d1 * vy + a5 * vz) + vz * (a4 * vx + a5 * vy + d2 * vz);
    T xd[3] = {bv, cs * cs * b11 + 2.0 * cs * sn * b12 + sn * sn * b22, sn * sn * b11 - 2.0 * cs * sn * b12 + cs * cs * b22};
    T lms = m_log(m * sig);
SMF_UNROLL(3)
    for (int i = 0; i < 3; i++) fl[i] = sel(m_abs(xd[i]) <= 0.5 * m, lms + m_log1p(xd[i] / m), fl[i]);
  }
SMF_UNROLL(3)
  for (int x = 0; x < 3; x++)
SMF_UNROLL(3)
    for (int y = x; y < 3; y++)
      F[vidx(x, y)] = V[x][0] * fl[0] * V[y][0] + V[x][1] * fl[1] * V[y][1] + V[x][2] * fl[2] * V[y][2];
  if (!J) return;
  // ---- Daleckii-Krein: J_ab,pp = C_abpp, J_ab,pq = C_abpq + C_abqp, C_abpq = sum_i P_i[a,p] T_i[b,q],
  //      P_i = v_i v_i^T, T_i = sum_j D_ij P_j, D_ij = f[lam_i, lam_j]
  T D01 = divdiff<KIND>(lam[0], lam[1], fl[0], fl[1], al), D02 = divdiff<KIND>(lam[0], lam[2], fl[0], fl[2], al);
  T D12 = divdiff<KIND>(lam[1], lam[2], fl[1], fl[2], al);
  T D[3][3] = {{deriv<KIND>(lam[0], fl[0], al), D01, D02}, {D01, deriv<KIND>(lam[1], fl[1], al), D12},
               {D02, D12, deriv<KIND>(lam[2], fl[2], al)}};
  T P[3][3][3], Q[3][3][3];
SMF_UNROLL(3)
  for (int i = 0; i < 3; i++)
SMF_UNROLL(3)
    for (int x = 0; x < 3; x++)
SMF_UNROLL(3)
      for (int y = 0; y < 3; y++) P[i][x][y] = V[x][i] * V[y][i];
SMF_UNROLL(3)
  for (int i = 0; i < 3; i++)
SMF_UNROLL(3)
    for (int x = 0; x < 3; x++)
SMF_UNROLL(3)
      for (int y = 0; y < 3; y++) Q[i][x][y] = D[i][0] * P[0][x][y] + D[i][1] * P[1][x][y] + D[i][2] * P[2][x][y];
  constexpr int VP[6] = {0, 1, 2, 0, 0, 1}, VQ[6] = {0, 1, 2, 1, 2, 2};
SMF_UNROLL(3)
  for (int x = 0; x < 3; x++)
SMF_UNROLL(3)
    for (int y = x; y < 3; y++)
SMF_UNROLL(6)
      for (int k = 0; k < 6; k++) {
        const int p = VP[k], q = VQ[k];
        T s = P[0][x][p] * Q[0][y][q] + P[1][x][p] * Q[1][y][q] + P[2][x][p] * Q[2][y][q];
        if (p != q) s += P[0][x][q] * Q[0][y][p] + P[1][x][q] * Q[1][y][p] + P[2][x][q] * Q[2][y][p];
        J[6 * vidx(x, y) + k] = s;
      }
  if (!H) return;
  // ---- second order: H^pq = V S^pq V^T,  S^pq = X + X^T,  X_ab = sum_m D2_amb Et^p_am Et^q_mb,  Et^p = V^T E_p V
  constexpr int TRI[10][3] = {{0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {0, 1, 1}, {0, 1, 2},
                              {0, 2, 2}, {1, 1, 1}, {1, 1, 2}, {1, 2, 2}, {2, 2, 2}};
  T D2[3][3][3];
SMF_UNROLL(10)
  for (int t = 0; t < 10; t++) {
    const int i = TRI[t][0], j = TRI[t][1], m = TRI[t][2];
    T d = divdiff2<KIND>(lam[i], lam[j], lam[m], fl[i], fl[j], fl[m], D[i][j], D[i][m], D[j][m], al, c2);
    D2[i][j][m] = D2[i][m][j] = D2[j][i][m] = D2[j][m][i] = D2[m][i][j] = D2[m][j][i] = d;
  }
  T Et[6][3][3];
SMF_UNROLL(6)
  for (int p = 0; p < 6; p++)
SMF_UNROLL(3)
    for (int a = 0; a < 3; a++)
SMF_UNROLL(3)
      for (int m = a; m < 3; m++) {
        const int k = VP[p], l = VQ[p];
        T e = V[k][a] * V[l][m];
        if (k != l) e += V[l][a] * V[k][m];
        Et[p][a][m] = Et[p][m][a] = e;
      }
SMF_UNROLL(6)
  for (int p = 0; p < 6; p++)
SMF_UNROLL(6)
    for (int q = p; q < 6; q++) {
      T S[3][3], W[3][3];
SMF_UNROLL(3)
      for (int a = 0; a < 3; a++)
SMF_UNROLL(3)
        for (int b = a; b < 3; b++) {
          T xab = D2[a][0][b] * Et[p][a][0] * Et[q][0][b] + D2[a][1][b] * Et[p][a][1] * Et[q][1][b] +
                  D2[a][2][b] * Et[p][a][2] * Et[q][2][b];
          T xba = D2[b][0][a] * Et[p][b][0] * Et[q][0][a] + D2[b][1][a] * Et[p][b][1] * Et[q][1][a] +
                  D2[b][2][a] * Et[p][b][2] * Et[q][2][a];
          S[a][b] = S[b][a] = xab + xba;
        }
SMF_UNROLL(3)
      for (int a = 0; a < 3; a++)
SMF_UNROLL(3)
        for (int j = 0; j < 3; j++) W[a][j] = S[a][0] * V[j][0] + S[a][1] * V[j][1] + S[a][2] * V[j][2];
SMF_UNROLL(3)
      for (int i = 0; i < 3; i++)
SMF_UNROLL(3)
        for (int j = i; j < 3; j++) {
          T h = V[i][0] * W[0][j] + V[i][1] * W[1][j] + V[i][2] * W[2][j];
          if (HPACK) {
            H[vidx(i, j) * 21 + hpack_pq(p, q)] = h;
          } else {
            H[(6 * vidx(i, j) + p) * 6 + q] = H[(6 * vidx(i, j) + q) * 6 + p] = h;
          }
        }
    }
}

// ------------------------------------------------------------------ full-matrix input and output
// Matrix layout (smf_eval_matrix), row-major, m = 3a + b:  F[m] (3x3);  J[9 m + c] = dF_ab / dA_cd (3x3x3x3);
// H[(9 m + c) 9 + e] = d2F_ab / dA_cd dA_ef (3x3x3x3x3x3), the derivatives of A -> f((A + A^T)/2): w.r.t. an
// off-diagonal entry half the Voigt derivative, so that J and H have the symmetries of the index pairs
SMF_HD constexpr int midx(int m) { return m % 4 == 0 ? m / 4 : (m == 1 || m == 3 ? 3 : (m == 2 || m == 6 ? 4 : 5)); }
SMF_HD constexpr double mwt(int m) { return m % 4 == 0 ? 1.0 : 0.5; }   // dv_midx(m) / dA_m

// the matrix layout of one matrix from its Voigt layout (J, H: nullptr if not wanted)
// (F and J unrolled: constant indices let nvcc keep Fv and Jv in registers)
SMF_HD inline void to_matrix(const double *Fv, const double *Jv, const double *Hv, double *F, double *J, double *H) {
SMF_UNROLL(9)
  for (int m = 0; m < 9; m++) F[m] = Fv[midx(m)];
  if (J)
SMF_UNROLL(9)
    for (int m = 0; m < 9; m++)
SMF_UNROLL(9)
      for (int c = 0; c < 9; c++) J[9 * m + c] = Jv[6 * midx(m) + midx(c)] * mwt(c);
  if (H)
    for (int m = 0; m < 9; m++)
      for (int c = 0; c < 9; c++)
        for (int e = 0; e < 9; e++)
          H[(9 * m + c) * 9 + e] = Hv[(6 * midx(m) + midx(c)) * 6 + midx(e)] * (mwt(c) * mwt(e));
}

// Voigt vector of the symmetric part of a full matrix a (row-major); false and NaN if a is not symmetric:
// |a_ij - a_ji| > SYM_TOL max |a_kl|  (NaN and Inf entries pass and give NaN results)
constexpr double SYM_TOL = 1e-8;
SMF_HD inline bool from_matrix(const double *a, double *v) {
  double mx = 0.0;
  for (int k = 0; k < 9; k++) mx = m_abs(a[k]) > mx ? m_abs(a[k]) : mx;
  const double tol = SYM_TOL * mx;
  const bool sym = !(m_abs(a[1] - a[3]) > tol) && !(m_abs(a[2] - a[6]) > tol) && !(m_abs(a[5] - a[7]) > tol);
  v[0] = a[0], v[1] = a[4], v[2] = a[8];
  v[3] = 0.5 * (a[1] + a[3]), v[4] = 0.5 * (a[2] + a[6]), v[5] = 0.5 * (a[5] + a[7]);
  if (!sym)
    for (int k = 0; k < 6; k++) v[k] = NAN;
  return sym;
}

}  // namespace SMF_NS
