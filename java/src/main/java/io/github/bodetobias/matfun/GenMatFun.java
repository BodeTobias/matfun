// GenMatFun.java -- matfun kernel for general (non-symmetric) real 3x3 matrices in Java: f(A) for f = exp, log,
// sqrt, x^alpha (principal branch) with the full first (J) and second (H) derivative.
// Port of gmf::eval<KIND, ORDER, double> in c/include/matfun/gmf_kernel.hpp, same steps, thresholds and case
// distinctions; the selects of the branch-free C++ kernel are plain conditionals here. Method: theory.md.
package io.github.bodetobias.matfun;

import java.util.Objects;
import java.util.stream.IntStream;

import io.github.bodetobias.matfun.SymMatFun.Kind;

/**
 * Matrix functions exp, log, sqrt and A^alpha of general (non-symmetric) real 3x3 matrices with first and second
 * derivatives, as {@code gmf_eval} of the C API.
 *
 * <p>Layout (flat, row-major, 0-based, one matrix after the other): {@code A[3a+b]} and {@code F[3a+b]} (9 values),
 * {@code J[9(3a+b)+3c+d] = dF_ab / dA_cd} (81), {@code H[(9(3a+b)+3c+d)*9+3e+f] = d2F_ab / dA_cd dA_ef} (729): the
 * full derivatives, exact also for defective A (Jordan blocks) and complex eigenvalues.
 *
 * <p>Principal branch; log, sqrt and pow give NaN for a matrix with a real eigenvalue &lt;= 0. All methods are
 * thread-safe.
 */
public final class GenMatFun {

  private GenMatFun() {}

  private static final int EXP = 0, LOG = 1, POW = 2;
  static final double RHO = 0.7, R_EXP = 4.0, RHO_POLE = 0.6;
  static final int KMAX = 200;
  private static final double TWO_PI_3 = 2.0943951023931957;
  private static final int[][] TMS = {{1, 0, 0}, {1, 1, 0}, {1, 1, 1}, {2, 0, 0}, {2, 1, 0}, {2, 1, 1}, {2, 2, 0},
      {2, 2, 1}, {2, 2, 2}, {3, 0, 0}, {3, 1, 0}, {3, 1, 1}, {3, 2, 0}, {3, 2, 1}, {3, 2, 2}, {3, 3, 0}, {3, 3, 1},
      {3, 3, 2}, {3, 3, 3}};
  private static final int[] TPAR = {-1, 0, 1, 0, 1, 2, 4, 5, 7, 3, 4, 5, 6, 7, 8, 12, 13, 14, 17};
  private static final int[] TNODE = {0, 1, 2, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2};
  private static final int[] NDD = {3, 9, 19};
  private static final double[] B_EXP = seriesCoef(EXP, 0.0), B_LOG = seriesCoef(LOG, 0.0), B_SQRT = seriesCoef(POW, 0.5);

  /** scratch arrays of one thread */
  private static final class Work {
    final double[] Epr = new double[19], Epi = new double[19], Ecr = new double[19], Eci = new double[19];
    final double[] accr = new double[19], acci = new double[19], Dr = new double[19], Di = new double[19];
    final double[] c = new double[2], f = new double[6];
    final double[] Y = new double[27], Z = new double[81], Yh = new double[243];
    double[] bPow;
    double bAl = Double.NaN;

    double[] coef(int kind, double al) {
      if (kind == EXP) return B_EXP;
      if (kind == LOG) return B_LOG;
      if (al == 0.5) return B_SQRT;
      if (bPow == null || !(al == bAl)) {
        bPow = seriesCoef(POW, al);
        bAl = al;
      }
      return bPow;
    }
  }

  private static final ThreadLocal<Work> WORK = ThreadLocal.withInitial(Work::new);

  /**
   * f(A) and optionally its first and second derivative for n = A.length / 9 general 3x3 matrices stored back to
   * back (matrix i at offset 9i in A and F, 81i in J and 729i in H). Runs in parallel (common fork-join pool) over
   * chunks of 256 matrices (1024 without H) when there is more than one chunk.
   *
   * @param kind  the function
   * @param alpha exponent for {@link Kind#POW} (&gt; 0); ignored otherwise (sqrt uses 0.5)
   * @param A     input, 3x3 matrices (row-major), 9n values
   * @param F     output f(A), 9n values
   * @param J     output dF/dA, 81n values (or null)
   * @param H     output d2F/dA2, 729n values (or null; needs J)
   * @throws IllegalArgumentException  if H is given without J, alpha is not &gt; 0 for POW, or A.length is not a
   *                                   multiple of 9
   * @throws IndexOutOfBoundsException if an output array is too short for n matrices
   */
  public static void eval(Kind kind, double alpha, double[] A, double[] F, double[] J, double[] H) {
    Objects.requireNonNull(kind, "kind");
    if (H != null && J == null) throw new IllegalArgumentException("H requires J");
    if (kind == Kind.POW && !(alpha > 0)) throw new IllegalArgumentException("POW needs alpha > 0, got " + alpha);
    final double al = kind == Kind.POW ? alpha : (kind == Kind.SQRT ? 0.5 : 0.0);
    final int k = kind == Kind.EXP ? EXP : (kind == Kind.LOG ? LOG : POW);
    if (A.length % 9 != 0) throw new IllegalArgumentException("input length must be a multiple of 9, got " + A.length);
    final int n = A.length / 9;
    Objects.checkFromIndexSize(0, 9 * n, F.length);
    if (J != null) Objects.checkFromIndexSize(0, (long) 81 * n > Integer.MAX_VALUE ? -1 : 81 * n, J.length);
    if (H != null) Objects.checkFromIndexSize(0, (long) 729 * n > Integer.MAX_VALUE ? -1 : 729 * n, H.length);
    final int order = H != null ? 2 : (J != null ? 1 : 0);
    final int chunk = H != null ? 256 : 1024;
    final int nchunks = (int) (((long) n + chunk - 1) / chunk);
    if (nchunks <= 1) {
      range(k, order, al, A, F, J, H, 0, n);
      return;
    }
    IntStream.range(0, nchunks).parallel().forEach(c -> range(k, order, al, A, F, J, H, c * chunk,
                                                                Math.min(n, c * chunk + chunk)));
  }

  private static void range(int k, int order, double al, double[] A, double[] F, double[] J, double[] H, int lo,
                            int hi) {
    final Work w = WORK.get();
    final double[] bt = w.coef(k, al);
    for (int i = lo; i < hi; i++) eval1(k, order, al, bt, A, 9 * i, F, 9 * i, J, 81 * i, H, 729 * i, w);
  }

  // series coefficients b_j of f(c (1 + u)) / scale: exp 1/j!; log (-1)^(j+1)/j (b_0 apart); pow binom(al, j)
  static double[] seriesCoef(int kind, double al) {
    final double[] b = new double[KMAX];
    b[0] = 1.0;
    for (int j = 1; j < KMAX; j++)
      b[j] = kind == EXP ? b[j - 1] / j : (kind == LOG ? (j % 2 != 0 ? 1.0 : -1.0) / j : b[j - 1] * (al - j + 1) / j);
    return b;
  }

  // terms K of the group series: sum_k b_(n+k) h_k with |h_k| <= C(k+n, n) r^k and of the pole series C(k+2, 2) q^k
  static int nterms(double[] b, int n, double r, double q) {
    int M = 1;
    if (r > 0) {
      double best = 0.0, c = 1.0, rk = 1.0;
      for (int k = 0; n + k < KMAX; k++) {
        final double t = Math.abs(b[n + k]) * c * rk;
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

  // ------------------------------------------------------------------ helpers
  private static double cabs(double re, double im) {
    final double ar = Math.abs(re), ai = Math.abs(im), big = ar > ai ? ar : ai, sml = ar < ai ? ar : ai;
    if (!(big > 0)) return big == 0 ? 0.0 : big;
    final double q = sml / big;
    return big * Math.sqrt(1.0 + q * q);
  }

  // 1 / (re + i im) into c (Smith)
  private static void cinv(double re, double im, double[] c) {
    final boolean sw = Math.abs(im) > Math.abs(re);
    final double p = sw ? im : re, q = sw ? re : im;
    final double r = q / (p != 0 ? p : 1.0), den = p + q * r;
    final double a = 1.0 / den, b = -r / den;
    c[0] = sw ? -b : a;
    c[1] = sw ? -a : b;
  }

  private static double pow2safe(double x) {
    return (x > 1e-300 && x < 1e300) ? Math.scalb(1.0, Math.getExponent(x)) : 1.0;
  }

  private static double copysgn(double x, double y) {
    return y < 0 ? -Math.abs(x) : Math.abs(x);
  }

  private static double det3(double[] M, int o) {
    return M[o] * (M[o + 4] * M[o + 8] - M[o + 5] * M[o + 7]) - M[o + 1] * (M[o + 3] * M[o + 8] - M[o + 5] * M[o + 6])
        + M[o + 2] * (M[o + 3] * M[o + 7] - M[o + 4] * M[o + 6]);
  }

  // det (M - s I) by elimination with partial pivoting (backward stable, exact pivots for diagonal M)
  private static double detLu(double[] M, double s) {
    final double m0 = M[0] - s, m4 = M[4] - s, m8 = M[8] - s;
    final double[] r0 = {m0, M[1], M[2]}, r1 = {M[3], m4, M[5]}, r2 = {M[6], M[7], m8};
    final double a0 = Math.abs(r0[0]), a1 = Math.abs(r1[0]), a2 = Math.abs(r2[0]);
    final boolean p1 = a1 > a0 && a1 >= a2, p2 = a2 > a0 && a2 > a1;
    final double[] P = p1 ? r1 : (p2 ? r2 : r0), R1 = (p1 || p2) ? r0 : r1, R2 = p2 ? r1 : r2;
    double sgn = p1 ? -1.0 : 1.0;
    final double piv = P[0] != 0 ? P[0] : 1.0;
    final double l1 = R1[0] / piv, l2 = R2[0] / piv;
    final double b = R1[1] - l1 * P[1], c = R1[2] - l1 * P[2], d = R2[1] - l2 * P[1], e = R2[2] - l2 * P[2];
    final boolean sw = Math.abs(d) > Math.abs(b);
    final double bb = sw ? d : b, cc = sw ? e : c, dd = sw ? b : d, ee = sw ? c : e;
    if (sw) sgn = -sgn;
    if (P[0] == 0 || bb == 0) return 0.0;
    return sgn * P[0] * bb * (ee - dd / bb * cc);
  }

  private static double minors2(double[] M) {
    return M[0] * M[4] - M[1] * M[3] + M[0] * M[8] - M[2] * M[6] + M[4] * M[8] - M[5] * M[7];
  }

  // eigenvalues of S (see roots3): x[0] real (re in xr, im in xi)
  private static void roots3(double[] S, double[] xr, double[] xi) {
    final double tau = (S[0] + S[4] + S[8]) / 3.0;
    final double[] Sp = new double[9];
    for (int k = 0; k < 9; k++) Sp[k] = k % 4 == 0 ? S[k] - tau : S[k];
    double s = Math.abs(Sp[0]);
    for (int k = 1; k < 9; k++) s = Math.max(s, Math.abs(Sp[k]));
    double t = pow2safe(s);
    final double it = 1.0 / t;
    for (int k = 0; k < 9; k++) Sp[k] = Sp[k] * it;
    double P = 0.5 * (Sp[0] * Sp[0] + Sp[4] * Sp[4] + Sp[8] * Sp[8]) + Sp[1] * Sp[3] + Sp[2] * Sp[6] + Sp[5] * Sp[7];
    double Q = det3(Sp, 0);
    final double u = pow2safe(Math.max(Math.sqrt(Math.abs(P)), Math.cbrt(Math.abs(Q)))), iu = 1.0 / u;
    P = P * iu * iu;
    Q = Q * iu * iu * iu;
    t = t * u;
    final boolean trig = P > 0 && 4.0 * P * P * P >= 27.0 * Q * Q;
    double s1;
    if (trig) {
      final double p3 = P / 3.0;
      double r = 0.5 * Q / (p3 * Math.sqrt(p3));
      r = r > 1.0 ? 1.0 : (r < -1.0 ? -1.0 : r);
      final double ph = Math.acos(r) / 3.0;
      s1 = 2.0 * Math.sqrt(p3) * Math.cos(r >= 0 ? ph : ph + TWO_PI_3);
    } else {
      final double D = 0.25 * Q * Q - P * P * P / 27.0;
      final double uc = Math.cbrt(0.5 * Q + copysgn(Math.sqrt(D > 0 ? D : 0.0), Q));
      s1 = uc + (uc != 0 ? P / (3.0 * uc) : 0.0);
    }
    final double d = P - 0.75 * s1 * s1;
    final boolean real = d >= 0;
    final double sq = Math.sqrt(Math.abs(d));
    final double sa = -0.5 * s1 - copysgn(sq, s1);
    final double prod = s1 != 0 ? Q / s1 : -P;
    final double sb = sa != 0 ? prod / sa : 0.0;
    final double im = real ? 0.0 : sq * t;
    xr[0] = s1 * t + tau;
    xi[0] = 0.0;
    xr[1] = (real ? sa : -0.5 * s1) * t + tau;
    xi[1] = im;
    xr[2] = (real ? sb : -0.5 * s1) * t + tau;
    xi[2] = -im;
  }

  // real node j: 3 Newton steps on det(A - mu I), derivative from the other nodes
  private static void newton(double[] As, double[] mr, double[] mi, int j, double[] c) {
    for (int it = 0; it < 3; it++) {
      double dr = 1.0, di = 0.0;
      for (int k = 0; k < 3; k++)
        if (k != j) {
          final double er = mr[k] - mr[j], ei = mi[k] - mi[j];
          final double nr = dr * er - di * ei;
          di = dr * ei + di * er;
          dr = nr;
        }
      if (dr == 0 && di == 0) continue;
      cinv(dr, di, c);
      mr[j] = mr[j] + detLu(As, mr[j]) * c[0];
    }
  }

  // f(mu) into c (k: order of the output; pow: times sig^(al - k)); exp: e^x (x relative to the shift)
  private static void coef0(int kind, double al, double sig, double m, double xr, double xi, double mur, double mui,
                            int k, double[] c) {
    if (kind == EXP) {
      final double e = Math.exp(xr);
      c[0] = e * Math.cos(xi);
      c[1] = e * Math.sin(xi);
      return;
    }
    final double th = Math.atan2(mui, mur);
    if (kind == LOG) {
      final boolean near = m > 0 && cabs(xr, xi) <= 0.5 * m;
      if (near) {
        final double zr = xr / m, zi = xi / m;
        c[0] = Math.log(sig * m) + 0.5 * Math.log1p(zr * (2.0 + zr) + zi * zi);
        c[1] = Math.atan2(zi, 1.0 + zr);
      } else {
        c[0] = Math.log(sig * cabs(mur, mui));
        c[1] = th;
      }
      return;
    }
    final double mag = SymMatFun.powK(sig * cabs(mur, mui), al, k), ph = (al - k) * th;
    double re = mag * Math.cos(ph), im = mag * Math.sin(ph);
    for (int i = 0; i < k; i++) {
      final double nr = re * mur - im * mui;
      im = re * mui + im * mur;
      re = nr;
    }
    c[0] = re;
    c[1] = im;
  }

  // ------------------------------------------------------------------ the kernel (gmf::eval)
  private static void eval1(int kind, int order, double al, double[] bt, double[] a, int ao, double[] F, int fo,
                            double[] J, int jo, double[] H, int ho, Work w) {
    final int nd = NDD[order];
    final double[] c = w.c;
    // ---- scaling, shift m, deviator S
    double amax = 0;
    for (int k = 0; k < 9; k++) amax = Math.max(amax, Math.abs(a[ao + k]));
    final double sig = kind == EXP ? 1.0 : pow2safe(amax), isig = 1.0 / sig;
    final double[] As = new double[9], S = new double[9];
    for (int k = 0; k < 9; k++) As[k] = a[ao + k] * isig;
    final double m = (As[0] + As[4] + As[8]) / 3.0;
    for (int k = 0; k < 9; k++) S[k] = k % 4 == 0 ? As[k] - m : As[k];
    final double[] xr = new double[3], xi = new double[3], mr = new double[3], mi = new double[3];
    roots3(S, xr, xi);
    for (int i = 0; i < 3; i++) { mr[i] = m + xr[i]; mi[i] = xi[i]; }
    // ---- the pair x[1], x[2] from its sum and product with A itself (see gmf_kernel.hpp)
    double smax = 0;
    for (int k = 0; k < 9; k++) smax = Math.max(smax, Math.abs(S[k]));
    final double mid = 0.5 * (mr[1] + mr[2]), Rp = 0.5 * cabs(mr[1] - mr[2], mi[1] - mi[2]);
    final boolean small = Math.abs(mid) + Rp <= 0.25 * Math.abs(mr[0]);
    final boolean aniso = Math.abs(m) <= smax && Rp <= 0.25 * Math.abs(mid - mr[0]);
    final boolean pc = mr[0] != 0 && (small || aniso);
    double pcen = 0, ps = 0;
    boolean preal = true;
    if (pc) {
      newton(As, mr, mi, 0, c);
      final double m0 = mr[0];
      final boolean big = Math.abs(m0) >= Math.abs(mid) + Rp;
      final double c2 = minors2(As);
      double sg, pi;
      if (big) {
        pi = detLu(As, 0.0) / m0;
        sg = (c2 - pi) / m0;
      } else {
        sg = As[0] + As[4] + As[8] - m0;
        pi = c2 - m0 * sg;
      }
      pcen = 0.5 * sg;
      final double disc = 0.25 * sg * sg - pi;
      ps = Math.sqrt(Math.abs(disc));
      preal = disc >= 0;
      if (preal) {
        final double r1 = pcen + copysgn(ps, pcen), r2 = r1 != 0 ? pi / r1 : pcen - copysgn(ps, pcen);
        mr[1] = r1; mi[1] = 0.0; mr[2] = r2; mi[2] = 0.0;
      } else {
        mr[1] = pcen; mi[1] = ps; mr[2] = pcen; mi[2] = -ps;
      }
    }
    final boolean[] ab = {pc, pc, pc};
    double[] og = {0.0, 1.0, 2.0};
    for (int i = 0; i < 3; i++)
      if (ab[i]) { xr[i] = mr[i] - m; xi[i] = mi[i]; }
    // ---- node order: exp largest real part last; log, pow smallest modulus last; a conjugate pair stays adjacent
    {
      final int[] perm = {0, 1, 2};
      if (xi[1] != 0) {
        if (!(key(kind, mr[0], mi[0]) <= key(kind, mr[1], mi[1]))) { perm[0] = 1; perm[1] = 2; perm[2] = 0; }
      } else {
        final int[][] NET = {{0, 1}, {1, 2}, {0, 1}};
        for (int[] p : NET)
          if (key(kind, mr[perm[p[0]]], mi[perm[p[0]]]) > key(kind, mr[perm[p[1]]], mi[perm[p[1]]])) {
            final int t = perm[p[0]];
            perm[p[0]] = perm[p[1]];
            perm[p[1]] = t;
          }
      }
      final double[] txr = xr.clone(), txi = xi.clone(), tmr = mr.clone(), tmi = mi.clone(), tog = og.clone();
      final boolean[] tab = ab.clone();
      for (int k = 0; k < 3; k++) {
        xr[k] = txr[perm[k]]; xi[k] = txi[perm[k]]; mr[k] = tmr[perm[k]]; mi[k] = tmi[perm[k]];
        ab[k] = tab[perm[k]]; og[k] = tog[perm[k]];
      }
    }
    // ---- groups: all three, else the closest real / conjugate pair, else none
    final double rhoF = kind == POW && 3.0 / al < RHO ? 3.0 / al : RHO;
    final double cs = (xr[0] + xr[1] + xr[2]) / 3.0;
    final double R3 = Math.max(Math.max(cabs(xr[0] - cs, xi[0]), cabs(xr[1] - cs, xi[1])), cabs(xr[2] - cs, xi[2]));
    final boolean triple = okF(kind, m, rhoF, cs, R3);
    final boolean[] psel = new boolean[3];
    double bestR = Double.POSITIVE_INFINITY;
    final int[] PI_ = {0, 0, 1}, PJ_ = {1, 2, 2};
    for (int p = 0; p < 3; p++) {
      final int i = PI_[p], j = PJ_[p], k = 3 - i - j;
      final boolean elig = (xi[i] == 0 && xi[j] == 0) || (xr[i] == xr[j] && xi[i] == -xi[j]);
      final double cc = 0.5 * (xr[i] + xr[j]), R = 0.5 * cabs(xr[i] - xr[j], xi[i] - xi[j]);
      final boolean okp = elig && okF(kind, m, rhoF, cc, R) && R <= RHO_POLE * cabs(cc - xr[k], -xi[k]) && !triple;
      final boolean better = okp && R < bestR;
      if (better) {
        bestR = R;
        for (int q = 0; q < p; q++) psel[q] = false;
      }
      psel[p] = better;
    }
    final boolean[] inG = {triple || psel[0] || psel[1], triple || psel[0] || psel[2], triple || psel[1] || psel[2]};
    final boolean[] pk = {psel[2], psel[1], psel[0]};
    // ---- group centre and offsets; the refined pair as a group: symmetric offsets
    final int gsum = (inG[0] ? 1 : 0) + (inG[1] ? 1 : 0) + (inG[2] ? 1 : 0);
    double cxg = ((inG[0] ? xr[0] : 0.0) + (inG[1] ? xr[1] : 0.0) + (inG[2] ? xr[2] : 0.0)) / (gsum > 0 ? gsum : 1);
    double cmu = m + cxg;
    final boolean[] is12 = {og[0] > 0.5, og[1] > 0.5, og[2] > 0.5};
    final boolean ovr = pc && gsum == 2 && ((inG[0] && inG[1] && is12[0] && is12[1]) || (inG[0] && inG[2] && is12[0]
        && is12[2]) || (inG[1] && inG[2] && is12[1] && is12[2]));
    if (ovr) { cxg = pcen - m; cmu = pcen; }
    final double dpr = preal ? ps : 0.0, dpi = preal ? 0.0 : ps;
    final double[] offr = new double[3], offi = new double[3];
    for (int i = 0; i < 3; i++) {
      final boolean neg = (mr[i] - pcen < 0) || (mi[i] < 0);
      final double dr = neg ? -dpr : dpr, di = neg ? -dpi : dpi;
      if (ovr && inG[i]) {
        mr[i] = pcen + dr; mi[i] = di;
        xr[i] = mr[i] - m; xi[i] = mi[i];
        offr[i] = dr; offi[i] = di;
      } else if (inG[i]) {
        offr[i] = xr[i] - cxg; offi[i] = xi[i];
      }
    }
    // ---- single real nodes far from m: Newton on det(A - mu I)
    for (int i = 0; i < 3; i++)
      if (!inG[i] && !ab[i] && xi[i] == 0 && Math.abs(xr[i]) > 0.5 * Math.abs(m)) {
        newton(As, mr, mi, i, c);
        ab[i] = true;
        xr[i] = mr[i] - m;
        xi[i] = mi[i];
      }
    final boolean absc = (inG[0] && ab[0]) || (inG[1] && ab[1]) || (inG[2] && ab[2]);
    double apole = 0.0;
    for (int k = 0; k < 3; k++)
      if (pk[k]) apole = (absc || ab[k]) ? cmu - mr[k] : cxg - xr[k];
    final double xshift = kind == EXP ? Math.max(Math.max(xr[0], xr[1]), xr[2]) : 0.0;
    boolean bad = false;
    if (kind != EXP)
      for (int i = 0; i < 3; i++) bad |= xi[i] == 0 && mr[i] <= 0;

    // ---- divided differences: group part (Taylor series streamed over the series index)
    final double[] Dr = w.Dr, Di = w.Di;
    final boolean hasg = inG[0] || inG[1] || inG[2];
    final double L = kind == EXP ? 1.0 : (hasg && cmu > 0 ? cmu : 1.0), iL = 1.0 / L;
    final double[] ur = new double[3], ui = new double[3];
    for (int i = 0; i < 3; i++) { ur[i] = offr[i] * iL; ui[i] = offi[i] * iL; }
    final double ap = Math.abs(apole) > 0 ? apole * iL : 1.0, iap = 1.0 / ap;
    double rr = Math.max(cabs(ur[0], ui[0]), Math.max(cabs(ur[1], ui[1]), cabs(ur[2], ui[2])));
    final double rq0 = rr * Math.abs(iap);
    final double rq = (pk[0] || pk[1] || pk[2]) && rq0 <= 1.0 ? rq0 : 0.0;
    rr = hasg && rr <= R_EXP + 0.5 ? rr : 0.0;
    for (int cc = 0; cc < nd; cc++) { Dr[cc] = 0.0; Di[cc] = 0.0; }
    if (hasg) {
      final int K = nterms(bt, 3 * (order + 1) - 1, rr, rq);
      double b0 = 1.0;
      if (kind == LOG) {
        coef0(LOG, al, sig, m, cxg, 0.0, cmu, 0.0, 0, c);
        b0 = c[0];
      }
      final double[] Epr = w.Epr, Epi = w.Epi, Ecr = w.Ecr, Eci = w.Eci, accr = w.accr, acci = w.acci;
      for (int cc = 0; cc < nd; cc++) {
        final int par = TPAR[cc];
        if (inG[TNODE[cc]]) { Epr[cc] = 0.0; Epi[cc] = 0.0; }
        else { Epr[cc] = par < 0 ? 1.0 : Epr[par]; Epi[cc] = par < 0 ? 0.0 : Epi[par]; }
        accr[cc] = 0.0;
        acci[cc] = 0.0;
      }
      double q1 = 0.0, q2 = 0.0, q3 = 0.0;
      final double[] q = new double[4];
      for (int j = 0; j < K; j++) {
        final double bj = j == 0 ? b0 : bt[j];
        final double g1 = (bj - q1) * iap, g2 = (g1 - q2) * iap, g3 = (g2 - q3) * iap;
        q1 = g1; q2 = g2; q3 = g3;
        q[0] = bj; q[1] = g1; q[2] = g2; q[3] = g3;
        for (int cc = 0; cc < nd; cc++) {
          final int par = TPAR[cc], i = TNODE[cc];
          if (inG[i]) {
            final double eppr = par < 0 ? (j == 0 ? 1.0 : 0.0) : Epr[par], eppi = par < 0 ? 0.0 : Epi[par];
            Ecr[cc] = eppr + (ur[i] * Epr[cc] - ui[i] * Epi[cc]);
            Eci[cc] = eppi + (ur[i] * Epi[cc] + ui[i] * Epr[cc]);
          } else {
            Ecr[cc] = par < 0 ? 0.0 : Ecr[par];
            Eci[cc] = par < 0 ? 0.0 : Eci[par];
          }
          final double g = pk[0] ? q[TMS[cc][0]] : (pk[1] ? q[TMS[cc][1]] : (pk[2] ? q[TMS[cc][2]] : q[0]));
          accr[cc] += Ecr[cc] * g;
          acci[cc] += Eci[cc] * g;
        }
        System.arraycopy(Ecr, 0, Epr, 0, nd);
        System.arraycopy(Eci, 0, Epi, 0, nd);
      }
      for (int cc = 0; cc < nd; cc++) {
        final int n = TMS[cc][0] + TMS[cc][1] + TMS[cc][2] - 1, k = TMS[cc][0] - 1;
        double fac;
        if (kind == EXP) {
          fac = Math.exp(cxg - xshift);
        } else {
          double iLn = 1.0;
          for (int r = 0; r < n - (kind == POW ? k : 0); r++) iLn = iLn * iL;
          fac = kind == LOG ? iLn : SymMatFun.powK(sig * L, al, k) * iLn;
        }
        Dr[cc] = accr[cc] * fac;
        Di[cc] = acci[cc] * fac;
      }
    }
    // ---- residues of the single nodes
    for (int i = 0; i < 3; i++) {
      if (inG[i]) continue;
      final double xsr = kind == EXP ? xr[i] - xshift : xr[i];
      final double[] f0 = w.f;
      final int kmax = kind == POW ? order : 0;
      for (int k = 0; k <= kmax; k++) {
        coef0(kind, al, sig, m, xsr, xi[i], mr[i], mi[i], k, c);
        f0[2 * k] = c[0];
        f0[2 * k + 1] = c[1];
      }
      cinv(mr[i], mi[i], c);
      final double imr = c[0], imi = c[1];
      final double[] wr = new double[3], wi = new double[3];
      for (int z = 0; z < 3; z++)
        if (z != i) {
          final boolean useMu = ab[i] || ab[z];
          cinv(useMu ? mr[i] - mr[z] : xr[i] - xr[z], useMu ? mi[i] - mi[z] : xi[i] - xi[z], c);
          wr[z] = c[0];
          wi[z] = c[1];
        }
      for (int cc = 0; cc < nd; cc++) {
        final int mi_ = TMS[cc][i];
        if (mi_ == 0) continue;
        final int k = TMS[cc][0] - 1;
        final double fbr = kind == POW ? f0[2 * k] : f0[0], fbi = kind == POW ? f0[2 * k + 1] : f0[1];
        final double[] sr = new double[3], si = new double[3];
        sr[0] = fbr;
        si[0] = fbi;
        if (kind == EXP) {
          sr[1] = fbr; si[1] = fbi; sr[2] = fbr * 0.5; si[2] = fbi * 0.5;
        } else if (kind == LOG) {
          sr[1] = imr; si[1] = imi;
          sr[2] = (imr * imr - imi * imi) * -0.5; si[2] = (2 * imr * imi) * -0.5;
        } else {
          sr[1] = (fbr * imr - fbi * imi) * al; si[1] = (fbr * imi + fbi * imr) * al;
          final double h = (al - 1.0) / 2.0;
          sr[2] = (sr[1] * imr - si[1] * imi) * h; si[2] = (sr[1] * imi + si[1] * imr) * h;
        }
        for (int z = 0; z < 3; z++) {
          final int mz = TMS[cc][z];
          if (z == i || mz == 0) continue;
          double pr = 1.0, pi = 0.0;
          for (int r = 0; r < mz; r++) {
            final double nr = pr * wr[z] - pi * wi[z];
            pi = pr * wi[z] + pi * wr[z];
            pr = nr;
          }
          final double t1r = pr * wr[z] - pi * wi[z], t1i = pr * wi[z] + pi * wr[z];
          final double p1r = t1r * -mz, p1i = t1i * -mz;
          final double t2r = t1r * wr[z] - t1i * wi[z], t2i = t1r * wi[z] + t1i * wr[z];
          final double p2r = t2r * (mz * (mz + 1) / 2.0), p2i = t2i * (mz * (mz + 1) / 2.0);
          final double n0r = sr[0] * pr - si[0] * pi, n0i = sr[0] * pi + si[0] * pr;
          final double n1r = (sr[1] * pr - si[1] * pi) + (sr[0] * p1r - si[0] * p1i);
          final double n1i = (sr[1] * pi + si[1] * pr) + (sr[0] * p1i + si[0] * p1r);
          final double n2r = ((sr[2] * pr - si[2] * pi) + (sr[1] * p1r - si[1] * p1i)) + (sr[0] * p2r - si[0] * p2i);
          final double n2i = ((sr[2] * pi + si[2] * pr) + (sr[1] * p1i + si[1] * p1r)) + (sr[0] * p2i + si[0] * p2r);
          sr[0] = n0r; si[0] = n0i; sr[1] = n1r; si[1] = n1i; sr[2] = n2r; si[2] = n2i;
        }
        Dr[cc] += sr[mi_ - 1];
        Di[cc] += si[mi_ - 1];
      }
    }
    // ---- real basis and assembly
    final double b0i = xi[0], b1i = xi[1];
    final double[] B0 = new double[9], B1 = new double[9], r = new double[27];
    for (int k = 0; k < 9; k++) {
      final boolean dg = k % 4 == 0;
      B0[k] = dg ? (ab[0] ? As[k] - mr[0] : S[k] - xr[0]) : S[k];
      B1[k] = dg ? (ab[1] ? As[k] - mr[1] : S[k] - xr[1]) : S[k];
      r[k] = dg ? 1.0 : 0.0;
      r[9 + k] = B0[k];
    }
    for (int i2 = 0; i2 < 3; i2++)
      for (int j2 = 0; j2 < 3; j2++)
        r[18 + 3 * i2 + j2] = B0[3 * i2] * B1[j2] + B0[3 * i2 + 1] * B1[3 + j2] + B0[3 * i2 + 2] * B1[6 + j2]
            - (i2 == j2 ? b0i * b1i : 0.0);
    final double a01 = (ab[0] || ab[1]) ? mr[0] - mr[1] : xr[0] - xr[1];
    final double N10 = -b0i, N20 = -b0i * a01, N21 = -(b0i + b1i);
    final double nanf = bad ? Double.NaN : 1.0;
    final double e = kind == EXP ? Math.exp(m + xshift) : 1.0;
    {
      final double d0 = Dr[pos1(0)] - (N10 * Di[pos1(1)] + N20 * Di[pos1(2)]);
      final double d1 = Dr[pos1(1)] - N21 * Di[pos1(2)], d2 = Dr[pos1(2)];
      for (int k = 0; k < 9; k++) F[fo + k] = (d0 * r[k] + d1 * r[9 + k] + d2 * r[18 + k]) * e * nanf;
    }
    if (order < 1) return;
    final double sc1 = (kind == EXP ? e : (kind == LOG ? isig : 1.0)) * nanf;
    final double[] D2 = new double[9];
    {
      final double[] Er = new double[9], Ei = new double[9];
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 3; k++) { Er[3 * j + k] = Dr[pos2(j, k)]; Ei[3 * j + k] = Di[pos2(j, k)]; }
      for (int k = 0; k < 3; k++) {                      // T on index 0
        final double vr0 = Er[k], vr1 = Er[3 + k], vr2 = Er[6 + k], vi0 = Ei[k], vi1 = Ei[3 + k], vi2 = Ei[6 + k];
        Er[k] = vr0 - (N10 * vi1 + N20 * vi2); Er[3 + k] = vr1 - N21 * vi2;
        Ei[k] = vi0 + (N10 * vr1 + N20 * vr2); Ei[3 + k] = vi1 + N21 * vr2;
      }
      for (int j = 0; j < 3; j++) {                      // T on index 1, real part
        D2[3 * j] = Er[3 * j] - (N10 * Ei[3 * j + 1] + N20 * Ei[3 * j + 2]);
        D2[3 * j + 1] = Er[3 * j + 1] - N21 * Ei[3 * j + 2];
        D2[3 * j + 2] = Er[3 * j + 2];
      }
    }
    final double[] Y = w.Y;
    for (int mm = 0; mm < 3; mm++)
      for (int k = 0; k < 9; k++) Y[9 * mm + k] = D2[3 * mm] * r[k] + D2[3 * mm + 1] * r[9 + k] + D2[3 * mm + 2] * r[18 + k];
    for (int aa = 0; aa < 3; aa++)
      for (int bb = 0; bb < 3; bb++)
        for (int cc = 0; cc < 3; cc++)
          for (int dd = 0; dd < 3; dd++)
            J[jo + ((3 * aa + bb) * 3 + cc) * 3 + dd] = (r[3 * aa + cc] * Y[3 * dd + bb] + r[9 + 3 * aa + cc]
                * Y[9 + 3 * dd + bb] + r[18 + 3 * aa + cc] * Y[18 + 3 * dd + bb]) * sc1;
    if (order < 2) return;
    final double sc2 = (kind == EXP ? e : (kind == LOG ? isig * isig : 1.0)) * nanf;
    final double[] D3 = new double[27];
    {
      final double[] Er = new double[27], Ei = new double[27];
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 3; k++)
          for (int l = 0; l < 3; l++) {
            Er[9 * j + 3 * k + l] = Dr[pos3(j, k, l)];
            Ei[9 * j + 3 * k + l] = Di[pos3(j, k, l)];
          }
      final int[] STR = {9, 3, 1};
      for (int ax = 0; ax < 3; ax++) {
        final int s = STR[ax];
        for (int base = 0; base < 27; base++) {
          if ((base / s) % 3 != 0) continue;
          final double vr0 = Er[base], vr1 = Er[base + s], vr2 = Er[base + 2 * s];
          final double vi0 = Ei[base], vi1 = Ei[base + s], vi2 = Ei[base + 2 * s];
          Er[base] = vr0 - (N10 * vi1 + N20 * vi2); Er[base + s] = vr1 - N21 * vi2;
          Ei[base] = vi0 + (N10 * vr1 + N20 * vr2); Ei[base + s] = vi1 + N21 * vr2;
        }
      }
      System.arraycopy(Er, 0, D3, 0, 27);
    }
    final double[] Z = w.Z, Yh = w.Yh;
    for (int mn = 0; mn < 9; mn++)
      for (int k = 0; k < 9; k++) Z[9 * mn + k] = D3[3 * mn] * r[k] + D3[3 * mn + 1] * r[9 + k] + D3[3 * mn + 2] * r[18 + k];
    for (int mm = 0; mm < 3; mm++)
      for (int de = 0; de < 9; de++)
        for (int fb = 0; fb < 9; fb++)
          Yh[81 * mm + 9 * de + fb] = r[de] * Z[9 * (3 * mm) + fb] + r[9 + de] * Z[9 * (3 * mm + 1) + fb]
              + r[18 + de] * Z[9 * (3 * mm + 2) + fb];
    for (int aa = 0; aa < 3; aa++)
      for (int bb = 0; bb < 3; bb++)
        for (int cc = 0; cc < 3; cc++)
          for (int dd = 0; dd < 3; dd++)
            for (int ee = 0; ee < 3; ee++)
              for (int ff = 0; ff < 3; ff++) {
                final int i1 = 9 * (3 * dd + ee) + 3 * ff + bb, i2 = 9 * (3 * ff + cc) + 3 * dd + bb;
                final double h = r[3 * aa + cc] * Yh[i1] + r[9 + 3 * aa + cc] * Yh[81 + i1] + r[18 + 3 * aa + cc] * Yh[162 + i1]
                    + r[3 * aa + ee] * Yh[i2] + r[9 + 3 * aa + ee] * Yh[81 + i2] + r[18 + 3 * aa + ee] * Yh[162 + i2];
                H[ho + ((((3 * aa + bb) * 3 + cc) * 3 + dd) * 3 + ee) * 3 + ff] = h * sc2;
              }
  }

  private static double key(int kind, double re, double im) {
    return kind == EXP ? re : -cabs(re, im);
  }

  private static boolean okF(int kind, double m, double rhoF, double c, double R) {
    return kind == EXP ? R <= R_EXP : (m + c > 0 && R <= rhoF * (m + c));
  }

  private static int tpos(int a, int b, int c) {
    return a == 1 ? (b + c) : (a == 2 ? (b == 0 ? 3 : (b == 1 ? 4 + c : 6 + c))
        : (b == 0 ? 9 : (b == 1 ? 10 + c : (b == 2 ? 12 + c : 15 + c))));
  }

  private static int pos1(int j) {
    return tpos(1, j >= 1 ? 1 : 0, j >= 2 ? 1 : 0);
  }

  private static int pos2(int j, int k) {
    return tpos(2, (j >= 1 ? 1 : 0) + (k >= 1 ? 1 : 0), (j >= 2 ? 1 : 0) + (k >= 2 ? 1 : 0));
  }

  private static int pos3(int j, int k, int l) {
    return tpos(3, (j >= 1 ? 1 : 0) + (k >= 1 ? 1 : 0) + (l >= 1 ? 1 : 0), (j >= 2 ? 1 : 0) + (k >= 2 ? 1 : 0)
        + (l >= 2 ? 1 : 0));
  }
}
