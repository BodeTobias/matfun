// SymMatFun.java -- matfun 3x3 kernel in Java: f(A) for f = exp, log, sqrt, x^alpha of a symmetric 3x3 matrix A,
// with the first (J) and second (H) derivative.
// Port of eval3<KIND, double> (HPACK = 0) in c/include/matfun/smf_kernel.hpp, same steps, thresholds and case
// distinctions and the same order of floating-point operations; the selects of the branch-free C++ kernel are plain
// conditionals here (only the chosen branch is computed, its value is the same). The 3x3 loops of F, J and H are
// unrolled into local variables (faster with HotSpot C2). Conventions: README.md.
package io.github.bodetobias.matfun;

import java.util.Objects;
import java.util.stream.IntStream;

/**
 * Matrix functions exp, log, sqrt and A^alpha of symmetric 3x3 matrices with first and second derivatives.
 *
 * <p>Layouts (flat, row-major, 0-based, one matrix after the other, as in the C API {@code matfun.h}):
 * <ul>
 *   <li>{@link #eval}: Voigt vectors {@code v = (A11, A22, A33, A12, A13, A23)} in, {@code F[i]} (6 values),
 *       {@code J[6i+k] = dF_i / dv_k} (36), {@code H[(6i+p)*6+q] = d2F_i / dv_p dv_q} (216) out, with i, k, p, q
 *       Voigt indices. The derivative w.r.t. an off-diagonal component, e.g. {@code v[3] = A12}, changes A12 and
 *       A21 together.</li>
 *   <li>{@link #evalMatrix}: matrices in, {@code F[3a+b]} (9), {@code J[9(3a+b)+3c+d] = dF_ab / dA_cd} (81),
 *       {@code H[(9(3a+b)+3c+d)*9+3e+f] = d2F_ab / dA_cd dA_ef} (729) out: the derivatives of
 *       {@code A -> f((A + A^T)/2)}, with the symmetries of the index pairs.</li>
 * </ul>
 *
 * <p>Domains: exp all symmetric matrices; log positive definite; sqrt and pow positive semidefinite (round-off
 * below 0 is clamped to 0). Arguments outside the domain are not checked (NaN or infinite results).
 * All methods are thread-safe.
 */
public final class SymMatFun {

  /** The matrix function. */
  public enum Kind {
    /** e^x */
    EXP(0),
    /** ln x */
    LOG(1),
    /** x^alpha, alpha &gt; 0 */
    POW(2),
    /** sqrt(x) (internally alpha = 0.5) */
    SQRT(3);

    /** code as in the C enum {@code smf_kind} */
    final int code;

    Kind(int code) {
      this.code = code;
    }

    /** the kind with a given C code 0..3 (IllegalArgumentException for an unknown code) */
    static Kind fromCode(int code) {
      switch (code) {
        case 0: return EXP;
        case 1: return LOG;
        case 2: return POW;
        case 3: return SQRT;
        default: throw new IllegalArgumentException("unknown kind code " + code);
      }
    }
  }

  private SymMatFun() {}

  private static final int EXP = 0, LOG = 1, POW = 2, SQRT = 3;

  // ---- second divided difference f[x, y, z]: Taylor series in a cluster around the middle point (see divdiff2)
  static final int NTERMS = 30;          // tau^30 * 31 < 1e-16 (exp: 1/32! * 31)
  static final double TAU = 0.25;        // cluster radius relative to m (exp: absolute 1; pow: min(TAU, 3/al))

  private static final double SQRT6 = 2.449489742783178, TWO_PI_3 = 2.0943951023931957;
  private static final int[] VP = {0, 1, 2, 0, 0, 1}, VQ = {0, 1, 2, 1, 2, 2};
  // matrix entry m = 3a + b -> Voigt index, and dv / dA_m (1 diagonal, 1/2 off-diagonal: symmetric part)
  private static final int[] MIDX = {0, 3, 4, 3, 1, 5, 4, 5, 2};
  private static final double[] MWT = {1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0};
  static final double SYM_TOL = 1e-8;    // matrices: |A_ij - A_ji| <= SYM_TOL max |A_kl|

  // series coefficients of the fixed kinds (pow: per alpha, cached in Work)
  private static final double[] C_EXP = ddCoef(EXP, 0.0), C_LOG = ddCoef(LOG, 0.0), C_SQRT = ddCoef(SQRT, 0.5);

  /** scratch arrays of one thread (no allocation per matrix) */
  private static final class Work {
    final double[] V = new double[9];     // V[3x+i]: component x of eigenvector i
    final double[] Et = new double[54];   // Et[9p+3a+m] = (V^T E_p V)_am
    final double[] v = new double[6], f = new double[6], j = new double[36], h = new double[216];   // matrix input
    double[] cPow;                        // dd2_coef for pow and alpha cAl
    double cAl = Double.NaN;

    double[] coef(int kind, double al) {
      if (kind == EXP) return C_EXP;
      if (kind == LOG) return C_LOG;
      if (kind == SQRT) return C_SQRT;
      if (cPow == null || !(al == cAl)) {
        cPow = ddCoef(POW, al);
        cAl = al;
      }
      return cPow;
    }
  }

  private static final ThreadLocal<Work> WORK = ThreadLocal.withInitial(Work::new);

  // ------------------------------------------------------------------ public API

  /**
   * f(A) and optionally its first and second derivative for n = V.length / 6 Voigt vectors stored back to back
   * (n = 1: one matrix): matrix i at offset 6i in V and F, 36i in J and 216i in H. Runs in parallel (common
   * fork-join pool) over chunks of 256 matrices (1024 without H) when there is more than one chunk.
   *
   * @param kind  the function
   * @param alpha exponent for {@link Kind#POW} (&gt; 0); ignored otherwise (sqrt uses 0.5)
   * @param V     input, Voigt vectors (A11, A22, A33, A12, A13, A23), 6n values
   * @param F     output f(A) as Voigt vectors, 6n values
   * @param J     output dF/dv, 36n values (or null)
   * @param H     output d2F/dv2, 216n values (or null; needs J)
   * @throws IllegalArgumentException  if H is given without J, alpha is not &gt; 0 for POW, or V.length is not a
   *                                   multiple of 6
   * @throws IndexOutOfBoundsException if an output array is too short for n matrices
   */
  public static void eval(Kind kind, double alpha, double[] V, double[] F, double[] J, double[] H) {
    run(kind, alpha, V, false, F, J, H);
  }

  /**
   * As {@link #eval(Kind, double, double[], double[], double[], double[])}, but for n = A.length / 9 symmetric
   * 3x3 matrices (row-major, matrix i at offset 9i in A and F, 81i in J and 729i in H), with the derivatives w.r.t.
   * the matrix entries.
   *
   * @param kind  the function
   * @param alpha exponent for {@link Kind#POW} (&gt; 0); ignored otherwise (sqrt uses 0.5)
   * @param A     input, symmetric 3x3 matrices, 9n values
   * @param F     output f(A), 9n values
   * @param J     output dF/dA, 81n values (or null)
   * @param H     output d2F/dA2, 729n values (or null; needs J)
   * @throws IllegalArgumentException  if a matrix is not symmetric ({@code |A_ij - A_ji| > 1e-8 max |A_kl|}), H is
   *                                   given without J, alpha is not &gt; 0 for POW, or A.length is not a multiple
   *                                   of 9
   * @throws IndexOutOfBoundsException if an output array is too short for n matrices
   */
  public static void evalMatrix(Kind kind, double alpha, double[] A, double[] F, double[] J, double[] H) {
    run(kind, alpha, A, true, F, J, H);
  }

  // n matrices from X: Voigt vectors (stride 6) or full matrices (mat, stride 9); outputs in the matching layout
  private static void run(Kind kind, double alpha, double[] X, boolean mat, double[] F, double[] J, double[] H) {
    final double al = checkArgs(kind, alpha, J, H);
    final int stride = mat ? 9 : 6, nj = mat ? 81 : 36, nh = mat ? 729 : 216;
    if (X.length % stride != 0)
      throw new IllegalArgumentException("input length must be a multiple of " + stride + ", got " + X.length);
    final int n = X.length / stride;
    Objects.checkFromIndexSize(0, (long) stride * n > Integer.MAX_VALUE ? -1 : stride * n, F.length);
    if (J != null) Objects.checkFromIndexSize(0, (long) nj * n > Integer.MAX_VALUE ? -1 : nj * n, J.length);
    if (H != null) Objects.checkFromIndexSize(0, (long) nh * n > Integer.MAX_VALUE ? -1 : nh * n, H.length);
    if (mat)
      for (int i = 0; i < n; i++) {
        final int o = 9 * i;
        double mx = 0;
        for (int k = 0; k < 9; k++) mx = Math.max(mx, Math.abs(X[o + k]));
        final double tol = SYM_TOL * mx;
        if (Math.abs(X[o + 1] - X[o + 3]) > tol || Math.abs(X[o + 2] - X[o + 6]) > tol
            || Math.abs(X[o + 5] - X[o + 7]) > tol)
          throw new IllegalArgumentException("matrix " + i + " is not symmetric (|A_ij - A_ji| > 1e-8 max |A_kl|)");
      }
    final int k = kind.code;
    final int chunk = H != null ? 256 : 1024;
    final int nchunks = (int) (((long) n + chunk - 1) / chunk);
    if (nchunks <= 1) {
      range(k, al, X, mat, F, J, H, 0, n);
      return;
    }
    IntStream.range(0, nchunks).parallel().forEach(c -> range(k, al, X, mat, F, J, H, c * chunk,
                                                                Math.min(n, c * chunk + chunk)));
  }

  private static void range(int k, double al, double[] X, boolean mat, double[] F, double[] J, double[] H, int lo,
                            int hi) {
    final Work w = WORK.get();
    final double[] c2 = w.coef(k, al);
    for (int i = lo; i < hi; i++) {
      if (!mat) {
        eval3(k, al, c2, X, 6 * i, F, 6 * i, J, 36 * i, H, 216 * i, w);
        continue;
      }
      final int o = 9 * i;                               // symmetric part (symmetric up to round-off, see run)
      final double[] v = w.v;
      v[0] = X[o]; v[1] = X[o + 4]; v[2] = X[o + 8];
      v[3] = 0.5 * (X[o + 1] + X[o + 3]); v[4] = 0.5 * (X[o + 2] + X[o + 6]); v[5] = 0.5 * (X[o + 5] + X[o + 7]);
      eval3(k, al, c2, v, 0, w.f, 0, J != null ? w.j : null, 0, H != null ? w.h : null, 0, w);
      for (int m = 0; m < 9; m++) F[o + m] = w.f[MIDX[m]];
      if (J != null)
        for (int m = 0; m < 9; m++)
          for (int c = 0; c < 9; c++) J[81 * i + 9 * m + c] = w.j[6 * MIDX[m] + MIDX[c]] * MWT[c];
      if (H != null)
        for (int m = 0; m < 9; m++)
          for (int c = 0; c < 9; c++)
            for (int e = 0; e < 9; e++)
              H[729 * i + 81 * m + 9 * c + e] = w.h[36 * MIDX[m] + 6 * MIDX[c] + MIDX[e]] * (MWT[c] * MWT[e]);
    }
  }

  /** checks the arguments as smf_eval does; returns the alpha used internally (sqrt 0.5, exp/log 0) */
  private static double checkArgs(Kind kind, double alpha, double[] J, double[] H) {
    Objects.requireNonNull(kind, "kind");
    if (H != null && J == null) throw new IllegalArgumentException("H requires J");
    if (kind == Kind.POW && !(alpha > 0)) throw new IllegalArgumentException("POW needs alpha > 0, got " + alpha);
    return kind == Kind.POW ? alpha : (kind == Kind.SQRT ? 0.5 : 0.0);
  }

  // ------------------------------------------------------------------ scalar operations

  // 0^e: inf for e < 0, 1 for e = 0, 0 for e > 0
  private static double zeroPow(double e) {
    return e < 0 ? Double.POSITIVE_INFINITY : (e == 0 ? 1.0 : 0.0);
  }

  // artanh(z) for |z| < 1; java.lang.Math has no atanh: 0.5 log1p(2z / (1 - z)) = 0.5 log((1 + z) / (1 - z)),
  // accurate to a few ulp (here used for 1e-3 <= |z| < 0.5 only)
  static double atanh(double z) {
    return 0.5 * Math.log1p(2.0 * z / (1.0 - z));
  }

  // x^(al - k) for x > 0 with the exponent al - k = bh + bl exact (TwoSum): the rounding of al - k alone would cost
  // |log x| times its error (1e-14 at x ~ 1e100); x^(bh + bl) = x^bh (1 + bl log x)
  static double powK(double x, double al, double k) {
    final double bh = al - k, bb = bh - al, bl = (al - (bh - bb)) + (-k - bb);
    final double p = Math.pow(x, bh);
    return bl == 0 ? p : p * (1.0 + bl * Math.log(x));
  }

  private static double fun(int kind, double x, double al) {
    switch (kind) {
      case EXP: return Math.exp(x);
      case LOG: return Math.log(x);
      case SQRT: return x > 0 ? Math.sqrt(x) : 0.0;
      default: return x > 0 ? Math.pow(x, al) : 0.0;
    }
  }

  // first divided difference f[a, b] (f[a, a] = f'(a)), numerically stable
  private static double divdiff(int kind, double a, double b, double fa, double fb, double al) {
    if (kind == EXP) {                                   // e^hi (1 - e^-d) / d, d = hi - lo >= 0 (no overflow)
      final double d = Math.abs(a - b);
      return (a > b ? fa : fb) * (d > 0 ? -Math.expm1(-d) / d : 1.0);
    }
    if (kind == LOG) {                                   // 2 artanh(z) / (z (a+b)), z = (a-b)/(a+b)
      final double s = 0.5 * (a + b), z = (a - b) / (a + b), az = Math.abs(z), z2 = z * z;
      if (az < 1e-3) return (1.0 + z2 / 3.0 + z2 * z2 / 5.0 + z2 * z2 * z2 / 7.0) / s;
      if (az >= 0.5) return (Math.log(a) - Math.log(b)) / (a - b);
      return (atanh(z) / z) / s;
    }
    a = a > 0 ? a : 0.0;                                 // powers: PSD round-off clamped to 0
    b = b > 0 ? b : 0.0;
    if (kind == SQRT) return 1.0 / (Math.sqrt(a) + Math.sqrt(b));
    final double s = 0.5 * (a + b), ss = s > 0 ? s : 1.0;
    final double z = (a - b) / (2.0 * ss), az = Math.abs(z);
    final double g;
    if (az == 0) g = al;
    else if (az < 0.5) g = (Math.expm1(al * Math.log1p(z)) - Math.expm1(al * Math.log1p(-z))) / (2.0 * z);
    else return (fa - fb) / (a - b);                     // far apart (or NaN): fa = f(a) = a^al (0 for a <= 0)
    return (s > 0 ? powK(ss, al, 1.0) : zeroPow(al - 1.0)) * g;
  }

  // f'(x) = f[x, x], same values as divdiff(x, x) but cheaper; fx = f(x)
  private static double deriv(int kind, double x, double fx, double al) {
    if (kind == EXP) return fx;
    if (kind == LOG) return 1.0 / x;
    x = x > 0 ? x : 0.0;
    if (kind == SQRT) return 1.0 / (2.0 * Math.sqrt(x));
    return x > 0 ? al * powK(x, al, 1.0) : al * zeroPow(al - 1.0);
  }

  // ---- second divided difference f[x, y, z] (symmetric; f[x, x, x] = f''(x)/2), as _dd2 in _jaxpure.py:
  //      all three points within a cluster around the middle one m -> Taylor series
  //          f[x, y, z] = sum_k f^(k+2)(m)/(k+2)! h_k(lo - m, hi - m),  h_k(u, w) = sum_i u^i w^(k-i)
  //      otherwise the difference formula over the most distant pair, from the first divided differences D

  // series coefficients c[k] = f^(k+2)(m)/(k+2)! without the m-dependent factor (exp: e^m, log: 1/m^2, pow: m^(al-2))
  static double[] ddCoef(int kind, double al) {
    final double[] c = new double[NTERMS];
    if (kind == EXP) {
      double f = 0.5;
      for (int k = 0; k < NTERMS; k++) { c[k] = f; f /= (k + 3.0); }
    } else if (kind == LOG) {
      for (int k = 0; k < NTERMS; k++) c[k] = (k % 2 != 0 ? 1.0 : -1.0) / (k + 2.0);
    } else {
      double b = al * (al - 1.0) / 2.0;
      for (int k = 0; k < NTERMS; k++) { c[k] = b; b *= (al - k - 2.0) / (k + 3.0); }
    }
    return c;
  }

  private static double divdiff2(int kind, double li, double lj, double lk, double fi, double fj, double fk,
                                 double Dij, double Dik, double Djk, double al, double[] c) {
    final boolean pw = kind == POW || kind == SQRT;
    if (pw) {
      li = li > 0 ? li : 0.0; lj = lj > 0 ? lj : 0.0; lk = lk > 0 ? lk : 0.0;
    }
    // Taylor series around the middle point
    final double mn = li < lj ? li : lj, mx = li < lj ? lj : li;
    final double lo = mn < lk ? mn : lk, hi = mx > lk ? mx : lk;
    final double t = mx < lk ? mx : lk;
    final double mid = mn > t ? mn : t;
    final double fm = mid == li ? fi : (mid == lj ? fj : fk);
    final boolean cluster;
    final double u, w, scale;
    if (kind == EXP) {
      cluster = (hi - lo) <= 1.0;
      u = lo - mid; w = hi - mid; scale = fm;
    } else {
      // pow: the coefficients binom(al, k+2) grow with al, so the radius shrinks to 3/al for al > 12
      final double tau = kind == POW && 3.0 / al < TAU ? 3.0 / al : TAU;
      cluster = (mid > 0) & (mid - lo <= tau * mid) & (hi - mid <= tau * mid);
      final double ms = cluster ? mid : 1.0;
      u = (lo - mid) / ms; w = (hi - mid) / ms;
      scale = (kind == LOG ? 1.0 : (cluster ? fm : 1.0)) / (ms * ms);    // pow: m^(al-2) = f(m) / m^2
    }
    double out;
    if (cluster) {
      double h = 1.0, wk = 1.0, s = c[0];
      for (int k = 1; k < NTERMS; k++) {
        wk = wk * w;
        h = u * h + wk;
        s = s + c[k] * h;
      }
      out = scale * s;
    } else {
      // difference formula f[x,y,z] = (f[y,z] - f[x,y]) / (z - x) with (x, z) the most distant pair
      final double dA = lk - li, dB = lj - li, dC = lk - lj;
      final double aA = Math.abs(dA), aB = Math.abs(dB), aC = Math.abs(dC);
      final boolean useA = (aA >= aB) & (aA >= aC);
      final boolean useB = !useA & (aB >= aC);
      final double num = useA ? Djk - Dij : (useB ? Djk - Dik : Dik - Dij);
      final double den = useA ? dA : (useB ? dB : dC);
      out = num / (den != 0 ? den : 1.0);
    }
    if (pw) {                                  // all three points 0: f''(0)/2 = binom(al, 2) 0^(al-2)
      final double z2 = al == 1.0 ? 0.0 : al * (al - 1.0) / 2.0 * zeroPow(al - 2.0);
      out = hi > 0 ? out : z2;
    }
    return out;
  }

  // ------------------------------------------------------------------ the kernel
  // a: Voigt (A11, A22, A33, A12, A13, A23);  Voigt layout out: F[i] (6), J[6i+k] = dF_i / dv_k (36, or null),
  // H[(6i+p)*6+q] = d2F_i / dv_p dv_q (216, or null; needs J and the coefficients c2 from ddCoef)
  private static void eval3(int kind, double al, double[] c2, double[] A, int ao, double[] F, int fo,
                            double[] J, int jo, double[] H, int ho, Work wk) {
    // ---- scale A by a power of two sig ~ max |A_ij| (exact), so that the products below (up to |A|^4 in the cross
    //      products) neither overflow nor underflow; eigenvalues are scaled back below, eigenvectors are unaffected
    final double b0 = Math.abs(A[ao]), b1 = Math.abs(A[ao + 1]), b2 = Math.abs(A[ao + 2]), b3 = Math.abs(A[ao + 3]);
    final double b4 = Math.abs(A[ao + 4]), b5 = Math.abs(A[ao + 5]);
    final double z01 = b0 > b1 ? b0 : b1, z23 = b2 > b3 ? b2 : b3, z45 = b4 > b5 ? b4 : b5;
    final double z03 = z01 > z23 ? z01 : z23;
    final double scale0 = (z03 > z45 ? z03 : z45) + 1e-300;
    final double sig = Math.scalb(1.0, Math.getExponent(scale0)), isig = 1.0 / sig;
    final double a0 = A[ao] * isig, a1 = A[ao + 1] * isig, a2 = A[ao + 2] * isig, a3 = A[ao + 3] * isig;
    final double a4 = A[ao + 4] * isig, a5 = A[ao + 5] * isig;
    // ---- simple eigenvalue (trigonometric formula) and its eigenvector (cross products), from the deviator
    //      A - m I (exact where the diagonal is close to m, so accurate also near I), less its trace residue tr3 from
    //      the rounding of m and normalized by its own magnitude (a power of two): the formula needs a traceless
    //      matrix, and the eigenvector then does not depend on how small the deviator is
    final double m = (a0 + a1 + a2) / 3.0, d0 = a0 - m, d1 = a1 - m, d2 = a2 - m;
    final double tr3 = (d0 + d1 + d2) / 3.0, e0 = d0 - tr3, e1 = d1 - tr3, e2 = d2 - tr3;
    final double dmax = Math.max(Math.max(Math.max(Math.abs(e0), Math.abs(e1)), Math.max(Math.abs(e2), Math.abs(a3))),
                                 Math.max(Math.abs(a4), Math.abs(a5)));
    final boolean iso = !(dmax > 1e-300);                          // A = m I
    final double ids = iso ? 1.0 : 1.0 / Math.scalb(1.0, Math.getExponent(dmax));
    final double g0 = e0 * ids, g1 = e1 * ids, g2 = e2 * ids, g3 = a3 * ids, g4 = a4 * ids, g5 = a5 * ids;
    final double p = g0 * g0 + g1 * g1 + g2 * g2 + 2.0 * (g3 * g3 + g4 * g4 + g5 * g5);
    final double q = g0 * (g1 * g2 - g5 * g5) - g3 * (g3 * g2 - g5 * g4) + g4 * (g3 * g5 - g1 * g4);
    final double pg = iso ? 1.0 : p, qg = iso ? 0.0 : q;
    double r = 3.0 * SQRT6 * qg / (pg * Math.sqrt(pg));
    r = r > 1.0 ? 1.0 : (r < -1.0 ? -1.0 : r);
    final double t = Math.sqrt(2.0 * pg / 3.0), ph = Math.acos(r) / 3.0;
    final double sh = t * Math.cos(r >= 0 ? ph : ph + TWO_PI_3);
    final double m00 = g0 - sh, m11 = g1 - sh, m22 = g2 - sh; // rows M0 = (m00, g3, g4), M1 = (g3, m11, g5), M2 = (g4, g5, m22)
    final double c0x = g3 * g5 - g4 * m11, c0y = g4 * g3 - m00 * g5, c0z = m00 * m11 - g3 * g3;
    final double c1x = g3 * m22 - g4 * g5, c1y = g4 * g4 - m00 * m22, c1z = m00 * g5 - g3 * g4;
    final double c2x = m11 * m22 - g5 * g5, c2y = g5 * g4 - g3 * m22, c2z = g3 * g5 - m11 * g4;
    final double n0 = c0x * c0x + c0y * c0y + c0z * c0z, n1 = c1x * c1x + c1y * c1y + c1z * c1z;
    final double n2 = c2x * c2x + c2y * c2y + c2z * c2z;
    final boolean s1 = n1 > n0;
    double cx = s1 ? c1x : c0x, cy = s1 ? c1y : c0y, cz = s1 ? c1z : c0z, nn = s1 ? n1 : n0;
    if (n2 > nn) { cx = c2x; cy = c2y; cz = c2z; nn = n2; }
    final double nrm = Math.sqrt(nn);
    final boolean ok = (nrm > 1e-300) & !iso;
    final double inv = 1.0 / (ok ? nrm : 1.0);
    final double vx = ok ? cx * inv : 1.0, vy = ok ? cy * inv : 0.0, vz = ok ? cz * inv : 0.0;
    // ---- orthonormal complement (u, w) of v, 2x2 Jacobi rotation in span(u, w)
    final double ax = Math.abs(vx), ay = Math.abs(vy), az = Math.abs(vz);
    final boolean k1 = ay < ax;
    final boolean k2 = az < (k1 ? ay : ax);
    final double ex = (k1 | k2) ? 0.0 : 1.0, ey = (k1 & !k2) ? 1.0 : 0.0, ez = k2 ? 1.0 : 0.0;
    final double vk = ex * vx + ey * vy + ez * vz;
    double ux = ex - vk * vx, uy = ey - vk * vy, uz = ez - vk * vz;
    final double nu = Math.sqrt(ux * ux + uy * uy + uz * uz);
    final boolean oku = nu > 1e-300;
    final double iu = 1.0 / (oku ? nu : 1.0);
    ux = oku ? ux * iu : 0.0; uy = oku ? uy * iu : 0.0; uz = oku ? uz * iu : 0.0;
    final double wx = vy * uz - vz * uy, wy = vz * ux - vx * uz, wz = vx * uy - vy * ux;
    // Rayleigh quotients of A (eigenvalues; accurate for spectra over orders of magnitude) and of the deviator A - m I
    // (rotation angle, and x = lambda - m for log near I)
    final double Aux = a0 * ux + a3 * uy + a4 * uz, Auy = a3 * ux + a1 * uy + a5 * uz, Auz = a4 * ux + a5 * uy + a2 * uz;
    final double Awx = a0 * wx + a3 * wy + a4 * wz, Awy = a3 * wx + a1 * wy + a5 * wz, Awz = a4 * wx + a5 * wy + a2 * wz;
    final double Avx = a0 * vx + a3 * vy + a4 * vz, Avy = a3 * vx + a1 * vy + a5 * vz, Avz = a4 * vx + a5 * vy + a2 * vz;
    final double a11 = ux * Aux + uy * Auy + uz * Auz, a12 = ux * Awx + uy * Awy + uz * Awz;
    final double a22 = wx * Awx + wy * Awy + wz * Awz, lv = vx * Avx + vy * Avy + vz * Avz;
    final double Dux = d0 * ux + a3 * uy + a4 * uz, Duy = a3 * ux + d1 * uy + a5 * uz, Duz = a4 * ux + a5 * uy + d2 * uz;
    final double Dwx = d0 * wx + a3 * wy + a4 * wz, Dwy = a3 * wx + d1 * wy + a5 * wz, Dwz = a4 * wx + a5 * wy + d2 * wz;
    final double b11 = ux * Dux + uy * Duy + uz * Duz, b12 = ux * Dwx + uy * Dwy + uz * Dwz;
    final double b22 = wx * Dwx + wy * Dwy + wz * Dwz;
    // theta = atan2(y, x) / 2 in (-pi/2, pi/2]:  cos = sqrt((r + x) / 2r) >= 0,  sin = y / (2 r cos)  (x >= 0, else swapped)
    final double jx = b11 - b22, jy = 2.0 * b12, jr = Math.sqrt(jx * jx + jy * jy);   // angle from the deviator
    final boolean jz = jr == 0, jpos = jx >= 0;
    final double jrs = jz ? 1.0 : jr;
    final double big = Math.sqrt((jrs + Math.abs(jx)) / (2.0 * jrs));
    final double sml = jy / (2.0 * jrs * big);
    final double sgn = (jy > 0 || (jy == 0 && Double.doubleToRawLongBits(jy) >= 0)) ? 1.0 : -1.0;  // +0 -> 1, -0 -> -1
    final double cs = jz ? 1.0 : (jpos ? big : sgn * sml);
    final double sn = jz ? 0.0 : (jpos ? sml : sgn * big);
    final double l0 = lv * sig, l1 = (cs * cs * a11 + 2.0 * cs * sn * a12 + sn * sn * a22) * sig;
    final double l2 = (sn * sn * a11 - 2.0 * cs * sn * a12 + cs * cs * a22) * sig;
    // V[x][i] = vxi: component x of eigenvector i (columns v, cos u + sin w, cos w - sin u)
    final double v00 = vx, v01 = cs * ux + sn * wx, v02 = cs * wx - sn * ux;
    final double v10 = vy, v11 = cs * uy + sn * wy, v12 = cs * wy - sn * uy;
    final double v20 = vz, v21 = cs * uz + sn * wz, v22 = cs * wz - sn * uz;
    // ---- F = V diag(f) V^T
    double f0 = fun(kind, l0, al), f1 = fun(kind, l1, al), f2 = fun(kind, l2, al);
    if (kind == LOG) {    // near m: log(m sig) + log1p(x/m), the rounding of m cancels against A - m I (near I)
      final double bv = vx * (d0 * vx + a3 * vy + a4 * vz) + vy * (a3 * vx + d1 * vy + a5 * vz)
          + vz * (a4 * vx + a5 * vy + d2 * vz);
      final double x0 = bv, x1 = cs * cs * b11 + 2.0 * cs * sn * b12 + sn * sn * b22;
      final double x2 = sn * sn * b11 - 2.0 * cs * sn * b12 + cs * cs * b22;
      final double lms = Math.log(m * sig);
      if (Math.abs(x0) <= 0.5 * m) f0 = lms + Math.log1p(x0 / m);
      if (Math.abs(x1) <= 0.5 * m) f1 = lms + Math.log1p(x1 / m);
      if (Math.abs(x2) <= 0.5 * m) f2 = lms + Math.log1p(x2 / m);
    }
    F[fo] = v00 * f0 * v00 + v01 * f1 * v01 + v02 * f2 * v02;
    F[fo + 3] = v00 * f0 * v10 + v01 * f1 * v11 + v02 * f2 * v12;
    F[fo + 4] = v00 * f0 * v20 + v01 * f1 * v21 + v02 * f2 * v22;
    F[fo + 1] = v10 * f0 * v10 + v11 * f1 * v11 + v12 * f2 * v12;
    F[fo + 5] = v10 * f0 * v20 + v11 * f1 * v21 + v12 * f2 * v22;
    F[fo + 2] = v20 * f0 * v20 + v21 * f1 * v21 + v22 * f2 * v22;
    if (J == null) return;
    // ---- Daleckii-Krein: J_ab,pp = C_abpp, J_ab,pq = C_abpq + C_abqp, C_abpq = sum_i P_i[a,p] T_i[b,q],
    //      P_i = v_i v_i^T, T_i = sum_j D_ij P_j, D_ij = f[lam_i, lam_j]
    // (unrolled into local variables pi_xy = P_i[x][y], qi_xy = T_i[x][y], both symmetric in x, y: the JIT cannot
    // keep array elements in registers across the stores into J and H)
    final double D01 = divdiff(kind, l0, l1, f0, f1, al), D02 = divdiff(kind, l0, l2, f0, f2, al);
    final double D12 = divdiff(kind, l1, l2, f1, f2, al);
    final double D00 = deriv(kind, l0, f0, al), D11 = deriv(kind, l1, f1, al), D22 = deriv(kind, l2, f2, al);
    final double p0_00 = v00 * v00, p0_01 = v00 * v10, p0_02 = v00 * v20;
    final double p0_11 = v10 * v10, p0_12 = v10 * v20, p0_22 = v20 * v20;
    final double p1_00 = v01 * v01, p1_01 = v01 * v11, p1_02 = v01 * v21;
    final double p1_11 = v11 * v11, p1_12 = v11 * v21, p1_22 = v21 * v21;
    final double p2_00 = v02 * v02, p2_01 = v02 * v12, p2_02 = v02 * v22;
    final double p2_11 = v12 * v12, p2_12 = v12 * v22, p2_22 = v22 * v22;
    final double q0_00 = D00 * p0_00 + D01 * p1_00 + D02 * p2_00, q0_01 = D00 * p0_01 + D01 * p1_01 + D02 * p2_01;
    final double q0_02 = D00 * p0_02 + D01 * p1_02 + D02 * p2_02, q0_11 = D00 * p0_11 + D01 * p1_11 + D02 * p2_11;
    final double q0_12 = D00 * p0_12 + D01 * p1_12 + D02 * p2_12, q0_22 = D00 * p0_22 + D01 * p1_22 + D02 * p2_22;
    final double q1_00 = D01 * p0_00 + D11 * p1_00 + D12 * p2_00, q1_01 = D01 * p0_01 + D11 * p1_01 + D12 * p2_01;
    final double q1_02 = D01 * p0_02 + D11 * p1_02 + D12 * p2_02, q1_11 = D01 * p0_11 + D11 * p1_11 + D12 * p2_11;
    final double q1_12 = D01 * p0_12 + D11 * p1_12 + D12 * p2_12, q1_22 = D01 * p0_22 + D11 * p1_22 + D12 * p2_22;
    final double q2_00 = D02 * p0_00 + D12 * p1_00 + D22 * p2_00, q2_01 = D02 * p0_01 + D12 * p1_01 + D22 * p2_01;
    final double q2_02 = D02 * p0_02 + D12 * p1_02 + D22 * p2_02, q2_11 = D02 * p0_11 + D12 * p1_11 + D22 * p2_11;
    final double q2_12 = D02 * p0_12 + D12 * p1_12 + D22 * p2_12, q2_22 = D02 * p0_22 + D12 * p1_22 + D22 * p2_22;
    // J for the Voigt row (x, y), x <= y, column k = (p, q): sum_i P_i[x][p] T_i[y][q] (+ the same with p, q swapped)
    put2(J, jo, 0, 0, 0, dot3(p0_00, q0_00, p1_00, q1_00, p2_00, q2_00));
    put2(J, jo, 0, 0, 1, dot3(p0_01, q0_01, p1_01, q1_01, p2_01, q2_01));
    put2(J, jo, 0, 0, 2, dot3(p0_02, q0_02, p1_02, q1_02, p2_02, q2_02));
    put2(J, jo, 0, 0, 3, dot3(p0_00, q0_01, p1_00, q1_01, p2_00, q2_01)
        + dot3(p0_01, q0_00, p1_01, q1_00, p2_01, q2_00));
    put2(J, jo, 0, 0, 4, dot3(p0_00, q0_02, p1_00, q1_02, p2_00, q2_02)
        + dot3(p0_02, q0_00, p1_02, q1_00, p2_02, q2_00));
    put2(J, jo, 0, 0, 5, dot3(p0_01, q0_02, p1_01, q1_02, p2_01, q2_02)
        + dot3(p0_02, q0_01, p1_02, q1_01, p2_02, q2_01));
    put2(J, jo, 0, 1, 0, dot3(p0_00, q0_01, p1_00, q1_01, p2_00, q2_01));
    put2(J, jo, 0, 1, 1, dot3(p0_01, q0_11, p1_01, q1_11, p2_01, q2_11));
    put2(J, jo, 0, 1, 2, dot3(p0_02, q0_12, p1_02, q1_12, p2_02, q2_12));
    put2(J, jo, 0, 1, 3, dot3(p0_00, q0_11, p1_00, q1_11, p2_00, q2_11)
        + dot3(p0_01, q0_01, p1_01, q1_01, p2_01, q2_01));
    put2(J, jo, 0, 1, 4, dot3(p0_00, q0_12, p1_00, q1_12, p2_00, q2_12)
        + dot3(p0_02, q0_01, p1_02, q1_01, p2_02, q2_01));
    put2(J, jo, 0, 1, 5, dot3(p0_01, q0_12, p1_01, q1_12, p2_01, q2_12)
        + dot3(p0_02, q0_11, p1_02, q1_11, p2_02, q2_11));
    put2(J, jo, 0, 2, 0, dot3(p0_00, q0_02, p1_00, q1_02, p2_00, q2_02));
    put2(J, jo, 0, 2, 1, dot3(p0_01, q0_12, p1_01, q1_12, p2_01, q2_12));
    put2(J, jo, 0, 2, 2, dot3(p0_02, q0_22, p1_02, q1_22, p2_02, q2_22));
    put2(J, jo, 0, 2, 3, dot3(p0_00, q0_12, p1_00, q1_12, p2_00, q2_12)
        + dot3(p0_01, q0_02, p1_01, q1_02, p2_01, q2_02));
    put2(J, jo, 0, 2, 4, dot3(p0_00, q0_22, p1_00, q1_22, p2_00, q2_22)
        + dot3(p0_02, q0_02, p1_02, q1_02, p2_02, q2_02));
    put2(J, jo, 0, 2, 5, dot3(p0_01, q0_22, p1_01, q1_22, p2_01, q2_22)
        + dot3(p0_02, q0_12, p1_02, q1_12, p2_02, q2_12));
    put2(J, jo, 1, 1, 0, dot3(p0_01, q0_01, p1_01, q1_01, p2_01, q2_01));
    put2(J, jo, 1, 1, 1, dot3(p0_11, q0_11, p1_11, q1_11, p2_11, q2_11));
    put2(J, jo, 1, 1, 2, dot3(p0_12, q0_12, p1_12, q1_12, p2_12, q2_12));
    put2(J, jo, 1, 1, 3, dot3(p0_01, q0_11, p1_01, q1_11, p2_01, q2_11)
        + dot3(p0_11, q0_01, p1_11, q1_01, p2_11, q2_01));
    put2(J, jo, 1, 1, 4, dot3(p0_01, q0_12, p1_01, q1_12, p2_01, q2_12)
        + dot3(p0_12, q0_01, p1_12, q1_01, p2_12, q2_01));
    put2(J, jo, 1, 1, 5, dot3(p0_11, q0_12, p1_11, q1_12, p2_11, q2_12)
        + dot3(p0_12, q0_11, p1_12, q1_11, p2_12, q2_11));
    put2(J, jo, 1, 2, 0, dot3(p0_01, q0_02, p1_01, q1_02, p2_01, q2_02));
    put2(J, jo, 1, 2, 1, dot3(p0_11, q0_12, p1_11, q1_12, p2_11, q2_12));
    put2(J, jo, 1, 2, 2, dot3(p0_12, q0_22, p1_12, q1_22, p2_12, q2_22));
    put2(J, jo, 1, 2, 3, dot3(p0_01, q0_12, p1_01, q1_12, p2_01, q2_12)
        + dot3(p0_11, q0_02, p1_11, q1_02, p2_11, q2_02));
    put2(J, jo, 1, 2, 4, dot3(p0_01, q0_22, p1_01, q1_22, p2_01, q2_22)
        + dot3(p0_12, q0_02, p1_12, q1_02, p2_12, q2_02));
    put2(J, jo, 1, 2, 5, dot3(p0_11, q0_22, p1_11, q1_22, p2_11, q2_22)
        + dot3(p0_12, q0_12, p1_12, q1_12, p2_12, q2_12));
    put2(J, jo, 2, 2, 0, dot3(p0_02, q0_02, p1_02, q1_02, p2_02, q2_02));
    put2(J, jo, 2, 2, 1, dot3(p0_12, q0_12, p1_12, q1_12, p2_12, q2_12));
    put2(J, jo, 2, 2, 2, dot3(p0_22, q0_22, p1_22, q1_22, p2_22, q2_22));
    put2(J, jo, 2, 2, 3, dot3(p0_02, q0_12, p1_02, q1_12, p2_02, q2_12)
        + dot3(p0_12, q0_02, p1_12, q1_02, p2_12, q2_02));
    put2(J, jo, 2, 2, 4, dot3(p0_02, q0_22, p1_02, q1_22, p2_02, q2_22)
        + dot3(p0_22, q0_02, p1_22, q1_02, p2_22, q2_02));
    put2(J, jo, 2, 2, 5, dot3(p0_12, q0_22, p1_12, q1_22, p2_12, q2_22)
        + dot3(p0_22, q0_12, p1_22, q1_12, p2_22, q2_12));
    if (H == null) return;
    // ---- second order: H^pq = V S^pq V^T,  S^pq = X + X^T,  X_ab = sum_m D2_amb Et^p_am Et^q_mb,  Et^p = V^T E_p V
    // D2 is symmetric in its three indices: dijm = f[lam_i, lam_j, lam_m] for i <= j <= m
    final double d000 = divdiff2(kind, l0, l0, l0, f0, f0, f0, D00, D00, D00, al, c2);
    final double d001 = divdiff2(kind, l0, l0, l1, f0, f0, f1, D00, D01, D01, al, c2);
    final double d002 = divdiff2(kind, l0, l0, l2, f0, f0, f2, D00, D02, D02, al, c2);
    final double d011 = divdiff2(kind, l0, l1, l1, f0, f1, f1, D01, D01, D11, al, c2);
    final double d012 = divdiff2(kind, l0, l1, l2, f0, f1, f2, D01, D02, D12, al, c2);
    final double d022 = divdiff2(kind, l0, l2, l2, f0, f2, f2, D02, D02, D22, al, c2);
    final double d111 = divdiff2(kind, l1, l1, l1, f1, f1, f1, D11, D11, D11, al, c2);
    final double d112 = divdiff2(kind, l1, l1, l2, f1, f1, f2, D11, D12, D12, al, c2);
    final double d122 = divdiff2(kind, l1, l2, l2, f1, f2, f2, D12, D12, D22, al, c2);
    final double d222 = divdiff2(kind, l2, l2, l2, f2, f2, f2, D22, D22, D22, al, c2);
    final double[] V = wk.V, Et = wk.Et;                 // V[3x+i] = vxi;  Et[9p+3a+m] = (V^T E_p V)_am
    V[0] = v00; V[1] = v01; V[2] = v02; V[3] = v10; V[4] = v11; V[5] = v12; V[6] = v20; V[7] = v21; V[8] = v22;
    for (int pp = 0; pp < 6; pp++) {
      final int k = VP[pp], l = VQ[pp];
      for (int a = 0; a < 3; a++)
        for (int mm = a; mm < 3; mm++) {
          double e = V[3 * k + a] * V[3 * l + mm];
          if (k != l) e += V[3 * l + a] * V[3 * k + mm];
          Et[9 * pp + 3 * a + mm] = Et[9 * pp + 3 * mm + a] = e;
        }
    }
    for (int pp = 0; pp < 6; pp++) {
      final int ep = 9 * pp;
      for (int qq = pp; qq < 6; qq++) {
        final int eq = 9 * qq;
        final double e00 = Et[ep], e01 = Et[ep + 1], e02 = Et[ep + 2], e11 = Et[ep + 4], e12 = Et[ep + 5];
        final double e22 = Et[ep + 8];
        final double g00 = Et[eq], g01 = Et[eq + 1], g02 = Et[eq + 2], g11 = Et[eq + 4], g12 = Et[eq + 5];
        final double g22 = Et[eq + 8];
        // S = X + X^T (sab), X_ab = sum_m D2_amb Et^p_am Et^q_mb
        final double x00 = d000 * e00 * g00 + d001 * e01 * g01 + d002 * e02 * g02;
        final double x11 = d011 * e01 * g01 + d111 * e11 * g11 + d112 * e12 * g12;
        final double x22 = d022 * e02 * g02 + d122 * e12 * g12 + d222 * e22 * g22;
        final double s00 = x00 + x00, s11 = x11 + x11, s22 = x22 + x22;
        final double s01 = d001 * e00 * g01 + d011 * e01 * g11 + d012 * e02 * g12
            + (d001 * e01 * g00 + d011 * e11 * g01 + d012 * e12 * g02);
        final double s02 = d002 * e00 * g02 + d012 * e01 * g12 + d022 * e02 * g22
            + (d002 * e02 * g00 + d012 * e12 * g01 + d022 * e22 * g02);
        final double s12 = d012 * e01 * g02 + d112 * e11 * g12 + d122 * e12 * g22
            + (d012 * e02 * g01 + d112 * e12 * g11 + d122 * e22 * g12);
        // W = S V^T, h = V W
        final double w00 = s00 * v00 + s01 * v01 + s02 * v02;
        final double w01 = s00 * v10 + s01 * v11 + s02 * v12;
        final double w02 = s00 * v20 + s01 * v21 + s02 * v22;
        final double w10 = s01 * v00 + s11 * v01 + s12 * v02;
        final double w11 = s01 * v10 + s11 * v11 + s12 * v12;
        final double w12 = s01 * v20 + s11 * v21 + s12 * v22;
        final double w20 = s02 * v00 + s12 * v01 + s22 * v02;
        final double w21 = s02 * v10 + s12 * v11 + s22 * v12;
        final double w22 = s02 * v20 + s12 * v21 + s22 * v22;
        final int o1 = ho + 6 * pp + qq, o2 = ho + 6 * qq + pp;   // H[(6r+p)*6+q] = H[o1 + 36 r]
        put4(H, o1, o2, 0, 0, v00 * w00 + v01 * w10 + v02 * w20);
        put4(H, o1, o2, 0, 1, v00 * w01 + v01 * w11 + v02 * w21);
        put4(H, o1, o2, 0, 2, v00 * w02 + v01 * w12 + v02 * w22);
        put4(H, o1, o2, 1, 1, v10 * w01 + v11 * w11 + v12 * w21);
        put4(H, o1, o2, 1, 2, v10 * w02 + v11 * w12 + v12 * w22);
        put4(H, o1, o2, 2, 2, v20 * w02 + v21 * w12 + v22 * w22);
      }
    }
  }

  // Voigt index of (x, y), x <= y
  private static int vidx(int x, int y) {
    return x == y ? x : x + y + 2;
  }

  // J_r,k = s, r the Voigt row of (x, y)
  private static void put2(double[] J, int jo, int x, int y, int k, double s) {
    J[jo + 6 * vidx(x, y) + k] = s;
  }

  private static double dot3(double a0, double b0, double a1, double b1, double a2, double b2) {
    return a0 * b0 + a1 * b1 + a2 * b2;
  }

  // H_r,pq = H_r,qp = h, r the Voigt row of (i, j)  (o1 = hOff + 6p + q, o2 = hOff + 6q + p)
  private static void put4(double[] H, int o1, int o2, int i, int j, double h) {
    final int r = 36 * vidx(i, j);
    H[o1 + r] = H[o2 + r] = h;
  }
}
