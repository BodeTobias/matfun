package io.github.bodetobias.matfun;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import io.github.bodetobias.matfun.SymMatFun.Kind;

/** Cross-language reference cases testdata/cases.txt (C kernel) and API checks. */
class SymMatFunTest {

  /** one line of cases.txt: kind alpha tol v[6] F[6] J[36] H[216] (Voigt layout) */
  static final class Case {
    int line, kind;
    double alpha, tol;
    double[] v = new double[6], F = new double[6], J = new double[36], H = new double[216];
  }

  static List<Case> cases;

  static Path casesPath() {
    String p = System.getProperty("matfun.cases");
    return p != null ? Paths.get(p) : Paths.get("..", "testdata", "cases.txt");
  }

  @BeforeAll
  static void load() throws IOException {
    cases = new ArrayList<>();
    List<String> lines = Files.readAllLines(casesPath());
    for (int ln = 0; ln < lines.size(); ln++) {
      String s = lines.get(ln).trim();
      if (s.isEmpty() || s.startsWith("#")) continue;
      String[] t = s.split("\\s+");
      assertEquals(267, t.length, "numbers in line " + (ln + 1));
      double[] x = new double[t.length];
      for (int i = 0; i < t.length; i++) x[i] = Double.parseDouble(t[i]);
      Case c = new Case();
      c.line = ln + 1;
      c.kind = (int) x[0];
      c.alpha = x[1];
      c.tol = x[2];
      System.arraycopy(x, 3, c.v, 0, 6);
      System.arraycopy(x, 9, c.F, 0, 6);
      System.arraycopy(x, 15, c.J, 0, 36);
      System.arraycopy(x, 51, c.H, 0, 216);
      cases.add(c);
    }
  }

  /** max|x - ref| / max|ref| */
  static double relErr(double[] x, double[] ref) {
    double num = 0, den = 0;
    for (int i = 0; i < ref.length; i++) {
      double d = Math.abs(x[i] - ref[i]);
      num = Double.isNaN(d) || d > num ? d : num;
      den = Math.max(den, Math.abs(ref[i]));
    }
    return num / den;
  }

  static String label(Case c) {
    return c.kind == 2 ? "pow " + c.alpha : Kind.fromCode(c.kind).name().toLowerCase(Locale.ROOT);
  }

  @Test
  void referenceCases() {
    assertTrue(cases.size() >= 150, "number of cases in " + casesPath() + ": " + cases.size());
    // per kind label: max error of F, J, H, number of cases
    java.util.Map<String, double[]> maxErr = new java.util.LinkedHashMap<>();
    List<String> failures = new ArrayList<>();
    for (Case c : cases) {
      double[] F = new double[6], J = new double[36], H = new double[216];
      SymMatFun.eval(Kind.fromCode(c.kind), c.alpha, c.v, F, J, H);
      double eF = relErr(F, c.F), eJ = relErr(J, c.J), eH = relErr(H, c.H);
      String group = label(c) + String.format(Locale.ROOT, ", tol %.0e", c.tol);   // larger tol: graded / wide
      double[] m = maxErr.computeIfAbsent(group, k -> new double[4]);
      m[0] = Math.max(m[0], eF); m[1] = Math.max(m[1], eJ); m[2] = Math.max(m[2], eH); m[3]++;
      if (!(eF <= c.tol && eJ <= c.tol && eH <= c.tol))
        failures.add(String.format(Locale.ROOT, "line %d (%s): F %.3e J %.3e H %.3e > tol %.0e", c.line, label(c),
                                   eF, eJ, eH, c.tol));
      // F only and F+J must give the same F (and J) as the full call
      double[] F2 = new double[6], J2 = new double[36];
      SymMatFun.eval(Kind.fromCode(c.kind), c.alpha, c.v, F2, null, null);
      assertArrayEquals(F, F2, 0.0);
      SymMatFun.eval(Kind.fromCode(c.kind), c.alpha, c.v, F2, J2, null);
      assertArrayEquals(J, J2, 0.0);
    }
    System.out.println("matfun reference cases " + casesPath().normalize() + ": max|x - ref| / max|ref|");
    System.out.println(String.format(Locale.ROOT, "  %-24s %5s %10s %10s %10s", "kind", "cases", "F", "J", "H"));
    for (var e : maxErr.entrySet()) {
      double[] m = e.getValue();
      System.out.println(String.format(Locale.ROOT, "  %-24s %5d %10.2e %10.2e %10.2e", e.getKey(), (int) m[3],
                                       m[0], m[1], m[2]));
    }
    assertTrue(failures.isEmpty(), String.join("\n", failures));
  }

  /** n random SPD matrices B B^T + 0.1 I, row-major */
  static double[] spd(int n, long seed) {
    Random rng = new Random(seed);
    double[] A = new double[9 * n];
    for (int i = 0; i < n; i++) {
      double[] b = new double[9];
      for (int k = 0; k < 9; k++) b[k] = rng.nextGaussian();
      for (int x = 0; x < 3; x++)
        for (int y = 0; y < 3; y++)
          A[9 * i + 3 * x + y] = b[3 * x] * b[3 * y] + b[3 * x + 1] * b[3 * y + 1] + b[3 * x + 2] * b[3 * y + 2]
              + (x == y ? 0.1 : 0.0);
    }
    return A;
  }

  static final int[] MIDX = {0, 3, 4, 3, 1, 5, 4, 5, 2};

  static double[] voigt(double[] A) {
    int n = A.length / 9;
    double[] V = new double[6 * n];
    for (int i = 0; i < n; i++)
      for (int m = 0; m < 9; m++) V[6 * i + MIDX[m]] = A[9 * i + m];
    return V;
  }

  @Test
  void batchEqualsEval() {
    for (int n : new int[] {0, 1, 5, 300, 3000}) {           // 3000 > 1024: parallel path also without H
      double[] V = voigt(spd(n, 7));
      for (Kind kind : Kind.values()) {
        double alpha = kind == Kind.POW ? 1.7 : 0.0;
        double[] F = new double[6 * n], J = new double[36 * n], H = new double[216 * n];
        SymMatFun.eval(kind, alpha, V, F, J, H);
        double[] F1 = new double[6], J1 = new double[36], H1 = new double[216];
        for (int i = 0; i < n; i++) {
          SymMatFun.eval(kind, alpha, java.util.Arrays.copyOfRange(V, 6 * i, 6 * i + 6), F1, J1, H1);
          assertArrayEquals(F1, java.util.Arrays.copyOfRange(F, 6 * i, 6 * i + 6), 0.0, kind + " F, n = " + n);
          assertArrayEquals(J1, java.util.Arrays.copyOfRange(J, 36 * i, 36 * i + 36), 0.0, kind + " J, n = " + n);
          assertArrayEquals(H1, java.util.Arrays.copyOfRange(H, 216 * i, 216 * i + 216), 0.0, kind + " H, n = " + n);
        }
        double[] F2 = new double[6 * n], J2 = new double[36 * n];
        SymMatFun.eval(kind, alpha, V, F2, null, null);
        assertArrayEquals(F, F2, 0.0, kind + " F only, n = " + n);
        SymMatFun.eval(kind, alpha, V, F2, J2, null);
        assertArrayEquals(J, J2, 0.0, kind + " F+J, n = " + n);
      }
    }
  }

  @Test
  void sqrtEqualsPowHalfAndAlphaIgnored() {
    double[] v = {2.0, 1.5, 1.0, 0.3, -0.2, 0.1};
    double[] F = new double[6], J = new double[36], H = new double[216];
    double[] F2 = new double[6], J2 = new double[36], H2 = new double[216];
    SymMatFun.eval(Kind.SQRT, 0.0, v, F, J, H);
    SymMatFun.eval(Kind.SQRT, 3.0, v, F2, J2, H2);           // alpha ignored for sqrt
    assertArrayEquals(F, F2, 0.0);
    assertArrayEquals(H, H2, 0.0);
    SymMatFun.eval(Kind.POW, 0.5, v, F2, J2, H2);            // same values via the pow branch (not bitwise)
    assertTrue(relErr(F2, F) < 1e-14 && relErr(J2, J) < 1e-14 && relErr(H2, H) < 1e-13);
    SymMatFun.eval(Kind.EXP, 0.0, v, F, J, H);
    SymMatFun.eval(Kind.EXP, -5.0, v, F2, J2, H2);           // alpha ignored for exp
    assertArrayEquals(H, H2, 0.0);
  }

  @Test
  void matrixInput() {
    // the matrix results are the Voigt ones with dv/dA_cd = 1 (diagonal) or 1/2 (exact), symmetric in the index pairs
    final int n = 300;
    double[] A = spd(n, 3), V = voigt(A);
    double[] F = new double[6 * n], J = new double[36 * n], H = new double[216 * n];
    double[] Fm = new double[9 * n], Jm = new double[81 * n], Hm = new double[729 * n];
    SymMatFun.eval(Kind.LOG, 0.0, V, F, J, H);
    SymMatFun.evalMatrix(Kind.LOG, 0.0, A, Fm, Jm, Hm);
    double asym = 0, scale = 0;
    for (int i = 0; i < n; i++)
      for (int m = 0; m < 9; m++) {
        assertEquals(F[6 * i + MIDX[m]], Fm[9 * i + m], 0.0);
        int a = m / 3, b = m % 3, mt = 3 * b + a;
        for (int c = 0; c < 9; c++) {
          double w = c % 4 == 0 ? 1.0 : 0.5;
          assertEquals(J[36 * i + 6 * MIDX[m] + MIDX[c]] * w, Jm[81 * i + 9 * m + c], 0.0);
          int ct = 3 * (c % 3) + c / 3;
          double j0 = Jm[81 * i + 9 * m + c];
          asym = Math.max(asym, Math.max(Math.abs(j0 - Jm[81 * i + 9 * mt + c]), Math.abs(j0 - Jm[81 * i + 9 * m + ct])));
          asym = Math.max(asym, Math.abs(j0 - Jm[81 * i + 9 * c + m]));
          scale = Math.max(scale, Math.abs(j0));
          for (int e = 0; e < 9; e++) {
            double we = e % 4 == 0 ? 1.0 : 0.5;
            assertEquals(H[216 * i + 36 * MIDX[m] + 6 * MIDX[c] + MIDX[e]] * w * we, Hm[729 * i + 81 * m + 9 * c + e], 0.0);
          }
        }
      }
    assertTrue(asym <= 1e-15 * scale, "symmetries of J: " + asym);
    double[] N = A.clone();
    N[9 * 17 + 1] += 1e-13;                                   // round-off asymmetry is accepted
    SymMatFun.evalMatrix(Kind.LOG, 0.0, N, Fm, null, null);
    N[9 * 17 + 1] += 0.1;                                     // a non-symmetric matrix is not
    assertThrows(IllegalArgumentException.class, () -> SymMatFun.evalMatrix(Kind.LOG, 0.0, N, Fm, null, null));
    assertThrows(IllegalArgumentException.class, () -> SymMatFun.evalMatrix(Kind.EXP, 0, new double[10], Fm, null, null));
  }

  @Test
  void argumentChecks() {
    double[] v = {1, 2, 3, 0, 0, 0}, F = new double[6], J = new double[36], H = new double[216];
    assertThrows(IllegalArgumentException.class, () -> SymMatFun.eval(Kind.EXP, 0, v, F, null, H));
    assertThrows(IllegalArgumentException.class, () -> SymMatFun.eval(Kind.POW, 0.0, v, F, J, H));
    assertThrows(IllegalArgumentException.class, () -> SymMatFun.eval(Kind.POW, -1.0, v, F, null, null));
    assertThrows(IllegalArgumentException.class, () -> SymMatFun.eval(Kind.POW, Double.NaN, v, F, null, null));
    assertThrows(IllegalArgumentException.class, () -> SymMatFun.eval(Kind.EXP, 0, new double[7], F, null, null));
    assertThrows(IllegalArgumentException.class, () -> Kind.fromCode(4));
    assertThrows(NullPointerException.class, () -> SymMatFun.eval(null, 0, v, F, null, null));
    assertThrows(IndexOutOfBoundsException.class, () -> SymMatFun.eval(Kind.EXP, 0, v, F, new double[35], null));
    assertThrows(IndexOutOfBoundsException.class, () -> SymMatFun.eval(Kind.EXP, 0, new double[12], F, null, null));
    for (Kind k : Kind.values()) assertEquals(k, Kind.fromCode(k.code));
    assertEquals(0, Kind.EXP.code); assertEquals(1, Kind.LOG.code);
    assertEquals(2, Kind.POW.code); assertEquals(3, Kind.SQRT.code);
    SymMatFun.eval(Kind.EXP, 0, new double[0], new double[0], null, null);   // n = 0 is fine
  }
}
