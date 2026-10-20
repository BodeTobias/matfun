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

/** Cross-language reference cases testdata/cases_general.txt (C kernel, general matrices) and API checks. */
class GenMatFunTest {

  /** one line of cases_general.txt: kind alpha tol A[9] F[9] J[81] H[729] (matrix layout) */
  static final class Case {
    int line, kind;
    double alpha, tol;
    double[] A = new double[9], F = new double[9], J = new double[81], H = new double[729];
  }

  static List<Case> cases;

  static Path casesPath() {
    String p = System.getProperty("matfun.cases.general");
    return p != null ? Paths.get(p) : Paths.get("..", "testdata", "cases_general.txt");
  }

  @BeforeAll
  static void load() throws IOException {
    cases = new ArrayList<>();
    List<String> lines = Files.readAllLines(casesPath());
    for (int ln = 0; ln < lines.size(); ln++) {
      String s = lines.get(ln).trim();
      if (s.isEmpty() || s.startsWith("#")) continue;
      String[] t = s.split("\\s+");
      assertEquals(831, t.length, "numbers in line " + (ln + 1));
      double[] x = new double[t.length];
      for (int i = 0; i < t.length; i++) x[i] = Double.parseDouble(t[i]);
      Case c = new Case();
      c.line = ln + 1;
      c.kind = (int) x[0];
      c.alpha = x[1];
      c.tol = x[2];
      System.arraycopy(x, 3, c.A, 0, 9);
      System.arraycopy(x, 12, c.F, 0, 9);
      System.arraycopy(x, 21, c.J, 0, 81);
      System.arraycopy(x, 102, c.H, 0, 729);
      cases.add(c);
    }
  }

  @Test
  void referenceCases() {
    assertTrue(cases.size() >= 150, "number of cases in " + casesPath() + ": " + cases.size());
    java.util.Map<String, double[]> maxErr = new java.util.LinkedHashMap<>();
    List<String> failures = new ArrayList<>();
    for (Case c : cases) {
      double[] F = new double[9], J = new double[81], H = new double[729];
      GenMatFun.eval(Kind.fromCode(c.kind), c.alpha, c.A, F, J, H);
      double eF = SymMatFunTest.relErr(F, c.F), eJ = SymMatFunTest.relErr(J, c.J), eH = SymMatFunTest.relErr(H, c.H);
      String label = c.kind == 2 ? "pow " + c.alpha : Kind.fromCode(c.kind).name().toLowerCase(Locale.ROOT);
      String group = label + String.format(Locale.ROOT, ", tol %.0e", c.tol);
      double[] m = maxErr.computeIfAbsent(group, k -> new double[4]);
      m[0] = Math.max(m[0], eF); m[1] = Math.max(m[1], eJ); m[2] = Math.max(m[2], eH); m[3]++;
      if (!(eF <= c.tol && eJ <= c.tol && eH <= c.tol))
        failures.add(String.format(Locale.ROOT, "line %d (%s): F %.3e J %.3e H %.3e > tol %.0e", c.line, label, eF,
                                   eJ, eH, c.tol));
      double[] F2 = new double[9], J2 = new double[81];
      GenMatFun.eval(Kind.fromCode(c.kind), c.alpha, c.A, F2, null, null);
      assertArrayEquals(F, F2, 0.0);
      GenMatFun.eval(Kind.fromCode(c.kind), c.alpha, c.A, F2, J2, null);
      assertArrayEquals(J, J2, 0.0);
    }
    System.out.println("matfun general reference cases " + casesPath().normalize() + ": max|x - ref| / max|ref|");
    System.out.println(String.format(Locale.ROOT, "  %-24s %5s %10s %10s %10s", "kind", "cases", "F", "J", "H"));
    for (var e : maxErr.entrySet()) {
      double[] m = e.getValue();
      System.out.println(String.format(Locale.ROOT, "  %-24s %5d %10.2e %10.2e %10.2e", e.getKey(), (int) m[3],
                                       m[0], m[1], m[2]));
    }
    assertTrue(failures.isEmpty(), String.join("\n", failures));
  }

  @Test
  void batchEqualsEvalAndDomain() {
    Random rng = new Random(11);
    for (int n : new int[] {0, 1, 5, 300, 3000}) {
      double[] A = new double[9 * n];
      for (int i = 0; i < n; i++)
        for (int k = 0; k < 9; k++) A[9 * i + k] = 0.3 * rng.nextGaussian() + (k % 4 == 0 ? 1.5 : 0.0);
      if (n > 3) {                                          // real eigenvalue -1: NaN for log only in matrix 3
        double[] neg = {-1, 0, 0, 0, 2, 0, 0, 0, 3};
        System.arraycopy(neg, 0, A, 27, 9);
      }
      double[] F = new double[9 * n], J = new double[81 * n], H = new double[729 * n];
      GenMatFun.eval(Kind.LOG, 0.0, A, F, J, H);
      for (int i = 0; i < Math.min(n, 40); i++) {
        double[] a = java.util.Arrays.copyOfRange(A, 9 * i, 9 * i + 9), f = new double[9], j = new double[81],
            h = new double[729];
        GenMatFun.eval(Kind.LOG, 0.0, a, f, j, h);
        assertArrayEquals(f, java.util.Arrays.copyOfRange(F, 9 * i, 9 * i + 9), 0.0);
        assertArrayEquals(j, java.util.Arrays.copyOfRange(J, 81 * i, 81 * i + 81), 0.0);
        assertArrayEquals(h, java.util.Arrays.copyOfRange(H, 729 * i, 729 * i + 729), 0.0);
        assertEquals(i == 3, Double.isNaN(f[0]), "NaN only for the matrix with eigenvalue -1, matrix " + i);
      }
    }
  }

  @Test
  void closedForms() {
    double th = 0.9, c = Math.cos(th), s = Math.sin(th);
    double[] K = {0, -th, 0, th, 0, 0, 0, 0, 0}, R = {c, -s, 0, s, c, 0, 0, 0, 1};
    double[] N = {0, 0.7, 0, 0, 0, 0, 0, 0, 0}, IN = {1, 0.7, 0, 0, 1, 0, 0, 0, 1}, F = new double[9];
    GenMatFun.eval(Kind.EXP, 0, K, F, null, null);
    assertTrue(SymMatFunTest.relErr(F, R) < 1e-15);
    GenMatFun.eval(Kind.LOG, 0, R, F, null, null);
    assertTrue(SymMatFunTest.relErr(F, K) < 1e-15);
    GenMatFun.eval(Kind.EXP, 0, N, F, null, null);
    assertTrue(SymMatFunTest.relErr(F, IN) < 1e-15);
    GenMatFun.eval(Kind.LOG, 0, IN, F, null, null);
    assertTrue(SymMatFunTest.relErr(F, N) < 1e-15);
  }

  @Test
  void argumentChecks() {
    double[] A = {1, 0, 0, 0, 1, 0, 0, 0, 1}, F = new double[9], H = new double[729];
    assertThrows(IllegalArgumentException.class, () -> GenMatFun.eval(Kind.POW, 0.0, A, F, null, null));
    assertThrows(IllegalArgumentException.class, () -> GenMatFun.eval(Kind.EXP, 0.0, A, F, null, H));
    assertThrows(IllegalArgumentException.class, () -> GenMatFun.eval(Kind.EXP, 0.0, new double[8], F, null, null));
    assertThrows(IndexOutOfBoundsException.class, () -> GenMatFun.eval(Kind.EXP, 0.0, A, new double[8], null, null));
  }
}
