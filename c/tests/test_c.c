/* test_c.c -- checks of the C API: closed-form cases, J and H against central differences, the matrix layout,
 * error codes; gmf_eval: closed forms (defective, rotation), J and H against central differences, batch == single.
 * The cross-language reference cases are checked in python/tests (testdata/cases.txt, cases_general.txt). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matfun/matfun.h"

static int fails = 0;
static void check(const char *what, double err, double tol) {
  int ok = err <= tol;
  printf("%-58s %.1e %s\n", what, err, ok ? "ok" : "FAIL");
  if (!ok) fails++;
}

static double maxrel(const double *a, const double *b, int n) {
  double d = 0, s = 0;
  for (int i = 0; i < n; i++) {
    d = fmax(d, fabs(a[i] - b[i]));
    s = fmax(s, fabs(b[i]));
  }
  return d / (s > 0 ? s : 1);
}

int main(void) {
  const int kinds[5] = {SMF_EXP, SMF_LOG, SMF_SQRT, SMF_POW, SMF_POW};
  const double alphas[5] = {0, 0, 0, 0.3, 1.7};
  const char *names[5] = {"exp", "log", "sqrt", "pow 0.3", "pow 1.7"};
  char buf[128];
  printf("matfun %s, SIMD %s\n", smf_version(), smf_simd_name());
  /* MATFUN_SIMD requests one kernel variant; if this CPU (or its libmvec) cannot run it, the library falls back to
     another one: report the test as skipped (exit code 77, SKIP_RETURN_CODE in CMakeLists.txt) */
  const char *want = getenv("MATFUN_SIMD");
  if (want && *want && strcmp(want, smf_simd_name()) != 0) {
    printf("variant %s not available here (using %s): skipped\n", want, smf_simd_name());
    return 77;
  }

  /* diagonal matrix: f applied to the diagonal */
  double xd[6] = {0.5, 2.0, 3.0, 0, 0, 0}, F[6];
  for (int k = 0; k < 5; k++) {
    smf_eval(kinds[k], alphas[k], 1, xd, F, NULL, NULL);
    double ref[6] = {0};
    for (int i = 0; i < 3; i++) {
      double l = xd[i];
      ref[i] = k == 0 ? exp(l) : k == 1 ? log(l) : k == 2 ? sqrt(l) : pow(l, alphas[k]);
    }
    snprintf(buf, sizeof buf, "%s: diagonal matrix", names[k]);
    check(buf, maxrel(F, ref, 6), 4e-15);   /* eigenvalues from the trigonometric formula: a few ulp */
  }

  /* exp(log(A)) = A */
  double x[6] = {2.0, 1.5, 1.2, 0.3, -0.2, 0.1}, L[6], E[6];
  smf_eval(SMF_LOG, 0, 1, x, L, NULL, NULL);
  smf_eval(SMF_EXP, 0, 1, L, E, NULL, NULL);
  check("exp(log(A)) = A", maxrel(E, x, 6), 1e-14);

  /* J and H against central differences (h = 1e-5: error ~ h^2) for a matrix with distinct eigenvalues and one
     with a double eigenvalue (2, 2, 5) */
  double xs[2][6] = {{2.0, 1.5, 1.2, 0.3, -0.2, 0.1}, {3.0, 3.0, 3.0, 1.0, 1.0, 1.0}};
  for (int s = 0; s < 2; s++)
    for (int k = 0; k < 5; k++) {
      double F0[6], J0[36], H0[216], Jfd[36], Hfd[216], Fp[6], Fm[6], Jp[36], Jm[36];
      const double h = 1e-5;
      smf_eval(kinds[k], alphas[k], 1, xs[s], F0, J0, H0);
      for (int p = 0; p < 6; p++) {
        double xp[6], xm[6];
        memcpy(xp, xs[s], sizeof xp);
        memcpy(xm, xs[s], sizeof xm);
        xp[p] += h;
        xm[p] -= h;
        smf_eval(kinds[k], alphas[k], 1, xp, Fp, Jp, NULL);
        smf_eval(kinds[k], alphas[k], 1, xm, Fm, Jm, NULL);
        for (int i = 0; i < 6; i++) Jfd[6 * i + p] = (Fp[i] - Fm[i]) / (2 * h);
        for (int i = 0; i < 6; i++)
          for (int q = 0; q < 6; q++) Hfd[36 * i + 6 * q + p] = (Jp[6 * i + q] - Jm[6 * i + q]) / (2 * h);
      }
      snprintf(buf, sizeof buf, "%s, %s: J vs central differences", names[k], s ? "eigenvalues 2, 2, 5" : "random");
      check(buf, maxrel(J0, Jfd, 36), 1e-8);
      snprintf(buf, sizeof buf, "%s, %s: H vs central differences of J", names[k], s ? "eigenvalues 2, 2, 5" : "random");
      check(buf, maxrel(H0, Hfd, 216), 1e-8);
    }

  /* batch == single calls (SIMD tail handling), n = 37 */
  enum { N = 37 };
  double *xb = malloc(sizeof(double) * 6 * N), *Fb = malloc(sizeof(double) * 6 * N), *Jb = malloc(sizeof(double) * 36 * N);
  double *Hb = malloc(sizeof(double) * 216 * N);
  for (int i = 0; i < N; i++) {
    double t = 0.1 * i;
    double v[6] = {2 + sin(t), 1.5 + cos(t), 1.2 + 0.5 * sin(2 * t), 0.3 * cos(3 * t), -0.2 * sin(t), 0.1};
    memcpy(xb + 6 * i, v, sizeof v);
  }
  smf_eval(SMF_LOG, 0, N, xb, Fb, Jb, Hb);
  double err = 0;
  for (int i = 0; i < N; i++) {
    double F1[6], J1[36], H1[216];
    smf_eval(SMF_LOG, 0, 1, xb + 6 * i, F1, J1, H1);
    err = fmax(err, fmax(maxrel(Fb + 6 * i, F1, 6), fmax(maxrel(Jb + 36 * i, J1, 36), maxrel(Hb + 216 * i, H1, 216))));
  }
  check("batch of 37 == single calls (F, J, H)", err, 0);
  free(xb); free(Fb); free(Jb); free(Hb);

  /* matrix layout: smf_eval_matrix == the Voigt results with the weights 1 (diagonal) and 1/2 (off-diagonal) of
     dv / dA_cd (exact), symmetries of the index pairs, batch of 37 with non-symmetric matrices -> -3 and NaN */
  {
    static const int vi[9] = {0, 3, 4, 3, 1, 5, 4, 5, 2};
    double Fv[6], Jv[36], Hv[216], Fm[9], Jm[81], Hm[729], Fr[9], Jr[81], Hr[729];
    double As[9] = {x[0], x[3], x[4], x[3], x[1], x[5], x[4], x[5], x[2]};
    smf_eval(SMF_LOG, 0, 1, x, Fv, Jv, Hv);
    int r = smf_eval_matrix(SMF_LOG, 0, 1, As, Fm, Jm, Hm);
    for (int m = 0; m < 9; m++) {
      Fr[m] = Fv[vi[m]];
      for (int c = 0; c < 9; c++) {
        Jr[9 * m + c] = Jv[6 * vi[m] + vi[c]] * (c % 4 ? 0.5 : 1.0);
        for (int e = 0; e < 9; e++)
          Hr[81 * m + 9 * c + e] = Hv[36 * vi[m] + 6 * vi[c] + vi[e]] * (c % 4 ? 0.5 : 1.0) * (e % 4 ? 0.5 : 1.0);
      }
    }
    check("smf_eval_matrix == smf_eval with weights (F, J, H)",
          r == 0 ? fmax(maxrel(Fm, Fr, 9), fmax(maxrel(Jm, Jr, 81), maxrel(Hm, Hr, 729))) : 1.0, 0);
    double asym = 0;               /* J_abcd = J_bacd = J_abdc = J_cdab; H likewise in each pair and (cd) <-> (ef) */
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++)
        for (int c = 0; c < 3; c++)
          for (int d = 0; d < 3; d++) {
            const int ab = 3 * a + b, ba = 3 * b + a, cd = 3 * c + d, dc = 3 * d + c;
            asym = fmax(asym, fmax(fabs(Jm[9 * ab + cd] - Jm[9 * ba + cd]), fabs(Jm[9 * ab + cd] - Jm[9 * ab + dc])));
            asym = fmax(asym, fabs(Jm[9 * ab + cd] - Jm[9 * cd + ab]));
            for (int ef = 0; ef < 9; ef++) {
              const double h0 = Hm[81 * ab + 9 * cd + ef];
              asym = fmax(asym, fmax(fabs(h0 - Hm[81 * ba + 9 * cd + ef]), fabs(h0 - Hm[81 * ab + 9 * dc + ef])));
              asym = fmax(asym, fabs(h0 - Hm[81 * ab + 9 * ef + cd]));
            }
          }
    check("matrix layout: symmetries of J and H", asym, 1e-15);
    double *Ab = malloc(sizeof(double) * 9 * N), *Fmb = malloc(sizeof(double) * 9 * N);
    for (int i = 0; i < N; i++) memcpy(Ab + 9 * i, As, sizeof As);
    Ab[9 * 5 + 1] += 1e-12;                                   /* tiny asymmetry (round-off): accepted */
    Ab[9 * 7 + 1] += 0.1;                                     /* clearly non-symmetric */
    Ab[9 * 30 + 5] -= 0.1;
    r = smf_eval_matrix(SMF_LOG, 0, N, Ab, Fmb, NULL, NULL);
    int ok = r == -3;
    for (int i = 0; i < N; i++)
      for (int m = 0; m < 9; m++) ok &= (i == 7 || i == 30) ? isnan(Fmb[9 * i + m]) : fabs(Fmb[9 * i + m] - Fm[m]) < 1e-12;
    check("non-symmetric matrices in a batch -> -3, their results NaN", ok ? 0 : 1, 0);
    free(Ab); free(Fmb);
  }

  /* extreme magnitudes (A scaled by a power of two inside the kernel): sqrt(sA) = sqrt(s) sqrt(A),
     log(sA) = log(s) I + log(A); derivatives w.r.t. s v: J_sqrt / sqrt(s), H_sqrt / s^1.5, J_log / s */
  {
    double F0[6], J0[36], H0[216], L0[6], JL0[36];
    smf_eval(SMF_SQRT, 0, 1, x, F0, J0, H0);
    smf_eval(SMF_LOG, 0, 1, x, L0, JL0, NULL);
    const double scales[4] = {1e100, 1e-100, 1e250, 1e-250};
    for (int k = 0; k < 4; k++) {
      const double s = scales[k];
      double xs[6], F[6], J[36], H[216], L[6], JL[36], Fr[6], Jr[36], Hr[216], Lr[6], JLr[36];
      for (int i = 0; i < 6; i++) xs[i] = s * x[i];
      smf_eval(SMF_SQRT, 0, 1, xs, F, J, k < 2 ? H : NULL);
      smf_eval(SMF_LOG, 0, 1, xs, L, JL, NULL);
      for (int i = 0; i < 6; i++) Fr[i] = sqrt(s) * F0[i], Lr[i] = L0[i] + (i < 3 ? log(s) : 0.0);
      for (int i = 0; i < 36; i++) Jr[i] = J0[i] / sqrt(s), JLr[i] = JL0[i] / s;
      for (int i = 0; i < 216; i++) Hr[i] = H0[i] / (s * sqrt(s));
      double err = fmax(maxrel(F, Fr, 6), fmax(maxrel(J, Jr, 36), maxrel(JL, JLr, 36)));
      err = fmax(err, maxrel(L, Lr, 6));
      if (k < 2) err = fmax(err, maxrel(H, Hr, 216));
      snprintf(buf, sizeof buf, "scaled by %.0e: sqrt F, J%s and log F, J", s, k < 2 ? ", H" : "");
      check(buf, err, 1e-14);
    }
  }

  /* error codes */
  check("invalid kind -> -1", smf_eval(7, 0, 1, x, F, NULL, NULL) == -1 ? 0 : 1, 0);
  check("pow with alpha <= 0 -> -1", smf_eval(SMF_POW, -1, 1, x, F, NULL, NULL) == -1 ? 0 : 1, 0);
  double Hx[216];
  check("H without J -> -1", smf_eval(SMF_EXP, 0, 1, x, F, NULL, Hx) == -1 ? 0 : 1, 0);

  /* ---- gmf_eval (general matrices) */
  {
    /* closed forms: N = 0.7 e1 e2^T (N^2 = 0): exp(N) = I + N, log(I + N) = N, sqrt(4 I + N) = 2 I + N / 4;
       rotation by th about z: exp of its generator, log and sqrt of the rotation */
    const double th = 0.9, c = cos(th), sn = sin(th), ch = cos(th / 2), sh = sin(th / 2);
    double Nil[9] = {0, 0.7, 0, 0, 0, 0, 0, 0, 0}, IN[9] = {1, 0.7, 0, 0, 1, 0, 0, 0, 1}, I4N[9] = {4, 0.7, 0, 0, 4, 0, 0, 0, 4};
    double K[9] = {0, -th, 0, th, 0, 0, 0, 0, 0}, R[9] = {c, -sn, 0, sn, c, 0, 0, 0, 1}, Rh[9] = {ch, -sh, 0, sh, ch, 0, 0, 0, 1};
    double S2[9] = {2, 0.175, 0, 0, 2, 0, 0, 0, 2}, G[9];
    gmf_eval(SMF_EXP, 0, 1, Nil, G, NULL, NULL);
    check("gmf: exp(N) = I + N (defective)", maxrel(G, IN, 9), 1e-15);
    gmf_eval(SMF_LOG, 0, 1, IN, G, NULL, NULL);
    check("gmf: log(I + N) = N (defective)", maxrel(G, Nil, 9), 1e-15);
    gmf_eval(SMF_SQRT, 0, 1, I4N, G, NULL, NULL);
    check("gmf: sqrt(4 I + N) = 2 I + N / 4 (defective)", maxrel(G, S2, 9), 1e-15);
    gmf_eval(SMF_EXP, 0, 1, K, G, NULL, NULL);
    check("gmf: exp(skew) = rotation (complex pair)", maxrel(G, R, 9), 1e-15);
    gmf_eval(SMF_LOG, 0, 1, R, G, NULL, NULL);
    check("gmf: log(rotation) = skew", maxrel(G, K, 9), 1e-15);
    gmf_eval(SMF_SQRT, 0, 1, R, G, NULL, NULL);
    check("gmf: sqrt(rotation) = half rotation", maxrel(G, Rh, 9), 1e-15);

    /* J and H against central differences: a random non-symmetric matrix and a nearly defective one */
    double Ag[2][9] = {{1.6, 0.3, -0.2, -0.4, 1.1, 0.5, 0.2, -0.3, 1.4}, {1.2, 0.8, 0.1, 0, 1.2, 0.5, 1e-9, 0, 1.2}};
    const int gk[4] = {SMF_EXP, SMF_LOG, SMF_SQRT, SMF_POW};
    const double ga[4] = {0, 0, 0, 1.7};
    const char *gn[4] = {"exp", "log", "sqrt", "pow 1.7"};
    for (int s = 0; s < 2; s++)
      for (int k = 0; k < 4; k++) {
        double F0[9], J0[81], H0[729], Jfd[81], Hfd[729], Fp[9], Fm[9], Jp[81], Jm[81];
        const double h = 1e-6;
        gmf_eval(gk[k], ga[k], 1, Ag[s], F0, J0, H0);
        for (int p = 0; p < 9; p++) {
          double ap[9], am[9];
          memcpy(ap, Ag[s], sizeof ap);
          memcpy(am, Ag[s], sizeof am);
          ap[p] += h;
          am[p] -= h;
          gmf_eval(gk[k], ga[k], 1, ap, Fp, Jp, NULL);
          gmf_eval(gk[k], ga[k], 1, am, Fm, Jm, NULL);
          for (int i = 0; i < 9; i++) Jfd[9 * i + p] = (Fp[i] - Fm[i]) / (2 * h);
          for (int i = 0; i < 81; i++) Hfd[9 * i + p] = (Jp[i] - Jm[i]) / (2 * h);
        }
        snprintf(buf, sizeof buf, "gmf %s, %s: J vs central differences", gn[k], s ? "nearly defective" : "random");
        check(buf, maxrel(J0, Jfd, 81), 1e-8);
        snprintf(buf, sizeof buf, "gmf %s, %s: H vs central differences of J", gn[k], s ? "nearly defective" : "random");
        check(buf, maxrel(H0, Hfd, 729), 1e-7);
      }

    /* batch == single calls (SIMD lanes with different groups / refinements), n = 37; NaN for a negative real
       eigenvalue only in its matrix */
    double *Ab = malloc(sizeof(double) * 9 * N), *Fg = malloc(sizeof(double) * 9 * N), *Jg = malloc(sizeof(double) * 81 * N);
    double *Hg = malloc(sizeof(double) * 729 * N);
    for (int i = 0; i < N; i++) {
      double t = 0.37 * i;
      double a[9] = {1.5 + sin(t), 0.3 * cos(2 * t), 0.2, -0.4 * sin(t), 1.2, 0.5 * cos(t), 0.1 * (i % 3), -0.3, 1.4 + 0.2 * cos(3 * t)};
      memcpy(Ab + 9 * i, a, sizeof a);
    }
    double neg[9] = {-1, 0, 0, 0, 2, 0, 0, 0, 3};
    memcpy(Ab + 9 * 11, neg, sizeof neg);
    gmf_eval(SMF_LOG, 0, N, Ab, Fg, Jg, Hg);
    double err = 0;
    int nanok = 1;
    for (int i = 0; i < N; i++) {
      double F1[9], J1[81], H1[729];
      gmf_eval(SMF_LOG, 0, 1, Ab + 9 * i, F1, J1, H1);
      if (i == 11) {
        for (int m = 0; m < 9; m++) nanok &= isnan(Fg[9 * i + m]) && isnan(F1[m]);
        continue;
      }
      err = fmax(err, fmax(maxrel(Fg + 9 * i, F1, 9), fmax(maxrel(Jg + 81 * i, J1, 81), maxrel(Hg + 729 * i, H1, 729))));
    }
    check("gmf: batch of 37 == single calls (F, J, H)", err, 0);
    check("gmf: real eigenvalue < 0 -> NaN (log), only that matrix", nanok ? 0 : 1, 0);
    free(Ab); free(Fg); free(Jg); free(Hg);
    check("gmf: invalid kind / alpha / H without J -> -1",
          gmf_eval(7, 0, 1, Nil, G, NULL, NULL) == -1 && gmf_eval(SMF_POW, 0, 1, Nil, G, NULL, NULL) == -1 &&
          gmf_eval(SMF_EXP, 0, 1, Nil, G, NULL, Hx) == -1 ? 0 : 1, 0);
  }

  printf(fails ? "%d FAILED\n" : "all passed\n", fails);
  return fails != 0;
}
