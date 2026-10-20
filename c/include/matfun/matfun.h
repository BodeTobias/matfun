/* matfun.h -- matrix functions exp, log, sqrt, A^alpha of symmetric (smf_*) and general (gmf_eval) 3x3 matrices with
 * first and second derivatives; C API of libmatfun.  Conventions: README.md.  All arrays row-major, one matrix after the other.
 *
 * smf_eval: Voigt vectors v = (A11, A22, A33, A12, A13, A23) in, the same layout out
 *   v  (n, 6)
 *   F  (n, 6)          Voigt vector of f(A)
 *   J  (n, 6, 6)       J[i][k]    = dF_i / dv_k            (or NULL)
 *   H  (n, 6, 6, 6)    H[i][p][q] = d2F_i / dv_p dv_q      (or NULL; needs J)
 *   The derivative w.r.t. an off-diagonal component, e.g. v_3 = A12, changes A12 and A21 together.
 *
 * smf_eval_matrix: full symmetric matrices A in, matrices out
 *   A  (n, 3, 3)
 *   F  (n, 3, 3)                   f(A)
 *   J  (n, 3, 3, 3, 3)             J[a][b][c][d]       = dF_ab / dA_cd             (or NULL)
 *   H  (n, 3, 3, 3, 3, 3, 3)       H[a][b][c][d][e][f] = d2F_ab / dA_cd dA_ef      (or NULL; needs J)
 *   These are the derivatives of A -> f((A + A^T)/2); they have the symmetries of the index pairs
 *   (J_abcd = J_bacd = J_abdc = J_cdab, likewise for H).
 */
#ifndef MATFUN_H
#define MATFUN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum smf_kind { SMF_EXP = 0, SMF_LOG = 1, SMF_POW = 2, SMF_SQRT = 3 };

/* f(A) and its derivatives for n matrices.  alpha: exponent for SMF_POW (> 0), ignored otherwise.
 * log needs A positive definite, sqrt / pow positive semidefinite (round-off below 0 is clamped).
 * Returns 0, or -1 for invalid arguments (unknown kind, alpha <= 0 for SMF_POW, H without J, n < 0). */
int smf_eval(int kind, double alpha, int64_t n, const double *v, double *F, double *J, double *H);

/* as smf_eval for full matrices.  Returns -3 if a matrix is not symmetric (|A_ij - A_ji| > 1e-8 max |A_kl|);
 * its results are then NaN, all other matrices are computed as usual. */
int smf_eval_matrix(int kind, double alpha, int64_t n, const double *A, double *F, double *J, double *H);

/* gmf_eval: f(A) and its derivatives for n general (non-symmetric) real matrices, principal branch:
 *   A  (n, 3, 3)                   input (row-major)
 *   F  (n, 3, 3)                   f(A)
 *   J  (n, 3, 3, 3, 3)             J[a][b][c][d]       = dF_ab / dA_cd             (or NULL)
 *   H  (n, 3, 3, 3, 3, 3, 3)       H[a][b][c][d][e][f] = d2F_ab / dA_cd dA_ef      (or NULL; needs J)
 * the full derivatives (no symmetrization), exact also for defective A (Jordan blocks) and complex eigenvalues.
 * alpha: exponent for SMF_POW (> 0), ignored otherwise.  log, sqrt and pow need A without real eigenvalues <= 0;
 * the results of such a matrix are NaN.  Returns 0, or -1 for invalid arguments (as smf_eval). */
int gmf_eval(int kind, double alpha, int64_t n, const double *A, double *F, double *J, double *H);

/* kernel variant chosen at run time (CPU and libmvec): "avx512", "avx2", "scalar", or "native" for a build with
 * MATFUN_SIMD=native; force one with the environment variable MATFUN_SIMD=avx512|avx2|scalar (if usable) */
const char *smf_simd_name(void);

const char *smf_version(void);

#ifdef __cplusplus
}
#endif

#endif /* MATFUN_H */
