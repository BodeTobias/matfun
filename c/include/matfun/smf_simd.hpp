// smf_simd.hpp -- vector number type for smf_kernel.hpp, chosen by the compile flags:
//   AVX-512 (8 doubles) or AVX2 (4 doubles), transcendental functions from glibc libmvec (glibc >= 2.35);
//   plain scalar code otherwise or with -DSMF_SCALAR.
// libmvec binding:
//   default            direct calls, link with -lmvec (build for one machine, -march=native)
//   SMF_MVEC_DYNAMIC   calls through the pointer table smf_mvec_table[W] (filled at run time with dlsym by
//                      c/src/dispatch.cc), so that the library loads on systems without (a recent enough) libmvec
#pragma once
#include <cstdint>

#if !defined(SMF_SCALAR) && defined(__AVX512F__)
#include <immintrin.h>
#define SMF_W 8
#define SMF_MV(f) _ZGVeN8v_##f
#define SMF_MV2(f) _ZGVeN8vv_##f
typedef __m512d smf_native;
#elif !defined(SMF_SCALAR) && defined(__AVX2__)
#include <immintrin.h>
#define SMF_W 4
#define SMF_MV(f) _ZGVdN4v_##f
#define SMF_MV2(f) _ZGVdN4vv_##f
typedef __m256d smf_native;
#else
#define SMF_W 1
#endif

#ifndef SMF_NS
#define SMF_NS smf
#endif

#if SMF_W > 1
#ifdef SMF_MVEC_DYNAMIC
// order of the entries: acos, cos, exp, log, atanh, expm1, log1p, pow, sin, cbrt, atan2; table index 4 (AVX2) or 8
// (AVX-512)
extern "C" void *smf_mvec_table[9][11];
#define SMF_MVCALL1(i, x) (((smf_native(*)(smf_native))smf_mvec_table[SMF_W][i])((smf_native)(x)))
#define SMF_MVCALL2(i, x, y) \
  (((smf_native(*)(smf_native, smf_native))smf_mvec_table[SMF_W][i])((smf_native)(x), (smf_native)(y)))
#else
extern "C" {
smf_native SMF_MV(acos)(smf_native);
smf_native SMF_MV(cos)(smf_native);
smf_native SMF_MV(exp)(smf_native);
smf_native SMF_MV(log)(smf_native);
smf_native SMF_MV(atanh)(smf_native);
smf_native SMF_MV(expm1)(smf_native);
smf_native SMF_MV(log1p)(smf_native);
smf_native SMF_MV2(pow)(smf_native, smf_native);
smf_native SMF_MV(sin)(smf_native);
smf_native SMF_MV(cbrt)(smf_native);
smf_native SMF_MV2(atan2)(smf_native, smf_native);
}
#define SMF_MVCALL1(i, x) (SMF_MVF##i((smf_native)(x)))
#define SMF_MVCALL2(i, x, y) (SMF_MVF##i((smf_native)(x), (smf_native)(y)))
#define SMF_MVF0 SMF_MV(acos)
#define SMF_MVF1 SMF_MV(cos)
#define SMF_MVF2 SMF_MV(exp)
#define SMF_MVF3 SMF_MV(log)
#define SMF_MVF4 SMF_MV(atanh)
#define SMF_MVF5 SMF_MV(expm1)
#define SMF_MVF6 SMF_MV(log1p)
#define SMF_MVF7 SMF_MV2(pow)
#define SMF_MVF8 SMF_MV(sin)
#define SMF_MVF9 SMF_MV(cbrt)
#define SMF_MVF10 SMF_MV2(atan2)
#endif

namespace SMF_NS {
typedef double vd __attribute__((vector_size(8 * SMF_W)));
typedef long vm __attribute__((vector_size(8 * SMF_W)));
inline vd sel(vm m, vd a, vd b) { return m ? a : b; }
inline vd sel(vm m, vd a, double b) { return m ? a : (vd{} + b); }
inline vd sel(vm m, double a, vd b) { return m ? (vd{} + a) : b; }
inline vd sel(vm m, double a, double b) { return m ? (vd{} + a) : (vd{} + b); }
inline vm mnot(vm m) { return ~m; }
inline vm sgnbit(vd x) { return (vm)x < 0; }
inline vd m_abs(vd x) { return (vd)((vm)x & (vm{} + 0x7fffffffffffffffL)); }
inline vd m_pow2floor(vd x) { return (vd)((vm)x & (vm{} + 0x7ff0000000000000L)); }   // exponent bits only
#if SMF_W == 8
inline vd m_sqrt(vd x) { return (vd)_mm512_sqrt_pd((__m512d)x); }
#else
inline vd m_sqrt(vd x) { return (vd)_mm256_sqrt_pd((__m256d)x); }
#endif
inline vd m_acos(vd x) { return (vd)SMF_MVCALL1(0, x); }
inline vd m_cos(vd x) { return (vd)SMF_MVCALL1(1, x); }
inline vd m_exp(vd x) { return (vd)SMF_MVCALL1(2, x); }
inline vd m_log(vd x) { return (vd)SMF_MVCALL1(3, x); }
inline vd m_atanh(vd x) { return (vd)SMF_MVCALL1(4, x); }
inline vd m_expm1(vd x) { return (vd)SMF_MVCALL1(5, x); }
inline vd m_log1p(vd x) { return (vd)SMF_MVCALL1(6, x); }
inline vd m_pow(vd x, vd y) { return (vd)SMF_MVCALL2(7, x, y); }
inline vd m_sin(vd x) { return (vd)SMF_MVCALL1(8, x); }
inline vd m_cbrt(vd x) { return (vd)SMF_MVCALL1(9, x); }
inline vd m_atan2(vd y, vd x) { return (vd)SMF_MVCALL2(10, y, x); }
inline bool anyl(vm m) {                       // any lane set
  for (int l = 0; l < SMF_W; l++)
    if (m[l]) return true;
  return false;
}
inline double hmax(vd x) {                     // largest lane (NaN ignored)
  double r = 0.0;
  for (int l = 0; l < SMF_W; l++) r = x[l] > r ? x[l] : r;
  return r;
}
}  // namespace SMF_NS
#endif

#include "smf_kernel.hpp"   // the kernel templates; included after the vector overloads so that they find them
#include "gmf_kernel.hpp"

namespace SMF_NS {
// n matrices, Voigt vectors x (n x 6) in; out (J, H may be nullptr, H needs J): Voigt layout F (n x 6), J (n x 36),
// H (n x 216), or with mat the matrix layout F (n x 9), J (n x 81), H (n x 729) (smf_kernel.hpp);
// SMF_W at a time, tail padded with the identity
template <int KIND>
void batch(int64_t n, const double *x, double *F, double *J, double al, double *H, bool mat) {
  double c2[NTERMS];
  dd2_coef(KIND, al, c2);
  double fs[6], js[36], hs[216];                          // one matrix in Voigt layout (matrix output)
#if SMF_W == 1
  for (int64_t i = 0; i < n; i++) {
    if (!mat) {
      eval3<KIND, double>(x + 6 * i, F + 6 * i, J ? J + 36 * i : nullptr, al, H ? H + 216 * i : nullptr, c2);
    } else {
      eval3<KIND, double>(x + 6 * i, fs, J ? js : nullptr, al, H ? hs : nullptr, c2);
      to_matrix(fs, js, hs, F + 9 * i, J ? J + 81 * i : nullptr, H ? H + 729 * i : nullptr);
    }
  }
#else
  constexpr int W = SMF_W;
  for (int64_t b = 0; b < n; b += W) {
    const int nb = n - b < W ? (int)(n - b) : W;
    vd a[6], f[6], j[36], h[216];
    for (int k = 0; k < 6; k++)
      for (int l = 0; l < W; l++) a[k][l] = l < nb ? x[6 * (b + l) + k] : (k < 3 ? 1.0 : 0.0);
    eval3<KIND>(a, f, J ? j : nullptr, al, H ? h : nullptr, c2);
    if (!mat) {
      for (int l = 0; l < nb; l++)
        for (int m = 0; m < 6; m++) F[6 * (b + l) + m] = f[m][l];
      if (J)
        for (int l = 0; l < nb; l++)
          for (int m = 0; m < 36; m++) J[36 * (b + l) + m] = j[m][l];
      if (H)
        for (int l = 0; l < nb; l++)
          for (int m = 0; m < 216; m++) H[216 * (b + l) + m] = h[m][l];
      continue;
    }
    for (int l = 0; l < nb; l++) {
      for (int m = 0; m < 6; m++) fs[m] = f[m][l];
      if (J)
        for (int m = 0; m < 36; m++) js[m] = j[m][l];
      if (H)
        for (int m = 0; m < 216; m++) hs[m] = h[m][l];
      to_matrix(fs, js, hs, F + 9 * (b + l), J ? J + 81 * (b + l) : nullptr, H ? H + 729 * (b + l) : nullptr);
    }
  }
#endif
}

inline void batch_any(int kind, int64_t n, const double *x, double *F, double *J, double al, double *H, bool mat) {
  switch (kind) {
    case EXP: batch<EXP>(n, x, F, J, al, H, mat); break;
    case LOG: batch<LOG>(n, x, F, J, al, H, mat); break;
    case SQRT: batch<SQRT>(n, x, F, J, al, H, mat); break;
    default: batch<POW>(n, x, F, J, al, H, mat); break;
  }
}
}  // namespace SMF_NS

namespace SMF_NS {
// general matrices (gmf_kernel.hpp): n matrices A (n x 9, row-major) -> F (n x 9), J (n x 81), H (n x 729) (J, H may
// be nullptr; H needs J); kind EXP, LOG or POW (sqrt: POW with al = 0.5); SMF_W at a time, tail padded with I
template <int KIND>
void gbatch(int64_t n, const double *A, double *F, double *J, double al, double *H) {
  const double *bt = gmf::series_table(KIND, al);
#if SMF_W == 1
  for (int64_t i = 0; i < n; i++) {
    if (H) gmf::eval<KIND, 2, double>(A + 9 * i, F + 9 * i, J + 81 * i, H + 729 * i, al, bt);
    else if (J) gmf::eval<KIND, 1, double>(A + 9 * i, F + 9 * i, J + 81 * i, nullptr, al, bt);
    else gmf::eval<KIND, 0, double>(A + 9 * i, F + 9 * i, nullptr, nullptr, al, bt);
  }
#else
  constexpr int W = SMF_W;
  for (int64_t b = 0; b < n; b += W) {
    const int nb = n - b < W ? (int)(n - b) : W;
    vd a[9], f[9], j[81], h[729];
    for (int k = 0; k < 9; k++)
      for (int l = 0; l < W; l++) a[k][l] = l < nb ? A[9 * (b + l) + k] : (k % 4 == 0 ? 1.0 : 0.0);
    if (H) gmf::eval<KIND, 2, vd>(a, f, j, h, al, bt);
    else if (J) gmf::eval<KIND, 1, vd>(a, f, j, nullptr, al, bt);
    else gmf::eval<KIND, 0, vd>(a, f, nullptr, nullptr, al, bt);
    for (int l = 0; l < nb; l++)
      for (int k = 0; k < 9; k++) F[9 * (b + l) + k] = f[k][l];
    if (J)
      for (int l = 0; l < nb; l++)
        for (int k = 0; k < 81; k++) J[81 * (b + l) + k] = j[k][l];
    if (H)
      for (int l = 0; l < nb; l++)
        for (int k = 0; k < 729; k++) H[729 * (b + l) + k] = h[k][l];
  }
#endif
}

inline void gbatch_any(int kind, int64_t n, const double *A, double *F, double *J, double al, double *H) {
  switch (kind) {
    case EXP: gbatch<EXP>(n, A, F, J, al, H); break;
    case LOG: gbatch<LOG>(n, A, F, J, al, H); break;
    default: gbatch<POW>(n, A, F, J, al, H); break;
  }
}
}  // namespace SMF_NS
