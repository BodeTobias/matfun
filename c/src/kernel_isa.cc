// kernel_isa.cc -- one kernel variant; compiled once per instruction set with -DSMF_ISA=<name> and matching flags
// (CMakeLists.txt): scalar (baseline), avx2 (-mavx2 -mfma -mbmi -mbmi2 -mf16c), avx512 (additionally
// -mavx512f/bw/cd/dq/vl), or native (-march=native).
// Each variant lives in its own namespace smf_<name>, see smf_kernel.hpp.
#define SMF_CAT_(a, b) a##b
#define SMF_CAT(a, b) SMF_CAT_(a, b)
#define SMF_NS SMF_CAT(smf_, SMF_ISA)

#include "matfun/smf_simd.hpp"

extern "C" void SMF_CAT(smf_batch_, SMF_ISA)(int kind, int64_t n, const double *x, double *F, double *J, double alpha,
                                             double *H, int mat) {
  SMF_NS::batch_any(kind, n, x, F, J, alpha, H, mat != 0);
}

extern "C" void SMF_CAT(smf_gbatch_, SMF_ISA)(int kind, int64_t n, const double *A, double *F, double *J,
                                              double alpha, double *H) {
  SMF_NS::gbatch_any(kind, n, A, F, J, alpha, H);
}

extern "C" int SMF_CAT(smf_width_, SMF_ISA)() { return SMF_W; }
