// dispatch.h -- choice of the kernel variant (instruction set) at run time; internal to libmatfun.
#pragma once
#include <cstdint>

namespace smf_dispatch {

// n Voigt vectors x -> F, J, H in the Voigt layout, or with mat != 0 the matrix layout (smf_kernel.hpp)
using batch_fn = void (*)(int kind, int64_t n, const double *x, double *F, double *J, double alpha, double *H,
                          int mat);

// general matrices: n matrices A (n x 9) -> F (n x 9), J (n x 81), H (n x 729) (gmf_kernel.hpp)
using gbatch_fn = void (*)(int kind, int64_t n, const double *A, double *F, double *J, double alpha, double *H);

struct Impl {
  batch_fn fn;        // smf::batch_any of the chosen variant
  gbatch_fn gfn;      // smf::gbatch_any of the chosen variant
  int width;          // matrices per SIMD vector
  const char *name;   // "avx512", "avx2", "native" or "scalar"
};

// chosen on first use (thread-safe): the widest variant the CPU supports and for which libmvec provides all
// vector functions; the environment variable MATFUN_SIMD=avx512|avx2|native|scalar restricts the choice to one
// variant if that one is usable
const Impl &impl();

}  // namespace smf_dispatch
