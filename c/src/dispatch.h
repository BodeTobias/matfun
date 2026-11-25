// dispatch.h -- choice of the kernel variant (instruction set) at run time, and the driver shared by the C API and
// the XLA handlers; internal to libmatfun.
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
  batch_fn fn1 = nullptr;     // for a call with one matrix: the scalar variant (a SIMD vector would mostly compute
  gbatch_fn gfn1 = nullptr;   // padding), or the variant forced by MATFUN_SIMD
};

// chosen on first use (thread-safe): the widest variant the CPU supports and for which libmvec provides all
// vector functions; the environment variable MATFUN_SIMD=avx512|avx2|native|scalar restricts the choice to one
// variant if that one is usable (then also for one matrix)
const Impl &impl();

// ---- one call: n inputs x of a family -> F, J, H (J, H may be nullptr; H needs J), layouts as in matfun.h:
//      VOIGT smf_eval, MATRIX smf_eval_matrix, GENERAL gmf_eval
enum Family { VOIGT, MATRIX, GENERAL };

struct Call {
  Family fam;
  int kind;
  double alpha;
  int64_t n;
  const double *x;
  double *F, *J, *H;
  batch_fn fn = nullptr;      // set by prepare
  gbatch_fn gfn = nullptr;
};

// checks the arguments (false: invalid, as smf_eval's -1), normalises kind and alpha, chooses the kernel
bool prepare(Call &c);

// matrices lo..hi-1 of a prepared call; false if one of them is not symmetric (MATRIX; its results are NaN)
bool run(const Call &c, int64_t lo, int64_t hi);

// matrices per task of a prepared call
int64_t chunk(const Call &c);

}  // namespace smf_dispatch
