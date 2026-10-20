// matfun.cc -- C API (matfun.h); the kernel variant (instruction set) is chosen at run time, dispatch.h.
#include "matfun/matfun.h"

#include <algorithm>

#include "dispatch.h"
#include "matfun/smf_kernel.hpp"

#ifndef SMF_VERSION
#define SMF_VERSION "0.0.0"
#endif

namespace {

constexpr int64_t CHUNK_H = 256, CHUNK = 1024;   // matrices per task with / without H

// n matrices, chunked and parallel (OpenMP); input either Voigt vectors (n x 6) or full matrices (mat, n x 9)
int eval_chunks(int kind, double alpha, int64_t n, const double *x, bool mat, double *F, double *J, double *H) {
  if (kind < SMF_EXP || kind > SMF_SQRT || n < 0 || (H && !J) || (kind == SMF_POW && !(alpha > 0))) return -1;
  if (kind != SMF_POW) alpha = kind == SMF_SQRT ? 0.5 : 0.0;
  const smf_dispatch::batch_fn batch = smf_dispatch::impl().fn;
  const int64_t nf = mat ? 9 : 6, nj = mat ? 81 : 36, nh = mat ? 729 : 216;   // output sizes per matrix
  const int64_t chunk = H ? CHUNK_H : CHUNK;
  const int64_t nchunks = (n + chunk - 1) / chunk;
  int asym = 0;
#pragma omp parallel for schedule(dynamic) if (nchunks > 1) reduction(| : asym)
  for (int64_t c = 0; c < nchunks; c++) {
    const int64_t lo = c * chunk, hi = std::min(n, lo + chunk);
    const double *v = x + 6 * lo;
    double buf[6 * CHUNK];                       // Voigt vectors of the matrices (full input only)
    if (mat) {
      for (int64_t i = lo; i < hi; i++) asym |= !smf::from_matrix(x + 9 * i, buf + 6 * (i - lo));
      v = buf;
    }
    batch(kind, hi - lo, v, F + nf * lo, J ? J + nj * lo : nullptr, alpha, H ? H + nh * lo : nullptr, mat);
  }
  return asym ? -3 : 0;
}

}  // namespace

extern "C" int smf_eval(int kind, double alpha, int64_t n, const double *v, double *F, double *J, double *H) {
  return eval_chunks(kind, alpha, n, v, false, F, J, H);
}

extern "C" int smf_eval_matrix(int kind, double alpha, int64_t n, const double *A, double *F, double *J, double *H) {
  return eval_chunks(kind, alpha, n, A, true, F, J, H);
}

// general matrices: chunked and parallel as eval_chunks
extern "C" int gmf_eval(int kind, double alpha, int64_t n, const double *A, double *F, double *J, double *H) {
  if (kind < SMF_EXP || kind > SMF_SQRT || n < 0 || (H && !J) || (kind == SMF_POW && !(alpha > 0))) return -1;
  if (kind == SMF_SQRT) kind = SMF_POW, alpha = 0.5;
  if (kind != SMF_POW) alpha = 0.0;
  const smf_dispatch::gbatch_fn gbatch = smf_dispatch::impl().gfn;
  const int64_t chunk = H ? CHUNK_H : CHUNK;
  const int64_t nchunks = (n + chunk - 1) / chunk;
#pragma omp parallel for schedule(dynamic) if (nchunks > 1)
  for (int64_t c = 0; c < nchunks; c++) {
    const int64_t lo = c * chunk, hi = std::min(n, lo + chunk);
    gbatch(kind, hi - lo, A + 9 * lo, F + 9 * lo, J ? J + 81 * lo : nullptr, alpha, H ? H + 729 * lo : nullptr);
  }
  return 0;
}

extern "C" const char *smf_simd_name(void) { return smf_dispatch::impl().name; }

extern "C" const char *smf_version(void) { return SMF_VERSION; }
