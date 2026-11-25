// matfun.cc -- C API (matfun.h) and the driver of dispatch.h; the kernel variant (instruction set) is chosen at run
// time.
#include "matfun/matfun.h"

#include <algorithm>

#include "dispatch.h"
#include "matfun/smf_kernel.hpp"

#ifndef SMF_VERSION
#define SMF_VERSION "0.0.0"
#endif

namespace smf_dispatch {

constexpr int64_t CHUNK_H = 256, CHUNK = 1024;   // matrices per task with / without H
constexpr int64_t SMALL = 64;                     // matrices per batch call of full symmetric input

bool prepare(Call &c) {
  if (c.kind < SMF_EXP || c.kind > SMF_SQRT || c.n < 0 || (c.H && !c.J) || (c.kind == SMF_POW && !(c.alpha > 0)))
    return false;
  if (c.fam == GENERAL && c.kind == SMF_SQRT) c.kind = SMF_POW, c.alpha = 0.5;
  if (c.kind != SMF_POW) c.alpha = c.kind == SMF_SQRT ? 0.5 : 0.0;
  const Impl &d = impl();
  c.fn = c.n == 1 ? d.fn1 : d.fn;
  c.gfn = c.n == 1 ? d.gfn1 : d.gfn;
  return true;
}

bool run(const Call &c, int64_t lo, int64_t hi) {
  const int64_t nf = c.fam == VOIGT ? 6 : 9, nj = nf * nf, nh = nj * nf;   // output sizes per matrix
  double *F = c.F + nf * lo, *J = c.J ? c.J + nj * lo : nullptr, *H = c.H ? c.H + nh * lo : nullptr;
  if (c.fam == GENERAL) {
    c.gfn(c.kind, hi - lo, c.x + 9 * lo, F, J, c.alpha, H);
    return true;
  }
  if (c.fam == VOIGT) {
    c.fn(c.kind, hi - lo, c.x + 6 * lo, F, J, c.alpha, H, 0);
    return true;
  }
  double buf[6 * SMALL];                         // Voigt vectors of the symmetric parts
  bool sym = true;
  for (int64_t b = 0; b < hi - lo; b += SMALL) {
    const int64_t m = std::min(hi - lo - b, SMALL);
    for (int64_t i = 0; i < m; i++) sym &= smf::from_matrix(c.x + 9 * (lo + b + i), buf + 6 * i);
    c.fn(c.kind, m, buf, F + 9 * b, J ? J + 81 * b : nullptr, c.alpha, H ? H + 729 * b : nullptr, 1);
  }
  return sym;
}

int64_t chunk(const Call &c) { return c.H ? CHUNK_H : CHUNK; }

}  // namespace smf_dispatch

namespace {

// a call of the C API: chunks in parallel (OpenMP); 0, -1 (invalid arguments) or -3 (a matrix not symmetric)
int eval(smf_dispatch::Call c) {
  if (!smf_dispatch::prepare(c)) return -1;
  const int64_t ch = smf_dispatch::chunk(c), nchunks = (c.n + ch - 1) / ch;
  if (nchunks <= 1)                              // small calls (e.g. one matrix): no OpenMP region
    return smf_dispatch::run(c, 0, c.n) ? 0 : -3;
  int asym = 0;
#pragma omp parallel for schedule(dynamic) reduction(| : asym)
  for (int64_t k = 0; k < nchunks; k++) asym |= !smf_dispatch::run(c, k * ch, std::min(c.n, (k + 1) * ch));
  return asym ? -3 : 0;
}

}  // namespace

extern "C" int smf_eval(int kind, double alpha, int64_t n, const double *v, double *F, double *J, double *H) {
  return eval({smf_dispatch::VOIGT, kind, alpha, n, v, F, J, H});
}

extern "C" int smf_eval_matrix(int kind, double alpha, int64_t n, const double *A, double *F, double *J, double *H) {
  return eval({smf_dispatch::MATRIX, kind, alpha, n, A, F, J, H});
}

extern "C" int gmf_eval(int kind, double alpha, int64_t n, const double *A, double *F, double *J, double *H) {
  return eval({smf_dispatch::GENERAL, kind, alpha, n, A, F, J, H});
}

extern "C" const char *smf_simd_name(void) { return smf_dispatch::impl().name; }

extern "C" const char *smf_version(void) { return SMF_VERSION; }
