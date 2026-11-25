// matfun_xla.cc -- XLA FFI handlers (CPU) for JAX, part of libmatfun when built with MATFUN_WITH_XLA_FFI.
// The batch is split over XLA's intra-op thread pool (not OpenMP).  Each handler takes x and returns F, (F, J) or
// (F, J, H); attributes kind (int64), alpha (f64).  Targets, registered by matfun/jax.py:
//   SmfValue, SmfValueJac, SmfValueJacHess: Voigt layout as smf_eval, x (..., 6), F (..., 6), J (..., 6, 6),
//   H (..., 6, 6, 6)
//   SmfMatrixValue, SmfMatrixValueJac, SmfMatrixValueJacHess: symmetric matrices as smf_eval_matrix, x (..., 3, 3),
//   F (..., 3, 3), J (..., 3, 3, 3, 3), H (..., 3, 3, 3, 3, 3, 3); NaN for a matrix that is not symmetric
//   GmfValue, GmfValueJac, GmfValueJacHess: general matrices as gmf_eval, x (..., 3, 3), F (..., 3, 3),
//   J (..., 3, 3, 3, 3), H (..., 3, 3, 3, 3, 3, 3)
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>

#include "dispatch.h"
#include "matfun/matfun.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using smf_dispatch::Family;

// dynamic chunks; the caller works too and only waits for chunks already claimed by a running thread
// (no deadlock if the pool has one thread or scheduled tasks start late)
template <typename Body>
static void parallel_chunks(ffi::ThreadPool &pool, int64_t n, int64_t chunk, Body body) {
  const int64_t nchunks = (n + chunk - 1) / chunk;
  const int64_t nt = std::min<int64_t>(pool.num_threads(), nchunks - 1);
  struct State { std::atomic<int64_t> next{0}, done{0}; };
  auto st = std::make_shared<State>();
  auto work = [st, n, nchunks, chunk, body]() {
    for (int64_t c; (c = st->next.fetch_add(1)) < nchunks;) {
      body(c * chunk, std::min(n, (c + 1) * chunk));
      st->done.fetch_add(1, std::memory_order_release);
    }
  };
  for (int64_t t = 0; t < nt; t++) pool.Schedule(decltype(work)(work));
  work();
  while (st->done.load(std::memory_order_acquire) < nchunks) std::this_thread::yield();
}

template <Family FAM>
static ffi::Error Eval(ffi::ThreadPool pool, ffi::Buffer<ffi::F64> x, ffi::RemainingRets out, int64_t kind,
                       double alpha) {
  const auto dims = x.dimensions();
  const size_t r = dims.size();
  if (FAM == smf_dispatch::VOIGT ? r < 1 || dims[r - 1] != 6 : r < 2 || dims[r - 1] != 3 || dims[r - 2] != 3)
    return ffi::Error::InvalidArgument(FAM == smf_dispatch::VOIGT ? "matfun: expected Voigt vectors of shape (..., 6)"
                                                                  : "matfun: expected matrices of shape (..., 3, 3)");
  if (out.size() < 1 || out.size() > 3) return ffi::Error::InvalidArgument("matfun: expected F, (F, J) or (F, J, H)");
  double *p[3] = {};
  for (size_t i = 0; i < out.size(); i++) {
    auto b = out.get<ffi::Buffer<ffi::F64>>(i);
    if (b.has_error()) return b.error();
    p[i] = b.value()->typed_data();
  }
  // (kind outside the int range: invalid, as an unknown kind)
  smf_dispatch::Call c{FAM, (int)std::clamp<int64_t>(kind, -1, SMF_SQRT + 1), alpha,
                       (int64_t)x.element_count() / (FAM == smf_dispatch::VOIGT ? 6 : 9), x.typed_data(), p[0],
                       p[1], p[2]};
  if (!smf_dispatch::prepare(c)) return ffi::Error::InvalidArgument("matfun: invalid kind or alpha");
  parallel_chunks(pool, c.n, smf_dispatch::chunk(c, (int)pool.num_threads() + 1),
                  [c](int64_t lo, int64_t hi) { smf_dispatch::run(c, lo, hi); });
  return ffi::Error::Success();
}

#define MATFUN_HANDLER(name, fam)                                              \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(name, Eval<smf_dispatch::fam>, ffi::Ffi::Bind() \
                                                                   .Ctx<ffi::ThreadPool>()                     \
                                                                   .Arg<ffi::Buffer<ffi::F64>>()               \
                                                                   .RemainingRets()                            \
                                                                   .Attr<int64_t>("kind")                      \
                                                                   .Attr<double>("alpha"))

MATFUN_HANDLER(SmfValue, VOIGT);
MATFUN_HANDLER(SmfValueJac, VOIGT);
MATFUN_HANDLER(SmfValueJacHess, VOIGT);
MATFUN_HANDLER(SmfMatrixValue, MATRIX);
MATFUN_HANDLER(SmfMatrixValueJac, MATRIX);
MATFUN_HANDLER(SmfMatrixValueJacHess, MATRIX);
MATFUN_HANDLER(GmfValue, GENERAL);
MATFUN_HANDLER(GmfValueJac, GENERAL);
MATFUN_HANDLER(GmfValueJacHess, GENERAL);

// XLA FFI API version the handlers were built against (major * 1000 + minor); matfun.jax only registers them with
// a jaxlib of the same major and at least this minor version
extern "C" int smf_xla_ffi_api_version(void) { return XLA_FFI_API_MAJOR * 1000 + XLA_FFI_API_MINOR; }
