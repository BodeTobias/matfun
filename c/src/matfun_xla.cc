// matfun_xla.cc -- XLA FFI handlers (CPU) for JAX, part of libmatfun when built with MATFUN_WITH_XLA_FFI.
// The batch is split over XLA's intra-op thread pool (not OpenMP).  Targets, registered by matfun/jax.py:
//   SmfValue (x -> F), SmfValueJac (x -> F, J), SmfValueJacHess (x -> F, J, H); attributes kind (int64), alpha (f64);
//   Voigt layout as smf_eval: x (..., 6), F (..., 6), J (..., 6, 6), H (..., 6, 6, 6)
//   GmfValue, GmfValueJac, GmfValueJacHess: general matrices as gmf_eval, x (..., 3, 3), F (..., 3, 3), J (..., 3, 3,
//   3, 3), H (..., 3, 3, 3, 3, 3, 3)
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>

#include "dispatch.h"
#include "matfun/matfun.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

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

static ffi::Error Eval(ffi::ThreadPool pool, ffi::Buffer<ffi::F64> x, double *F, double *J, int64_t kind, double alpha,
                       double *H = nullptr) {
  auto dims = x.dimensions();
  if (dims.size() == 0 || dims[dims.size() - 1] != 6)
    return ffi::Error::InvalidArgument("matfun: expected Voigt vectors of shape (..., 6)");
  // same checks and normalisation as smf_eval (c/src/matfun.cc)
  if (kind < SMF_EXP || kind > SMF_SQRT || (kind == SMF_POW && !(alpha > 0)))
    return ffi::Error::InvalidArgument("matfun: invalid kind or alpha");
  if (kind != SMF_POW) alpha = kind == SMF_SQRT ? 0.5 : 0.0;
  const int64_t n = x.element_count() / 6;
  const double *xp = x.typed_data();
  const smf_dispatch::batch_fn batch = smf_dispatch::impl().fn;
  parallel_chunks(pool, n, H ? 256 : 1024, [=](int64_t lo, int64_t hi) {
    batch((int)kind, hi - lo, xp + 6 * lo, F + 6 * lo, J ? J + 36 * lo : nullptr, alpha, H ? H + 216 * lo : nullptr, 0);
  });
  return ffi::Error::Success();
}

// general matrices (gmf_eval): x (..., 3, 3) -> F (..., 3, 3), J (..., 3, 3, 3, 3), H (..., 3, 3, 3, 3, 3, 3)
static ffi::Error GEval(ffi::ThreadPool pool, ffi::Buffer<ffi::F64> x, double *F, double *J, int64_t kind, double alpha,
                        double *H = nullptr) {
  auto dims = x.dimensions();
  if (dims.size() < 2 || dims[dims.size() - 1] != 3 || dims[dims.size() - 2] != 3)
    return ffi::Error::InvalidArgument("matfun: expected matrices of shape (..., 3, 3)");
  if (kind < SMF_EXP || kind > SMF_SQRT || (kind == SMF_POW && !(alpha > 0)))
    return ffi::Error::InvalidArgument("matfun: invalid kind or alpha");
  if (kind == SMF_SQRT) kind = SMF_POW, alpha = 0.5;
  if (kind != SMF_POW) alpha = 0.0;
  const int64_t n = x.element_count() / 9;
  const double *xp = x.typed_data();
  const smf_dispatch::gbatch_fn gbatch = smf_dispatch::impl().gfn;
  parallel_chunks(pool, n, H ? 256 : 1024, [=](int64_t lo, int64_t hi) {
    gbatch((int)kind, hi - lo, xp + 9 * lo, F + 9 * lo, J ? J + 81 * lo : nullptr, alpha, H ? H + 729 * lo : nullptr);
  });
  return ffi::Error::Success();
}

static ffi::Error GValueImpl(ffi::ThreadPool pool, ffi::Buffer<ffi::F64> x, ffi::ResultBuffer<ffi::F64> F,
                             int64_t kind, double alpha) {
  return GEval(pool, x, F->typed_data(), nullptr, kind, alpha);
}

static ffi::Error GValueJacImpl(ffi::ThreadPool pool, ffi::Buffer<ffi::F64> x, ffi::ResultBuffer<ffi::F64> F,
                                ffi::ResultBuffer<ffi::F64> J, int64_t kind, double alpha) {
  return GEval(pool, x, F->typed_data(), J->typed_data(), kind, alpha);
}

static ffi::Error GValueJacHessImpl(ffi::ThreadPool pool, ffi::Buffer<ffi::F64> x, ffi::ResultBuffer<ffi::F64> F,
                                    ffi::ResultBuffer<ffi::F64> J, ffi::ResultBuffer<ffi::F64> H, int64_t kind,
                                    double alpha) {
  return GEval(pool, x, F->typed_data(), J->typed_data(), kind, alpha, H->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GmfValue, GValueImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::ThreadPool>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Attr<int64_t>("kind")
                                  .Attr<double>("alpha"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(GmfValueJac, GValueJacImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::ThreadPool>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Attr<int64_t>("kind")
                                  .Attr<double>("alpha"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(GmfValueJacHess, GValueJacHessImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::ThreadPool>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Attr<int64_t>("kind")
                                  .Attr<double>("alpha"));

static ffi::Error ValueImpl(ffi::ThreadPool pool, ffi::Buffer<ffi::F64> x, ffi::ResultBuffer<ffi::F64> F,
                            int64_t kind, double alpha) {
  return Eval(pool, x, F->typed_data(), nullptr, kind, alpha);
}

static ffi::Error ValueJacImpl(ffi::ThreadPool pool, ffi::Buffer<ffi::F64> x, ffi::ResultBuffer<ffi::F64> F,
                               ffi::ResultBuffer<ffi::F64> J, int64_t kind, double alpha) {
  return Eval(pool, x, F->typed_data(), J->typed_data(), kind, alpha);
}

static ffi::Error ValueJacHessImpl(ffi::ThreadPool pool, ffi::Buffer<ffi::F64> x, ffi::ResultBuffer<ffi::F64> F,
                                   ffi::ResultBuffer<ffi::F64> J, ffi::ResultBuffer<ffi::F64> H, int64_t kind,
                                   double alpha) {
  return Eval(pool, x, F->typed_data(), J->typed_data(), kind, alpha, H->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(SmfValue, ValueImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::ThreadPool>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Attr<int64_t>("kind")
                                  .Attr<double>("alpha"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(SmfValueJac, ValueJacImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::ThreadPool>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Attr<int64_t>("kind")
                                  .Attr<double>("alpha"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(SmfValueJacHess, ValueJacHessImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::ThreadPool>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Attr<int64_t>("kind")
                                  .Attr<double>("alpha"));

// XLA FFI API version the handlers were built against (major * 1000 + minor); matfun.jax only registers them with
// a jaxlib of the same major and at least this minor version
extern "C" int smf_xla_ffi_api_version(void) { return XLA_FFI_API_MAJOR * 1000 + XLA_FFI_API_MINOR; }
