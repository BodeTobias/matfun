// matfun_cuda.cu -- CUDA kernels around smf_kernel.hpp and gmf_kernel.hpp (T = double, one thread per matrix), C API
// smf_cuda_eval / smf_cuda_eval_matrix / gmf_cuda_eval and, with MATFUN_WITH_XLA_FFI, the XLA FFI handlers
// SmfValueCuda / SmfValueJacCuda / SmfValueJacHessCuda for JAX (Voigt layout) and GmfValueCuda / GmfValueJacCuda /
// GmfValueJacHessCuda (general matrices, matrix layout).
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

#include "matfun/smf_kernel.hpp"
#include "matfun/gmf_kernel.hpp"
#include "matfun/matfun_cuda.h"

namespace {

struct Coef {                      // series coefficients of the second divided differences (computed on the host)
  double c[smf::NTERMS];
};

#ifndef SMF_BLOCK
#define SMF_BLOCK 32               // threads per block of smf_eval_kernel (value, value + J)
#endif

// one thread per matrix; H is always nullptr here (H goes to smf_hess_kernel), but in the Voigt branch it is passed
// as a runtime value instead of a literal: that was markedly faster for value + J (different register allocation)
template <int KIND, bool MAT>
__global__ void smf_eval_kernel(int64_t n, const double *x, double *F, double *J, double *H, double al, Coef c2) {
  const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (!MAT) {
    smf::eval3<KIND, double>(x + 6 * i, F + 6 * i, J ? J + 36 * i : nullptr, al, H ? H + 216 * i : nullptr, c2.c);
    return;
  }
  double v[6], f[6], j[36];                  // (two calls: a conditional pointer to j would put j in local memory)
  smf::from_matrix(x + 9 * i, v);
  if (J) {
    smf::eval3<KIND, double>(v, f, j, al, nullptr, c2.c);
    smf::to_matrix(f, j, nullptr, F + 9 * i, J + 81 * i, nullptr);
  } else {
    smf::eval3<KIND, double>(v, f, nullptr, al, nullptr, c2.c);
    smf::to_matrix(f, nullptr, nullptr, F + 9 * i, nullptr, nullptr);
  }
}

// F, J and H; one warp per block, one thread per matrix.  Writing H (216 or 729 doubles) directly from each thread
// puts neighbouring threads far apart (uncoalesced).  Instead each thread stores the 126 distinct entries of its H
// in shared memory, then the warp writes the 32 full H blocks one after another, consecutive lanes -> consecutive
// doubles.
constexpr int HWARP = 32, HPACKED = 126;

template <int KIND, bool MAT>
__global__ void __launch_bounds__(HWARP) smf_hess_kernel(int64_t n, const double *x, double *F, double *J,
                                                         double *H, double al, Coef c2) {
  __shared__ double buf[HWARP * HPACKED];                  // 31.5 KiB
  const int lane = threadIdx.x;
  const int64_t base = blockIdx.x * (int64_t)HWARP, i = base + lane;
  if (i < n) {
    if (!MAT) {
      smf::eval3<KIND, double, 1>(x + 6 * i, F + 6 * i, J + 36 * i, al, buf + HPACKED * lane, c2.c);
    } else {
      double v[6], f[6], j[36];
      smf::from_matrix(x + 9 * i, v);
      smf::eval3<KIND, double, 1>(v, f, j, al, buf + HPACKED * lane, c2.c);
      smf::to_matrix(f, j, nullptr, F + 9 * i, J + 81 * i, nullptr);
    }
  }
  __syncwarp();
  // entry e of a full H block (Voigt: e = (6 r + p) 6 + q;  matrix: e = (9 m + c) 9 + d) -> packed index and weight;
  // lane handles e = lane + 32 k
  constexpr int NH = MAT ? 729 : 216, NK = (NH + HWARP - 1) / HWARP;
  int src[NK];
  double wt[NK];
#pragma unroll
  for (int k = 0; k < NK; k++) {
    const int e = min(lane + HWARP * k, NH - 1);
    int r, p, q;
    if (MAT) {
      r = smf::midx(e / 81), p = smf::midx(e / 9 % 9), q = smf::midx(e % 9);
      wt[k] = smf::mwt(e / 9 % 9) * smf::mwt(e % 9);
    } else {
      r = e / 36, p = e / 6 % 6, q = e % 6;
      wt[k] = 1.0;
    }
    src[k] = r * 21 + smf::hpack_pq(min(p, q), max(p, q));
  }
  const int cnt = (int)min((int64_t)HWARP, n - base);
  for (int m = 0; m < cnt; m++) {
    double *o = H + NH * (base + m);
    const double *s = buf + HPACKED * m;
#pragma unroll
    for (int k = 0; k < NK; k++)
      if (lane + HWARP * k < NH) o[lane + HWARP * k] = MAT ? s[src[k]] * wt[k] : s[src[k]];
  }
}

template <int KIND, bool MAT>
void launch_kind(cudaStream_t st, unsigned blocks, int64_t n, const double *x, double *F, double *J, double *H,
                 double al, const Coef &c2) {
  if (H) {
    smf_hess_kernel<KIND, MAT><<<blocks, HWARP, 0, st>>>(n, x, F, J, H, al, c2);
  } else {
    smf_eval_kernel<KIND, MAT><<<blocks, SMF_BLOCK, 0, st>>>(n, x, F, J, H, al, c2);
  }
}

template <bool MAT>
void launch_any(int kind, cudaStream_t st, unsigned blocks, int64_t n, const double *x, double *F, double *J,
                double *H, double al, const Coef &c2) {
  switch (kind) {
    case SMF_EXP: launch_kind<smf::EXP, MAT>(st, blocks, n, x, F, J, H, al, c2); break;
    case SMF_LOG: launch_kind<smf::LOG, MAT>(st, blocks, n, x, F, J, H, al, c2); break;
    case SMF_SQRT: launch_kind<smf::SQRT, MAT>(st, blocks, n, x, F, J, H, al, c2); break;
    default: launch_kind<smf::POW, MAT>(st, blocks, n, x, F, J, H, al, c2); break;
  }
}

// 0, -1 (arguments), -2 (CUDA error, returned in *err: cudaGetLastError() also resets the error, so a second call
// would only see cudaSuccess)
int launch(cudaStream_t st, int kind, double alpha, int64_t n, const double *x, bool mat, double *F, double *J,
           double *H, cudaError_t *err = nullptr) {
  if (kind < SMF_EXP || kind > SMF_SQRT || n < 0 || (H && !J) || (kind == SMF_POW && !(alpha > 0))) return -1;
  if (kind != SMF_POW) alpha = kind == SMF_SQRT ? 0.5 : 0.0;
  if (n == 0) return 0;
  // gridDim.x is limited to 2^31 - 1 (larger values would also wrap around in the unsigned int of dim3)
  const int64_t blocks = (n - 1) / (H ? HWARP : SMF_BLOCK) + 1;
  if (blocks > INT32_MAX) {
    if (err) *err = cudaErrorInvalidConfiguration;
    return -2;
  }
  Coef c2;
  smf::dd2_coef(kind, alpha, c2.c);
  if (mat)
    launch_any<true>(kind, st, (unsigned)blocks, n, x, F, J, H, alpha, c2);
  else
    launch_any<false>(kind, st, (unsigned)blocks, n, x, F, J, H, alpha, c2);
  const cudaError_t e = cudaGetLastError();
  if (err) *err = e;
  return e == cudaSuccess ? 0 : -2;
}

// ------------------------------------------------------------------ general matrices (gmf_kernel.hpp)
struct GCoef {                     // series coefficients of the group series (computed on the host)
  double b[smf::gmf::KMAX];
};

#ifndef GMF_BLOCK
#define GMF_BLOCK 64
#endif

template <int KIND, int ORDER>
__global__ void __launch_bounds__(GMF_BLOCK) gmf_eval_kernel(int64_t n, const double *A, double *F, double *J,
                                                              double *H, double al, GCoef bt) {
  const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  smf::gmf::eval<KIND, ORDER, double>(A + 9 * i, F + 9 * i, ORDER >= 1 ? J + 81 * i : nullptr,
                                      ORDER >= 2 ? H + 729 * i : nullptr, al, bt.b);
}

template <int KIND>
void glaunch_kind(cudaStream_t st, unsigned blocks, int64_t n, const double *A, double *F, double *J, double *H,
                  double al, const GCoef &bt) {
  if (H) gmf_eval_kernel<KIND, 2><<<blocks, GMF_BLOCK, 0, st>>>(n, A, F, J, H, al, bt);
  else if (J) gmf_eval_kernel<KIND, 1><<<blocks, GMF_BLOCK, 0, st>>>(n, A, F, J, H, al, bt);
  else gmf_eval_kernel<KIND, 0><<<blocks, GMF_BLOCK, 0, st>>>(n, A, F, J, H, al, bt);
}

int glaunch(cudaStream_t st, int kind, double alpha, int64_t n, const double *A, double *F, double *J, double *H,
            cudaError_t *err = nullptr) {
  if (kind < SMF_EXP || kind > SMF_SQRT || n < 0 || (H && !J) || (kind == SMF_POW && !(alpha > 0))) return -1;
  if (kind == SMF_SQRT) kind = SMF_POW, alpha = 0.5;
  if (kind != SMF_POW) alpha = 0.0;
  if (n == 0) return 0;
  const int64_t blocks = (n - 1) / GMF_BLOCK + 1;
  if (blocks > INT32_MAX) {
    if (err) *err = cudaErrorInvalidConfiguration;
    return -2;
  }
  GCoef bt;
  smf::gmf::series_coef(kind, alpha, bt.b);
  switch (kind) {
    case SMF_EXP: glaunch_kind<smf::EXP>(st, (unsigned)blocks, n, A, F, J, H, alpha, bt); break;
    case SMF_LOG: glaunch_kind<smf::LOG>(st, (unsigned)blocks, n, A, F, J, H, alpha, bt); break;
    default: glaunch_kind<smf::POW>(st, (unsigned)blocks, n, A, F, J, H, alpha, bt); break;
  }
  const cudaError_t e = cudaGetLastError();
  if (err) *err = e;
  return e == cudaSuccess ? 0 : -2;
}

}  // namespace

extern "C" int gmf_cuda_eval(int kind, double alpha, int64_t n, const double *A, double *F, double *J, double *H,
                             void *stream) {
  return glaunch((cudaStream_t)stream, kind, alpha, n, A, F, J, H);
}

extern "C" int smf_cuda_eval(int kind, double alpha, int64_t n, const double *v, double *F, double *J, double *H,
                             void *stream) {
  return launch((cudaStream_t)stream, kind, alpha, n, v, false, F, J, H);
}

extern "C" int smf_cuda_eval_matrix(int kind, double alpha, int64_t n, const double *A, double *F, double *J,
                                    double *H, void *stream) {
  return launch((cudaStream_t)stream, kind, alpha, n, A, true, F, J, H);
}

#ifdef MATFUN_WITH_XLA_FFI
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

ffi::Error Ffi(cudaStream_t stream, ffi::Buffer<ffi::F64> x, double *F, double *J, double *H, int64_t kind,
               double alpha) {
  auto dims = x.dimensions();
  if (dims.size() == 0 || dims[dims.size() - 1] != 6)
    return ffi::Error::InvalidArgument("matfun: expected Voigt vectors of shape (..., 6)");
  cudaError_t err = cudaSuccess;
  const int r = launch(stream, (int)kind, alpha, x.element_count() / 6, x.typed_data(), false, F, J, H, &err);
  if (r == -1) return ffi::Error::InvalidArgument("matfun: invalid kind or alpha");
  if (r != 0) return ffi::Error::Internal(std::string("matfun: CUDA launch failed: ") + cudaGetErrorString(err));
  return ffi::Error::Success();
}

ffi::Error ValueImpl(cudaStream_t st, ffi::Buffer<ffi::F64> x, ffi::ResultBuffer<ffi::F64> F, int64_t kind,
                     double alpha) {
  return Ffi(st, x, F->typed_data(), nullptr, nullptr, kind, alpha);
}

ffi::Error ValueJacImpl(cudaStream_t st, ffi::Buffer<ffi::F64> x, ffi::ResultBuffer<ffi::F64> F,
                        ffi::ResultBuffer<ffi::F64> J, int64_t kind, double alpha) {
  return Ffi(st, x, F->typed_data(), J->typed_data(), nullptr, kind, alpha);
}

ffi::Error ValueJacHessImpl(cudaStream_t st, ffi::Buffer<ffi::F64> x, ffi::ResultBuffer<ffi::F64> F,
                            ffi::ResultBuffer<ffi::F64> J, ffi::ResultBuffer<ffi::F64> H, int64_t kind,
                            double alpha) {
  return Ffi(st, x, F->typed_data(), J->typed_data(), H->typed_data(), kind, alpha);
}

ffi::Error GFfi(cudaStream_t stream, ffi::Buffer<ffi::F64> x, double *F, double *J, double *H, int64_t kind,
                double alpha) {
  auto dims = x.dimensions();
  if (dims.size() < 2 || dims[dims.size() - 1] != 3 || dims[dims.size() - 2] != 3)
    return ffi::Error::InvalidArgument("matfun: expected matrices of shape (..., 3, 3)");
  cudaError_t err = cudaSuccess;
  const int r = glaunch(stream, (int)kind, alpha, x.element_count() / 9, x.typed_data(), F, J, H, &err);
  if (r == -1) return ffi::Error::InvalidArgument("matfun: invalid kind or alpha");
  if (r != 0) return ffi::Error::Internal(std::string("matfun: CUDA launch failed: ") + cudaGetErrorString(err));
  return ffi::Error::Success();
}

ffi::Error GValueImpl(cudaStream_t st, ffi::Buffer<ffi::F64> x, ffi::ResultBuffer<ffi::F64> F, int64_t kind,
                      double alpha) {
  return GFfi(st, x, F->typed_data(), nullptr, nullptr, kind, alpha);
}

ffi::Error GValueJacImpl(cudaStream_t st, ffi::Buffer<ffi::F64> x, ffi::ResultBuffer<ffi::F64> F,
                         ffi::ResultBuffer<ffi::F64> J, int64_t kind, double alpha) {
  return GFfi(st, x, F->typed_data(), J->typed_data(), nullptr, kind, alpha);
}

ffi::Error GValueJacHessImpl(cudaStream_t st, ffi::Buffer<ffi::F64> x, ffi::ResultBuffer<ffi::F64> F,
                             ffi::ResultBuffer<ffi::F64> J, ffi::ResultBuffer<ffi::F64> H, int64_t kind,
                             double alpha) {
  return GFfi(st, x, F->typed_data(), J->typed_data(), H->typed_data(), kind, alpha);
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(GmfValueCuda, GValueImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Attr<int64_t>("kind")
                                  .Attr<double>("alpha"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(GmfValueJacCuda, GValueJacImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Attr<int64_t>("kind")
                                  .Attr<double>("alpha"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(GmfValueJacHessCuda, GValueJacHessImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Attr<int64_t>("kind")
                                  .Attr<double>("alpha"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(SmfValueCuda, ValueImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Attr<int64_t>("kind")
                                  .Attr<double>("alpha"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(SmfValueJacCuda, ValueJacImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Attr<int64_t>("kind")
                                  .Attr<double>("alpha"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(SmfValueJacHessCuda, ValueJacHessImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Attr<int64_t>("kind")
                                  .Attr<double>("alpha"));

// XLA FFI API version the handlers were built against (major * 1000 + minor); matfun.jax only registers them with
// a jaxlib of the same major and at least this minor version
extern "C" int smf_xla_ffi_api_version(void) { return XLA_FFI_API_MAJOR * 1000 + XLA_FFI_API_MINOR; }
#endif
