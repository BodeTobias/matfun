// test_cuda_api.cu -- smf_cuda_eval / smf_cuda_eval_matrix / gmf_cuda_eval (device pointers) against smf_eval /
// smf_eval_matrix / gmf_eval (CPU) for random SPD (smf) and general (gmf) matrices, all kinds, F / F+J / F+J+H, n not
// a multiple of the block sizes; and the error codes.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "matfun/matfun.h"
#include "matfun/matfun_cuda.h"

static double maxrel(const std::vector<double> &a, const std::vector<double> &b) {
  double d = 0, s = 0;
  for (size_t i = 0; i < a.size(); i++) {
    if (std::isnan(a[i]) && std::isnan(b[i])) continue;
    d = std::fmax(d, std::isnan(a[i] - b[i]) ? INFINITY : std::fabs(a[i] - b[i]));
    s = std::fmax(s, std::fabs(b[i]));
  }
  return d / s;
}

int main() {
  const int64_t n = 1000;
  std::mt19937_64 rng(7);
  std::normal_distribution<double> nd;
  std::vector<double> x(6 * n), A(9 * n);
  for (int64_t i = 0; i < n; i++) {                     // A = B B^T + 0.5 I
    double B[9], *a = A.data() + 9 * i;
    for (double &b : B) b = nd(rng);
    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 3; c++) a[3 * r + c] = B[3 * r] * B[3 * c] + B[3 * r + 1] * B[3 * c + 1] + B[3 * r + 2] * B[3 * c + 2] + (r == c ? 0.5 : 0);
    double v[6] = {a[0], a[4], a[8], a[1], a[2], a[5]};
    for (int k = 0; k < 6; k++) x[6 * i + k] = v[k];
  }
  A[9 * 17 + 1] += 0.1;                                 // one non-symmetric matrix: NaN on the GPU, -3 on the CPU
  double *dx, *dA, *dF, *dJ, *dH;
  cudaMalloc(&dx, 6 * n * sizeof(double));
  cudaMalloc(&dA, 9 * n * sizeof(double));
  cudaMalloc(&dF, 9 * n * sizeof(double));
  cudaMalloc(&dJ, 81 * n * sizeof(double));
  cudaMalloc(&dH, 729 * n * sizeof(double));
  cudaMemcpy(dx, x.data(), 6 * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dA, A.data(), 9 * n * sizeof(double), cudaMemcpyHostToDevice);

  const int kinds[5] = {SMF_EXP, SMF_LOG, SMF_SQRT, SMF_POW, SMF_POW};
  const double alphas[5] = {0, 0, 0, 0.3, 1.7};
  const char *names[5] = {"exp", "log", "sqrt", "pow 0.3", "pow 1.7"};
  int fails = 0;
  for (int mat = 0; mat < 2; mat++)
    for (int k = 0; k < 5; k++)
      for (int order = 0; order < 3; order++) {
        const int64_t nf = mat ? 9 : 6, nj = mat ? 81 : 36, nh = mat ? 729 : 216;
        std::vector<double> F(nf * n), J(nj * n), H(nh * n), Fg(nf * n), Jg(nj * n), Hg(nh * n);
        double *Jc = order ? J.data() : nullptr, *Hc = order > 1 ? H.data() : nullptr;
        double *Jd = order ? dJ : nullptr, *Hd = order > 1 ? dH : nullptr;
        const int rc = mat ? smf_eval_matrix(kinds[k], alphas[k], n, A.data(), F.data(), Jc, Hc)
                           : smf_eval(kinds[k], alphas[k], n, x.data(), F.data(), Jc, Hc);
        const int r = mat ? smf_cuda_eval_matrix(kinds[k], alphas[k], n, dA, dF, Jd, Hd, nullptr)
                          : smf_cuda_eval(kinds[k], alphas[k], n, dx, dF, Jd, Hd, nullptr);
        cudaMemcpy(Fg.data(), dF, nf * n * sizeof(double), cudaMemcpyDeviceToHost);
        double e = maxrel(Fg, F);
        if (order) {
          cudaMemcpy(Jg.data(), dJ, nj * n * sizeof(double), cudaMemcpyDeviceToHost);
          e = std::fmax(e, maxrel(Jg, J));
        }
        if (order > 1) {
          cudaMemcpy(Hg.data(), dH, nh * n * sizeof(double), cudaMemcpyDeviceToHost);
          e = std::fmax(e, maxrel(Hg, H));
        }
        bool nan_ok = true;                              // matrix 17: NaN on both (maxrel skips NaN pairs)
        if (mat)
          for (int64_t m = 0; m < nf; m++) nan_ok &= std::isnan(Fg[nf * 17 + m]) && std::isnan(F[nf * 17 + m]);
        const bool ok = r == 0 && rc == (mat ? -3 : 0) && e <= 1e-13 && nan_ok;
        printf("%-6s %-8s %-8s GPU vs CPU max rel. diff %.1e %s\n", mat ? "matrix" : "Voigt", names[k],
               order == 0 ? "F" : order == 1 ? "F+J" : "F+J+H", e, ok ? "ok" : "FAIL");
        fails += !ok;
      }
  // general matrices: B / 2 + 1.5 I (with a defective and a rotation block and one matrix outside the domain of log)
  std::vector<double> G(9 * n);
  for (int64_t i = 0; i < n; i++)
    for (int k = 0; k < 9; k++) G[9 * i + k] = 0.3 * nd(rng) + (k % 4 == 0 ? 1.5 : 0.0);
  const double shear[9] = {1, 0.7, 0, 0, 1, 0, 0, 0, 1}, rot[9] = {0.6, -0.8, 0, 0.8, 0.6, 0, 0, 0, 1.2},
               neg[9] = {-1, 0, 0, 0, 2, 0, 0, 0, 3};
  for (int k = 0; k < 9; k++) G[9 * 3 + k] = shear[k], G[9 * 4 + k] = rot[k], G[9 * 5 + k] = neg[k];
  cudaMemcpy(dA, G.data(), 9 * n * sizeof(double), cudaMemcpyHostToDevice);
  for (int k = 0; k < 5; k++)
    for (int order = 0; order < 3; order++) {
      std::vector<double> F(9 * n), J(81 * n), H(729 * n), Fg(9 * n), Jg(81 * n), Hg(729 * n);
      const int rc = gmf_eval(kinds[k], alphas[k], n, G.data(), F.data(), order ? J.data() : nullptr,
                              order > 1 ? H.data() : nullptr);
      const int r = gmf_cuda_eval(kinds[k], alphas[k], n, dA, dF, order ? dJ : nullptr, order > 1 ? dH : nullptr,
                                  nullptr);
      cudaMemcpy(Fg.data(), dF, 9 * n * sizeof(double), cudaMemcpyDeviceToHost);
      double e = maxrel(Fg, F);
      if (order) {
        cudaMemcpy(Jg.data(), dJ, 81 * n * sizeof(double), cudaMemcpyDeviceToHost);
        e = std::fmax(e, maxrel(Jg, J));
      }
      if (order > 1) {
        cudaMemcpy(Hg.data(), dH, 729 * n * sizeof(double), cudaMemcpyDeviceToHost);
        e = std::fmax(e, maxrel(Hg, H));
      }
      bool nan_ok = true;                                // matrix 5 (eigenvalue -1): NaN on both for log, sqrt, pow
      for (int m = 0; m < 9; m++) nan_ok &= std::isnan(Fg[45 + m]) == (kinds[k] != SMF_EXP) && std::isnan(F[45 + m]) == std::isnan(Fg[45 + m]);
      const bool ok = r == 0 && rc == 0 && e <= 1e-13 && nan_ok;
      printf("gmf    %-8s %-8s GPU vs CPU max rel. diff %.1e %s\n", names[k], order == 0 ? "F" : order == 1 ? "F+J" : "F+J+H",
             e, ok ? "ok" : "FAIL");
      fails += !ok;
    }
  const bool gcodes = gmf_cuda_eval(9, 0, n, dA, dF, nullptr, nullptr, nullptr) == -1 &&
                      gmf_cuda_eval(SMF_POW, 0, n, dA, dF, nullptr, nullptr, nullptr) == -1 &&
                      gmf_cuda_eval(SMF_EXP, 0, n, dA, dF, nullptr, dH, nullptr) == -1 &&
                      gmf_cuda_eval(SMF_EXP, 0, 0, dA, dF, nullptr, nullptr, nullptr) == 0;
  printf("gmf error codes %s\n", gcodes ? "ok" : "FAIL");
  fails += !gcodes;
  const bool codes = smf_cuda_eval(9, 0, n, dx, dF, nullptr, nullptr, nullptr) == -1 &&
                     smf_cuda_eval(SMF_POW, 0, n, dx, dF, nullptr, nullptr, nullptr) == -1 &&
                     smf_cuda_eval(SMF_EXP, 0, n, dx, dF, nullptr, dH, nullptr) == -1 &&
                     smf_cuda_eval_matrix(SMF_EXP, 0, -1, dA, dF, nullptr, nullptr, nullptr) == -1 &&
                     smf_cuda_eval(SMF_EXP, 0, 0, dx, dF, nullptr, nullptr, nullptr) == 0;
  // grids beyond the limit of 2^31 - 1 blocks (32 matrices each) -> -2 without a launch: exactly 2^31 blocks, and
  // 2^35 blocks (would wrap around to 0 in the unsigned int of dim3)
  const bool launch_err =
      smf_cuda_eval(SMF_EXP, 0, (int64_t)INT32_MAX * 32 + 1, dx, dF, nullptr, nullptr, nullptr) == -2 &&
      smf_cuda_eval(SMF_EXP, 0, (int64_t)1 << 40, dx, dF, nullptr, nullptr, nullptr) == -2 &&
      cudaGetLastError() == cudaSuccess;
  printf("launch error -> -2 %s\n", launch_err ? "ok" : "FAIL");
  fails += !launch_err;
  printf("error codes %s\n", codes ? "ok" : "FAIL");
  fails += !codes;
  cudaFree(dx); cudaFree(dA); cudaFree(dF); cudaFree(dJ); cudaFree(dH);
  printf(fails ? "%d FAILED\n" : "all passed\n", fails);
  return fails != 0;
}
