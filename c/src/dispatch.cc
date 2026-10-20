// dispatch.cc -- run-time choice of the kernel variant (see dispatch.h).
// Which variants exist is set by CMake: SMF_WITH_AVX512, SMF_WITH_AVX2, SMF_WITH_NATIVE; scalar always.
// The SIMD variants call libmvec (glibc >= 2.35) through smf_mvec_table, filled here with dlopen/dlsym, so the
// library itself only needs the baseline glibc and falls back to the scalar kernel where libmvec is missing.
#include "dispatch.h"

#include <cstdlib>
#include <cstring>

#if defined(SMF_WITH_AVX512) || defined(SMF_WITH_AVX2)
#include <dlfcn.h>
#endif

extern "C" {
void *smf_mvec_table[9][11] = {};
void smf_batch_scalar(int, int64_t, const double *, double *, double *, double, double *, int);
void smf_gbatch_scalar(int, int64_t, const double *, double *, double *, double, double *);
int smf_width_scalar();
#ifdef SMF_WITH_AVX512
void smf_batch_avx512(int, int64_t, const double *, double *, double *, double, double *, int);
void smf_gbatch_avx512(int, int64_t, const double *, double *, double *, double, double *);
int smf_width_avx512();
#endif
#ifdef SMF_WITH_AVX2
void smf_batch_avx2(int, int64_t, const double *, double *, double *, double, double *, int);
void smf_gbatch_avx2(int, int64_t, const double *, double *, double *, double, double *);
int smf_width_avx2();
#endif
#ifdef SMF_WITH_NATIVE
void smf_batch_native(int, int64_t, const double *, double *, double *, double, double *, int);
void smf_gbatch_native(int, int64_t, const double *, double *, double *, double, double *);
int smf_width_native();
#endif
}

namespace smf_dispatch {
namespace {

#if defined(SMF_WITH_AVX512) || defined(SMF_WITH_AVX2)
// vector ABI names: 'e' = AVX-512 (8 doubles), 'd' = AVX2 (4 doubles); order as in smf_simd.hpp
bool load_mvec(int w) {
  static const char *const fn[11] = {"acos", "cos", "exp", "log", "atanh", "expm1", "log1p", "pow", "sin", "cbrt",
                                     "atan2"};
  void *h = dlopen("libmvec.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!h) return false;
  for (int i = 0; i < 11; i++) {
    char name[32];
    std::strcpy(name, w == 8 ? "_ZGVeN8v" : "_ZGVdN4v");
    if (i == 7 || i == 10) std::strcat(name, "v");       // pow, atan2: two vector arguments
    std::strcat(name, "_");
    std::strcat(name, fn[i]);
    void *p = dlsym(h, name);
    if (!p) return false;
    smf_mvec_table[w][i] = p;
  }
  return true;
}
#endif

bool wanted(const char *name) {
  const char *force = std::getenv("MATFUN_SIMD");
  return !force || !*force || std::strcmp(force, name) == 0;
}

Impl choose_among(bool honour_env) {
  auto ok = [&](const char *name) { return !honour_env || wanted(name); };
#ifdef SMF_WITH_NATIVE
  if (ok("native")) return {smf_batch_native, smf_gbatch_native, smf_width_native(), "native"};
#endif
#ifdef SMF_WITH_AVX512
  if (ok("avx512") && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
      __builtin_cpu_supports("avx512cd") && __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl") &&
      __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && load_mvec(8))
    return {smf_batch_avx512, smf_gbatch_avx512, smf_width_avx512(), "avx512"};
#endif
#ifdef SMF_WITH_AVX2
  if (ok("avx2") && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
      __builtin_cpu_supports("bmi") && __builtin_cpu_supports("bmi2") && __builtin_cpu_supports("f16c") &&
      load_mvec(4))
    return {smf_batch_avx2, smf_gbatch_avx2, smf_width_avx2(), "avx2"};
#endif
  if (ok("scalar")) return {smf_batch_scalar, smf_gbatch_scalar, smf_width_scalar(), "scalar"};
  return choose_among(false);              // MATFUN_SIMD names a variant that is not usable here: automatic choice
}

}  // namespace

const Impl &impl() {
  static const Impl chosen = choose_among(true);
  return chosen;
}

}  // namespace smf_dispatch
