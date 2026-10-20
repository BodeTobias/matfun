/* matfun_cuda.h -- CUDA API of libmatfun_cuda; same functions, layouts and return codes as smf_eval,
 * smf_eval_matrix and gmf_eval in matfun.h, but all pointers are device pointers and the work is queued on `stream`
 * (asynchronous).  smf_cuda_eval_matrix does not report non-symmetric matrices; their results are NaN. */
#ifndef MATFUN_CUDA_H
#define MATFUN_CUDA_H

#include <stdint.h>

#include "matfun/matfun.h"

#ifdef __cplusplus
extern "C" {
#endif

/* stream: a cudaStream_t (0 = default stream).  Returns 0, -1 for invalid arguments, -2 for a CUDA launch error. */
int smf_cuda_eval(int kind, double alpha, int64_t n, const double *v, double *F, double *J, double *H, void *stream);

int smf_cuda_eval_matrix(int kind, double alpha, int64_t n, const double *A, double *F, double *J, double *H,
                         void *stream);

/* general matrices, as gmf_eval (matfun.h) */
int gmf_cuda_eval(int kind, double alpha, int64_t n, const double *A, double *F, double *J, double *H, void *stream);

#ifdef __cplusplus
}
#endif

#endif /* MATFUN_CUDA_H */
