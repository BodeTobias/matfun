"""
matfun -- exp, log, sqrt and A^alpha of 3x3 matrices with first and second derivatives: symmetric (smf) and general
(non-symmetric, gmf) matrices.

    F, J, H = matfun.smf("log", A, order=2)        NumPy, compiled C kernel (no JAX needed)
    F, J, H = matfun.jax.smf("log", A, order=2)    JAX: C kernel on the CPU, CUDA kernel on NVIDIA GPUs; jit, vmap, AD
    F, J, H = matfun.gmf("log", A, order=2)        general matrices (also matfun.jax.gmf)

A: matrices (..., 3, 3) -> F (..., 3, 3), J (..., 3, 3, 3, 3) = dF_ab/dA_cd, H (..., 3, 3, 3, 3, 3, 3)
   or (smf) Voigt vectors (..., 6) = (A11, A22, A33, A12, A13, A23) -> F (..., 6), J (..., 6, 6), H (..., 6, 6, 6).
kind: "exp", "log", "sqrt" or "pow" (with alpha > 0).
"""
from ._capi import gmf, smf

__version__ = "0.1.0"
__all__ = ["smf", "gmf", "backends"]


def backends():
    """which compiled kernels could be loaded: C library and its instruction set, JAX, CUDA plugin"""
    from . import _capi
    ok = _capi.available()
    info = {"version": __version__, "c": _capi.library_path() if ok else None, "c_simd": _capi.simd_name() if ok else None}
    try:
        from . import jax as _sjax
        info.update(jax=True, jax_cpu_kernel=_sjax._lib is not None, jax_cuda_kernel=_sjax._lib_cuda is not None,
                    cuda_library=_sjax._lib_cuda_path)
    except ImportError:
        info.update(jax=False)
    return info
