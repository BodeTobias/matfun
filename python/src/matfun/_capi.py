"""
matfun._capi -- NumPy interface to the C library libmatfun (no JAX needed); public as matfun.smf and matfun.gmf.
Threads: OpenMP (OMP_NUM_THREADS).  Instruction set: chosen at run time (MATFUN_SIMD=avx512|avx2|scalar forces one).
"""
import ctypes
import glob
import os

import numpy as np

KINDS = {"exp": 0, "log": 1, "pow": 2, "sqrt": 3}
_HERE = os.path.dirname(os.path.abspath(__file__))
_lib = None


def library_path():
    """path of libmatfun (MATFUN_LIB overrides the copy installed with the package)"""
    if os.environ.get("MATFUN_LIB"):
        return os.environ["MATFUN_LIB"]
    found = sorted(glob.glob(os.path.join(_HERE, "libmatfun.*")) + glob.glob(os.path.join(_HERE, "matfun.dll")))
    return found[0] if found else os.path.join(_HERE, "libmatfun.so")


def _load():
    global _lib
    if _lib is None:
        lib = ctypes.cdll.LoadLibrary(library_path())
        for fn in (lib.smf_eval, lib.smf_eval_matrix, lib.gmf_eval):
            fn.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_int64] + [ctypes.c_void_p] * 4
            fn.restype = ctypes.c_int
        lib.smf_version.restype = ctypes.c_char_p
        lib.smf_simd_name.restype = ctypes.c_char_p
        _lib = lib
    return _lib


def available():
    """True if the C library can be loaded"""
    try:
        _load()
        return True
    except OSError:
        return False


def version():
    return _load().smf_version().decode()


def simd_name():
    """kernel variant chosen at run time: "avx512", "avx2", "scalar" (or "native"); environment variable
    MATFUN_SIMD=avx512|avx2|scalar forces one if it is usable (set before the first call)"""
    return _load().smf_simd_name().decode()


def smf(kind, A, alpha=None, order=0):
    """f(A) and its first and second derivative for symmetric 3x3 matrices, given as matrices A (..., 3, 3) or as
    Voigt vectors A (..., 6) = (A11, A22, A33, A12, A13, A23); the results have the same form:
        matrices:  F (..., 3, 3),  J (..., 3, 3, 3, 3) = dF_ab/dA_cd,  H (..., 3, 3, 3, 3, 3, 3) = d2F_ab/dA_cd dA_ef
        Voigt:     F (..., 6),     J (..., 6, 6) = dF_i/dv_k,          H (..., 6, 6, 6) = d2F_i/dv_p dv_q
    order 0: F, 1: (F, J), 2: (F, J, H).  kind: "exp", "log", "sqrt" or "pow" (alpha > 0).  Matrices must be
    symmetric (|A_ij - A_ji| <= 1e-8 max |A_kl|, otherwise ValueError)."""
    if kind not in KINDS:
        raise ValueError(f"unknown function {kind!r}, expected one of {sorted(KINDS)}")
    if kind == "pow" and not (alpha is not None and alpha > 0):
        raise ValueError("pow needs an exponent alpha > 0")
    if order not in (0, 1, 2):
        raise ValueError("order must be 0, 1 or 2")
    A = np.ascontiguousarray(A, dtype=np.float64)
    if A.shape[-2:] == (3, 3):
        sh, shapes, fn = A.shape[:-2], ((3, 3), (3, 3, 3, 3), (3, 3, 3, 3, 3, 3)), _load().smf_eval_matrix
    elif A.shape[-1:] == (6,):
        sh, shapes, fn = A.shape[:-1], ((6,), (6, 6), (6, 6, 6)), _load().smf_eval
    else:
        raise ValueError(f"expected matrices (..., 3, 3) or Voigt vectors (..., 6), got shape {A.shape}")
    out = [np.empty(sh + s) for s in shapes[:order + 1]] + [None] * (2 - order)
    ptr = lambda a: None if a is None else a.ctypes.data
    r = fn(KINDS[kind], float(alpha or 0.0), int(np.prod(sh, dtype=np.int64)), ptr(A), *map(ptr, out))
    if r == -3:
        raise ValueError("matrix input must be symmetric: |A_ij - A_ji| <= 1e-8 max |A_kl| for each matrix")
    if r != 0:
        raise ValueError(f"smf_eval returned {r}")
    return tuple(out[:order + 1]) if order else out[0]


def gmf(kind, A, alpha=None, order=0):
    """f(A) and its first and second derivative for general (non-symmetric) real 3x3 matrices A (..., 3, 3),
    principal branch:  F (..., 3, 3),  J (..., 3, 3, 3, 3) = dF_ab/dA_cd,  H (..., 3, 3, 3, 3, 3, 3) = d2F_ab/dA_cd dA_ef
    (the full derivatives, exact also for defective A and complex eigenvalues).  order 0: F, 1: (F, J), 2: (F, J, H).
    kind: "exp", "log", "sqrt" or "pow" (alpha > 0); log, sqrt and pow give NaN for a matrix with a real eigenvalue
    <= 0."""
    if kind not in KINDS:
        raise ValueError(f"unknown function {kind!r}, expected one of {sorted(KINDS)}")
    if kind == "pow" and not (alpha is not None and alpha > 0):
        raise ValueError("pow needs an exponent alpha > 0")
    if order not in (0, 1, 2):
        raise ValueError("order must be 0, 1 or 2")
    A = np.ascontiguousarray(A, dtype=np.float64)
    if A.shape[-2:] != (3, 3):
        raise ValueError(f"expected matrices (..., 3, 3), got shape {A.shape}")
    sh = A.shape[:-2]
    out = [np.empty(sh + s) for s in ((3, 3), (3, 3, 3, 3), (3, 3, 3, 3, 3, 3))[:order + 1]] + [None] * (2 - order)
    ptr = lambda a: None if a is None else a.ctypes.data
    r = _load().gmf_eval(KINDS[kind], float(alpha or 0.0), int(np.prod(sh, dtype=np.int64)), ptr(A), *map(ptr, out))
    if r != 0:
        raise ValueError(f"gmf_eval returned {r}")
    return tuple(out[:order + 1]) if order else out[0]
