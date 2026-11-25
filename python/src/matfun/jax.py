"""
matfun.jax -- matrix functions of 3x3 matrices for JAX.

    smf(kind, A, alpha=None, order=0)     as matfun.smf (symmetric):  A (..., 3, 3) -> F (..., 3, 3),
                                          J (..., 3, 3, 3, 3), H (..., 3, 3, 3, 3, 3, 3);  Voigt vectors (..., 6) ->
                                          F (..., 6), J (..., 6, 6), H (..., 6, 6, 6);  F, (F, J) or (F, J, H) for
                                          order 0, 1, 2
    gmf(kind, A, alpha=None, order=0)     as matfun.gmf (general, non-symmetric):  A (..., 3, 3) -> F, J, H as above
                                          (the full derivatives)

Works under jit and vmap and is differentiable (jvp, jacfwd, jacrev, grad, hessian, nested) up to second order, also
the J of order 1 (dJ = H dA); third derivatives return NaN.  smf: the derivative w.r.t. a matrix A is that of
f((A + A^T)/2), the same as J.  Matrices must be symmetric; since no error can be raised inside jit, the results
of a matrix with |A_ij - A_ji| > 1e-8 max |A_kl| (float32: 1e-4) are NaN.  gmf: log, sqrt and pow give NaN for a
matrix with a real eigenvalue <= 0.

Which code runs is decided per backend when a function is lowered (a primitive with a lowering per platform): the C
kernel of libmatfun on the CPU, the CUDA kernel of the plugin matfun-cuda13 / matfun-cuda12 on NVIDIA GPUs
(pip install "matfun[cuda13]"), float32 in float64 there (with jax_enable_x64); pure JAX otherwise (other backends).
The pure-JAX version is also the primitive's impl, for tools that inline primitives.
MATFUN_LIB / MATFUN_CUDA_LIB override the library paths.
"""
import ctypes
import functools
import importlib
import os
import re
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import Primitive
from jax.interpreters import batching, mlir

from . import _capi
from . import _jaxgen as _gmf
from . import _jaxpure as _smf

__all__ = ["smf", "gmf"]

_KIND = {"exp": 0, "log": 1, "pow": 2, "sqrt": 3}
# families: Voigt vectors and symmetric matrices (smf), general matrices (gmf); targets and handler symbols per order
_TARGETS = {"voigt": ("matfun_value", "matfun_value_jac", "matfun_value_jac_hess"),
            "matrix": ("matfun_matrix_value", "matfun_matrix_value_jac", "matfun_matrix_value_jac_hess"),
            "general": ("matfun_gen_value", "matfun_gen_value_jac", "matfun_gen_value_jac_hess")}
_SYMBOLS = {"voigt": ("SmfValue", "SmfValueJac", "SmfValueJacHess"),
            "matrix": ("SmfMatrixValue", "SmfMatrixValueJac", "SmfMatrixValueJacHess"),
            "general": ("GmfValue", "GmfValueJac", "GmfValueJacHess")}
_SHAPES = {"voigt": ((6,), (6, 6), (6, 6, 6)), "matrix": ((3, 3), (3, 3, 3, 3), (3, 3, 3, 3, 3, 3))}
_SHAPES["general"] = _SHAPES["matrix"]
_NIN = {"voigt": 1, "matrix": 2, "general": 2}              # trailing axes of one input


def _jaxlib_ffi_api():
    """XLA FFI API version of the installed jaxlib (major * 1000 + minor), from its c_api.h; None if unknown"""
    try:
        with open(os.path.join(jax.ffi.include_dir(), "xla", "ffi", "api", "c_api.h")) as fh:
            txt = fh.read()
        major, minor = (int(re.search(rf"#define XLA_FFI_API_{k} (\d+)", txt).group(1)) for k in ("MAJOR", "MINOR"))
        return 1000 * major + minor
    except Exception:
        return None


def _register(lib, platform, suffix=""):
    """register the handlers of lib; the families it has (a library older than a family has none of its symbols).
    OSError if the jaxlib is too old for them: a rejected handler would make XLA fail to initialise the whole backend
    (then matfun.jax uses pure JAX instead)"""
    ours, theirs = lib.smf_xla_ffi_api_version(), _jaxlib_ffi_api()
    if theirs is not None and (ours // 1000 != theirs // 1000 or ours % 1000 > theirs % 1000):
        warnings.warn(f"matfun: {os.path.basename(lib._name)} needs XLA FFI API >= {ours // 1000}.{ours % 1000}, the "
                      f"installed jaxlib has {theirs // 1000}.{theirs % 1000}; using pure JAX on {platform}",
                      RuntimeWarning, stacklevel=2)
        raise OSError("incompatible XLA FFI API")
    fams = set()
    for fam, symbols in _SYMBOLS.items():
        try:
            caps = [getattr(lib, sym + suffix) for sym in symbols]
        except AttributeError:
            continue
        for name, cap in zip(_TARGETS[fam], caps):
            jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(cap), platform=platform)
        fams.add(fam)
    if "voigt" not in fams:
        raise AttributeError("no matfun handlers")
    return frozenset(fams)


try:                                    # C kernel (built with MATFUN_WITH_XLA_FFI)
    _lib = ctypes.cdll.LoadLibrary(_capi.library_path())
    _fams_cpu = _register(_lib, "cpu")
except (OSError, AttributeError):
    _lib, _fams_cpu = None, frozenset()


_PLUGINS = ("matfun_cuda13", "matfun_cuda12")


def _cuda_candidates():
    """MATFUN_CUDA_LIB, else the libraries of the installed plugin packages (CUDA 13 first)"""
    if os.environ.get("MATFUN_CUDA_LIB"):
        return [os.environ["MATFUN_CUDA_LIB"]]
    paths = []
    for name in _PLUGINS:
        try:
            paths.append(importlib.import_module(name).library_path())
        except ImportError:
            pass
    return paths


def _load_cuda():
    for path in _cuda_candidates():     # the first one that loads (its libcudart must be found) and registers
        try:
            lib = ctypes.cdll.LoadLibrary(path)
            return lib, path, _register(lib, "CUDA", "Cuda")
        except (OSError, AttributeError):
            continue
    return None, None, frozenset()


_lib_cuda, _lib_cuda_path, _fams_cuda = _load_cuda()


def _norm(kind, alpha):
    if kind == "sqrt":
        kind, alpha = "pow", 0.5
    if kind == "pow":
        try:
            alpha = float(alpha)
        except Exception:
            raise TypeError("smf: the exponent alpha must be a concrete number (Python float), not a traced value") from None
        if not alpha > 0:
            raise ValueError("smf: pow needs an exponent alpha > 0")
        if alpha == 0.5:
            kind = "sqrt"
    elif kind not in ("exp", "log"):
        raise ValueError(f"unknown function {kind!r}, expected one of ['exp', 'log', 'pow', 'sqrt']")
    else:
        alpha = 0.0
    return kind, alpha


# --------------------------------------------------------------------------------------
# matrix layout of symmetric matrices: input (A + A^T)/2, results from the Voigt layout
# --------------------------------------------------------------------------------------
_VI = np.array([[0, 3, 4], [3, 1, 5], [4, 5, 2]])        # Voigt index of A_ab
_W = np.where(np.eye(3, dtype=bool), 1.0, 0.5)           # dv / dA_cd for the symmetric part (A + A^T)/2


def _sym(A):
    """(A + A^T)/2 (exactly symmetric); NaN for a matrix that is not symmetric"""
    S = 0.5 * (A + jnp.swapaxes(A, -1, -2))
    B = jax.lax.stop_gradient(A)
    tol = 1e-8 if A.dtype == jnp.float64 else 1e-4
    asym = jnp.max(jnp.abs(B - jnp.swapaxes(B, -1, -2)), axis=(-2, -1)) > tol * jnp.max(jnp.abs(B), axis=(-2, -1))
    return jnp.where(asym[..., None, None], jnp.nan, S)


def _voigt(S):
    return jnp.stack([S[..., i, j] for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))], axis=-1)


def _to_matrix(out):
    """(F,), (F, J), (F, J, H) in the Voigt layout -> matrix layout: F_ab = Fv[VI_ab], J_abcd = Jv[VI_ab, VI_cd] W_cd,
    H_abcdef = Hv[VI_ab, VI_cd, VI_ef] W_cd W_ef"""
    W = jnp.asarray(_W, out[0].dtype)
    res = [out[0][..., _VI]]
    if len(out) > 1:
        res.append(out[1][..., _VI[:, :, None, None], _VI] * W)
    if len(out) > 2:
        res.append(out[2][..., _VI[:, :, None, None, None, None], _VI[:, :, None, None], _VI] * (W[:, :, None, None] * W))
    return tuple(res)


# --------------------------------------------------------------------------------------
# primitive: x -> (F,), (F, J) or (F, J, H) of a family.  Tracing only needs the shapes; lowered to the kernel on a
# platform that has one, else to pure JAX, which is also the impl
# --------------------------------------------------------------------------------------
def _pure(x, *, fam, order, kind, alpha):
    """pure-JAX version of the kernels"""
    if fam == "matrix":
        return _to_matrix(_pure(_voigt(x), fam="voigt", order=order, kind=kind, alpha=alpha))
    jkind, jalpha = ("pow", 0.5) if kind == "sqrt" else (kind, alpha)
    k = _NIN[fam]
    evaluate = _smf.evaluate if fam == "voigt" else _gmf.evaluate
    out = jax.vmap(lambda u: evaluate(jkind, jalpha, u, order))(x.reshape((-1,) + x.shape[-k:]))
    return tuple(y.reshape(x.shape[:-k] + y.shape[1:]) for y in out)


def _abstract(x, *, fam, order, kind, alpha):
    return [jax.core.ShapedArray(x.shape[:-_NIN[fam]] + s, x.dtype) for s in _SHAPES[fam][:order + 1]]


def _lowering(fams):
    """lowering of a platform with the kernels of the families fams: matrices through the Voigt kernel where the
    matrix one is missing; float32 in float64 (if x64 is enabled); pure JAX for other dtypes"""
    def lower(x, *, fam, order, kind, alpha):
        p = dict(order=order, kind=kind, alpha=alpha)
        if x.dtype == jnp.float32 and jax.dtypes.canonicalize_dtype(np.float64) == np.float64:
            return tuple(y.astype(x.dtype) for y in lower(x.astype(jnp.float64), fam=fam, **p))
        if x.dtype != jnp.float64 or fam not in fams and (fam != "matrix" or "voigt" not in fams):
            return _pure(x, fam=fam, **p)
        if fam not in fams:
            return _to_matrix(lower(_voigt(x), fam="voigt", **p))
        out = [jax.ShapeDtypeStruct(x.shape[:-_NIN[fam]] + s, x.dtype) for s in _SHAPES[fam][:order + 1]]
        res = jax.ffi.ffi_call(_TARGETS[fam][order], out)(x, kind=np.int64(_KIND[kind]), alpha=np.float64(alpha))
        return tuple(res)
    return lower


def _batch(args, dims, **params):
    (x,), (d,) = args, dims
    return _p.bind(jnp.moveaxis(x, d, 0), **params), (0,) * (params["order"] + 1)


_p = Primitive("matfun")
_p.multiple_results = True
_p.def_impl(_pure)
_p.def_abstract_eval(_abstract)
batching.primitive_batchers[_p] = _batch
mlir.register_lowering(_p, mlir.lower_fun(_pure, multiple_results=True))
if _lib is not None:
    mlir.register_lowering(_p, mlir.lower_fun(_lowering(_fams_cpu), multiple_results=True), platform="cpu")
if _lib_cuda is not None:
    mlir.register_lowering(_p, mlir.lower_fun(_lowering(_fams_cuda), multiple_results=True), platform="cuda")


# --------------------------------------------------------------------------------------
# derivatives: dF = J dx, dJ = H dx with J and H of the next order (so that second derivatives also run through the
# kernel); the derivative of H is NaN
# --------------------------------------------------------------------------------------
_EINSUM = {1: ("...ip,...p->...i", "...ipq,...q->...ip"), 2: ("...abcd,...cd->...ab", "...abcdef,...ef->...abcd")}


@functools.lru_cache(maxsize=None)
def _make(fam, kind, alpha, order):
    """order 0: F, 1: (F, J), 2: (F, J, H), differentiable"""
    ein = _EINSUM[_NIN[fam]]

    @jax.custom_jvp
    def g(x):
        out = _p.bind(x, fam=fam, order=order, kind=kind, alpha=alpha)
        return tuple(out) if order else out[0]

    @g.defjvp
    def g_jvp(primals, tangents):
        (x,), (dx,) = primals, tangents
        if order == 2:
            out = g(x)
            return out, tuple(jnp.full_like(y, jnp.nan) for y in out)
        nxt = _make(fam, kind, alpha, order + 1)(x)
        dF = jnp.einsum(ein[0], nxt[1], dx)
        if order == 0:
            return nxt[0], dF
        return nxt[:2], (dF, jnp.einsum(ein[1], nxt[2], dx))

    return g


@functools.lru_cache(maxsize=None)
def _jitted(fam, kind, alpha, order):
    g = _make(fam, kind, alpha, order)
    return jax.jit((lambda A: g(_sym(A))) if fam == "matrix" else g)


def smf(kind, A, alpha=None, order=0):
    """f(A) and its first and second derivative for symmetric 3x3 matrices, as matfun.smf:
        A (..., 3, 3):  F (..., 3, 3),  J (..., 3, 3, 3, 3) = dF_ab/dA_cd,  H (..., 3, 3, 3, 3, 3, 3)
        A (..., 6) Voigt vectors (A11, A22, A33, A12, A13, A23):  F (..., 6),  J (..., 6, 6),  H (..., 6, 6, 6)
    order 0: F, 1: (F, J), 2: (F, J, H).  kind in {"exp", "log", "sqrt", "pow"}; pow needs alpha > 0 (a Python
    float).  Differentiable w.r.t. A; non-symmetric matrices give NaN."""
    if order not in (0, 1, 2):
        raise ValueError("order must be 0, 1 or 2")
    kind, alpha = _norm(kind, alpha)
    A = jnp.asarray(A)
    if A.shape[-2:] == (3, 3):
        return _jitted("matrix", kind, alpha, order)(A)
    if A.shape[-1:] == (6,):
        return _jitted("voigt", kind, alpha, order)(A)
    raise ValueError(f"expected matrices (..., 3, 3) or Voigt vectors (..., 6), got shape {A.shape}")


def gmf(kind, A, alpha=None, order=0):
    """f(A) and its first and second derivative for general (non-symmetric) real 3x3 matrices, as matfun.gmf:
        A (..., 3, 3):  F (..., 3, 3),  J (..., 3, 3, 3, 3) = dF_ab/dA_cd,  H (..., 3, 3, 3, 3, 3, 3)
    order 0: F, 1: (F, J), 2: (F, J, H).  kind in {"exp", "log", "sqrt", "pow"}; pow needs alpha > 0 (a Python
    float).  Principal branch; NaN for a real eigenvalue <= 0 (log, sqrt, pow).  Differentiable w.r.t. A."""
    if order not in (0, 1, 2):
        raise ValueError("order must be 0, 1 or 2")
    kind, alpha = _norm(kind, alpha)
    A = jnp.asarray(A)
    if A.shape[-2:] != (3, 3):
        raise ValueError(f"expected matrices (..., 3, 3), got shape {A.shape}")
    return _jitted("general", kind, alpha, order)(A)
