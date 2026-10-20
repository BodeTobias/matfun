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

Which code runs is decided per backend when a function is compiled (jax.lax.platform_dependent): the C kernel of
libmatfun on the CPU, the CUDA kernel of the plugin matfun-cuda13 / matfun-cuda12 on NVIDIA GPUs
(pip install "matfun[cuda13]"), pure JAX otherwise (other backends, float32).
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

from . import _capi
from . import _jaxgen as _gmf
from . import _jaxpure as _smf

__all__ = ["smf", "gmf"]

_KIND = {"exp": 0, "log": 1, "pow": 2, "sqrt": 3}
_TARGETS = ("matfun_value", "matfun_value_jac", "matfun_value_jac_hess")
_GTARGETS = ("matfun_gen_value", "matfun_gen_value_jac", "matfun_gen_value_jac_hess")


def _jaxlib_ffi_api():
    """XLA FFI API version of the installed jaxlib (major * 1000 + minor), from its c_api.h; None if unknown"""
    try:
        with open(os.path.join(jax.ffi.include_dir(), "xla", "ffi", "api", "c_api.h")) as fh:
            txt = fh.read()
        major, minor = (int(re.search(rf"#define XLA_FFI_API_{k} (\d+)", txt).group(1)) for k in ("MAJOR", "MINOR"))
        return 1000 * major + minor
    except Exception:
        return None


def _register(lib, symbols, platform, targets=_TARGETS):
    """register the handlers of lib, unless the jaxlib is too old for them: a rejected handler would make XLA fail to
    initialise the whole backend (then matfun.jax uses pure JAX instead)"""
    ours, theirs = lib.smf_xla_ffi_api_version(), _jaxlib_ffi_api()
    if theirs is not None and (ours // 1000 != theirs // 1000 or ours % 1000 > theirs % 1000):
        warnings.warn(f"matfun: {os.path.basename(lib._name)} needs XLA FFI API >= {ours // 1000}.{ours % 1000}, the "
                      f"installed jaxlib has {theirs // 1000}.{theirs % 1000}; using pure JAX on {platform}",
                      RuntimeWarning, stacklevel=2)
        raise OSError("incompatible XLA FFI API")
    caps = [getattr(lib, sym) for sym in symbols]            # all or none (AttributeError: older library)
    for name, cap in zip(targets, caps):
        jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(cap), platform=platform)


def _register_gen(lib, symbols, platform):
    """the handlers of gmf; False if lib has none (a library older than gmf: then gmf uses pure JAX there)"""
    try:
        _register(lib, symbols, platform, _GTARGETS)
        return True
    except (OSError, AttributeError):
        return False


try:                                    # C kernel (built with MATFUN_WITH_XLA_FFI)
    _lib = ctypes.cdll.LoadLibrary(_capi.library_path())
    _register(_lib, ("SmfValue", "SmfValueJac", "SmfValueJacHess"), "cpu")
    _gen_cpu = _register_gen(_lib, ("GmfValue", "GmfValueJac", "GmfValueJacHess"), "cpu")
except (OSError, AttributeError):
    _lib, _gen_cpu = None, False


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
            _register(lib, ("SmfValueCuda", "SmfValueJacCuda", "SmfValueJacHessCuda"), "CUDA")
            return lib, path
        except (OSError, AttributeError):
            continue
    return None, None


_lib_cuda, _lib_cuda_path = _load_cuda()
_gen_cuda = _lib_cuda is not None and _register_gen(_lib_cuda, ("GmfValueCuda", "GmfValueJacCuda",
                                                                "GmfValueJacHessCuda"), "CUDA")


def _dispatch(name, nout, v, kind, alpha, jax_impl):
    """compiled kernel on CPU / CUDA where available (chosen per backend at compile time), else jax_impl"""
    branches = {}
    if _lib is not None:
        branches["cpu"] = lambda v: _ffi(name, v, kind, alpha, nout)
    if _lib_cuda is not None:
        branches["cuda"] = lambda v: _ffi(name, v, kind, alpha, nout)
    if not branches or v.dtype != jnp.float64:
        return jax_impl(v)
    return jax.lax.platform_dependent(v, default=jax_impl, **branches)


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
# Voigt vectors v (..., 6) -> F (..., 6), J (..., 6, 6), H (..., 6, 6, 6): compiled kernel or pure-JAX fallback
# --------------------------------------------------------------------------------------
_SHAPES = ((6,), (6, 6), (6, 6, 6))


def _ffi(name, v, kind, alpha, nout):
    """nout = 1: F;  2: F, J;  3: F, J, H"""
    out = [jax.ShapeDtypeStruct(v.shape[:-1] + s, v.dtype) for s in _SHAPES[:nout]]
    res = jax.ffi.ffi_call(name, out, vmap_method="broadcast_all")(v, kind=np.int64(_KIND[kind]), alpha=np.float64(alpha))
    return tuple(res) if nout > 1 else res[0]


def _impl(order, kind, alpha):
    """pure-JAX version of the kernel: v (..., 6) -> (F,), (F, J) or (F, J, H)"""
    jkind, jalpha = ("pow", 0.5) if kind == "sqrt" else (kind, alpha if kind == "pow" else None)

    def f(v):
        out = jax.vmap(lambda u: _smf.evaluate(jkind, jalpha, u, order))(v.reshape(-1, 6))
        return tuple(x.reshape(v.shape[:-1] + x.shape[1:]) for x in out)
    return f


@functools.lru_cache(maxsize=None)
def _make_vjh(kind, alpha):
    """(F, J, H); third derivative = NaN"""
    @jax.custom_jvp
    def vjh(v):
        return _dispatch("matfun_value_jac_hess", 3, v, kind, alpha, _impl(2, kind, alpha))

    @vjh.defjvp
    def vjh_jvp(primals, tangents):
        out = vjh(primals[0])
        return out, tuple(jnp.full_like(x, jnp.nan) for x in out)

    return vjh


@functools.lru_cache(maxsize=None)
def _make_vj(kind, alpha):
    """(F, J) with derivative rule dF = J dv, dJ = H dv (H from the kernel), so that second derivatives also run
    through the compiled kernel"""
    @jax.custom_jvp
    def vj(v):
        return _dispatch("matfun_value_jac", 2, v, kind, alpha, _impl(1, kind, alpha))

    @vj.defjvp
    def vj_jvp(primals, tangents):
        (v,), (dv,) = primals, tangents
        Fv, J, H = _make_vjh(kind, alpha)(v)
        return (Fv, J), (jnp.einsum("...ip,...p->...i", J, dv), jnp.einsum("...ipq,...q->...ip", H, dv))

    return vj


@functools.lru_cache(maxsize=None)
def _make_v(kind, alpha):
    """F with derivative rule dF = J dv (J, and H for second derivatives, from the kernel)"""
    @jax.custom_jvp
    def v_(v):
        return _dispatch("matfun_value", 1, v, kind, alpha, lambda v: _impl(0, kind, alpha)(v)[0])

    @v_.defjvp
    def v_jvp(primals, tangents):
        (v,), (dv,) = primals, tangents
        Fv, J = _make_vj(kind, alpha)(v)
        return Fv, jnp.einsum("...ip,...p->...i", J, dv)

    return v_


# --------------------------------------------------------------------------------------
# matrices: Voigt vector of A in, results expanded to the matrix layout (linear, differentiated by JAX itself)
# --------------------------------------------------------------------------------------
_VI = np.array([[0, 3, 4], [3, 1, 5], [4, 5, 2]])        # Voigt index of A_ab
_W = np.where(np.eye(3, dtype=bool), 1.0, 0.5)           # dv / dA_cd for the symmetric part (A + A^T)/2


def _from_matrix(A):
    """Voigt vectors (..., 6) of the symmetric parts of A (..., 3, 3); NaN for a matrix that is not symmetric"""
    S = 0.5 * (A + jnp.swapaxes(A, -1, -2))
    v = jnp.stack([S[..., i, j] for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))], axis=-1)
    B = jax.lax.stop_gradient(A)
    tol = 1e-8 if A.dtype == jnp.float64 else 1e-4
    asym = jnp.max(jnp.abs(B - jnp.swapaxes(B, -1, -2)), axis=(-2, -1)) > tol * jnp.max(jnp.abs(B), axis=(-2, -1))
    return jnp.where(asym[..., None], jnp.nan, v)


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


@functools.lru_cache(maxsize=None)
def _jitted(kind, alpha, order, mat):
    core = (_make_v, _make_vj, _make_vjh)[order](kind, alpha)
    if not mat:
        return jax.jit(core)

    def f(A):
        out = core(_from_matrix(A))
        out = _to_matrix(out if order else (out,))
        return out if order else out[0]
    return jax.jit(f)


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
        return _jitted(kind, alpha, order, True)(A)
    if A.shape[-1:] == (6,):
        return _jitted(kind, alpha, order, False)(A)
    raise ValueError(f"expected matrices (..., 3, 3) or Voigt vectors (..., 6), got shape {A.shape}")


# --------------------------------------------------------------------------------------
# general matrices: A (..., 3, 3) -> F (..., 3, 3), J (..., 3, 3, 3, 3), H (..., 3, 3, 3, 3, 3, 3)
# --------------------------------------------------------------------------------------
_GSHAPES = ((3, 3), (3, 3, 3, 3), (3, 3, 3, 3, 3, 3))


def _gffi(name, A, kind, alpha, nout):
    out = [jax.ShapeDtypeStruct(A.shape[:-2] + s, A.dtype) for s in _GSHAPES[:nout]]
    res = jax.ffi.ffi_call(name, out, vmap_method="broadcast_all")(A, kind=np.int64(_KIND[kind]),
                                                                   alpha=np.float64(alpha))
    return tuple(res)


def _gimpl(order, kind, alpha):
    """pure-JAX version of the kernel: A (..., 3, 3) -> (F,), (F, J) or (F, J, H)"""
    jkind, jalpha = ("pow", 0.5) if kind == "sqrt" else (kind, alpha)

    def f(A):
        out = jax.vmap(lambda X: _gmf.evaluate(jkind, jalpha, X, order))(A.reshape(-1, 3, 3))
        return tuple(x.reshape(A.shape[:-2] + x.shape[1:]) for x in out)
    return f


def _gdispatch(order, A, kind, alpha):
    """compiled kernel on CPU / CUDA where available (chosen per backend at compile time), else pure JAX"""
    name, nout = _GTARGETS[order], order + 1
    branches = {}
    if _gen_cpu:
        branches["cpu"] = lambda A: _gffi(name, A, kind, alpha, nout)
    if _gen_cuda:
        branches["cuda"] = lambda A: _gffi(name, A, kind, alpha, nout)
    impl = _gimpl(order, kind, alpha)
    if not branches or A.dtype != jnp.float64:
        return impl(A)
    return jax.lax.platform_dependent(A, default=impl, **branches)


@functools.lru_cache(maxsize=None)
def _gmake(kind, alpha, order):
    """order 0: F, 1: (F, J), 2: (F, J, H), differentiable: dF = J dA, dJ = H dA (from the kernel of the next order);
    the derivative of H is NaN"""
    if order == 2:
        @jax.custom_jvp
        def g(A):
            return _gdispatch(2, A, kind, alpha)

        @g.defjvp
        def g_jvp(primals, tangents):
            out = g(primals[0])
            return out, tuple(jnp.full_like(x, jnp.nan) for x in out)
        return g

    @jax.custom_jvp
    def g(A):
        out = _gdispatch(order, A, kind, alpha)
        return out if order else out[0]

    @g.defjvp
    def g_jvp(primals, tangents):
        (A,), (dA,) = primals, tangents
        nxt = _gmake(kind, alpha, order + 1)(A)
        dF = jnp.einsum("...abcd,...cd->...ab", nxt[1], dA)
        if order == 0:
            return nxt[0], dF
        return nxt[:2], (dF, jnp.einsum("...abcdef,...ef->...abcd", nxt[2], dA))
    return g


@functools.lru_cache(maxsize=None)
def _gjitted(kind, alpha, order):
    return jax.jit(_gmake(kind, alpha, order))


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
    return _gjitted(kind, alpha, order)(A)
