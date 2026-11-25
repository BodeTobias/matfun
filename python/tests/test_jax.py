"""
test_jax.py -- matfun.jax: kernels vs matfun.smf (C) and the pure-JAX fallback, AD (jacfwd / jacrev / hessian) through
the kernels against the returned J and H and against independent references (AD through eigh, expm), dispatch per
device (CPU: C kernel, GPU: CUDA kernel if the plugin is installed).  Skipped without jax.   pytest python/tests -v
"""
import re

import numpy as np
import pytest

import matfun

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import matfun.jax as mf  # noqa: E402
from matfun import _jaxpure as jaxpure  # noqa: E402

DEVICES = jax.devices() + ([jax.devices("cpu")[0]] if jax.default_backend() != "cpu" else [])
KINDS = [("exp", None), ("log", None), ("sqrt", None), ("pow", 0.3)]


def spd(shape, seed=0):
    A = np.random.default_rng(seed).standard_normal(shape + (3, 3))
    return A @ np.swapaxes(A, -1, -2) + 0.5 * np.eye(3)


def voigt(A):
    A = np.asarray(A)
    return np.stack([A[..., 0, 0], A[..., 1, 1], A[..., 2, 2], A[..., 0, 1], A[..., 0, 2], A[..., 1, 2]], axis=-1)


def custom_calls(f, *x):
    txt = jax.jit(f).lower(*x).compile().as_text()
    return sorted(set(re.findall(r'custom_call_target="(matfun\w*)"', txt)))


def jrel(x, y):
    return float(jnp.max(jnp.abs(x - y)) / jnp.max(jnp.abs(y)))


@pytest.mark.parametrize("dev", DEVICES, ids=lambda d: d.platform)
def test_jax_dispatch(dev):
    with jax.default_device(dev):
        fams = {"cpu": mf._fams_cpu, "gpu": mf._fams_cuda}[dev.platform]
        for A, fam in ((jnp.asarray(voigt(spd((8,)))), "voigt"), (jnp.asarray(spd((8,))), "matrix")):
            calls = custom_calls(lambda A: mf.smf("log", A, order=2), A)
            target = mf._TARGETS[fam if fam in fams else "voigt"][2]     # matrices: else the Voigt kernel
            assert calls == ([target] if fams else []), calls


def test_jax_trace_without_the_pure_version():
    """tracing needs only the shapes (the pure-JAX version is not traced, e.g. no while loops of gmf); derivatives too"""
    for f, A in ((lambda A: mf.gmf("log", A, order=2), gen_matrices(4, 0)),
                 (jax.hessian(lambda A: jnp.sum(mf.smf("log", A) ** 2)), spd(()))):
        txt = str(jax.make_jaxpr(f)(jnp.asarray(A)))
        assert "matfun" in txt and "while" not in txt


def test_jax_export():
    """exported for the CPU alone and for CPU and CUDA (kernel or pure JAX per platform), the CPU module runs"""
    from jax import export
    A = spd((5,), 2)
    f = jax.jit(lambda A: (mf.smf("log", A, order=2), mf.gmf("log", A, order=1),
                           mf.smf("sqrt", A[..., (0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)], order=1)))
    ref = jax.tree.leaves((matfun.smf("log", A, order=2), matfun.gmf("log", A, order=1),
                           matfun.smf("sqrt", voigt(A), order=1)))
    checks = [export.DisabledSafetyCheck.custom_call(t) for ts in mf._TARGETS.values() for t in ts]
    with jax.default_device(jax.devices("cpu")[0]):
        for platforms in (("cpu",), ("cpu", "cuda")):
            out = export.export(f, platforms=platforms, disabled_checks=checks)(jnp.asarray(A)).call(jnp.asarray(A))
            assert all(jrel(x, jnp.asarray(y)) < 1e-13 for x, y in zip(jax.tree.leaves(out), ref))


def test_jax_kernelax_inlines_the_pure_version():
    """kernelax builds C kernels from the primitive's impl (pure JAX), also of the derivatives"""
    kernelax = pytest.importorskip("kernelax")
    A = spd((8,), 3)
    with jax.default_device(jax.devices("cpu")[0]):
        F, J = jax.vmap(kernelax.kernel(lambda X: mf.smf("log", X, order=1), n_threads=1))(jnp.asarray(A))
        G = jax.vmap(kernelax.kernel(jax.grad(lambda X: jnp.sum(mf.gmf("exp", X) ** 2)), n_threads=1))(jnp.asarray(A))
        ref = jax.vmap(jax.grad(lambda X: jnp.sum(mf.gmf("exp", X) ** 2)))(jnp.asarray(A))
    assert all(jrel(x, jnp.asarray(y)) < 1e-13 for x, y in zip((F, J), matfun.smf("log", A, order=1)))
    assert jrel(G, ref) < 1e-13


@pytest.mark.parametrize("dev", DEVICES, ids=lambda d: d.platform)
@pytest.mark.parametrize("kind,alpha", KINDS)
def test_jax_vs_capi(dev, kind, alpha):
    """every order, Voigt and matrix input, matches the C library; pure JAX matches too"""
    with jax.default_device(dev):
        A = spd((64,), 1)
        for X in (voigt(A), A):
            ref = matfun.smf(kind, X, alpha, order=2)
            for order in (0, 1, 2):
                o = mf.smf(kind, jnp.asarray(X), alpha, order=order)
                for x, y in zip(o if order else (o,), ref):
                    assert x.shape == y.shape and jrel(x, jnp.asarray(y)) < 1e-13
        k, a = ("pow", 0.5) if kind == "sqrt" else (kind, alpha)
        pure = mf._pure(jnp.asarray(voigt(A)), fam="voigt", order=2, kind=k, alpha=a)
        for x, y in zip(pure, matfun.smf(kind, voigt(A), alpha, order=2)):
            assert jrel(x, jnp.asarray(y)) < 1e-13


@pytest.mark.parametrize("dev", DEVICES, ids=lambda d: d.platform)
def test_jax_ad_vs_returned_derivatives(dev):
    """jacfwd / jacrev / hessian of order 0 and jacfwd of the J of order 1 reproduce J and H, for Voigt vectors and
    matrices, also under vmap; third derivatives are NaN"""
    with jax.default_device(dev):
        A = jnp.asarray(spd((), 2))
        for X in (jnp.asarray(voigt(A)), A):
            F, J, H = mf.smf("log", X, order=2)
            f0 = lambda X: mf.smf("log", X)
            assert jrel(jax.jacfwd(f0)(X), J) < 1e-13
            assert jrel(jax.jacrev(f0)(X), J) < 1e-13
            assert jrel(jax.jacfwd(jax.jacrev(f0))(X), H) < 1e-13
            assert jrel(jax.jacrev(jax.jacfwd(f0))(X), H) < 1e-13
            assert jrel(jax.jacfwd(lambda X: mf.smf("log", X, order=1)[1])(X), H) < 1e-13
            third = jax.jacfwd(jax.jacfwd(jax.jacfwd(f0)))(X)
            assert bool(jnp.all(jnp.isnan(third)))
        B = jnp.asarray(spd((4, 5), 3))
        Jb = mf.smf("log", B, order=1)[1]
        assert jrel(jax.vmap(jax.vmap(jax.jacfwd(lambda X: mf.smf("log", X))))(B), Jb) < 1e-13


@pytest.mark.parametrize("dev", DEVICES, ids=lambda d: d.platform)
def test_jax_vs_independent_references(dev):
    """matrix layout against AD through jnp.linalg.eigh (exact for distinct eigenvalues) of A -> f((A + A^T)/2), and
    the second derivative of exp in symmetric directions against jax.scipy.linalg.expm"""
    from jax.scipy.linalg import expm

    def eigh_fun(f):
        def g(A):
            lam, V = jnp.linalg.eigh(0.5 * (A + A.T))
            return (V * f(lam)) @ V.T
        return g

    with jax.default_device(dev):
        A = jnp.asarray(spd((), 4))
        for kind, alpha, f in (("log", None, jnp.log), ("pow", 1.7, lambda x: x ** 1.7), ("exp", None, jnp.exp)):
            F, J, H = mf.smf(kind, A, alpha, order=2)
            g = eigh_fun(f)
            assert jrel(F, g(A)) < 1e-13
            assert jrel(J, jax.jacfwd(g)(A)) < 1e-11
            assert jrel(H, jax.jacfwd(jax.jacfwd(g))(A)) < 1e-10
        A = 0.3 * A
        rng = np.random.default_rng(6)
        E1, E2 = (jnp.asarray(x + x.T) for x in rng.standard_normal((2, 3, 3)))
        d2 = jax.jvp(lambda X: jax.jvp(expm, (X,), (E1,))[1], (A,), (E2,))[1]
        H = mf.smf("exp", A, order=2)[2]
        assert jrel(jnp.einsum("abcdef,cd,ef->ab", H, E1, E2), d2) < 1e-12


@pytest.mark.parametrize("dev", DEVICES, ids=lambda d: d.platform)
def test_jax_matrix_symmetries_and_check(dev):
    """J and H of matrices have the symmetries of the index pairs; a non-symmetric matrix gives NaN (round-off
    asymmetry is accepted), the others in the batch are unaffected"""
    with jax.default_device(dev):
        A = jnp.asarray(spd((3,), 5))
        F, J, H = mf.smf("sqrt", A, order=2)
        for T, perms in ((J, [(1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)]),
                         (H, [(1, 0, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5), (0, 1, 4, 5, 2, 3)])):
            for p in perms:
                assert jrel(jnp.transpose(T, (0,) + tuple(1 + q for q in p)), T) < 1e-15
        N = A.at[1, 0, 2].add(0.1).at[2, 1, 2].add(1e-13)
        FN = mf.smf("sqrt", N)
        assert bool(jnp.all(jnp.isnan(FN[1]))) and jrel(FN[0], F[0]) == 0 and jrel(FN[2], F[2]) < 1e-12
        with pytest.raises(ValueError):
            mf.smf("log", jnp.eye(2))


@pytest.mark.parametrize("dev", DEVICES, ids=lambda d: d.platform)
def test_jax_float32_with_x64(dev):
    """float32 input with jax_enable_x64 (the float64 kernels, pure JAX without): results and derivatives stay
    float32. Precision "highest": on GPUs, float32 matrix products default to TF32 (errors ~1e-4)"""
    A = spd((), 3)
    with jax.default_device(dev), jax.default_matmul_precision("highest"):
        for X in (voigt(A), A):
            X32 = jnp.asarray(X, jnp.float32)
            out, ref = mf.smf("log", X32, order=2), mf.smf("log", jnp.asarray(X), order=2)
            assert [x.dtype for x in out] == [jnp.float32] * 3
            assert all(jrel(x, y) < 1e-5 for x, y in zip(out, ref))
            Hs = jax.hessian(lambda X: jnp.sum(mf.smf("log", X) ** 2))
            assert Hs(X32).dtype == jnp.float32 and jrel(Hs(X32), Hs(jnp.asarray(X))) < 1e-5


@pytest.mark.parametrize("dev", DEVICES, ids=lambda d: d.platform)
@pytest.mark.parametrize("kind,alpha,scale", [("log", None, 1e100), ("sqrt", None, 1e-100), ("pow", 1.7, 1e100),
                                              ("pow", 0.3, 1e-250), ("pow", 60.5, 1.0), ("pow", 300.5, 0.15)])
def test_jax_extreme_scales_and_large_alpha(dev, kind, alpha, scale):
    """kernels on CPU / GPU vs the C library for extreme magnitudes and large exponents (same kernel code); the
    parameters keep F and J representable (e.g. (1e250)^1.7 would overflow)"""
    with jax.default_device(dev):
        V = scale * voigt(spd((16,), 7))
        ref = matfun.smf(kind, V, alpha, order=1)
        out = mf.smf(kind, jnp.asarray(V), alpha, order=1)
        for x, y in zip(out, ref):
            assert np.all(np.isfinite(y)) and jrel(x, jnp.asarray(y)) < 1e-13
        k, a = mf._norm(kind, alpha)                                   # pure JAX: other rounding, x^alpha cond. alpha
        pure = mf._pure(jnp.asarray(V), fam="voigt", order=1, kind=k, alpha=a)
        for x, y in zip(pure, ref):
            assert jrel(x, jnp.asarray(y)) < max(1e-13, 10 * (alpha or 1) * np.finfo(float).eps)


def test_jaxpure_scaling_is_an_exact_power_of_two():
    """the scaling of the pure-JAX path: an exact power of two within a factor 4 of x (1 for 0, NaN, Inf), for
    float64 and float32, also on the GPU (no frexp: bit casts; no exp2: inexact in XLA)"""
    rng = np.random.default_rng(0)
    for dt, lo, hi in ((np.float64, -1021, 1023), (np.float32, -125, 127)):
        e = np.concatenate([np.arange(lo, hi), rng.integers(lo, hi, 2000)])
        x = np.ldexp(np.concatenate([np.ones(hi - lo), rng.uniform(1, 2, 2000)]), e).astype(dt)
        x = np.concatenate([x, x * dt(1 - np.finfo(dt).eps)])
        for dev in DEVICES:
            with jax.default_device(dev):
                s = np.asarray(jax.jit(jax.vmap(jaxpure._pow2))(jnp.asarray(x)))
                special = np.asarray(jax.jit(jax.vmap(jaxpure._pow2))(jnp.asarray([0.0, np.nan, np.inf], dt)))
            assert s.dtype == dt and np.all(np.frexp(s)[0] == 0.5) and np.all((x / s >= 0.5) & (x / s < 4))
            assert np.array_equal(special, np.ones(3, dt))


def test_jaxpure_has_no_bit_casts():
    """the pure-JAX path (F, J, H) uses no bit-level operations (frexp, ldexp, bit casts, shifts), so that code
    generators translating the jaxpr (autopdex C element kernels) can take it"""
    txt = str(jax.make_jaxpr(lambda v: mf._pure(v, fam="voigt", order=2, kind="log", alpha=0.0))(jnp.ones((2, 6))))
    txt = txt.lower()
    assert not [w for w in ("bitcast", "frexp", "ldexp", "shift_", "ffi_call", "callback") if w in txt]


# ------------------------------------------------------------------ general matrices (gmf)
GKINDS = [("exp", None), ("log", None), ("sqrt", None), ("pow", 1.7)]


def gen_matrices(n, seed):
    A = 1.5 * np.eye(3) + 0.3 * np.random.default_rng(seed).standard_normal((n, 3, 3))
    A[1] = np.eye(3) + 0.7 * np.outer([1, 0, 0], [0, 1, 0])                  # defective
    A[2] = np.array([[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.2]])    # complex pair
    return A


@pytest.mark.parametrize("dev", DEVICES, ids=lambda d: d.platform)
def test_jax_gmf_dispatch(dev):
    with jax.default_device(dev):
        expect = "general" in {"cpu": mf._fams_cpu, "gpu": mf._fams_cuda}[dev.platform]
        A = jnp.asarray(gen_matrices(8, 0))
        calls = custom_calls(lambda A: mf.gmf("log", A, order=2), A)
        assert calls == (["matfun_gen_value_jac_hess"] if expect else []), calls


@pytest.mark.parametrize("dev", DEVICES, ids=lambda d: d.platform)
@pytest.mark.parametrize("kind,alpha", GKINDS)
def test_jax_gmf_vs_capi(dev, kind, alpha):
    """every order matches the C library; the pure-JAX version too"""
    with jax.default_device(dev):
        A = gen_matrices(64, 1)
        ref = matfun.gmf(kind, A, alpha, order=2)
        for order in (0, 1, 2):
            o = mf.gmf(kind, jnp.asarray(A), alpha, order=order)
            for x, y in zip(o if order else (o,), ref):
                assert x.shape == y.shape and jrel(x, jnp.asarray(y)) < 1e-13
        k, a = mf._norm(kind, alpha)
        pure = mf._pure(jnp.asarray(A), fam="general", order=2, kind=k, alpha=a)
        for x, y in zip(pure, ref):
            assert jrel(x, jnp.asarray(y)) < 1e-13


@pytest.mark.parametrize("dev", DEVICES, ids=lambda d: d.platform)
def test_jax_gmf_ad_vs_returned_derivatives(dev):
    """jacfwd / jacrev / hessian of order 0 and jacfwd of the J of order 1 reproduce J and H, also under vmap; third
    derivatives are NaN"""
    with jax.default_device(dev):
        A = jnp.asarray(gen_matrices(4, 2))
        for X in (A[0], A[1], A[2]):
            F, J, H = mf.gmf("log", X, order=2)
            f0 = lambda X: mf.gmf("log", X)
            assert jrel(jax.jacfwd(f0)(X), J) < 1e-13
            assert jrel(jax.jacrev(f0)(X), J) < 1e-13
            assert jrel(jax.jacfwd(jax.jacrev(f0))(X), H) < 1e-13
            assert jrel(jax.jacfwd(lambda X: mf.gmf("log", X, order=1)[1])(X), H) < 1e-13
            assert bool(jnp.all(jnp.isnan(jax.jacfwd(jax.jacfwd(jax.jacfwd(f0)))(X))))
        Jb = mf.gmf("exp", A, order=1)[1]
        assert jrel(jax.vmap(jax.jacfwd(lambda X: mf.gmf("exp", X)))(A), Jb) < 1e-13


@pytest.mark.parametrize("dev", DEVICES, ids=lambda d: d.platform)
def test_jax_gmf_vs_independent_references(dev):
    """exp against jax.scipy.linalg.expm and its derivatives by AD (non-symmetric, also defective)"""
    from jax.scipy.linalg import expm
    with jax.default_device(dev):
        for X in jnp.asarray(gen_matrices(3, 4) - np.eye(3)):
            F, J, H = mf.gmf("exp", X, order=2)
            assert jrel(F, expm(X)) < 1e-13
            assert jrel(J, jax.jacfwd(expm)(X)) < 1e-12
            assert jrel(H, jax.jacfwd(jax.jacfwd(expm))(X)) < 1e-11


@pytest.mark.parametrize("dev", DEVICES, ids=lambda d: d.platform)
def test_jax_gmf_float32_with_x64(dev):
    """float32 input (the float64 kernels, pure JAX without): results stay float32, close to float64"""
    A = gen_matrices(4, 5)
    with jax.default_device(dev), jax.default_matmul_precision("highest"):
        out = mf.gmf("log", jnp.asarray(A, jnp.float32), order=2)
        ref = mf.gmf("log", jnp.asarray(A), order=2)
        assert [x.dtype for x in out] == [jnp.float32] * 3
        assert all(jrel(x, y) < 1e-4 for x, y in zip(out, ref))


def test_jaxgen_has_no_bit_casts():
    """the pure-JAX gmf (F, J, H) uses no bit-level operations (code generators that translate the jaxpr, kernelax, take
    it; its while loops over the series terms too, also nested in a local Newton loop)"""
    txt = str(jax.make_jaxpr(lambda A: mf._pure(A, fam="general", order=2, kind="log", alpha=0.0))(jnp.ones((2, 3, 3))))
    txt = txt.lower()
    assert not [w for w in ("bitcast", "frexp", "ldexp", "shift_", "ffi_call", "callback") if w in txt]


@pytest.mark.parametrize("kind, alpha", GKINDS)
def test_jaxgen_series_terms_as_the_c_kernel(kind, alpha):
    """the pure-JAX gmf sums as many series terms as the C kernel's bound gives (per matrix, at most 128): F, J and H
    as the C library's, also for clustered nodes, a pair next to the negative real axis and wide spectra"""
    from matfun import _jaxgen
    A = gen_matrices(200, 7)
    A[3] = np.diag([1.0, 1.0 + 1e-9, 1.0 - 1e-9])                              # clustered
    A[4] = np.array([[-0.9, -0.43, 0.0], [0.43, -0.9, 0.0], [0.0, 0.0, 0.5]])  # pair near the negative real axis
    A[5:50] = 2.0 * np.random.default_rng(8).standard_normal((45, 3, 3))       # wide spectra
    jk, ja = ("pow", 0.5) if kind == "sqrt" else (kind, 0.0 if alpha is None else alpha)
    out = jax.jit(jax.vmap(lambda X: _jaxgen.evaluate(jk, ja, X, 2)))(jnp.asarray(A))
    for x, z in zip(out, matfun.gmf(kind, A, alpha, order=2)):
        x, ok = np.asarray(x), np.isfinite(z)
        assert np.array_equal(np.isnan(x), np.isnan(z))
        assert float(np.max(np.abs(x - z)[ok]) / np.max(np.abs(z[ok]))) < 1e-13
