"""
test_capi.py -- matfun.smf (NumPy -> C library) against testdata/cases.txt, the same check as in all other
languages, plus shapes and argument errors.  Needs no JAX.      pytest python/tests -v
"""
import os

import numpy as np
import pytest

import matfun

CASES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "testdata", "cases.txt")
NAMES = {0: "exp", 1: "log", 2: "pow", 3: "sqrt"}


def load_cases():
    D = np.loadtxt(CASES)
    assert D.shape[1] == 267
    return D


def relerr(x, ref):
    return np.max(np.abs(x - ref)) / max(np.max(np.abs(ref)), 1e-300)


def matrix(v):
    """symmetric matrix (..., 3, 3) from Voigt vectors (..., 6)"""
    v = np.asarray(v)
    return v[..., [[0, 3, 4], [3, 1, 5], [4, 5, 2]]]


# ------------------------------------------------------------------ C library (NumPy)
def test_capi_cases():
    D = load_cases()
    worst = {}
    for row in D:
        kind, alpha, tol = int(row[0]), row[1], row[2]
        F, J, H = matfun.smf(NAMES[kind], row[3:9], alpha if kind == 2 else None, order=2)
        for name, x, ref in (("F", F, row[9:15]), ("J", J, row[15:51]), ("H", H, row[51:])):
            e = relerr(x.ravel(), ref)
            assert e <= tol, (NAMES[kind], alpha, name, e)
            worst[name] = max(worst.get(name, 0.0), e)
    print("capi vs cases.txt:", worst)


def test_capi_shapes_and_matrix_layout():
    """Voigt and matrix input, shapes; the matrix results are the Voigt ones with dv/dA_cd = 1 (diagonal) or 1/2
    (exact), and have the symmetries of the index pairs"""
    V = np.tile([2.0, 1.5, 1.2, 0.3, -0.2, 0.1], (4, 5, 1)) + 0.01 * np.arange(20).reshape(4, 5, 1)
    F, J, H = matfun.smf("log", V, order=2)
    assert F.shape == (4, 5, 6) and J.shape == (4, 5, 6, 6) and H.shape == (4, 5, 6, 6, 6)
    Fm, Jm, Hm = matfun.smf("log", matrix(V), order=2)
    assert Fm.shape == (4, 5, 3, 3) and Jm.shape == (4, 5, 3, 3, 3, 3) and Hm.shape == (4, 5) + (3,) * 6
    I, W = np.array([[0, 3, 4], [3, 1, 5], [4, 5, 2]]), np.where(np.eye(3, dtype=bool), 1.0, 0.5)
    Jr = J[..., I[:, :, None, None], I] * W
    Hr = H[..., I[:, :, None, None, None, None], I[:, :, None, None], I] * (W[:, :, None, None] * W)
    assert np.array_equal(Fm, matrix(F)) and np.array_equal(Jm, Jr) and np.array_equal(Hm, Hr)
    for T, perms in ((Jm, [(1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)]),
                     (Hm, [(1, 0, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5), (0, 1, 4, 5, 2, 3)])):
        for p in perms:
            assert np.max(np.abs(T - np.transpose(T, (0, 1) + tuple(2 + q for q in p)))) <= 1e-15 * np.max(np.abs(T))


def test_capi_errors():
    V = np.tile([2.0, 1.5, 1.2, 0.3, -0.2, 0.1], (3, 1))
    A = matrix(V)
    N = A.copy()
    N[1, 0, 1] += 0.1                                            # one non-symmetric matrix
    with pytest.raises(ValueError, match="symmetric"):
        matfun.smf("log", N)
    T = A.copy()
    T[1, 0, 1] += 1e-13                                          # round-off asymmetry is accepted
    assert np.allclose(matfun.smf("log", T), matfun.smf("log", A), rtol=1e-12, atol=0)
    with pytest.raises(ValueError):
        matfun.smf("log", V[..., :4].reshape(3, 2, 2))
    with pytest.raises(ValueError):
        matfun.smf("pow", V)
    with pytest.raises(ValueError):
        matfun.smf("cosh", V)
    with pytest.raises(ValueError):
        matfun.smf("exp", V[..., :5])


@pytest.mark.parametrize("variant", ["avx512", "avx2", "scalar"])
def test_capi_simd_variants(variant):
    """every kernel variant of the run-time dispatch against cases.txt (fresh process: the choice is made once)"""
    import subprocess
    import sys
    code = ("import numpy as np, sys; sys.path.insert(0, %r); import test_capi as t; from matfun import _capi; "
            "print(_capi.simd_name()); t.test_capi_cases()" % os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, MATFUN_SIMD=variant), capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    used = r.stdout.split()[0]
    if used != variant:
        pytest.skip(f"{variant} not usable on this machine (got {used})")
