"""
test_gmf.py -- matfun.gmf (general, non-symmetric matrices; NumPy -> C library) against testdata/cases_general.txt,
the same check as in all other languages, every kernel variant, closed-form cases, identities, consistency with smf
for symmetric input, NaN outside the domain, shapes and argument errors.  Needs no JAX.      pytest python/tests -v
"""
import os

import numpy as np
import pytest

import matfun

CASES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "testdata", "cases_general.txt")
NAMES = {0: "exp", 1: "log", 2: "pow", 3: "sqrt"}


def load_cases():
    D = np.loadtxt(CASES)
    assert D.shape[1] == 831
    return D


def relerr(x, ref):
    return np.max(np.abs(x - ref)) / max(np.max(np.abs(ref)), 1e-300)


def test_gmf_cases():
    D = load_cases()
    worst = {}
    for row in D:
        kind, alpha, tol = int(row[0]), row[1], row[2]
        F, J, H = matfun.gmf(NAMES[kind], row[3:12].reshape(3, 3), alpha if kind == 2 else None, order=2)
        for name, x, ref in (("F", F, row[12:21]), ("J", J, row[21:102]), ("H", H, row[102:])):
            e = relerr(x.ravel(), ref)
            assert e <= tol, (NAMES[kind], alpha, name, e)
            worst[name] = max(worst.get(name, 0.0), e)
    print("gmf vs cases_general.txt:", worst)


@pytest.mark.parametrize("variant", ["avx512", "avx2", "scalar"])
def test_gmf_simd_variants(variant):
    """every kernel variant of the run-time dispatch against cases_general.txt (fresh process)"""
    import subprocess
    import sys
    code = ("import numpy as np, sys; sys.path.insert(0, %r); import test_gmf as t; from matfun import _capi; "
            "print(_capi.simd_name()); t.test_gmf_cases()" % os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, MATFUN_SIMD=variant), capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    used = r.stdout.split()[0]
    if used != variant:
        pytest.skip(f"{variant} not usable on this machine (got {used})")


def test_gmf_closed_forms():
    """defective (Jordan) and rotation cases with closed-form results"""
    I = np.eye(3)
    N = np.zeros((3, 3))
    N[0, 1] = 0.7                                                # N^2 = 0
    assert relerr(matfun.gmf("exp", N), I + N) < 1e-15
    assert relerr(matfun.gmf("log", I + N), N - N @ N / 2) < 1e-15
    assert relerr(matfun.gmf("sqrt", 4 * I + N), 2 * I + N / 4) < 1e-15
    assert relerr(matfun.gmf("pow", I + N, 2.5), I + 2.5 * N) < 1e-15
    th = 0.9                                                     # rotation about z: exp of the skew generator
    K = np.array([[0, -th, 0], [th, 0, 0], [0, 0, 0]])
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    assert relerr(matfun.gmf("exp", K), R) < 1e-15
    assert relerr(matfun.gmf("log", R), K) < 1e-15
    Rh = np.array([[np.cos(th / 2), -np.sin(th / 2), 0], [np.sin(th / 2), np.cos(th / 2), 0], [0, 0, 1]])
    assert relerr(matfun.gmf("sqrt", R), Rh) < 1e-15
    # derivative of exp at 0: J = identity (dF_ab/dA_cd = delta_ac delta_bd), H_abcdef = (d_ac d_de d_fb + d_ae d_fc d_db) / 2
    F, J, H = matfun.gmf("exp", np.zeros((3, 3)), order=2)
    d = np.eye(3)
    assert np.array_equal(F, I) and relerr(J, np.einsum("ac,bd->abcd", d, d)) < 1e-16
    Href = 0.5 * (np.einsum("ac,de,fb->abcdef", d, d, d) + np.einsum("ae,fc,db->abcdef", d, d, d))
    assert relerr(H, Href) < 1e-16


def test_gmf_identities_and_derivatives():
    """exp(log A) = A, sqrt(A)^2 = A, chain rule J_exp(log A) J_log(A) = I, J and H against central differences"""
    rng = np.random.default_rng(1)
    A = np.eye(3) + 0.4 * rng.standard_normal((3, 3)) + np.outer([0, 1, 0], [1, 0, 0])
    L, JL = matfun.gmf("log", A, order=1)
    E, JE = matfun.gmf("exp", L, order=1)
    S = matfun.gmf("sqrt", A)
    assert relerr(E, A) < 1e-14 and relerr(S @ S, A) < 1e-14
    assert np.max(np.abs(np.einsum("abcd,cdef->abef", JE, JL).reshape(9, 9) - np.eye(9))) < 1e-14
    for kind, alpha in (("exp", None), ("log", None), ("pow", 1.7)):
        F, J, H = matfun.gmf(kind, A, alpha, order=2)
        h = 1e-6
        for c in range(9):
            Ec = np.zeros(9)
            Ec[c] = h
            Fp, Jp = matfun.gmf(kind, A + Ec.reshape(3, 3), alpha, order=1)
            Fm, Jm = matfun.gmf(kind, A - Ec.reshape(3, 3), alpha, order=1)
            assert relerr((Fp - Fm).ravel() / (2 * h), J.reshape(9, 9)[:, c]) < 1e-9
            assert relerr((Jp - Jm).reshape(9, 9) / (2 * h), H.reshape(9, 9, 9)[:, :, c]) < 1e-8
        assert relerr(H, np.transpose(H, (0, 1, 4, 5, 2, 3))) < 1e-15            # symmetric in (cd) <-> (ef)


def test_gmf_symmetric_input_matches_smf():
    """for symmetric A, gmf's derivatives projected on symmetric directions are those of smf (matrix layout)"""
    rng = np.random.default_rng(2)
    B = rng.standard_normal((5, 3, 3))
    A = B @ np.swapaxes(B, -1, -2) + 0.5 * np.eye(3)
    for kind, alpha in (("exp", None), ("log", None), ("sqrt", None), ("pow", 1.7)):
        F, J, H = matfun.gmf(kind, A, alpha, order=2)
        Fs, Js, Hs = matfun.smf(kind, A, alpha, order=2)
        Jp = 0.5 * (J + np.swapaxes(J, -1, -2))
        Hp = 0.5 * (H + np.swapaxes(H, -3, -4))
        Hp = 0.5 * (Hp + np.swapaxes(Hp, -1, -2))
        assert relerr(F, Fs) < 1e-14 and relerr(Jp, Js) < 1e-14 and relerr(Hp, Hs) < 1e-13, kind


def test_gmf_domain_and_batch():
    """NaN for a real eigenvalue <= 0 (log, sqrt, pow), only for that matrix of a batch; complex eigenvalues with
    negative real part are in the domain (principal branch); results do not depend on the batch neighbours"""
    rng = np.random.default_rng(3)
    A = 1.5 * np.eye(3) + 0.3 * rng.standard_normal((13, 3, 3))
    A[4] = np.diag([-1.0, 2.0, 3.0])
    A[9] = np.array([[-1.0, 2.0, 0], [-2.0, -1.0, 0], [0, 0, 1.0]])             # -1 +- 2i, 1
    for kind in ("log", "sqrt"):
        F, J, H = matfun.gmf(kind, A, order=2)
        bad = np.array([np.any(np.isnan(x)) for x in F])
        assert bad.tolist() == [i == 4 for i in range(13)], kind
        for i in (0, 9, 12):
            Fi, Ji, Hi = matfun.gmf(kind, A[i], order=2)
            assert np.array_equal(Fi, F[i]) and np.array_equal(Ji, J[i]) and np.array_equal(Hi, H[i])
    L = matfun.gmf("log", A[9])
    assert relerr(matfun.gmf("exp", L), A[9]) < 1e-14
    assert np.all(np.isfinite(matfun.gmf("exp", A, order=2)[2]))


def test_gmf_shapes_and_errors():
    A = np.tile(np.eye(3) * 1.5, (4, 5, 1, 1)) + 0.01 * np.arange(20).reshape(4, 5, 1, 1)
    F, J, H = matfun.gmf("log", A, order=2)
    assert F.shape == (4, 5, 3, 3) and J.shape == (4, 5) + (3,) * 4 and H.shape == (4, 5) + (3,) * 6
    assert matfun.gmf("exp", A[0, 0]).shape == (3, 3)
    F1, J1 = matfun.gmf("log", A, order=1)
    assert np.array_equal(F1, F) and np.array_equal(J1, J) and np.array_equal(matfun.gmf("log", A), F)
    with pytest.raises(ValueError):
        matfun.gmf("log", np.zeros(6))
    with pytest.raises(ValueError):
        matfun.gmf("pow", A)
    with pytest.raises(ValueError):
        matfun.gmf("cosh", A)
    with pytest.raises(ValueError):
        matfun.gmf("exp", A, order=3)
