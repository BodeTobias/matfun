"""
test_mpref.py -- the C kernel against an independent reference in 60-digit arithmetic (mpmath):
F from mpmath's symmetric eigensolver, J and H by central finite differences of F in the same precision. None of
the Daleckii-Krein formulas or divided differences of matfun are used, so this checks correctness, whereas
testdata/cases.txt checks that all implementations agree with each other.

    pytest python/tests/test_mpref.py -v          (needs mpmath; about 30-60 s)
"""
import os

import numpy as np
import pytest

import matfun

mp = pytest.importorskip("mpmath")
mp.mp.dps = 60

CASES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "testdata", "cases.txt")
NAMES = {0: "exp", 1: "log", 2: "pow", 3: "sqrt"}
VOIGT = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def mp_fun(kind, alpha):
    if kind == 0:
        return mp.exp
    if kind == 1:
        return mp.log
    a = mp.mpf(0.5) if kind == 3 else mp.mpf(alpha)
    return lambda x: x ** a


def mp_reference(kind, alpha, v):
    """F (6), J (6,6), H (6,6,6) (Voigt layout) of the matrix with Voigt vector v, in 60 digits"""
    f = mp_fun(kind, alpha)
    A0 = mp.matrix(3, 3)
    for p, (i, j) in enumerate(VOIGT):
        A0[i, j] = A0[j, i] = mp.mpf(float(v[p]))

    def F(A):
        E, Q = mp.eigsy(A)
        return Q * mp.diag([f(E[i]) for i in range(3)]) * Q.T

    def E(p):
        M = mp.zeros(3, 3)
        i, j = VOIGT[p]
        M[i, j] = M[j, i] = 1
        return M

    s = max(abs(x) for x in A0)
    F0 = F(A0)
    h1, h2 = s * mp.mpf("1e-20"), s * mp.mpf("1e-15")       # truncation O(h^2), round-off 1e-60 / h^2
    vec = lambda M: [float(M[i, j]) for i, j in VOIGT]
    J = np.zeros((6, 6))
    for p in range(6):
        J[:, p] = vec((F(A0 + h1 * E(p)) - F(A0 - h1 * E(p))) / (2 * h1))
    H = np.zeros((6, 6, 6))
    for p in range(6):
        for q in range(p, 6):
            Ep, Eq = h2 * E(p), h2 * E(q)
            H[:, p, q] = H[:, q, p] = vec((F(A0 + Ep + Eq) - F(A0 + Ep - Eq) - F(A0 - Ep + Eq) + F(A0 - Ep - Eq))
                                          / (4 * h2 * h2))
    return np.array(vec(F0)), J, H


def relerr(x, ref):
    return float(np.max(np.abs(x - ref)) / max(np.max(np.abs(ref)), 1e-300))


def check(kind, alpha, v, tol, what):
    ref = mp_reference(kind, alpha, v)
    out = matfun.smf(NAMES[kind], np.asarray(v), alpha if kind == 2 else None, order=2)
    errs = [relerr(x, r) for x, r in zip(out, ref)]
    assert max(errs) <= tol, (what, errs)
    return errs


def test_cases_against_mpmath():
    """all cases of testdata/cases.txt (inputs only): F, J, H of the C kernel against the 60-digit reference"""
    D = np.loadtxt(CASES)
    worst = {}
    for row in D:
        kind, alpha, tol = int(row[0]), row[1], row[2]
        ill = tol > 1e-12                                     # graded / wide spectrum
        tol = 1e-11 if ill else max(1e-13, 10 * alpha * np.finfo(float).eps)   # x^alpha: condition number alpha
        errs = check(kind, alpha, row[3:9], tol, (NAMES[kind], alpha, row[3:9]))
        key = f"{NAMES[kind]} {alpha:g}" + (" ill-conditioned" if ill else "")
        worst[key] = [max(a, b) for a, b in zip(worst.get(key, [0, 0, 0]), errs)]
    for k, (eF, eJ, eH) in sorted(worst.items()):
        print(f"{k:28s} F {eF:.1e}  J {eJ:.1e}  H {eH:.1e}")


@pytest.mark.parametrize("alpha", [0.3, 1.7, 12.0, 13.5, 40.5, 60.5, 100.5, 300.5])
@pytest.mark.parametrize("lam", [(0.8, 1.0, 1.25), (1.0, 1.04, 1.06), (0.95, 1.0, 1.06)])
def test_pow_large_alpha(alpha, lam):
    """second divided differences of x^alpha in and near the cluster of the Taylor series, also for large alpha.
    x^alpha has condition number alpha: an eigenvalue error of a few ulp becomes ~ alpha * eps in F, J and H (measured:
    0.4-1.5 alpha*eps on Linux, 2 alpha*eps with MSVC on Windows), so the tolerance grows with alpha."""
    Q = np.linalg.qr(np.random.default_rng(5).standard_normal((3, 3)))[0]
    A = Q @ np.diag(lam) @ Q.T
    check(2, alpha, [A[i, j] for i, j in VOIGT], max(1e-13, 10 * alpha * np.finfo(float).eps), (alpha, lam))


# ------------------------------------------------------------------ general matrices (gmf)
CASES_GENERAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "testdata", "cases_general.txt")


def mp_eigfun(kind, alpha):
    """principal f(A) = V f(D) V^-1 from mpmath's eigendecomposition (diagonalizable A; mpmath's logm / sqrtm are
    inaccurate for a complex pair next to the negative real axis)"""
    def fm(A):
        E, V = mp.eig(A)
        fe = [mp.log(e) if kind == 1 else mp.exp((mp.mpf(0.5) if kind == 3 else mp.mpf(alpha)) * mp.log(e)) for e in E]
        return V * mp.diag(fe) * mp.inverse(V)
    return fm


def mp_reference_general(kind, alpha, a, fm=None):
    """F (9), J (9, 9), H (9, 9, 9)[:, :, HCOLS] (matrix layout) of the general matrix a (9,) in 60 digits: F from
    mpmath's expm / logm / sqrtm / powm (principal branch, none of gmf's formulas), J and H by central differences
    (truncation O(h^2), round-off 1e-60 / h^2; mpmath's logm does not always converge at 40 digits)"""
    fm = fm or {0: mp.expm, 1: mp.logm, 3: mp.sqrtm}.get(kind, lambda X: mp.powm(X, mp.mpf(alpha)))
    A0 = mp.matrix(3, 3)
    for k in range(9):
        A0[k // 3, k % 3] = mp.mpf(float(a[k]))

    def F(A):
        X = fm(A)
        return [mp.re(X[k // 3, k % 3]) for k in range(9)]

    def E(c):
        M = mp.zeros(3, 3)
        M[c // 3, c % 3] = 1
        return M

    s = max(abs(x) for x in A0)
    h1, h2 = s * mp.mpf("1e-20"), s * mp.mpf("1e-14")
    J = np.zeros((9, 9))
    for c in range(9):
        J[:, c] = [float(x) for x in (mp.matrix(F(A0 + h1 * E(c))) - mp.matrix(F(A0 - h1 * E(c)))) / (2 * h1)]
    H = np.zeros((9, 9, len(HCOLS)))
    for c in range(9):
        for k, e in enumerate(HCOLS):
            Ec, Ee = h2 * E(c), h2 * E(e)
            d = (mp.matrix(F(A0 + Ec + Ee)) - mp.matrix(F(A0 + Ec - Ee)) - mp.matrix(F(A0 - Ec + Ee))
                 + mp.matrix(F(A0 - Ec - Ee))) / (4 * h2 * h2)
            H[:, c, k] = [float(x) for x in d]
    return np.array([float(x) for x in F(A0)]), J, H


HCOLS = (0, 5, 7)                                            # columns e of H checked (a diagonal, two off-diagonal)


def test_general_cases_against_mpmath():
    """every eighth case of testdata/cases_general.txt (inputs only): F, J, H of gmf against the 60-digit reference;
    the tolerance is that of the case (1e-12, ill-conditioned 1e-11) times 10 for J, H of the finite differences
    (about 2 minutes)"""
    D = np.loadtxt(CASES_GENERAL)
    worst = {}
    with mp.workdps(60):
        for row in D[::8]:
            kind, alpha, tol = int(row[0]), row[1], row[2]
            a = row[3:12]
            ev = np.linalg.eigvals(a.reshape(3, 3))
            near_cut = kind != 0 and np.any((ev.imag != 0) & (ev.real < 0))   # mpmath's logm / sqrtm fail there
            ref = mp_reference_general(kind, alpha, a, mp_eigfun(kind, alpha) if near_cut else None)
            out = matfun.gmf(NAMES[kind], a.reshape(3, 3), alpha if kind == 2 else None, order=2)
            out = [out[0].ravel(), out[1].reshape(9, 9), out[2].reshape(9, 9, 9)[:, :, list(HCOLS)]]
            errs = [relerr(x, r) for x, r in zip(out, ref)]
            lim = max(tol, 10 * alpha * np.finfo(float).eps)
            assert errs[0] <= lim and errs[1] <= 10 * lim and errs[2] <= 10 * lim, (NAMES[kind], alpha, a, errs)
            key = f"{NAMES[kind]} {alpha:g}"
            worst[key] = [max(x, y) for x, y in zip(worst.get(key, [0, 0, 0]), errs)]
    for k, (eF, eJ, eH) in sorted(worst.items()):
        print(f"gmf {k:12s} F {eF:.1e}  J {eJ:.1e}  H {eH:.1e}")


@pytest.mark.parametrize("deg", [179.0, 179.99])
def test_general_near_180_against_mpmath(deg):
    """log and x^1.7 of a rotation close to 180 degrees (complex pair next to the branch cut): F, J, H against the
    60-digit reference (eigendecomposition, see mp_eigfun); the tolerance follows the condition number
    ~ 1 / (pi - angle)^k (k = 1, 2, 3 for F, J, H)"""
    th = np.radians(deg)
    k = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    d = np.pi - th
    with mp.workdps(60):
        for kind, alpha, code in (("log", None, 1), ("pow", 1.7, 2)):
            ref = mp_reference_general(code, alpha or 0.0, R.ravel(), mp_eigfun(code, alpha))
            out = matfun.gmf(kind, R, alpha, order=2)
            out = [out[0].ravel(), out[1].reshape(9, 9), out[2].reshape(9, 9, 9)[:, :, list(HCOLS)]]
            for i, (x, r) in enumerate(zip(out, ref)):
                assert relerr(x, r) < max(1e-13, 30 * np.finfo(float).eps / d ** (i + 1)), (kind, deg, i, relerr(x, r))
