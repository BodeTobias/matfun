"""
make_testdata.py -- writes testdata/cases.txt (smf) and testdata/cases_general.txt (gmf), the cross-language reference
cases (F, J, H from the C kernel).

    python testdata/make_testdata.py [path/to/libmatfun.so]      (default: build/libmatfun.so)
    python testdata/make_testdata.py --compare other_lib.so         (max. differences of another build to cases.txt)

Format (plain text, one case per line, whitespace separated, %.17g):
    kind alpha tol v[6] F[6] J[36] H[216]                           (267 numbers)
kind: 0 exp, 1 log, 2 pow, 3 sqrt; alpha only used for pow.  tol: allowed max|x - ref| / max|ref| for each of F, J, H
of the case: 1e-12, and 1e-11 for the ill-conditioned cases (graded, wide spectrum), where the C kernel deviates from
a 60-digit reference by up to 1.2e-12 (python/tests/test_mpref.py) and two correct implementations (e.g. SIMD vs
scalar libm) differ by up to 5.8e-13.  Voigt layout of smf_eval (c/include/matfun/matfun.h), row-major:
F[i], J[6i+k] = dF_i/dv_k, H[(6i+p)*6+q] = d2F_i/dv_p dv_q, i, k, p, q Voigt indices, v = (A11, A22, A33, A12, A13, A23).
Lines starting with # are comments.  The cases check that all implementations agree with the C kernel.

cases_general.txt (gmf_eval, general real matrices, matrix layout, row-major):
    kind alpha tol A[9] F[9] J[81] H[729]                           (831 numbers)
J[9 m + c] = dF_m/dA_c, H[(9 m + c) 9 + e] = d2F_m/dA_c dA_e (m, c, e = 3 row + column); tol 1e-12, 1e-11 for the
ill-conditioned cases (graded / wide spectrum, nearly defective), up to 1e-9 for rotations within 1e-4 degrees of
180 (log, powers; condition ~ 1 / (pi - angle)^k).
"""
import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cases.txt")
OUT_GENERAL = os.path.join(HERE, "cases_general.txt")
KINDS = [(0, 0.0, "exp"), (1, 0.0, "log"), (3, 0.0, "sqrt"), (2, 0.3, "pow 0.3"), (2, 1.7, "pow 1.7"),
         (2, 60.5, "pow 60.5")]


def load(path):
    lib = ctypes.cdll.LoadLibrary(os.path.abspath(path))
    lib.smf_eval.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_int64] + [ctypes.c_void_p] * 4
    lib.gmf_eval.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_int64] + [ctypes.c_void_p] * 4
    lib.smf_version.restype = ctypes.c_char_p
    return lib


def evaluate(lib, kind, alpha, V):
    n = len(V)
    V = np.ascontiguousarray(V, dtype=np.float64)
    F, J, H = np.empty((n, 6)), np.empty((n, 36)), np.empty((n, 216))
    r = lib.smf_eval(kind, alpha, n, V.ctypes.data, F.ctypes.data, J.ctypes.data, H.ctypes.data)
    assert r == 0
    return F, J, H


def sym(Q, lams):
    return Q @ np.diag(lams) @ Q.T


def matrices(kind, alpha=0.0):
    """list of (label, 3x3 matrix); SPD for log/sqrt/pow, also indefinite for exp.  New cases are appended at the
    end, so that the earlier ones keep their place in cases.txt."""
    rng = np.random.default_rng(2026)
    Q = [np.linalg.qr(rng.standard_normal((3, 3)))[0] for _ in range(40)]
    out = []
    for i in range(8):
        out.append(("random", sym(Q[i], 10 ** rng.uniform(-1, 0.5, 3))))
    for i, d in enumerate([1e-1, 1e-4, 1e-8, 1e-12, 0.0]):
        out.append((f"near double {d:g}", sym(Q[8 + i], [1.3, 1.3 * (1 + d), 0.4])))
        out.append((f"near double below {d:g}", sym(Q[13 + i], [0.7, 0.7 * (1 + d), 2.5])))
    for i, d in enumerate([1e-3, 1e-7, 1e-11]):
        out.append((f"near triple {d:g}", sym(Q[18 + i], [0.9, 0.9 * (1 + d), 0.9 * (1 - 0.7 * d)])))
    out.append(("isotropic 2 I", 2.0 * np.eye(3)))
    out.append(("exact eigenvalues 1, 1, 4", np.array([[2.0, 1, 1], [1, 2, 1], [1, 1, 2]])))
    out.append(("diagonal", np.diag([0.5, 2.0, 3.0])))
    for i, r in enumerate([1.2, 1.24, 1.26, 0.8]):                        # cluster boundaries of f[a, b, c]
        out.append((f"cluster boundary {r}", sym(Q[21 + i], [1.0, r, 1.0 / r])))
    out.append(("graded 1e-4", sym(Q[25], [1e-4, 1e-4 * (1 + 1e-3), 1.0])))
    out.append(("wide 1e-3 .. 1e2", sym(Q[26], [1e-3, 1.0, 1e2])))
    if kind == 0:
        out.append(("large exp", sym(Q[27], [5.0, 20.0, 30.0])))
        out.append(("indefinite", sym(Q[28], [-2.0, 0.5, 1.5])))
        out.append(("deviatoric", sym(Q[29], [-1.0, 0.3, 0.7])))
        out.append(("negative double", sym(Q[30], [-1.0, -1.0, 2.0])))
    if kind != 0 and alpha < 10:              # magnitudes far outside the range of the unscaled cross products
        for e in (100, -100):
            out.append((f"scaled 1e{e:+d}", 10.0 ** e * sym(Q[31], 10 ** rng.uniform(-1, 0.5, 3))))
    for e in (1e-6, 1e-10, 1e-14):            # near I: f(A) ~ e; log needs log(m) + log1p(x/m) of the deviator
        out.append((f"near identity {e:g}", np.eye(3) + e * sym(Q[32], rng.uniform(-1, 1, 3))))
    return out


def evaluate_general(lib, kind, alpha, As):
    n = len(As)
    As = np.ascontiguousarray(As, dtype=np.float64)
    F, J, H = np.empty((n, 9)), np.empty((n, 81)), np.empty((n, 729))
    r = lib.gmf_eval(kind, alpha, n, As.ctypes.data, F.ctypes.data, J.ctypes.data, H.ctypes.data)
    assert r == 0
    return F, J, H


def rot(axis, ang):
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def matrices_general(kind, alpha=0.0):
    """list of (label, 3x3 matrix) for gmf: spectrum in the domain (no real eigenvalue <= 0) except for exp.  New
    cases are appended at the end."""
    rng = np.random.default_rng(2027)
    I = np.eye(3)
    G = [rng.standard_normal((3, 3)) for _ in range(20)]
    Q = [np.linalg.qr(rng.standard_normal((3, 3)))[0] for _ in range(10)]
    X = [I + 0.5 * rng.standard_normal((3, 3)) for _ in range(10)]
    sim = lambda k, D: X[k] @ (np.diag(D) if np.ndim(D) == 1 else np.asarray(D)) @ np.linalg.inv(X[k])
    out = []
    for i in range(4):
        out.append(("random", 1.5 * I + 0.4 * G[i]))
    for i, e in enumerate((1e-3, 1e-8, 1e-13)):
        out.append((f"near identity {e:g}", I + e * G[4 + i]))
    out.append(("simple shear (defective)", I + 0.7 * np.outer([1, 0, 0], [0, 1, 0])))
    out.append(("shear, rotated basis", Q[0] @ (I + 1.3 * np.outer([1, 0, 0], [0, 1, 0])) @ Q[0].T))
    N = np.array([[0, 0.5, 0.3], [0, 0, 0.8], [0, 0, 0]])
    out.append(("Jordan 3", Q[1] @ (1.2 * I + N) @ Q[1].T))
    out.append(("Jordan 3 + 1e-10", Q[1] @ (1.2 * I + N + 1e-10 * np.outer([0, 0, 1], [1, 0, 0])) @ Q[1].T))
    for i, deg in enumerate((1e-3, 20.0, 90.0, 160.0)):
        U = Q[2 + i] @ np.diag(10 ** rng.uniform(-0.3, 0.3, 3)) @ Q[2 + i].T
        out.append((f"rotation {deg:g} deg x stretch", rot(rng.standard_normal(3), np.radians(deg)) @ U))
    for i, d in enumerate((1e-4, 1e-10, 0.0)):
        out.append((f"double real {d:g}, nonnormal", sim(i, [1.3, 1.3 * (1 + d), 0.5])))
        out.append((f"triple real {d:g}, nonnormal", sim(3 + i, [0.8, 0.8 * (1 + d), 0.8 * (1 - d)])))
    out.append(("symmetric", G[8] @ G[8].T + 0.5 * I))
    out.append(("graded 1e-4", sim(6, [1.0, 0.3, 1e-4])))
    out.append(("wide 1e-2 .. 1e2", Q[6] @ np.diag([1e-2, 1.0, 1e2]) @ Q[6].T))
    out.append(("cluster boundary", sim(7, [0.7, 1.0, 1.3])))
    out.append(("complex cluster boundary", sim(8, [[1.0, 0.45, 0], [-0.45, 1.0, 0], [0, 0, 0.6]])))
    if kind == 0:
        out.append(("exp: indefinite", 2.0 * G[9]))
        out.append(("exp: large", 40.0 * I + 2.0 * G[10]))
        out.append(("exp: spread", sim(9, [-20.0, 3.0, 15.0])))
        out.append(("exp: skew (rotation rate)", 3.0 * (G[11] - G[11].T)))
    if kind != 0 and alpha < 10:
        for e in (100, -100):
            out.append((f"scaled 1e{e:+d}", 10.0 ** e * (1.5 * I + 0.4 * G[12])))
    # complex pair near the negative real axis (rotations close to 180 degrees; log, powers: across the branch cut)
    for deg in (179.0, 179.99, 179.9999):
        out.append((f"near 180: rotation {deg} deg", rot(G[13][0], np.radians(deg))))
        out.append((f"near 180: rotation {deg} deg, non-normal", sim(9, rot(G[13][1], np.radians(deg)))))
    return out


def voigt(A):
    return np.array([A[0, 0], A[1, 1], A[2, 2], A[0, 1], A[0, 2], A[1, 2]])


if __name__ == "__main__":
    if "--compare" in sys.argv:
        other = load(sys.argv[sys.argv.index("--compare") + 1])
        D = np.loadtxt(OUT)
        worst = {}
        for row in D:
            kind, alpha, v = int(row[0]), row[1], row[3:9]
            F, J, H = (x[0] for x in evaluate(other, kind, alpha, v[None]))
            for name, x, y in (("F", F, row[9:15]), ("J", J, row[15:51]), ("H", H, row[51:])):
                e = np.max(np.abs(x - y)) / max(np.max(np.abs(y)), 1e-300)
                worst[name] = max(worst.get(name, 0), e)
        print(f"{len(D)} cases, max. relative difference (max-norm per case): " +
              ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))
        sys.exit(0)
    lib = load(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "build", "libmatfun.so"))
    lines = [f"# matfun cross-language test cases from libmatfun {lib.smf_version().decode()} (C kernel), "
             "make_testdata.py", "# kind alpha tol v[6] F[6] J[36] H[216]; kind: 0 exp, 1 log, 2 pow, 3 sqrt; "
             "Voigt layout of smf_eval in matfun.h"]
    n = 0
    for kind, alpha, name in KINDS:
        mats = matrices(kind, alpha)
        V = np.array([voigt(A) for _, A in mats])
        F, J, H = evaluate(lib, kind, alpha, V)
        lines.append(f"# {name}: " + "; ".join(lbl for lbl, _ in mats))
        for i in range(len(V)):
            tol = 1e-11 if mats[i][0].startswith(("graded", "wide")) else 1e-12
            nums = [kind, alpha, tol] + list(V[i]) + list(F[i]) + list(J[i]) + list(H[i])
            lines.append(" ".join(f"{x:.17g}" for x in nums))
            n += 1
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {n} cases to {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")
    lines = [f"# matfun cross-language test cases of gmf from libmatfun {lib.smf_version().decode()} (C kernel), "
             "make_testdata.py", "# kind alpha tol A[9] F[9] J[81] H[729]; kind: 0 exp, 1 log, 2 pow, 3 sqrt; "
             "matrix layout of gmf_eval in matfun.h"]
    n = 0
    for kind, alpha, name in KINDS:
        mats = matrices_general(kind, alpha)
        As = np.array([A for _, A in mats])
        F, J, H = evaluate_general(lib, kind, alpha, As)
        lines.append(f"# {name}: " + "; ".join(lbl for lbl, _ in mats))
        for i in range(len(As)):
            lbl = mats[i][0]
            tol = 1e-11 if lbl.startswith(("graded", "wide", "Jordan 3 +", "triple real 1e-10")) else 1e-12
            if lbl.startswith("near 180"):                      # condition ~ 1 / (pi - angle)^k
                tol = 1e-9 if "179.9999" in lbl else (1e-11 if "179.99" in lbl else 1e-12)
            nums = [kind, alpha, tol] + list(As[i].ravel()) + list(F[i]) + list(J[i]) + list(H[i])
            lines.append(" ".join(f"{x:.17g}" for x in nums))
            n += 1
    with open(OUT_GENERAL, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {n} cases to {OUT_GENERAL} ({os.path.getsize(OUT_GENERAL) / 1e6:.1f} MB)")
