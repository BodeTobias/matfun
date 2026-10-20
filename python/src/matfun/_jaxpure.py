"""
matfun._jaxpure -- the kernel in pure JAX: F, J and H of one Voigt vector by the Daleckii-Krein formulas (fallback of
matfun.jax where no compiled kernel runs, e.g. float32 or other backends).  Never differentiated: the derivative
rules of matfun.jax supply J and H themselves.

Method (as the C kernel, c/include/matfun/smf_kernel.hpp):
  * eigen-decomposition A = V diag(lam) V^T in closed form, orthonormal V by construction: A scaled by a power of
    two, simple eigenvalue from the trigonometric formula, its eigenvector by cross products, 2x2 Jacobi rotation
    in the orthogonal complement
  * f(A) = V f(lam) V^T,   derivative  L(A, E) = V (D o (V^T E V)) V^T  with numerically stable divided
    differences D_ij = f[lam_i, lam_j] (D_ii = f'(lam_i)), exact also for repeated eigenvalues
  * second derivative  d2F[E1, E2] = V S V^T,  S_ab = sum_m f[lam_a, lam_m, lam_b] (E1~_am E2~_mb + E2~_am E1~_mb),
    E~ = V^T E V, with stable second divided differences (Taylor series for clustered eigenvalues)
  * powers: eigenvalues < 0 from round-off are clamped to 0
"""
import functools
import math

import numpy as np

import jax.numpy as jnp

_SQRT6 = math.sqrt(6.0)
_TWO_PI_3 = 2.0 * math.pi / 3.0


# --------------------------------------------------------------------------------------
# scalar function and its stable first divided difference f[a, b] (f[a, a] = f'(a))
# --------------------------------------------------------------------------------------
def _f(kind, alpha, x):
    if kind == "exp":
        return jnp.exp(x)
    if kind == "log":
        return jnp.log(x)
    if alpha == 0.5:
        return jnp.sqrt(jnp.where(x > 0, x, 0.0))
    xs = jnp.where(x > 0, x, 1.0)
    return jnp.where(x > 0, xs ** alpha, 0.0)


def _dd(kind, alpha, a, b):
    if kind == "exp":                                   # e^hi (1 - e^-d) / d, d = hi - lo >= 0 (no overflow)
        hi = jnp.maximum(a, b)
        d = hi - jnp.minimum(a, b)
        ds = jnp.where(d > 0, d, 1.0)
        return jnp.exp(hi) * jnp.where(d > 0, -jnp.expm1(-ds) / ds, 1.0)
    if kind == "log":                                   # 2 artanh(z) / (z (a+b)), z = (a-b)/(a+b)
        s = 0.5 * (a + b)
        z = (a - b) / (a + b)
        az = jnp.abs(z)
        tiny = az < 1e-3
        mid = (~tiny) & (az < 0.5)
        zs = jnp.where(mid, z, 0.5)
        atc = jnp.where(tiny, 1.0 + z * z / 3.0 + z ** 4 / 5.0 + z ** 6 / 7.0, jnp.arctanh(zs) / zs)
        far = ~(tiny | mid)
        direct = (jnp.log(a) - jnp.log(b)) / jnp.where(far, a - b, 1.0)
        return jnp.where(far, direct, atc / s)
    a = jnp.where(a > 0, a, 0.0)                        # powers: PSD round-off clamped to 0
    b = jnp.where(b > 0, b, 0.0)
    if alpha == 0.5:
        return 1.0 / (jnp.sqrt(a) + jnp.sqrt(b))
    s = 0.5 * (a + b)
    ssafe = jnp.where(s > 0, s, 1.0)
    z = (a - b) / (2.0 * ssafe)
    az = jnp.abs(z)
    zero = az == 0
    near = (~zero) & (az < 0.5)
    zs = jnp.where(near, z, 0.25)
    g = (jnp.expm1(alpha * jnp.log1p(zs)) - jnp.expm1(alpha * jnp.log1p(-zs))) / (2.0 * zs)
    g = jnp.where(zero, alpha, g)
    far = ~(zero | near)
    asafe, bsafe = jnp.where(a > 0, a, 1.0), jnp.where(b > 0, b, 1.0)
    direct = (jnp.where(a > 0, asafe ** alpha, 0.0) - jnp.where(b > 0, bsafe ** alpha, 0.0)) / jnp.where(far, a - b, 1.0)
    sp = jnp.where(s > 0, _pow_k(ssafe, alpha, 1), _zero_pow(alpha - 1.0))
    return jnp.where(far, direct, sp * g)


def _pow_k(x, alpha, k):
    """x ** (alpha - k) with the exponent alpha - k = bh + bl exact (TwoSum): its rounding alone would cost |log x|
    times its error (1e-14 at x ~ 1e100); x^(bh + bl) = x^bh (1 + bl log x)"""
    bh = alpha - k
    bb = bh - alpha
    bl = (alpha - (bh - bb)) + (-k - bb)
    p = x ** bh
    return p if bl == 0 else p * (1.0 + bl * jnp.log(x))


def _zero_pow(e):
    """0^e (Python float): inf for e < 0, 1 for e = 0, 0 for e > 0"""
    return math.inf if e < 0 else (1.0 if e == 0 else 0.0)


_TAU = 0.25          # cluster radius (relative; exp: absolute 1; pow: min(_TAU, 3/alpha)) for the series of f[a, b, c]
_NTERMS = 30         # series terms: tau^30 * 31 < 1e-16 (exp: 1/32! * 31)


def _dd2(kind, alpha, a, b, c):
    """second divided difference f[a, b, c] (symmetric in a, b, c; f[a, a, a] = f''(a)/2), numerically stable:
    all three points in a cluster around the middle one m -> Taylor series
        f[a, b, c] = sum_k f^(k+2)(m)/(k+2)! h_k(lo - m, hi - m),   h_k(u, w) = sum_i u^i w^(k-i),
    otherwise (f[mid, hi] - f[lo, mid]) / (hi - lo) with the stable first divided differences."""
    if kind == "pow":
        a, b, c = [jnp.where(x > 0, x, 0.0) for x in (a, b, c)]
    lo = jnp.minimum(jnp.minimum(a, b), c)
    hi = jnp.maximum(jnp.maximum(a, b), c)
    mid = jnp.maximum(jnp.minimum(a, b), jnp.minimum(jnp.maximum(a, b), c))
    spread = hi - lo
    direct = (_dd(kind, alpha, mid, hi) - _dd(kind, alpha, lo, mid)) / jnp.where(spread > 0, spread, 1.0)
    if kind == "exp":
        cluster = spread <= 1.0
        u, w, scale = lo - mid, hi - mid, jnp.exp(mid)
        coef = [1.0 / math.factorial(k + 2) for k in range(_NTERMS)]
    else:
        # pow: the coefficients binom(alpha, k+2) grow with alpha, so the radius shrinks to 3/alpha for alpha > 12
        tau = min(_TAU, 3.0 / alpha) if kind == "pow" else _TAU
        cluster = (mid > 0) & (mid - lo <= tau * mid) & (hi - mid <= tau * mid)
        ms = jnp.where(cluster, mid, 1.0)
        u, w = (lo - mid) / ms, (hi - mid) / ms
        if kind == "log":                               # f^(j)(m)/j! = (-1)^(j+1) / (j m^j)
            scale = 1.0 / (ms * ms)
            coef = [(-1.0) ** (k + 3) / (k + 2) for k in range(_NTERMS)]
        else:                                           # f^(j)(m)/j! = binom(alpha, j) m^(alpha - j)
            scale = _pow_k(ms, alpha, 2)
            coef, bj = [], alpha * (alpha - 1.0) / 2.0
            for k in range(_NTERMS):
                coef.append(bj)
                bj *= (alpha - k - 2.0) / (k + 3.0)
    h, wk, series = jnp.ones_like(u), jnp.ones_like(u), coef[0] * jnp.ones_like(u)
    for k in range(1, _NTERMS):                         # h_k = u h_(k-1) + w^k
        wk = wk * w
        h = u * h + wk
        series = series + coef[k] * h
    out = jnp.where(cluster, scale * series, direct)
    if kind == "pow":                                   # all three points 0: f''(0)/2 = binom(alpha, 2) 0^(alpha-2)
        z2 = alpha * (alpha - 1.0) / 2.0 * _zero_pow(alpha - 2.0) if alpha != 1.0 else 0.0
        out = jnp.where(hi > 0, out, z2)
    return out


# --------------------------------------------------------------------------------------
# closed-form symmetric eigen-decompositions (orthonormal V by construction)
# --------------------------------------------------------------------------------------
def _det3(A):
    return (A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
            - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
            + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]))


def _normalize(v):
    n = jnp.sqrt(jnp.sum(v * v))
    ok = n > 1e-300
    return jnp.where(ok, v / jnp.where(ok, n, 1.0), 0.0 * v), ok


def _pow2(x):
    """a power of two within a factor 4 of x > 0 (1 for x = 0, NaN or Inf), exact, from compares, selects and
    products only: no bit casts (frexp) and no exp2 (inexact in XLA), so that code generators that translate the
    jaxpr (e.g. the C element kernels of autopdex) cover it"""
    ok = (x > 0) & (x < jnp.inf)
    emax = jnp.finfo(x.dtype).maxexp - 1                        # 2^-(emax-1) .. 2^emax are normal
    k = jnp.clip(jnp.floor(jnp.log(jnp.where(ok, x, 1.0)) * (1.0 / math.log(2.0))), 1 - emax, emax)
    r, s = jnp.abs(k), jnp.ones_like(x)
    for b in reversed(range(emax.bit_length())):               # 2^|k| from the binary digits of |k|
        take = r >= 2.0 ** b
        s = jnp.where(take, s * 2.0 ** (2 ** b), s)
        r = jnp.where(take, r - 2.0 ** b, r)
    return jnp.where(k < 0, 1.0 / s, s)


def _eig3(A):
    """A = V diag(lam) V^T; also m sig and x / m with lam = (m + x) sig, x the eigenvalues of A / sig - m I (for
    log lam = log(m sig) + log1p(x / m), accurate also near I)"""
    # scale A by a power of two sig ~ max |A_ij| (exact), so that the products below (up to |A|^4 in the cross
    # products) neither overflow nor underflow; the eigenvalues are scaled back at the end
    sig = _pow2(jnp.max(jnp.abs(A)) + 1e-300)
    A = A / sig
    I = jnp.eye(3, dtype=A.dtype)
    m = (A[0, 0] + A[1, 1] + A[2, 2]) / 3.0
    Ad = A - m * I                                      # exact where the diagonal is close to m (near I)
    # simple eigenvalue and its eigenvector from the deviator less its trace residue (rounding of m; the formula
    # needs a traceless matrix), normalized by its own magnitude: then independent of how small the deviator is
    Ed = Ad - (jnp.trace(Ad) / 3.0) * I
    dmax = jnp.max(jnp.abs(Ed))
    iso = dmax <= 1e8 * jnp.finfo(A.dtype).tiny         # A = m I
    Ed = Ed / _pow2(jnp.where(iso, 1.0, dmax))
    p = jnp.sum(Ed * Ed)
    q = _det3(Ed)
    pg = jnp.where(iso, 1.0, p)
    qg = jnp.where(iso, 0.0, q)
    r = jnp.clip(3.0 * _SQRT6 * qg / pg ** 1.5, -1.0, 1.0)
    t = jnp.sqrt(2.0 * pg / 3.0)
    ph = jnp.arccos(r) / 3.0
    c = jnp.where(r >= 0, jnp.cos(ph), jnp.cos(ph + _TWO_PI_3))      # the well separated (simple) eigenvalue
    M = Ed - (t * c) * I
    cands = jnp.stack([jnp.cross(M[0], M[1]), jnp.cross(M[0], M[2]), jnp.cross(M[1], M[2])])
    v = cands[jnp.argmax(jnp.sum(cands * cands, axis=1))]
    v, ok = _normalize(v)
    v = jnp.where(ok & (~iso), v, I[0])
    ek = I[jnp.argmin(jnp.abs(v))]                      # orthonormal complement (u, w) of v
    u, _ = _normalize(ek - jnp.sum(ek * v) * v)
    w = jnp.cross(v, u)
    # eigenvalues from the Rayleigh quotients of A (accurate for spectra over orders of magnitude), the rotation angle
    # and x = lambda - m from those of the deviator (accurate near I)
    b11, b12, b22, bv = u @ Ad @ u, u @ Ad @ w, w @ Ad @ w, v @ Ad @ v
    th = 0.5 * jnp.arctan2(2.0 * b12, b11 - b22)       # 2x2 Jacobi rotation in span(u, w)
    cs, sn = jnp.cos(th), jnp.sin(th)

    def rot(q11, q12, q22, qv):
        return jnp.stack([qv, cs * cs * q11 + 2 * cs * sn * q12 + sn * sn * q22,
                          sn * sn * q11 - 2 * cs * sn * q12 + cs * cs * q22])
    lam = rot(u @ A @ u, u @ A @ w, w @ A @ w, v @ A @ v)
    x = rot(b11, b12, b22, bv)
    return lam * sig, jnp.stack([v, cs * u + sn * w, -sn * u + cs * w], axis=1), m * sig, x / m


# --------------------------------------------------------------------------------------
# Daleckii-Krein formulas in the Voigt layout
# --------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _triples(n):
    """unique index triples a <= m <= b and the map (a, m, b) -> position in that list"""
    U = [(a, m, b) for a in range(n) for m in range(a, n) for b in range(m, n)]
    pos = np.array([[[U.index(tuple(sorted((a, m, b)))) for b in range(n)] for m in range(n)] for a in range(n)])
    return np.array(U).T, pos


def _d2_tensor(kind, alpha, lam):
    """D2[a, m, b] = f[lam_a, lam_m, lam_b] (n, n, n), from the n(n+1)(n+2)/6 distinct triples"""
    (ia, im, ib), pos = _triples(lam.shape[0])
    return _dd2(kind, alpha, lam[ia], lam[im], lam[ib])[pos]


_VP, _VQ = [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]        # Voigt index -> (row, column)
_VI = np.array([[0, 3, 4], [3, 1, 5], [4, 5, 2]])        # (row, column) -> Voigt index


def _voigt_basis():
    """E[p] (6, 3, 3): symmetric basis matrices of the Voigt components"""
    E = np.zeros((6, 3, 3))
    for p, (i, j) in enumerate(zip(_VP, _VQ)):
        E[p, i, j] = E[p, j, i] = 1.0
    return E


def evaluate(kind, alpha, v, order):
    """one Voigt vector v (6,) -> (F,), (F, J) or (F, J, H) for order 0, 1, 2 in the Voigt layout:
    F (6), J (6, 6) = dF_i/dv_k, H (6, 6, 6) = d2F_i/dv_p dv_q.  kind "exp", "log" or "pow" (alpha: Python float)."""
    lam, V, msig, xm = _eig3(v[_VI])
    Vp, Vq = V[np.array(_VP)], V[np.array(_VQ)]               # rows of the Voigt components
    fl = _f(kind, alpha, lam)
    if kind == "log":                                   # near m: log(m sig) + log1p(x/m), no cancellation near I
        near = jnp.abs(xm) <= 0.5
        fl = jnp.where(near, jnp.log(msig) + jnp.log1p(jnp.where(near, xm, 0.0)), fl)
    out = [jnp.einsum("ia,ia,a->i", Vp, Vq, fl)]
    if order >= 1:
        D = _dd(kind, alpha, lam[:, None], lam[None, :])
        Et = jnp.einsum("ka,pkl,lb->pab", V, jnp.asarray(_voigt_basis(), V.dtype), V)   # V^T E_p V
        out.append(jnp.einsum("ia,ib,ab,pab->ip", Vp, Vq, D, Et))
    if order >= 2:
        X = jnp.einsum("amb,pam,qmb->pqab", _d2_tensor(kind, alpha, lam), Et, Et)
        out.append(jnp.einsum("ia,ib,pqab->ipq", Vp, Vq, X + jnp.transpose(X, (1, 0, 2, 3))))
    return tuple(out)
