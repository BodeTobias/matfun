"""
matfun._jaxgen -- gmf in pure JAX: F, J and H of one general (non-symmetric) real 3x3 matrix A (fallback of
matfun.jax.gmf where no compiled kernel runs, e.g. float32 or other backends).  Never differentiated: the derivative
rules of matfun.jax supply J and H themselves.  Real arithmetic only (complex numbers as (re, im) pairs), no bit
casts and no data-dependent control flow, so that code generators translating the jaxpr (e.g. the C element kernels
of autopdex) cover it.  Same method as c/include/matfun/gmf_kernel.hpp (theory.md, "General matrices"):

  * eigenvalues mu = m + x of A (scaled by a power of two sig), x of S = A - m I in closed form (trigonometric /
    Cardano on the traceless, normalized deviator); eigenvalues far from m refined (Newton on det(A - mu I) with
    pivoting; a pair small against the third eigenvalue from det A and the principal minors)
  * Newton-Hermite form over the eigenvalues (no eigenvectors, exact also for defective A):
        F = sum_j f[N_j] w_j,  L(E) = sum_jk f[N_j+N_k] w_j E w_k,  L2 = sum_jkl f[N_j+N_k+N_l] (w_j E1 w_k E2 w_l + ..)
    w_0 = I, w_1 = S - x_1 I, w_2 = (S - x_1 I)(S - x_2 I), N_j = {x_1..x_j+1}; a complex pair through a real basis
  * the 19 confluent divided differences f[x1^a x2^b x3^c] by residues: a group of close nodes (all three, or the
    closest pair) by the Taylor series around its real centre (streamed over the series index, divided by the
    poles of the other node), single nodes by their Taylor coefficients
"""
import math

import numpy as np

import jax
import jax.numpy as jnp

from ._jaxpure import _pow2, _pow_k

_RHO, _R_EXP, _RHO_POLE = 0.7, 4.0, 0.6     # group radius / distance to the singularity (exp: radius), pole ratio
_K = 128                                     # series terms (radius ratio 0.7: 0.7^k C(k+8, 8) < 1e-17 x max for k ~ 112)
_TWO_PI_3 = 2.0 * math.pi / 3.0

# the 19 divided differences f[x1^a x2^b x3^c] of the Newton form (a >= b >= c; a = order + 1), each one node more
# than its parent in this list (-1: none): (multiplicities, parent, node added)
_TREE = [((1, 0, 0), -1, 0), ((1, 1, 0), 0, 1), ((1, 1, 1), 1, 2),
         ((2, 0, 0), 0, 0), ((2, 1, 0), 1, 0), ((2, 1, 1), 2, 0), ((2, 2, 0), 4, 1), ((2, 2, 1), 5, 1), ((2, 2, 2), 7, 2),
         ((3, 0, 0), 3, 0), ((3, 1, 0), 4, 0), ((3, 1, 1), 5, 0), ((3, 2, 0), 6, 0), ((3, 2, 1), 7, 0),
         ((3, 2, 2), 8, 0), ((3, 3, 0), 12, 1), ((3, 3, 1), 13, 1), ((3, 3, 2), 14, 1), ((3, 3, 3), 17, 2)]
_NDD = (3, 9, 19)                            # divided differences needed for order 0, 1, 2
_POS = {ms: i for i, (ms, _, _) in enumerate(_TREE)}


def _mult(idx):
    """multiplicities of x1, x2, x3 in N_j + N_k + ...: (#indices, #indices >= 1, #indices >= 2)"""
    return (len(idx), sum(i >= 1 for i in idx), sum(i >= 2 for i in idx))


# --------------------------------------------------------------------------------------
# complex numbers as (re, im)
# --------------------------------------------------------------------------------------
def _cadd(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _csub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _cmul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def _csum(terms):
    out = terms[0]
    for t in terms[1:]:
        out = _cadd(out, t)
    return out


def _cscale(a, s):
    return (a[0] * s, a[1] * s)


def _cabs(a):
    """|a| without overflow / underflow of the squares"""
    big = jnp.maximum(jnp.abs(a[0]), jnp.abs(a[1]))
    sml = jnp.minimum(jnp.abs(a[0]), jnp.abs(a[1]))
    bs = jnp.where(big > 0, big, 1.0)
    return jnp.where(big > 0, big * jnp.sqrt(1.0 + (sml / bs) ** 2), 0.0)


def _cinv(b):
    """1 / b (Smith: no overflow / underflow of |b|^2)"""
    swap = jnp.abs(b[1]) > jnp.abs(b[0])
    p, q = jnp.where(swap, b[1], b[0]), jnp.where(swap, b[0], b[1])     # |p| >= |q|
    ps = jnp.where(p != 0, p, 1.0)
    r = q / ps
    den = p + q * r
    ds = jnp.where(den != 0, den, 1.0)
    re, im = 1.0 / ds, -r / ds
    return jnp.where(swap, -im, re) * jnp.where(den != 0, 1.0, jnp.inf), jnp.where(swap, -re, im)


def _cwhere(m, a, b):
    return (jnp.where(m, a[0], b[0]), jnp.where(m, a[1], b[1]))


def _copysign(x, y):
    return jnp.where(y < 0, -jnp.abs(x), jnp.abs(x))


# --------------------------------------------------------------------------------------
# eigenvalues
# --------------------------------------------------------------------------------------
def _det3(M):
    return (M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]) - M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0])
            + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0]))


def _det_lu(M):
    """det M by elimination with partial pivoting: backward stable, exact pivots for diagonal M"""
    a = jnp.abs(M[:, 0])
    p1 = (a[1] > a[0]) & (a[1] >= a[2])
    p2 = (a[2] > a[0]) & (a[2] > a[1])
    P = jnp.where(p1, M[1], jnp.where(p2, M[2], M[0]))                 # pivot row; (2, 0, 1) is an even permutation
    R1 = jnp.where(p1 | p2, M[0], M[1])
    R2 = jnp.where(p2, M[1], M[2])
    sgn = jnp.where(p1, -1.0, 1.0)
    piv = jnp.where(P[0] != 0, P[0], 1.0)
    l1, l2 = R1[0] / piv, R2[0] / piv
    b, c = R1[1] - l1 * P[1], R1[2] - l1 * P[2]
    d, e = R2[1] - l2 * P[1], R2[2] - l2 * P[2]
    sw = jnp.abs(d) > jnp.abs(b)
    b, c, d, e = jnp.where(sw, d, b), jnp.where(sw, e, c), jnp.where(sw, b, d), jnp.where(sw, c, e)
    sgn = jnp.where(sw, -sgn, sgn)
    bs = jnp.where(b != 0, b, 1.0)
    det = sgn * P[0] * b * (e - d / bs * c)
    return jnp.where((P[0] != 0) & (b != 0), det, 0.0)


def _minors2(M):
    """sum of the principal 2x2 minors"""
    return (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0] + M[0, 0] * M[2, 2] - M[0, 2] * M[2, 0]
            + M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1])


def _roots(S):
    """eigenvalues of S: tau + those of the traceless S - tau I (tau = tr S / 3, the rounding residue of the shift),
    normalized twice (by max |S| and by the size of the roots), from x^3 - P x - Q = 0: x[0] the real root farthest
    from the others (trigonometric formula) or the only real one (Cardano), x[1], x[2] from the deflated quadratic
    (a real pair or a conjugate pair).  [(re, im)] * 3"""
    I = jnp.eye(3, dtype=S.dtype)
    tau = (S[0, 0] + S[1, 1] + S[2, 2]) / 3.0
    S = S - tau * I
    t = _pow2(jnp.max(jnp.abs(S)))
    Sn = S / t
    P = 0.5 * (Sn[0, 0] ** 2 + Sn[1, 1] ** 2 + Sn[2, 2] ** 2) + Sn[0, 1] * Sn[1, 0] + Sn[0, 2] * Sn[2, 0] + Sn[1, 2] * Sn[2, 1]
    Q = _det3(Sn)
    u = _pow2(jnp.maximum(jnp.sqrt(jnp.abs(P)), jnp.cbrt(jnp.abs(Q))))
    P, Q, t = P / u / u, Q / u / u / u, t * u
    trig = (P > 0) & (4.0 * P ** 3 >= 27.0 * Q * Q)
    Ps = jnp.where(P > 0, P, 1.0)
    r = jnp.clip(0.5 * Q / (Ps / 3.0) ** 1.5, -1.0, 1.0)
    ph = jnp.arccos(r) / 3.0
    s1t = 2.0 * jnp.sqrt(Ps / 3.0) * jnp.cos(jnp.where(r >= 0, ph, ph + _TWO_PI_3))
    uc = jnp.cbrt(0.5 * Q + _copysign(jnp.sqrt(jnp.maximum(0.25 * Q * Q - P ** 3 / 27.0, 0.0)), Q))
    s1c = uc + jnp.where(uc != 0, P / (3.0 * jnp.where(uc != 0, uc, 1.0)), 0.0)
    s1 = jnp.where(trig, s1t, s1c)
    d = P - 0.75 * s1 * s1                                  # x^2 + s1 x + (s1^2 - P) = 0
    real = d >= 0
    sq = jnp.sqrt(jnp.abs(d))
    sa = -0.5 * s1 - _copysign(sq, s1)
    prod = jnp.where(s1 != 0, Q / jnp.where(s1 != 0, s1, 1.0), -P)
    sb = jnp.where(sa != 0, prod / jnp.where(sa != 0, sa, 1.0), 0.0)
    im = jnp.where(real, 0.0, sq) * t
    return [(s1 * t + tau, 0.0 * t), (jnp.where(real, sa, -0.5 * s1) * t + tau, im),
            (jnp.where(real, sb, -0.5 * s1) * t + tau, -im)]


def _newton(As, mu, j, mask):
    """real node j: 3 Newton steps on det(A - mu I), derivative from the other nodes (where mask)"""
    I = jnp.eye(3, dtype=As.dtype)
    for _ in range(3):
        den = (1.0, 0.0)
        for k in range(3):
            if k != j:
                den = _cmul(den, _csub(mu[k], mu[j]))
        n2 = den[0] * den[0] + den[1] * den[1]
        ok = mask & (n2 > 0)
        # Re(det / den) = det Re(den) / |den|^2, formed without |den|^2 under/overflow
        inv = _cinv(den)
        step = _det_lu(As - mu[j][0] * I) * inv[0]
        mu[j] = (jnp.where(ok, mu[j][0] + step, mu[j][0]), mu[j][1])
    return mu


# --------------------------------------------------------------------------------------
# the divided differences
# --------------------------------------------------------------------------------------
def _coef0(kind, alpha, sig, m, x, mu, k):
    """f(mu) (k: the output order, pow: times sig^(alpha - k)), f(t) = exp(t) (x: relative to the shift),
    log(sig t) (log1p of x/m near m), (sig t)^alpha"""
    if kind == "exp":
        e = jnp.exp(x[0])
        return (e * jnp.cos(x[1]), e * jnp.sin(x[1]))
    th = jnp.arctan2(mu[1], mu[0])
    if kind == "log":
        near = (m > 0) & (_cabs(x) <= 0.5 * m)
        ms = jnp.where(m > 0, m, 1.0)
        zr, zi = x[0] / ms, x[1] / ms
        l1p = (0.5 * jnp.log1p(zr * (2.0 + zr) + zi * zi), jnp.arctan2(zi, 1.0 + zr))
        return (jnp.where(near, jnp.log(sig * ms) + l1p[0], jnp.log(sig * _cabs(mu))), jnp.where(near, l1p[1], th))
    lam = sig * _cabs(mu)
    mag = _pow_k(lam, alpha, k)
    ph = (alpha - k) * th
    out = (mag * jnp.cos(ph), mag * jnp.sin(ph))
    for _ in range(k):
        out = _cmul(out, mu)
    return out


def _series_coefs(kind, alpha, K):
    """b_j, j < K, of f(c (1 + u)) / scale: exp e^u: 1/j!; log: (-1)^(j+1)/j (b_0 apart); pow: binom(alpha, j)"""
    b = np.zeros(K)
    if kind == "exp":
        b[0] = 1.0
        for j in range(1, K):
            b[j] = b[j - 1] / j
    elif kind == "log":
        for j in range(1, K):
            b[j] = (1.0 if j % 2 else -1.0) / j
    else:
        b[0] = 1.0
        for j in range(1, K):
            b[j] = b[j - 1] * (alpha - j + 1) / j
    return b


def _divided_differences(kind, alpha, nd, sig, m, x, mu, isabs, inG, grp, xshift):
    """the first nd entries of _TREE: group part (Taylor series of the group, streamed over the series index) plus
    the residues of the single nodes; [(re, im)] * nd.  grp = (cx, cmu, offsets, pk, a): group centre relative to m
    and absolute, offsets of the nodes in the group, one-hot outside node (pair) and its pole distance"""
    cx, cmu, off, pk, apole = grp
    hasg = inG[0] | inG[1] | inG[2]
    expk = kind == "exp"
    L = jnp.where(hasg & (cmu > 0), cmu, 1.0) if not expk else 1.0     # normalization of the series variable
    u = [_cscale(o, 1.0 / L) for o in off]
    ap = jnp.where(jnp.abs(apole) > 0, apole / L, 1.0)
    # coefficient b_0 of the group (log: lane dependent), the rest from the table
    cxs = cx - xshift if expk else cx
    b0 = _coef0(kind, alpha, sig, m, (cxs, 0.0 * cx), (cmu, 0.0 * cx), 0)[0] if kind == "log" else 1.0 + 0.0 * cx
    btab = jnp.asarray(_series_coefs(kind, alpha, _K), dtype=m.dtype)
    first = jnp.arange(_K) == 0
    zero = (0.0 * m, 0.0 * m)

    # dual vectors e^(ms)_j of the group nodes, at j = -1: root 1, others the parent unless their node is in the group
    E0 = []
    for ms, par, i in _TREE[:nd]:
        ep = (1.0 + 0.0 * m, 0.0 * m) if par < 0 else E0[par]
        E0.append(_cwhere(inG[i], zero, ep))
    q0 = (0.0 * m, 0.0 * m, 0.0 * m)                         # G^(r)_(j-1), r = 1..3

    def step(carry, xs):
        Eprev, qprev, acc = carry
        bt, fst = xs
        bj = jnp.where(fst, b0, bt)
        q = [bj]
        for r in range(3):                                   # series of f / (a + u)^(r+1)
            q.append((q[r] - qprev[r]) / ap)
        Ecur, accn = [], []
        for c, (ms, par, i) in enumerate(_TREE[:nd]):
            epp = (1.0 + 0.0 * m, 0.0 * m) if par < 0 else Eprev[par]     # parent at j-1 (root: 1 at j = -1 only)
            if par < 0:
                epp = _cwhere(fst, epp, zero)
            epc = zero if par < 0 else Ecur[par]                            # parent at j (root: 0 for j >= 0)
            add = _cadd(epp, _cmul(u[i], Eprev[c]))
            Ecur.append(_cwhere(inG[i], add, epc))
            g = jnp.where(pk[0], q[ms[0]], jnp.where(pk[1], q[ms[1]], jnp.where(pk[2], q[ms[2]], q[0])))
            accn.append(_cadd(acc[c], _cscale(Ecur[c], g)))
        return (Ecur, tuple(q[1:]), accn), None

    acc0 = [zero] * nd
    (_, _, acc), _ = jax.lax.scan(step, (E0, q0, acc0), (btab, first))
    out = []
    for c, (ms, par, i) in enumerate(_TREE[:nd]):
        n = sum(ms) - 1
        if expk:
            fac = jnp.exp(cx - xshift)
        elif kind == "log":
            fac = L ** (-n)
        else:
            k = ms[0] - 1
            fac = _pow_k(sig * L, alpha, k) * L ** (k - n)
        out.append(_cwhere(hasg, _cscale(acc[c], fac), zero))

    # residues of the single nodes i: coefficient of (t - mu_i)^(m_i - 1) in f(t) / prod_(z != i) (t - mu_z)^m_z
    for i in range(3):
        single = ~inG[i]
        xs_i = (x[i][0] - xshift, x[i][1]) if expk else x[i]
        f0 = [_coef0(kind, alpha, sig, m, xs_i, mu[i], k) for k in (range(3) if kind != "exp" and kind != "log" else range(1))]
        imu = _cinv(mu[i])
        winv = {}
        for z in range(3):
            if z != i:
                dz = _cwhere(isabs[i] | isabs[z], _csub(mu[i], mu[z]), _csub(x[i], x[z]))
                winv[z] = _cinv(dz)
        for c, (ms, par, _) in enumerate(_TREE[:nd]):
            mi = ms[i]
            if mi == 0:
                continue
            k = ms[0] - 1
            fb = f0[k] if len(f0) > 1 else f0[0]
            if kind == "exp":
                fs = [fb, fb, _cscale(fb, 0.5)]
            elif kind == "log":
                fs = [fb, imu, _cscale(_cmul(imu, imu), -0.5)]
            else:
                f1 = _cscale(_cmul(fb, imu), alpha)
                fs = [fb, f1, _cscale(_cmul(f1, imu), (alpha - 1.0) / 2.0)]
            ser = fs[:mi]                                    # times the series of 1 / (d + tau)^m per other node
            for z, w in winv.items():
                mz = ms[z]
                if mz == 0:
                    continue
                wp = (1.0 + 0.0 * m, 0.0 * m)
                for _ in range(mz):
                    wp = _cmul(wp, w)
                ps = [wp, _cscale(_cmul(wp, w), -mz), _cscale(_cmul(_cmul(wp, w), w), mz * (mz + 1) / 2.0)]
                ser = [_csum([_cmul(ser[a], ps[b - a]) for a in range(b + 1)]) for b in range(len(ser))]
            res = ser[mi - 1]
            out[c] = _cwhere(single, _cadd(out[c], res), out[c])
    return out


# --------------------------------------------------------------------------------------
# F, J, H
# --------------------------------------------------------------------------------------
def evaluate(kind, alpha, A, order):
    """one matrix A (3, 3) -> (F,), (F, J) or (F, J, H): F (3, 3), J (3, 3, 3, 3) = dF_ab/dA_cd,
    H (3, 3, 3, 3, 3, 3) = d2F_ab/dA_cd dA_ef.  kind "exp", "log" or "pow" (alpha: Python float)."""
    dt = A.dtype
    I = jnp.eye(3, dtype=dt)
    sig = 1.0 + 0.0 * A[0, 0] if kind == "exp" else _pow2(jnp.max(jnp.abs(A)))
    As = A / sig
    m = (As[0, 0] + As[1, 1] + As[2, 2]) / 3.0
    S = As - m * I
    x = _roots(S)
    mu = [(m + v[0], v[1]) for v in x]
    isabs = [jnp.zeros((), bool)] * 3

    # ---- the pair x[1], x[2] from its sum and product with A itself, after mu[0] by Newton, where the roots of S are
    #      not the more accurate ones (their pair loses eps |S|^2 / gap, the shift by m eps m): the pair small against
    #      mu[0], or the deviator not dominated by the shift and mu[0] separated from the pair
    mid = 0.5 * (mu[1][0] + mu[2][0])
    Rp = 0.5 * _cabs(_csub(mu[1], mu[2]))
    small = jnp.abs(mid) + Rp <= 0.25 * jnp.abs(mu[0][0])
    aniso = (jnp.abs(m) <= jnp.max(jnp.abs(S))) & (Rp <= 0.25 * jnp.abs(mid - mu[0][0]))
    pc = (mu[0][0] != 0) & (small | aniso)
    mu = _newton(As, list(mu), 0, pc)
    m0 = jnp.where(pc, mu[0][0], 1.0)
    big = jnp.abs(m0) >= jnp.abs(mid) + Rp                  # pi = det A / mu0, sigma = (c2 - pi) / mu0
    pi_d = _det_lu(As) / m0
    sg_t = As[0, 0] + As[1, 1] + As[2, 2] - m0              # else sigma = tr A - mu0, pi = c2 - mu0 sigma
    sg = jnp.where(big, (_minors2(As) - pi_d) / m0, sg_t)
    pi = jnp.where(big, pi_d, _minors2(As) - m0 * sg_t)
    pcen, disc = 0.5 * sg, 0.25 * sg * sg - pi
    ps = jnp.sqrt(jnp.abs(disc))
    preal = disc >= 0
    r1 = pcen + _copysign(ps, pcen)
    r2 = jnp.where(r1 != 0, pi / jnp.where(r1 != 0, r1, 1.0), pcen - _copysign(ps, pcen))
    mu[1] = _cwhere(pc, (jnp.where(preal, r1, pcen), jnp.where(preal, 0.0, ps)), mu[1])
    mu[2] = _cwhere(pc, (jnp.where(preal, r2, pcen), jnp.where(preal, 0.0, -ps)), mu[2])
    isabs = [pc, pc, pc]
    x = [_cwhere(isabs[i], (mu[i][0] - m, mu[i][1]), x[i]) for i in range(3)]
    orig = [0.0 * m, 1.0 + 0.0 * m, 2.0 + 0.0 * m]

    # ---- node order: the node with the largest high derivatives last (exp: largest real part; log, pow: smallest
    #      modulus); a conjugate pair stays adjacent (real node first or last)
    def key(v):
        return v[0] if kind == "exp" else -_cabs(v)

    nodes = [dict(x=x[i], mu=mu[i], isabs=isabs[i], orig=orig[i]) for i in range(3)]

    def pick(cond, a, b):
        return {k: (_cwhere(cond, a[k], b[k]) if isinstance(a[k], tuple) else jnp.where(cond, a[k], b[k])) for k in a}

    srt = list(nodes)
    for i, j in ((0, 1), (1, 2), (0, 1)):                    # stable sorting network (real triple)
        sw = key(srt[i]["mu"]) > key(srt[j]["mu"])
        srt[i], srt[j] = pick(sw, srt[j], srt[i]), pick(sw, srt[i], srt[j])
    first = key(nodes[0]["mu"]) <= key(nodes[1]["mu"])        # complex pair: real node first or last
    cpl = [pick(first, nodes[k], nodes[(k + 1) % 3]) for k in range(3)]
    cplx = nodes[1]["x"][1] != 0
    nodes = [pick(cplx, cpl[k], srt[k]) for k in range(3)]
    x = [n_["x"] for n_ in nodes]
    mu = [n_["mu"] for n_ in nodes]
    isabs = [n_["isabs"] for n_ in nodes]
    orig = [n_["orig"] for n_ in nodes]

    # ---- groups: all three, else the closest real / conjugate pair (with the pole of the third node), else none
    rho_f = min(_RHO, 3.0 / alpha) if kind == "pow" else _RHO

    def ok_f(c, R):
        return (R <= _R_EXP) if kind == "exp" else ((m + c > 0) & (R <= rho_f * (m + c)))

    cs = (x[0][0] + x[1][0] + x[2][0]) / 3.0
    R3 = jnp.maximum(jnp.maximum(_cabs((x[0][0] - cs, x[0][1])), _cabs((x[1][0] - cs, x[1][1]))),
                     _cabs((x[2][0] - cs, x[2][1])))
    triple = ok_f(cs, R3)
    best, bestR = None, None
    pairsel = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        k = 3 - i - j
        elig = ((x[i][1] == 0) & (x[j][1] == 0)) | ((x[i][0] == x[j][0]) & (x[i][1] == -x[j][1]))
        c = 0.5 * (x[i][0] + x[j][0])
        Rp_ = 0.5 * _cabs(_csub(x[i], x[j]))
        okp = elig & ok_f(c, Rp_) & (Rp_ <= _RHO_POLE * _cabs((c - x[k][0], -x[k][1]))) & ~triple
        better = okp if bestR is None else okp & (Rp_ < bestR)
        bestR = jnp.where(better, Rp_, jnp.inf if bestR is None else bestR)
        pairsel = [s & ~better for s in pairsel] + [better]
    p01, p02, p12 = pairsel
    inG = [triple | p01 | p02, triple | p01 | p12, triple | p02 | p12]
    pk = [p12, p02, p01]                                     # outside node of the pair

    # ---- the refined pair as a group: symmetric offsets around its centre
    off = [None] * 3
    gsum = (inG[0] * 1.0 + inG[1] * 1.0 + inG[2] * 1.0)
    cx = jnp.where(gsum > 0, ((x[0][0] * inG[0]) + (x[1][0] * inG[1]) + (x[2][0] * inG[2])) / jnp.maximum(gsum, 1.0), 0.0)
    cmu = m + cx
    is12 = [(orig[i] == 1) | (orig[i] == 2) for i in range(3)]
    ovr = pc & (gsum == 2) & ((inG[0] & inG[1] & is12[0] & is12[1]) | (inG[0] & inG[2] & is12[0] & is12[2])
                              | (inG[1] & inG[2] & is12[1] & is12[2]))
    cx = jnp.where(ovr, pcen - m, cx)
    cmu = jnp.where(ovr, pcen, cmu)
    dpair = (jnp.where(preal, ps, 0.0), jnp.where(preal, 0.0, ps))
    for i in range(3):
        neg = (mu[i][0] - pcen < 0) | (mu[i][1] < 0)
        d_i = _cwhere(neg, (-dpair[0], -dpair[1]), dpair)
        ov_i = ovr & inG[i]
        mu[i] = _cwhere(ov_i, (pcen + d_i[0], d_i[1]), mu[i])
        x[i] = _cwhere(ov_i, (mu[i][0] - m, mu[i][1]), x[i])
        off[i] = _cwhere(inG[i], _cwhere(ov_i, d_i, (x[i][0] - cx, x[i][1])), (0.0 * m, 0.0 * m))

    # ---- single real nodes far from m: Newton on det(A - mu I)
    for i in range(3):
        mask = ~inG[i] & ~isabs[i] & (x[i][1] == 0) & (jnp.abs(x[i][0]) > 0.5 * jnp.abs(m))
        mu = _newton(As, mu, i, mask)
        isabs[i] = isabs[i] | mask
        x[i] = _cwhere(mask, (mu[i][0] - m, mu[i][1]), x[i])

    # pole distance of the outside node of a pair group (normalized in _divided_differences)
    absc = (inG[0] & isabs[0]) | (inG[1] & isabs[1]) | (inG[2] & isabs[2])
    apole = 0.0 * m
    for k in range(3):
        a_k = jnp.where(absc | isabs[k], cmu - mu[k][0], cx - x[k][0])
        apole = jnp.where(pk[k], a_k, apole)
    xshift = jnp.maximum(jnp.maximum(x[0][0], x[1][0]), x[2][0]) if kind == "exp" else 0.0 * m
    bad = jnp.zeros((), bool) if kind == "exp" else (((x[0][1] == 0) & (mu[0][0] <= 0)) | ((x[1][1] == 0) & (mu[1][0] <= 0))
                                                    | ((x[2][1] == 0) & (mu[2][0] <= 0)))

    nd = _NDD[order]
    D = _divided_differences(kind, alpha, nd, sig, m, x, mu, isabs, inG, (cx, cmu, off, pk, apole), xshift)

    # ---- real basis: w = T r, r_0 = I, r_1 = B_0, r_2 = B_0 B_1 - b0 b1 I, B_j = S - Re x_j I (or A - Re mu_j I),
    #      T = I + i N (N strictly lower), D' = Re(T^T D T ...)
    b0, b1 = x[0][1], x[1][1]
    B = [jnp.where(isabs[j], As - mu[j][0] * I, S - x[j][0] * I) for j in range(2)]
    a01 = jnp.where(isabs[0] | isabs[1], mu[0][0] - mu[1][0], x[0][0] - x[1][0])
    r = jnp.stack([I, B[0], B[0] @ B[1] - b0 * b1 * I])
    N = jnp.zeros((3, 3), dt).at[1, 0].set(-b0).at[2, 0].set(-b0 * a01).at[2, 1].set(-(b0 + b1))

    def real_basis(Dr, Di, ndim):
        """Re of D contracted with T = I + i N on each of its ndim indices"""
        for ax in range(ndim):
            Nr = jnp.moveaxis(jnp.tensordot(N, jnp.moveaxis(Dr, ax, 0), axes=(0, 0)), 0, ax)
            Ni = jnp.moveaxis(jnp.tensordot(N, jnp.moveaxis(Di, ax, 0), axes=(0, 0)), 0, ax)
            Dr, Di = Dr - Ni, Di + Nr
        return Dr

    def gather(idxs):
        re = jnp.stack([D[_POS[_mult(ix)]][0] for ix in idxs])
        im = jnp.stack([D[_POS[_mult(ix)]][1] for ix in idxs])
        return re, im

    nanf = jnp.where(bad, jnp.nan, 1.0)
    re, im = gather([(j,) for j in range(3)])
    D1 = real_basis(re, im, 1)
    F = jnp.einsum("m,mab->ab", D1, r)
    out = [F]
    if order >= 1:
        idx = [(j, k) for j in range(3) for k in range(3)]
        re, im = gather(idx)
        D2 = real_basis(re.reshape(3, 3), im.reshape(3, 3), 2)
        out.append(jnp.einsum("mn,mac,ndb->abcd", D2, r, r))
    if order >= 2:
        idx = [(j, k, l) for j in range(3) for k in range(3) for l in range(3)]
        re, im = gather(idx)
        D3 = real_basis(re.reshape(3, 3, 3), im.reshape(3, 3, 3), 3)
        Tt = jnp.einsum("mnp,mac,nde,pfb->abcdef", D3, r, r, r)
        out.append(Tt + jnp.transpose(Tt, (0, 1, 4, 5, 2, 3)))
    if kind == "exp":
        e = jnp.exp(m + xshift)
        out = [o * e for o in out]
    elif kind == "log":                                      # pow: sig^(alpha - k) is in the coefficients
        out = [o if k == 0 else o / sig ** k for k, o in enumerate(out)]
    return tuple(o * nanf for o in out)
