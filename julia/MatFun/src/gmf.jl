# gmf.jl -- general (non-symmetric) real 3x3 matrices: f(A) for f = exp, log, sqrt, x^alpha (principal branch) with
# the full first and second derivative. Port of gmf::eval<KIND, ORDER, double> in c/include/matfun/gmf_kernel.hpp
# (same steps, thresholds and case distinctions; the selects of the branch-free C++ kernel are plain conditionals
# here). Internally row-major flat indices k = 3 (row - 1) + (column - 1) as in C; method: theory.md.

const G_RHO = 0.7
const G_R_EXP = 4.0
const G_RHO_POLE = 0.6
const G_KMAX = 200
const G_TMS = ((1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 0, 0), (2, 1, 0), (2, 1, 1), (2, 2, 0), (2, 2, 1), (2, 2, 2),
               (3, 0, 0), (3, 1, 0), (3, 1, 1), (3, 2, 0), (3, 2, 1), (3, 2, 2), (3, 3, 0), (3, 3, 1), (3, 3, 2),
               (3, 3, 3))
const G_TPAR = (0, 1, 2, 1, 2, 3, 5, 6, 8, 4, 5, 6, 7, 8, 9, 13, 14, 15, 18)       # 1-based, 0: none
const G_TNODE = (1, 2, 3, 1, 1, 1, 2, 2, 3, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3)
const G_NDD = (3, 9, 19)

# series coefficients b_j (j = 0..KMAX-1 at index j+1) of f(c (1 + u)) / scale: exp 1/j!; log (-1)^(j+1)/j
# (b_0 apart); pow binom(al, j)
function _gseries(k::Kind, al::Float64)
    b = Vector{Float64}(undef, G_KMAX)
    b[1] = 1.0
    for j in 1:G_KMAX-1
        b[j+1] = k == SMF_EXP ? b[j] / j : (k == SMF_LOG ? (isodd(j) ? 1.0 : -1.0) / j : b[j] * (al - j + 1) / j)
    end
    return b
end

# terms of the group series (bound C(k+n, n) r^k |b_(n+k)|) and of the pole series (C(k+2, 2) q^k)
function _gnterms(b, n::Int, r::Float64, q::Float64)
    M = 1
    if r > 0
        best, c, rk = 0.0, 1.0, 1.0
        k = 0
        while n + k < G_KMAX
            t = abs(b[n+k+1]) * c * rk
            best = t > best ? t : best
            M = k
            ((k > 2 && t <= 1e-17 * best) || !(rk > 0)) && break
            c = c * (k + n + 1) / (k + 1)
            rk *= r
            k += 1
        end
    end
    if q > 0
        qk = 1.0
        k = 0
        while n + k < G_KMAX
            (k > M && (k + 1.0) * (k + 2.0) / 2.0 * qk <= 1e-17) && break
            M = max(k, M)
            qk *= q
            k += 1
        end
    end
    return min(n + M + 1, G_KMAX)
end

@inline function _cabs(re, im)
    ar, ai = abs(re), abs(im)
    big, sml = ar > ai ? ar : ai, ar < ai ? ar : ai
    big > 0 || return big == 0 ? zero(big) : big
    q = sml / big
    return big * sqrt(1 + q * q)
end

@inline function _cinv(re, im)                    # 1 / (re + i im) (Smith)
    sw = abs(im) > abs(re)
    p, q = sw ? im : re, sw ? re : im
    r = q / (p != 0 ? p : one(p))
    den = p + q * r
    a, b = 1 / den, -r / den
    return (sw ? -b : a, sw ? -a : b)
end

@inline _cmul(ar, ai, br, bi) = (ar * br - ai * bi, ar * bi + ai * br)
@inline _gpow2(x) = (x > 1e-300 && x < 1e300) ? _pow2floor(x) : one(x)
@inline _copysgn(x, y) = y < 0 ? -abs(x) : abs(x)
@inline _gdet3(M) = M[1] * (M[5] * M[9] - M[6] * M[8]) - M[2] * (M[4] * M[9] - M[6] * M[7]) +
                    M[3] * (M[4] * M[8] - M[5] * M[7])
@inline _gminors2(M) = M[1] * M[5] - M[2] * M[4] + M[1] * M[9] - M[3] * M[7] + M[5] * M[9] - M[6] * M[8]

# det(M - s I) by elimination with partial pivoting (backward stable, exact pivots for diagonal M)
function _gdetlu(M, s)
    r0 = (M[1] - s, M[2], M[3])
    r1 = (M[4], M[5] - s, M[6])
    r2 = (M[7], M[8], M[9] - s)
    a0, a1, a2 = abs(r0[1]), abs(r1[1]), abs(r2[1])
    p1 = a1 > a0 && a1 >= a2
    p2 = a2 > a0 && a2 > a1
    P = p1 ? r1 : (p2 ? r2 : r0)
    R1 = (p1 || p2) ? r0 : r1
    R2 = p2 ? r1 : r2
    sgn = p1 ? -1.0 : 1.0
    piv = P[1] != 0 ? P[1] : one(P[1])
    l1, l2 = R1[1] / piv, R2[1] / piv
    b, c = R1[2] - l1 * P[2], R1[3] - l1 * P[3]
    d, e = R2[2] - l2 * P[2], R2[3] - l2 * P[3]
    sw = abs(d) > abs(b)
    bb, cc, dd, ee = sw ? d : b, sw ? e : c, sw ? b : d, sw ? c : e
    sw && (sgn = -sgn)
    (P[1] == 0 || bb == 0) && return zero(b)
    return sgn * P[1] * bb * (ee - dd / bb * cc)
end

# scratch arrays of one evaluation (allocated once per gmf! call and reused for all matrices of a chunk); flat,
# 0-based layouts as in the C++ kernel: matrices row-major (k = 3 row + column), r[9 m + k], D2[3 j + k],
# D3[9 j + 3 k + l], Z[9 (3 m + n) + k], Yh[81 m + 9 de + fb]
struct _GWork{W}
    a::Vector{W}; As::Vector{W}; S::Vector{W}; Sp::Vector{W}
    xr::Vector{W}; xi::Vector{W}; mr::Vector{W}; mi::Vector{W}; ab::Vector{Bool}
    offr::Vector{W}; offi::Vector{W}; ur::Vector{W}; ui::Vector{W}
    Dr::Vector{W}; Di::Vector{W}; Epr::Vector{W}; Epi::Vector{W}; Ecr::Vector{W}; Eci::Vector{W}
    accr::Vector{W}; acci::Vector{W}
    B0::Vector{W}; B1::Vector{W}; r::Vector{W}
    E2r::Vector{W}; E2i::Vector{W}; D2::Vector{W}; Y::Vector{W}
    E3r::Vector{W}; E3i::Vector{W}; Z::Vector{W}; Yh::Vector{W}
end
function _GWork{W}() where {W}
    z(n) = zeros(W, n)
    return _GWork{W}(z(9), z(9), z(9), z(9), z(3), z(3), z(3), z(3), zeros(Bool, 3), z(3), z(3), z(3), z(3),
                     z(19), z(19), z(19), z(19), z(19), z(19), z(19), z(19), z(9), z(9), z(27),
                     z(9), z(9), z(9), z(27), z(27), z(27), z(81), z(243))
end

# eigenvalues of w.S (flat, row-major) into w.xr, w.xi: x[1] real, x[2], x[3] a real or a conjugate pair
@inline function _groots!(w::_GWork{W}) where {W}
    S, Sp = w.S, w.Sp
    @inbounds begin
        tau = (S[1] + S[5] + S[9]) / 3
        s = zero(W)
        for k in 1:9
            Sp[k] = k % 4 == 1 ? S[k] - tau : S[k]
            s = max(s, abs(Sp[k]))
        end
        t = _gpow2(s)
        it = 1 / t
        for k in 1:9
            Sp[k] *= it
        end
        P = (Sp[1] * Sp[1] + Sp[5] * Sp[5] + Sp[9] * Sp[9]) / 2 + Sp[2] * Sp[4] + Sp[3] * Sp[7] + Sp[6] * Sp[8]
        Q = _gdet3(Sp)
        u = _gpow2(max(sqrt(abs(P)), cbrt(abs(Q))))
        iu = 1 / u
        P = P * iu * iu
        Q = Q * iu * iu * iu
        t = t * u
        if P > 0 && 4 * P * P * P >= 27 * Q * Q
            p3 = P / 3
            r = Q / 2 / (p3 * sqrt(p3))
            r = r > 1 ? one(r) : (r < -1 ? -one(r) : r)
            ph = acos(r) / 3
            s1 = 2 * sqrt(p3) * cos(r >= 0 ? ph : ph + TWO_PI_3)
        else
            D = Q * Q / 4 - P * P * P / 27
            uc = cbrt(Q / 2 + _copysgn(sqrt(D > 0 ? D : zero(D)), Q))
            s1 = uc + (uc != 0 ? P / (3 * uc) : zero(uc))
        end
        d = P - 0.75 * s1 * s1
        real = d >= 0
        sq = sqrt(abs(d))
        sa = -s1 / 2 - _copysgn(sq, s1)
        prod = s1 != 0 ? Q / s1 : -P
        sb = sa != 0 ? prod / sa : zero(sa)
        im = real ? zero(sq) : sq * t
        w.xr[1], w.xr[2], w.xr[3] = s1 * t + tau, (real ? sa : -s1 / 2) * t + tau, (real ? sb : -s1 / 2) * t + tau
        w.xi[1], w.xi[2], w.xi[3] = zero(im), im, -im
    end
    return nothing
end

# real node j: 3 Newton steps on det(A - mu I), derivative from the other nodes
@inline function _gnewton!(As, mr, mi, j)
    @inbounds for _ in 1:3
        dr, di = one(mr[1]), zero(mr[1])
        for k in 1:3
            k == j && continue
            dr, di = _cmul(dr, di, mr[k] - mr[j], mi[k] - mi[j])
        end
        (dr == 0 && di == 0) && continue
        ir, _ = _cinv(dr, di)
        mr[j] = mr[j] + _gdetlu(As, mr[j]) * ir
    end
    return nothing
end

# T = I + i N (N strictly lower: N10, N20, N21) applied to the index of stride s of the 3-vectors in (Er, Ei)
# starting at base (0-based)
@inline function _gapplyT!(Er, Ei, base, s, N10, N20, N21)
    @inbounds begin
        i0, i1, i2 = base + 1, base + s + 1, base + 2s + 1
        vr0, vr1, vr2, vi0, vi1, vi2 = Er[i0], Er[i1], Er[i2], Ei[i0], Ei[i1], Ei[i2]
        Er[i0] = vr0 - (N10 * vi1 + N20 * vi2); Er[i1] = vr1 - N21 * vi2
        Ei[i0] = vi0 + (N10 * vr1 + N20 * vr2); Ei[i1] = vi1 + N21 * vr2
    end
    return nothing
end

# f(mu) (k: order of the output; pow: times sig^(al - k)); exp: e^x (x relative to the shift)
function _gcoef0(k::Kind, al, sig, m, xr, xi, mur, mui, ko)
    if k == SMF_EXP
        e = exp(xr)
        return (e * cos(xi), e * sin(xi))
    end
    th = atan(mui, mur)
    if k == SMF_LOG
        if m > 0 && _cabs(xr, xi) <= m / 2
            zr, zi = xr / m, xi / m
            return (_log(sig * m) + _log1p(zr * (2 + zr) + zi * zi) / 2, atan(zi, 1 + zr))
        end
        return (_log(sig * _cabs(mur, mui)), th)
    end
    mag = _pow_k(sig * _cabs(mur, mui), al, Float64(ko))
    ph = (al - ko) * th
    re, im = mag * cos(ph), mag * sin(ph)
    for _ in 1:ko
        re, im = _cmul(re, im, mur, mui)
    end
    return (re, im)
end

_gkey(k::Kind, re, im) = k == SMF_EXP ? re : -_cabs(re, im)
_gok(k::Kind, m, rhoF, c, R) = k == SMF_EXP ? R <= G_R_EXP : (m + c > 0 && R <= rhoF * (m + c))
_gtpos(a, b, c) = a == 1 ? (b + c) : (a == 2 ? (b == 0 ? 3 : (b == 1 ? 4 + c : 6 + c)) :
                  (b == 0 ? 9 : (b == 1 ? 10 + c : (b == 2 ? 12 + c : 15 + c))))
_gpos1(j) = _gtpos(1, Int(j >= 1), Int(j >= 2)) + 1                       # 1-based positions, j = 0..2
_gpos2(j, k) = _gtpos(2, Int(j >= 1) + Int(k >= 1), Int(j >= 2) + Int(k >= 2)) + 1
_gpos3(j, k, l) = _gtpos(3, Int(j >= 1) + Int(k >= 1) + Int(l >= 1), Int(j >= 2) + Int(k >= 2) + Int(l >= 2)) + 1

# the kernel for the matrix w.a (flat row-major); writes F[a,b] (3x3), J[a,b,c,d], H[a,b,c,d,e,f] if given
function _geval!(w::_GWork{W}, k::Kind, order::Int, al::Float64, bt, F, J, H) where {W}
    nd = G_NDD[order+1]
    a, As, S, xr, xi, mr, mi, ab = w.a, w.As, w.S, w.xr, w.xi, w.mr, w.mi, w.ab
    @inbounds begin
    # ---- scaling, shift m, deviator S
    sig = k == SMF_EXP ? one(W) : _gpow2(maximum(abs, a))
    isig = 1 / sig
    for i in 1:9
        As[i] = a[i] * isig
    end
    m = (As[1] + As[5] + As[9]) / 3
    smax = zero(W)
    for i in 1:9
        S[i] = i % 4 == 1 ? As[i] - m : As[i]
        smax = max(smax, abs(S[i]))
    end
    _groots!(w)
    for i in 1:3
        mr[i], mi[i] = m + xr[i], xi[i]
    end
    # ---- the pair x[2], x[3] from its sum and product with A itself (see gmf_kernel.hpp)
    mid, Rp = (mr[2] + mr[3]) / 2, _cabs(mr[2] - mr[3], mi[2] - mi[3]) / 2
    small = abs(mid) + Rp <= abs(mr[1]) / 4
    aniso = abs(m) <= smax && Rp <= abs(mid - mr[1]) / 4
    pc = mr[1] != 0 && (small || aniso)
    pcen, ps, preal = zero(W), zero(W), true
    if pc
        _gnewton!(As, mr, mi, 1)
        m0 = mr[1]
        c2 = _gminors2(As)
        if abs(m0) >= abs(mid) + Rp
            pi_ = _gdetlu(As, zero(W)) / m0
            sg = (c2 - pi_) / m0
        else
            sg = As[1] + As[5] + As[9] - m0
            pi_ = c2 - m0 * sg
        end
        pcen = sg / 2
        disc = sg * sg / 4 - pi_
        ps = sqrt(abs(disc))
        preal = disc >= 0
        if preal
            r1 = pcen + _copysgn(ps, pcen)
            r2 = r1 != 0 ? pi_ / r1 : pcen - _copysgn(ps, pcen)
            mr[2], mi[2], mr[3], mi[3] = r1, zero(W), r2, zero(W)
        else
            mr[2], mi[2], mr[3], mi[3] = pcen, ps, pcen, -ps
        end
    end
    # ---- a conjugate pair near the negative real axis (log, powers): f jumps between its two nodes, so the result
    #      depends on the imaginary part beta itself. The pair from the 2x2 block of the deviator on its invariant
    #      subspace (orthogonal complement of the left eigenvector u of the real eigenvalue mr[1]):
    #      beta^2 = -(d^2 + b12 b21), d = (b11 - b22)/2, without the cancellation of the discriminant. The real node
    #      is ordered first (below)
    # (decided after the refinement above: a small pair next to a large eigenvalue has an inaccurate centre in x)
    cut = k != SMF_EXP && mr[2] + mr[3] < 0
    if cut
        pc || _gnewton!(As, mr, mi, 1)
        l0 = mr[1]
        M = w.Sp                                            # (free after _groots!): A - mu0 I, row-major
        for i in 1:9
            M[i] = i % 4 == 1 ? As[i] - l0 : As[i]
        end
        u0, u1, u2, nn = zero(W), zero(W), zero(W), -one(W)
        for (i, j) in ((0, 1), (0, 2), (1, 2))              # the largest cross product of two columns
            v0 = M[3+i+1] * M[6+j+1] - M[6+i+1] * M[3+j+1]
            v1 = M[6+i+1] * M[j+1] - M[i+1] * M[6+j+1]
            v2 = M[i+1] * M[3+j+1] - M[3+i+1] * M[j+1]
            n2 = v0 * v0 + v1 * v1 + v2 * v2
            if n2 > nn
                u0, u1, u2, nn = v0, v1, v2, n2
            end
        end
        if nn > 0
            iu = 1 / sqrt(nn)
            u0, u1, u2 = u0 * iu, u1 * iu, u2 * iu
            a0, a1, a2 = abs(u0), abs(u1), abs(u2)
            k1 = a1 < a0
            k2 = a2 < (k1 ? a1 : a0)
            ex, ey, ez = (k1 || k2) ? zero(W) : one(W), (k1 && !k2) ? one(W) : zero(W), k2 ? one(W) : zero(W)
            eu = ex * u0 + ey * u1 + ez * u2
            q0, q1, q2 = ex - eu * u0, ey - eu * u1, ez - eu * u2
            iq = 1 / sqrt(q0 * q0 + q1 * q1 + q2 * q2)
            q0, q1, q2 = q0 * iq, q1 * iq, q2 * iq
            r0, r1, r2 = u1 * q2 - u2 * q1, u2 * q0 - u0 * q2, u0 * q1 - u1 * q0
            sq0, sq1, sq2 = S[1] * q0 + S[2] * q1 + S[3] * q2, S[4] * q0 + S[5] * q1 + S[6] * q2,
                            S[7] * q0 + S[8] * q1 + S[9] * q2
            sr0, sr1, sr2 = S[1] * r0 + S[2] * r1 + S[3] * r2, S[4] * r0 + S[5] * r1 + S[6] * r2,
                            S[7] * r0 + S[8] * r1 + S[9] * r2
            b11, b12 = q0 * sq0 + q1 * sq1 + q2 * sq2, q0 * sr0 + q1 * sr1 + q2 * sr2
            b21, b22 = r0 * sq0 + r1 * sq1 + r2 * sq2, r0 * sr0 + r1 * sr1 + r2 * sr2
            dd = (b11 - b22) / 2
            beta2 = -(dd * dd + b12 * b21)
            cc = m + (b11 + b22) / 2
            be = sqrt(abs(beta2))
            preal = beta2 <= 0
            pcen, ps = cc, be
            if preal
                mr[2], mi[2], mr[3], mi[3] = cc + be, zero(W), cc - be, zero(W)
            else
                mr[2], mi[2], mr[3], mi[3] = cc, be, cc, -be
            end
            pc = true
        end
    end
    for i in 1:3
        ab[i] = pc
        if pc
            xr[i], xi[i] = mr[i] - m, mi[i]
        end
    end
    # ---- node order: exp largest real part last; log, pow smallest modulus last; a conjugate pair stays adjacent
    perm = (1, 2, 3)
    if xi[2] != 0
        # real node first or last; first for a pair near the negative real axis (see above)
        cut || _gkey(k, mr[1], mi[1]) <= _gkey(k, mr[2], mi[2]) || (perm = (2, 3, 1))
    else
        for (p, q) in ((1, 2), (2, 3), (1, 2))
            if _gkey(k, mr[perm[p]], mi[perm[p]]) > _gkey(k, mr[perm[q]], mi[perm[q]])
                perm = p == 1 ? (perm[2], perm[1], perm[3]) : (perm[1], perm[3], perm[2])
            end
        end
    end
    txr, txi, tmr, tmi, tab = (xr[1], xr[2], xr[3]), (xi[1], xi[2], xi[3]), (mr[1], mr[2], mr[3]), (mi[1], mi[2], mi[3]),
                              (ab[1], ab[2], ab[3])
    for i in 1:3
        xr[i], xi[i], mr[i], mi[i], ab[i] = txr[perm[i]], txi[perm[i]], tmr[perm[i]], tmi[perm[i]], tab[perm[i]]
    end
    og = (perm[1] - 1.0, perm[2] - 1.0, perm[3] - 1.0)
    # ---- groups: all three, else the closest real / conjugate pair, else none
    rhoF = k == SMF_POW && 3.0 / al < G_RHO ? 3.0 / al : G_RHO
    cs = (xr[1] + xr[2] + xr[3]) / 3
    R3 = max(_cabs(xr[1] - cs, xi[1]), _cabs(xr[2] - cs, xi[2]), _cabs(xr[3] - cs, xi[3]))
    triple = _gok(k, m, rhoF, cs, R3)
    ps1, ps2, ps3 = false, false, false
    bestR = W(Inf)
    for (p, (i, j)) in enumerate(((1, 2), (1, 3), (2, 3)))
        kk = 6 - i - j
        elig = (xi[i] == 0 && xi[j] == 0) || (xr[i] == xr[j] && xi[i] == -xi[j])
        cc, R = (xr[i] + xr[j]) / 2, _cabs(xr[i] - xr[j], xi[i] - xi[j]) / 2
        okp = elig && _gok(k, m, rhoF, cc, R) && R <= G_RHO_POLE * _cabs(cc - xr[kk], -xi[kk]) && !triple
        if okp && R < bestR
            bestR = R
            ps1, ps2, ps3 = p == 1, p == 2, p == 3
        end
    end
    inG = (triple || ps1 || ps2, triple || ps1 || ps3, triple || ps2 || ps3)
    pk = (ps3, ps2, ps1)
    # ---- group centre and offsets; the refined pair as a group: symmetric offsets
    gsum = count(inG)
    cxg = ((inG[1] ? xr[1] : zero(W)) + (inG[2] ? xr[2] : zero(W)) + (inG[3] ? xr[3] : zero(W))) / max(gsum, 1)
    cmu = m + cxg
    is12 = (og[1] > 0.5, og[2] > 0.5, og[3] > 0.5)
    ovr = pc && gsum == 2 && ((inG[1] && inG[2] && is12[1] && is12[2]) || (inG[1] && inG[3] && is12[1] && is12[3]) ||
                              (inG[2] && inG[3] && is12[2] && is12[3]))
    if ovr
        cxg, cmu = pcen - m, pcen
    end
    dpr, dpi = preal ? ps : zero(W), preal ? zero(W) : ps
    offr, offi = w.offr, w.offi
    for i in 1:3
        neg = (mr[i] - pcen < 0) || (mi[i] < 0)
        dr, di = neg ? -dpr : dpr, neg ? -dpi : dpi
        offr[i], offi[i] = zero(W), zero(W)
        if ovr && inG[i]
            mr[i], mi[i] = pcen + dr, di
            xr[i], xi[i] = mr[i] - m, mi[i]
            offr[i], offi[i] = dr, di
        elseif inG[i]
            offr[i], offi[i] = xr[i] - cxg, xi[i]
        end
    end
    # ---- single real nodes far from m: Newton on det(A - mu I)
    for i in 1:3
        if !inG[i] && !ab[i] && xi[i] == 0 && abs(xr[i]) > abs(m) / 2
            _gnewton!(As, mr, mi, i)
            ab[i] = true
            xr[i], xi[i] = mr[i] - m, mi[i]
        end
    end
    absc = (inG[1] && ab[1]) || (inG[2] && ab[2]) || (inG[3] && ab[3])
    apole = zero(W)
    for kk in 1:3
        pk[kk] && (apole = (absc || ab[kk]) ? cmu - mr[kk] : cxg - xr[kk])
    end
    xshift = k == SMF_EXP ? max(xr[1], xr[2], xr[3]) : zero(W)
    bad = k != SMF_EXP && ((xi[1] == 0 && mr[1] <= 0) || (xi[2] == 0 && mr[2] <= 0) || (xi[3] == 0 && mr[3] <= 0))

    # ---- divided differences: group part (Taylor series streamed over the series index)
    Dr, Di = w.Dr, w.Di
    fill!(Dr, zero(W)); fill!(Di, zero(W))
    if inG[1] || inG[2] || inG[3]
        L = k == SMF_EXP ? one(W) : (cmu > 0 ? cmu : one(W))
        iL = 1 / L
        ur, ui = w.ur, w.ui
        for i in 1:3
            ur[i], ui[i] = offr[i] * iL, offi[i] * iL
        end
        ap = abs(apole) > 0 ? apole * iL : one(W)
        iap = 1 / ap
        rr = max(_cabs(ur[1], ui[1]), _cabs(ur[2], ui[2]), _cabs(ur[3], ui[3]))
        rq0 = rr * abs(iap)
        rq = (pk[1] || pk[2] || pk[3]) && rq0 <= 1 ? rq0 : zero(W)
        rr = rr <= G_R_EXP + 0.5 ? rr : zero(W)
        K = _gnterms(bt, 3 * (order + 1) - 1, Float64(_val(rr)), Float64(_val(rq)))
        b0 = k == SMF_LOG ? _gcoef0(SMF_LOG, al, sig, m, cxg, zero(W), cmu, zero(W), 0)[1] : one(W)
        Epr, Epi, Ecr, Eci, accr, acci = w.Epr, w.Epi, w.Ecr, w.Eci, w.accr, w.acci
        for c in 1:nd
            par = G_TPAR[c]
            if inG[G_TNODE[c]]
                Epr[c], Epi[c] = zero(W), zero(W)
            else
                Epr[c], Epi[c] = par == 0 ? one(W) : Epr[par], par == 0 ? zero(W) : Epi[par]
            end
            accr[c], acci[c] = zero(W), zero(W)
        end
        q1, q2, q3 = zero(W), zero(W), zero(W)
        for j in 0:K-1
            bj = j == 0 ? b0 : W(bt[j+1])
            g1 = (bj - q1) * iap
            g2 = (g1 - q2) * iap
            g3 = (g2 - q3) * iap
            q1, q2, q3 = g1, g2, g3
            q = (bj, g1, g2, g3)
            for c in 1:nd
                par, i = G_TPAR[c], G_TNODE[c]
                if inG[i]
                    eppr = par == 0 ? (j == 0 ? one(W) : zero(W)) : Epr[par]
                    eppi = par == 0 ? zero(W) : Epi[par]
                    Ecr[c] = eppr + (ur[i] * Epr[c] - ui[i] * Epi[c])
                    Eci[c] = eppi + (ur[i] * Epi[c] + ui[i] * Epr[c])
                else
                    Ecr[c] = par == 0 ? zero(W) : Ecr[par]
                    Eci[c] = par == 0 ? zero(W) : Eci[par]
                end
                ms = G_TMS[c]
                g = pk[1] ? q[ms[1]+1] : (pk[2] ? q[ms[2]+1] : (pk[3] ? q[ms[3]+1] : q[1]))
                accr[c] += Ecr[c] * g
                acci[c] += Eci[c] * g
            end
            for c in 1:nd
                Epr[c], Epi[c] = Ecr[c], Eci[c]
            end
        end
        for c in 1:nd
            ms = G_TMS[c]
            n, ko = ms[1] + ms[2] + ms[3] - 1, ms[1] - 1
            if k == SMF_EXP
                fac = exp(cxg - xshift)
            else
                iLn = one(W)
                for _ in 1:(n-(k == SMF_POW ? ko : 0))
                    iLn = iLn * iL
                end
                fac = k == SMF_LOG ? iLn : _pow_k(sig * L, al, Float64(ko)) * iLn
            end
            Dr[c], Di[c] = accr[c] * fac, acci[c] * fac
        end
    end
    # ---- residues of the single nodes
    for i in 1:3
        inG[i] && continue
        xsr = k == SMF_EXP ? xr[i] - xshift : xr[i]
        f00 = _gcoef0(k, al, sig, m, xsr, xi[i], mr[i], mi[i], 0)
        f01 = k == SMF_POW && order >= 1 ? _gcoef0(k, al, sig, m, xsr, xi[i], mr[i], mi[i], 1) : f00
        f02 = k == SMF_POW && order >= 2 ? _gcoef0(k, al, sig, m, xsr, xi[i], mr[i], mi[i], 2) : f00
        imr, imi = _cinv(mr[i], mi[i])
        wz = ntuple(Val(3)) do z
            z == i && return (zero(W), zero(W))
            useMu = ab[i] || ab[z]
            _cinv(useMu ? mr[i] - mr[z] : xr[i] - xr[z], useMu ? mi[i] - mi[z] : xi[i] - xi[z])
        end
        for c in 1:nd
            ms = G_TMS[c]
            mi_ = ms[i]
            mi_ == 0 && continue
            fbr, fbi = k == SMF_POW ? (ms[1] == 1 ? f00 : (ms[1] == 2 ? f01 : f02)) : f00
            if k == SMF_EXP
                s0r, s0i, s1r, s1i, s2r, s2i = fbr, fbi, fbr, fbi, fbr * 0.5, fbi * 0.5
            elseif k == SMF_LOG
                s0r, s0i, s1r, s1i = fbr, fbi, imr, imi
                s2r, s2i = (imr * imr - imi * imi) * -0.5, (2 * imr * imi) * -0.5
            else
                t1r, t1i = _cmul(fbr, fbi, imr, imi)
                s1r, s1i = t1r * al, t1i * al
                t2r, t2i = _cmul(s1r, s1i, imr, imi)
                h = (al - 1.0) / 2.0
                s0r, s0i, s2r, s2i = fbr, fbi, t2r * h, t2i * h
            end
            for z in 1:3
                mz = ms[z]
                (z == i || mz == 0) && continue
                wr, wi = wz[z]
                pr, pim = one(W), zero(W)
                for _ in 1:mz
                    pr, pim = _cmul(pr, pim, wr, wi)
                end
                t1r, t1i = _cmul(pr, pim, wr, wi)
                p1r, p1i = t1r * -mz, t1i * -mz
                t2r, t2i = _cmul(t1r, t1i, wr, wi)
                p2r, p2i = t2r * (mz * (mz + 1) / 2.0), t2i * (mz * (mz + 1) / 2.0)
                n0r, n0i = _cmul(s0r, s0i, pr, pim)
                a1r, a1i = _cmul(s1r, s1i, pr, pim)
                b1r, b1i = _cmul(s0r, s0i, p1r, p1i)
                a2r, a2i = _cmul(s2r, s2i, pr, pim)
                b2r, b2i = _cmul(s1r, s1i, p1r, p1i)
                c2r, c2i = _cmul(s0r, s0i, p2r, p2i)
                s0r, s0i = n0r, n0i
                s1r, s1i = a1r + b1r, a1i + b1i
                s2r, s2i = (a2r + b2r) + c2r, (a2i + b2i) + c2i
            end
            Dr[c] += mi_ == 1 ? s0r : (mi_ == 2 ? s1r : s2r)
            Di[c] += mi_ == 1 ? s0i : (mi_ == 2 ? s1i : s2i)
        end
    end
    # ---- real basis and assembly
    b0i, b1i = xi[1], xi[2]
    B0, B1, r = w.B0, w.B1, w.r
    for i in 1:9
        dg = i % 4 == 1
        B0[i] = dg ? (ab[1] ? As[i] - mr[1] : S[i] - xr[1]) : S[i]
        B1[i] = dg ? (ab[2] ? As[i] - mr[2] : S[i] - xr[2]) : S[i]
        r[i] = dg ? one(W) : zero(W)                    # r[9 m + k]: basis matrix m (I, B0, B0 B1 - b0 b1 I)
        r[9+i] = B0[i]
    end
    for i2 in 0:2, j2 in 0:2
        r[18+3i2+j2+1] = B0[3i2+1] * B1[j2+1] + B0[3i2+2] * B1[3+j2+1] + B0[3i2+3] * B1[6+j2+1] -
                         (i2 == j2 ? b0i * b1i : zero(W))
    end
    a01 = (ab[1] || ab[2]) ? mr[1] - mr[2] : xr[1] - xr[2]
    N10, N20, N21 = -b0i, -b0i * a01, -(b0i + b1i)
    nanf = bad ? W(NaN) : one(W)
    e = k == SMF_EXP ? exp(m + xshift) : one(W)
    d0 = Dr[_gpos1(0)] - (N10 * Di[_gpos1(1)] + N20 * Di[_gpos1(2)])
    d1 = Dr[_gpos1(1)] - N21 * Di[_gpos1(2)]
    d2 = Dr[_gpos1(2)]
    for aa in 0:2, bb in 0:2
        kk = 3aa + bb + 1
        F[aa+1, bb+1] = (d0 * r[kk] + d1 * r[9+kk] + d2 * r[18+kk]) * e * nanf
    end
    order < 1 && return nothing
    sc1 = (k == SMF_EXP ? e : (k == SMF_LOG ? isig : one(W))) * nanf
    E2r, E2i, D2, Y = w.E2r, w.E2i, w.D2, w.Y
    for j in 0:2, l in 0:2
        E2r[3j+l+1], E2i[3j+l+1] = Dr[_gpos2(j, l)], Di[_gpos2(j, l)]
    end
    for l in 0:2                                         # T on index 1 (stride 3)
        _gapplyT!(E2r, E2i, l, 3, N10, N20, N21)
    end
    for j in 0:2                                         # T on index 2 (stride 1), real part
        D2[3j+1] = E2r[3j+1] - (N10 * E2i[3j+2] + N20 * E2i[3j+3])
        D2[3j+2] = E2r[3j+2] - N21 * E2i[3j+3]
        D2[3j+3] = E2r[3j+3]
    end
    for mm in 0:2, kk in 1:9
        Y[9mm+kk] = D2[3mm+1] * r[kk] + D2[3mm+2] * r[9+kk] + D2[3mm+3] * r[18+kk]
    end
    for aa in 0:2, bb in 0:2, cc in 0:2, dd in 0:2
        i1, i2 = 3aa + cc + 1, 3dd + bb + 1
        J[aa+1, bb+1, cc+1, dd+1] = (r[i1] * Y[i2] + r[9+i1] * Y[9+i2] + r[18+i1] * Y[18+i2]) * sc1
    end
    order < 2 && return nothing
    sc2 = (k == SMF_EXP ? e : (k == SMF_LOG ? isig * isig : one(W))) * nanf
    E3r, E3i, Z, Yh = w.E3r, w.E3i, w.Z, w.Yh
    for j in 0:2, l in 0:2, p in 0:2
        E3r[9j+3l+p+1], E3i[9j+3l+p+1] = Dr[_gpos3(j, l, p)], Di[_gpos3(j, l, p)]
    end
    for s in (9, 3, 1), base in 0:26                    # T on each index
        (base ÷ s) % 3 == 0 && _gapplyT!(E3r, E3i, base, s, N10, N20, N21)
    end
    for mn in 0:8, kk in 1:9                            # Z[mn][k] = sum_p D3[mn p] r[p][k]
        Z[9mn+kk] = E3r[3mn+1] * r[kk] + E3r[3mn+2] * r[9+kk] + E3r[3mn+3] * r[18+kk]
    end
    for mm in 0:2, de in 0:8, fb in 1:9                 # Yh[m][de fb] = sum_n r[n][de] Z[mn][fb]
        Yh[81mm+9de+fb] = r[de+1] * Z[9(3mm)+fb] + r[9+de+1] * Z[9(3mm+1)+fb] + r[18+de+1] * Z[9(3mm+2)+fb]
    end
    for aa in 0:2, bb in 0:2, cc in 0:2, dd in 0:2, ee in 0:2, ff in 0:2
        ac, ae = 3aa + cc + 1, 3aa + ee + 1
        i1, i2 = 9(3dd + ee) + 3ff + bb + 1, 9(3ff + cc) + 3dd + bb + 1
        h = r[ac] * Yh[i1] + r[9+ac] * Yh[81+i1] + r[18+ac] * Yh[162+i1] +
            r[ae] * Yh[i2] + r[9+ae] * Yh[81+i2] + r[18+ae] * Yh[162+i2]
        H[aa+1, bb+1, cc+1, dd+1, ee+1, ff+1] = h * sc2
    end
    end # @inbounds
    return nothing
end

# plain value of a number (Dual: its value) for the number of series terms
_val(x::AbstractFloat) = x
_val(x::Real) = hasproperty(x, :value) ? _val(getproperty(x, :value)) : Float64(x)

"""
    gmf(kind, A; alpha=nothing, order=0)

f(A) and its first and second derivative for general (non-symmetric) real 3x3 matrices `A` (3 x 3, or 3 x 3 x n),
principal branch. Returns `F` (3 x 3), `(F, J)` or `(F, J, H)` for `order = 0, 1, 2` with
`J[a,b,c,d] = dF[a,b]/dA[c,d]` and `H[a,b,c,d,e,f] = d2F[a,b]/dA[c,d] dA[e,f]` (the full derivatives, exact also for
defective `A` and complex eigenvalues); with `n` matrices a trailing dimension `n`. `kind` as for [`smf`](@ref);
log, sqrt and pow give NaN for a matrix with a real eigenvalue <= 0.
"""
function gmf(kind, A::AbstractArray; alpha::Union{Nothing,Real} = nothing, order = 0)
    k = tokind(kind)
    if alpha === nothing
        k == SMF_POW && throw(ArgumentError("alpha is required for pow"))
        alpha = 0.0
    end
    o = order isa Val ? typeof(order).parameters[1] : Int(order)
    o in (0, 1, 2) || throw(ArgumentError("order must be 0, 1 or 2, got $o"))
    T = _wtype(eltype(A))
    n = _glayout(A)
    tail = n === nothing ? () : (n,)
    F = Array{T}(undef, 3, 3, tail...)
    J = o >= 1 ? Array{T}(undef, 3, 3, 3, 3, tail...) : nothing
    H = o >= 2 ? Array{T}(undef, 3, 3, 3, 3, 3, 3, tail...) : nothing
    gmf!(F, J, H, k, A; alpha = alpha)
    return o == 0 ? F : (o == 1 ? (F, J) : (F, J, H))
end

function _glayout(A)
    sz = size(A)
    ndims(A) == 2 && sz == (3, 3) && return nothing
    ndims(A) == 3 && sz[1] == 3 && sz[2] == 3 && return sz[3]
    throw(DimensionMismatch("expected a matrix 3 x 3 or 3 x 3 x n, got size $sz"))
end

# matrices lo..hi of A (batch: the last index) with the workspace w
function _grange!(w::_GWork, k, order, al, bt, F, J, H, A, lo, hi, ::Val{B}) where {B}
    @inbounds for i in lo:hi
        for r in 1:3, c in 1:3
            w.a[3(r-1)+c] = B ? A[r, c, i] : A[r, c]
        end
        if B
            _geval!(w, k, order, al, bt, _last(F, i), J === nothing ? nothing : _last(J, i),
                    H === nothing ? nothing : _last(H, i))
        else
            _geval!(w, k, order, al, bt, F, J, H)
        end
    end
    return nothing
end

"""
    gmf!(F, J, H, kind, A; alpha=0.0) -> F

In-place version of [`gmf`](@ref): `F`, `J` (or `nothing`) and `H` (or `nothing`; needs `J`) of the sizes listed
there are overwritten; multithreaded over chunks of matrices like [`smf!`](@ref).
"""
function gmf!(F::AbstractArray, J::Union{Nothing,AbstractArray}, H::Union{Nothing,AbstractArray}, kind,
              A::AbstractArray; alpha::Real = 0.0)
    k = tokind(kind)
    al = _alpha(k, alpha)
    _check(J, H)
    if k == SMF_SQRT
        k = SMF_POW
    end
    order = H !== nothing ? 2 : (J !== nothing ? 1 : 0)
    bt = _gseries(k, al)
    n = _glayout(A)
    T = _wtype(eltype(A))
    if n === nothing
        _chkout(F, J, H, (3, 3), (3, 3, 3, 3), (3, 3, 3, 3, 3, 3))
        _grange!(_GWork{T}(), k, order, al, bt, F, J, H, A, 1, 1, Val(false))
        return F
    end
    _chkout(F, J, H, (3, 3, n), (3, 3, 3, 3, n), (3, 3, 3, 3, 3, 3, n))
    chunk = H === nothing ? 1024 : 256
    nchunks = cld(n, chunk)
    if nchunks <= 1 || Threads.nthreads() == 1
        _grange!(_GWork{T}(), k, order, al, bt, F, J, H, A, 1, n, Val(true))
    else
        Threads.@threads :dynamic for ic in 1:nchunks      # one workspace per chunk
            lo = (ic - 1) * chunk + 1
            _grange!(_GWork{T}(), k, order, al, bt, F, J, H, A, lo, min(n, lo + chunk - 1), Val(true))
        end
    end
    return F
end
