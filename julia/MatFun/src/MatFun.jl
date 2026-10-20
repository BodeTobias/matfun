"""
    MatFun

Matrix functions exp, log, sqrt and A^alpha of 3x3 matrices with first and second derivatives: symmetric matrices
[`smf`](@ref), general (non-symmetric) matrices [`gmf`](@ref). Native Julia port of the C++ kernels
`c/include/matfun/smf_kernel.hpp` (`eval3`, HPACK = 0) and `gmf_kernel.hpp`; conventions in the repository README.
"""
module MatFun

export smf!, smf, gmf!, gmf

# function kind, same codes as the C enum smf_kind; the public functions take :exp, :log, :pow, :sqrt (or the
# codes 0-3, or these values)
@enum Kind::Int32 SMF_EXP = 0 SMF_LOG = 1 SMF_POW = 2 SMF_SQRT = 3

const NTERMS = 30      # Taylor terms in divdiff2 (tau^30 * 31 < 1e-16)
const TAU = 0.25       # cluster radius relative to the middle point (exp: absolute 1; pow: min(TAU, 3/al))
const SQRT6 = 2.449489742783178
const TWO_PI_3 = 2.0943951023931957

tokind(k::Kind) = k
tokind(k::Integer) = 0 <= k <= 3 ? Kind(k) : throw(ArgumentError("unknown kind code $k (0 exp, 1 log, 2 pow, 3 sqrt)"))
function tokind(s::Symbol)
    s === :exp && return SMF_EXP
    s === :log && return SMF_LOG
    s === :pow && return SMF_POW
    s === :sqrt && return SMF_SQRT
    throw(ArgumentError("unknown kind :$s (:exp, :log, :pow, :sqrt)"))
end

# exponent used internally: alpha for pow (must be > 0), 0.5 for sqrt, 0 for exp/log (as c/src/matfun.cc)
function _alpha(k::Kind, alpha::Real)
    if k == SMF_POW
        a = Float64(alpha)
        a > 0 || throw(ArgumentError("alpha must be > 0 for pow, got $alpha"))
        return a
    end
    return k == SMF_SQRT ? 0.5 : 0.0
end

# working number type: at least Float64 (Float32/Int inputs are computed in Float64), Duals stay Duals
_wtype(::Type{T}) where {T<:Real} = promote_type(float(T), Float64)

# ------------------------------------------------------------------ scalar helpers
# C semantics instead of DomainError: log(x < 0) = NaN
@inline _log(x) = x < 0 ? oftype(x, NaN) : log(x)
@inline _log1p(x) = x < -1 ? oftype(x, NaN) : log1p(x)
# 0^e: inf for e < 0, 1 for e = 0, 0 for e > 0
@inline _zero_pow(e::Float64) = e < 0 ? Inf : (e == 0 ? 1.0 : 0.0)

# x^(al - k) for x > 0 with the exponent al - k = bh + bl exact (TwoSum): the rounding of al - k alone would cost
# |log x| times its error (1e-14 at x ~ 1e100); x^(bh + bl) = x^bh (1 + bl log x)
@inline function _pow_k(x, al::Float64, k::Float64)
    bh = al - k
    bb = bh - al
    bl = (al - (bh - bb)) + (-k - bb)
    p = x^bh
    return bl == 0 ? p : p * (1.0 + bl * log(x))
end

# log of the eigenvalues; near m (|x| <= m/2): log(m sig) + log1p(x/m) with x = lambda/sig - m from the Rayleigh
# quotients of the deviator (the rounding of m cancels against A - m I, accurate also near I)
@inline function _log_near(lam, m, sig, d0, d1, d2, a3, a4, a5, vx, vy, vz, cs, sn, b11, b12, b22)
    bv = vx * (d0 * vx + a3 * vy + a4 * vz) + vy * (a3 * vx + d1 * vy + a5 * vz) + vz * (a4 * vx + a5 * vy + d2 * vz)
    xd = (bv, cs * cs * b11 + 2.0 * cs * sn * b12 + sn * sn * b22, sn * sn * b11 - 2.0 * cs * sn * b12 + cs * cs * b22)
    lms = _log(m * sig)
    return ntuple(i -> abs(xd[i]) <= m / 2 ? lms + _log1p(xd[i] / m) : _log(lam[i]), Val(3))
end

@inline function _fun(::Val{K}, x::W, al::Float64) where {K,W}
    K === SMF_EXP && return exp(x)
    K === SMF_LOG && return _log(x)
    x > 0 || return zero(x)                    # powers: PSD round-off clamped to 0
    return K === SMF_SQRT ? sqrt(x) : x^al
end

# first divided difference f[a, b] (f[a, a] = f'(a)), numerically stable
@inline function _divdiff(::Val{K}, a::W, b::W, fa::W, fb::W, al::Float64) where {K,W}
    if K === SMF_EXP                           # e^hi (1 - e^-d) / d, d = hi - lo >= 0 (no overflow)
        d = abs(a - b)
        g = d > 0 ? -expm1(-d) / d : one(d)
        return (a > b ? fa : fb) * g
    elseif K === SMF_LOG                       # 2 artanh(z) / (z (a+b)), z = (a-b)/(a+b)
        s = 0.5 * (a + b)
        z = (a - b) / (a + b)
        az = abs(z)
        z2 = z * z
        az < 1e-3 && return (1.0 + z2 / 3.0 + z2 * z2 / 5.0 + z2 * z2 * z2 / 7.0) / s
        az >= 0.5 && return (_log(a) - _log(b)) / (a - b)
        return (atanh(z) / z) / s              # also reached for z = NaN, as in C
    else
        a = a > 0 ? a : zero(a)                # powers: PSD round-off clamped to 0
        b = b > 0 ? b : zero(b)
        K === SMF_SQRT && return 1.0 / (sqrt(a) + sqrt(b))
        s = 0.5 * (a + b)
        ss = s > 0 ? s : one(s)
        z = (a - b) / (2.0 * ss)
        az = abs(z)
        # az <= 0 means z == 0 (written so that Dual partials do not matter); az >= 0.5 or NaN: far apart
        (az <= 0 || az < 0.5) || return (fa - fb) / (a - b)     # fa = f(a) = a^al (0 for a <= 0)
        g = az <= 0 ? W(al) : (expm1(al * log1p(z)) - expm1(al * log1p(-z))) / (2.0 * z)
        return (s > 0 ? _pow_k(ss, al, 1.0) : W(_zero_pow(al - 1.0))) * g
    end
end

# f'(x) = f[x, x], fx = f(x)
@inline function _deriv(::Val{K}, x::W, fx::W, al::Float64) where {K,W}
    K === SMF_EXP && return fx
    K === SMF_LOG && return 1.0 / x
    x = x > 0 ? x : zero(x)
    K === SMF_SQRT && return 1.0 / (2.0 * sqrt(x))
    return x > 0 ? al * _pow_k(x, al, 1.0) : W(al * _zero_pow(al - 1.0))
end

# series coefficients c[k] = f^(k+2)(m)/(k+2)! without the m-dependent factor (exp: e^m, log: 1/m^2,
# pow: m^(al-2)); same recurrences and rounding as dd2_coef in the C kernel
function _dd2_coef(k::Kind, al::Float64)::NTuple{NTERMS,Float64}
    if k == SMF_EXP
        f_1 = 0.5
        Base.Cartesian.@nexprs 29 i -> f_{i + 1} = f_i / ((i - 1) + 3.0)
        return Base.Cartesian.@ntuple 30 f
    elseif k == SMF_LOG
        return ntuple(i -> (isodd(i - 1) ? 1.0 : -1.0) / ((i - 1) + 2.0), Val(NTERMS))
    else
        b_1 = al * (al - 1.0) / 2.0
        Base.Cartesian.@nexprs 29 i -> b_{i + 1} = b_i * ((al - (i - 1) - 2.0) / ((i - 1) + 3.0))
        return Base.Cartesian.@ntuple 30 b
    end
end

# second divided difference f[x, y, z] (symmetric; f[x, x, x] = f''(x)/2):
#   all three points in a cluster around the middle one m -> Taylor series
#       f[x, y, z] = sum_k f^(k+2)(m)/(k+2)! h_k(lo - m, hi - m),  h_k(u, w) = sum_i u^i w^(k-i)
#   otherwise the difference formula over the most distant pair, from the first divided differences
@inline function _divdiff2(::Val{K}, li::W, lj::W, lk::W, fi::W, fj::W, fk::W, Dij::W, Dik::W, Djk::W,
                           al::Float64, c::NTuple{NTERMS,Float64}) where {K,W}
    if K === SMF_POW || K === SMF_SQRT
        li = li > 0 ? li : zero(li)
        lj = lj > 0 ? lj : zero(lj)
        lk = lk > 0 ? lk : zero(lk)
    end
    # difference formula (f[y,z] - f[x,y]) / (z - x) with (x, z) the most distant pair
    dA = lk - li
    dB = lj - li
    dC = lk - lj
    aA = abs(dA)
    aB = abs(dB)
    aC = abs(dC)
    useA = (aA >= aB) & (aA >= aC)
    useB = !useA & (aB >= aC)
    num = useA ? Djk - Dij : (useB ? Djk - Dik : Dik - Dij)
    den = useA ? dA : (useB ? dB : dC)
    aden = useA ? aA : (useB ? aB : aC)
    # middle point, extremes
    mn = li < lj ? li : lj
    mx = li < lj ? lj : li
    lo = mn < lk ? mn : lk
    hi = mx > lk ? mx : lk
    t = mx < lk ? mx : lk
    mid = mn > t ? mn : t
    fm = mid == li ? fi : (mid == lj ? fj : fk)
    if K === SMF_EXP
        cluster = (hi - lo) <= 1.0
        u = lo - mid
        w = hi - mid
        scale = fm
    else
        # pow: the coefficients binom(al, k+2) grow with al, so the radius shrinks to 3/al for al > 12
        tau = K === SMF_POW && 3.0 / al < TAU ? 3.0 / al : TAU
        cluster = (mid > 0) & (mid - lo <= tau * mid) & (hi - mid <= tau * mid)
        ms = cluster ? mid : one(mid)
        u = (lo - mid) / ms
        w = (hi - mid) / ms
        scale = (K === SMF_LOG ? one(ms) : (cluster ? fm : one(ms))) / (ms * ms)   # pow: m^(al-2) = f(m)/m^2
    end
    if cluster
        h = one(u)
        wk = h
        s = c[1] + zero(u)
        @inbounds for n in 2:NTERMS
            wk = wk * w
            h = u * h + wk
            s = s + c[n] * h
        end
        out = scale * s
    else
        out = num / (aden <= 0 ? one(den) : den)        # den != 0 (NaN counts as != 0, as in C)
    end
    if K === SMF_POW || K === SMF_SQRT     # all three points 0: f''(0)/2 = binom(al, 2) 0^(al-2)
        z2 = al == 1.0 ? 0.0 : al * (al - 1.0) / 2.0 * _zero_pow(al - 2.0)
        out = hi > 0 ? out : W(z2)
    end
    return out
end

# Voigt index -> matrix indices (1-based)
const VP = (1, 2, 3, 1, 1, 2)
const VQ = (1, 2, 3, 2, 3, 3)

# 3x3 matrices as column-major NTuple{9}
@inline _ix(i, j) = i + 3 * (j - 1)
# position of (i, j, m) in the 10 distinct entries of a symmetric 3x3x3 tensor, ordered as TRI in the C kernel
# (used when generating the H code)
function _t3(i, j, m)
    a, b, c = sort([i, j, m])
    return (1, 2, 3, 0, 4, 5, 0, 0, 6, 0, 0, 0, 0, 7, 8, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 10)[(a - 1) * 9 + (b - 1) * 3 + c]
end

# ------------------------------------------------------------------ the kernel (eval3 of smf_kernel.hpp)
# largest power of two <= x (x > 0, normal), used to scale A exactly before the eigendecomposition; number types
# without an exponent (e.g. ForwardDiff.Dual) are not scaled, nor is NaN/Inf input (exponent would throw; the
# result is NaN as in the C library)
_pow2floor(x::AbstractFloat) = isfinite(x) ? ldexp(one(x), exponent(x)) : one(x)
_pow2floor(x::Real) = one(x)

@inline function _eval3!(::Val{K}, F, J, H, v, al::Float64, c::NTuple{NTERMS,Float64}) where {K}
    W = _wtype(eltype(v))
    one_ = one(W)
    zero_ = zero(W)
    @inbounds b0, b1, b2, b3, b4, b5 = W(v[1]), W(v[2]), W(v[3]), W(v[4]), W(v[5]), W(v[6])
    # ---- scale A by a power of two sig ~ max |A_ij| (exact), so that the products below (up to |A|^4 in the cross
    #      products) neither overflow nor underflow; eigenvalues are scaled back below, eigenvectors are unaffected
    s01 = abs(b0) > abs(b1) ? abs(b0) : abs(b1)
    s23 = abs(b2) > abs(b3) ? abs(b2) : abs(b3)
    s45 = abs(b4) > abs(b5) ? abs(b4) : abs(b5)
    s03 = s01 > s23 ? s01 : s23
    scale0 = (s03 > s45 ? s03 : s45) + 1e-300
    sig = _pow2floor(scale0)
    isig = 1 / sig
    a0, a1, a2, a3, a4, a5 = b0 * isig, b1 * isig, b2 * isig, b3 * isig, b4 * isig, b5 * isig
    # ---- simple eigenvalue (trigonometric formula) and its eigenvector (cross products), from the deviator
    #      A - m I (exact where the diagonal is close to m, so accurate also near I), less its trace residue tr3 from
    #      the rounding of m and normalized by its own magnitude (a power of two): the formula needs a traceless
    #      matrix, and the eigenvector then does not depend on how small the deviator is
    m = (a0 + a1 + a2) / 3.0
    d0 = a0 - m
    d1 = a1 - m
    d2 = a2 - m
    tr3 = (d0 + d1 + d2) / 3.0
    e0, e1, e2 = d0 - tr3, d1 - tr3, d2 - tr3
    dmax = max(max(abs(e0), abs(e1)), max(abs(e2), abs(a3)), max(abs(a4), abs(a5)))
    iso = !(dmax > 1e-300)                            # A = m I
    ids = iso ? one_ : 1 / _pow2floor(dmax)
    g0, g1, g2, g3, g4, g5 = e0 * ids, e1 * ids, e2 * ids, a3 * ids, a4 * ids, a5 * ids
    p = g0 * g0 + g1 * g1 + g2 * g2 + 2.0 * (g3 * g3 + g4 * g4 + g5 * g5)
    q = g0 * (g1 * g2 - g5 * g5) - g3 * (g3 * g2 - g5 * g4) + g4 * (g3 * g5 - g1 * g4)
    pg = iso ? one_ : p
    qg = iso ? zero_ : q
    r = 3.0 * SQRT6 * qg / (pg * sqrt(pg))
    r = r > 1.0 ? one_ : (r < -1.0 ? -one_ : r)
    t = sqrt(2.0 * pg / 3.0)
    ph = acos(r) / 3.0
    sh = t * cos(r >= 0 ? ph : ph + TWO_PI_3)
    m00 = g0 - sh
    m11 = g1 - sh
    m22 = g2 - sh                    # rows M0 = (m00, g3, g4), M1 = (g3, m11, g5), M2 = (g4, g5, m22)
    c0x = g3 * g5 - g4 * m11; c0y = g4 * g3 - m00 * g5; c0z = m00 * m11 - g3 * g3
    c1x = g3 * m22 - g4 * g5; c1y = g4 * g4 - m00 * m22; c1z = m00 * g5 - g3 * g4
    c2x = m11 * m22 - g5 * g5; c2y = g5 * g4 - g3 * m22; c2z = g3 * g5 - m11 * g4
    n0 = c0x * c0x + c0y * c0y + c0z * c0z
    n1 = c1x * c1x + c1y * c1y + c1z * c1z
    n2 = c2x * c2x + c2y * c2y + c2z * c2z
    s1 = n1 > n0
    cx = s1 ? c1x : c0x; cy = s1 ? c1y : c0y; cz = s1 ? c1z : c0z; nn = s1 ? n1 : n0
    s2 = n2 > nn
    cx = s2 ? c2x : cx; cy = s2 ? c2y : cy; cz = s2 ? c2z : cz; nn = s2 ? n2 : nn
    nrm = sqrt(nn)
    ok = (nrm > 1e-300) & !iso
    inv = 1.0 / (ok ? nrm : one_)
    vx = ok ? cx * inv : one_
    vy = ok ? cy * inv : zero_
    vz = ok ? cz * inv : zero_
    # ---- orthonormal complement (u, w) of v, 2x2 Jacobi rotation in span(u, w)
    ax = abs(vx); ay = abs(vy); az = abs(vz)
    k1 = ay < ax
    k2 = az < (k1 ? ay : ax)
    ex = (k1 | k2) ? zero_ : one_
    ey = (k1 & !k2) ? one_ : zero_
    ez = k2 ? one_ : zero_
    vk = ex * vx + ey * vy + ez * vz
    ux = ex - vk * vx; uy = ey - vk * vy; uz = ez - vk * vz
    nu = sqrt(ux * ux + uy * uy + uz * uz)
    oku = nu > 1e-300
    iu = 1.0 / (oku ? nu : one_)
    ux = oku ? ux * iu : zero_; uy = oku ? uy * iu : zero_; uz = oku ? uz * iu : zero_
    wx = vy * uz - vz * uy; wy = vz * ux - vx * uz; wz = vx * uy - vy * ux
    # Rayleigh quotients of A (eigenvalues; accurate for spectra over orders of magnitude) and of the deviator A - m I
    # (rotation angle, and x = lambda - m for log near I)
    Aux = a0 * ux + a3 * uy + a4 * uz; Auy = a3 * ux + a1 * uy + a5 * uz; Auz = a4 * ux + a5 * uy + a2 * uz
    Awx = a0 * wx + a3 * wy + a4 * wz; Awy = a3 * wx + a1 * wy + a5 * wz; Awz = a4 * wx + a5 * wy + a2 * wz
    Avx = a0 * vx + a3 * vy + a4 * vz; Avy = a3 * vx + a1 * vy + a5 * vz; Avz = a4 * vx + a5 * vy + a2 * vz
    Dux = d0 * ux + a3 * uy + a4 * uz; Duy = a3 * ux + d1 * uy + a5 * uz; Duz = a4 * ux + a5 * uy + d2 * uz
    Dwx = d0 * wx + a3 * wy + a4 * wz; Dwy = a3 * wx + d1 * wy + a5 * wz; Dwz = a4 * wx + a5 * wy + d2 * wz
    b11 = ux * Dux + uy * Duy + uz * Duz
    b12 = ux * Dwx + uy * Dwy + uz * Dwz
    b22 = wx * Dwx + wy * Dwy + wz * Dwz
    a11 = ux * Aux + uy * Auy + uz * Auz
    a12 = ux * Awx + uy * Awy + uz * Awz
    a22 = wx * Awx + wy * Awy + wz * Awz
    lv = vx * Avx + vy * Avy + vz * Avz
    # theta = atan2(y, x)/2 in (-pi/2, pi/2]: cos = sqrt((r + x)/2r) >= 0, sin = y/(2 r cos) (x >= 0, else swapped)
    jx = b11 - b22                                   # angle from the deviator
    jy = 2.0 * b12
    jr = sqrt(jx * jx + jy * jy)
    jz = jr <= 0                                     # jr == 0 (jr >= 0 or NaN)
    jpos = jx >= 0
    jrs = jz ? one_ : jr
    big = sqrt((jrs + abs(jx)) / (2.0 * jrs))
    sml = jy / (2.0 * jrs * big)
    sgn = ((jy > 0) | ((abs(jy) <= 0) & !signbit(jy))) ? one_ : -one_
    cs = jz ? one_ : (jpos ? big : sgn * sml)
    sn = jz ? zero_ : (jpos ? sml : sgn * big)
    lam = (lv * sig, (cs * cs * a11 + 2.0 * cs * sn * a12 + sn * sn * a22) * sig,
           (sn * sn * a11 - 2.0 * cs * sn * a12 + cs * cs * a22) * sig)
    V = (vx, vy, vz, cs * ux + sn * wx, cs * uy + sn * wy, cs * uz + sn * wz,
         cs * wx - sn * ux, cs * wy - sn * uy, cs * wz - sn * uz)        # V[x, i] = V[_ix(x, i)]
    # ---- F = V diag(f) V^T
    vk_ = Val(K)
    # log: log(m sig) + log1p(x/m), the rounding of m cancels against A - m I (near I)
    fl = K === SMF_LOG ? _log_near(lam, m, sig, d0, d1, d2, a3, a4, a5, vx, vy, vz, cs, sn, b11, b12, b22) :
         (_fun(vk_, lam[1], al), _fun(vk_, lam[2], al), _fun(vk_, lam[3], al))
    @inbounds for x in 1:3, y in x:3
        f = V[_ix(x, 1)] * fl[1] * V[_ix(y, 1)] + V[_ix(x, 2)] * fl[2] * V[_ix(y, 2)] +
            V[_ix(x, 3)] * fl[3] * V[_ix(y, 3)]
        _setF!(F, _vidx(x, y), f)
    end
    J === nothing && return nothing
    D01 = _divdiff(vk_, lam[1], lam[2], fl[1], fl[2], al)
    D02 = _divdiff(vk_, lam[1], lam[3], fl[1], fl[3], al)
    D12 = _divdiff(vk_, lam[2], lam[3], fl[2], fl[3], al)
    D = (_deriv(vk_, lam[1], fl[1], al), D01, D02,
         D01, _deriv(vk_, lam[2], fl[2], al), D12,
         D02, D12, _deriv(vk_, lam[3], fl[3], al))                     # symmetric, D[i, j] = D[_ix(i, j)]
    _first!(J, V, D)
    H === nothing && return nothing
    d2(i, j, m) = _divdiff2(vk_, lam[i], lam[j], lam[m], fl[i], fl[j], fl[m],
                            D[_ix(i, j)], D[_ix(i, m)], D[_ix(j, m)], al, c)
    D2u = (d2(1, 1, 1), d2(1, 1, 2), d2(1, 1, 3), d2(1, 2, 2), d2(1, 2, 3),
           d2(1, 3, 3), d2(2, 2, 2), d2(2, 2, 3), d2(2, 3, 3), d2(3, 3, 3))  # order as TRI in the C kernel
    _second!(H, V, D2u)
    return nothing
end

# ---- the two output layouts, chosen by the dimensions of the output arrays; r, k, p, q: Voigt indices
#      Voigt:   F[r] (6),    J[r, k] = dF_r/dv_k (6x6),         H[r, p, q] = d2F_r/dv_p dv_q (6x6x6)
#      matrix:  F[a, b],     J[a, b, c, d] = dF_ab/dA_cd,       H[a, b, c, d, e, f] = d2F_ab/dA_cd dA_ef: derivatives of
#               A -> f((A + A^T)/2), i.e. w.r.t. an off-diagonal entry half the Voigt derivative
@inline _vidx(x, y) = x == y ? x : x + y + 1                     # Voigt index of (x, y), x <= y
@inline _half(t, c, d) = c == d ? t : t / 2
@inline _setF!(F::AbstractVector, r, f) = (@inbounds F[r] = f; nothing)
@inline function _setF!(F::AbstractMatrix, r, f)
    @inbounds F[VP[r], VQ[r]] = f
    @inbounds F[VQ[r], VP[r]] = f
    return nothing
end
@inline _setJ!(J::AbstractMatrix, r, k, t) = (@inbounds J[r, k] = t; nothing)
@inline function _setJ!(J::AbstractArray{<:Any,4}, r, k, t)
    a, b, c, d = VP[r], VQ[r], VP[k], VQ[k]
    tk = _half(t, c, d)
    @inbounds for (x, y) in ((a, b), (b, a)), (u, w) in ((c, d), (d, c))
        J[x, y, u, w] = tk
    end
    return nothing
end
@inline function _setH!(H::AbstractArray{<:Any,3}, r, p, q, h)
    @inbounds H[r, p, q] = h
    @inbounds H[r, q, p] = h
    return nothing
end
@inline function _setH!(H::AbstractArray{<:Any,6}, r, p, q, h)
    a, b, c, d, e, f = VP[r], VQ[r], VP[p], VQ[p], VP[q], VQ[q]
    hw = _half(_half(h, c, d), e, f)
    @inbounds for (x, y) in ((a, b), (b, a)), (u, w) in ((c, d), (d, c)), (s, t) in ((e, f), (f, e))
        H[x, y, u, w, s, t] = hw
        H[x, y, s, t, u, w] = hw
    end
    return nothing
end

# ---- J and H: straight-line code generated at load time (the C kernel fully unrolls these loops),
#      V[x, i], D[i, j] as scalars V_x_i, D_i_j
_s(parts...) = Symbol(join(parts, "_"))
function _unpack!(ex, name, tup, n)
    for i in 1:n, j in 1:n
        push!(ex.args, :($(_s(name, i, j)) = $tup[$(_ix(i, j))]))
    end
end

# Daleckii-Krein: J_ab,pp = C_abpp, J_ab,pq = C_abpq + C_abqp, C_abpq = sum_i P_i[a,p] Q_i[b,q],
# P_i = v_i v_i^T, Q_i = sum_j D_ij P_j, D_ij = f[lam_i, lam_j]
function _first_code()
    ex = Expr(:block)
    _unpack!(ex, :V, :V, 3)
    _unpack!(ex, :D, :D, 3)
    for i in 1:3, x in 1:3, y in 1:3
        push!(ex.args, :($(_s(:P, i, x, y)) = $(_s(:V, x, i)) * $(_s(:V, y, i))))
    end
    for i in 1:3, x in 1:3, y in 1:3
        push!(ex.args, :($(_s(:Q, i, x, y)) = $(_s(:D, i, 1)) * $(_s(:P, 1, x, y)) + $(_s(:D, i, 2)) * $(_s(:P, 2, x, y)) +
                                              $(_s(:D, i, 3)) * $(_s(:P, 3, x, y))))
    end
    for x in 1:3, y in x:3, k in 1:6
        p, q = VP[k], VQ[k]
        sk = :($(_s(:P, 1, x, p)) * $(_s(:Q, 1, y, q)) + $(_s(:P, 2, x, p)) * $(_s(:Q, 2, y, q)) +
               $(_s(:P, 3, x, p)) * $(_s(:Q, 3, y, q)))
        if p != q
            sk = :($sk + ($(_s(:P, 1, x, q)) * $(_s(:Q, 1, y, p)) + $(_s(:P, 2, x, q)) * $(_s(:Q, 2, y, p)) +
                          $(_s(:P, 3, x, q)) * $(_s(:Q, 3, y, p))))
        end
        t = _s(:J, x, y, k)
        push!(ex.args, :($t = $sk), :(_setJ!(J, $(_vidx(x, y)), $k, $t)))
    end
    return ex
end

# second order: H^pq = V S^pq V^T, S^pq = X + X^T, X_ab = sum_m D2_amb Et^p_am Et^q_mb, Et^p = V^T E_p V
function _second_code()
    ex = Expr(:block)
    _unpack!(ex, :V, :V, 3)
    for t in 1:10
        push!(ex.args, :($(_s(:D2, t)) = D2u[$t]))
    end
    d2(i, j, m) = _s(:D2, _t3(i, j, m))
    et(p, a, m) = _s(:Et, p, min(a, m), max(a, m))
    for p in 1:6, a in 1:3, m in a:3
        k, l = VP[p], VQ[p]
        e = :($(_s(:V, k, a)) * $(_s(:V, l, m)))
        k != l && (e = :($e + $(_s(:V, l, a)) * $(_s(:V, k, m))))
        push!(ex.args, :($(et(p, a, m)) = $e))
    end
    for p in 1:6, q in p:6
        S(a, b) = _s(:S, p, q, min(a, b), max(a, b))
        for a in 1:3, b in a:3
            xab = :($(d2(a, 1, b)) * $(et(p, a, 1)) * $(et(q, 1, b)) + $(d2(a, 2, b)) * $(et(p, a, 2)) * $(et(q, 2, b)) +
                    $(d2(a, 3, b)) * $(et(p, a, 3)) * $(et(q, 3, b)))
            xba = :($(d2(b, 1, a)) * $(et(p, b, 1)) * $(et(q, 1, a)) + $(d2(b, 2, a)) * $(et(p, b, 2)) * $(et(q, 2, a)) +
                    $(d2(b, 3, a)) * $(et(p, b, 3)) * $(et(q, 3, a)))
            push!(ex.args, :($(S(a, b)) = $xab + $xba))
        end
        for a in 1:3, j in 1:3         # W = S V^T
            push!(ex.args, :($(_s(:W, p, q, a, j)) = $(S(a, 1)) * $(_s(:V, j, 1)) + $(S(a, 2)) * $(_s(:V, j, 2)) +
                                                    $(S(a, 3)) * $(_s(:V, j, 3))))
        end
        for i in 1:3, j in i:3         # V W
            h = _s(:h, p, q, i, j)
            push!(ex.args, :($h = $(_s(:V, i, 1)) * $(_s(:W, p, q, 1, j)) + $(_s(:V, i, 2)) * $(_s(:W, p, q, 2, j)) +
                                  $(_s(:V, i, 3)) * $(_s(:W, p, q, 3, j))))
            push!(ex.args, :(_setH!(H, $(_vidx(i, j)), $p, $q, $h)))
        end
    end
    return ex
end

@eval @inline function _first!(J, V, D)
    @inbounds $(_first_code())
    return nothing
end

@eval @inline function _second!(H, V, D2u)
    @inbounds $(_second_code())
    return nothing
end

@inline function _dispatch!(k::Kind, F, J, H, v, al, c)
    if k == SMF_EXP
        _eval3!(Val(SMF_EXP), F, J, H, v, al, c)
    elseif k == SMF_LOG
        _eval3!(Val(SMF_LOG), F, J, H, v, al, c)
    elseif k == SMF_POW
        _eval3!(Val(SMF_POW), F, J, H, v, al, c)
    else
        _eval3!(Val(SMF_SQRT), F, J, H, v, al, c)
    end
    return nothing
end

const _NOCOEF = ntuple(_ -> 0.0, Val(NTERMS))

function _check(J, H)
    H !== nothing && J === nothing && throw(ArgumentError("H requires J (pass a J array as well)"))
    return nothing
end

_chk(name, A, sz) = size(A) == sz || throw(DimensionMismatch("$name must have size $sz, got $(size(A))"))
function _chkout(F, J, H, sf, sj, sh)
    _chk("F", F, sf)
    J === nothing || _chk("J", J, sj)
    H === nothing || _chk("H", H, sh)
    return nothing
end

# ------------------------------------------------------------------ public API
"""
    smf(kind, A; alpha=nothing, order=0)

f(A) and its first and second derivative for symmetric 3x3 matrices, given as matrices or as Voigt vectors
`v = (A11, A22, A33, A12, A13, A23)`; the results have the same form. Returns `F` for `order = 0`, `(F, J)` for
`order = 1` and `(F, J, H)` for `order = 2`:

| input `A`                | `F`         | `J`                              | `H`                                    |
|:-------------------------|:------------|:---------------------------------|:---------------------------------------|
| matrix 3 x 3             | 3 x 3       | 3 x 3 x 3 x 3, `dF[a,b]/dA[c,d]` | 3 x 3 x 3 x 3 x 3 x 3, `d2F[a,b]/dA[c,d] dA[e,f]` |
| Voigt vector (6)         | 6           | 6 x 6, `dF[i]/dv[k]`             | 6 x 6 x 6, `d2F[i]/dv[p] dv[q]`         |

`n` matrices: `A` of size (3, 3, n) or (6, n), with a trailing dimension `n` in all results.
The matrix derivatives are those of `A -> f((A + A^T)/2)` and have the symmetries of the index pairs
(`J[a,b,c,d] = J[b,a,c,d] = J[a,b,d,c] = J[c,d,a,b]`); the derivative w.r.t. an off-diagonal Voigt component (e.g.
`v[4] = A12`) changes A12 and A21 together. Matrices must be symmetric: `ArgumentError` if
`|A[i,j] - A[j,i]| > 1e-8 max|A|`.

`kind`: `:exp`, `:log`, `:sqrt` or `:pow` (or the codes 0 exp, 1 log, 2 pow, 3 sqrt); `alpha` is required for pow
(> 0). log needs A positive definite, sqrt/pow positive semidefinite (round-off below 0 is clamped to 0); there
are no domain checks, invalid input gives NaN/Inf as in the C library. `order` may also be `Val(0)`, `Val(1)`,
`Val(2)`. Any `eltype(A) <: Real` works (e.g. `ForwardDiff.Dual`); the computation runs in
`promote_type(float(eltype(A)), Float64)`.

```julia
F, J, H = smf(:log, [2.0 0.1 0.2; 0.1 3.0 0.3; 0.2 0.3 4.0]; order = 2)
```
"""
function smf(kind, A::Union{AbstractArray,NTuple{6,Real}}; alpha::Union{Nothing,Real} = nothing, order = 0)
    k = tokind(kind)
    if alpha === nothing
        k == SMF_POW && throw(ArgumentError("alpha is required for pow"))
        alpha = 0.0
    end
    o = order isa Val ? order : Val(Int(order))
    return _smf(k, A, alpha, o)
end

function _smf(k::Kind, A, alpha, ::Val{N}) where {N}
    N in (0, 1, 2) || throw(ArgumentError("order must be 0, 1 or 2, got $N"))
    T = _wtype(eltype(A))
    mat, n = _layout(A)
    tail = n === nothing ? () : (n,)
    F = Array{T}(undef, (mat ? (3, 3) : (6,))..., tail...)
    J = N >= 1 ? Array{T}(undef, (mat ? (3, 3, 3, 3) : (6, 6))..., tail...) : nothing
    H = N >= 2 ? Array{T}(undef, (mat ? (3, 3, 3, 3, 3, 3) : (6, 6, 6))..., tail...) : nothing
    smf!(F, J, H, k, A; alpha = alpha)
    return N == 0 ? F : (N == 1 ? (F, J) : (F, J, H))
end

# (matrix input?, n or nothing for one matrix) from the size of the input
function _layout(A)
    A isa Tuple && return (false, nothing)
    nd, sz = ndims(A), size(A)
    nd == 1 && sz[1] == 6 && return (false, nothing)
    nd == 2 && sz == (3, 3) && return (true, nothing)
    nd == 2 && sz[1] == 6 && return (false, sz[2])
    nd == 3 && sz[1] == 3 && sz[2] == 3 && return (true, sz[3])
    throw(DimensionMismatch("expected a matrix 3 x 3, 3 x 3 x n, a Voigt vector (6) or 6 x n, got size $sz"))
end

"""
    smf!(F, J, H, kind, A; alpha=0.0) -> F

In-place version of [`smf`](@ref): `F`, `J` (or `nothing`) and `H` (or `nothing`; needs `J`) of the sizes listed
there are overwritten. For one matrix (Float64, `Array` outputs) it does not allocate. For `n` matrices the series
coefficients are computed once per call, and the work runs multithreaded (`Threads.@threads :dynamic`) over chunks
of 1024 matrices (256 with H), as the C library does with OpenMP; start Julia with `-t N` to use N threads.
Throws `ArgumentError` for an unknown kind, alpha <= 0 with pow, H without J or a non-symmetric matrix, and
`DimensionMismatch` for wrong sizes.
"""
function smf!(F::AbstractArray, J::Union{Nothing,AbstractArray}, H::Union{Nothing,AbstractArray}, kind,
              A::Union{AbstractArray,NTuple{6,Real}}; alpha::Real = 0.0)
    k = tokind(kind)
    al = _alpha(k, alpha)
    _check(J, H)
    A isa AbstractArray && Base.require_one_based_indexing(A)
    Base.require_one_based_indexing(F)
    J === nothing || Base.require_one_based_indexing(J)
    H === nothing || Base.require_one_based_indexing(H)
    mat, n = _layout(A)
    c = H === nothing ? _NOCOEF : _dd2_coef(k, al)
    if n === nothing                                 # (separate branches: concrete sizes, no allocation)
        if mat
            _chkout(F, J, H, (3, 3), (3, 3, 3, 3), (3, 3, 3, 3, 3, 3))
            _dispatch!(k, F, J, H, _voigt_sym(A), al, c)
        else
            _chkout(F, J, H, (6,), (6, 6), (6, 6, 6))
            _dispatch!(k, F, J, H, A, al, c)
        end
    elseif mat
        _chkout(F, J, H, (3, 3, n), (3, 3, 3, 3, n), (3, 3, 3, 3, 3, 3, n))
        for i in 1:n                                 # all matrices are checked before any result is written
            _voigt_sym(view(A, :, :, i), i)
        end
        _batch!(k, F, J, H, A, Val(true), al, c)
    else
        _chkout(F, J, H, (6, n), (6, 6, n), (6, 6, 6, n))
        _batch!(k, F, J, H, A, Val(false), al, c)
    end
    return F
end

# Voigt vector (tuple) of the symmetric part of a 3 x 3 matrix; ArgumentError if it is not symmetric
const SYM_TOL = 1e-8
@inline function _voigt_sym(A::AbstractMatrix, i = nothing)
    @inbounds begin
        tol = SYM_TOL * maximum(abs, A)
        if abs(A[1, 2] - A[2, 1]) > tol || abs(A[1, 3] - A[3, 1]) > tol || abs(A[2, 3] - A[3, 2]) > tol
            throw(ArgumentError("matrix $(i === nothing ? "" : "$i ")is not symmetric (|A[i,j] - A[j,i]| > 1e-8 max|A|)"))
        end
        T = float(eltype(A))
        h = T(1) / 2
        return (T(A[1, 1]), T(A[2, 2]), T(A[3, 3]), h * (A[1, 2] + A[2, 1]), h * (A[1, 3] + A[3, 1]),
                h * (A[2, 3] + A[3, 2]))
    end
end

# view of slice i along the last dimension
@inline _last(X::AbstractArray{<:Any,N}, i) where {N} = view(X, ntuple(_ -> Colon(), Val(N - 1))..., i)

function _range!(vk::Val, F, J, H, A, ::Val{M}, al, c, lo, hi) where {M}
    @inbounds for i in lo:hi
        Ji = J === nothing ? nothing : _last(J, i)
        Hi = H === nothing ? nothing : _last(H, i)
        v = M ? _voigt_sym(_last(A, i)) : _last(A, i)
        _eval3!(vk, _last(F, i), Ji, Hi, v, al, c)
    end
    return nothing
end

function _batch!(k::Kind, F, J, H, A, mat::Val, al, c)
    if k == SMF_EXP
        _batch!(Val(SMF_EXP), F, J, H, A, mat, al, c)
    elseif k == SMF_LOG
        _batch!(Val(SMF_LOG), F, J, H, A, mat, al, c)
    elseif k == SMF_POW
        _batch!(Val(SMF_POW), F, J, H, A, mat, al, c)
    else
        _batch!(Val(SMF_SQRT), F, J, H, A, mat, al, c)
    end
end

function _batch!(vk::Val, F, J, H, A, mat::Val, al, c)
    n = size(A, ndims(A))
    chunk = H === nothing ? 1024 : 256
    nchunks = cld(n, chunk)
    if nchunks <= 1 || Threads.nthreads() == 1
        _range!(vk, F, J, H, A, mat, al, c, 1, n)
    else
        Threads.@threads :dynamic for ic in 1:nchunks
            lo = (ic - 1) * chunk + 1
            _range!(vk, F, J, H, A, mat, al, c, lo, min(n, lo + chunk - 1))
        end
    end
    return nothing
end

include("gmf.jl")

end # module
