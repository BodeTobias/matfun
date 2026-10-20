using MatFun
using MatFun: SMF_EXP, SMF_LOG, SMF_POW, SMF_SQRT
using Test
using Printf
using Aqua

const CASES = normpath(joinpath(@__DIR__, "..", "..", "..", "testdata", "cases.txt"))

# one case per data line: kind alpha tol v[6] F[6] J[36] H[216] (Voigt layout), C row-major -> 1-based column-major
function read_cases(path)
    cases = NamedTuple[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, '#')) && continue
        x = parse.(Float64, split(s))
        length(x) == 267 || error("expected 267 numbers per line, got $(length(x))")
        kind = Int(x[1]); alpha = x[2]; tol = x[3]
        v = x[4:9]
        F = x[10:15]                                                      # F[i]
        J = permutedims(reshape(x[16:51], 6, 6), (2, 1))                  # J[6i+k]
        H = permutedims(reshape(x[52:267], 6, 6, 6), (3, 2, 1))           # H[(6i+p)*6+q]
        push!(cases, (; kind, alpha, tol, v, F, J, H))
    end
    return cases
end

relerr(x, ref) = maximum(abs.(x .- ref)) / maximum(abs.(ref))
label(c) = c.kind == 2 ? "pow $(c.alpha)" : ("exp", "log", "pow", "sqrt")[c.kind + 1]

const cases = read_cases(CASES)

const CASES_GENERAL = normpath(joinpath(@__DIR__, "..", "..", "..", "testdata", "cases_general.txt"))

# one case per data line: kind alpha tol A[9] F[9] J[81] H[729] (matrix layout, C row-major) -> F[a,b], J[a,b,c,d], ...
function read_cases_general(path)
    cases = NamedTuple[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, '#')) && continue
        x = parse.(Float64, split(s))
        length(x) == 831 || error("expected 831 numbers per line, got $(length(x))")
        kind = Int(x[1]); alpha = x[2]; tol = x[3]
        A = permutedims(reshape(x[4:12], 3, 3), (2, 1))
        F = permutedims(reshape(x[13:21], 3, 3), (2, 1))
        J = permutedims(reshape(x[22:102], 3, 3, 3, 3), (4, 3, 2, 1))
        H = permutedims(reshape(x[103:831], 3, 3, 3, 3, 3, 3), (6, 5, 4, 3, 2, 1))
        push!(cases, (; kind, alpha, tol, A, F, J, H))
    end
    return cases
end

const gcases = read_cases_general(CASES_GENERAL)
mat(v) = [v[1] v[4] v[5]; v[4] v[2] v[6]; v[5] v[6] v[3]]
const VI = [1 4 5; 4 2 6; 5 6 3]
const W = [a == b ? 1.0 : 0.5 for a in 1:3, b in 1:3]

# bytes allocated by one call after a warm-up call (concrete argument types, no splatting)
function alloc_single(F, J, H, kind, v, al)
    smf!(F, J, H, kind, v; alpha = al)
    return @allocated smf!(F, J, H, kind, v; alpha = al)
end
function alloc_batch(F, J, H, kind, V, al)
    smf!(F, J, H, kind, V; alpha = al)
    return @allocated smf!(F, J, H, kind, V; alpha = al)
end

@testset "MatFun" begin
    @testset "reference cases (testdata/cases.txt)" begin
        @test length(cases) >= 150
        # per kind: max error over the well-conditioned cases (tol 1e-12) and over the ill-conditioned ones (1e-11)
        maxerr = Dict{Tuple{String,Float64},Vector{Float64}}()
        order = String[]
        nfail = 0
        for c in cases
            F, J, H = smf(c.kind, c.v; alpha = c.alpha, order = 2)
            e = (relerr(F, c.F), relerr(J, c.J), relerr(H, c.H))
            ok = all(<=(c.tol), e)
            ok || (nfail += 1; @printf("FAIL %s v=%s errors F %.2e J %.2e H %.2e tol %.0e\n", label(c), c.v, e..., c.tol))
            @test e[1] <= c.tol
            @test e[2] <= c.tol
            @test e[3] <= c.tol
            l = label(c)
            l in order || push!(order, l)
            m = get!(() -> zeros(3), maxerr, (l, c.tol))
            m .= max.(m, e)
        end
        println("max relative error max|x-ref|/max|ref| ($(length(cases)) cases, $nfail outside tol):")
        for l in order, tol in sort(unique(c.tol for c in cases))
            n = count(c -> label(c) == l && c.tol == tol, cases)
            n == 0 && continue
            @printf("  %-8s tol %.0e (%2d cases)  F %.2e   J %.2e   H %.2e\n", l, tol, n, maxerr[(l, tol)]...)
        end
    end

    @testset "batch == single" begin
        for (kind, al) in ((SMF_EXP, 0.0), (SMF_LOG, 0.0), (SMF_SQRT, 0.0), (SMF_POW, 0.3), (SMF_POW, 1.7))
            sel = [c for c in cases if c.kind == Int(kind) && c.alpha == al]
            # repeat so that several chunks (and threads, if any) are used
            reps = 20
            V = reduce(hcat, [c.v for c in sel for _ in 1:reps])
            n = size(V, 2)
            F = zeros(6, n); J = zeros(6, 6, n); H = zeros(6, 6, 6, n)
            smf!(F, J, H, kind, V; alpha = al)
            F0 = zeros(6, n)
            smf!(F0, nothing, nothing, kind, V; alpha = al)
            same = true
            for i in 1:n
                Fs, Js, Hs = smf(kind, V[:, i]; alpha = al, order = 2)
                same &= Fs == F[:, i] && Js == J[:, :, i] && Hs == H[:, :, :, i] && F0[:, i] == Fs
            end
            @test same
            Fa, Ja, Ha = smf(kind, V; alpha = al == 0.0 ? nothing : al, order = 2)    # allocating, 6 x n input
            @test Fa == F && Ja == J && Ha == H
            A = reduce((x, y) -> cat(x, y; dims = 3), [mat(V[:, i]) for i in 1:n])   # 3 x 3 x n input
            Fm, Jm, Hm = smf(kind, A; alpha = al == 0.0 ? nothing : al, order = 2)
            @test Fm[:, :, n] == smf(kind, A[:, :, n]; alpha = al == 0.0 ? nothing : al)
            @test Hm[:, :, :, :, :, :, 1] == smf(kind, A[:, :, 1]; alpha = al == 0.0 ? nothing : al, order = 2)[3]
        end
        # n = 0 and a Float32 input
        @test smf!(zeros(6, 0), nothing, nothing, SMF_EXP, zeros(6, 0)) == zeros(6, 0)
        F32 = zeros(Float32, 6, 1)
        smf!(F32, nothing, nothing, :sqrt, Float32[2, 3, 4, 0.1, 0.2, 0.3][:, :])
        @test F32[:, 1] ≈ smf(:sqrt, [2, 3, 4, 0.1, 0.2, 0.3]) rtol = 1e-6
    end

    @testset "matrix layout" begin
        v = [2.0, 1.5, 1.0, 0.3, -0.2, 0.1]
        A = mat(v)
        F, J, H = smf(:log, v; order = 2)
        Fm, Jm, Hm = smf(:log, A; order = 2)
        # the Voigt results with dv/dA[c,d] = 1 (diagonal) or 1/2 (exact)
        @test Fm == [F[VI[a, b]] for a in 1:3, b in 1:3]
        @test Jm == [J[VI[a, b], VI[c, d]] * W[c, d] for a in 1:3, b in 1:3, c in 1:3, d in 1:3]
        @test Hm == [H[VI[a, b], VI[c, d], VI[e, f]] * W[c, d] * W[e, f]
                     for a in 1:3, b in 1:3, c in 1:3, d in 1:3, e in 1:3, f in 1:3]
        # symmetries of the index pairs
        sym(X, p) = maximum(abs.(X .- permutedims(X, p))) <= 1e-15 * maximum(abs.(X))
        @test sym(Jm, (2, 1, 3, 4)) && sym(Jm, (1, 2, 4, 3)) && sym(Jm, (3, 4, 1, 2))
        @test sym(Hm, (2, 1, 3, 4, 5, 6)) && sym(Hm, (1, 2, 4, 3, 5, 6)) && sym(Hm, (1, 2, 5, 6, 3, 4))
        # symmetric input only: round-off asymmetry is accepted, more is not (also in a batch, before any output)
        @test smf(:log, A + [0 1e-13 0; 0 0 0; 0 0 0]) ≈ Fm rtol = 1e-12
        N = A + [0 0.1 0; 0 0 0; 0 0 0]
        @test_throws ArgumentError smf(:log, N)
        @test_throws ArgumentError smf(:log, cat(A, N; dims = 3))
        @test_throws DimensionMismatch smf(:log, zeros(4, 4))
        F = zeros(3, 3); J = zeros(3, 3, 3, 3); H = zeros(3, 3, 3, 3, 3, 3)
        smf!(F, J, H, :log, A)
        @test (@allocated smf!(F, J, H, :log, A)) == 0
        @test F == Fm && J == Jm && H == Hm
    end

    @testset "kinds and alpha" begin
        v = [2.0, 3.0, 4.0, 0.1, 0.2, 0.3]
        A = mat(v)
        @test smf(:exp, v) == smf(SMF_EXP, v) == smf(0, v)
        @test smf(:sqrt, v) ≈ smf(:pow, v; alpha = 0.5) rtol = 1e-14
        @test smf(:pow, A; alpha = 1.0) ≈ A rtol = 1e-14
        @test smf(:exp, v; alpha = 7.0) == smf(:exp, v)          # alpha ignored for exp
        S = smf(:sqrt, A)
        @test S * S ≈ A rtol = 1e-14
        @test smf(:exp, smf(:log, v)) ≈ v rtol = 1e-13
    end

    @testset "argument errors" begin
        v = [2.0, 3.0, 4.0, 0.1, 0.2, 0.3]
        F = zeros(6); J = zeros(6, 6); H = zeros(6, 6, 6)
        @test_throws ArgumentError smf!(F, J, H, 4, v)                  # unknown kind
        @test_throws ArgumentError smf!(F, J, H, -1, v)
        @test_throws ArgumentError smf!(F, J, H, :cos, v)
        @test_throws ArgumentError smf!(F, J, H, SMF_POW, v; alpha = 0.0)
        @test_throws ArgumentError smf!(F, J, H, SMF_POW, v; alpha = -1.0)
        @test_throws ArgumentError smf!(F, J, H, SMF_POW, v; alpha = NaN)
        @test_throws ArgumentError smf!(F, nothing, H, SMF_EXP, v)      # H without J
        @test_throws ArgumentError smf(SMF_POW, v)                        # alpha required
        @test_throws ArgumentError smf(SMF_EXP, v; order = 3)
        @test_throws DimensionMismatch smf!(F, J, H, SMF_EXP, v[1:5])
        @test_throws DimensionMismatch smf!(zeros(3, 3), J, H, SMF_EXP, v)
        @test_throws DimensionMismatch smf!(F, zeros(6, 5), nothing, SMF_EXP, v)
        @test_throws DimensionMismatch smf!(F, J, zeros(6, 6, 5), SMF_EXP, v)
        @test_throws DimensionMismatch smf!(zeros(3, 3), zeros(3, 3, 6), nothing, SMF_EXP, mat(v))
        V = reshape(repeat(v, 4), 6, 4)
        @test_throws ArgumentError smf!(zeros(6, 4), nothing, zeros(6, 6, 6, 4), SMF_EXP, V)
        @test_throws ArgumentError smf!(zeros(6, 4), nothing, nothing, SMF_POW, V; alpha = 0.0)
        @test_throws DimensionMismatch smf!(zeros(6, 3), nothing, nothing, SMF_EXP, V)
        @test_throws DimensionMismatch smf!(zeros(6, 4), zeros(6, 6, 3), nothing, SMF_EXP, V)
        @test_throws DimensionMismatch smf!(zeros(6, 4), nothing, nothing, SMF_EXP, V[1:5, :])
    end

    @testset "NaN / Inf input gives NaN as in the C library (no exception)" begin
        for x in (NaN, Inf, -Inf), p in 1:6
            v = [2.0, 3.0, 4.0, 0.1, 0.2, 0.3]
            v[p] = x
            F, J, H = smf(:log, v; order = 2)
            @test all(isnan, F) && all(isnan, J) && all(isnan, H)
            V = repeat([2.0, 3.0, 4.0, 0.1, 0.2, 0.3], 1, 3)
            V[:, 2] = v
            F = smf(:sqrt, V)
            @test all(isnan, F[:, 2]) && all(isfinite, F[:, [1, 3]])
            @test all(isnan, smf(:sqrt, mat(v)))
        end
    end

    @testset "type stability and allocations (Float64)" begin
        v = [2.0, 3.0, 4.0, 0.1, 0.2, 0.3]
        F = zeros(6); J = zeros(6, 6); H = zeros(6, 6, 6)
        A = mat(v); Fm = zeros(3, 3); Jm = zeros(3, 3, 3, 3); Hm = zeros(3, 3, 3, 3, 3, 3)
        V = repeat(v, 1, 3000)
        Fb = zeros(6, 3000); Jb = zeros(6, 6, 3000); Hb = zeros(6, 6, 6, 3000)
        for (kind, al) in ((SMF_EXP, 0.0), (SMF_LOG, 0.0), (SMF_POW, 0.3), (SMF_POW, 1.7), (SMF_SQRT, 0.0))
            @test (@inferred smf!(F, J, H, kind, v; alpha = al)) === F
            @test (@inferred smf!(F, J, nothing, kind, v; alpha = al)) === F
            @test (@inferred smf!(F, nothing, nothing, kind, v; alpha = al)) === F
            @test alloc_single(F, J, H, kind, v, al) == 0
            @test alloc_single(F, J, nothing, kind, v, al) == 0
            @test alloc_single(F, nothing, nothing, kind, v, al) == 0
            @test (@inferred smf!(Fm, Jm, Hm, kind, A; alpha = al)) === Fm
            @test alloc_single(Fm, Jm, Hm, kind, A, al) == 0
            al2 = kind == SMF_POW ? al : nothing
            @inferred smf(kind, v; alpha = al2, order = Val(2))
            @inferred smf(kind, v; alpha = al2, order = Val(0))
            if Threads.nthreads() == 1      # with threads, @threads itself allocates its tasks
                @test alloc_batch(Fb, Jb, Hb, kind, V, al) == 0
            end
        end
        a = alloc_single(F, J, H, SMF_LOG, v, 0.0)
        println("allocations of smf!(F, J, H, ...) after warm-up: $a bytes; threads: $(Threads.nthreads())")
    end

    @testset "ForwardDiff: J = dF/dv, H = dJ/dv" begin
        fd = try
            @eval import ForwardDiff
            true
        catch err
            @warn "ForwardDiff not available (no test environment / registry?); skipping the ForwardDiff check" err
            false
        end
        if fd
            # random cases (well separated eigenvalues; at exactly multiple eigenvalues the eigenvectors, and so the
            # AD derivative of the decomposition, are not differentiable)
            # (the first 8 cases of each block of one function in cases.txt are random matrices: take the first two)
            firsts = [findfirst(c -> (c.kind, c.alpha) == ka, cases) for ka in unique((c.kind, c.alpha) for c in cases)]
            picks = cases[sort(vcat(firsts, firsts .+ 1))]
            maxe = zeros(2)
            for c in picks
                F, J, H = smf(c.kind, c.v; alpha = c.alpha, order = 2)
                kind, al = c.kind, c.kind == 2 ? c.alpha : nothing
                JF = ForwardDiff.jacobian(x -> smf(kind, x; alpha = al), c.v)                    # (6, 6)
                JJ = ForwardDiff.jacobian(x -> vec(smf(kind, x; alpha = al, order = 1)[2]), c.v) # (36, 6)
                e1 = relerr(JF, J)
                e2 = relerr(JJ, reshape(H, 36, 6))
                @test e1 <= 1e-11
                @test e2 <= 1e-10
                maxe .= max.(maxe, (e1, e2))
            end
            @printf("ForwardDiff (%d cases, ForwardDiff %s): max rel. diff dF/dv vs J %.2e, dJ/dv vs H %.2e\n",
                    length(picks), pkgversion(ForwardDiff), maxe...)
        end
    end

    @testset "gmf: reference cases (testdata/cases_general.txt)" begin
        @test length(gcases) >= 150
        maxerr = Dict{String,Vector{Float64}}()
        for c in gcases
            F, J, H = gmf(c.kind, c.A; alpha = c.alpha, order = 2)
            e = (relerr(F, c.F), relerr(J, c.J), relerr(H, c.H))
            all(<=(c.tol), e) || @printf("FAIL %s A=%s errors F %.2e J %.2e H %.2e tol %.0e\n", label(c), c.A, e..., c.tol)
            @test e[1] <= c.tol
            @test e[2] <= c.tol
            @test e[3] <= c.tol
            m = get!(() -> zeros(3), maxerr, label(c))
            m .= max.(m, e)
            @test gmf(c.kind, c.A; alpha = c.alpha) == F
        end
        println("gmf: max relative error max|x-ref|/max|ref| ($(length(gcases)) cases):")
        for (l, m) in sort(collect(maxerr))
            @printf("  %-8s F %.2e   J %.2e   H %.2e\n", l, m...)
        end
    end

    @testset "gmf: batch == single, domain, closed forms, errors" begin
        As = cat([c.A for c in gcases if c.kind == 1]...; dims = 3)
        As[:, :, 3] = [-1.0 0 0; 0 2 0; 0 0 3]                                 # real eigenvalue -1: NaN for log
        F, J, H = gmf(:log, As; order = 2)
        for i in axes(As, 3)
            Fi, Ji, Hi = gmf(:log, As[:, :, i]; order = 2)
            @test isequal(Fi, F[:, :, i]) && isequal(Ji, J[:, :, :, :, i]) && isequal(Hi, H[:, :, :, :, :, :, i])
            @test all(isnan, Fi) == (i == 3)
        end
        th = 0.9
        K = [0 -th 0; th 0 0; 0 0 0]
        R = [cos(th) -sin(th) 0; sin(th) cos(th) 0; 0 0 1]
        @test relerr(gmf(:exp, K), R) < 1e-15
        @test relerr(gmf(:log, R), K) < 1e-15
        N = [0 0.7 0; 0 0 0; 0 0 0]
        @test relerr(gmf(:exp, N), [1 0.7 0; 0 1 0; 0 0 1]) < 1e-15
        @test_throws ArgumentError gmf(:pow, R)
        @test_throws DimensionMismatch gmf(:exp, zeros(2, 2))
        @test_throws ArgumentError gmf!(zeros(3, 3), nothing, zeros(3, 3, 3, 3, 3, 3), :exp, R)
    end

    @testset "gmf: ForwardDiff, J = dF/dA, H = dJ/dA" begin
        fd = try
            @eval import ForwardDiff
            true
        catch err
            false
        end
        if fd
            # the first four cases of each function are random matrices (separated eigenvalues; at (nearly) multiple
            # eigenvalues the roots, and so the AD derivative of the algorithm, are not differentiable)
            firsts = [findfirst(c -> (c.kind, c.alpha) == ka, gcases) for ka in unique((c.kind, c.alpha) for c in gcases)]
            for c in gcases[sort(vcat(firsts, firsts .+ 1))]
                F, J, H = gmf(c.kind, c.A; alpha = c.alpha, order = 2)
                kind, al = c.kind, c.kind == 2 ? c.alpha : nothing
                JF = ForwardDiff.jacobian(X -> gmf(kind, X; alpha = al), c.A)
                JJ = ForwardDiff.jacobian(X -> vec(gmf(kind, X; alpha = al, order = 1)[2]), c.A)
                @test relerr(JF, reshape(J, 9, 9)) <= 1e-11
                @test relerr(JJ, reshape(H, 81, 9)) <= 1e-10
            end
        end
    end

    @testset "Aqua (package quality, as checked before registration)" begin
        Aqua.test_all(MatFun)
    end
end
