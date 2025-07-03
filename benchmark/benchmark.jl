using FastGaussTransforms
using FastGaussTransforms.StaticArrays
using FastGaussTransforms.Distances
using FastGaussTransforms.NearestNeighbors
using Chairmarks
using Profile
using PProf

function fast(data)
    (; xs, ys, qs, std) = data
    fgt = FastGaussTransform(xs, qs, std; rtol = eps())
    fgt(ys)
end

function slow(data)
    (; xs, ys, qs, std) = data
    sgt = SlowGaussTransform(xs, qs, std)
    [sgt(y) for y in ys]
end

function naive(data)
    (; xs, ys, qs, std) = data
    G = pairwise(sqeuclidean, ys, xs)
    c = -0.5 / std^2
    G .= exp.(c .* G)
    G * qs
end

function kdtree(data)
    (; xs, ys, qs, std) = data
    tree = KDTree(xs)
    c = -0.5 / std^2
    gs = Vector{typeof(std)}(undef, length(ys))
    idcs = Int[]
    for i in eachindex(ys, gs)
        y = ys[i]
        empty!(idcs)
        inrange!(idcs, tree, y, 3std)
        g = zero(eltype(gs))
        for j in idcs
            g += qs[j] * exp(c * sqeuclidean(xs[j], y))
        end
        gs[i] = g
    end
    gs
end

make_data(n, d) = (
    xs = [randn(SVector{d}) for _ in 1:n],
    qs = rand(n),
    ys = [randn(SVector{d}) for _ in 1:n],
    std = 1.,
)

function (@main)(args)
    n = 10^1
    d = 2
    data = make_data(n, d)
    @info "Running benchmarks." n d data.std

    fgt = FastGaussTransform(data.xs, data.qs, data.std; rtol = eps())
    @info "precomputed values" size(fgt.coefficients) minimum(abs, fgt.coefficients) fgt.tree fgt.ry

    @info "Fast Gauss Transform"
    display(@be data fast evals=2 samples=1 seconds=Inf)
    # @info "Slow Gauss Transform"
    # display(@be data slow evals=2 samples=1 seconds=Inf)
    @info "k-d tree"
    display(@be data kdtree evals=2 samples=1 seconds=Inf)
    @info "Naive"
    display(@be data naive evals=2 samples=1 seconds=Inf)

    return 0
end
