using FastGaussTransforms
using FastGaussTransforms.StaticArrays
using FastGaussTransforms.Distances
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

make_data(n, d) = (
    xs = [randn(SVector{d}) for _ in 1:n],
    qs = rand(n),
    ys = [randn(SVector{d}) for _ in 1:n],
    std = .1,
)

function (@main)(args)
    n = 10^4
    d = 2
    @info "Running benchmarks with $n times $n data points in $d dimensions."
    data = make_data(n, d)

    fgt = FastGaussTransform(data.xs, data.qs, data.std; rtol = eps())
    @info "precomputed values" size(fgt.coefficients) minimum(abs, fgt.coefficients) fgt.tree fgt.ry

    @info "Fast Gauss Transform"
    display(@be data fast evals=2 samples=1 seconds=Inf)
    @info "Slow Gauss Transform"
    display(@be data slow evals=2 samples=1 seconds=Inf)
    @info "Naive"
    display(@be data naive evals=2 samples=1 seconds=Inf)

    return 0
end
