module FastGaussTransforms

using Distances
using NearestNeighbors
using Accessors
using StaticArrays
using LinearAlgebra
import LinearAlgebra: norm_sqr

export FastGaussTransform, SlowGaussTransform

include("farthest_point_clustering.jl")

struct FastGaussTransform{T <: Real, N, Tree <: KDTree{SVector{N, T}}}
    tree::Tree
    coefficients::Matrix{T}
    h::T # sqrt(2)*std
    ry::Int # neighbor radius used in evaluation
end

# Relative error is bounded above by
#
# sum(qs) * ((2*rx*ry)^order/factorial(order) + exp(-ry^2))
#
# Choose ry=ceil(Int, sqrt(-log(rtol))), rx=0.5, and
# find order s.t. the error bound is less than rtol
function errorconstants(rtol::T) where {T}
    half = one(T)/2
    lrtol = log(rtol)
    rx = half
    ry = ceil(Int, sqrt(-log(rtol/2)))
    order = 0
    # numerical prefactor here is determined empirically. Theory says it should be
    # 2, but in practice it appears that a smaller number of terms is sufficient.
    c = half*rx*ry
    error = one(T)
    while rtol<error
        order += 1
        error *= c/order
    end
    return rx, ry, order
end

num_terms(order, dim) = binomial(order + dim, order)

function graded_lexicographic_monomials!(monomials, x)
    monomials[1] = one(eltype(monomials))
    starts = map(_ -> 1, x)
    stop = curr_idx = 1
    while stop < lastindex(monomials)
        stop = curr_idx
        for i in eachindex(x)
            from_range = starts[i]:stop
            lr = length(from_range)
            to_range = curr_idx .+ (1:lr)
            last(to_range) > lastindex(monomials) && return
            @views monomials[to_range] .= monomials[from_range] .* x[i]
            @reset starts[i] = first(to_range)
            curr_idx += lr
        end
    end
end

function graded_lexicographic_monomials(x, order::Integer)
    monomials = Vector{eltype(x)}(undef, num_terms(order, length(x)))
    graded_lexicographic_monomials!(monomials, x)
    monomials
end

function graded_lexicographic_prefactors(T::Type{<: Real}, order::Integer, ::Val{dim}) where { dim}
    nt = num_terms(order, dim)
    alphas = Vector{NTuple{dim, Int}}(undef, nt)
    prefactors = Vector{T}(undef, nt)
    alphas[1] = ntuple(_ -> 0, Val(dim))
    prefactors[1] = one(T)
    
    starts = ntuple(_ -> 1, Val(dim))
    stop = curr_idx = 1
    while stop < nt
        stop = curr_idx
        for i in StaticArrays.SOneTo(dim)
            from_range = starts[i]:stop
            lr = length(from_range)
            to_range = curr_idx .+ (1:lr)
            last(to_range) > nt && return prefactors
            for (j, k) in zip(from_range, to_range)
                alpha = alphas[j]
                prefactors[k] = prefactors[j] * 2 / (alpha[i] + 1)
                alphas[k] = alpha .+ ntuple(==(i), Val(dim))
            end
            @reset starts[i] = first(to_range)
            curr_idx += lr
        end
    end
    prefactors
end

function FastGaussTransform(
    xs::AbstractVector{<: StaticVector{N, TX}},
    qs,
    std;
    rtol = eps(promote_type(TX, eltype(qs))),
) where {N,TX}
    T = promote_type(TX, eltype(qs))
    rx, ry, order = errorconstants(rtol)
    h = convert(T, sqrt(2)*std)
    tree = farthest_point_clustering(xs, std * rx)
    ncenters = length(tree.data)
    prefactors = graded_lexicographic_prefactors(T, order, Val(N))
    monomials = Vector{T}(undef, length(prefactors))
    coefficients = zeros(T, num_terms(order, N), ncenters)
    ks, dists = nn(tree, xs)
    for (x, q, k, d) in zip(xs, qs, ks, dists)
        center = tree.data[k]
        t = (x - center) / h
        graded_lexicographic_monomials!(monomials, t)
        c = q * exp(-(d / h)^2)
        @view(coefficients[:, k]) .+= c .* prefactors .* monomials
    end
    return FastGaussTransform(tree, coefficients, h, ry)
end

function neighborindices(f::FastGaussTransform, x)
    centers = f.centers
    ncenters = length(centers)
    range = centers[end] - centers[1]
    imin = (x-f.ry*f.h-centers[1])/range*(ncenters-1) + 1
    imax = (x+f.ry*f.h-centers[1])/range*(ncenters-1) + 1
    return max(floor(Int, imin), 1):min(ceil(Int, imax), ncenters)
end

function (f::FastGaussTransform{T})(y) where {T}
    g = zero(T)
    monomials = Vector{T}(undef, size(f.coefficients, 1))
    for k in inrange(f.tree, y, f.h * f.ry)
        center = f.tree.data[k]
        t = (y - center) / f.h
        graded_lexicographic_monomials!(monomials, t)
        s = dot(monomials, @view(f.coefficients[:, k]))
        g += exp(- norm_sqr(t)) * s
    end
    return g
end

# Dummy type used for comparison that just stores points directly,
# and then performs naive summation
struct SlowGaussTransform{T <: Real, N, X <: AbstractVector{<: SVector{N, T}}}
    xs::X
    qs::Vector{T}
    h2::T

    function SlowGaussTransform(xs, qs, std)
        T = promote_type(eltype(eltype(xs)), eltype(qs))
        new{T, length(eltype(xs)), typeof(xs)}(xs, qs, 2*std^2)
    end
end

function (f::SlowGaussTransform{T})(y) where {T}
    g = zero(T)
    for (x, q) in zip(f.xs, f.qs)
        g += q * exp(-sqeuclidean(x, y) / f.h2)
    end
    return g
end

end # module
