module FastGaussTransforms

export FastGaussTransform, SlowGaussTransform

struct FastGaussTransform{T <: Real}
    centers::Vector{T} # In general, an array of points
    coefficients::Array{T, 2} # In general, a rank-d tensor for every point
    h::T # sqrt(2)*std
    ry::Int # neighbor radius used in evaluation
end

# Relative error is bounded above by
#
# (2*rx*ry)^order/factorial(order) + exp(-ry^2)
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

function FastGaussTransform(
    xs,
    qs,
    std;
    rtol = eps(promote_type(eltype(xs), eltype(qs))),
)
    T = promote_type(eltype(xs), eltype(qs))
    rx, ry, order = errorconstants(rtol)
    h = convert(T, sqrt(2)*std)
    xmin, xmax = extrema(xs)
    range = xmax - xmin
    centers = Vector{T}(xmin:(2 * h * rx):(xmax + 2 * h * rx))
    xmin, xmax = extrema(centers)
    range = xmax - xmin
    ncenters = length(centers)
    coefficients = zeros(T, order + 1, ncenters)
    for (x, q) in zip(xs, qs)
        k = ncenters == 1 ? 1 : 1 + round(Int, (x - xmin)/(2*h*rx))
        center = centers[k]
        t = (x-center)/h
        c = q*exp(-t^2)
        z = one(T)
        coefficients[1, k] += c
        for n in 1:order
            # nth term is q e^(-t^2) 2^n/n! t^n
            z *= 2/n*t
            coefficients[n + 1, k] += c*z
        end
    end
    return FastGaussTransform(centers, coefficients, h, ry)
end

function neighborindices(f::FastGaussTransform, x)
    centers = f.centers
    ncenters = length(centers)
    range = centers[end] - centers[1]
    imin = (x-f.ry*f.h-centers[1])/range*(ncenters-1) + 1
    imax = (x+f.ry*f.h-centers[1])/range*(ncenters-1) + 1
    return max(floor(Int, imin), 1):min(ceil(Int, imax), ncenters)
end

function (f::FastGaussTransform{T})(x) where {T}
    g = zero(T)
    for k in neighborindices(f, x)
        center = f.centers[k]
        t = (x - center)/f.h
        s = f.coefficients[end, k]
        # Horner's method
        for n in (size(f.coefficients, 1) - 1):-1:1
            s = f.coefficients[n, k] + t*s
        end
        g += exp(-t^2)*s
    end
    return g
end

# Dummy type used for comparison that just stores points directly,
# and then performs naive summation
struct SlowGaussTransform{T <: Real}
    xs::Vector{T}
    qs::Vector{T}
    h2::T

    function SlowGaussTransform(xs, qs, std)
        T = promote_type(eltype(xs), eltype(qs))
        new{T}(xs, qs, 2*std^2)
    end
end

function (f::SlowGaussTransform{T})(x) where {T}
    g = zero(T)
    for (xp, q) in zip(f.xs, f.qs)
        g += q*exp(-(x-xp)^2/f.h2)
    end
    return g
end

end # module
