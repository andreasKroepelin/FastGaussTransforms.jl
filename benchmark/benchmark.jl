using FastGaussTransforms
let points = rand(1000000), values = rand(size(points))
    evaluate(fastgausstransform(points, values, 0.01), 0.0) # warm up JIT

    @time f = fastgausstransform(points, values, 0.01)
    @time [evaluate(f, x) for x in linspace(-10, 10, 100000)]
end
