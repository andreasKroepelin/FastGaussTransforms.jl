function farthest_point_clustering(points, max_radius)
    centers = [first(points)]
    while length(centers) < length(points) ÷ 10
        dist, idx = findmax(points) do point
            minimum(centers) do center
                sqeuclidean(point, center)
            end
        end
        if dist < max_radius
            return KDTree(centers)
        end
        push!(centers, points[idx])
    end

    KDTree(centers)
end
