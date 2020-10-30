module MyUtils

export flatten, myColors, cartesian2matrix, cartesian2polar, polar2cartesian

using Plots

function flatten(x)
    return collect(Iterators.flatten(x))
end

function myColors()
    return distinguishable_colors(25, [parse(Colorant,c) for c in  ["green", "yellow"]])[[1, 3, 5, 8, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24]]
end


function cartesian2matrix(ci::AbstractArray)
    ### array of cartesian indices to matrix N x DIM
    dims = collect(1:length(ci[1]))
    sparse_idx = reduce(vcat, [[t[d] for d in dims]' for t in ci])
end


function cartesian2polar(x, y)
    ### cartesian coordinates to polar coordinates
    if size(x) == size(y)
        rphi = zeros(length(x), 2)
        rphi[:,1] = sqrt.(x.^2 .+ y.^2)
        rphi[:,2] = atan.(x ./ y)
        return rphi
    else
        @warn "Dimensions not matching"
    end
end

function polar2cartesian(r, phi)
    ###
    xy = zeros(length(r), 2)
    xy[:,1] = r.*cos.(phi)
    xy[:,2] = r.*sin.(phi)
    return xy
end



end
