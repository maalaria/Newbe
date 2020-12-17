module MyUtils

export flatten, myColors, cartesian2matrix, cartesian2polar, polar2cartesian, append_sessions, convexHullCircum, foo

using Plots
using Polyhedra
import GLPK
lib = DefaultLibrary{Float64}(GLPK.Optimizer)

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


function append_sessions( rd_a, rd; which_sessions)

    #
    # INPUT
    # rd_a: empty DataFrame() !!!(required since recursive implementation)!!!
    # rd: Array{Newbe.Experiment.data_of_run,1}
    # which_sessions: array with indicies of sessions in rd to concatenate
    #
    # OUTPUT
    # rd_a: concatenated DataFrame()
    #

    if !isempty(which_sessions)
        rd_a = [rd_a; getfield( rd[which_sessions[1]], :trial_list)]
        which_sessions = which_sessions[2:end]
        append_sessions( rd_a, rd; which_sessions=which_sessions)
    else
        return rd_a
    end

end

function foo(x)
    print(x)
end

function convexHullCircum(xy::Array{Array{Float64,1},1})

    xy_ = copy(xy)
    [push!(xy_[ii], xy_[ii][1]) for ii in [1,2]]

    P1 = polyhedron(vrep(
            hcat(xy_[1], xy_[2])
            ), lib)
    removevredundancy!(P1)

    ext = [P1.vrep.V; [P1.vrep.V[1,1] P1.vrep.V[1,2]]]

    diffs = diff(ext, dims=1).^2 # compute the differences along x and y dimensions
    return [P1, sum(sqrt.(sum(diffs, dims=2)))] #return of the length of the hull circumference

end

end
