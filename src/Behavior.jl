module Behavior

export i_dt, i_dt_wrapper, identify_lookedAtTargets, plot_saccades, plot_fixations

using DataFrames
using MAT
using JLD
using Dates
using LinearAlgebra
using DSP
using Plots
using Statistics
using KernelDensity
using StatsBase
using Images

using ..MyUtils


### functions


#################################################################################
#################################################################################


function i_dt(
    xy_o::Array{Array{Float64,1},1},
    sf,
    fix_duration_threshold_sec,
    fix_dispersio00n_threshold,
    saccade_velocity_threshold,
    saccade_duration_threshold
    )

    ###########
    #
    # Implements I-DT algorithm to detect fixations, sacades
    # extended to identify and interpolate blinks
    #
    # INPUT
    # fix_duration_threshold_sec: in sec
    # fix_dispersion_threshold: can be left empty then it is inferred from central fixation period in beginning of trials
    # saccade_velocity_threshold: not normalized to sampling rate!
    # saccade_duration_threshold: in sec
    #
    # OUTPUT
    #   xy_: eye trace with blinks interpolated
    #   sac_idx:
    #
    #
    #
    ###########

    if size(xy_o[1],1) > 1000 # if number of samples is lower empty arrays are returned

        xy = deepcopy(xy_o)
        [deleteat!(xy[ii], findall(isnan.(xy[ii]))) for ii in 1:length(xy)] # remove NaNs
        xy_ = deepcopy(xy) # to be returned with blinks interpolated
        xy_idxs = [collect(1:length(xy[1])), collect(1:length(xy[2]))] # array with corresponding indices
        fix_window_length = Int(fix_duration_threshold_sec * sf)

        if fix_dispersion_threshold == []

            fix_dispersion_threshold = maximum(xy[1][1:499])-minimum(xy[1][1:499]) + maximum(xy[2][1:499])-minimum(xy[2][1:499]) # deg

        end

        #
        sac_idx = Array{Array{Int,1}, 1}(undef,0)#[]
        fix_idx = Array{Array{Int,1}, 1}(undef,0)#[]
        blink_idx = Array{Array{Int,1}, 1}(undef,0)#[]

        ### I-DT loop:
        while !isempty(xy_idxs[1])



            if length(xy_idxs[1]) > fix_window_length # check if length(xy) > fix_window_length (lower duration threshold of saccades)
                current_window_idxs = collect(1:fix_window_length)
            else
                # @show length(xy_idxs[1])
                current_window_idxs = collect(1:length(xy_idxs[1])) # if length(xy) > fix_window_length
            end

            # println(current_window_idxs)

            current_window_vals = [xy[ii][current_window_idxs] for ii in 1:2]
            wd = (maximum(current_window_vals[1])-minimum(current_window_vals[1])) + (maximum(current_window_vals[2])-minimum(current_window_vals[2]))

            # if dispersion in current window is smaller than threshold
            if wd <= fix_dispersion_threshold

                # while this is the case and window does not exceed amount of data points add next point to window
                while wd<=fix_dispersion_threshold && length(current_window_idxs)<length(xy[1])

                    ### append to indices of current window
                    current_window_idxs = append!( current_window_idxs, maximum(current_window_idxs)+1 )
                    ### append to values of current widnow
                    current_window_vals[1] = xy[1][current_window_idxs]
                    current_window_vals[2] = xy[2][current_window_idxs]
                    wd = (maximum(current_window_vals[1])-minimum(current_window_vals[1])) + (maximum(current_window_vals[2])-minimum(current_window_vals[2]))

                end

                ### add fixation to array and delete window from data
                # push!( fix_idx, [xy_idxs[ii][current_window_idxs] for ii in 1:2] )
                push!( fix_idx, xy_idxs[1][current_window_idxs] )
                [deleteat!( xy[ii], current_window_idxs ) for ii in 1:length(xy)]
                [deleteat!( xy_idxs[ii], current_window_idxs ) for ii in 1:length(xy_idxs)]

            else # if dispersion in current window is greater than threshold delete first data point

                [deleteat!( xy[ii], 1 ) for ii in 1:length(xy)]
                [deleteat!( xy_idxs[ii], 1 ) for ii in 1:length(xy_idxs)]

            end

        end

        ### get saccade-like indices, i.e. those not identified as fixations
        for (idx,fixs) in enumerate(fix_idx[1:end-1])
            #if (fix_idx[idx+1][1][1]-fixs[1][end]) > blink_velocity_threshold # blink velocity threshold
                # push!( sac_idx, fixs[1][end]:fix_idx[idx+1][1][1] )
                push!( sac_idx, fixs[end]:fix_idx[idx+1][1] )

            #end
        end

        ### filter blinks: saccades with velocities > threshold
        saccade_velocities = [diff(xy_[1][sac]) for sac in sac_idx]
        blinks = findall(maximum.(saccade_velocities) .> saccade_velocity_threshold)

        for bl in blinks

            push!( blink_idx, sac_idx[bl] )

        end
        # delete blinks from saccades collection
        deleteat!( sac_idx, blinks )
        deleteat!(sac_idx, findall(length.(sac_idx) .< saccade_duration_threshold*sf)) # remove implausible short saccades

        ### inerpolate blinks (linear)
        interpolated_blinks = []

        for bl in 1:length(blink_idx)

            push!(interpolated_blinks, range( xy_[1][minimum(blink_idx[bl])], xy_[1][maximum(blink_idx[bl])], length=length(blink_idx[bl]) ))

        end

        [xy_[dim][blink_idx[bl]] .= interpolated_blinks[bl] for bl in 1:length(blink_idx), dim=1:2 ] # dim is x,y dimensions of eyedata -> bot needed


        return xy_, fix_idx, sac_idx, blink_idx

    else
        return [], [], [], []
    end

end # i_dt


end # end Behavior.jl
