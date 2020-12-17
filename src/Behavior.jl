module Behavior

export i_dt, i_dt_wrapper, identify_fixatedTargets, plot_eye_events

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

function i_dt_wrapper(
    df,
    sf,
    fix_duration_threshold,
    fix_dispersion_threshold,
    saccade_velocity_threshold,
    saccade_duration_threshold;
    plot_saccades=false,
    plot_fixations=false,
    which_trials="all",
    how_ = "mean",
    which="Fixations_clean"
    )

    ####
    # functions takes a dataframe containg trial data of an experimental session as input
    # and returns the dataframe with filled :Fixsation and :Saccades columns
    ####

    for trl in range(1, size(df, 1), step=1)

        xy_, fix_idx, sac_idx, blink_idx = i_dt( df[!, "EyeXY"][trl], sf, fix_duration_threshold, fix_dispersion_threshold, saccade_velocity_threshold, saccade_duration_threshold);#, blink_velocity_threshold );
        df.Fixations[trl] = fix_idx
        df.Saccades[trl] = sac_idx

    end

end







function plot_eye_events(
    df;
    what="both",
    how_="mean",
    which_fixations="all_fixations",
    p=plot(legend=false) )

    if what == "both" || what == "saccades"
        ### plot saccades
        for (itrl,trls_saccades) in enumerate(df[:Saccades])
            for sac_idxs in trls_saccades
                xx = df[itrl,:EyeXY][1][sac_idxs]
                yy = df[itrl,:EyeXY][2][sac_idxs]
                plot!(xx,yy)
            end
        end
    end

    if which_fixations == "all_fixations"
        clean_fixations_bool = [[fix in [-1,0,1,2,3,4,5,6,7,8] for fix in el] for el in df.LookedAtTarget]
    elseif which_fixations == "clean_fixations"
        clean_fixations_bool = [[fix in [0,1,2,3,4,5,6,7,8] for fix in el] for el in df.LookedAtTarget]
    end

    if what == "both" || what == "fixations"
        for (itrl,trls_fix) in enumerate(df.Fixations)
            for (ifix, fix) in enumerate(trls_fix)
                if clean_fixations_bool[itrl][ifix]
                    scatter!( [mean( df.EyeXY[itrl][1][fix] )], [mean( df.EyeXY[itrl][2][fix] )] )
                end
            end
        end

        # for (itrl,trls_fixations) in enumerate(df[:Fixations])
        #     for fix_idxs in trls_fixations
        #         # print(df[itrl,:EyeXY][1][sacd])
        #         xx = df[itrl,:EyeXY][1][fix_idxs]
        #         yy = df[itrl,:EyeXY][2][fix_idxs]
        #         if how_ == "mean"
        #             scatter!([mean(xx)],[mean(yy)])
        #         elseif how_ == "raw"
        #             scatter!(xx,yy, markersize=0.5)
        #         end
        #     end
        # end
    end

    return p
end




#################################################################################
#################################################################################

function identify_fixatedTargets(
    df;
    nbins_length=150,
    nbins_angles=150,
    kde_bandwidth_length=1/3,
    kde_bandwidth_angle=1/15,
    plot_distributions=false,
    plot_only=false,
    target_rotation=22.5)

    #
    # also identifies Invalid trials, i.e. those that are inititated and Fritz went for the WRONG target
    #
    #

    if !isempty(df)

        ### rotate targets to prevent splitting a cluster between 0 and 360 degrees
        rot_angl = target_rotation
        rot_rad = 2*pi/360 * rot_angl ### rotate the data points by 22.5 degrees to get target3 as one cluster
        rot_matr = [cos(rot_rad) -sin(rot_rad); sin(rot_rad) cos(rot_rad) ]
        ref_dist = (2pi)/360 * 45 ### theoretical distance of targets in rad

        ### determin theoretical angles of targets
        angle_of_first_target = (2*pi)/360 * rot_angl
        angle_of_last_target = 7*(2*pi)/360*45 + (2*pi)/360*rot_angl
        theoretical_x_vals_targets_va = collect(range(angle_of_first_target, angle_of_last_target; step=ref_dist))

        ### get vector length and angles of fixations
        vec_length = Array{Any}(undef, 0)
        vec_angl = Array{Any}(undef, 0)

        for (itrl,trls_fix) in enumerate(df.Fixations)
            push!(vec_length, Array{Float64,1}(undef,0))
            push!(vec_angl, Array{Float64,1}(undef,0))

            for (ifix,fix) in enumerate(trls_fix)

                # vec = [mean(xy_nb[itrl][1][fix]), mean(xy_nb[itrl][2][fix])]
                vec = [mean(df.EyeXY[itrl][1][fix]), mean(df.EyeXY[itrl][2][fix])]

                ### rotate fixations to prevent the cluster at 0deg to be splitted between >0 and <360
                vec_rotated = rot_matr*vec

                ### length
                push!(vec_length[itrl], round(norm(vec_rotated), digits=1))

                ### angle relative to x axis
                if vec_rotated[2] >= 0
                    push!(vec_angl[itrl], acos(dot([1,0],vec_rotated)/(norm([1,0])*norm(vec_rotated))))
                else
                    push!(vec_angl[itrl], acos(dot([1,0],vec_rotated)/(norm([1,0])*norm(vec_rotated)))+pi)
                end

            end
        end

        ### fit histograms and kernel densitys to length
        vec_length_ = MyUtils.flatten(vec_length)
        h_vl = fit(Histogram, vec_length_, nbins=nbins_length)
        h_norm_vl = normalize(h_vl, mode=:pdf)
        vec_len_dens = kde(vec_length_, bandwidth=kde_bandwidth_length)
        ### find modes of fitted distributions
        x_idx_all_modes_vl = findlocalmaxima(vec_len_dens.density)
        x_idx_two_largest_modes_vl = x_idx_all_modes_vl[sortperm(vec_len_dens.density[x_idx_all_modes_vl], rev=true)][1:2]
        x_val_two_largest_modes_vl = vec_len_dens.x[x_idx_two_largest_modes_vl]
        y_val_two_largest_modes_vl = vec_len_dens.density[x_idx_two_largest_modes_vl]
        ### sort saccades according fixation vector length
        # min and max vector lengths of central fixations
        c_fix = [x_val_two_largest_modes_vl[1]-1, x_val_two_largest_modes_vl[1]+1]
        margin_t_fix_i = 3
        margin_t_fix_o = 3
        t_fix = [x_val_two_largest_modes_vl[2]-margin_t_fix_i, x_val_two_largest_modes_vl[2]+margin_t_fix_o]


        fixation_types = Array{Any,1}(undef, 0)

        for (itrl,trls_fix) in enumerate(df.Fixations)
            clean_fixations = [] ### Fixations that fall into the defined areas of central fixation point and targets

            push!(fixation_types, Array{Float64,1}(undef,0))

            for (ifix,fix) in enumerate(trls_fix)

                ### catch central fixations
                if c_fix[1] < vec_length[itrl][ifix] < c_fix[2]
                    push!(fixation_types[itrl], 0)
                elseif t_fix[1] < vec_length[itrl][ifix] < t_fix[2]
                    push!(fixation_types[itrl], 10)
                else
                    push!(fixation_types[itrl], -1)
                end

            end
        end

        bool_list_central_fixations = [ el.==10.0 for el in fixation_types ]
        vec_angl_targets_only = [vec_angl[ii][el] for (ii,el) in enumerate(bool_list_central_fixations)]

        ### fit histograms and kernel densitys to angles distributions
        vec_angl_ = MyUtils.flatten(vec_angl_targets_only)
        h_va = fit(Histogram, vec_angl_, nbins=nbins_angles)
        h_norm_va = normalize(h_va, mode=:pdf)
        vec_angl_dens = kde(vec_angl_, bandwidth=kde_bandwidth_angle)
        ### find modes of fitted distributions
        x_idx_all_modes_va = findlocalmaxima(vec_angl_dens.density)
        x_val_all_modes_va = vec_angl_dens.x[x_idx_all_modes_va]
        which_modes = [argmin(abs.(tx .- x_val_all_modes_va)) for tx in theoretical_x_vals_targets_va] ### find modes closest to theoretical location of modes
        x_idx_eight_largest_modes_va = x_idx_all_modes_va[which_modes]
        x_val_eight_largest_modes_va = x_val_all_modes_va[which_modes]
        y_val_eight_largest_modes_va = vec_angl_dens.density[x_idx_eight_largest_modes_va]
        ### correct for potentially wrong modes: replace modes that deviate more than 0.5 from theoretical mode
        false_modes = x_val_eight_largest_modes_va .- theoretical_x_vals_targets_va .> 0.5
        x_val_eight_largest_modes_va[false_modes] .=  theoretical_x_vals_targets_va[false_modes]

        ### angular range of target fixations
        margin_ts = 0.15
        t1 = [x_val_eight_largest_modes_va[1]-margin_ts, x_val_eight_largest_modes_va[1]+margin_ts]
        t2 = [x_val_eight_largest_modes_va[2]-margin_ts, x_val_eight_largest_modes_va[2]+margin_ts]
        t3 = [x_val_eight_largest_modes_va[3]-margin_ts, x_val_eight_largest_modes_va[3]+margin_ts]
        t4 = [x_val_eight_largest_modes_va[4]-margin_ts, x_val_eight_largest_modes_va[4]+margin_ts]
        t5 = [x_val_eight_largest_modes_va[5]-margin_ts, x_val_eight_largest_modes_va[5]+margin_ts]
        t6 = [x_val_eight_largest_modes_va[6]-margin_ts, x_val_eight_largest_modes_va[6]+margin_ts]
        t7 = [x_val_eight_largest_modes_va[7]-margin_ts, x_val_eight_largest_modes_va[7]+margin_ts]
        t8 = [x_val_eight_largest_modes_va[8]-margin_ts, x_val_eight_largest_modes_va[8]+margin_ts]

        for (itrl,trls_fix) in enumerate(fixation_types)
            clean_fixations = Array{Array{Float64,1},1}(undef,0)

            for (ifix,fix) in enumerate(trls_fix)

                ### catch target fixations
                if fix == 10.0

                    ### sort into traget IDs
                    if t1[1] < vec_angl[itrl][ifix] < t1[2]
                        fixation_types[itrl][ifix] = 3
                    elseif t2[1] < vec_angl[itrl][ifix] < t2[2]
                        fixation_types[itrl][ifix] = 2
                    elseif t3[1] < vec_angl[itrl][ifix] < t3[2]
                        fixation_types[itrl][ifix] = 1
                    elseif t4[1] < vec_angl[itrl][ifix] < t4[2]
                        fixation_types[itrl][ifix] = 8
                    elseif t5[1] < vec_angl[itrl][ifix] < t5[2]
                        fixation_types[itrl][ifix] = 4
                    elseif t6[1] < vec_angl[itrl][ifix] < t6[2]
                        fixation_types[itrl][ifix] = 5
                    elseif t7[1] < vec_angl[itrl][ifix] < t7[2]
                        fixation_types[itrl][ifix] = 6
                    elseif t8[1] < vec_angl[itrl][ifix] < t8[2]
                        fixation_types[itrl][ifix] = 7
                    else
                        fixation_types[itrl][ifix] = -1
                    end

                end

                ### store fixations belonging to target or central fixation dot
                if !(fixation_types[itrl][end] == -1)
                    push!( clean_fixations, df.Fixations[itrl][ifix] )
                end

            end

            if !plot_only
                df.LookedAtTarget[itrl] = fixation_types[itrl]

                ### add columns Values
                df.Valid[itrl] = (!df.WasFixationFailureFixPt[itrl]) & (any(unique(df[itrl, :LookedAtTarget]) .> 0))
                df.Invalid[itrl] = (!df.WasFixationFailureFixPt[itrl]) & (!any(unique(df[itrl, :LookedAtTarget]) .> 0))
            #     # df.Hit[itrl] = (df.Valid[itrl]) & (df[itrl, :LookedAtTarget][end] == df[itrl, :ValidTargetIndex])
                if !isempty(df[itrl, :LookedAtTarget])
                    df.FA[itrl] = (df.Valid[itrl]) & (df[itrl, :LookedAtTarget][end] != df[itrl, :ValidTargetIndex])
                end
            end

        end


        if plot_distributions | plot_only
            p = plot(layout=2, size=(1000,500), xlabel="saccade length / AU")
            plot!(h_norm_vl, subplot=1, xlims=[0, 50])
            plot!(vec_len_dens.x, vec_len_dens.density, linewidth=3, subplot=1)
            scatter!(x_val_two_largest_modes_vl, y_val_two_largest_modes_vl, subplot=1)
            #
            plot!(h_norm_va, subplot=2, xlabel="saccade angles / rad")
            plot!(vec_angl_dens.x, vec_angl_dens.density, linewidth=3, subplot=2)
            scatter!(x_val_eight_largest_modes_va, y_val_eight_largest_modes_va, subplot=2)
            return p
        end

    end

end


#################################################################################
#################################################################################


function i_dt(
    xy_o::Array{Array{Float64,1},1},
    sf,
    fix_duration_threshold_sec,
    fix_dispersion_threshold,
    saccade_velocity_threshold,
    saccade_duration_threshold
    )

    ###########
    #
    # Implements I-DT algorithm to detect fixations, sacades
    # extended to identify and interpolate blinks
    #
    # INPUT
    # fix_duration_threshold_sec: minimum duration of a fixation to be counted in sec
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

            ### if dispersion in current window is smaller than threshold
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
