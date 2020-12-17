module Experiment



export trialList2DataFrame
export data_of_run
export mat2jdl


using DataFrames
using MAT
using JLD
using Dates

using ..MyUtils
using ..Neurons


##################################################
#
#
#
##################################################
struct data_of_run
    paradigm::String
    date::Date
    time::DateTime
    sf::Float64
    trial_list::DataFrame
end




##################################################
#
#
#
##################################################
function trialList2DataFrame( data_dict )


    ### orderd according to intended column order in Julia
    ### (f1, f2): f1 -> column index in .mat.TrialList,
    ordered_column_name_pairs = [
        ([ Neurons.recording() for ii in 1:size(data_dict["TrialList"],1) ], "Neurons"),
        ([ [ data_dict["EyeX"][.!isnan.(data_dict["EyeX"][:,trl]),trl], data_dict["EyeY"][.!isnan.(data_dict["EyeY"][:,trl]),trl] ] for trl in 1:size(data_dict["EyeY"],2) ], "EyeXY"),
        (fill([[]],size(data_dict["TrialList"],1)), "Saccades"),
        (fill([[]],size(data_dict["TrialList"],1)), "Saccades_clean"),
        (fill([[]],size(data_dict["TrialList"],1)), "Fixations"),
        (fill([],size(data_dict["TrialList"],1)), "LookedAtTarget"),
        (data_dict["TrialList"][:,9], "TrialStartTime"),
        (data_dict["TrialList"][:,10], "TrialEndTime"),
        (fill(false, size(data_dict["TrialList"],1)), "Valid"),
        (data_dict["TrialList"][:,24].==1, "Hit"),
        (fill(false, size(data_dict["TrialList"],1)), "FA"),
        (fill(false, size(data_dict["TrialList"],1)), "Invalid"),
        (data_dict["TrialList"][:,23].==1, "WasMissed"),
        (data_dict["TrialList"][:,22].==1, "WasTimeout"),
        (data_dict["TrialList"][:,16].+1, "ValidTargetIndex"),
        (data_dict["TrialList"][:,12].+1, "CueTargetIndex"),
        (data_dict["TrialList"][:,17].==1, "WasCueValid"),
        (data_dict["TrialList"][:,18].==1, "WasFixationFailureFixPt"),
        (data_dict["TrialList"][:,19].==1, "WasFixationFailureTarget"),
        (data_dict["TrialList"][:,14], "LuminanceChange"),
        (data_dict["TrialList"][:,11], "LuminanceChangeTime"),
        (data_dict["TrialList"][:,21], "PauseDuration"),
        (data_dict["TrialList"][:,13], "ITI"),
        (data_dict["TrialList"][:,15], "NumberOfNonFixFailTrials"),
        (data_dict["TrialList"][:,20].==1, "WasLEDChangeActive")]


    ### -- Now in Behavior.jl --- ###
    # ### extract eye trace to store events in TrialList
    # xy = [[data_dict["EyeX"][:,trl], data_dict["EyeY"][:,trl]] for trl in 1:size(data_dict["EyeY"], 2)];
    # [deleteat!(xy[trl][dim], findall(isnan.(xy[trl][dim]))) for dim in [1,2], trl in 1:size(data_dict["EyeY"], 2)] # remove NaNs
    # sf = data_dict["samplingRate"]

    ### generate dataframe from ordered_column_name_pairs
    trial_list = DataFrame([el[1] for el in ordered_column_name_pairs], [Symbol(el[2]) for el in ordered_column_name_pairs])

    return trial_list
end


##################################################
#
# reads .mat file and generates raw_data stuct
#
##################################################
function mat2jdl(ifi, ofi)
    for (idx, mf_) in enumerate(ifi)

        ### read mat files of runs specified in paths_to_mat_files and store the in data_dicts array
        data_dict = matread(mf_)
        #data_dict_ = deepcopy(data_dict)

        ### remove missed trials in data_dict
        missed_trials_idx = findall(data_dict["TrialList"][:, 19].==1)
        ks = ["TargetY", "VoltX", "TargetX", "EyeT", "TargetTime", "EyeX", "EyeY", "VoltY"]
        for k in ks
            data_dict[k] = data_dict[k][:, setdiff(1:end, missed_trials_idx)]
        end
        data_dict["TrialList"] = data_dict["TrialList"][setdiff(1:end, missed_trials_idx), :]

        ### remove trials with all NaN
        xy = [[data_dict["EyeX"][:,trl], data_dict["EyeY"][:,trl]] for trl in 1:size(data_dict["EyeY"], 2)];
        only_nans = [trl for trl in 1:size(xy, 1) if all(isnan.(xy[trl][1]))] # find all NaN trials
        for k in ks
            data_dict[k] = data_dict[k][:, setdiff(1:end, only_nans)]
        end
        data_dict["TrialList"] = data_dict["TrialList"][setdiff(1:end, only_nans), :];

        print(".mat file loaded: ", idx,"/",size(ifi)[1])

        ### convert trialList from mat file to julia table
        trial_list = trialList2DataFrame(data_dict)
        print(" | trial_list extracted")

        ## get indices of valid and invalid trials
        # valid_trials_idx = findall( trial_list[!, :Valid] )

        raw_data = data_of_run(
            "...",
            Date(mf_[end-26:end-17],"y-m-d"),
            DateTime(mf_[end-15:end-8],"H-M-S"),
            data_dict["samplingRate"],
            trial_list
        )

        ### store processed data to file
        save(joinpath(ofi, mf_[end-41:end-4]*".jld"), "raw_data", raw_data)
        println(" | .jdl saved")

        #return eye_dat
    end
end




end
