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
    # specify columns names
    column_names = [
    "EyeXY",
    "Saccades",
    "Fixations",
    "LookedAtTarget",
    "Neurons",
    "TrialStartTime",               # 9
    "TrialEndTime",                 #10
    "LuminanceChangeTime",          #11
    "CueTargetIndex",               #12
    "ITI",                          #13
    "LuminanceChange",              #14
#    "NumberOfNonFixFailTrials",    #15
#    "NumberOfValidTrials",         #16
    "ValidTargetIndex",             #17
    "WasCueValid",                  #18
    "WasFixationFailureFixPt",      #19
    "WasFixationFailureTarget",     #20
    "WasLEDChangeActive",           #21
    "WasPauseAdded",                #22
    "WasTimeout",                   #23
    "WasMissed",                    #24
    "WasInvalidTargetChosen",       #25
    "WasValidTargetChosen",         #26
    "Valid",
    "Invalid",
    "Hit",
    "FA"
    ]

    ### extract eye trace to store events in TrialList
    xy = [[data_dict["EyeX"][:,trl], data_dict["EyeY"][:,trl]] for trl in 1:size(data_dict["EyeY"], 2)];
    [deleteat!(xy[trl][dim], findall(isnan.(xy[trl][dim]))) for dim in [1,2], trl in 1:size(data_dict["EyeY"], 2)] # remove NaNs
    sf = data_dict["samplingRate"]
    fix_duration_threshold = 0.07
    ### generate dataframe with list of trial paramters
    trial_list = DataFrame(
        vEyeXY              = [ [data_dict["EyeX"][.!isnan.(data_dict["EyeX"][:,trl]),trl], data_dict["EyeY"][.!isnan.(data_dict["EyeY"][:,trl]),trl]] for trl in 1:size(data_dict["EyeY"],2) ],
        vSaccades           = fill([[]],size(data_dict["TrialList"],1)),
        vFixations          = fill([[]],size(data_dict["TrialList"],1)),
        vLookedAtTargets    = fill([],size(data_dict["TrialList"],1)),
        vNeurons    = [ Neurons.recording() for ii in 1:size(data_dict["TrialList"],1) ],
        v9          = data_dict["TrialList"][:,9],       # TrialStartTime
        v10         = data_dict["TrialList"][:,10],   # TrialEndTime
        v11         = data_dict["TrialList"][:,11],     # LuminanceChangeTime
        v12         = data_dict["TrialList"][:,12],     # CueTargetIndex
        v13         = data_dict["TrialList"][:,13],     # ITI
        v14         = data_dict["TrialList"][:,14],     # LuminanceChange
 #       v15        = data_dict["TrialList"][:,15],    # NumberOfNonFixFailTrials
 #       v16        = data_dict["TrialList"][:,16],    # NumberOfValidTrials
        v17         = data_dict["TrialList"][:,17] .+ 1,     # ValidTargetIndex, map to 1-8
        ### the following are converted into booleans
        v18         = data_dict["TrialList"][:,18].==1, # WasCueValid
        v19         = data_dict["TrialList"][:,19].==1, # WasFixationFailureFixPt, i.e. trial is initiated if 0 and ignored/Miss if 1
        v20         = data_dict["TrialList"][:,20].==1, # WasFixationFailureTarget
        v21         = data_dict["TrialList"][:,21].==1, # WasLEDChangeActive
        v22         = data_dict["TrialList"][:,22].==1, # WasPauseAdded
        v23         = data_dict["TrialList"][:,23].==1, # WasTimeout
        v24         = data_dict["TrialList"][:,24].==1, # WasMissed
        v25         = data_dict["TrialList"][:,25].==1, # WasInvalidTargetChosen
        v26         = data_dict["TrialList"][:,26].==1, # WasValidTargetChosen
        vValid      = fill(-10, size(data_dict["TrialList"],1)), # initiated and going for a target
        vInvalid    = fill(-10, size(data_dict["TrialList"],1)),  # initiated but not going for a target
        vHit        = (data_dict["TrialList"][:,19].==0) .& (data_dict["TrialList"][:,26].==1), # valid and chosen target was correct
        vFA         = (data_dict["TrialList"][:,19].==0) .& (data_dict["TrialList"][:,25].==1), # valid but chosen target was wrong
    )
    # rename columns
    # names!(trial_list, [Symbol(cn) for cn in column_names])
    rename!(trial_list, [Symbol(cn) for cn in column_names])
    # names!(df::AbstractDataFrame, vals::Vector{Symbol}; makeunique::Bool = false)
    # rename!(df, vals, makeunique = false)
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
        valid_trials_idx = findall( trial_list[!, :Valid] )

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
