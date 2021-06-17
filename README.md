### experiment.jl


```experiment.jl``` implements ```data_of_run::struct``` intended to hold the data of a single experimental session. ```data_of_run``` has the fields:

+ ```date::Date```
+ ```time::DateTime```
+ ```sf::Float64```
+ ```trial_list::DataFrame```

```trial_list``` is its heart and intended to hold all data, eg. neural recodings of any form, eyetracking and other pshychophysical measures as well as all relevant independent variables in form of a ```DataFrame```. Each row represents one trial of the given session/ run.





#######################
.data_structure
|
|---- data_of_run::struct
      |
      | ### FIELDS ###
      |---- date::Date
      |---- time::DateTime
      |---- sf::Float64                                     ##### needs to be extended to contain eye tracking sf and myEphys sf
      |---- trial_list_all::DataFrame
            |
      |---- trial_list_valid::DataFrame
            |
      |---- trial_list_invalid::DataFrame
            |
            | ### COLUMNS ### rows contain data of trials
            |---- EyeXY
            |---- Neurons                                   ######
            |---- TrialTime
            |---- ChangeTime
            |---- CueTargetIndex
            |---- ITI
            |---- LuminanceChange
            |---- ValidTargetIndex
            |---- WasCueValid
            |---- WasFixationFailureFixPt
            |---- WasFixationFailureTarget
            |---- WasLEDChangeActive
            |---- WasPauseAdded
            |---- WasTimeout
            |---- Valid
            |---- Invalid
            |---- NumberOfSaccades
            |---- NumberOfBlinks

#######################
.neurons
|
|---- neuron::mutable struct
      |
      |
|
|---- detect_spikes::function
|
|---- filter_trace::function
            filtered data is stored on neuron.trace_preprocessed
|
|---- resample::function

#######################
.behavior
|
|---- getEyeEvents::function
