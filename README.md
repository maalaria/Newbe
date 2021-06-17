
### experiment.jl

```experiment.jl``` implements ```data_of_run::struct``` which is intended to hold all the data of one experimental sessions/ run. It has the fields:

+ ```date::Date```
+ ```time::DateTime```
+ ```sf::Float64```                         
+ ```trial_list::DataFrame```

```trial_list``` is its heart, a DataFrame containing all dependent (eg. enural recordings, eyetracking data) and independent data. Each row represents a trial of the given session/ run.




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

.behavior
|
|---- getEyeEvents::function
