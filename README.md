### A typical top-level directory layout

    .
    ├── build                   # Compiled files (alternatively `dist`)
    ├── docs                    # Documentation files (alternatively `doc`)
    ├── src                     # Source files (alternatively `lib` or `app`)
    ├── test                    # Automated tests (alternatively `spec` or `tests`)
    ├── tools                   # Tools and utilities
    ├── LICENSE
    └── README.md



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
