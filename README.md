# Newbe

*Analyze neurons and behavior with Julia*  


## experiment.jl

```experiment.jl``` implements ```data_of_run::struct``` which is intended to hold all the data of one experimental sessions/ run. It has the fields:

+ ```date::Date```
+ ```time::DateTime```
+ ```sf::Float64```                         
+ ```trial_list::DataFrame```

```trial_list``` is its heart, a DataFrame containing all dependent (eg. enural recordings, eyetracking data) and independent data. Each row represents a trial of the given session/ run.

**Functions**  

+ ```trialList2DataFrame```   
+ ```mat2jdl```



## behavior.jl

**Functions**  

+ ```i_dt_wrapper```   
+ ```i_dt```
+ ```plot_eye_events```
+ ```identify_fixatedTargets```



## neurons.jl

**Functions**  

+ ```normalize_trace```   
+ ```filter_trace```
+ ```resample```
+ ```get_PSD```
+ ```find_redundant```
+ ```detect_spikes```
+ ```extract_isih```
+ ```extract_spikes```
+ ```extract_features```
+ ```cluster_spikes```

PLOTTING  

+ ```plot_detection```
+ ```plot_waveforms```
