### order matters!!!
# if Experiment.jl loads Neurons.jl the latter one has to be included before

include("MyUtils.jl")
include("Behavior.jl")
include("Neurons.jl")
include("Experiment.jl")
include("MyDSP.jl")
include("MyLinearAlgebra.jl")
