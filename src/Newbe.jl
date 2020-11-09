module Newbe

__precompile__(true)

include("allmodules.jl")

### this allows to do Newbe.myColors() instal of Newbe.MyUtils.myColors()
using .MyUtils
export MyUtils

using .Behavior
export Behavior

#test

end
