module AXMethods

# Dependencies
using DataFrames
using Distributions
using LinearAlgebra
using Random
using Statistics

# Module Export
export AXEstimator, AXLeastSquares, AX2StageLeastSquares
export coef, predict, infer

# Module Content
include("G:\\My Drive\\PhD\\AXMethods\\src\\FUN\\AXEstimator.jl")
include("G:\\My Drive\\PhD\\AXMethods\\src\\FUN\\AXLeastSquares.jl")
include("G:\\My Drive\\PhD\\AXMethods\\src\\FUN\\AX2StageLeastSquares.jl")

end
