"""
AX2StageLeastSquares(y, D, instrument, control)
A simple two stage least squares implementation.
"""
struct AX2StageLeastSquares <: AXEstimator
    β::Array{Float64} # coefficient
    FS::Array{Float64} # first stage coefficients
    y::Matrix{Float64} # response
	Z::Matrix{Float64} # combined first stage variables
    X::Matrix{Float64} # combined second stage variables

	# Define constructor function
	function AX2StageLeastSquares(y::Matrix{Float64}, D::Matrix{Float64},
                                  instrument::Matrix{Float64}, control = nothing)
        # Add constant if no control is passed
        if isnothing(control) control = ones(length(y)) end

		# Define data matrices
		Z = hcat(control, instrument) # combined first stage variables
		X = hcat(D, control)          # combined second stage variables

		# Calculate TSLS coefficient
        PZ = Z * (Z' * Z)^(-1) * Z'
        β  = inv(X' * PZ * PZ * X) * (X' * PZ * y)
        FS = inv(Z' * Z) * (Z' * X)

		# Return output
		new(β, FS, y, Z, X)
	end #AX2StageLeastSquares
end #AX2StageLeastSquares

# Methods ======================================================================
"""
infer(fit::AX2StageLeastSquares; heteroskedastic, print_df)
A method to calculate standard errors of an AX2StageLeastSquares object.

"""
function infer(fit::AX2StageLeastSquares;
               heteroskedastic::Bool=false, cluster=nothing,
               print_df::Bool=true)
    # Retrieve necessary parameters
    N = length(fit.y)
    Kx = size(fit.X, 2)
    Kz = size(fit.Z, 2)

	# Calculate the variance-covariance matrix
    FShat = fit.Z * fit.FS
    FSinv = inv(FShat' * FShat)
    resid = fit.y - predict(fit)
	if heteroskedastic
        residFS = FShat .* resid
        PZuuZP  = (residFS' * residFS)
		vcv     = FSinv * PZuuZP * FSinv .* (N / (N - Kz))
	else
		vcv = FSinv .* (resid' * resid) ./ (N - Kz)
	end

	# Get standard errors, t-statistics and p-values
    se = sqrt.(vcv[diagind(vcv)])
    t_stat = fit.β ./ se
    p_val = 2 * cdf.(Normal(), -abs.(t_stat))

    # Print estimates
    if print_df
        out_df = DataFrame(hcat(fit.β, se, t_stat, p_val), :auto)
        rename!(out_df, ["coef", "se", "t-stat", "p-val"])
        display(out_df)
    end

    # Organize and return output
    output = (β = fit.β, se = se, t = t_stat, p = p_val)
    return output
end #infer.AX2StageLeastSquares
