"""
AXLeastSquares(y, X)

An Ordinary Least Squares implementation.
"""

struct AXLeastSquares <: AXEstimator
    β::Array{Float64} # Coefficient
    y::Array{Float64} # Response
    X::Array{Float64} # Covariates

    # Constructor Function
    function AXLeastSquares(y::Array{Float64}, X::Array{Float64})
        β = (X' * X)^(-1) * (X' * y)
        new(β, y, X)
    end
end #AXLeastSquares

# Methods ======================================================================
"""
infer(fit::AXLeastSquares; heteroskedastic, print_df)

A method to calculate standard errors of an AXLeastSquares object.
"""
function infer(fit::AXLeastSquares; heteroskedastic::Bool=false, print_df::Bool=true)
    # Retrieve necessary parameters
    N = length(fit.y)
    K = size(fit.X, 2) # number of covariates

    # Calculate the variance-covariance matrix
    resid  = fit.y - predict(fit)
    XX     = fit.X' * fit.X
    residX = fit.X .* resid
    if heteroskedastic
        vcv = XX^(-1) * (residX' * residX) * XX^(-1) .* (N / (N-K))
    else
        # homoskedasticity
        vcv = XX^(-1) .* (resid' * resid) ./ (N-K)
    end

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
end #infer.AXLeastSquares