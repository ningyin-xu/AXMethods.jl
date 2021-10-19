"""
AXLeastSquares(y, X)

An Ordinary Least Squares implementation.
"""

struct AXLeastSquares <: AXEstimator
    β::Array{Float64} # Coefficient
    y::Matrix{Float64} # Response
    X::Matrix{Float64} # Covariates

    # Constructor Function
    function AXLeastSquares(y::Matrix{Float64}, X::Matrix{Float64})
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

    se = sqrt.(diag(vcv))
    t_stat = fit.β ./ se
    p_val = 2 * cdf.(Normal(), -abs.(t_stat))
    r2 = 1 - sum(resid.^2)/sum((fit.y.-mean(fit.y)).^2)

    # Print estimates
    if print_df
        out_df = DataFrame(hcat(fit.β, se, t_stat, p_val, r2), :auto)
        rename!(out_df, ["coef", "se", "t-stat", "p-val", "R-sqaure"])
        display(out_df)
    end

    # Organize and return output
    output = (β = fit.β, se = se, t = t_stat, p = p_val, r = r2)
    return output
end #infer.AXLeastSquares
