"""
AXEstimator
An abstract type for estimators.
"""
abstract type AXEstimator end

# Methods
"""
coef(fit::AXEstimator)
A method to get the coefficient from an AXEstimator object.
"""
function coef(fit::AXEstimator)
    return fit.β
end

"""
predict(fit::AXEstimator)
A method to predict data based on an AXEstimator object.
"""
function predict(fit::AXEstimator, newdata = nothing)
    isnothing(newdata) ? prediction = fit.X * fit.β : prediction = newdata * fit.β
    return(prediction)
end
