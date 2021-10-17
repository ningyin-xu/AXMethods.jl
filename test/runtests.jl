using AXMethods
using Test

@testset "AXLeastSquares.jl" begin
    # Generate example data
    n_i = 1000
    X   = 2 * randn((n_i, 2))
    y   = X * [1, -2] + randn((n_i, 1))

    # Estimate the least square regression
    ls_fit = AXLeastSquares(y, X)

    # Check the methods
    β_hat = coef(ls_fit)
    y_hat = predict(ls_fit)
    se_hat = infer(ls_fit, print_df = false)
    se_robust_hat = infer(ls_fit, heteroskedastic = true, print_df = false)

    # Let's check that everything is of correct type.
    @test typeof(ls_fit) == AXLeastSquares
    @test typeof(β_hat) == Array{Float64,2}
    @test typeof(y_hat) == Array{Float64,2}
    @test typeof(se_hat) == NamedTuple{(:β, :se, :t, :p),
                                       Tuple{Array{Float64,2},
                                             Array{Float64,1},
                                             Array{Float64,2},
                                             Array{Float64,2}}}
    @test typeof(se_robust_hat) == NamedTuple{(:β, :se, :t, :p),
                                          Tuple{Array{Float64,2},
                                                Array{Float64,1},
                                                Array{Float64,2},
                                                Array{Float64,2}}}
end

@testset "AX2StageLeastSquares.jl" begin
    # Generate example data
    n_i = 1000
    ν = randn((n_i, 1))
    instrument = 2 * randn((n_i, 2))
    D = instrument * [1, -2] + ν
    y = D + 0.3 * ν + randn((n_i, 1))

    # Estimate the least square regression
    tsls_fit = AX2StageLeastSquares(y, D, instrument)

    # Check the methods
    β_hat = coef(tsls_fit)
    y_hat = predict(tsls_fit)
    se_hat = infer(tsls_fit, print_df = false)
    se_hc_hat = infer(tsls_fit, heteroskedastic = true, print_df = false)

    # Let's check that everything is of correct type.
    @test typeof(tsls_fit) == AX2StageLeastSquares
    @test typeof(β_hat) == Array{Float64,2}
    @test typeof(y_hat) == Array{Float64,2}
    @test typeof(se_hat) == NamedTuple{(:β, :se, :t, :p),
                                       Tuple{Array{Float64,2},
                                             Array{Float64,1},
                                             Array{Float64,2},
                                             Array{Float64,2}}}
    @test typeof(se_hc_hat) == NamedTuple{(:β, :se, :t, :p),
                                          Tuple{Array{Float64,2},
                                                Array{Float64,1},
                                                Array{Float64,2},
                                                Array{Float64,2}}}
end
