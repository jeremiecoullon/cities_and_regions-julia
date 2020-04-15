using Plots
using Optim
using DelimitedFiles
using LinearAlgebra
using BenchmarkTools

include("./src/Potential.jl")


cost_mat = readdlm("data/london_n/cost_mat.txt")
cost_adj = convert(Array, cost_mat')
orig = readdlm("data/london_n/P.txt")

xd = readdlm("data/london_n/xd0.txt")
nn, mm = size(cost_mat)


alpha = 1
beta = 0
delta = 0.3/mm
gamma = 10000
kappa = 1.3

function optimise_model(m, delta, valuePot, gradPot!)
    """
    Parameters:
    ----------
    m: default minimizer for the potential function if no result is found
    delta: to determine the IC
    valuePot: potential function
    gradPot: gradient of potential function

    Returns:
    -------
    m: minimizer for the potential function
    min_value: value of potential at minimum
    """
    f_val = Inf16
    min_value = 0
    Threads.@threads for k in 1:mm
        g = log(delta)*ones(mm)
        g[k] = log(1+delta)
        # g[k] = log(delta)+ 0.1
        f = optimize(valuePot, gradPot!, g, LBFGS())
        if f.minimum < f_val
            f_val = f.minimum
            m = f.minimizer
            min_value = f.minimum
        end
    end
    m, min_value
end

function estimatePosterior(alpha, beta)
    lap_c1 = 0.5*mm*log(2*pi)
    theta = [alpha, beta, delta, gamma, kappa]
    levaluePot = X -> Potential.potential_val(X, orig, cost_adj, theta, nn, mm)[1]
    legradPot! = (grad, X) -> Potential.potential_grad!(grad, X, orig, cost_adj, theta, nn, mm)[2]

    minimizer, le_min = optimise_model(xd, delta, levaluePot, legradPot!)
    A = Potential.hessian(minimizer, orig, cost_mat, theta, nn, mm)
    L = cholesky(A).L
    half_log_det_A = sum(map(log, diag(L)))
    lap = - le_min + lap_c1 - half_log_det_A
    pot_data = Potential.potential_val(xd[:,1], orig, cost_adj, [alpha, beta, delta, gamma, kappa], nn, mm)[1]
    lik_value = - lap - pot_data
end

println("Time estimate posterior")
# @btime estimatePosterior(1,1)

println(estimatePosterior(1,1))
