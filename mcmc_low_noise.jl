#!/usr/bin/env julia
using Distributions
using DelimitedFiles

"""
MCMC scheme for low-noise regime.
To use threads, run `JULIA_NUM_THREADS=4 julia mcmc_low_noise.jl`
Threads give approximately *2.5 speedup
"""


include("./src/Potential.jl")
include("./src/hmc.jl")
include("./src/Parallel_tempering.jl")


println("Loading data..")
cost_mat = readdlm("data/london_n/cost_mat.txt")
cost_adj = convert(Array, cost_mat')
orig = readdlm("data/london_n/P.txt")
xd = readdlm("data/london_n/xd0.txt")
# origin, destination
nn, mm = size(cost_mat)


# low noise: gamma = 10000.
alpha = 2.0
beta = 0.3*0.7e6
delta = 0.3/mm
gamma = 10000.
kappa = 1.3
theta = [alpha, beta, delta, gamma, kappa]

function z_inv(m, delta, valuePot, gradPot!)
    f_val = Inf16
    min_value = 0
    Threads.@threads for k in 1:mm
        g = log(delta)*ones(mm)
        g[k] = log(1+delta)
        # g[k] = log(delta)+0.1
        f = optimize(valuePot, gradPot!, g, LBFGS())
        if f.minimum < f_val
            f_val = f.minimum
            m = f.minimizer
        end
    end
    A = Potential.hessian(minimizer, orig, cost_mat, theta, nn, mm)
    L = cholesky(A).L
    half_log_det_A = sum(map(log, diag(L)))
    ret = zeros(Float64, 2)
    ret[1] = f_val + half_log_det_A
    ret[2] = 1.
    ret
end

# MCMC tuning parameters
Ap = [0.00749674 0.00182529 ; 0.00182529 0.00709968]   # Randomwalk covariance
L2 = 50                                                # Number of leapfrog steps
eps2 = 0.02                                            # Leapfrog step size

# Set-up MCMC
mcmc_start = 1
mcmc_n = 20000

samples = Array{Float64, 1}(undef, mcmc_n, 2)   # Theta-values
samples1 = Array{Float64, 1}(undef, mcmc_n, mm) # X-values
samples2 = Array{Float64, 1}(undef, mcmc_n)     # Sign-values

samples_init = readdlm("output/low_noise_samples.txt")
samples2_init = readdlm("output/low_noise_samples2.txt")
samples3_init = readdlm("output/low_noise_samples3.txt")

samples[1:mcmc_start, :] = samples_init
samples2[1:mcmc_start, :] = samples2_init
samples3[1:mcmc_start, :] = samples3_init

function U_fun(x::Array{Float64,1}, theta::Array{Float64,1}, alpha::Float64, beta::Float64)
    theta[1] = alpha
    theta[2] = beta
    grad = Array{Float64,1}(undef, mm)
    Potential.potential!(grad, x, orig, cost_adj, theta, nn, mm)
end

# Initialize MCMC
println("Starting at $(mcmc_start)")
kk = samples[mcmc_start, :]
xx = samples1[mcmc_start, :]
valuePot = X -> Potential.potential_val(X, orig, cost_adj, theta, nn, mm)
gradPot! = (grad, X) -> Potential.potential_grad!(grad, X, orig, cost_adj, theta, nn, mm)
lnzinv, ss = z_inv(xd, delta, valuePot, gradPot!)
V, gradV = pot_value(xx)



function runMCMC(mcmc_n)
    # Counts to keep track of accept rates
    ac = 0
    pc = 0
    ac2 = 0
    pc2 = 0
    # MCMC algorithm
    for i in mcmc_start:mcmc_n
        print("Iteration: $i")

        # Theta-proposal (random walk with reflecting boundaries)


        # Theta-accept/reject

        # Reset theta for HMC

        # Initialize leapfrog integrator for HMC proposal

        # X-Proposal

        # X-accept/reject


        # Update stored Markov-chain

        # Savedown and output details every 100 iterations

    end
    print("Done")
end

runMCMC(mcmc_n)
