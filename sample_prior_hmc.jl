#!/usr/bin/env julia
using Distributions
using DelimitedFiles

"""
To use threads, run `JULIA_NUM_THREADS=4 julia sample_prior_hmc.jl`
"""

include("./src/Potential.jl")
include("./src/hmc.jl")
include("./src/Parallel_tempering.jl")


println("Loading data..")
cost_mat = readdlm("data/london_n/cost_mat.txt")
cost_adj = convert(Array, cost_mat')
orig = readdlm("data/london_n/P.txt")
# origin, destination
nn, mm = size(cost_mat)

# ============
# potential and kinetic energy functions
alpha = 2.0
beta = 0.3*0.7e6
delta = 0.3/mm
gamma = 100.
kappa = 1.3
theta = [alpha, beta, delta, gamma, kappa]

function U_fun(x::Array{Float64,1})
    grad = Array{Float64,1}(undef, mm)
    Potential.potential!(grad, x, orig, cost_adj, theta, nn, mm)
end

L = 10
eps = 0.1

# set-up MCMC
mcmc_n = 10000
inverse_temps = [1., 1/2, 1/4, 1/8, 1/16]
temp_n = length(inverse_temps)
samples = zeros(Float64, mcmc_n, mm)

# initialise MCMC
xx = -log(mm)*ones(Float64, temp_n, mm)
V = zeros(Float64, temp_n)
gradV = zeros(Float64, temp_n, mm)
for j in 1:temp_n
    V[j], gradV[j, :] = U_fun(xx[j,:])
end

function runHMC(mcmc_n)
    # Counts to keep track of accept rates
    ac = zeros(Int64, temp_n)
    pc = zeros(Int64, temp_n)
    acs = 0
    pcs = 1
    # MCMC algorithm
    for i in 1:mcmc_n
        Threads.@threads for j in 1:temp_n
            xNew, V_New, gradV_New, ac_New = HMC_Prior.hmc_step(xx[j,:], U_fun, V[j],
                                        gradV[j,:], inverse_temps[j], eps, L, mm)
            xx[j,:] = xNew
            V[j], gradV[j,:] = V_New, gradV_New
            ac[j] += ac_New
            pc[j] += 1
        end

        # Perform a swap
        xx[:,:], V[:], gradV[:,:], acsNew = PT.swap_step(xx, V, gradV, inverse_temps)
        acs += acsNew
        pcs += 1

        # Update stored Markov-chain
        samples[i,:] = xx[1,:]

        if (i+1)%100 == 0
            println("Saving iteration $(i+1)")
            println("X AR: $(ac/pc)")
            println("Swap AR: $(acs/pcs)")
            # writedlm("outputs/hmc_samples$(alpha).txt", samples)
        end
    end

    println("Done")
end

runHMC(mcmc_n)
