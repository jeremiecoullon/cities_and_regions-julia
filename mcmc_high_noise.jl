#!/usr/bin/env julia
using Distributions
using DelimitedFiles
using Optim
using LinearAlgebra
"""
MCMC scheme for low-noise regime.
To use threads, run `JULIA_NUM_THREADS=4 julia mcmc_low_noise.jl`
Threads give approximately *2.5 speedup
"""


include("./src/Potential.jl")
include("./src/hmc.jl")
include("./src/MCMC_full.jl")


println("Loading data..")
cost_mat = readdlm("data/london_n/cost_mat.txt")
cost_adj = convert(Array, cost_mat')
orig = readdlm("data/london_n/P.txt")
xd = readdlm("data/london_n/xd0.txt")
# origin, destination
nn, mm = size(cost_mat)


# high noise: gamma = 100.
alpha = 1
beta = 0
delta = 0.3/mm
gamma = 100.
kappa = 1.3
theta = [alpha, beta, delta, gamma, kappa]

stopping = readdlm("data/stopping.txt")

function ais_ln_z()
    1.
end

function unbiased_z_inv()
    1.
end


# MCMC tuning parameters
rwk_sd = 0.3                # Randomwalk covariance
L2 = 50                     # Number of leapfrog steps
eps2 = 0.02                 # Leapfrog step size

# Set-up MCMC
mcmc_start = 10000
mcmc_n = 20000

samples = zeros(Float64, mcmc_n, 2)   # Theta-values
samples2 = ones(Float64, mcmc_n, mm) # X-values
samples3 = zeros(Float64, mcmc_n)     # Sign-values

samples_init = readdlm("outputs/paper_outputs/high_noise_samples.txt")
samples2_init = readdlm("outputs/paper_outputs/high_noise_samples2.txt")
samples3_init = readdlm("outputs/paper_outputs/high_noise_samples3.txt")

samples[1:mcmc_start, :] = samples_init[1:mcmc_start,:]
samples2[1:mcmc_start, :] = samples2_init[1:mcmc_start,:]
samples3[1:mcmc_start] = samples3_init[1:mcmc_start]

function ValuePot(x::Array{Float64,1}, theta::Array{Float64,1}, alpha::Float64, beta::Float64)
    theta[1] = alpha
    theta[2] = beta
    grad = Array{Float64,1}(undef, mm)
    Potential.potential!(grad, x, orig, cost_adj, theta, nn, mm)
end


function runMCMC(mcmc_n)
    # Counts to keep track of accept rates
    ac = 0
    pc = 0
    ac2 = 0
    pc2 = 0
    # Initialize MCMC
    println("Starting at $(mcmc_start)")
    println("Warning max random stopping is  + $(maximum(stopping[mcmc_start:mcmc_n]))")
    kk = samples[mcmc_start, :]
    xx = samples2[mcmc_start, :]
    lnzinv, ss = unbiased_z_inv()
    V, gradV = ValuePot(xx, theta, kk[1], kk[2]*0.7e6)
    # MCMC algorithm
    for i in mcmc_start:mcmc_n
        println("\nIteration: $i")

        # Theta-proposal (random walk with reflecting boundaries)
        kk_p = kk + rand(Normal(0, rwk_sd),2)
        for j in 1:2
            if kk_p[j] < 0.
                kk_p[j] = -kk_p[j]
            elseif kk_p[j] >2.
                kk_p[j] = 2. - (kk_p[j]-2.)
            end
        end
        # Theta-accept/reject
        if minimum(kk_p)>0 && maximum(kk_p)<=2
            try
                lnzinv_p, ss_p = unbiased_z_inv()
                V_p, gradV_p = ValuePot(xx, theta, kk_p[1], kk_p[2]*0.7e6)
                pp_p = lnzinv_p - V_p
                pp = lnzinv - V
                println("Proposing $kk_p with $ss_p")
                println("$pp_p vs $pp")

                pc += 1
                if log(rand(Uniform(0,1))) < pp_p - pp
                    println("Theta-Accept")
                    kk = kk_p
                    V, gradV = V_p, gradV_p
                    ac += 1
                    lnzinv, ss = lnzinv_p, ss_p
                else
                    println("Theta-reject")
                end
            catch
                nothing
            end
        end

        # Initialize leapfrog integrator for HMC proposal
        U_fun = X -> ValuePot(X, theta, kk[1], kk[2]*0.7e6) + MCMC_full.likeValue(X, xd[:,1])
        potLik, gradLik = MCMC_full.likeValue(xx, xd[:,1])
        Wpot, gradW = V + potLik, gradV + gradLik

        # X-Proposal
        xx, W_New, gradW_new, ac_New = HMC_Prior.hmc_step(xx, U_fun, Wpot,
                                    gradW, 1., eps2, L2, mm)
        potLik_new, gradLik_new = MCMC_full.likeValue(xx, xd[:,1])
        V, gradV = W_New-potLik_new, gradW_new-gradLik_new
        ac2 += ac_New
        pc2 += 1
        # Update stored Markov-chain
        samples[i,:] = kk
        samples2[i,:] = xx
        samples3[i] = ss

        # Savedown and output details every 100 iterations
        if (i+1)%10 == 0
            println("Saving iteration $(i)")
            writedlm("outputs/high_noise_samples.txt", samples)
            writedlm("outputs/high_noise_samples2.txt", samples2)
            writedlm("outputs/high_noise_samples3.txt", samples3)
            println("Theta AR: $(ac/pc)")
            println("X AR: $(ac2/pc2)")
            println("Net + ves: $(sum(samples3[1:i+1,1])))")
        end
    end
    print("Done")
end

runMCMC(mcmc_n)
