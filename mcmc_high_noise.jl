#!/usr/bin/env julia
using Distributed
addprocs(7)
@everywhere using Distributions
@everywhere using DelimitedFiles

"""
MCMC scheme for high-noise regime.
"""

@everywhere include("./src/Potential.jl")
include("./src/hmc.jl")
@everywhere include("./src/MCMC_full.jl")


println("Loading data..")
@everywhere cost_mat = readdlm("data/london_n/cost_mat.txt")
@everywhere cost_adj = convert(Array, cost_mat')
@everywhere orig = readdlm("data/london_n/P.txt")
xd = readdlm("data/london_n/xd0.txt")
# origin, destination
@everywhere nn, mm = size(cost_mat)

# high noise: gamma = 100.
alpha = 1
beta = 0
delta = 0.3/mm
gamma = 100.
kappa = 1.3
theta = [alpha, beta, delta, gamma, kappa]

stopping = readdlm("data/stoppingArray.txt")

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

@everywhere function ValuePot(x::Array{Float64,1}, theta::Array{Float64,1}, alpha::Float64, beta::Float64)
    theta[1] = alpha
    theta[2] = beta
    grad = Array{Float64,1}(undef, mm)
    Potential.potential!(grad, x, orig, cost_adj, theta, nn, mm)
end



@everywhere function ais_ln_z(i::Int64, alpha::Float64, beta::Float64, ValuePot::Function)
    """
    usage: `map(x -> ais_ln_z(x, alpha, beta), 1:4)`
    5.4 seconds per iteration
    """
    p_n = 10
    t_n = 50
    L = 10
    eps = 0.1
    temp = [elem for elem in LinRange(0, 1, t_n)]
    minustemp = ones(Float64, t_n) - temp
    ac = 0
    pc = 0
    log_weights = -log(p_n)*ones(p_n)
    delta = 0.3/mm
    gamma = 100.
    kappa = 1.3
    theta = [alpha, beta, delta, gamma, kappa]
    V0_p, V1_p = 0., 0.
    gradV0_p, gradV1_p = ones(Float64, mm), ones(Float64, mm)

    # For each particle...
    for ip in 1:p_n
        # Initialize
        xx = map(log, rand(Gamma(gamma*(delta + 1. /mm), 1. /(gamma*kappa)), mm))     #Log-gamma model with alpha,beta->0
        V0, gradV0 = MCMC_full.pot0Value(xx)
        V1, gradV1 = ValuePot(xx, theta, alpha, beta)

        # Anneal...
        for it in 2:t_n
            log_weights[ip] += (temp[it] - temp[it-1])*(V0 - V1)

            # Initialize HMC
            p = rand(Normal(0., 1.), mm)
            V, gradV = minustemp[it]*V0 + temp[it]*V1, minustemp[it]*gradV0 + temp[it]*gradV1
            H = 0.5 * p'*p + V

            # HMC leapfrog integrator
            x_p = xx
            p_p = p
            V_p, gradV_p = V, gradV
            for j in 1:L
                p_p = p_p - 0.5*eps*gradV_p
                x_p = x_p + eps*p_p
                V0_p, gradV0_p = MCMC_full.pot0Value(x_p)
                V1_p, gradV1_p = ValuePot(x_p, theta, alpha, beta)
                V_p, gradV_p = minustemp[it]*V0_p + temp[it]*V1_p, minustemp[it]*gradV0_p + temp[it]*gradV1_p
                p_p = p_p - 0.5*eps*gradV_p
            end
            # HMC accept/reject
            pc += 1
            H_p = 0.5 * p_p'*p_p + V_p
            if log(rand(Uniform(0, 1))) < H - H_p
                xx = x_p
                V0, gradV0 = V0_p, gradV0_p
                V1, gradV1 = V1_p, gradV1_p
                ac += 1
            end
        end
    end
    MCMC_full.logsumexp(log_weights)
end

stoppingArray = convert(Array{Int64}, readdlm(joinpath(@__DIR__, "data/stoppingArray.txt")))

function unbiased_z_inv(cc::Int64, alpha::Float64, beta::Float64, ValuePot::Function)
    N = stoppingArray[cc]
    k_pow = 1.1
    println("Debiasing with N=$N")

    # Get importance sampling estimate of z(theta) in parallel
    log_weights = pmap(x -> ais_ln_z(x, alpha, beta, ValuePot), 1:(N+1))

    # Compute S = Y[0] + sum_i (Y[i] - Y[i-1])/P(K > i) using logarithms
    ln_Y = zeros(Float64, N+1)
    ln_Y_pos = zeros(Float64, N+1)
    ln_Y_neg = zeros(Float64, N)

    for i in 1:N+1
        ln_Y[i] = log(i) - MCMC_full.logsumexp(log_weights[1:i])
    end

    ln_Y_pos[1] = ln_Y[1]
    for i in 2:N+1
        ln_Y_pos[i] = ln_Y[i] + k_pow*log(i-1)
        ln_Y_neg[i-1] = ln_Y[i-1] + k_pow*log(i-1)
    end
    positive_sum = MCMC_full.logsumexp(ln_Y_pos)
    negative_sum = MCMC_full.logsumexp(ln_Y_neg)

    ret = zeros(2)
    if(positive_sum >= negative_sum)
        ret[1] = positive_sum + log(1. - exp(negative_sum - positive_sum))
        ret[2] = 1.
    else
        ret[1] = negative_sum + log(1. - exp(positive_sum - negative_sum))
        ret[2] = -1.
    end
    ret
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
    lnzinv, ss = unbiased_z_inv(mcmc_start-1, kk[1], kk[2]*0.7e6, ValuePot)
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
            lnzinv_p, ss_p = unbiased_z_inv(i, kk_p[1], kk_p[2]*0.7e6, ValuePot)
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
