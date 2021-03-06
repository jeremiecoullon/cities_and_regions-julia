#!/usr/bin/env julia

using DelimitedFiles
using BenchmarkTools
include("./src/Potential.jl")

"Time the potential and hessian functions"

cost_mat = readdlm("data/london_n/cost_mat.txt")
cost_adj = convert(Array, cost_mat')
orig = readdlm("data/london_n/P.txt")

N, M = size(cost_mat)


alpha = 2.
beta = 0.3*0.7e6
delta = 0.3/M
gamma = 100.
kappa = 1. + delta*M
theta = [alpha, beta, delta, gamma, kappa]

xval = [0.35783767264345334, -0.2209858163853502, 0.07694316282590163, -0.5592493820167688, 0.45017948071235336, -0.13187935288390085, 0.08173972881194702, 0.4697604252725478, -0.7930020302715701, 0.636736964081372, 0.04159245658542776, -0.5011748638445761, -0.5322225987871061, -0.766410957379569, 0.958442692383763, 0.626177847931769, 0.610182863131767, 0.24255338002914106, -0.3249586529212869, 0.8402797620935014, 0.9584105409783232, 0.3838693989980384, 0.2554425489860863, 0.009636452399528928, 0.8633302667076528, 0.4971420481429969, 0.7620931640627635, -0.6214045967132642, 0.6885390087081147, -0.8332058405202001, -0.8445006788282816, 0.058018387092042545, 0.7448517737345446, -0.1102485355047067, 0.6091525003404037, 0.4703273849893166, -0.3186939596967906, -0.6148175125420252, -0.6271015543044971, 0.7662334229195285, 0.40033467437582004, -0.49266705127946153, 0.3845870729035221, 0.3956082166537964, 0.9261066681673142, 0.054700333751995345, 0.16057794939771997, -0.23830634738546452, 0.6484556249254738]

# println("Timing the Hessian function")
# @btime Potential.hessian(xval, orig, cost_mat, theta, N, M)
# println("Done")
#
# println("\nNow timing the Potential function (both value and gradient)")
# grad = Array{Float64,1}(undef, M)
# @btime Potential.potential!(grad, xval, orig, cost_adj, theta, N, M)
# println("Done")

println("\nTiming the Potential function - value only")
@btime Potential.potential_val(xval, orig, cost_adj, theta, N, M)
println("Done")

println("\nTiming the Potential function - gradient only")
grad = Array{Float64,1}(undef, M)
@btime Potential.potential_grad!(grad, xval, orig, cost_adj, theta, N, M)
println("Done")
