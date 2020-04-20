#!/usr/bin/env julia

using Plots
using DelimitedFiles

include("./src/Potential.jl")

println("Loading data..")
cost_mat = readdlm("data/london_n/cost_mat.txt")
cost_adj = convert(Array, cost_mat')
orig = readdlm("data/london_n/P.txt")

# Keep only 2 destination zones and normalise the matrix
cost_mat = cost_mat[:, 1:2] / sum(cost_mat[:, 1:2])

nn, mm = size(cost_mat)

alpha = 2.
beta = 1000
delta = 0.3/mm
gamma = 20.
kappa = 1. + delta*mm
theta = [alpha, beta, delta, gamma, kappa]

# potential for given parameter values
function valuePot(X, alpha)
    theta[1] = alpha
    grad = Array{Float64,1}(undef, mm)
    val, grad = Potential.potential!(grad, X, orig, cost_adj, theta, nn, mm)
    val
end


x = -4:(9/100):(.5 - 9/100)
y = -4:(9/100):(.5 - 9/100)

println("Creating figure..")
learr = []
for alpha in [0.5, 1., 1.5, 2.]
    zfun = (x,y) -> exp(-valuePot([x,y], alpha))
    push!(learr, contourf(x,y,zfun, xlabel="X1", ylabel="X2",
            title="alpha = $alpha ", colorbar=false))
end

plot(learr[1],
    learr[2],
    learr[3],
    learr[4],
    size=(650, 600)
)
savefig("images/2d_pot.png")
