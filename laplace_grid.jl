using Distributed
addprocs(7)
@everywhere using Optim
@everywhere using DelimitedFiles
@everywhere using LinearAlgebra
using Plots

@everywhere include("./src/Potential.jl")

"Evaluates p(x | theta) over grid of alpha and beta values"


@everywhere cost_mat = readdlm("data/london_n/cost_mat.txt")
@everywhere cost_adj = convert(Array, cost_mat')
@everywhere orig = readdlm("data/london_n/P.txt")
@everywhere xd = readdlm("data/london_n/xd0.txt")
@everywhere nn, mm = size(cost_mat)

# Set theta for low-noise model
@everywhere alpha = 1
@everywhere beta = 0
@everywhere delta = 0.3/mm
@everywhere gamma = 100
@everywhere kappa = 1.3

@everywhere theta = [alpha, beta, delta, gamma, kappa]


@everywhere function optimise_model(m, delta, valuePot, gradPot!)
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
    for k in 1:mm
        g = log(delta)*ones(mm)
        g[k] = log(1+delta)
        f = optimize(valuePot, gradPot!, g, LBFGS())
        if f.minimum < f_val
            f_val = f.minimum
            m = f.minimizer
            min_value = f.minimum
        end
    end
    m, min_value
end

@everywhere function estimatePosterior(alpha, beta)
    # println("post for (alpha, beta) = ($alpha, $beta)")
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

# lala = estimatePosterior(0.5, (1.4e6)/2)
# println("$lala")

@everywhere grid_n = 5
x = [el for el in LinRange(0.1,2, grid_n)]
@everywhere y = [el for el in LinRange(0.1,1.4e6, grid_n)]


# With threads: Run `JULIA_NUM_THREADS=4 julia laplace_grid.jl`
# println("Running grid search:\n")
# Threads.@threads for xval in x
#     lala = estimatePosterior(xval, (1.4e6)/2)
#     println("Thread $(Threads.threadid()): ($xval, $((1.4e6)/2)) has likelihood $lala\n")
# end

# Distributed
println("Running grid search:\n")
laplaceArray = Array{Float64, 2}(undef, grid_n, grid_n)
for j in 1:grid_n
    println("computing log-lik for all x values and with y=$(round(y[j]/1.4e6, digits=3)) (y in (0,1))\n")
    laplaceArray[:,j] = pmap(xval -> estimatePosterior(xval, x[j]), x)
end
writedlm("outputs/laplace_analysis$(gamma).txt", laplaceArray)


# laplaceArray = readdlm("outputs/laplace_analysis100.txt")
# x = [el for el in LinRange(0.1,2, grid_n)]
# modifiedy = [el for el in LinRange(0.1,1.4e6, grid_n)] * 2/1.4e6
# contourf(x,y, modifiedy)
savefig("images/laplace_grid$(gamma).png")
# plot(laplaceArray)
