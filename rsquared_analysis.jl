using Distributions
using Optim
using DelimitedFiles
using LinearAlgebra
using Plots

include("./src/Potential.jl")

println("Loading data..")
cost_mat = readdlm("data/london_n/cost_mat.txt")
cost_adj = convert(Array, cost_mat')
orig = readdlm("data/london_n/P.txt")
xd = readdlm("data/london_n/xd0.txt")
nn, mm = size(cost_mat)

# Set theta for low-noise model
alpha = 1.
beta = 0.
delta = 0.3/mm
gamma = 100.
kappa = 1.3

# initialise grid search
grid_n = 100
rsquaredArray = zeros(Float64, grid_n, grid_n)
x = [el for el in LinRange(0.1, 2, grid_n)]
y = [el for el in LinRange(0.1, 1.4e6, grid_n)]


# Total sum squares
w_data = map(exp, xd)
w_data_centred = w_data - ones(mm)*mean(w_data)
ss_tot = (w_data_centred' * w_data_centred)[1]

function run_rsquared(rsquaredArray)
    # Search values
    last_r2 = -Inf16
    for j in 1:grid_n
        println("computing rsquared for all y values and with x=$(round(x[j], digits=3)) (x in (0,2))\n")
            Threads.@threads for i in 1:grid_n
            theta = [x[j], y[i], delta, gamma, kappa]
            levaluePot = X -> Potential.potential_val(X, orig, cost_adj, theta, nn, mm)[1]
            legradPot! = (grad, X) -> Potential.potential_grad!(grad, X, orig, cost_adj, theta, nn, mm)[2]
            try
                f = optimize(levaluePot, legradPot!, xd[:,1], LBFGS())
                w_pred = map(exp, f.minimizer)
                res = w_pred - w_data
                ss_res = (res' * res)[1]

                # Regression sum squares
                RSS = 1. - ss_res/ss_tot
                rsquaredArray[i, j] = RSS
            finally
                0.
            end
            if rsquaredArray[i, j] == 0
                rsquaredArray[i, j] = last_r2
            else
                last_r2 = rsquaredArray[i, j]
            end
        end
    end
end

run_rsquared(rsquaredArray)

writedlm("outputs/rsquared_analysis.txt", rsquaredArray)

idx = findall(x -> x==maximum(rsquaredArray), rsquaredArray)[1]
x_min = round(x[idx[2]], digits=3)
y_min = round(y[idx[1]]* 2/1.4e6, digits=3)
R_2 = round(rsquaredArray[idx], digits=3)
println("Fitted alpha and beta values:\n")
println("alpha=$x_min, beta=$y_min. R_2=$R_2\n")


plot(contourf(x, y* 2/1.4e6, rsquaredArray, xlabel="alpha", ylabel="beta", title="Rsquared values"))
savefig("images/rsquared.png")
