using Optim
using DelimitedFiles

include("./src/Potential.jl")

"Find global minimum of p(x | theta)"


cost_mat = readdlm("data/london_n/cost_mat.txt")
cost_adj = convert(Array, cost_mat')
orig = readdlm("data/london_n/P.txt")
xd = readdlm("data/london_n/xd0.txt")

nn, mm = size(cost_mat)

# Set theta for low-noise model
alpha = 0.5
beta = 0.3*0.7e6
delta = 0.3/mm
gamma = 10000.
kappa = 1.3

theta = [alpha, beta, delta, gamma, kappa]

# potential for given parameter values
function valuePot(X)
    Potential.potential_val(X, orig, cost_adj, theta, nn, mm)
end

function gradPot!(grad, X)
    Potential.potential_grad!(grad,X, orig, cost_adj, theta, nn, mm)
end


function optimise_model(m, delta, valuePot, gradPot!)
    """
    Parameters:
    ----------
    m: default minimizer for the potential function if no result is found
    delta: to determine the IC
    valuePot: potential function
    gradPot: gradient of potential function
    """
    f_val = Inf16
    for k in 1:mm
        g = log(delta)*ones(mm)
        g[k] = log(1+delta)
        f = optimize(valuePot, gradPot!, g, LBFGS())
        if f.minimum < f_val
            f_val = f.minimum
            m = f.minimizer
        end
    end
    m
end
m = optimise_model(xd, delta, valuePot, gradPot!)

writedlm("outputs/opt$alpha.txt", m)
