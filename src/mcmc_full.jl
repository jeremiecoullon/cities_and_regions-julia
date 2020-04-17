module MCMC_full
    using Distributions
    mm = 49

    function pot0Value(xx::Array{Float64, 1})
        delta = 0.3/mm
        gamm = 100.
        kk = 1.3
        gamm_kk_exp_xx = gamm*kk*map(exp, xx)
        gradV = -gamm*(delta + 1. /mm)*ones(mm) + gamm_kk_exp_xx
        V = -gamm*(delta + 1. /mm)*sum(xx) + sum(gamm_kk_exp_xx)
        V, gradV
    end

    # Potential function of the likelihood (so this is actually -log_likelihood)
    function likeValue(xx::Array{Float64,1}, xd::Array{Float64,1})
        s2_inv = 100.
        diff = xx - xd
        grad = s2_inv*diff
        pot = 0.5*s2_inv * diff'*diff
        [pot, grad]
    end

    function logsumexp(xx)
        x_max = maximum(xx)
        x_max + log(sum(map(x -> exp(x-x_max), xx)))
    end
end
