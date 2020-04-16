module MCMC_full
    using Distributions

    # Potential function of the likelihood (so this is actually -log_likelihood)
    function likeValue(xx::Array{Float64,1}, xd::Array{Float64,1})
        s2_inv = 100.
        diff = xx - xd
        grad = s2_inv*diff
        pot = 0.5*s2_inv * diff'*diff
        [pot, grad]
    end
end
