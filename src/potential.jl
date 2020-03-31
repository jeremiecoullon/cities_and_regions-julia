module Potential

    function potential(X::Array{Float64,1}, orig::Array{Float64,2},
            cost_mat::Array{Float64,2}, theta::Array{Float64,1}, N::Int64, M::Int64)
            """
            Returns the value of the potential as well as the gradient
            """
        α, β, δ, γ, κ = theta
        value_sum_N = 0
        grad_sum_N = zeros(M)
        cost_adj = cost_mat'

        for i in 1:N
            sum_M = 0
            for j in 1:M
                sum_M += exp(α*X[j] - β*cost_adj[j,i])
            end
            value_sum_N += orig[i]*log(sum_M)
            grad_top_term = orig[i]*(α*map(exp, α*X - β*cost_mat[i, :]))
            grad_sum_N += grad_top_term / sum_M
        end
        value_part1 = -(1/α)*value_sum_N
        value_part2 = κ * sum(map(x -> exp(x), X))
        value_part3 = - δ * sum(X)

        grad_part1 = -(1/α)*grad_sum_N
        grad_part2 = κ * map(x -> exp(x), X)
        grad_part3 = - δ * ones(M)

        value = γ * (value_part1 + value_part2 + value_part3)
        grad = γ * (grad_part1 + grad_part2 + grad_part3)
        [value, grad]
    end


    function hessian(X::Array{Float64,1}, orig::Array{Float64,2},
                    cost_mat::Array{Float64,2}, theta::Array{Float64,1}, N::Int64, M::Int64)
            """
            Returns the Hessian matrix
            """
        α, β, δ, γ, κ = theta
        hess = zeros(Float64, M, M)
        alpha_inv = 1/α
        cost_adj = cost_mat'

        for l in 1:M
            for k in l:M
                B = 0
                if k==l
                    B = κ*exp(X[k])
                end
                sum_N = 0
                for i in 1:N
                    sum_M = 0
                    for j in 1:M
                        sum_M += exp(α*X[j] - β*cost_adj[j,i])
                    end
                    exp_val = exp(α*X[k] - β*cost_mat[i,k])
                    if k==l
                        C = α^2 * exp_val * (sum_M - exp_val)
                    else
                        C = -α^2 * exp_val * exp(α*X[l] - β*cost_mat[i,l])
                    end
                    A = C/(sum_M^2)
                    sum_N += orig[i]*A
                end
                hess[k,l] = γ * (-alpha_inv * sum_N + B)
                if k!=l
                    hess[l,k] = hess[k,l]
                end
            end
        end
        hess
    end

end