module PT
    using Distributions
    
    function swap_step(xx::Array{Float64, 2}, V::Array{Float64, 1}, gradV::Array{Float64, 2}, inverse_temps::Array{Float64, 1})
        temp_n = length(inverse_temps)
        j0 = rand(1:(temp_n-1))
        j1 = j0 + 1
        logA = (inverse_temps[j1] - inverse_temps[j0])*(-V[j1] + V[j0])
        if log(rand(Uniform(0,1))) < logA
            x0 = xx[j0, :]
            x1 = xx[j1, :]
            xx[j0, :] = x1
            xx[j1, :] = x0
            V0, V1 = V[j0], V[j1]
            V[j0] = V1
            V[j1] = V0
            gradV0 = gradV[j0,:]
            gradV1 = gradV[j1,:]
            gradV[j0,:] = gradV1
            gradV[j1,:] = gradV0

            xx, V, gradV, 1
        else
            xx, V, gradV, 0
        end

    end

end
