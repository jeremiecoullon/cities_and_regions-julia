using DelimitedFiles
using Distributions

function stoppingFun()
    "To generate random stopping times from P(K > k) = 1./k^1.1"
    stoppingArray = Array{Int64}(undef, 20000)

    for i in 1:20000
        N = 1
        k_pow = 1.1
        u = rand(Uniform(0, 1))
        while(u < ((N+1)^(-k_pow)))
            N += 1
        end
        stoppingArray[i] = N

    end
    stoppingArray
end

stoppingArray = stoppingFun()
writedlm(joinpath(@__DIR__, "../data/stoppingArray.txt"), stoppingArray)
