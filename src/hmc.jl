module HMC_Prior
    using Distributions

    function hmc_step(xCurrent::Array{Float64, 1}, U_fun::Function, V::Float64, gradV::Array{Float64, 1}, inv_temp::Float64, eps::Float64, L::Int64, mm::Int64)
        """
        HMC step for a given inverse temperature.
        Uses a standard Normal for the Kinetic Energy
        """

        # Initialize leapfrog integrator for HMC proposal
        p = rand(Normal(0.,1.), mm)
        # current value of Hamiltonian
        H = 0.5*p'*p + inv_temp*V

        # X-proposal
        x_p = xCurrent
        p_p = p
        V_p, gradV_p = V, gradV
        for l in 1:L
            p_p = p_p - 0.5*eps*inv_temp*gradV_p
            x_p = x_p + eps*p_p
            V_p, gradV_p = U_fun(x_p)
            p_p = p_p - 0.5*eps*inv_temp*gradV_p
        end

        # X-accept/reject
        H_p = 0.5*p_p'*p_p + inv_temp*V_p
        if log(rand(Uniform(0,1))) < H - H_p
            x_p, V_p, gradV_p, 1
        else
            xCurrent, V, gradV, 0
        end
    end
end
