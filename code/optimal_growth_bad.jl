using Random, Parameters, Distributions, Plots
Random.seed!(123) # Setting the seed

@with_kw struct Primitives
    β::Float64 = 0.99          # discount factor
    θ::Float64 = 0.36          # production
    δ::Float64 = 0.025         # depreciation

    Z::Array{Float64, 1} = [1.25, 0.2]    # Productivity levels
    transition_matrix::Array{Float64, 2} = [0.977 0.023; 0.074 0.926] # Transition matrix

    k_grid::Array{Float64, 1} = collect(range(0.1, length=1000, stop=45.0)) # capital grid
    nk::Int64 = length(k_grid) # number of capital grid states
end

mutable struct Results
    val_func::Array{Float64,2} #value function
    pol_func::Array{Float64,2} #policy function
end

function Solve_model()
    #initialize primitives and results
    prim = Primitives() # Calls on primitives struct
    val_func, pol_func = zeros(prim.nk, 2), zeros(prim.nk, 2) 
    res = Results(val_func, pol_func)   

    error, n = 100, 0
    while error>eps() #loop until convergence
        n+=1
        v_next = Bellman(prim, res) #next guess of value function

        error = maximum(abs.(v_next .- res.val_func)) #check for convergence, v_next is based on a contraction mapping 
        # that on stokey lucas 
        res.val_func = v_next #update

        #println("Current error: ", error)
        if mod(n, 5000) == 0 || error < eps()
            println(" ")
            println("*************************************************")
            println("AT ITERATION = ", n)
            println("MAX DIFFERENCE = ", error)
            println("*************************************************")
        end
    end
    prim, res
end

function Bellman(prim::Primitives, res::Results)
    @unpack β, δ, θ, nk, k_grid, Z, transition_matrix = prim # unpack primitive structure
    v_next = zeros(nk, 2) # preallocate next guess of value function

    for i_k = 1:nk # Looping over the capital state space to find optimality
        for i_z = 1:length(Z) # Loop over productivity states
            max_util = -1e10 # Start very low
            k = k_grid[i_k] # Current capital level
            
            # Calculate budget based on the current state of productivity
            budget = Z[i_z] * k^θ + (1 - δ) * k 

            for kp_index = 1:nk         
                k_prime = k_grid[kp_index] # Choosing a K' from the k_grid
                c = budget - k_prime  # Consumption

                if c > 0  # Check if consumption is positive
                    # Calculate utility for this productivity state
                    transition_prob = transition_matrix[i_z, :] # Get transition probabilities
                    # Using the Bellman equation with the transition matrix
                    val = log(c) + β * (
                        transition_prob[1] * res.val_func[kp_index] + 
                        transition_prob[2] * res.val_func[kp_index]
                    )

                    if val > max_util  # Update if better utility found
                        max_util = val
                        res.pol_func[i_k, i_z] = k_prime # Update policy function for current state
                    end
                end
            end 
            v_next[i_k, i_z] = max_util # Update value function for current state
        end
    end 
    return v_next
end

@elapsed prim, res = Solve_model() #
Plots.plot(prim.k_grid, res.val_func[:, 1], label="Z_g", legend=:topright, xaxis = "k-grid", yaxis = "Value")
Plots.plot!(prim.k_grid, res.val_func[:, 2], label="Z_b", legend=:topright)

Plots.plot(prim.k_grid, res.pol_func[:, 1], label="Z_g", legend=:topright, xaxis = "k-grid", yaxis = "policy k")
Plots.plot!(prim.k_grid, res.pol_func[:, 2], label="Z_b", legend=:topright)

