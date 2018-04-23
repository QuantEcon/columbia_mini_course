
using QuantEcon
using Distributions


mutable struct SavingsProblem

    β::Float64 
    γ::Float64 
    ρ::Float64 
    d::Float64 
    σ::Float64 
    r::Float64 
    w::Float64 
    z_grid_size::Int64
    x_grid_size::Int64
    x_grid_max::Int64
    R::Float64
    Q::Array{Float64,2}
    z_grid::Vector{Float64}
    x_grid::Vector{Float64}
end
    
function SavingsProblem(β=0.96,
                        γ=2.5,
                        ρ=0.9,
                        d=0.0,
                        σ=0.1,
                        r=0.04,
                        w=1.0,
                        z_grid_size=25,
                        x_grid_size=200,
                        x_grid_max=15)
    
    R = 1 + r
    mc = rouwenhorst(z_grid_size, d, σ, ρ)
    Q = mc.p
    z_grid = exp.(mc.state_values)
    x_grid = collect(linspace(0.0, x_grid_max, x_grid_size))
    
    SavingsProblem(β, γ, ρ, d, σ, r, w, z_grid_size, x_grid_size, x_grid_max, R, Q, z_grid, x_grid)
    
end 




self = SavingsProblem()

function pack_parameters(self::SavingsProblem)
    return self.β, self.γ, self.R, self.w, self.Q, self.x_grid, self.z_grid
end

function u(c, γ)
    return (c + 1e-10)^(1 - γ) / (1 - γ)
end

function value_function_iteration(self::SavingsProblem,
                             tol=1e-4, 
                             max_iter=1000, 
                             verbose=true,
                             print_skip=25)
    # Set initial condition, set up storage
    v_init = ones(self.x_grid_size, self.z_grid_size)
    v_out = similar(v_init)
    π = similar(trunc.(Int, v_init))
    # Set up loop
    params = pack_parameters(self)
    v_in = v_init
    i = 0
    error = tol + 1

    while i < max_iter && error > tol
        T(v_in, v_out, π, params)
        error = maximum(abs.(v_in - v_out))
        i += 1
        if i % print_skip == 0
            println("Error at iteration", i, " is ", error)
        end
        v_in[:] = v_out
    end
        

    if i == max_iter 
        print("Failed to converge!")
    end

    if verbose && i < max_iter
        print("\nConverged in ", i," iterations.")
    end

    return v_out, π
end



function T(v, v_out, π, params)
    
    """
    Given v, compute Tv and write it to v_out.

    At the same time, compute the v-greedy policy and write it to π

    """
    n, m = size(v)
    β, γ, R, w, Q, x_grid, z_grid = params
    k_star = 1
    for j=1:m
        z = z_grid[j]

        for i=1:n
    
            x = x_grid[i]

            # Cash in hand at start of period
            y = R * x + w * z  
            # A variable to store largest recorded value
            max_so_far = - Inf
            # Find largest x_grid index s.t. x' <= y
            #idx = searchsortedfirst(x_grid, y)-1
            idx = searchsortedlast(x_grid, y)
            # Step through x' with 0 <= x' <= y, find max
            for k=1:idx
                x_next = x_grid[k]
                val = u(y - x_next, γ) + β * sum(v[k, :] .* Q[j, :])

                if val > max_so_far
                    max_so_far = val
                    k_star = k
                end

            end

            π[i, j] = k_star 

            v_out[i, j] = max_so_far


        end
    end
end
      
