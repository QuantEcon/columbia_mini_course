mutable struct JS
    μ::Float64 
    s::Float64
    d::Float64
    ρ::Float64
    σ::Float64
    β::Float64
    c::Int64
    mc_size::Int64
    grid_size::Int64
    z_mean::Float64
    z_sd::Float64
    k::Int64
    a::Float64
    b::Float64
    z_grid::Vector{Float64}
    e_draws::Array{Float64,2}
    f_star::Vector{Float64}
end

function JS(μ=0.0, 
            s=1.0, 
            d=0.0, 
            ρ=0.9, 
            σ=0.1, 
            β=0.98, 
            c=5,
            mc_size=5000,
            grid_size=200)
    
    z_mean = d / (1 - ρ)
    z_sd = sqrt(σ / (1 - ρ^2))
    k = 3 
    a, b = z_mean - k * z_sd, z_mean + k * z_sd
    z_grid = collect(linspace(a, b, grid_size))

    e_draws = randn(2, mc_size)


    f_star = zeros(0)

    JS(μ, s ,d, ρ, σ, β,c,  mc_size, grid_size, z_mean, z_sd, k,
        a, b, z_grid, e_draws, f_star)   
end




self = JS()

function pack_parameters(self::JS)
    return  self.μ, self.s, self.d, self.ρ, self.σ, self.β, self.c
end

function compute_fixed_point(self::JS, 
                             tol=1e-4, 
                             max_iter=1000, 
                             verbose= true,
                             print_skip=25)
    # Set initial condition
    f_init = log(self.c) * ones(length(self.z_grid))
    f_out = similar(self.z_grid)

    # Set up loop
    params = pack_parameters(self)
    f_in = f_init
    i = 0
    error = tol + 1

    while i < max_iter && error > tol
        Q(f_in, f_out, params, self.z_grid, self.e_draws)
        error = maximum(abs.(f_in - f_out))
        i += 1
        if i % print_skip == 0
            println("Error at iteration",i," is ", error)
        end
        f_in[:] = f_out
    end

    if i == max_iter
        print("Failed to converge!")
    end

    if verbose && i < max_iter
        print("\nConverged in ", i," iterations.")
    end

    self.f_star = f_out
end

function Q(f_in, f_out, params, z_grid, e_draws)

    μ, s, d, ρ, σ, β, c = params
    M = size(e_draws,2)

    # For every grid point
    for i=1:length(z_grid)

        z = z_grid[i]

        # Compute expectation by MC
        expectation = 0.0
        for m=1:M
            e1, e2 = e_draws[:, m]
            z_next = d + ρ * z + σ * e1
            go_val = interp1d(z_grid, f_in, z_next)      # f(z') draw
            y_next = exp(μ + s * e2)                     # y' draw
            w_next = exp(z_next) + y_next                # w' draw
            stop_val = log(w_next) / (1 - β)             # u(w') / (1 - β)
            expectation += max(stop_val, go_val)
        end

        expectation = expectation / M 

        f_out[i] = log(c) + β * expectation
    end
end



