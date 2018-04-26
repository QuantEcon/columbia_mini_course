function compute_fixed_point_multithread(self::JS, 
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
        Q_multithread(f_in, f_out, params, self.z_grid, self.e_draws)
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

function Q_multithread(f_in, f_out, params, z_grid, e_draws)

    μ, s, d, ρ, σ, β, c = params
    M = size(e_draws,2)

    # For every grid point
    Threads.@threads for i=1:length(z_grid)

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



