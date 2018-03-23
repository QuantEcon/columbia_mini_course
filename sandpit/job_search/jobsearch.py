"""

Job search with persistent and transitory components to wages.

Wages are given by

    w = exp(z) + y

    y ~ exp(μ + s ζ)  

    z' = d + ρ z + σ ε

with ζ and ε both iid and N(0, 1).   The value function is

    v(w, z) = max{ u(w) / (1-β), u(c) + β E v(w', z')}

The continuation value function satisfies

    f(z) = u(c) + β E max{ u(w') / (1-β), f(z') }

From f we can solve the optimal stopping problem by stopping when

    u(w) / (1-β) > f(z)

For utility we take u(c) = ln(c).  The reservation wage is the wage where
equality holds, or

    w^*(z) = exp(f^*(z) (1-β))

Our aim is to solve for the reservation rule.  We do this by first computing
f^* as the fixed point of the contraction map

    Qf(z) = u(c) + β E max{ u(w') / (1-β), f(z') }

When we iterate, f is stored as a vector of values on a grid and these points
are interpolated into a function as necessary.

Interpolation is piecewise linear.

The integral in the definition of Qf is calculated by Monte Carlo.

"""

import numpy as np
from numpy.random import randn
from lininterp import interp1d
from numba import jit, prange

class JobSearch:

    def __init__(self,
                 μ=0.0, 
                 s=1.0, 
                 d=0.0, 
                 ρ=0.9, 
                 σ=0.1, 
                 β=0.98, 
                 c=5,
                 mc_size=5000,
                 grid_size=200):

        self.μ, self.s, self.d, self.ρ, self.σ, self.β, self.c = \
            μ, s, d, ρ, σ, β, c 

        # Set up grid
        z_mean = d / (1 - ρ)
        z_sd = np.sqrt(σ / (1 - ρ**2))
        k = 3  # Number of standard devations from mean
        a, b = z_mean - k * z_sd, z_mean + k * z_sd
        self.z_grid = np.linspace(a, b, grid_size)

        # Store shocks
        self.mc_size = mc_size
        self.e_draws = randn(2, mc_size)

        # Store the continuation value function after it's computed
        self.f_star = None

    def pack_parameters(self):
        return self.μ, self.s, self.d, self.ρ, self.σ, self.β, self.c

    def compute_fixed_point(self, 
            tol=1e-4, 
            max_iter=1000, 
            verbose=True,
            print_skip=25): 

        # Set initial condition
        f_init = np.log(self.c) * np.ones(len(self.z_grid))
        f_out = np.empty_like(self.z_grid)

        # Set up loop
        params = self.pack_parameters()
        f_in = f_init
        i = 0
        error = tol + 1

        while i < max_iter and error > tol:
            Q(f_in, f_out, params, self.z_grid, self.e_draws)
            error = np.max(np.abs(f_in - f_out))
            i += 1
            if i % print_skip == 0:
                print(f"Error at iteration {i} is {error}.")
            f_in[:] = f_out

        if i == max_iter: 
            print("Failed to converge!")

        if verbose and i < max_iter:
            print(f"\nConverged in {i} iterations.")

        self.f_star = f_out


@jit(nopython=True, parallel=True)
def Q(f_in, f_out, params, z_grid, e_draws):

    μ, s, d, ρ, σ, β, c = params
    M = e_draws.shape[1]

    # For every grid point
    for i in prange(len(z_grid)):

        z = z_grid[i]

        # Compute expectation by MC
        expectation = 0.0
        for m in range(M):
            e1, e2 = e_draws[:, m]
            z_next = d + ρ * z + σ * e1
            go_val = interp1d(z_grid, f_in, z_next)      # f(z') draw
            y_next = np.exp(μ + s * e2)                  # y' draw
            w_next = np.exp(z_next) + y_next             # w' draw
            stop_val = np.log(w_next) / (1 - β)          # u(w') / (1 - β)
            expectation += max(stop_val, go_val)

        expectation = expectation / M 

        f_out[i] = np.log(c) + β * expectation



