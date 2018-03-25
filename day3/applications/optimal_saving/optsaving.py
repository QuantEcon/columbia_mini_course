"""

A simple optimal savings problem.  The Bellman equation is

    v(x, z) = max_x' { u(R x + w z - x') + β E v(x', z')}

where 0 <= x' <= R x + w z and

    E v(x', z') = Σ_{z'} v(x', z') Q(z, z')

We take 

    u(c) = c^{1 - γ} / (1 - γ)

and obtain the transition kernel p by discretizing

    log z' = ρ log z + d + σ η
    
using Rouwenhorst's method.

"""

import numpy as np
import quantecon as qe
from numba import jit, prange


@jit(nopython=True)
def u(c, γ):
    return (c + 1e-10)**(1 - γ) / (1 - γ)


class SavingsProblem:

    def __init__(self, 
                 β=0.96,
                 γ=2.5,
                 ρ=0.9,
                 d=0.0,
                 σ=0.1,
                 r=0.04,
                 w=1.0,
                 z_grid_size=25,
                 x_grid_size=200,
                 x_grid_max=15):

        self.β, self.γ = β, γ
        self.R = 1 + r
        self.w = w
        self.z_grid_size, self.x_grid_size = z_grid_size, x_grid_size

        mc = qe.rouwenhorst(z_grid_size, d, σ, ρ)
        self.Q = mc.P
        self.z_grid = np.exp(mc.state_values)

        self.x_grid = np.linspace(0.0, x_grid_max, x_grid_size)

    def pack_parameters(self):
        return self.β, self.γ, self.R, self.w, self.Q, self.x_grid, self.z_grid

    def value_function_iteration(self, 
                                tol=1e-4, 
                                max_iter=1000, 
                                verbose=True,
                                print_skip=25): 

        # Set initial condition, set up storage
        v_in = np.ones((self.x_grid_size, self.z_grid_size))
        v_out = np.empty_like(v_in)
        π = np.empty_like(v_in, dtype=np.int)

        # Set up loop
        params = self.pack_parameters()
        i = 0
        error = tol + 1

        while i < max_iter and error > tol:
            T(v_in, v_out, π, params)
            error = np.max(np.abs(v_in - v_out))
            i += 1
            if i % print_skip == 0:
                print(f"Error at iteration {i} is {error}.")
            v_in[:] = v_out

        if i == max_iter: 
            print("Failed to converge!")

        if verbose and i < max_iter:
            print(f"\nConverged in {i} iterations.")

        return v_out, π


@jit(nopython=True, parallel=True)
def T(v, v_out, π, params):
    """
    Given v, compute Tv and write it to v_out.

    At the same time, compute the v-greedy policy and write it to π

    """
    n, m = v.shape
    β, γ, R, w, Q, x_grid, z_grid = params

    for j in prange(m):
        z = z_grid[j]

        for i in range(n):
            x = x_grid[i]

            # Cash in hand at start of period
            y = R * x + w * z  
            # A variable to store largest recorded value
            max_so_far = - np.inf

            # Find largest x_grid index s.t. x' <= y
            idx = np.searchsorted(x_grid, y)

            # Step through x' with 0 <= x' <= y, find max
            for k in range(idx):
                x_next = x_grid[k]
                val = u(y - x_next, γ) + β * np.sum(v[k, :] * Q[j, :])
                if val > max_so_far:
                    max_so_far = val
                    k_star = k

            π[i, j] = k_star
            v_out[i, j] = max_so_far


