
function interp1d(grid, vals, x)

    a, b, G = minimum(grid), maximum(grid), length(grid)

    s = (x - a) / (b - a)

    q_0 = max(min(trunc.(Int, s * (G - 1)), (G - 2)), 1)
    v_0 = vals[q_0]
    v_1 = vals[q_0 + 1]

    λ = s * (G - 1) - q_0

    return (1 - λ) * v_0 + λ * v_1
end

function interp1d_vectorized(grid, vals, x_vec)
    
    out = similar(x_vec)

    for (i, x) in enumerate(x_vec)
        out[i] = interp1d(grid, vals, x)
    end
    
    return out
end


