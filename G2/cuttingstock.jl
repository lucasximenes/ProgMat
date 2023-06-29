using JuMP, CPLEX, BPPLib, SparseArrays

data = loadCSP(:PM_u120_01)


function params(instance)
    n = length(instance.weights)
    I = 1:n
    w = instance.weights
    c = instance.capacity
    d = instance.demands
    return I, w, c, d, n
end

function pricing(dual, widths, roll_width)
    n = length(dual)
    m = Model(CPLEX.Optimizer)
    
    set_silent(m)
    
    @variable(m, λ[1:n] >= 0, Int)
    
    @objective(m, Max, sum(λ[i]*dual[i] for i in 1:n))

    @constraint(m, sum(widths[i]*λ[i] for i in 1:n) <= roll_width)

    optimize!(m)
    
    if 1 - objective_value(m) < -1e-5
        return value.(λ)
    end
    
    return nothing

end

function dantzig_wolfe_csp(combinations, I, capacity, weights, demands)
    n, m = size(combinations)

    m = Model(CPLEX.Optimizer)
    set_silent(m)
    @variable(m, x[1:n] >= 0)

    @constraint(m, con[i in I], combinations[i, :]' * x >= demands[i])

    @objective(m, Min, sum(x))

    while true

        optimize!(m)

        duals = dual.(con)

        S = pricing(duals, weights, capacity)

        if S === nothing
            println("Finished adding variables")
            println("Solution to linear relaxation of master problem: ")
            println("X = ", value.(x))
            println("Objective value = ", objective_value(m))
            break
        end
        
        # println("Adding combination: $S")

        combinations = [combinations S]

        push!(x, @variable(m, lower_bound = 0))

        set_objective_coefficient(m, x[end], 1.0)
        
        for i in I
            if S[i] > 0
                set_normalized_coefficient(con[i], x[end], S[i])
            end
        end

    end

    set_integer.(x)

    optimize!(m)

    return value.(x), objective_value(m)
end

I, w, c, d, n = params(data)

patterns = spzeros(Int, n, n)
for i in I
    patterns[i, i] = floor(Int, min(c / w[i], d[i]))
end

patterns

x, obj = dantzig_wolfe_csp(patterns, I, c, w, d)
