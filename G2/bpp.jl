using BPPLib, JuMP, CPLEX, Combinatorics

data = loadBPP(:PM_u010_01)

function params(instance)
    n = length(instance.weights)
    I = 1:n
    U = 1:n
    w = instance.weights
    c = instance.capacity
    return I, U, w, c
end

I, U, w, c = params(data)

function gilmory_gomory(w, c, I, U)
    m = Model(CPLEX.Optimizer)
    set_silent(m)
    set_time_limit_sec(m, 60)
    
    @variable(m, x[I, U], Bin)
    @variable(m, y[U], Bin)

    @constraint(m, [i in I], sum(x[i, j] for j in U) == 1)
    @constraint(m, [j in U], sum(w[i]*x[i,j] for i in I) <= c*y[j])
    @constraint(m, [i in U; i > 1], y[i] <= y[i - 1])

    @objective(m, Min, sum(y[j] for j in U))
    optimize!(m)

    return value.(y), value.(x)
end

# @time bins, itens = gilmory_gomory(w, c, I, U)

function naive_dantzig_wolfe_bpp(w, c, I, U)
    Ss = powerset(U)
    Ss = filter(S -> length(S) == 0 ? false : sum(w[i] for i in S) <= c, collect(Ss))

    m = Model(CPLEX.Optimizer)
    @variable(m, λ[1:length(Ss)], Bin)

    @objective(m, Min, sum(λ[i] for i in 1:length(Ss)))

    @constraint(m, [i in I], sum(λ[j] for j in 1:length(Ss) if i in Ss[j]) == 1)

    optimize!(m)

    return value.(λ), Ss
end

# vals, Ss = naive_dantzig_wolfe_bpp(w, c, I, U)

function pricing(dual, weights, capacity)
    n = length(dual)
    m = Model(CPLEX.Optimizer)
    set_silent(m)
    @variable(m, λ[1:n], Bin)
    @objective(m, Max, sum(λ[i]*dual[i] for i in 1:n))

    @constraint(m, sum(weights[i]*λ[i] for i in 1:n) <= capacity)

    optimize!(m)
    
    S = []
    if 1 - objective_value(m) < -1e-5
        for i in I
            if value(λ[i]) > 1e-5
                push!(S, i)
            end
        end
    end
    
    return S

end

function dantzig_wolfe_bpp(combinations, I, capacity, weights)
    n = length(combinations)

    m = Model(CPLEX.Optimizer)
    set_silent(m)
    @variable(m, x[1:n] >= 0)

    @constraint(m, con[i in I], sum(x[j] for j in 1:n if i in combinations[j]) == 1)

    @objective(m, Min, sum(x[i] for i in 1:n))

    while true

        optimize!(m)

        # println("Number of bins: ", objective_value(m))

        duals = dual.(con)

        S = pricing(duals, weights, capacity)

        # println("Adding combination: $S")

        if length(S) == 0
            
            println("Finished adding variables")

            println("Solution to linear relaxation of master problem: ")
            println("X = ", value.(x))
            println("Objective value = ", objective_value(m))


            break
        end

        push!(combinations, S)

        push!(x, @variable(m, lower_bound = 0))

        set_objective_coefficient(m, x[end], 1.0)
        
        for item in S
            set_normalized_coefficient(con[item], x[end], 1)
        end

    end

    set_binary.(x)

    optimize!(m)

    return value.(x), objective_value(m)
end

combinations = [[i] for i in I]

@time x, obj = dantzig_wolfe_bpp(combinations, I, c, w)
