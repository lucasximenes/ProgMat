using JuMP, CPLEX, BPPLib, SparseArrays

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

function dantzig_wolfe_csp(combinations, I, capacity, weights, demands, info=false)
    n, m = size(combinations)

    m = Model(CPLEX.Optimizer)
    set_silent(m)
    set_time_limit_sec(m, 300)
    @variable(m, x[1:n] >= 0)

    @constraint(m, con[i in I], combinations[i, :]' * x >= demands[i])

    @objective(m, Min, sum(x))

    while true

        optimize!(m)

        duals = dual.(con)

        S = pricing(duals, weights, capacity)

        if S === nothing
            if info
                println("Finished adding variables")
                println("Solution to linear relaxation of master problem: $(objective_value(m))")
                println("X = ", value.(x))
            end
            break
        end

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

    return value.(x), objective_value(m), objective_bound(m)
end

function execute_tests()
    instances = [:PM_u010_01,
    :PM_u010_02,
    :PM_u010_03,
    :PM_u010_04,
    :PM_u010_05,
    :PM_u010_06,
    :PM_u010_07,
    :PM_u010_08,
    :PM_u010_09,
    :PM_u010_10,
    :PM_u020_01,
    :PM_u020_02,
    :PM_u020_03,
    :PM_u020_04,
    :PM_u020_05,
    :PM_u020_06,
    :PM_u020_07,
    :PM_u020_08,
    :PM_u020_09,
    :PM_u020_10,
    :PM_u030_01,
    :PM_u030_02,
    :PM_u030_03,
    :PM_u030_04,
    :PM_u030_05,
    :PM_u030_06,
    :PM_u030_07,
    :PM_u030_08,
    :PM_u030_09,
    :PM_u030_10,
    :PM_u060_01,
    :PM_u060_02,
    :PM_u060_03,
    :PM_u060_04,
    :PM_u060_05,
    :PM_u060_06,
    :PM_u060_07,
    :PM_u060_08,
    :PM_u060_09,
    :PM_u060_10,
    :PM_u120_01,
    :PM_u120_02,
    :PM_u120_03,
    :PM_u120_04,
    :PM_u120_05,
    :PM_u120_06,
    :PM_u120_07,
    :PM_u120_08,
    :PM_u120_09,
    :PM_u120_10]

    open("CuttingStockResults.txt", "w") do f


        for instance in instances
            write(f, "Solving instance $instance\n")
            data = loadCSP(instance)
            I, w, c, d, n = params(data)

            #gera padrões iniciais
            patterns = spzeros(Int, n, n)
            for i in I
                patterns[i, i] = floor(Int, min(c / w[i], d[i]))
            end

            time = @elapsed x, obj, bound = dantzig_wolfe_csp(patterns, I, c, w, d)
            if bound == obj
                write(f, "UB = LB =  $obj, time (s): $time\n")
            else
                write(f, "UB = $obj, LB = $bound, time (s): $time\n")
            end
            write(f, "==================================\n")
        end
    end
end

execute_tests()

data = loadCSP(:PM_u010_01)

I, w, c, d, n = params(data)

patterns = spzeros(Int, n, n)
for i in I
    patterns[i, i] = floor(Int, min(c / w[i], d[i]))
end

x, obj = dantzig_wolfe_csp(patterns, I, c, w, d)
