using JuMP, CPLEX, CVRPLIB, CVRPSEP, Graphs, MathOptInterface

cvrp, _, _ = readCVRPLIB("E-n13-k4")

cvrp.capacity

function solve(instance)

    C = instance.capacity
    n = instance.dimension
    d = instance.demand
    w = instance.weights


    m = Model(CPLEX.Optimizer)
    set_silent(m)

    @variable(m, x[1:n, 1:n], Bin)

    @variable(m, K >= 4, Int)

    @objective(m, Min, sum(x[i,j]*w[i,j] for i in 1:n, j in 1:n))

    ## flow conservation
    # @constraint(m, [i in 2:n], sum(x[i, j] for j in 1:n) == 1)
    # @constraint(m, [i in 2:n], sum(x[j, i] for j in 1:n) == 1)
    @constraint(m, [i in 2:n], sum(x[i, j] for j in 2:n) == 1)
    @constraint(m, [i in 2:n], sum(x[j, i] for j in 2:n) == 1)

    ## vehicles leaving depot
    @constraint(m, sum(x[1, i] for i in 1:n) == K)
    @constraint(m, sum(x[i, 1] for i in 1:n) == K)

    ## capacity constraint
    cut_manager = CutManager()

    function separateCapacityCuts(cb_data)
        println("callback foi chamado")

        edge_tail = Int64[]
        edge_head = Int64[]
        edge_x = Float64[]

        mat = callback_value.(cb_data, x)
        
        for i in 2:n, j in 2:n
            if mat[i,j] > 1e-5
                push!(edge_tail, i)
                push!(edge_head, j)
                push!(edge_x, mat[i,j])
            end
        end


        SS, rhs = rounded_capacity_inequalities!(cut_manager, d, C, edge_tail, edge_head, edge_x,
        integrality_tolerance = 1e-4, max_n_cuts = 1000)

        @show SS
        @show rhs

        for (index, subtour) in enumerate(SS)
            # lhs = 0.0
            # for i in subtour, j in subtour
            #     if i != j
            #         lhs += mat[i,j]
            #     end
            # end
            # subtour_size = length(subtour)
            # if lhs > subtour_size - 1
            st_size = length(subtour)
            ks = st_size/C
            println("Comparando ks e rhs", ks, rhs[index])
            con = @build_constraint(sum(x[i,j] for i in subtour, j in subtour if i != j) <= rhs[index])
            MOI.submit(m, MOI.LazyConstraint(cb_data), con)
            # end

        end

    end

    set_attribute(m, MOI.LazyConstraintCallback(), separateCapacityCuts)
    
    optimize!(m)

    if termination_status(m) != MathOptInterface.OPTIMAL
        println("Valor ótimo não encontrado")
        return
    end
    return value.(x), value(K), objective_value(m)

end

x, K, obj = solve(cvrp)

x
obj