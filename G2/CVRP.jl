using JuMP, CPLEX, CVRPLIB, CVRPSEP, Graphs, MathOptInterface

function find_subtours(mat::Matrix{Float64})
    mat = round.(mat)
    graph = SimpleDiGraph(mat)
    return connected_components(graph)
end


function solve(instance)

    C = instance.capacity
    n = instance.dimension
    d = instance.demand
    w = instance.weights


    m = Model(CPLEX.Optimizer)
    set_silent(m)
    @variable(m, x[1:n, 1:n], Bin)

    @variable(m, K >= 0, Int)

    @objective(m, Min, sum(x[i,j]*w[i,j] for i in 1:n, j in 1:n))

    ## flow conservation
    @constraint(m, [i in 2:n], sum(x[i, j] for j in 1:n if i != j) == 1)
    @constraint(m, [i in 2:n], sum(x[j, i] for j in 1:n if i != j) == 1)
    
    ## vehicles entering/leaving depot
    @constraint(m, sum(x[1, i] for i in 1:n) == K)
    @constraint(m, sum(x[i, 1] for i in 1:n) == K)
    
    @constraint(m,[i in 1:n], x[i,i] == 0)

    ## capacity constraint
    cut_manager = CutManager()

    function separateCapacityCuts(cb_data)

        status = callback_node_status(cb_data, m)
        
        if status == MOI.CALLBACK_NODE_STATUS_INTEGER

            edge_tail = Int64[]
            edge_head = Int64[]
            edge_x = Float64[]

            mat = callback_value.(cb_data, x)
            
            for i in 1:n, j in 1:n
                if mat[i,j] > 1e-5
                    push!(edge_tail, i)
                    push!(edge_head, j)
                    push!(edge_x, mat[i,j])
                end
            end

            SS, rhs = rounded_capacity_inequalities!(cut_manager, d, C, edge_tail, edge_head, edge_x,
            integrality_tolerance = 1e-4, max_n_cuts = 1000)

            for (index, subtour) in enumerate(SS)
                rhs_aux = ceil(sum(d[i] for i in subtour)/C)

                not_in_subtour = setdiff(collect(1:n), subtour)

                lhs = 0.0
                for i in subtour, j in not_in_subtour
                        lhs += mat[i,j]
                end

                con = @build_constraint(sum(x[i,j] for i in subtour, j in not_in_subtour) >= rhs_aux)
                MOI.submit(m, MOI.LazyConstraint(cb_data), con)
            end
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

cvrp, _, _ = readCVRPLIB("E-n31-k7")

x, K, obj = solve(cvrp)

if K == 7 && abs(obj - 379) < 1e-5
    println("Teste passou")
else
    println("Teste falhou")
end 
