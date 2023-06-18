using TSPLIB, JuMP, HiGHS, MathOptInterface, Graphs, CPLEX, CVRPSEP

tsp_tokens = [:burma14, :ulysses16, :gr17, :gr21]

function MTZ_TSP(instance::TSP)
    n = instance.dimension
    
    m = Model(CPLEX.Optimizer)
    set_silent(m)
    @variable(m, x[1:n, 1:n], Bin)
    @variable(m, u[1:n], Int)
    
    @constraint(m, [i in 1:n], sum(x[i, j] for j in 1:n if j != i) == 1)
    @constraint(m, [j in 1:n], sum(x[i, j] for i in 1:n if i != j) == 1)
    @constraint(m, [i in 2:n, j in 2:n], u[i] - u[j] + 1 <= n*(1 - x[i,j]))
    @constraint(m, u[1] == 1) ## não precisa, 
    # restrição acima só torna u[1..n] tomar valores entre 1 e n, mas isso é irrelevante
    
    @objective(m, Min, sum(instance.weights[i,j]*x[i,j] for i in 1:n, j in 1:n if i != j))
    
    optimize!(m)
    return value.(x), objective_value(m)
end

function find_subtours(mat::Matrix{Float64})
    graph = SimpleDiGraph(mat)
    return connected_components(graph)
end


function lazy_constraint_callback_TSP(instance::TSP)
    n = instance.dimension
    m = Model(CPLEX.Optimizer)
    set_silent(m)
    @variable(m, x[1:n, 1:n], Bin)
    @constraint(m, [i in 1:n], sum(x[i, j] for j in 1:n if j != i) == 1)
    @constraint(m, [j in 1:n], sum(x[i, j] for i in 1:n if i != j) == 1)
    @objective(m, Min, sum(instance.weights[i,j]*x[i,j] for i in 1:n, j in 1:n if i != j))

    cut_manager = CutManager()
    demands = ones(Int64, n)
    capacity = n

    function separateCapacityCuts(cb_data)
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

        SS, rhs = rounded_capacity_inequalities!(cut_manager, 
                                                demands, 
                                                capacity, 
                                                edge_tail, 
                                                edge_head, 
                                                edge_x,
                                                integrality_tolerance = 1e-4, 
                                                max_n_cuts = 1000)

        for subtour in SS
            lhs = 0.0
            for i in subtour, j in subtour
                lhs += mat[i, j] 
            end
            subtour_size = length(subtour)
            if lhs > subtour_size - 1 + 1e-5
                con = @build_constraint(sum(x[i,j] for i in subtour, j in subtour if i != j) <= subtour_size - 1)
                MOI.submit(m, MOI.LazyConstraint(cb_data), con)
            end
        end

        end
    
    function subtour_callback(cb_data)
        status = callback_node_status(cb_data, m)
        if status == MOI.CALLBACK_NODE_STATUS_INTEGER
            mat = callback_value.(cb_data, x)
            subtours = find_subtours(mat)
            if length(subtours) != 1
                for subtour in subtours
                    subtour_size = length(subtour)
                    # if subtour_size <= n/2
                    con = @build_constraint(sum(x[i,j] for i in subtour, j in subtour if i != j) <= subtour_size - 1)
                    MOI.submit(m, MOI.LazyConstraint(cb_data), con)
                    # end
                end
            end
        end
    end

    set_attribute(m, MOI.LazyConstraintCallback(), separateCapacityCuts)
    # set_attribute(m, MOI.LazyConstraintCallback(), subtour_callback)
    
    optimize!(m)

    if termination_status(m) != MathOptInterface.OPTIMAL
        println("problema ao otimizar")
        return
    end
    return value.(x), objective_value(m)
end

function read_tour(adj_mat::Matrix{Float64})
    n = size(adj_mat)[1]
    tour = [" " for i in 1:n]
    start = 1
    for i in 1:n
        j = findfirst(adj_mat[start, :] .== 1)
        if j != 0
            tour[i] = "$start -> $j"
        end
        start = j
    end
    return tour
end

results = Dict{Tuple{Symbol, String}, Float64}()

for token in tsp_tokens
    tsp_object = readTSPLIB(token)
    @show token
    @show tsp_object.optimal
    @time x, obj = MTZ_TSP(tsp_object)
    results[(token, "MTZ")] = obj
    @time x, obj = lazy_constraint_callback_TSP(tsp_object)
    results[(token, "callback_lazy_constraint")] = obj
end

results

instance = readTSPLIB(:ulysses16)

instance = readTSPLIB(:ch150)

@time x, obj = lazy_constraint_callback_TSP(instance)

obj - instance.optimal

