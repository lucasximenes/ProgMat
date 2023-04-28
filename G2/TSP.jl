using TSPLIB, JuMP, HiGHS, MathOptInterface, Graphs, CPLEX

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
    @constraint(m, u[1] == 1) ## parece não precisar
    
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
    
    function subtour_callback(cb_data)
        mat = callback_value.(cb_data, x)
        subtours = find_subtours(mat)
        if length(subtours) != 1
            for subtour in subtours
                subtour_size = length(subtour)
                con = @build_constraint(sum(x[i,j] for i in subtour, j in subtour if i != j) <= subtour_size - 1)
                MOI.submit(m, MOI.LazyConstraint(cb_data), con)
            end
        end
    end

    set_attribute(m, MOI.LazyConstraintCallback(), subtour_callback)
    
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


results = Dict{Tuple{Symbol, String}, Tuple{Matrix{Float64}, Float64}}()

for token in tsp_tokens
    tsp_object = readTSPLIB(token)
    @show token
    @time x, obj = MTZ_TSP(tsp_object)
    results[(token, "MTZ")] = (x, obj)
    @time x, obj = lazy_constraint_callback_TSP(tsp_object)
    results[(token, "callback_lazy_constraint")] = (x, obj)
end

results

instance = readTSPLIB(:si175)

@time x, obj = lazy_constraint_graph_callback_TSP(instance)

obj == instance.optimal