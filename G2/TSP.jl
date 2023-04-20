using TSPLIB, JuMP, HiGHS, MathOptInterface

tsp_tokens = [:burma14, :ulysses16, :gr17, :gr21]

function MTZ_TSP(instance::TSP)
    n = instance.dimension
    
    m = Model(HiGHS.Optimizer)
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

function find_subtour(mat::Matrix{Float64})
    n = size(mat)[1]
    visited = zeros(n)
    
    start = 1
    visited[1] = 1
    visited_amnt = 1
    for i in 1:14
        next = findfirst(mat[start, :] .> 0.5)
        if visited[next] == 1
            break 
        else
            visited[next] = 1
            visited_amnt += 1
        end
        start = next
    end
    if visited_amnt < n
        return findall(visited .== 1)
    else
        return false
    end
end

function lazy_constraint_TSP(instance::TSP)
    n = instance.dimension
    
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x[1:n, 1:n], Bin)
    @constraint(m, [i in 1:n], sum(x[i, j] for j in 1:n if j != i) == 1)
    @constraint(m, [j in 1:n], sum(x[i, j] for i in 1:n if i != j) == 1)
    @objective(m, Min, sum(instance.weights[i,j]*x[i,j] for i in 1:n, j in 1:n if i != j))
    optimize!(m)
    
    if termination_status(m) != MathOptInterface.OPTIMAL
        println("problema ao otimizar")
        return
    end
    
    subtour = find_subtour(value.(x))
    while subtour != false
        subtour_size = length(subtour)
        @constraint(m, sum(x[i,j] for i in subtour, j in subtour if i != j) <= subtour_size - 1)
        optimize!(m)
        if termination_status(m) != MathOptInterface.OPTIMAL
            println("problema ao otimizar")
            return
        end
        subtour = find_subtour(value.(x))
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
    @time x, obj = MTZ_TSP(tsp_object)
    results[(token, "MTZ")] = (x, obj)
    @time x, obj = lazy_constraint_TSP(tsp_object)
    results[(token, "lazy_constraint")] = (x, obj)
end

results