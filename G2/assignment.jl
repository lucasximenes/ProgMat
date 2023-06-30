using JuMP, CPLEX, AssignmentProblems, Coluna, BlockDecomposition, Knapsacks, MathOptInterface

const BD = BlockDecomposition;
const MOI = MathOptInterface;

data = loadAssignmentProblem(:a05100)

function unpack_data(instance_data)
    weights = instance_data.consumptions
    costs = instance_data.costs
    capacities = instance_data.capacities
    
    a, j = size(weights)
    
    agents_range = 1:a
    jobs_range = 1:j
    return weights, costs, capacities, agents_range, jobs_range
end


function solve_problem(W, C, cp, A, J)

    coluna = optimizer_with_attributes(
        Coluna.Optimizer,
        "params" => Coluna.Params(
            solver = Coluna.Algorithm.TreeSearchAlgorithm(
                timelimit = 300
            )
        ),
        "default_optimizer" => CPLEX.Optimizer
    )

    @axis(M_axis, A)

    model = BlockModel(coluna)
    @variable(model, x[m in M_axis, j in J], Bin)
    @constraint(model, cov[j in J], sum(x[m, j] for m in M_axis) >= 1)
    @constraint(model, knp[m in M_axis], sum(W[m, j] * x[m, j] for j in J) <= cp[m])
    @objective(model, Min, sum(C[m, j] * x[m, j] for m in M_axis, j in J))
    @dantzig_wolfe_decomposition(model, decomposition, M_axis)

    master = getmaster(decomposition)
    subproblems = getsubproblems(decomposition)

    optimize!(model)

    return value.(x), objective_value(model)
end

function solve_problem_specify(W, C, cp, A, J)

    coluna = optimizer_with_attributes(
        Coluna.Optimizer,
        "params" => Coluna.Params(
            solver = Coluna.Algorithm.TreeSearchAlgorithm(
                timelimit = 300
            )
        ),
        "default_optimizer" => CPLEX.Optimizer
    )

    @axis(M_axis, A)

    model = BlockModel(coluna)
    @variable(model, x[m in M_axis, j in J], Bin)
    @constraint(model, cov[j in J], sum(x[m, j] for m in M_axis) >= 1)
    @constraint(model, knp[m in M_axis], sum(W[m, j] * x[m, j] for j in J) <= cp[m])
    @objective(model, Min, sum(C[m, j] * x[m, j] for m in M_axis, j in J))
    @dantzig_wolfe_decomposition(model, decomposition, M_axis)

    subproblems = getsubproblems(decomposition)

    specify!.(subproblems, lower_multiplicity = 0, upper_multiplicity = 1)

    optimize!(model)

    return value.(x), objective_value(model)
end


function solve_problem_pricing_callback_mip(W, C, cp, A, J)
    coluna = optimizer_with_attributes(
        Coluna.Optimizer,
        "params" => Coluna.Params(
            solver = Coluna.Algorithm.TreeSearchAlgorithm(
                timelimit = 300
            )
        ),
        "default_optimizer" => CPLEX.Optimizer
    )

    @axis(M_axis, A)

    model = BlockModel(coluna)
    @variable(model, x[m in M_axis, j in J], Bin)
    @constraint(model, cov[j in J], sum(x[m, j] for m in M_axis) >= 1)
    @constraint(model, knp[m in M_axis], sum(W[m, j] * x[m, j] for j in J) <= cp[m])
    @objective(model, Min, sum(C[m, j] * x[m, j] for m in M_axis, j in J))
    @dantzig_wolfe_decomposition(model, decomposition, M_axis)


    function solve_knapsack(cost, weight, capacity)
        sp_model = Model(CPLEX.Optimizer)
        items = 1:length(weight)
        @variable(sp_model, x[i in items], Bin)
        @constraint(sp_model, weight' * x <= capacity)
        @objective(sp_model, Min, cost' * x)
        optimize!(sp_model)
        x_val = value.(x)
        return filter(i -> x_val[i] ≈ 1, collect(items))
    end

    function my_pricing_callback(cbdata)
        ## Retrieve the index of the subproblem (it will be one of the values in M_axis)
        cur_machine = BD.callback_spid(cbdata, model)
        
        ## Uncomment to see that the pricing callback is called.
        ## println("Pricing callback for machine $(cur_machine).")
    
        ## Retrieve reduced costs of subproblem variables
        red_costs = [BD.callback_reduced_cost(cbdata, x[cur_machine, j]) for j in J]
    
        ## Run the knapsack algorithm
        jobs_assigned_to_cur_machine = solve_knapsack(red_costs, w[cur_machine, :], Q[cur_machine])

        ## Create the solution (send only variables with non-zero values)
        sol_vars = [x[cur_machine, j] for j in jobs_assigned_to_cur_machine]
        sol_vals = [1.0 for _ in jobs_assigned_to_cur_machine]
        sol_cost = sum(red_costs[j] for j in jobs_assigned_to_cur_machine)

        if red_costs .> 0
            println("All reduced costs are positive")
            println("sol_cost = ", sol_cost)
            println("sol_vars = ", sol_vars)
            println("sol_vals = ", sol_vals)
        end

        ## Submit the solution to the subproblem to Coluna
        MOI.submit(model, BD.PricingSolution(cbdata), sol_cost, sol_vars, sol_vals)
        
        ## Submit the dual bound to the solution of the subproblem
        ## This bound is used to compute the contribution of the subproblem to the lagrangian
        ## bound in column generation.
        MOI.submit(model, BD.PricingDualBound(cbdata), sol_cost) # optimal solution
        return
    end
    


    subproblems = getsubproblems(decomposition)

    specify!.(subproblems, lower_multiplicity = 0, upper_multiplicity = 1)

    optimize!(model)

    return value.(x), objective_value(model)
end


function solve_problem_pricing_callback_knapsacks(W, C, cp, A, J)

    coluna = optimizer_with_attributes(
        Coluna.Optimizer,
        "params" => Coluna.Params(
            solver = Coluna.Algorithm.TreeSearchAlgorithm(
                timelimit = 300
            )
        ),
        "default_optimizer" => CPLEX.Optimizer
    )

    @axis(M_axis, A)

    model = BlockModel(coluna)
    @variable(model, x[m in M_axis, j in J], Bin)
    @constraint(model, cov[j in J], sum(x[m, j] for m in M_axis) >= 1)
    @constraint(model, knp[m in M_axis], sum(W[m, j] * x[m, j] for j in J) <= cp[m])
    @objective(model, Min, sum(C[m, j] * x[m, j] for m in M_axis, j in J))
    @dantzig_wolfe_decomposition(model, decomposition, M_axis)

    function pricing_callback(cbdata)

        cur_machine = BD.callback_spid(cbdata, model)
        red_costs = [BD.callback_reduced_cost(cbdata, x[cur_machine, j]) for j in J]

        negatives = findall(x -> x < 0, red_costs)
        
        negative_red_costs = red_costs[negatives]

        negative_red_costs = -1 .* negative_red_costs

        negative_red_costs = convert(Array{Int64}, negative_red_costs .* 1e3)

        backpack_data = Knapsack(cp[cur_machine], W[cur_machine, negatives], negative_red_costs)

        println("Objeto mochila: ", backpack_data)
        println("capacidade = ", backpack_data.capacity)
        println("pesos = ", backpack_data.weights)
        println("valores = ", backpack_data.profits)

        println(solveKnapsack(backpack_data, :DynammicProgramming))
    
        cost, jobs_assigned_to_cur_machine = solveKnapsack(backpack_data)

        println(jobs_assigned_to_cur_machine)
    
        sol_vars = [x[cur_machine, j] for j in jobs_assigned_to_cur_machine]
        sol_vals = [1.0 for _ in jobs_assigned_to_cur_machine]
        sol_cost = sum(negative_red_costs[j] for j in jobs_assigned_to_cur_machine) / 10e5

        # Submit the solution to the subproblem to Coluna
        MOI.submit(model, BD.PricingSolution(cbdata), sol_cost, sol_vars, sol_vals)
    
        # Submit the dual bound to the solution of the subproblem
        # This bound is used to compute the contribution of the subproblem to the lagrangian
        # bound in column generation.

        MOI.submit(model, BD.PricingDualBound(cbdata), sol_cost) # optimal solution
        return
    
    end

    subproblems = BD.getsubproblems(decomposition);
    BD.specify!.(subproblems, lower_multiplicity = 0, solver = pricing_callback);

    optimize!(model)

    return value.(x), objective_value(model)
end



params = unpack_data(data)


@time solve_problem(params...)
@time solve_problem_specify(params...)
@time solve_problem_pricing_callback_mip(params...)
@time solve_problem_pricing_callback_knapsacks(params...)

