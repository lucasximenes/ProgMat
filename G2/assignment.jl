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

function solve_problem_pricing_callback(W, C, cp, A, J)

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
    
        positives = findall(x -> x > 1e-5, red_costs)

        if length(positives) == 0
        end

        positive_red_costs = red_costs[positives]

        positive_red_costs = convert(Array{Int64}, red_costs .* 10e5)

        backpack_data = Knapsack(cp[cur_machine], W[cur_machine, positives], positive_red_costs)
    
        cost, jobs_assigned_to_cur_machine = solveKnapsack(backpack_data)
    
        sol_vars = [x[cur_machine, j] for j in jobs_assigned_to_cur_machine]
        sol_vals = [1.0 for _ in jobs_assigned_to_cur_machine]
        sol_cost = sum(positive_red_costs[j] for j in jobs_assigned_to_cur_machine) / 10e5

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
@time solve_problem_pricing_callback(params...)