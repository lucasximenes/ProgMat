using JuMP, CPLEX, AssignmentProblems, Coluna, BlockDecomposition, Knapsacks, MathOptInterface

const BD = BlockDecomposition;
const MOI = MathOptInterface;

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

        negative_red_costs = convert(Array{Int64}, floor.(negative_red_costs .* 1e3))

        sol_vars = nothing
        sol_vals = nothing
        sol_cost = nothing

        if sum(W[cur_machine, negatives]) <= cp[cur_machine]

            sol_vars = [x[cur_machine, j] for j in negatives]
            sol_vals = [1.0 for _ in negatives]
            sol_cost = sum(red_costs[j] for j in negatives)
        
        else

            backpack_data = Knapsack(cp[cur_machine], W[cur_machine, negatives], negative_red_costs)
        
            cost, jobs_assigned_to_cur_machine = solveKnapsack(backpack_data)
        
            sol_vars = [x[cur_machine, j] for j in negatives[jobs_assigned_to_cur_machine]]
            sol_vals = [1.0 for _ in jobs_assigned_to_cur_machine]
            sol_cost = sum(red_costs[j] for j in negatives[jobs_assigned_to_cur_machine])

        end


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


function execute_tests()
    instances = [:a05100,
    :a05200,
    :a10100,
    :a10200,
    :a20100,
    :a20200,
    :b05100,
    :b05200,
    :b10100,
    :b10200,
    :b20100,
    :b20200,
    :c05100,
    :c0515_1,
    :c0515_2,
    :c0515_3,
    :c0515_4,
    :c0515_5,
    :c05200,
    :c0520_1,
    :c0520_2,
    :c0520_3,
    :c0520_4,
    :c0520_5,
    :c0525_1,
    :c0525_2,
    :c0525_3,
    :c0525_4,
    :c0525_5,
    :c0530_1,
    :c0530_2,
    :c0530_3,
    :c0530_4,
    :c0530_5,
    :c0824_1,
    :c0824_2,
    :c0824_3,
    :c0824_4,
    :c0824_5,
    :c0832_1,
    :c0832_2,
    :c0832_3,
    :c0832_4,
    :c0832_5,
    :c0840_1,
    :c0840_2,
    :c0840_3,
    :c0840_4,
    :c0840_5,
    :c0848_1,
    :c0848_2,
    :c0848_3,
    :c0848_4,
    :c0848_5,
    :c10100,
    :c10200,
    :c1030_1,
    :c1030_2,
    :c1030_3,
    :c1030_4,
    :c1030_5,
    :c10400,
    :c1040_1,
    :c1040_2,
    :c1040_3,
    :c1040_4,
    :c1040_5,
    :c1050_1,
    :c1050_2,
    :c1050_3,
    :c1050_4,
    :c1050_5,
    :c1060_1,
    :c1060_2,
    :c1060_3,
    :c1060_4,
    :c1060_5,
    :c15900,
    :c20100,
    :c201600,
    :c20200,
    :c20400,
    :c30900,
    :c401600,
    :c40400,
    :c60900,
    :c801600,
    :d05100,
    :d05200,
    :d10100,
    :d10200,
    :d10400,
    :d15900,
    :d20100,
    :d201600,
    :d20200,
    :d20400,
    :d30900,
    :d401600,
    :d40400,
    :d60900,
    :d801600,
    :e05100,
    :e05200,
    :e10100,
    :e10200,
    :e10400,
    :e15900,
    :e20100,
    :e201600,
    :e20200,
    :e20400,
    :e30900,
    :e401600,
    :e40400,
    :e60900,
    :e801600]

    for instance in instances
        println("Solving instance $instance")
        data = read_data(instance)
        params = unpack_data(data)
        time = @elapsed solve_problem(params...)
    end
end



data = loadAssignmentProblem(:a05200)
params = unpack_data(data)

@time solve_problem(params...)
@time solve_problem_specify(params...)
@time solve_problem_pricing_callback_knapsacks(params...)

