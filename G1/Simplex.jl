using LinearAlgebra, Printf, Combinatorics

mutable struct SimplexTableau
    z_c     ::Array{Rational{Int64}} 
    Y       ::Array{Rational{Int64}}
    x_B     ::Array{Rational{Int64}}
    obj     ::Rational{Int64}
    b_idx   ::Array{Int64}
end


function initial_BFS(A, b)
    m, n = size(A)

    b_idx = zeros(Int64, m)
    for i in 1:n
        if sum(A[:,i]) == 1 && all(x -> x == 0 || x == 1, A[:,i])
            b_idx[ findfirst(A[:,i] .== 1) ] = i
        end
    end

    if is_nonnegative(b)
        return b_idx, b, A[:, b_idx]
    else
        error("Infeasible")
    end

end

function pivoting!(t::SimplexTableau, minimizing::Bool=false)
    m, n = size(t.Y)

    entering, exiting = pivot_point(t, minimizing)
    println("Pivoting: entering = x_$entering, exiting = x_$(t.b_idx[exiting])")

    # Pivoting: exiting-row, entering-column
    # updating exiting-row
    coef = t.Y[exiting, entering]
    t.Y[exiting, :] /= coef
    t.x_B[exiting] /= coef

    # updating other rows of Y
    for i in setdiff(1:m, exiting)
        coef = t.Y[i, entering]
        t.Y[i, :] -= coef * t.Y[exiting, :]
        t.x_B[i] -= coef * t.x_B[exiting]
    end

    # updating the row for the reduced costs
    coef = t.z_c[entering]
    t.z_c -= coef * t.Y[exiting, :]'
    t.obj -= coef * t.x_B[exiting]

    # Updating b_idx
    t.b_idx[ findfirst(t.b_idx .== t.b_idx[exiting]) ] = entering
end

function pivot_point(t::SimplexTableau, minimizing::Bool=false)
    # Finding the entering variable index
    entering = 0
    if minimizing
        entering = findfirst( t.z_c .> 0)[2]
    else
        entering = findfirst( t.z_c .< 0)[2]
    end
    
    if entering == 0
        error("Optimal")
    end

    # min ratio test / finding the exiting variable index
    pos_idx = findall( t.Y[:, entering] .> 0 )
    if length(pos_idx) == 0
        error("Unbounded")
    end
    exiting = pos_idx[ argmin( t.x_B[pos_idx] ./ t.Y[pos_idx, entering] ) ]

    return entering, exiting
end


function is_nonnegative(x::Union{Vector{Rational{Int64}}, Array{Rational{Int64}}})
    return all(i -> i >= 0, x)
end

function is_nonpositive(x::Union{Vector{Rational{Int64}}, Array{Rational{Int64}}})
    return all(i -> i <= 0, x)
end

function first_stage_initialize(A, b, c, s)
    m, n = size(A)
    slices = []
    for signal in s
        if signal == "≤"
            push!(slices, 1)    
        elseif signal == "≥"
            push!(slices, 2)
        else
            push!(slices, 1)
        end
    end
    new_vars_amnt = sum(slices)

    additional_matrix = zeros(Rational{Int64}, m, new_vars_amnt)
    index = 1
    artificials = []
    for (i, type) in enumerate(s)
        if type == "≥"
            additional_matrix[i, [index, index+1]] = [-1, 1]
            push!(artificials, [i, index+1])
        elseif type == "≤"
            additional_matrix[i, index] = 1
        else
            additional_matrix[i, index] = 1
            push!(artificials, [i, index])
        end
        index += slices[i]
    end

    n_total = n + new_vars_amnt

    A_total = [A additional_matrix]
    c_total = zeros(Rational{Int64}, n_total)
    for artificial in artificials
        c_total[n + artificial[2]] = 1
    end

    b_idx, x_B, B = initial_BFS(A_total, b)
    Y = inv(B) * A_total
    c_B = c_total[b_idx]
    obj = dot(c_B, x_B)

    # z_c is a row vector
    z_c = zeros(1,n_total)
    n_idx = setdiff(1:n_total, b_idx)
    z_c[n_idx] = c_B' * inv(B) * A_total[:,n_idx] - c_total[n_idx]'

    return SimplexTableau(z_c, Y, x_B, obj, b_idx), artificials
end

function initialize(A, b, c) 
    m, n = size(A)

    n_slack = n + m

    A_slack = [A Matrix{Rational{Int64}}(I, m, m)]
    c_slack = [c; zeros(Rational{Int64}, m)]

    # Finding an initial BFS
    b_idx, x_B, B = initial_BFS(A_slack,b)

    Y = inv(B) * A_slack
    c_B = c_slack[b_idx]
    obj = dot(c_B, x_B)

    # z_c is a row vector
    z_c = zeros(1,n_slack)
    n_idx = setdiff(1:n_slack, b_idx)
    z_c[n_idx] = c_B' * inv(B) * A_slack[:,n_idx] - c_slack[n_idx]'

    return SimplexTableau(z_c, Y, x_B, obj, b_idx)
end

function print_tableau(t::SimplexTableau, message::String)
    println(message)

    m, n = size(t.Y)

    hline0 = repeat("-", 6)
    hline1 = repeat("-", 7*n)
    hline2 = repeat("-", 7)
    hline = join([hline0, "+", hline1, "+", hline2])

    println(hline)

    @printf("%6s|", "")
    for j in 1:length(t.z_c)
    @printf("%6.2f ", t.z_c[j])
    end
    @printf("| %6.2f\n", t.obj)

    println(hline)

    for i in 1:m
    @printf("x[%2d] |", t.b_idx[i])
    for j in 1:n
        @printf("%6.2f ", t.Y[i,j])
    end
    @printf("| %6.2f\n", t.x_B[i])
    end

    println(hline)
end


function simplex(A::Matrix{Rational{Int64}}, b::Vector{Rational{Int64}}, c::Vector{Rational{Int64}}, s::Vector{String})
    
    tableau = nothing

    ################################
    ## First stage of the simplex ##
    ################################
    
    if issubset(["≥"], s) || issubset(["="], s)
        artificials = nothing
        try
            tableau, artificials = first_stage_initialize(A, b, c, s)
        catch err
            if err.msg == "inviável"
                println("Infeasible problem erro")
                return nothing, nothing
            end
        end
        iter = 1
        while !is_nonpositive(tableau.z_c)
            try
                pivoting!(tableau, true)
            catch err
                if err.msg == "Optimal"
                    if tableau.obj != 0
                        println("what is a man")
                        return nothing, nothing
                    else
                        println("Optimal first stage")
                    end
                    break
                elseif err.msg == "Unbounded"
                    println("Unbounded solution")
                    return 1//0, nothing
                end
            end
            print_tableau(tableau, "Iteration number $iter")
            iter += 1
        end


        println("Fim pivoteamento")

        if tableau.obj == 0
            print_tableau(tableau, "Optimal solution found in first stage")
        else
            println("Infeasible problem")
            return nothing, nothing
        end

        # get vector of all second values of artificials
        tableau.Y = tableau.Y[:, setdiff(1:length(tableau.z_c), length(c) .+ [artificials[i][2] for i in eachindex(artificials)])]

        new_A = tableau.Y
        new_B = tableau.x_B

        amount_slack = length(findall(x -> x != "=", s))
        n_vars = length(c) + amount_slack

        new_C = [c ; zeros(Rational{Int64}, amount_slack)]
        aux_C = deepcopy(new_C)

        obj = 0
        for i in 1:length(tableau.b_idx)
            aux_C += -new_C[tableau.b_idx[i]] * new_A[i, :]
            obj += new_C[tableau.b_idx[i]] * new_B[i]
        end

        new_C = aux_C
        b_idx, x_B, B = initial_BFS(new_A, new_B)

        Y = inv(B) * new_A
        c_B = new_C[b_idx]
        # obj = dot(c_B, x_B)

        # z_c is a row vector
        z_c = zeros(1,n_vars)
        n_idx = setdiff(1:n_vars, b_idx)
        z_c[n_idx] = c_B' * inv(B) * new_A[:,n_idx] - new_C[n_idx]'

        tableau = SimplexTableau(z_c, Y, x_B, obj, b_idx)

        print_tableau(tableau, "Initial second stage tableau")
    
    else
        try
            tableau = initialize(A, b, c)
            
        catch err
            if err.msg == "inviável"
                println("Infeasible problem")
                return nothing, nothing
            end
        end
        print_tableau(tableau, "Initial tableau")
    end
    
    iter = 1
    while !is_nonnegative(tableau.z_c)
        try
            pivoting!(tableau)
        catch err
            if err.msg == "Optimal"
                println("Optimal solution in the last iteration")
                break
            elseif err.msg == "Unbounded"
                println("Unbounded solution")
                return 1//0, nothing
            end
        end
        print_tableau(tableau, "Iteration number $iter")
        iter += 1
        if iter > 5
            break
        end
    end 

    println("end of loop")
    
    x = zeros(Rational{Int64}, length(tableau.z_c))
    x[tableau.b_idx] = tableau.x_B
    
    println("Optimal solution found: x = $x, obj = $(tableau.obj)")

    return tableau.obj, x
end

A = [
    2//1 1//1
    1//1 2//1
]
b = [ 4//1, 4//1 ]
c = [ 4//1, 3//1 ]
s = [ "≤", "≤" ]

#usei deepcopy porque os valores estavam sendo alterados (por algum motivo que não identificado) e isso impedia
# que o simplex fosse executado mais de uma vez no mesmo problema

simplex(deepcopy(A), deepcopy(b), deepcopy(c), s)

A = [
    1//1  1//1 1//1
    1//1 -1//1 0//1
    2//1  3//1 1//1
]
b = [ 10//1, 1//1, 20//1 ]
c = [ 4//1, 5//1, -3//1 ]
s = [ "≤", "≥", "=" ]

obj, x = simplex(deepcopy(A), deepcopy(b), deepcopy(c), s)


# transform obj into float
obj_float = convert(Float64, obj)

A = [
    2//1  5//1 
    1//1 4//1 
]
b = [ 5//1, 3//1 ]
c = [ 5//1, 7//1 ]
s = [ "≤", "=" ]

obj, x = simplex(deepcopy(A), deepcopy(b), deepcopy(c), s)


# transform obj into float
obj_float = convert(Float64, obj)