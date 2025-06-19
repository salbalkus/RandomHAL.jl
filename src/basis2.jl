basis_function(Xcol::AbstractVector{<:Real}) = Xcol .>= transpose(Xcol)
basis_function(Xcol::AbstractVector{<:Real}, smoothness::Int) = basis_function(Xcol) .* (Xcol .- transpose(Xcol)).^smoothness ./ factorial(smoothness)

function ha_basis_matrix(X::Tables.Columns, smoothness::Int; interaction_limit::Int = nothing)
    if isnothing(interaction_limit)
        interaction_limit = length(X)
    end
    
    coltypes = Tables.schema(X).types
    main_terms = (smoothness == 0) ? 
        [coltypes[i] == Bool ? reshape(X[i], :, 1) : basis_function(X[i]) for i in 1:length(X)] :
        [coltypes[i] == Bool ? reshape(X[i], :, 1) : basis_function(X[i], smoothness) for i in 1:length(X)]
    
    main_terms_and_interactions, _ = all_interactions(main_terms, interaction_limit)

    return main_terms_and_interactions
end


function interact(x_next::T, x::Vector{T}, next_order::Int, orders::Vector{Int}, limit::Int) where T <: Int
    interactions = Vector{T}()
    interaction_orders = Vector{Int}()
    for i in 1:length(x)
        if orders[i] < limit
            x_next_i = x_next .* x[i]
            interaction_orders_i = orders[i] + 1
            push!(interactions, x_next_i)
            push!(interaction_orders, interaction_orders_i)
        end
    end
    x_output = vcat([x_next], x, interactions)
    orders_output = vcat([next_order], orders, interaction_orders)
    return x_output, orders_output
end

function all_interactions(main_terms::Vector{T}, limit::Int) where T <: Int
    combined_interactions_and_main = [main_terms[1]]
    orders = [1]
    for i in 2:length(main_terms)
        combined_interactions_and_main, orders = interact(main_terms[i], combined_interactions_and_main, 1, orders, limit)
    end
    return combined_interactions_and_main, orders
end





