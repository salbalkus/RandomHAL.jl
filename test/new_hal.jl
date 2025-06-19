
using RandomHAL
using Pkg
Pkg.activate("test")
using Random
using Test
using CausalTables
using Distributions
using MLJ
import LogExpFunctions: logistic
using Tables
using Plots
using Optim
using Combinatorics

Random.seed!(1234)
dgp = @dgp(
        X2 ~ Beta(2, 3),
        X3 ~ Beta(3, 2),
        X4 ~ Beta(3, 3),
        A ~ (@. Bernoulli(logistic((X2 + X2^2 + X3 + X3^2 + X4 + X4^2 + X2 * X3) - 2.5))),
        Y ~ (@. Normal(A + X2 * X3 + A * X2 + A * X4 + 0.2 * (sqrt(10*X3*X4) + sqrt(10 * X2) + sqrt(10 * X3) + sqrt(10*X4)), 0.1))
    )
scm = StructuralCausalModel(dgp, :A, :Y)

n = 100
d = 4
ct = rand(scm, n)
X = Tables.Columns(responseparents(ct))
y = vec(responsematrix(ct))
y = y

cttest = rand(scm, n)
Xtest = responseparents(cttest)
ytest = vec(responsematrix(cttest))

cttestA = intervene(cttest, treat_all)
XAtest = responseparents(cttestA)
yAtest = vec(responsematrix(cttestA))
true_mean = conmean(scm, cttest, :Y)

# HAL
model = HALRegressor(0, "standard")
@time hal = machine(model, X, y) |> fit!

halpreds = MLJ.predict(hal, Xtest)
halmse = mean((halpreds .- true_mean).^2)
scatter(true_mean, [halpreds], labels = ["Mean HAL"])


## New HAL
B = map(x -> RandomHAL.basis_function(x), X)
β = zeros(n * d + 1)
λ = 0
main_terms = [(B[i] * β[((i - 1)*n + 2 ): ((i*n) + 1)]) for i in 1:d]
preds = β[1] .+ sum([reduce(.*, main_term) for main_term in powerset(main_terms,1)])
mean((preds - y).^2)

function loss(β)
    main_terms = [(B[i] * β[((i - 1)*n + 2 ): ((i*n) + 1)]) for i in 1:d]
    preds = β[1] .+ sum([reduce(.*, main_term) for main_term in powerset(main_terms,1)])
    return mean((preds - y).^2) + λ .* sum(abs.(β))
end

# Initial
#sol = optimize(loss, β, ConjugateGradient(), Optim.Options(iterations = 100))
#β_best = Optim.minimizer(sol)

# After Initial
#main_terms = [(B[i] * β_best[((i - 1)*n + 2 ): ((i*n) + 1)]) for i in 1:d]
#preds = β_best[1] .+ sum([reduce(.*, main_term) for main_term in powerset(main_terms,1)])
#mean((preds .- y).^2)

# Nelder-Mead
sol2 = optimize(loss, β, NelderMead(), Optim.Options(iterations = 20000))
β_best = Optim.minimizer(sol2)

# After Nelder-Mead
main_terms = [(B[i] * β_best[((i - 1)*n + 2 ): ((i*n) + 1)]) for i in 1:d]
preds = β_best[1] .+ sum([reduce(.*, main_term) for main_term in powerset(main_terms,1)])
mean((preds .- y).^2)

mean((preds .- true_mean).^2)

halmse = mean((halpreds .- true_mean).^2)
scatter(true_mean, [preds, halpreds], labels = ["New HAL" "Old HAL"])

