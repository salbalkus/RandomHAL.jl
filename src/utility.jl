# Expit function
expit(x) = 1 / (1 + exp(-x))

# Function for creating cross-validation folds
split_folds(v, n, K) = [v[collect(i:K:n)] for i in 1:K]

