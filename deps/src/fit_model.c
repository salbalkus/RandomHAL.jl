#include "fit_model.h"
#include "coordinate_descent.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * Configuration and model creation
 * ============================================================================ */

basis_config_t* basis_config_create(idx_t num_sections) {
    basis_config_t *cfg = (basis_config_t *)malloc(sizeof(basis_config_t));
    if (!cfg) return NULL;
    
    cfg->num_sections = num_sections;
    cfg->section_sizes = (idx_t *)calloc((size_t)num_sections, sizeof(idx_t));
    cfg->sections = (idx_t **)malloc((size_t)num_sections * sizeof(idx_t *));
    cfg->smoothness = 0;
    cfg->max_basis_size = 500;
    cfg->min_support = 1.0 / sqrt(-1.0);  /* sqrt(n) will be computed at fit time */
    
    if (!cfg->section_sizes || !cfg->sections) {
        if (cfg->section_sizes) free(cfg->section_sizes);
        if (cfg->sections) free(cfg->sections);
        free(cfg);
        return NULL;
    }
    
    for (idx_t i = 0; i < num_sections; i++) {
        cfg->sections[i] = NULL;
    }
    
    return cfg;
}

void basis_config_destroy(basis_config_t *config) {
    if (!config) return;
    if (config->sections) {
        for (idx_t i = 0; i < config->num_sections; i++) {
            if (config->sections[i]) free(config->sections[i]);
        }
        free(config->sections);
    }
    if (config->section_sizes) free(config->section_sizes);
    free(config);
}

void basis_config_add_section(basis_config_t *config, idx_t s_idx,
                               const idx_t *columns, idx_t num_columns) {
    if (!config || s_idx >= config->num_sections) return;
    
    if (config->sections[s_idx]) free(config->sections[s_idx]);
    
    config->sections[s_idx] = (idx_t *)malloc((size_t)num_columns * sizeof(idx_t));
    if (config->sections[s_idx]) {
        memcpy(config->sections[s_idx], columns, (size_t)num_columns * sizeof(idx_t));
        config->section_sizes[s_idx] = num_columns;
    }
}

fit_config_t* fit_config_create(void) {
    fit_config_t *cfg = (fit_config_t *)malloc(sizeof(fit_config_t));
    if (!cfg) return NULL;
    
    cfg->K = 5;
    cfg->max_iters = 1000;
    cfg->tolerance = 1e-7;
    cfg->alpha = 1.0;  /* LASSO */
    cfg->n_lambda = 100;
    cfg->lambda_min_ratio = 1e-3;
    cfg->auto_lambda = true;
    
    return cfg;
}

void fit_config_destroy(fit_config_t *config) {
    if (config) free(config);
}

model_t* model_create(idx_t d, idx_t n_lambda) {
    model_t *m = (model_t *)malloc(sizeof(model_t));
    if (!m) return NULL;
    
    m->β = (real_t *)calloc((size_t)d, sizeof(real_t));
    m->β_path = (real_t *)calloc((size_t)(d * n_lambda), sizeof(real_t));
    m->β0_path = (real_t *)calloc((size_t)n_lambda, sizeof(real_t));
    m->λ = (real_t *)calloc((size_t)n_lambda, sizeof(real_t));
    
    if (!m->β || !m->β_path || !m->β0_path || !m->λ) {
        if (m->β) free(m->β);
        if (m->β_path) free(m->β_path);
        if (m->β0_path) free(m->β0_path);
        if (m->λ) free(m->λ);
        free(m);
        return NULL;
    }
    
    m->β0 = 0.0;
    m->n_lambda = n_lambda;
    m->best_lambda_idx = 0;
    m->best_lambda = 0.0;
    m->best_cv_error = REAL_MAX;
    m->family = FAMILY_GAUSSIAN;
    
    return m;
}

void model_destroy(model_t *model) {
    if (!model) return;
    if (model->β) free(model->β);
    if (model->β_path) free(model->β_path);
    if (model->β0_path) free(model->β0_path);
    if (model->λ) free(model->λ);
    free(model);
}

/* ============================================================================
 * Phase 3: Column statistics and preprocessing
 * ============================================================================ */

typedef struct {
    real_t *means;           /* Column means (p) */
    real_t *stds;            /* Column standard deviations (p) */
    bool *is_standardized;   /* Whether each column is standardized */
} column_stats_t;

static column_stats_t* column_stats_create(idx_t p) {
    column_stats_t *stats = (column_stats_t *)malloc(sizeof(column_stats_t));
    if (!stats) return NULL;
    
    stats->means = (real_t *)calloc((size_t)p, sizeof(real_t));
    stats->stds = (real_t *)calloc((size_t)p, sizeof(real_t));
    stats->is_standardized = (bool *)calloc((size_t)p, sizeof(bool));
    
    if (!stats->means || !stats->stds || !stats->is_standardized) {
        if (stats->means) free(stats->means);
        if (stats->stds) free(stats->stds);
        if (stats->is_standardized) free(stats->is_standardized);
        free(stats);
        return NULL;
    }
    
    return stats;
}

static void column_stats_destroy(column_stats_t *stats) {
    if (!stats) return;
    if (stats->means) free(stats->means);
    if (stats->stds) free(stats->stds);
    if (stats->is_standardized) free(stats->is_standardized);
    free(stats);
}

/* Compute column means and standard deviations */
static int compute_column_stats(const real_t *X, idx_t n, idx_t p,
                                 column_stats_t *stats) {
    if (!X || !stats) return FASTHAL_ERROR_INVALID_ARGS;
    
    /* Compute means */
    for (idx_t j = 0; j < p; j++) {
        real_t sum = 0.0;
        for (idx_t i = 0; i < n; i++) {
            sum += X[i + j * n];  /* Column-major */
        }
        stats->means[j] = sum / n;
    }
    
    /* Compute standard deviations */
    for (idx_t j = 0; j < p; j++) {
        real_t sum_sq_dev = 0.0;
        for (idx_t i = 0; i < n; i++) {
            real_t dev = X[i + j * n] - stats->means[j];
            sum_sq_dev += dev * dev;
        }
        stats->stds[j] = sqrt(sum_sq_dev / (n - 1));
        
        /* Avoid division by zero */
        if (stats->stds[j] < 1e-10) {
            stats->stds[j] = 1.0;
        }
        stats->is_standardized[j] = true;
    }
    
    return FASTHAL_SUCCESS;
}

/* ============================================================================
 * Phase 3: Lambda grid generation
 * ============================================================================ */

/* Generate lambda grid: n_lambda values from λ_max down to λ_min */
static int generate_lambda_grid(const real_t *X, idx_t n, idx_t p,
                                 const real_t *y,
                                 const column_stats_t *stats,
                                 idx_t n_lambda, real_t lambda_min_ratio,
                                 real_t *lambda_out) {
    if (!X || !y || !stats || !lambda_out) {
        return FASTHAL_ERROR_INVALID_ARGS;
    }
    
    /* Compute λ_max as max(|X^T * y|) / n */
    real_t *Xty = (real_t *)malloc((size_t)p * sizeof(real_t));
    if (!Xty) return FASTHAL_ERROR_ALLOCATION;
    
    /* X^T * y */
    for (idx_t j = 0; j < p; j++) {
        real_t sum = 0.0;
        for (idx_t i = 0; i < n; i++) {
            sum += X[i + j * n] * y[i];
        }
        Xty[j] = sum / n;
    }
    
    /* Find max |X^T * y| */
    real_t lambda_max = 0.0;
    for (idx_t j = 0; j < p; j++) {
        real_t abs_val = fabs(Xty[j]);
        if (abs_val > lambda_max) {
            lambda_max = abs_val;
        }
    }
    
    free(Xty);
    
    /* If no signal, use default */
    if (lambda_max < 1e-10) {
        lambda_max = 1.0;
    }
    
    /* Generate logarithmic grid from λ_max to λ_min */
    real_t lambda_min = lambda_min_ratio * lambda_max;
    
    for (idx_t i = 0; i < n_lambda; i++) {
        real_t t = (real_t)i / (n_lambda - 1);  /* 0 to 1 */
        real_t log_lambda = (1.0 - t) * log(lambda_max) + t * log(lambda_min);
        lambda_out[i] = exp(log_lambda);
    }
    
    return FASTHAL_SUCCESS;
}

/* ============================================================================
 * Phase 3: K-fold cross-validation (TODO: full implementation in Phase 3.2)
 * ============================================================================ */

/* Fold split functions - disabled for now, will be enabled with full CV */
/*
typedef struct {
    idx_t *fold_assignment;      
    idx_t *fold_start;           
    idx_t *fold_size;            
} fold_split_t;

static fold_split_t* fold_split_create(idx_t n, idx_t K) {
    fold_split_t *split = (fold_split_t *)malloc(sizeof(fold_split_t));
    if (!split) return NULL;
    
    split->fold_assignment = (idx_t *)malloc((size_t)n * sizeof(idx_t));
    split->fold_start = (idx_t *)calloc((size_t)K, sizeof(idx_t));
    split->fold_size = (idx_t *)calloc((size_t)K, sizeof(idx_t));
    
    if (!split->fold_assignment || !split->fold_start || !split->fold_size) {
        if (split->fold_assignment) free(split->fold_assignment);
        if (split->fold_start) free(split->fold_start);
        if (split->fold_size) free(split->fold_size);
        free(split);
        return NULL;
    }
    
    for (idx_t i = 0; i < n; i++) {
        split->fold_assignment[i] = i % K;
    }
    
    for (idx_t k = 0; k < K; k++) {
        split->fold_size[k] = 0;
    }
    for (idx_t i = 0; i < n; i++) {
        split->fold_size[split->fold_assignment[i]]++;
    }
    
    idx_t cumsum = 0;
    for (idx_t k = 0; k < K; k++) {
        split->fold_start[k] = cumsum;
        cumsum += split->fold_size[k];
    }
    
    return split;
}

static void fold_split_destroy(fold_split_t *split) {
    if (!split) return;
    if (split->fold_assignment) free(split->fold_assignment);
    if (split->fold_start) free(split->fold_start);
    if (split->fold_size) free(split->fold_size);
    free(split);
}

static real_t compute_cv_error_gaussian(
    const real_t *y_actual, const real_t *y_predicted, idx_t n) {
    
    real_t sum_sq_error = 0.0;
    for (idx_t i = 0; i < n; i++) {
        real_t residual = y_actual[i] - y_predicted[i];
        sum_sq_error += residual * residual;
    }
    
    return sum_sq_error / n;
}
*/

/* ============================================================================
 * Phase 3: Simplified coordinate descent wrapper
 * ============================================================================ */

/* Simple wrapper for coordinate descent on dense basis matrix */
static int coordinate_descent_simple(
    const real_t *B, idx_t n, idx_t d,
    const real_t *y,
    real_t *μ, real_t *invσ, real_t *σ2,
    const real_t *lambda_values, idx_t n_lambda,
    real_t tol, idx_t max_iters,
    real_t *beta_out, real_t *beta0_out) {
    
    /* Initialize output */
    memset(beta_out, 0, (size_t)(d * n_lambda) * sizeof(real_t));
    memset(beta0_out, 0, (size_t)n_lambda * sizeof(real_t));
    
    /* For each lambda */
    for (idx_t l = 0; l < n_lambda; l++) {
        real_t lambda = lambda_values[l];
        
        /* Get starting point: all zeros */
        real_t *beta = beta_out + l * d;
        real_t *beta0 = beta0_out + l;
        
        /* Compute residuals and intercept */
        real_t y_mean = 0.0;
        for (idx_t i = 0; i < n; i++) y_mean += y[i];
        y_mean /= n;
        *beta0 = y_mean;
        
        /* Residuals = y - y_mean */
        real_t *residuals = (real_t *)malloc((size_t)n * sizeof(real_t));
        if (!residuals) return FASTHAL_ERROR_ALLOCATION;
        
        for (idx_t i = 0; i < n; i++) {
            residuals[i] = y[i] - y_mean;
        }
        
        /* Coordinate descent iterations */
        real_t *beta_prev = (real_t *)calloc((size_t)d, sizeof(real_t));
        if (!beta_prev) {
            free(residuals);
            return FASTHAL_ERROR_ALLOCATION;
        }
        
        for (idx_t iter = 0; iter < max_iters; iter++) {
            /* Update each coefficient */
            real_t max_change = 0.0;
            
            for (idx_t j = 0; j < d; j++) {
                /* Compute gradient: X[, j]^T * residuals / n */
                real_t grad = 0.0;
                for (idx_t i = 0; i < n; i++) {
                    grad += B[i + j * n] * residuals[i];
                }
                grad /= n;
                
                /* Update: soft_threshold(β + grad / σ²) */
                real_t beta_new = beta[j] + grad / σ2[j];
                
                /* Soft thresholding */
                if (beta_new > lambda) {
                    beta_new -= lambda;
                } else if (beta_new < -lambda) {
                    beta_new += lambda;
                } else {
                    beta_new = 0.0;
                }
                
                /* Update residuals */
                real_t delta = beta_new - beta[j];
                if (fabs(delta) > 1e-12) {
                    for (idx_t i = 0; i < n; i++) {
                        residuals[i] -= B[i + j * n] * delta;
                    }
                }
                
                beta[j] = beta_new;
                real_t change = fabs(delta);
                if (change > max_change) max_change = change;
            }
            
            /* Check convergence */
            if (max_change < tol) break;
        }
        
        free(residuals);
        free(beta_prev);
    }
    
    return FASTHAL_SUCCESS;
}

/* ============================================================================
 * Phase 3: Main fitting function
 * ============================================================================ */

int fit_gaussian(const real_t *X, idx_t n, idx_t p,
                 const real_t *y,
                 const basis_config_t *basis_cfg,
                 const fit_config_t *fit_cfg,
                 model_t **model_out) {
    
    if (!X || !y || !basis_cfg || !fit_cfg || !model_out) {
        return FASTHAL_ERROR_INVALID_ARGS;
    }
    
    /* Step 1: Allocate model */
    model_t *model = model_create(p, fit_cfg->n_lambda);
    if (!model) return FASTHAL_ERROR_ALLOCATION;
    
    model->family = FAMILY_GAUSSIAN;
    
    /* Step 2: Compute column statistics */
    column_stats_t *stats = column_stats_create(p);
    if (!stats) {
        model_destroy(model);
        return FASTHAL_ERROR_ALLOCATION;
    }
    
    if (compute_column_stats(X, n, p, stats) != FASTHAL_SUCCESS) {
        column_stats_destroy(stats);
        model_destroy(model);
        return FASTHAL_ERROR_INVALID_ARGS;
    }
    
    /* Step 3: Generate lambda grid */
    int ret = generate_lambda_grid(X, n, p, y, stats, fit_cfg->n_lambda,
                                    fit_cfg->lambda_min_ratio, model->λ);
    if (ret != FASTHAL_SUCCESS) {
        column_stats_destroy(stats);
        model_destroy(model);
        return ret;
    }
    
    /* Step 4: Simplified: fit directly on full data (no CV for now) */
    
    /* Standardize X for fitting */
    real_t *X_std = (real_t *)malloc((size_t)(n * p) * sizeof(real_t));
    if (!X_std) {
        column_stats_destroy(stats);
        model_destroy(model);
        return FASTHAL_ERROR_ALLOCATION;
    }
    
    for (idx_t j = 0; j < p; j++) {
        for (idx_t i = 0; i < n; i++) {
            X_std[i + j * n] = (X[i + j * n] - stats->means[j]) / stats->stds[j];
        }
    }
    
    /* Create sigma and invσ arrays */
    real_t *sigma2 = (real_t *)malloc((size_t)p * sizeof(real_t));
    real_t *invσ = (real_t *)malloc((size_t)p * sizeof(real_t));
    if (!sigma2 || !invσ) {
        free(X_std);
        free(sigma2);
        free(invσ);
        column_stats_destroy(stats);
        model_destroy(model);
        return FASTHAL_ERROR_ALLOCATION;
    }
    
    /* For standardized X, sigma2 and invσ are 1.0 */
    for (idx_t j = 0; j < p; j++) {
        sigma2[j] = 1.0;
        invσ[j] = 1.0;
    }
    
    /* Fit coordinate descent on standardized data */
    ret = coordinate_descent_simple(X_std, n, p, y,
                                    stats->means, invσ, sigma2,
                                    model->λ, fit_cfg->n_lambda,
                                    fit_cfg->tolerance, fit_cfg->max_iters,
                                    model->β_path, model->β0_path);
    
    free(X_std);
    free(sigma2);
    free(invσ);
    
    if (ret != FASTHAL_SUCCESS) {
        column_stats_destroy(stats);
        model_destroy(model);
        return ret;
    }
    
    /* Step 5: For now, select middle lambda as "best" */
    model->best_lambda_idx = fit_cfg->n_lambda / 2;
    model->best_lambda = model->λ[model->best_lambda_idx];
    
    /* Copy best solution */
    memcpy(model->β, model->β_path + model->best_lambda_idx * p,
           (size_t)p * sizeof(real_t));
    model->β0 = model->β0_path[model->best_lambda_idx];
    
    column_stats_destroy(stats);
    *model_out = model;
    
    return FASTHAL_SUCCESS;
}

int fit_binomial(const real_t *X, idx_t n, idx_t p,
                 const real_t *y_binary,
                 const basis_config_t *basis_cfg,
                 const fit_config_t *fit_cfg,
                 model_t **model_out) {
    
    (void)X;  /* Unused for now */
    (void)n;  /* Unused for now */
    
    /* TODO: Implement Newton-Raphson + coordinate descent */
    
    if (!X || !y_binary || !basis_cfg || !fit_cfg || !model_out) {
        return FASTHAL_ERROR_INVALID_ARGS;
    }
    
    model_t *model = model_create(p, fit_cfg->n_lambda);
    if (!model) return FASTHAL_ERROR_ALLOCATION;
    
    model->family = FAMILY_BINOMIAL;
    model->β0 = 0.0;
    model->best_lambda_idx = fit_cfg->n_lambda / 2;
    
    *model_out = model;
    return FASTHAL_SUCCESS;
}

/* ============================================================================
 * Prediction functions
 * ============================================================================ */

int predict_gaussian(const model_t *model,
                     const real_t *X_new, idx_t n_new, idx_t p,
                     const basis_config_t *basis_cfg,
                     real_t *y_pred_out) {
    
    if (!model || !X_new || !basis_cfg || !y_pred_out || !model->β) {
        return FASTHAL_ERROR_INVALID_ARGS;
    }
    
    /* Simple prediction: X_new * β + β0 */
    /* Assuming X_new is in same format as training data */
    
    for (idx_t i = 0; i < n_new; i++) {
        real_t pred = model->β0;
        for (idx_t j = 0; j < p; j++) {
            pred += X_new[i + j * n_new] * model->β[j];
        }
        y_pred_out[i] = pred;
    }
    
    return FASTHAL_SUCCESS;
}

int predict_binomial(const model_t *model,
                     const real_t *X_new, idx_t n_new, idx_t p,
                     const basis_config_t *basis_cfg,
                     real_t *prob_pred_out) {
    
    if (!model || !X_new || !basis_cfg || !prob_pred_out || !model->β) {
        return FASTHAL_ERROR_INVALID_ARGS;
    }
    
    /* Compute linear predictor and apply logistic transformation */
    for (idx_t i = 0; i < n_new; i++) {
        real_t eta = model->β0;
        for (idx_t j = 0; j < p; j++) {
            eta += X_new[i + j * n_new] * model->β[j];
        }
        
        /* Logistic: 1 / (1 + exp(-η)) */
        if (eta > 100.0) {
            prob_pred_out[i] = 1.0;
        } else if (eta < -100.0) {
            prob_pred_out[i] = 0.0;
        } else {
            prob_pred_out[i] = 1.0 / (1.0 + exp(-eta));
        }
    }
    
    return FASTHAL_SUCCESS;
}
