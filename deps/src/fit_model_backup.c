#include "fit_model.h"
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
 * Main fitting functions (stubs for Phase 2-3)
 * ============================================================================ */

int fit_gaussian(const real_t *X, idx_t n, idx_t p,
                 const real_t *y,
                 const basis_config_t *basis_cfg,
                 const fit_config_t *fit_cfg,
                 model_t **model_out) {
    /* TODO: Phase 2-3 implementation
     * 1. Compute ranks for all columns
     * 2. Create basis matrices for each section
     * 3. Compute column statistics (means, variances)
     * 4. Generate λ grid
     * 5. Run K-fold CV with coordinate descent
     * 6. Select best λ and refit
     * 7. Return model
     */
    
    if (!X || !y || !basis_cfg || !fit_cfg || !model_out) {
        return FASTHAL_ERROR_INVALID_ARGS;
    }
    
    /* Allocate model structure */
    model_t *model = model_create(p, fit_cfg->n_lambda);
    if (!model) return FASTHAL_ERROR_ALLOCATION;
    
    model->family = FAMILY_GAUSSIAN;
    
    /* Placeholder: set default outputs */
    model->β0 = 0.0;
    model->best_lambda_idx = fit_cfg->n_lambda / 2;
    
    *model_out = model;
    return FASTHAL_SUCCESS;
}

int fit_binomial(const real_t *X, idx_t n, idx_t p,
                 const real_t *y_binary,
                 const basis_config_t *basis_cfg,
                 const fit_config_t *fit_cfg,
                 model_t **model_out) {
    /* TODO: Phase 2 binomial (Newton-Raphson inside coordinate descent) */
    
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
    /* TODO: Phase 3 prediction
     * 1. Reconstruct basis matrices from config and new X
     * 2. Compute y_pred = X_new * β + β0
     */
    
    if (!model || !X_new || !basis_cfg || !y_pred_out) {
        return FASTHAL_ERROR_INVALID_ARGS;
    }
    
    /* Placeholder: return zero predictions */
    memset(y_pred_out, 0, (size_t)n_new * sizeof(real_t));
    
    return FASTHAL_SUCCESS;
}

int predict_binomial(const model_t *model,
                     const real_t *X_new, idx_t n_new, idx_t p,
                     const basis_config_t *basis_cfg,
                     real_t *prob_pred_out) {
    /* TODO: Phase 3 binomial prediction
     * 1. Reconstruct basis matrices from config and new X
     * 2. Compute linear predictor: η = X_new * β + β0
     * 3. Apply logit: prob = 1 / (1 + exp(-η))
     */
    
    if (!model || !X_new || !basis_cfg || !prob_pred_out) {
        return FASTHAL_ERROR_INVALID_ARGS;
    }
    
    /* Placeholder: return 0.5 predictions */
    for (idx_t i = 0; i < n_new; i++) {
        prob_pred_out[i] = 0.5;
    }
    
    return FASTHAL_SUCCESS;
}
