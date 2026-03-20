#ifndef FASTHAL_FIT_MODEL_H
#define FASTHAL_FIT_MODEL_H

#include "common.h"
#include "basis_matrix.h"
#include "memory_pool.h"

/* High-level model fitting interface.
 * 
 * This module provides end-to-end fitting with:
 * - Basis construction from raw data
 * - Variance estimation
 * - λ grid selection
 * - K-fold cross-validation
 * - Final model refitting
 */

typedef enum {
    FAMILY_GAUSSIAN,
    FAMILY_BINOMIAL
} family_t;

typedef struct {
    idx_t num_sections;           /* Number of interaction sections */
    idx_t *section_sizes;         /* Size of each section */
    idx_t **sections;             /* Section indices (variable subsets) */
    int smoothness;               /* Smoothness parameter (0, 1, 2, ...) */
    idx_t max_basis_size;         /* Max basis functions per section */
    real_t min_support;           /* Min proportion of observations for basis */
} basis_config_t;

typedef struct {
    idx_t K;                      /* Number of folds for CV */
    idx_t max_iters;              /* Max iterations per λ */
    real_t tolerance;             /* Convergence tolerance */
    real_t alpha;                 /* Elastic net: α=1 is LASSO, α=0 is ridge */
    idx_t n_lambda;               /* Number of λ values */
    real_t lambda_min_ratio;      /* λ_min = ratio * λ_max */
    bool auto_lambda;             /* Auto-generate λ grid */
} fit_config_t;

typedef struct {
    real_t *β;                    /* Final coefficients */
    real_t β0;                    /* Final intercept */
    real_t *β_path;               /* Coefficient paths (d x n_λ) */
    real_t *β0_path;              /* Intercept paths (n_λ) */
    real_t *λ;                    /* λ values used */
    idx_t n_lambda;               /* Number of λ values */
    idx_t best_lambda_idx;        /* Index of best λ */
    real_t best_lambda;           /* Best λ value */
    real_t best_cv_error;         /* CV error at best λ */
    family_t family;              /* Model family */
} model_t;

/* Create basis config */
basis_config_t* basis_config_create(idx_t num_sections);
void basis_config_destroy(basis_config_t *config);
void basis_config_add_section(basis_config_t *config, idx_t s_idx,
                               const idx_t *columns, idx_t num_columns);

/* Create fit config with defaults */
fit_config_t* fit_config_create(void);
void fit_config_destroy(fit_config_t *config);

/* Create model result structure */
model_t* model_create(idx_t d, idx_t n_lambda);
void model_destroy(model_t *model);

/* Main fitting function: Gaussian */
int fit_gaussian(const real_t *X, idx_t n, idx_t p,
                 const real_t *y,
                 const basis_config_t *basis_cfg,
                 const fit_config_t *fit_cfg,
                 model_t **model_out);

/* Main fitting function: Binomial */
int fit_binomial(const real_t *X, idx_t n, idx_t p,
                 const real_t *y_binary,
                 const basis_config_t *basis_cfg,
                 const fit_config_t *fit_cfg,
                 model_t **model_out);

/* Prediction functions */
int predict_gaussian(const model_t *model,
                     const real_t *X_new, idx_t n_new, idx_t p,
                     const basis_config_t *basis_cfg,
                     real_t *y_pred_out);

int predict_binomial(const model_t *model,
                     const real_t *X_new, idx_t n_new, idx_t p,
                     const basis_config_t *basis_cfg,
                     real_t *prob_pred_out);

#endif /* FASTHAL_FIT_MODEL_H */
