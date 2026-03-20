#ifndef FASTHAL_COORDINATE_DESCENT_H
#define FASTHAL_COORDINATE_DESCENT_H

#include "common.h"
#include "basis_matrix.h"
#include "memory_pool.h"

/* Gaussian coordinate descent solver using active set strategy.
 * 
 * Solves the elastic net problem:
 *   min_{β} 0.5*||y - X*β||^2 + λ*(α*||β||_1 + (1-α)/2*||β||_2^2)
 * 
 * Uses coordinate descent with active set screening to accelerate convergence.
 */

typedef struct {
    memory_context_t *mem_ctx;         /* Memory pooling context */
    real_t tolerance;                   /* Convergence tolerance */
    idx_t max_iterations;               /* Max iterations per λ */
    idx_t max_active_iterations;        /* Max iterations on active set */
} cd_gaussian_params_t;

/* Create parameter struct with defaults */
cd_gaussian_params_t coord_descent_gaussian_params_default(void);

/* Note: soft_threshold is defined in common.h as a static inline function */

/* Fit Gaussian model using coordinate descent.
 * 
 * Args:
 *   B: Basis matrix blocks
 *   y: Centered and scaled response (n elements)
 *   μ: Column means of B (d elements)
 *   invσ: 1/sqrt(var) for each column (d elements)
 *   σ2: Column variance (d elements), for convergence tracking
 *   λ_values: Array of regularization parameters (len n_λ)
 *   n_λ: Number of λ values
 *   params: Algorithm parameters
 *   β_out: Output coefficient paths (d x n_λ), pre-allocated
 *   β0_out: Intercept path (n_λ), pre-allocated
 * 
 * Returns: FASTHAL_SUCCESS or error code
 */
typedef struct {
    basis_matrix_t **blocks;            /* Array of basis matrices per section */
    idx_t num_blocks;                   /* Number of blocks */
    idx_t total_cols;                   /* Total number of columns across blocks */
    idx_t n;                            /* Number of observations */
} basis_matrix_blocks_t;

/* Create basis matrix blocks (empty structure, filled with matrices) */
basis_matrix_blocks_t* basis_matrix_blocks_create(idx_t num_blocks, idx_t n);
void basis_matrix_blocks_destroy(basis_matrix_blocks_t *b);

int coord_descent_gaussian(const basis_matrix_blocks_t *B,
                           const real_t *y,
                           const real_t *μ,
                           const real_t *invσ,
                           const real_t *σ2,
                           const real_t *λ_values,
                           idx_t n_λ,
                           const cd_gaussian_params_t *params,
                           real_t *β_out,
                           real_t *β0_out);

#endif /* FASTHAL_COORDINATE_DESCENT_H */
