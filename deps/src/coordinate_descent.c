#include "coordinate_descent.h"
#include "basis_matrix.h"
#include "memory_pool.h"
#include <string.h>
#include <math.h>
#include <stdbool.h>

cd_gaussian_params_t coord_descent_gaussian_params_default(void) {
    cd_gaussian_params_t params;
    params.mem_ctx = NULL;
    params.tolerance = 1e-7;
    params.max_iterations = 1000;
    params.max_active_iterations = 1000;
    return params;
}

basis_matrix_blocks_t* basis_matrix_blocks_create(idx_t num_blocks, idx_t n) {
    basis_matrix_blocks_t *b = (basis_matrix_blocks_t *)malloc(sizeof(basis_matrix_blocks_t));
    if (!b) return NULL;
    
    b->blocks = (basis_matrix_t **)malloc((size_t)num_blocks * sizeof(basis_matrix_t *));
    if (!b->blocks) {
        free(b);
        return NULL;
    }
    
    b->num_blocks = num_blocks;
    b->total_cols = 0;
    b->n = n;
    
    return b;
}

void basis_matrix_blocks_destroy(basis_matrix_blocks_t *b) {
    if (!b) return;
    if (b->blocks) {
        for (idx_t i = 0; i < b->num_blocks; i++) {
            if (b->blocks[i]) basis_matrix_destroy(b->blocks[i]);
        }
        free(b->blocks);
    }
    free(b);
}

/* ============================================================================
 * Compute gradient of squared error loss for one column:
 * grad[j] = -2 * (B[:,j]^T * residuals) / n
 * ============================================================================ */

static real_t compute_gradient_j(const basis_matrix_t *B,
                                 const real_t *residuals,
                                 idx_t j) {
    if (!B || !residuals || j >= B->ncol) return 0.0;
    
    /* Compute B[:,j]^T * residuals efficiently using basis structure
     * B[i,j] = (l[i] - r[j]) * F[i,j]  where F[i,j] = indicator(order[i] >= j+1)
     * 
     * grad = sum_i B[i,j] * residuals[i]
     *      = sum_i (l[i] - r[j]) * F[i,j] * residuals[i]
     *      = sum_i l[i] * F[i,j] * residuals[i] - r[j] * sum_i F[i,j] * residuals[i]
     */
    
    real_t sum_f_res = 0.0;      /* sum_i F[i,j] * residuals[i] */
    real_t sum_lf_res = 0.0;     /* sum_i l[i] * F[i,j] * residuals[i] */
    
    for (idx_t i = 0; i < B->nrow; i++) {
        if (B->F->order[i] >= j + 1) {  /* F[i,j] = 1 */
            sum_f_res += residuals[i];
            sum_lf_res += B->l[i] * residuals[i];
        }
    }
    
    return sum_lf_res - B->r[j] * sum_f_res;
}

/* Update residuals after coefficient change: residuals -= delta * B[:,j]
 * Efficiently updates using basis structure
 */
static void update_residuals_j(const basis_matrix_t *B,
                               real_t *residuals,
                               idx_t j,
                               real_t delta) {
    if (!B || !residuals || j >= B->ncol || fabs(delta) < 1e-15) return;
    
    for (idx_t i = 0; i < B->nrow; i++) {
        if (B->F->order[i] >= j + 1) {  /* B[i,j] non-zero */
            real_t bij = (B->l[i] - B->r[j]);
            residuals[i] -= delta * bij;
        }
    }
}

/* ============================================================================
 * Single pass through active set in coordinate descent
 * ============================================================================ */

static void cycle_active_set(const basis_matrix_blocks_t *blocks,
                             real_t *residuals,
                             const real_t *invσ,
                             real_t *β,
                             real_t *β_prev,
                             real_t λ,
                             real_t α,
                             uint8_t *active,
                             idx_t n) {
    if (!blocks || !residuals || !β || !active || !invσ || !β_prev) return;
    
    idx_t col_idx = 0;
    real_t lasso_λ = λ * α;
    real_t ridge_factor = 1.0 / (1.0 + λ * (1.0 - α));
    
    for (idx_t b = 0; b < blocks->num_blocks; b++) {
        basis_matrix_t *B = blocks->blocks[b];
        if (!B) continue;
        
        for (idx_t j = 0; j < B->ncol; j++) {
            if (col_idx >= blocks->total_cols) break;  /* Safety check */
            
            uint8_t byte_idx = col_idx / 8;
            uint8_t bit_idx = col_idx % 8;
            bool is_active = (active[byte_idx] & (1U << bit_idx)) != 0;
            
            if (!is_active) {
                col_idx++;
                continue;
            }
            
            real_t grad = compute_gradient_j(B, residuals, j);
            real_t σ2_j = invσ[col_idx] * invσ[col_idx];
            
            if (σ2_j < 1e-15) {
                col_idx++;
                continue;
            }
            
            real_t β_unpenalized = β[col_idx] + grad / (σ2_j * (real_t)n);
            real_t β_new = soft_threshold(β_unpenalized, lasso_λ / σ2_j) * ridge_factor;
            
            real_t delta = β_new - β[col_idx];
            if (fabs(delta) > 1e-15) {
                update_residuals_j(B, residuals, j, delta);
            }
            
            β_prev[col_idx] = β[col_idx];
            β[col_idx] = β_new;
            col_idx++;
        }
    }
}

/* ============================================================================
 * Main Gaussian coordinate descent solver with warm starts over lambda path
 * ============================================================================ */

int coord_descent_gaussian(const basis_matrix_blocks_t *B,
                           const real_t *y,
                           const real_t *μ,
                           const real_t *invσ,
                           const real_t *σ2,
                           const real_t *λ_values,
                           idx_t n_λ,
                           const cd_gaussian_params_t *params,
                           real_t *β_out,
                           real_t *β0_out) {
    if (!B || !y || !μ || !invσ || !σ2 || !λ_values || !β_out || !β0_out) {
        return FASTHAL_ERROR_INVALID_ARGS;
    }
    
    if (n_λ <= 0 || B->total_cols <= 0 || B->n <= 0) {
        return FASTHAL_ERROR_INVALID_ARGS;
    }
    
    idx_t n = B->n;
    idx_t d = B->total_cols;
    real_t α = 1.0;
    
    /* Early return if no columns to optimize */
    if (d <= 0) {
        /* Just return mean intercept for all lambda values */
        real_t mean_y = 0.0;
        for (idx_t i = 0; i < n; i++) {
            mean_y += y[i];
        }
        mean_y /= (real_t)n;
        
        for (idx_t λ_idx = 0; λ_idx < n_λ; λ_idx++) {
            β0_out[λ_idx] = mean_y;
        }
        return FASTHAL_SUCCESS;
    }
    
    /* Allocate working arrays */
    real_t *β = (real_t *)calloc((size_t)d, sizeof(real_t));
    real_t *β_prev = (real_t *)calloc((size_t)d, sizeof(real_t));
    real_t *residuals = (real_t *)malloc((size_t)n * sizeof(real_t));
    
    if (!β || !β_prev || !residuals) {
        free(β);
        free(β_prev);
        free(residuals);
        return FASTHAL_ERROR_ALLOCATION;
    }
    
    /* Compute mean of y for intercept computation */
    real_t mean_y = 0.0;
    for (idx_t i = 0; i < n; i++) {
        mean_y += y[i];
        residuals[i] = y[i];  /* Initialize residuals */
    }
    mean_y /= (real_t)n;
    
    /* For each λ value (warm-started from previous solution) */
    for (idx_t λ_idx = 0; λ_idx < n_λ; λ_idx++) {
        real_t λ = λ_values[λ_idx];
        
        /* Initialize active set (all non-zero variance columns) */
        idx_t active_bytes = (d + 7) / 8;
        uint8_t *active = (uint8_t *)malloc((size_t)active_bytes * sizeof(uint8_t));
        if (!active) {
            free(β);
            free(β_prev);
            free(residuals);
            return FASTHAL_ERROR_ALLOCATION;
        }
        memset(active, 0xFF, (size_t)active_bytes);  /* All true initially */
        
        /* Outer loop: check if active set changes */
        for (idx_t iter = 0; iter < params->max_active_iterations; iter++) {
            /* Inner loop: cycle through active set until convergence */
            for (idx_t cd_iter = 0; cd_iter < params->max_iterations; cd_iter++) {
                cycle_active_set(B, residuals, invσ, β, β_prev, λ, α, active, n);
                
                /* Check convergence: max coefficient change */
                real_t max_change = 0.0;
                for (idx_t j = 0; j < d; j++) {
                    real_t change = fabs(β[j] - β_prev[j]);
                    if (change > max_change) max_change = change;
                }
                
                if (max_change < params->tolerance) {
                    break;  /* Converged for this lambda */
                }
            }
            
            /* Screen active set: deactivate columns below threshold */
            bool any_removed = false;
            for (idx_t j = 0; j < d; j++) {
                uint8_t byte_idx = j / 8;
                uint8_t bit_idx = j % 8;
                bool is_active = (active[byte_idx] & (1U << bit_idx)) != 0;
                
                if (!is_active) continue;
                
                /* Compute KKT condition: |gradient| <= λ indicates can be inactive */
                real_t grad = 0.0;  /* Would compute full gradient here if screened */
                if (fabs(grad) <= λ) {
                    active[byte_idx] &= ~(1U << bit_idx);  /* Deactivate */
                    any_removed = true;
                }
            }
            
            if (!any_removed) {
                break;  /* Converged: no more variables to screen */
            }
        }
        
        /* Store result: scale by inverse sigma and store coefficient path */
        for (idx_t j = 0; j < d; j++) {
            β_out[λ_idx * d + j] = β[j] * invσ[j];
        }
        
        /* Compute intercept: β0 = mean(y) - mean(X) * β_scaled */
        real_t mean_xβ = 0.0;
        for (idx_t j = 0; j < d; j++) {
            mean_xβ += μ[j] * β_out[λ_idx * d + j];
        }
        β0_out[λ_idx] = mean_y - mean_xβ;
        
        /* Prepare for next lambda: reset coefficients but warm-start from current
         * (Warm starts significantly speed up the path computation)
         */
        memset(β_prev, 0, (size_t)d * sizeof(real_t));
        
        free(active);
    }
    
    /* Cleanup */
    free(β);
    free(β_prev);
    free(residuals);
    
    return FASTHAL_SUCCESS;
}
