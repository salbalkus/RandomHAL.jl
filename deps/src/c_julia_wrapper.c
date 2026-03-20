#include "coordinate_descent.h"
#include <stdlib.h>
#include <string.h>

/* Simple wrapper for Julia: takes pre-computed basis matrix
 * and returns coordinate descent solution for a single lambda value.
 * 
 * This wrapper avoids the complexity of marshaling C structures across languages.
 * Instead, it takes simple arrays that represent the basis matrix already
 * computed in Julia.
 */

int coord_descent_gaussian_simple(
        const real_t *B_data,     /* Dense basis matrix (n x d) column-major */
        idx_t n,                   /* Number of observations */
        idx_t d,                   /* Number of basis columns */
        const real_t *y,           /* Response vector (n) */
        const real_t *μ,           /* Column means (d) */
        const real_t *invσ,        /* 1/sqrt(variance) for columns (d) */
        const real_t *σ2,          /* Column variance (d) */
        const real_t *λ_values,    /* Lambda path (n_λ) */
        idx_t n_λ,                 /* Number of lambda values */
        real_t tolerance,          /* Convergence tolerance */
        idx_t max_iterations,      /* Max iterations per lambda */
        real_t *β_out,             /* Output coefficients (d x n_λ) column-major */
        real_t *β0_out)            /* Output intercepts (n_λ) */
{
    if (!B_data || !y || !μ || !invσ || !σ2 || !λ_values || !β_out || !β0_out) {
        return FASTHAL_ERROR_INVALID_ARGS;
    }
    
    if (n <= 0 || d <= 0 || n_λ <= 0) {
        return FASTHAL_ERROR_INVALID_ARGS;
    }
    
    real_t α = 1.0;  /* LASSO */
    
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
    
    /* Compute mean of y */
    real_t mean_y = 0.0;
    for (idx_t i = 0; i < n; i++) {
        mean_y += y[i];
        residuals[i] = y[i];
    }
    mean_y /= (real_t)n;
    
    /* For each lambda */
    for (idx_t λ_idx = 0; λ_idx < n_λ; λ_idx++) {
        real_t λ = λ_values[λ_idx];
        real_t lasso_λ = λ * α;
        real_t ridge_factor = 1.0 / (1.0 + λ * (1.0 - α));
        
        /* Coordinate descent iterations */
        for (idx_t iter = 0; iter < max_iterations; iter++) {
            /* Cycle through all columns */
            real_t max_change = 0.0;
            
            for (idx_t j = 0; j < d; j++) {
                /* Compute gradient: grad = -B[:,j]^T * residuals */
                real_t grad = 0.0;
                for (idx_t i = 0; i < n; i++) {
                    grad += B_data[i + j * n] * residuals[i];  /* Column-major indexing */
                }
                
                real_t σ2_j = invσ[j] * invσ[j];
                if (σ2_j < 1e-15) continue;
                
                /* Unpenalized update */
                real_t β_unpenalized = β[j] + grad / (σ2_j * (real_t)n);
                
                /* Soft threshold */
                real_t β_new = soft_threshold(β_unpenalized, lasso_λ / σ2_j) * ridge_factor;
                
                /* Update residuals */
                real_t delta = β_new - β[j];
                if (fabs(delta) > 1e-15) {
                    for (idx_t i = 0; i < n; i++) {
                        residuals[i] -= delta * B_data[i + j * n];
                    }
                }
                
                real_t change = fabs(β_new - β[j]);
                if (change > max_change) max_change = change;
                
                β_prev[j] = β[j];
                β[j] = β_new;
            }
            
            /* Check convergence */
            if (max_change < tolerance) {
                break;
            }
        }
        
        /* Store result */
        for (idx_t j = 0; j < d; j++) {
            β_out[λ_idx * d + j] = β[j] * invσ[j];
        }
        
        /* Compute intercept */
        real_t mean_xβ = 0.0;
        for (idx_t j = 0; j < d; j++) {
            mean_xβ += μ[j] * β_out[λ_idx * d + j];
        }
        β0_out[λ_idx] = mean_y - mean_xβ;
        
        /* Reset for next lambda */
        memset(β, 0, (size_t)d * sizeof(real_t));
        memset(β_prev, 0, (size_t)d * sizeof(real_t));
        memcpy(residuals, y, (size_t)n * sizeof(real_t));
    }
    
    free(β);
    free(β_prev);
    free(residuals);
    
    return FASTHAL_SUCCESS;
}
