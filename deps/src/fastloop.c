#include <stdint.h>
#include <math.h>

// Helper: Standard Soft Thresholding function
static inline double soft_threshold(double x, double lambda) {
    double abs_x = fabs(x);
    if (abs_x <= lambda) {
        return 0.0;
    }
    return (x > 0 ? 1.0 : -1.0) * (abs_x - lambda);
}

/**
 * Updates 'beta' and 'delta_r' in-place. 
 * Returns nothing (void).
 */
void coordinate_descent_update(
    int64_t* indices,
    int32_t n_indices,
    int8_t* active,
    double* beta,
    double* beta_unpenalized,
    double* mu_inv_sigma_diff,
    double* mu_inv_sigma,
    double lasso_penalty,
    double ridge_penalty
) {
    double delta_r;  
    for (int i = 0; i < n_indices; i++) {
        int64_t k = indices[i];

        if (active[k]) {
            double beta_prev = beta[k];
            
            // Update beta in-place
            beta[k] = soft_threshold(beta_unpenalized[k] - (mu_inv_sigma_diff[k] * delta_r), lasso_penalty) / ridge_penalty;

            // Update delta_r in-place via pointer dereferencing
            delta_r += mu_inv_sigma[k] * (beta[k] - beta_prev);
        }
    }
}