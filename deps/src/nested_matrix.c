#include "nested_matrix.h"
#include <string.h>

nested_matrix_t* nested_matrix_create(const idx_t *order, idx_t nrow, idx_t ncol) {
    if (!order || nrow <= 0 || ncol <= 0) return NULL;
    
    nested_matrix_t *m = (nested_matrix_t *)malloc(sizeof(nested_matrix_t));
    if (!m) return NULL;
    
    m->order = (idx_t *)malloc((size_t)nrow * sizeof(idx_t));
    if (!m->order) {
        free(m);
        return NULL;
    }
    
    memcpy(m->order, order, (size_t)nrow * sizeof(idx_t));
    m->ncol = ncol;
    m->nrow = nrow;
    
    return m;
}

void nested_matrix_destroy(nested_matrix_t *m) {
    if (!m) return;
    if (m->order) free(m->order);
    free(m);
}

nested_matrix_transpose_t* nested_matrix_transpose(const nested_matrix_t *m) {
    if (!m) return NULL;
    
    nested_matrix_transpose_t *mt = (nested_matrix_transpose_t *)malloc(sizeof(nested_matrix_transpose_t));
    if (!mt) return NULL;
    
    mt->order = (idx_t *)malloc((size_t)m->nrow * sizeof(idx_t));
    if (!mt->order) {
        free(mt);
        return NULL;
    }
    
    memcpy(mt->order, m->order, (size_t)m->nrow * sizeof(idx_t));
    mt->ncol = m->nrow;
    mt->nrow = m->ncol;
    
    return mt;
}

void nested_matrix_transpose_destroy(nested_matrix_transpose_t *mt) {
    if (!mt) return;
    if (mt->order) free(mt->order);
    free(mt);
}

/* Main forward multiplication: y = B * v using cumulative sum trick
 * 
 * Algorithm:
 * 1. Compute reverse cumulative sum: v_sum[i] = sum(v[i:])
 * 2. For each observation, output v_sum[order[i]] (the cumsum at that bin)
 * 
 * This is O(ncol + nrow) instead of O(ncol * nrow).
 */
int nested_matrix_mul(const nested_matrix_t *B, const real_t *v, real_t *y_out) {
    if (!B || !v || !y_out) return FASTHAL_ERROR_INVALID_ARGS;
    
    idx_t p = B->ncol;
    idx_t n = B->nrow;
    
    if (p <= 0 || n <= 0) return FASTHAL_ERROR_INVALID_ARGS;
    
    /* Compute reverse cumulative sum of v */
    real_t *v_sum = (real_t *)malloc((size_t)p * sizeof(real_t));
    if (!v_sum) return FASTHAL_ERROR_ALLOCATION;
    
    v_sum[p - 1] = v[p - 1];
    for (idx_t i = p - 2; i >= 0; i--) {
        v_sum[i] = v_sum[i + 1] + v[i];
    }
    
    /* For each observation, look up the cumsum at its bin */
    for (idx_t i = 0; i < n; i++) {
        idx_t bin = B->order[i];
        if (bin == 0) {
            y_out[i] = 0.0;
        } else {
            /* bin is 1-indexed, convert to 0-indexed */
            y_out[i] = v_sum[bin - 1];
        }
    }
    
    free(v_sum);
    return FASTHAL_SUCCESS;
}

/* Transpose multiplication: y = B^T * v
 * 
 * For each output bin j (1-indexed), we compute:
 *   y[j-1] = sum of v[i] for all observations i where order[i] >= j
 * 
 * This iterates through all bins j and accumulates contributions from
 * all observations whose bin assignment is >= j.
 */
int nested_matrix_transpose_mul(const nested_matrix_transpose_t *Bt,
                                 const real_t *v, real_t *y_out) {
    if (!Bt || !v || !y_out) return FASTHAL_ERROR_INVALID_ARGS;
    
    idx_t nrows_orig = Bt->ncol;  /* Number of observations */
    idx_t ncols_orig = Bt->nrow;  /* Number of basis functions */
    
    if (nrows_orig <= 0 || ncols_orig <= 0) return FASTHAL_ERROR_INVALID_ARGS;
    
    /* For each output bin j (1-indexed from 1 to ncols_orig) */
    for (idx_t j = 0; j < ncols_orig; j++) {
        real_t sum = 0.0;
        idx_t threshold = j + 1;  /* j-th output corresponds to threshold j+1 */
        
        /* Sum v[i] where order[i] >= threshold */
        for (idx_t i = 0; i < nrows_orig; i++) {
            if (Bt->order[i] >= threshold) {
                sum += v[i];
            }
        }
        
        y_out[j] = sum;
    }
    
    return FASTHAL_SUCCESS;
}
