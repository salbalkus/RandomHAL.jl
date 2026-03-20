#include "basis_matrix.h"
#include <math.h>
#include <string.h>

/* Helper: compute factorial */
static real_t factorial(int n) {
    if (n <= 1) return 1.0;
    real_t result = 1.0;
    for (int i = 2; i <= n; i++) {
        result *= (real_t)i;
    }
    return result;
}



basis_matrix_t* basis_matrix_create(const real_t *X, idx_t n, idx_t p,
                                     const idx_t *section, idx_t section_size,
                                     int smoothness, const idx_t *ranks_matrix) {
    if (!X || !section || !ranks_matrix || n <= 0 || p <= 0 || section_size <= 0) {
        return NULL;
    }
    
    basis_matrix_t *b = (basis_matrix_t *)malloc(sizeof(basis_matrix_t));
    if (!b) return NULL;
    
    /* Copy section */
    b->section = (idx_t *)malloc((size_t)section_size * sizeof(idx_t));
    if (!b->section) {
        free(b);
        return NULL;
    }
    memcpy(b->section, section, (size_t)section_size * sizeof(idx_t));
    b->section_size = section_size;
    b->smoothness = smoothness;
    
    /* Compute l vector: product of features raised to smoothness power */
    b->l = (real_t *)malloc((size_t)n * sizeof(real_t));
    if (!b->l) {
        free(b->section);
        free(b);
        return NULL;
    }
    
    real_t smooth_factorial = factorial(smoothness);
    for (idx_t i = 0; i < n; i++) {
        real_t prod = 1.0;
        for (idx_t s = 0; s < section_size; s++) {
            idx_t col = section[s];
            real_t x = X[col * n + i];
            prod *= pow(x, (real_t)smoothness);
        }
        b->l[i] = prod / smooth_factorial;
    }
    
    /* Sample nested path and compute nested matrix F */
    idx_t *path = (idx_t *)malloc((size_t)n * sizeof(idx_t));
    if (!path) {
        free(b->l);
        free(b->section);
        free(b);
        return NULL;
    }
    
    idx_t path_len = 0;
    
    /* Extract ranks for this section */
    idx_t *ranks_section = (idx_t *)malloc((size_t)(section_size * n) * sizeof(idx_t));
    if (!ranks_section) {
        free(path);
        free(b->l);
        free(b->section);
        free(b);
        return NULL;
    }
    
    for (idx_t s = 0; s < section_size; s++) {
        idx_t col = section[s];
        for (idx_t i = 0; i < n; i++) {
            ranks_section[s * n + i] = ranks_matrix[col * n + i];
        }
    }
    
    int status = path_sample(ranks_section, n, section_size, path, &path_len);
    if (status != FASTHAL_SUCCESS) {
        free(ranks_section);
        free(path);
        free(b->l);
        free(b->section);
        free(b);
        return NULL;
    }
    
    /* Compute binary bin search to get order array for nested matrix */
    idx_t *order = (idx_t *)malloc((size_t)n * sizeof(idx_t));
    if (!order) {
        free(ranks_section);
        free(path);
        free(b->l);
        free(b->section);
        free(b);
        return NULL;
    }
    
    status = binary_bin_search(X, n, X, path, section, section_size, path_len, order);
    if (status != FASTHAL_SUCCESS) {
        free(order);
        free(ranks_section);
        free(path);
        free(b->l);
        free(b->section);
        free(b);
        return NULL;
    }
    
    /* Create nested matrix F */
    b->F = nested_matrix_create(order, n, path_len);
    if (!b->F) {
        free(order);
        free(ranks_section);
        free(path);
        free(b->l);
        free(b->section);
        free(b);
        return NULL;
    }
    
    /* Compute r vector: intercepts from knot values */
    b->r = (real_t *)malloc((size_t)path_len * sizeof(real_t));
    if (!b->r) {
        nested_matrix_destroy(b->F);
        free(order);
        free(ranks_section);
        free(path);
        free(b->l);
        free(b->section);
        free(b);
        return NULL;
    }
    
    for (idx_t j = 0; j < path_len; j++) {
        idx_t knot_idx = path[j];
        real_t prod = 1.0;
        for (idx_t s = 0; s < section_size; s++) {
            idx_t col = section[s];
            real_t x = X[col * n + knot_idx];
            prod *= pow(x, (real_t)smoothness);
        }
        b->r[j] = prod / smooth_factorial;
    }
    
    /* Reverse r (to match Julia's "mirrored" basis behavior) */
    for (idx_t j = 0; j < path_len / 2; j++) {
        real_t temp = b->r[j];
        b->r[j] = b->r[path_len - 1 - j];
        b->r[path_len - 1 - j] = temp;
    }
    
    b->nrow = n;
    b->ncol = path_len;
    
    /* Cleanup */
    free(order);
    free(ranks_section);
    free(path);
    
    return b;
}

void basis_matrix_destroy(basis_matrix_t *b) {
    if (!b) return;
    if (b->section) free(b->section);
    if (b->l) free(b->l);
    if (b->r) free(b->r);
    if (b->F) nested_matrix_destroy(b->F);
    free(b);
}

basis_matrix_transpose_t* basis_matrix_transpose(const basis_matrix_t *b) {
    if (!b) return NULL;
    
    basis_matrix_transpose_t *bt = (basis_matrix_transpose_t *)malloc(sizeof(basis_matrix_transpose_t));
    if (!bt) return NULL;
    
    bt->section = (idx_t *)malloc((size_t)b->section_size * sizeof(idx_t));
    if (!bt->section) {
        free(bt);
        return NULL;
    }
    memcpy(bt->section, b->section, (size_t)b->section_size * sizeof(idx_t));
    
    bt->section_size = b->section_size;
    bt->smoothness = b->smoothness;
    bt->nrow = b->ncol;
    bt->ncol = b->nrow;
    
    bt->l = b->l;
    bt->r = b->r;
    
    bt->F = nested_matrix_transpose(b->F);
    if (!bt->F) {
        free(bt->section);
        free(bt);
        return NULL;
    }
    
    return bt;
}

void basis_matrix_transpose_destroy(basis_matrix_transpose_t *bt) {
    if (!bt) return;
    if (bt->section) free(bt->section);
    if (bt->F) nested_matrix_transpose_destroy(bt->F);
    free(bt);
}

/* y = (l ✓ F*v) - (r ✓ F*(r.*v))
 * 
 * Can be rewritten as:
 *   temp1 = F * v
 *   temp2 = F * (r .* v)
 *   y = (l .* temp1) - (r .* temp2)
 * 
 * But this requires two nested_matrix_mul calls. For efficiency, we could
 * optimize this, but for now we prioritize correctness.
 */
int basis_matrix_mul(const basis_matrix_t *B, const real_t *v, real_t *y_out) {
    if (!B || !v || !y_out) return FASTHAL_ERROR_INVALID_ARGS;
    
    /* Compute temp1 = F * v */
    real_t *temp1 = (real_t *)malloc((size_t)B->nrow * sizeof(real_t));
    if (!temp1) return FASTHAL_ERROR_ALLOCATION;
    
    int status = nested_matrix_mul(B->F, v, temp1);
    if (status != FASTHAL_SUCCESS) {
        free(temp1);
        return status;
    }
    
    /* Compute temp2 = F * (r .* v) */
    real_t *rv = (real_t *)malloc((size_t)B->ncol * sizeof(real_t));
    if (!rv) {
        free(temp1);
        return FASTHAL_ERROR_ALLOCATION;
    }
    for (idx_t j = 0; j < B->ncol; j++) {
        rv[j] = B->r[j] * v[j];
    }
    
    real_t *temp2 = (real_t *)malloc((size_t)B->nrow * sizeof(real_t));
    if (!temp2) {
        free(rv);
        free(temp1);
        return FASTHAL_ERROR_ALLOCATION;
    }
    
    status = nested_matrix_mul(B->F, rv, temp2);
    if (status != FASTHAL_SUCCESS) {
        free(temp2);
        free(rv);
        free(temp1);
        return status;
    }
    
    /* Compute y = (l .* temp1) - (r .* temp2)
     * Wait, that's not quite right. Let me re-think the Hadamard product.
     * 
     * The basis matrix is defined as:
     *   B[i,j] = (l[i] - r[j]) * F[i,j]
     * 
     * So:
     *   (B * v)[i] = sum_j B[i,j] * v[j]
     *              = sum_j (l[i] - r[j]) * F[i,j] * v[j]
     *              = l[i] * (F*v)[i] - (F*(r.*v))[i]
     */
    for (idx_t i = 0; i < B->nrow; i++) {
        y_out[i] = B->l[i] * temp1[i] - temp2[i];
    }
    
    free(temp2);
    free(rv);
    free(temp1);
    return FASTHAL_SUCCESS;
}

int basis_matrix_transpose_mul(const basis_matrix_transpose_t *Bt,
                                const real_t *v, real_t *y_out) {
    if (!Bt || !v || !y_out) return FASTHAL_ERROR_INVALID_ARGS;
    
    /* y = B^T * v
     * where B[i,j] = (l[i] - r[j]) * F[i,j]
     * 
     * So:
     *   (B^T * v)[j] = sum_i B[i,j] * v[i]
     *                = sum_i (l[i] - r[j]) * F[i,j] * v[i]
     *                = (F^T * (l .* v))[j] - r[j] * (F^T * v)[j]
     */
    
    /* Compute l .* v */
    real_t *lv = (real_t *)malloc((size_t)Bt->ncol * sizeof(real_t));
    if (!lv) return FASTHAL_ERROR_ALLOCATION;
    
    for (idx_t i = 0; i < Bt->ncol; i++) {
        lv[i] = Bt->l[i] * v[i];
    }
    
    /* Compute temp1 = F^T * (l .* v) */
    real_t *temp1 = (real_t *)malloc((size_t)Bt->nrow * sizeof(real_t));
    if (!temp1) {
        free(lv);
        return FASTHAL_ERROR_ALLOCATION;
    }
    
    int status = nested_matrix_transpose_mul(Bt->F, lv, temp1);
    if (status != FASTHAL_SUCCESS) {
        free(temp1);
        free(lv);
        return status;
    }
    
    /* Compute temp2 = F^T * v */
    real_t *temp2 = (real_t *)malloc((size_t)Bt->nrow * sizeof(real_t));
    if (!temp2) {
        free(temp1);
        free(lv);
        return FASTHAL_ERROR_ALLOCATION;
    }
    
    status = nested_matrix_transpose_mul(Bt->F, v, temp2);
    if (status != FASTHAL_SUCCESS) {
        free(temp2);
        free(temp1);
        free(lv);
        return status;
    }
    
    /* Compute y = temp1 - (r .* temp2) */
    for (idx_t j = 0; j < Bt->nrow; j++) {
        y_out[j] = temp1[j] - Bt->r[j] * temp2[j];
    }
    
    free(temp2);
    free(temp1);
    free(lv);
    return FASTHAL_SUCCESS;
}
