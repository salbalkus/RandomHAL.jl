#ifndef FASTHAL_BASIS_MATRIX_H
#define FASTHAL_BASIS_MATRIX_H

#include "common.h"
#include "nested_matrix.h"
#include "rank_and_path.h"
#include "memory_pool.h"

/* Basis matrix: Hadamard product structure B = (l ✓ F) - (r ✓ F)
 * 
 * This represents a single smooth HAL basis function block as:
 *   B[i,j] = l[i] * F[i,j] - r[j] * F[i,j]
 *          = (l[i] - r[j]) * F[i,j]
 * 
 * where:
 *   - F is a nested matrix (indicator basis)
 *   - l is the product of features: l[i] = ∏(X[i,s]^smooth) / smooth!
 *   - r is a vector of intercepts computed from knot values
 *   - smooth is the smoothness parameter (0, 1, 2, ...)
 */

typedef struct {
    idx_t *section;              /* Subset of columns (smoother dimensions) */
    idx_t section_size;          /* Length of section */
    nested_matrix_t *F;          /* Nested indicator basis */
    real_t *l;                   /* Product vector for all observations */
    real_t *r;                   /* Intercept vector for all basis functions */
    int smoothness;              /* Smoothness parameter */
    idx_t nrow;                  /* Number of observations */
    idx_t ncol;                  /* Number of basis functions */
} basis_matrix_t;

typedef struct {
    idx_t *section;
    idx_t section_size;
    nested_matrix_transpose_t *F;
    real_t *l;
    real_t *r;
    int smoothness;
    idx_t nrow;
    idx_t ncol;
} basis_matrix_transpose_t;

/* Create a basis matrix from a section of X and ranks.
 * 
 * Args:
 *   X: Full data matrix (n x p), column-major
 *   n: Number of observations
 *   p: Number of features
 *   section: Subset of column indices to use (section_size elements)
 *   section_size: Length of section
 *   smoothness: Smoothness parameter (0, 1, 2, ...)
 *   ranks_matrix: Pre-computed ranks for all columns (n x p)
 * Returns: Pointer to new basis_matrix_t or NULL on error
 */
basis_matrix_t* basis_matrix_create(const real_t *X, idx_t n, idx_t p,
                                     const idx_t *section, idx_t section_size,
                                     int smoothness, const idx_t *ranks_matrix);

void basis_matrix_destroy(basis_matrix_t *b);

basis_matrix_transpose_t* basis_matrix_transpose(const basis_matrix_t *b);
void basis_matrix_transpose_destroy(basis_matrix_transpose_t *bt);

/* Matrix-vector multiplication: y = B * v
 * Computes: y = (l ✓ F*v) - (r ✓ F*(r.*v))
 * 
 * Args:
 *   B: Basis matrix
 *   v: Coefficient vector (ncol elements)
 *   y_out: Output vector (nrow elements, pre-allocated)
 * Returns: FASTHAL_SUCCESS or error
 */
int basis_matrix_mul(const basis_matrix_t *B, const real_t *v, real_t *y_out);

/* Transpose multiplication: y = B^T * v */
int basis_matrix_transpose_mul(const basis_matrix_transpose_t *Bt,
                                const real_t *v, real_t *y_out);

#endif /* FASTHAL_BASIS_MATRIX_H */
