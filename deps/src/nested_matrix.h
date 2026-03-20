#ifndef FASTHAL_NESTED_MATRIX_H
#define FASTHAL_NESTED_MATRIX_H

#include "common.h"
#include "memory_pool.h"

/* Nested matrix: matrix-free basis structure using cumulative sum trick.
 * 
 * A nested matrix represents indicator basis functions evaluated at observations,
 * where the basis functions are ordered by nested knots. Instead of storing
 * the full n x p matrix, we store:
 *   - order: For each observation, which bin it falls into (1-indexed, 0 if before all)
 *   - ncol: Number of basis functions (bins)
 *   - nrow: Number of observations
 * 
 * Multiplication by a vector is done using a cumulative sum trick:
 *   1. Compute reverse cumulative sum of v
 *   2. For each observation, look up pre-computed sum based on its bin
 * This is O(n + p) instead of O(n*p).
 */

typedef struct {
    idx_t *order;       /* order[i] = bin index for observation i (1-indexed) */
    idx_t ncol;         /* Number of basis functions */
    idx_t nrow;         /* Number of observations */
} nested_matrix_t;

typedef struct {
    idx_t *order;       /* Same as nested_matrix_t */
    idx_t ncol;         /* Number of observations (for transpose) */
    idx_t nrow;         /* Number of basis functions */
} nested_matrix_transpose_t;

/* Create a nested matrix from order array.
 * Args:
 *   order: Pre-computed bin indices (n elements)
 *   nrow: Number of observations
 *   ncol: Number of basis functions
 * Returns: Pointer to new nested_matrix_t or NULL on error
 */
nested_matrix_t* nested_matrix_create(const idx_t *order, idx_t nrow, idx_t ncol);

/* Destroy a nested matrix */
void nested_matrix_destroy(nested_matrix_t *m);

/* Transpose wrapper */
nested_matrix_transpose_t* nested_matrix_transpose(const nested_matrix_t *m);
void nested_matrix_transpose_destroy(nested_matrix_transpose_t *m);

/* Matrix-vector multiplication: y = B * v
 * Uses cumulative sum trick: O(n + ncol) instead of O(n * ncol)
 * 
 * Args:
 *   B: Nested matrix
 *   v: Coefficient vector (ncol elements)
 *   y_out: Output vector (nrow elements, pre-allocated)
 * Returns: FASTHAL_SUCCESS or error
 */
int nested_matrix_mul(const nested_matrix_t *B, const real_t *v, real_t *y_out);

/* Transpose multiplication: y = B^T * v
 * This computes the reverse cumulative sum operation.
 * 
 * Args:
 *   Bt: Transposed nested matrix
 *   v: Vector (nrow elements of original matrix)
 *   y_out: Output vector (ncol elements, pre-allocated)
 * Returns: FASTHAL_SUCCESS or error
 */
int nested_matrix_transpose_mul(const nested_matrix_transpose_t *Bt,
                                 const real_t *v, real_t *y_out);

#endif /* FASTHAL_NESTED_MATRIX_H */
