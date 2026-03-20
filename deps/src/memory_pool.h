#ifndef FASTHAL_MEMORY_POOL_H
#define FASTHAL_MEMORY_POOL_H

#include "common.h"

/* Memory pool for reusing vector allocations across iterations.
 * This is critical for coordinate descent where we allocate vectors
 * repeatedly across λ values and folds. */

typedef struct {
    vector_t **vectors;
    idx_t num_vectors;
    idx_t capacity;
} vector_pool_t;

typedef struct {
    ivector_t **ivectors;
    idx_t num_vectors;
    idx_t capacity;
} ivector_pool_t;

typedef struct {
    bitvector_t **bitvectors;
    idx_t num_vectors;
    idx_t capacity;
} bitvector_pool_t;

typedef struct {
    matrix_t **matrices;
    idx_t num_matrices;
    idx_t capacity;
} matrix_pool_t;

/* Context for a single fitting session - holds all pools */
typedef struct {
    vector_pool_t vector_pool;
    ivector_pool_t ivector_pool;
    bitvector_pool_t bitvector_pool;
    matrix_pool_t matrix_pool;
} memory_context_t;

/* Initialize/destroy context */
memory_context_t* memory_context_create(void);
void memory_context_destroy(memory_context_t *ctx);

/* Vector pool operations - get a vector of at least length 'n', cleared to zero */
vector_t* vector_pool_get(memory_context_t *ctx, idx_t length);
void vector_pool_reset(memory_context_t *ctx);  /* Clear all vectors for next iteration */

/* Integer vector pool */
ivector_t* ivector_pool_get(memory_context_t *ctx, idx_t length);
void ivector_pool_reset(memory_context_t *ctx);

/* Bit vector pool */
bitvector_t* bitvector_pool_get(memory_context_t *ctx, idx_t length);
void bitvector_pool_reset(memory_context_t *ctx);

/* Matrix pool */
matrix_t* matrix_pool_get(memory_context_t *ctx, idx_t nrows, idx_t ncols);
void matrix_pool_reset(memory_context_t *ctx);

/* Low-level vector utilities */
vector_t* vector_create(idx_t length);
void vector_destroy(vector_t *v);
void vector_fill(vector_t *v, real_t value);
void vector_zero(vector_t *v);
real_t vector_sum(const vector_t *v);
real_t vector_mean(const vector_t *v);
real_t vector_sq_norm(const vector_t *v);

/* Low-level integer vector utilities */
ivector_t* ivector_create(idx_t length);
void ivector_destroy(ivector_t *v);
void ivector_zero(ivector_t *v);

/* Low-level bit vector utilities */
bitvector_t* bitvector_create(idx_t length);
void bitvector_destroy(bitvector_t *v);
void bitvector_set_all(bitvector_t *v, bool value);
bool bitvector_get(const bitvector_t *v, idx_t i);
void bitvector_set(bitvector_t *v, idx_t i, bool value);
idx_t bitvector_count_true(const bitvector_t *v);

/* Low-level matrix utilities */
matrix_t* matrix_create(idx_t nrows, idx_t ncols);
void matrix_destroy(matrix_t *m);
void matrix_zero(matrix_t *m);

#endif /* FASTHAL_MEMORY_POOL_H */
