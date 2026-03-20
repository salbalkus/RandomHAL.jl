#include "memory_pool.h"

/* ============================================================================
 * Low-level vector/matrix allocation utilities
 * ============================================================================ */

vector_t* vector_create(idx_t length) {
    if (length <= 0) return NULL;
    vector_t *v = (vector_t *)malloc(sizeof(vector_t));
    if (!v) return NULL;
    v->data = (real_t *)calloc((size_t)length, sizeof(real_t));
    if (!v->data) {
        free(v);
        return NULL;
    }
    v->length = length;
    v->capacity = length;
    return v;
}

void vector_destroy(vector_t *v) {
    if (!v) return;
    if (v->data) free(v->data);
    free(v);
}

void vector_fill(vector_t *v, real_t value) {
    if (!v || !v->data) return;
    for (idx_t i = 0; i < v->length; i++) {
        v->data[i] = value;
    }
}

void vector_zero(vector_t *v) {
    if (!v || !v->data) return;
    memset(v->data, 0, (size_t)v->length * sizeof(real_t));
}

real_t vector_sum(const vector_t *v) {
    if (!v || !v->data) return 0.0;
    real_t sum = 0.0;
    for (idx_t i = 0; i < v->length; i++) {
        sum += v->data[i];
    }
    return sum;
}

real_t vector_mean(const vector_t *v) {
    if (!v || !v->data || v->length == 0) return 0.0;
    return vector_sum(v) / (real_t)v->length;
}

real_t vector_sq_norm(const vector_t *v) {
    if (!v || !v->data) return 0.0;
    real_t norm = 0.0;
    for (idx_t i = 0; i < v->length; i++) {
        norm += v->data[i] * v->data[i];
    }
    return sqrt(norm);
}

/* ============================================================================
 * Integer vector utilities
 * ============================================================================ */

ivector_t* ivector_create(idx_t length) {
    if (length <= 0) return NULL;
    ivector_t *v = (ivector_t *)malloc(sizeof(ivector_t));
    if (!v) return NULL;
    v->data = (idx_t *)calloc((size_t)length, sizeof(idx_t));
    if (!v->data) {
        free(v);
        return NULL;
    }
    v->length = length;
    v->capacity = length;
    return v;
}

void ivector_destroy(ivector_t *v) {
    if (!v) return;
    if (v->data) free(v->data);
    free(v);
}

void ivector_zero(ivector_t *v) {
    if (!v || !v->data) return;
    memset(v->data, 0, (size_t)v->length * sizeof(idx_t));
}

/* ============================================================================
 * Bit vector utilities (packed booleans)
 * ============================================================================ */

bitvector_t* bitvector_create(idx_t length) {
    if (length <= 0) return NULL;
    bitvector_t *v = (bitvector_t *)malloc(sizeof(bitvector_t));
    if (!v) return NULL;
    idx_t num_bytes = (length + 7) / 8;
    v->data = (uint8_t *)calloc((size_t)num_bytes, sizeof(uint8_t));
    if (!v->data) {
        free(v);
        return NULL;
    }
    v->length = length;
    v->capacity = num_bytes;
    return v;
}

void bitvector_destroy(bitvector_t *v) {
    if (!v) return;
    if (v->data) free(v->data);
    free(v);
}

void bitvector_set_all(bitvector_t *v, bool value) {
    if (!v || !v->data) return;
    memset(v->data, value ? 0xFF : 0, (size_t)v->capacity);
}

bool bitvector_get(const bitvector_t *v, idx_t i) {
    if (!v || !v->data || i < 0 || i >= v->length) return false;
    idx_t byte_idx = i / 8;
    idx_t bit_idx = i % 8;
    return (v->data[byte_idx] & (1U << bit_idx)) != 0;
}

void bitvector_set(bitvector_t *v, idx_t i, bool value) {
    if (!v || !v->data || i < 0 || i >= v->length) return;
    idx_t byte_idx = i / 8;
    idx_t bit_idx = i % 8;
    if (value) {
        v->data[byte_idx] |= (1U << bit_idx);
    } else {
        v->data[byte_idx] &= ~(1U << bit_idx);
    }
}

idx_t bitvector_count_true(const bitvector_t *v) {
    if (!v || !v->data) return 0;
    idx_t count = 0;
    for (idx_t byte_idx = 0; byte_idx < v->capacity; byte_idx++) {
        uint8_t byte = v->data[byte_idx];
        while (byte) {
            count += byte & 1;
            byte >>= 1;
        }
    }
    return count;
}

/* ============================================================================
 * Matrix utilities (column-major)
 * ============================================================================ */

matrix_t* matrix_create(idx_t nrows, idx_t ncols) {
    if (nrows <= 0 || ncols <= 0) return NULL;
    matrix_t *m = (matrix_t *)malloc(sizeof(matrix_t));
    if (!m) return NULL;
    idx_t size = nrows * ncols;
    m->data = (real_t *)calloc((size_t)size, sizeof(real_t));
    if (!m->data) {
        free(m);
        return NULL;
    }
    m->nrows = nrows;
    m->ncols = ncols;
    m->capacity = size;
    return m;
}

void matrix_destroy(matrix_t *m) {
    if (!m) return;
    if (m->data) free(m->data);
    free(m);
}

void matrix_zero(matrix_t *m) {
    if (!m || !m->data) return;
    memset(m->data, 0, (size_t)(m->nrows * m->ncols) * sizeof(real_t));
}

/* ============================================================================
 * Memory pools for reuse
 * ============================================================================ */

/* Vector pool implementation */
vector_pool_t* vector_pool_create(void) {
    vector_pool_t *pool = (vector_pool_t *)malloc(sizeof(vector_pool_t));
    if (!pool) return NULL;
    pool->capacity = 10;
    pool->num_vectors = 0;
    pool->vectors = (vector_t **)malloc((size_t)pool->capacity * sizeof(vector_t *));
    if (!pool->vectors) {
        free(pool);
        return NULL;
    }
    return pool;
}

void vector_pool_destroy(vector_pool_t *pool) {
    if (!pool) return;
    for (idx_t i = 0; i < pool->num_vectors; i++) {
        vector_destroy(pool->vectors[i]);
    }
    free(pool->vectors);
    free(pool);
}

vector_t* vector_pool_get(memory_context_t *ctx, idx_t length) {
    if (!ctx) return vector_create(length);
    vector_pool_t *pool = &ctx->vector_pool;
    
    /* Check if any existing vector can be reused */
    for (idx_t i = 0; i < pool->num_vectors; i++) {
        if (pool->vectors[i]->capacity >= length) {
            pool->vectors[i]->length = length;
            vector_zero(pool->vectors[i]);
            return pool->vectors[i];
        }
    }
    
    /* Create new vector and add to pool */
    vector_t *v = vector_create(length);
    if (!v) return NULL;
    
    if (pool->num_vectors >= pool->capacity) {
        pool->capacity *= 2;
        vector_t **new_vectors = (vector_t **)realloc(pool->vectors,
                                                       (size_t)pool->capacity * sizeof(vector_t *));
        if (!new_vectors) {
            vector_destroy(v);
            return NULL;
        }
        pool->vectors = new_vectors;
    }
    
    pool->vectors[pool->num_vectors++] = v;
    return v;
}

void vector_pool_reset(memory_context_t *ctx) {
    if (!ctx) return;
    vector_pool_t *pool = &ctx->vector_pool;
    for (idx_t i = 0; i < pool->num_vectors; i++) {
        vector_zero(pool->vectors[i]);
    }
}

/* Integer vector pool */
ivector_pool_t* ivector_pool_create(void) {
    ivector_pool_t *pool = (ivector_pool_t *)malloc(sizeof(ivector_pool_t));
    if (!pool) return NULL;
    pool->capacity = 10;
    pool->num_vectors = 0;
    pool->ivectors = (ivector_t **)malloc((size_t)pool->capacity * sizeof(ivector_t *));
    if (!pool->ivectors) {
        free(pool);
        return NULL;
    }
    return pool;
}

void ivector_pool_destroy(ivector_pool_t *pool) {
    if (!pool) return;
    for (idx_t i = 0; i < pool->num_vectors; i++) {
        ivector_destroy(pool->ivectors[i]);
    }
    free(pool->ivectors);
    free(pool);
}

ivector_t* ivector_pool_get(memory_context_t *ctx, idx_t length) {
    if (!ctx) return ivector_create(length);
    ivector_pool_t *pool = &ctx->ivector_pool;
    
    for (idx_t i = 0; i < pool->num_vectors; i++) {
        if (pool->ivectors[i]->capacity >= length) {
            pool->ivectors[i]->length = length;
            ivector_zero(pool->ivectors[i]);
            return pool->ivectors[i];
        }
    }
    
    ivector_t *v = ivector_create(length);
    if (!v) return NULL;
    
    if (pool->num_vectors >= pool->capacity) {
        pool->capacity *= 2;
        ivector_t **new_vectors = (ivector_t **)realloc(pool->ivectors,
                                                        (size_t)pool->capacity * sizeof(ivector_t *));
        if (!new_vectors) {
            ivector_destroy(v);
            return NULL;
        }
        pool->ivectors = new_vectors;
    }
    
    pool->ivectors[pool->num_vectors++] = v;
    return v;
}

void ivector_pool_reset(memory_context_t *ctx) {
    if (!ctx) return;
    ivector_pool_t *pool = &ctx->ivector_pool;
    for (idx_t i = 0; i < pool->num_vectors; i++) {
        ivector_zero(pool->ivectors[i]);
    }
}

/* Bit vector pool */
bitvector_pool_t* bitvector_pool_create(void) {
    bitvector_pool_t *pool = (bitvector_pool_t *)malloc(sizeof(bitvector_pool_t));
    if (!pool) return NULL;
    pool->capacity = 10;
    pool->num_vectors = 0;
    pool->bitvectors = (bitvector_t **)malloc((size_t)pool->capacity * sizeof(bitvector_t *));
    if (!pool->bitvectors) {
        free(pool);
        return NULL;
    }
    return pool;
}

void bitvector_pool_destroy(bitvector_pool_t *pool) {
    if (!pool) return;
    for (idx_t i = 0; i < pool->num_vectors; i++) {
        bitvector_destroy(pool->bitvectors[i]);
    }
    free(pool->bitvectors);
    free(pool);
}

bitvector_t* bitvector_pool_get(memory_context_t *ctx, idx_t length) {
    if (!ctx) return bitvector_create(length);
    bitvector_pool_t *pool = &ctx->bitvector_pool;
    
    for (idx_t i = 0; i < pool->num_vectors; i++) {
        if (pool->bitvectors[i]->length >= length) {
            bitvector_set_all(pool->bitvectors[i], false);
            return pool->bitvectors[i];
        }
    }
    
    bitvector_t *v = bitvector_create(length);
    if (!v) return NULL;
    
    if (pool->num_vectors >= pool->capacity) {
        pool->capacity *= 2;
        bitvector_t **new_vectors = (bitvector_t **)realloc(pool->bitvectors,
                                                            (size_t)pool->capacity * sizeof(bitvector_t *));
        if (!new_vectors) {
            bitvector_destroy(v);
            return NULL;
        }
        pool->bitvectors = new_vectors;
    }
    
    pool->bitvectors[pool->num_vectors++] = v;
    return v;
}

void bitvector_pool_reset(memory_context_t *ctx) {
    if (!ctx) return;
    bitvector_pool_t *pool = &ctx->bitvector_pool;
    for (idx_t i = 0; i < pool->num_vectors; i++) {
        bitvector_set_all(pool->bitvectors[i], false);
    }
}

/* Matrix pool */
matrix_pool_t* matrix_pool_create(void) {
    matrix_pool_t *pool = (matrix_pool_t *)malloc(sizeof(matrix_pool_t));
    if (!pool) return NULL;
    pool->capacity = 10;
    pool->num_matrices = 0;
    pool->matrices = (matrix_t **)malloc((size_t)pool->capacity * sizeof(matrix_t *));
    if (!pool->matrices) {
        free(pool);
        return NULL;
    }
    return pool;
}

void matrix_pool_destroy(matrix_pool_t *pool) {
    if (!pool) return;
    for (idx_t i = 0; i < pool->num_matrices; i++) {
        matrix_destroy(pool->matrices[i]);
    }
    free(pool->matrices);
    free(pool);
}

matrix_t* matrix_pool_get(memory_context_t *ctx, idx_t nrows, idx_t ncols) {
    if (!ctx) return matrix_create(nrows, ncols);
    matrix_pool_t *pool = &ctx->matrix_pool;
    
    for (idx_t i = 0; i < pool->num_matrices; i++) {
        matrix_t *m = pool->matrices[i];
        if (m->nrows >= nrows && m->ncols >= ncols) {
            m->nrows = nrows;
            m->ncols = ncols;
            matrix_zero(m);
            return m;
        }
    }
    
    matrix_t *m = matrix_create(nrows, ncols);
    if (!m) return NULL;
    
    if (pool->num_matrices >= pool->capacity) {
        pool->capacity *= 2;
        matrix_t **new_matrices = (matrix_t **)realloc(pool->matrices,
                                                       (size_t)pool->capacity * sizeof(matrix_t *));
        if (!new_matrices) {
            matrix_destroy(m);
            return NULL;
        }
        pool->matrices = new_matrices;
    }
    
    pool->matrices[pool->num_matrices++] = m;
    return m;
}

void matrix_pool_reset(memory_context_t *ctx) {
    if (!ctx) return;
    matrix_pool_t *pool = &ctx->matrix_pool;
    for (idx_t i = 0; i < pool->num_matrices; i++) {
        matrix_zero(pool->matrices[i]);
    }
}

/* ============================================================================
 * Context management
 * ============================================================================ */

memory_context_t* memory_context_create(void) {
    memory_context_t *ctx = (memory_context_t *)malloc(sizeof(memory_context_t));
    if (!ctx) return NULL;
    
    ctx->vector_pool = *(vector_pool_create());
    if (!ctx->vector_pool.vectors) {
        free(ctx);
        return NULL;
    }
    
    ctx->ivector_pool = *(ivector_pool_create());
    if (!ctx->ivector_pool.ivectors) {
        vector_pool_destroy(&ctx->vector_pool);
        free(ctx);
        return NULL;
    }
    
    ctx->bitvector_pool = *(bitvector_pool_create());
    if (!ctx->bitvector_pool.bitvectors) {
        vector_pool_destroy(&ctx->vector_pool);
        ivector_pool_destroy(&ctx->ivector_pool);
        free(ctx);
        return NULL;
    }
    
    ctx->matrix_pool = *(matrix_pool_create());
    if (!ctx->matrix_pool.matrices) {
        vector_pool_destroy(&ctx->vector_pool);
        ivector_pool_destroy(&ctx->ivector_pool);
        bitvector_pool_destroy(&ctx->bitvector_pool);
        free(ctx);
        return NULL;
    }
    
    return ctx;
}

void memory_context_destroy(memory_context_t *ctx) {
    if (!ctx) return;
    vector_pool_destroy(&ctx->vector_pool);
    ivector_pool_destroy(&ctx->ivector_pool);
    bitvector_pool_destroy(&ctx->bitvector_pool);
    matrix_pool_destroy(&ctx->matrix_pool);
    free(ctx);
}
