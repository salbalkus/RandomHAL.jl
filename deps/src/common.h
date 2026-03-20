#ifndef FASTHAL_COMMON_H
#define FASTHAL_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <float.h>

/* Error handling macros */
#define FASTHAL_SUCCESS 0
#define FASTHAL_ERROR_INVALID_ARGS -1
#define FASTHAL_ERROR_ALLOCATION -2
#define FASTHAL_ERROR_DIMENSION_MISMATCH -3
#define FASTHAL_ERROR_NUMERICAL -4

/* Floating point type */
typedef double real_t;
#define REAL_EPSILON DBL_EPSILON
#define REAL_MAX DBL_MAX

/* Integer type for indices */
typedef int32_t idx_t;

/* Comparison tolerance for floating point */
#define FTOL 1e-10

/* Inline min/max macros */
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/* Matrix layout: column-major (Fortran-style for BLAS compatibility) */
#define MAT_IDX(i, j, nrows) ((j) * (nrows) + (i))

/* Structure to encapsulate a dense matrix */
typedef struct {
    real_t *data;  /* Column-major data array */
    idx_t nrows;
    idx_t ncols;
    idx_t capacity;  /* For memory pooling */
} matrix_t;

/* Structure for a vector */
typedef struct {
    real_t *data;
    idx_t length;
    idx_t capacity;  /* For memory pooling */
} vector_t;

/* Structure for an integer vector */
typedef struct {
    idx_t *data;
    idx_t length;
    idx_t capacity;  /* For memory pooling */
} ivector_t;

/* Structure for a bit vector (packed boolean) */
typedef struct {
    uint8_t *data;  /* Each byte holds 8 bools */
    idx_t length;
    idx_t capacity;
} bitvector_t;

/* Allocation error handler - returns non-zero on failure */
typedef int (*alloc_error_handler_t)(const char *msg);

/* Soft thresholding function: sign(z) * max(0, |z| - lambda) */
static inline real_t soft_threshold(real_t z, real_t lambda) {
    if (z > lambda) return z - lambda;
    if (z < -lambda) return z + lambda;
    return 0.0;
}

/* Sigmoid/expit function: 1 / (1 + exp(-x)) */
static inline real_t expit(real_t x) {
    if (x > 500.0) return 1.0;
    if (x < -500.0) return 0.0;
    return 1.0 / (1.0 + exp(-x));
}

#endif /* FASTHAL_COMMON_H */
