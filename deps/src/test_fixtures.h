#ifndef FASTHAL_TEST_FIXTURES_H
#define FASTHAL_TEST_FIXTURES_H

#include "common.h"
#include "memory_pool.h"

/* Test fixtures for validating basis operations against known results */

typedef struct {
    real_t *X;           /* Data matrix (n x p), column-major */
    real_t *y;           /* Target vector (n) */
    real_t *y_binomial;  /* Binary target */
    idx_t n;
    idx_t p;
} test_dataset_t;

/* Create standard small test dataset (n=100, p=5) */
test_dataset_t* test_dataset_create_small(void);

void test_dataset_destroy(test_dataset_t *ds);

/* Comparison utilities for floating point */
bool doubles_approx_equal(real_t a, real_t b, real_t tol);

bool vectors_approx_equal(const real_t *v1, const real_t *v2, idx_t n, real_t tol);

bool ivectors_equal(const idx_t *v1, const idx_t *v2, idx_t n);

/* Print utilities for debugging */
void vector_print(const char *name, const real_t *v, idx_t n);

void ivector_print(const char *name, const idx_t *v, idx_t n);

void matrix_print(const char *name, const real_t *X, idx_t nrows, idx_t ncols);

#endif /* FASTHAL_TEST_FIXTURES_H */
