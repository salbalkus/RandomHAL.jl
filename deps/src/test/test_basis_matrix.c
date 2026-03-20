#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../common.h"
#include "../memory_pool.h"
#include "../basis_matrix.h"
#include "../rank_and_path.h"
#include "../test_fixtures.h"

/* ============================================================================
 * Test: Basis matrix creation and destruction
 * ============================================================================ */

static void test_basis_matrix_create_destroy(void) {
    test_dataset_t *ds = test_dataset_create_small();
    CU_ASSERT_PTR_NOT_NULL(ds);
    
    /* Compute ranks for the data */
    idx_t *ranks_matrix = (idx_t *)malloc((size_t)(ds->n * ds->p) * sizeof(idx_t));
    CU_ASSERT_PTR_NOT_NULL(ranks_matrix);
    
    int status = rank_all_columns(ds->X, ds->n, ds->p, ranks_matrix);
    CU_ASSERT_EQUAL(status, FASTHAL_SUCCESS);
    
    /* Create a basis matrix for the first feature */
    idx_t section[] = {0};
    int smoothness = 0;
    
    basis_matrix_t *b = basis_matrix_create(ds->X, ds->n, ds->p,
                                            section, 1, smoothness, ranks_matrix);
    CU_ASSERT_PTR_NOT_NULL(b);
    CU_ASSERT_EQUAL(b->nrow, ds->n);
    CU_ASSERT_TRUE(b->ncol > 0);  /* Should have at least one basis function */
    
    basis_matrix_destroy(b);
    free(ranks_matrix);
    test_dataset_destroy(ds);
}

/* ============================================================================
 * Test: Basis matrix multiplication dimensions
 * ============================================================================ */

static void test_basis_matrix_mul_dimensions(void) {
    test_dataset_t *ds = test_dataset_create_small();
    CU_ASSERT_PTR_NOT_NULL(ds);
    
    idx_t *ranks_matrix = (idx_t *)malloc((size_t)(ds->n * ds->p) * sizeof(idx_t));
    int status = rank_all_columns(ds->X, ds->n, ds->p, ranks_matrix);
    CU_ASSERT_EQUAL(status, FASTHAL_SUCCESS);
    
    idx_t section[] = {0};
    basis_matrix_t *b = basis_matrix_create(ds->X, ds->n, ds->p,
                                            section, 1, 0, ranks_matrix);
    CU_ASSERT_PTR_NOT_NULL(b);
    
    /* Create coefficient vector */
    real_t *v = (real_t *)malloc((size_t)b->ncol * sizeof(real_t));
    for (idx_t j = 0; j < b->ncol; j++) {
        v[j] = 1.0;
    }
    
    /* Multiply */
    real_t *y = (real_t *)malloc((size_t)b->nrow * sizeof(real_t));
    status = basis_matrix_mul(b, v, y);
    CU_ASSERT_EQUAL(status, FASTHAL_SUCCESS);
    
    /* Check output dimensions */
    for (idx_t i = 0; i < b->nrow; i++) {
        CU_ASSERT_TRUE(isfinite(y[i]));  /* Should be finite */
    }
    
    basis_matrix_destroy(b);
    free(y);
    free(v);
    free(ranks_matrix);
    test_dataset_destroy(ds);
}

/* ============================================================================
 * Test: Basis matrix transpose consistency
 * ============================================================================ */

static void test_basis_matrix_transpose_consistency(void) {
    test_dataset_t *ds = test_dataset_create_small();
    CU_ASSERT_PTR_NOT_NULL(ds);
    
    idx_t *ranks_matrix = (idx_t *)malloc((size_t)(ds->n * ds->p) * sizeof(idx_t));
    int status = rank_all_columns(ds->X, ds->n, ds->p, ranks_matrix);
    CU_ASSERT_EQUAL(status, FASTHAL_SUCCESS);
    
    idx_t section[] = {0};
    basis_matrix_t *b = basis_matrix_create(ds->X, ds->n, ds->p,
                                            section, 1, 0, ranks_matrix);
    CU_ASSERT_PTR_NOT_NULL(b);
    
    basis_matrix_transpose_t *bt = basis_matrix_transpose(b);
    CU_ASSERT_PTR_NOT_NULL(bt);
    
    /* Check dimensions are swapped */
    CU_ASSERT_EQUAL(bt->nrow, b->ncol);
    CU_ASSERT_EQUAL(bt->ncol, b->nrow);
    
    basis_matrix_transpose_destroy(bt);
    basis_matrix_destroy(b);
    free(ranks_matrix);
    test_dataset_destroy(ds);
}

/* ============================================================================
 * Test: Basis matrix multiplication with smoothness
 * ============================================================================ */

static void test_basis_matrix_with_smoothness(void) {
    test_dataset_t *ds = test_dataset_create_small();
    CU_ASSERT_PTR_NOT_NULL(ds);
    
    idx_t *ranks_matrix = (idx_t *)malloc((size_t)(ds->n * ds->p) * sizeof(idx_t));
    int status = rank_all_columns(ds->X, ds->n, ds->p, ranks_matrix);
    CU_ASSERT_EQUAL(status, FASTHAL_SUCCESS);
    
    /* Test with smoothness = 1 */
    idx_t section[] = {0};
    basis_matrix_t *b = basis_matrix_create(ds->X, ds->n, ds->p,
                                            section, 1, 1, ranks_matrix);
    CU_ASSERT_PTR_NOT_NULL(b);
    CU_ASSERT_EQUAL(b->smoothness, 1);
    
    /* l vector should be non-zero (product of features to power 1) */
    for (idx_t i = 0; i < MIN(ds->n, 10); i++) {
        CU_ASSERT_TRUE(b->l[i] > 0);
    }
    
    basis_matrix_destroy(b);
    free(ranks_matrix);
    test_dataset_destroy(ds);
}

/* ============================================================================
 * Test: Multi-dimensional basis
 * ============================================================================ */

static void test_basis_matrix_multidim(void) {
    test_dataset_t *ds = test_dataset_create_small();
    CU_ASSERT_PTR_NOT_NULL(ds);
    
    idx_t *ranks_matrix = (idx_t *)malloc((size_t)(ds->n * ds->p) * sizeof(idx_t));
    int status = rank_all_columns(ds->X, ds->n, ds->p, ranks_matrix);
    CU_ASSERT_EQUAL(status, FASTHAL_SUCCESS);
    
    /* Two-dimensional basis */
    idx_t section[] = {0, 1};
    basis_matrix_t *b = basis_matrix_create(ds->X, ds->n, ds->p,
                                            section, 2, 0, ranks_matrix);
    CU_ASSERT_PTR_NOT_NULL(b);
    CU_ASSERT_EQUAL(b->section_size, 2);
    
    /* Should have fewer basis functions than single dimension 
     * (due to nesting requirement) */
    CU_ASSERT_TRUE(b->ncol > 0);
    
    basis_matrix_destroy(b);
    free(ranks_matrix);
    test_dataset_destroy(ds);
}

/* ============================================================================
 * Test: Numerical stability (no NaN/Inf)
 * ============================================================================ */

static void test_basis_matrix_numerical_stability(void) {
    test_dataset_t *ds = test_dataset_create_small();
    CU_ASSERT_PTR_NOT_NULL(ds);
    
    idx_t *ranks_matrix = (idx_t *)malloc((size_t)(ds->n * ds->p) * sizeof(idx_t));
    int status = rank_all_columns(ds->X, ds->n, ds->p, ranks_matrix);
    CU_ASSERT_EQUAL(status, FASTHAL_SUCCESS);
    
    idx_t section[] = {0};
    basis_matrix_t *b = basis_matrix_create(ds->X, ds->n, ds->p,
                                            section, 1, 0, ranks_matrix);
    CU_ASSERT_PTR_NOT_NULL(b);
    
    /* Check l and r vectors are finite */
    for (idx_t i = 0; i < b->nrow; i++) {
        CU_ASSERT_TRUE(isfinite(b->l[i]));
    }
    for (idx_t j = 0; j < b->ncol; j++) {
        CU_ASSERT_TRUE(isfinite(b->r[j]));
    }
    
    /* Multiply and check output */
    real_t *v = (real_t *)malloc((size_t)b->ncol * sizeof(real_t));
    for (idx_t j = 0; j < b->ncol; j++) {
        v[j] = 1.0 / (real_t)(j + 1);
    }
    
    real_t *y = (real_t *)malloc((size_t)b->nrow * sizeof(real_t));
    status = basis_matrix_mul(b, v, y);
    CU_ASSERT_EQUAL(status, FASTHAL_SUCCESS);
    
    for (idx_t i = 0; i < b->nrow; i++) {
        CU_ASSERT_TRUE(isfinite(y[i]));
    }
    
    basis_matrix_destroy(b);
    free(y);
    free(v);
    free(ranks_matrix);
    test_dataset_destroy(ds);
}

/* ============================================================================
 * Test Suite Setup
 * ============================================================================ */

int suite_basis_matrix(void) {
    CU_pSuite pSuite = NULL;
    
    pSuite = CU_add_suite("Basis Matrix", NULL, NULL);
    if (NULL == pSuite) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    
    CU_add_test(pSuite, "Create and destroy", test_basis_matrix_create_destroy);
    CU_add_test(pSuite, "Multiplication dimensions", test_basis_matrix_mul_dimensions);
    CU_add_test(pSuite, "Transpose consistency", test_basis_matrix_transpose_consistency);
    CU_add_test(pSuite, "With smoothness", test_basis_matrix_with_smoothness);
    CU_add_test(pSuite, "Multi-dimensional", test_basis_matrix_multidim);
    CU_add_test(pSuite, "Numerical stability", test_basis_matrix_numerical_stability);
    
    return CU_get_error();
}
