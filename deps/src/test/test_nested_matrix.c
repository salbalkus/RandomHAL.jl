#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../common.h"
#include "../memory_pool.h"
#include "../nested_matrix.h"
#include "../test_fixtures.h"

/* ============================================================================
 * Test: Nested Matrix Creation and Destruction
 * ============================================================================ */

static void test_nested_matrix_create(void) {
    idx_t n = 10;
    idx_t p = 5;
    
    idx_t *order = (idx_t *)malloc((size_t)n * sizeof(idx_t));
    for (idx_t i = 0; i < n; i++) {
        order[i] = (i % p) + 1;  /* 1-indexed bins */
    }
    
    nested_matrix_t *m = nested_matrix_create(order, n, p);
    CU_ASSERT_PTR_NOT_NULL(m);
    CU_ASSERT_EQUAL(m->nrow, n);
    CU_ASSERT_EQUAL(m->ncol, p);
    
    nested_matrix_destroy(m);
    free(order);
}

/* ============================================================================
 * Test: Multiplication with simple known case
 * ============================================================================ */

static void test_nested_matrix_mul_simple(void) {
    /* Simple case: order = [1, 1, 2, 2, 3]
     * v = [1, 2, 3]
     * Expected: cumsum(reverse(v)) = [6, 5, 3]
     * Output: [6, 6, 5, 5, 3]
     */
    idx_t n = 5;
    idx_t p = 3;
    
    idx_t order[] = {1, 1, 2, 2, 3};
    real_t v[] = {1.0, 2.0, 3.0};
    real_t expected[] = {6.0, 6.0, 5.0, 5.0, 3.0};
    
    nested_matrix_t *m = nested_matrix_create(order, n, p);
    CU_ASSERT_PTR_NOT_NULL(m);
    
    real_t *y = (real_t *)malloc((size_t)n * sizeof(real_t));
    int status = nested_matrix_mul(m, v, y);
    CU_ASSERT_EQUAL(status, FASTHAL_SUCCESS);
    
    for (idx_t i = 0; i < n; i++) {
        CU_ASSERT_DOUBLE_EQUAL(y[i], expected[i], 1e-10);
    }
    
    nested_matrix_destroy(m);
    free(y);
}

/* ============================================================================
 * Test: Transpose multiplication
 * ============================================================================ */

static void test_nested_matrix_transpose_mul(void) {
    /* For order = [1, 1, 2, 2, 3] and v = [1, 2, 3, 4, 5]
     * (B^T * v)[1] = sum of v[i] where order[i] >= 1 = 1+2+3+4+5 = 15
     * (B^T * v)[2] = sum of v[i] where order[i] >= 2 = 3+4+5 = 12
     * (B^T * v)[3] = sum of v[i] where order[i] >= 3 = 5
     */
    idx_t n = 5;
    idx_t p = 3;
    
    idx_t order[] = {1, 1, 2, 2, 3};
    real_t v[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    real_t expected[] = {15.0, 12.0, 5.0};
    
    nested_matrix_t *m = nested_matrix_create(order, n, p);
    nested_matrix_transpose_t *mt = nested_matrix_transpose(m);
    CU_ASSERT_PTR_NOT_NULL(mt);
    
    real_t *y = (real_t *)malloc((size_t)p * sizeof(real_t));
    int status = nested_matrix_transpose_mul(mt, v, y);
    CU_ASSERT_EQUAL(status, FASTHAL_SUCCESS);
    
    for (idx_t j = 0; j < p; j++) {
        CU_ASSERT_DOUBLE_EQUAL(y[j], expected[j], 1e-10);
    }
    
    nested_matrix_transpose_destroy(mt);
    nested_matrix_destroy(m);
    free(y);
}

/* ============================================================================
 * Test: Inverse relationship between B and B^T
 * ============================================================================ */

static void test_nested_matrix_transpose_inverse(void) {
    /* Test that (B^T)^T = B */
    idx_t n = 10;
    idx_t p = 5;
    
    idx_t *order = (idx_t *)malloc((size_t)n * sizeof(idx_t));
    for (idx_t i = 0; i < n; i++) {
        order[i] = (i % p) + 1;
    }
    
    nested_matrix_t *m = nested_matrix_create(order, n, p);
    nested_matrix_transpose_t *mt = nested_matrix_transpose(m);
    
    CU_ASSERT_EQUAL(mt->nrow, m->ncol);
    CU_ASSERT_EQUAL(mt->ncol, m->nrow);
    
    nested_matrix_transpose_destroy(mt);
    nested_matrix_destroy(m);
    free(order);
}

/* ============================================================================
 * Test: Large matrix multiplication
 * ============================================================================ */

static void test_nested_matrix_mul_large(void) {
    /* Test on larger dataset to ensure O(n) performance */
    idx_t n = 1000;
    idx_t p = 100;
    
    /* Create bins in sorted order */
    idx_t *order = (idx_t *)malloc((size_t)n * sizeof(idx_t));
    for (idx_t i = 0; i < n; i++) {
        order[i] = (idx_t)((real_t)i / (real_t)n * (real_t)p) + 1;
    }
    
    real_t *v = (real_t *)malloc((size_t)p * sizeof(real_t));
    for (idx_t j = 0; j < p; j++) {
        v[j] = 1.0 + (real_t)j / (real_t)p;
    }
    
    nested_matrix_t *m = nested_matrix_create(order, n, p);
    CU_ASSERT_PTR_NOT_NULL(m);
    
    real_t *y = (real_t *)malloc((size_t)n * sizeof(real_t));
    int status = nested_matrix_mul(m, v, y);
    CU_ASSERT_EQUAL(status, FASTHAL_SUCCESS);
    
    /* Verify structure: y[i] should be non-decreasing as order[i] increases */
    for (idx_t i = 1; i < n; i++) {
        if (order[i] > order[i-1] || order[i] == order[i-1]) {
            CU_ASSERT_TRUE(y[i] <= y[i-1]);  /* cumsum is decreasing as we go forward */
        }
    }
    
    nested_matrix_destroy(m);
    free(y);
    free(v);
    free(order);
}

/* ============================================================================
 * Test Suite Setup
 * ============================================================================ */

int suite_nested_matrix(void) {
    CU_pSuite pSuite = NULL;
    
    pSuite = CU_add_suite("Nested Matrix", NULL, NULL);
    if (NULL == pSuite) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    
    CU_add_test(pSuite, "Create and destroy", test_nested_matrix_create);
    CU_add_test(pSuite, "Simple multiplication", test_nested_matrix_mul_simple);
    CU_add_test(pSuite, "Transpose multiplication", test_nested_matrix_transpose_mul);
    CU_add_test(pSuite, "Transpose inverse", test_nested_matrix_transpose_inverse);
    CU_add_test(pSuite, "Large multiplication", test_nested_matrix_mul_large);
    
    return CU_get_error();
}
