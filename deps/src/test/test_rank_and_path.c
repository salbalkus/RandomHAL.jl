#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../common.h"
#include "../rank_and_path.h"
#include "../test_fixtures.h"

/* ============================================================================
 * Test: Rank column - simple case
 * ============================================================================ */

static void test_rank_column_simple(void) {
    /* col = [3.0, 1.0, 2.0, 1.0, 4.0]
     * Expected ranks: [3, 1, 2, 1, 4] (competitive ranking)
     */
    real_t col[] = {3.0, 1.0, 2.0, 1.0, 4.0};
    idx_t n = 5;
    idx_t *ranks = (idx_t *)malloc((size_t)n * sizeof(idx_t));
    
    int status = rank_column(col, n, ranks);
    CU_ASSERT_EQUAL(status, FASTHAL_SUCCESS);
    
    /* Check sorted properties: identical values should have same rank */
    CU_ASSERT_EQUAL(ranks[1], ranks[3]);  /* Both 1.0 */
    
    free(ranks);
}

/* ============================================================================
 * Test: Rank all columns
 * ============================================================================ */

static void test_rank_all_columns(void) {
    idx_t n = 5;
    idx_t p = 2;
    
    /* X = [[1.0, 4.0],
     *      [2.0, 2.0],
     *      [3.0, 1.0],
     *      [4.0, 3.0],
     *      [5.0, 5.0]]
     * Column-major: [1, 2, 3, 4, 5, 4, 2, 1, 3, 5]
     */
    real_t X[] = {1.0, 2.0, 3.0, 4.0, 5.0,
                  4.0, 2.0, 1.0, 3.0, 5.0};
    
    idx_t *ranks = (idx_t *)malloc((size_t)(n * p) * sizeof(idx_t));
    
    int status = rank_all_columns(X, n, p, ranks);
    CU_ASSERT_EQUAL(status, FASTHAL_SUCCESS);
    
    /* Check that ranks are in range [1, n] */
    for (idx_t i = 0; i < n * p; i++) {
        CU_ASSERT_TRUE(ranks[i] >= 1 && ranks[i] <= n);
    }
    
    free(ranks);
}

/* ============================================================================
 * Test: Path sampling - simple nested case
 * ============================================================================ */

static void test_path_sample_simple(void) {
    /* n=5, section_size=1
     * ranks_section = [1, 2, 3, 4, 5] (already sorted)
     * Expected path: all indices (each dominates previous)
     */
    idx_t n = 5;
    idx_t section_size = 1;
    idx_t ranks_section[] = {1, 2, 3, 4, 5};
    
    idx_t *path = (idx_t *)malloc((size_t)n * sizeof(idx_t));
    idx_t path_len = 0;
    
    int status = path_sample(ranks_section, n, section_size, path, &path_len);
    CU_ASSERT_EQUAL(status, FASTHAL_SUCCESS);
    CU_ASSERT_TRUE(path_len > 0);
    CU_ASSERT_TRUE(path_len <= n);
    
    free(path);
}

/* ============================================================================
 * Test: Binary bin search - simple case
 * ============================================================================ */

static void test_binary_bin_search_simple(void) {
    idx_t n = 5;
    idx_t num_bins = 3;
    
    /* X = [0.1, 0.4, 0.6, 0.8, 0.9] (column-major, single column) */
    real_t X[] = {0.1, 0.4, 0.6, 0.8, 0.9};
    
    /* X_bins at knot indices 0, 2, 4 */
    real_t X_bins[] = {0.1, 0.6, 0.9};
    
    /* Path: [0, 2, 4] (knot indices) */
    idx_t path[] = {0, 2, 4};
    
    /* Section: [0] (first column) */
    idx_t section[] = {0};
    idx_t section_size = 1;
    
    idx_t *order = (idx_t *)malloc((size_t)n * sizeof(idx_t));
    
    int status = binary_bin_search(X, n, X_bins, path, section, section_size, num_bins, order);
    CU_ASSERT_EQUAL(status, FASTHAL_SUCCESS);
    
    /* Check that all bin assignments are in range [0, num_bins] */
    for (idx_t i = 0; i < n; i++) {
        CU_ASSERT_TRUE(order[i] >= 0 && order[i] <= num_bins);
    }
    
    free(order);
}

/* ============================================================================
 * Test: Rank consistency across large dataset
 * ============================================================================ */

static void test_rank_consistency_large(void) {
    idx_t n = 100;
    
    real_t *col = (real_t *)malloc((size_t)n * sizeof(real_t));
    idx_t *ranks = (idx_t *)malloc((size_t)n * sizeof(idx_t));
    
    /* Fill with random values */
    for (idx_t i = 0; i < n; i++) {
        col[i] = (real_t)i + 0.5 * (real_t)(i % 3);
    }
    
    int status = rank_column(col, n, ranks);
    CU_ASSERT_EQUAL(status, FASTHAL_SUCCESS);
    
    /* Check rank invariant: if col[i] < col[j], then ranks[i] < ranks[j] */
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = i + 1; j < MIN(n, i + 10); j++) {
            if (col[i] < col[j] - FTOL) {
                CU_ASSERT_TRUE(ranks[i] < ranks[j]);
            }
        }
    }
    
    free(col);
    free(ranks);
}

/* ============================================================================
 * Test Suite Setup
 * ============================================================================ */

int suite_rank_and_path(void) {
    CU_pSuite pSuite = NULL;
    
    pSuite = CU_add_suite("Rank and Path", NULL, NULL);
    if (NULL == pSuite) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    
    CU_add_test(pSuite, "Rank column simple", test_rank_column_simple);
    CU_add_test(pSuite, "Rank all columns", test_rank_all_columns);
    CU_add_test(pSuite, "Path sample simple", test_path_sample_simple);
    CU_add_test(pSuite, "Binary bin search simple", test_binary_bin_search_simple);
    CU_add_test(pSuite, "Rank consistency large", test_rank_consistency_large);
    
    return CU_get_error();
}
