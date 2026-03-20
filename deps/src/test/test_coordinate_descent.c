#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "../coordinate_descent.h"
#include "../common.h"

/* Test 1: Parameters default */
void test_defaults(void) {
    cd_gaussian_params_t params = coord_descent_gaussian_params_default();
    CU_ASSERT_PTR_NULL(params.mem_ctx);
    CU_ASSERT(params.tolerance > 0);
}

/* Test 2: Soft threshold */
void test_soft_thresh(void) {
    CU_ASSERT_DOUBLE_EQUAL(soft_threshold(5.0, 2.0), 3.0, 1e-10);
    CU_ASSERT_DOUBLE_EQUAL(soft_threshold(-5.0, 2.0), -3.0, 1e-10);
    CU_ASSERT_DOUBLE_EQUAL(soft_threshold(0.0, 2.0), 0.0, 1e-10);
}

/* Test 3: Suite registration */
int suite_coordinate_descent(void) {
    CU_pSuite pSuite = CU_add_suite("Coordinate Descent", NULL, NULL);
    if (NULL == pSuite) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    
    CU_add_test(pSuite, "Defaults", test_defaults);
    CU_add_test(pSuite, "Soft threshold", test_soft_thresh);
    
    return CU_get_error();
}
