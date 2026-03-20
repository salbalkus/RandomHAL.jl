#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdio.h>
#include <stdlib.h>

/* Forward declarations of test suite functions */
extern int suite_nested_matrix(void);
extern int suite_rank_and_path(void);
extern int suite_basis_matrix(void);

int main(int argc, char *argv[]) {
    /* Initialize CUnit test registry */
    if (CUE_SUCCESS != CU_initialize_registry())
        return CU_get_error();
    
    /* Add test suites */
    if (CUE_SUCCESS != suite_nested_matrix())
        goto cleanup;
    
    if (CUE_SUCCESS != suite_rank_and_path())
        goto cleanup;
    
    if (CUE_SUCCESS != suite_basis_matrix())
        goto cleanup;
    
    /* Run tests */
    if (argc > 1 && argv[1][0] == '-' && argv[1][1] == 'v') {
        /* Verbose mode */
        CU_basic_set_mode(CU_BRM_VERBOSE);
    } else {
        /* Normal mode */
        CU_basic_set_mode(CU_BRM_NORMAL);
    }
    
    CU_basic_run_tests();
    
cleanup:
    CU_cleanup_registry();
    return CU_get_error();
}
