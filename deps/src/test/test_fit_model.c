#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "../fit_model.h"

/* Simple test utilities */
#define TEST_TOLERANCE 1e-6
#define ASSERT_NEAR(a, b, tol) do { \
    double diff = fabs((a) - (b)); \
    if (diff > tol) { \
        printf("  FAIL: %.6f != %.6f (diff: %.6e)\n", (a), (b), diff); \
        return 0; \
    } \
} while(0)

#define ASSERT_NEAR_ARRAY(arr1, arr2, n, tol) do { \
    for (idx_t i = 0; i < n; i++) { \
        double diff = fabs((arr1)[i] - (arr2)[i]); \
        if (diff > tol) { \
            printf("  FAIL: arr[%ld]: %.6f != %.6f (diff: %.6e)\n", i, (arr1)[i], (arr2)[i], diff); \
            return 0; \
        } \
    } \
} while(0)

typedef int (*test_func_t)(void);

/* ============================================================================
 * Test 1: Configuration creation and destruction
 * ============================================================================ */

int test_basis_config_creation(void) {
    printf("Test 1: Basis config creation\n");
    
    basis_config_t *cfg = basis_config_create(2);
    assert(cfg != NULL);
    assert(cfg->num_sections == 2);
    
    /* Add sections */
    idx_t section_0[] = {0, 1};
    idx_t section_1[] = {2, 3, 4};
    
    basis_config_add_section(cfg, 0, section_0, 2);
    basis_config_add_section(cfg, 1, section_1, 3);
    
    assert(cfg->section_sizes[0] == 2);
    assert(cfg->section_sizes[1] == 3);
    assert(cfg->sections[0][0] == 0);
    assert(cfg->sections[1][2] == 4);
    
    basis_config_destroy(cfg);
    printf("  PASS\n");
    return 1;
}

/* ============================================================================
 * Test 2: Fit config creation
 * ============================================================================ */

int test_fit_config_creation(void) {
    printf("Test 2: Fit config creation\n");
    
    fit_config_t *cfg = fit_config_create();
    assert(cfg != NULL);
    assert(cfg->K == 5);
    assert(cfg->alpha == 1.0);  /* LASSO */
    assert(cfg->n_lambda == 100);
    assert(cfg->auto_lambda == true);
    
    fit_config_destroy(cfg);
    printf("  PASS\n");
    return 1;
}

/* ============================================================================
 * Test 3: Model creation and destruction
 * ============================================================================ */

int test_model_creation(void) {
    printf("Test 3: Model creation\n");
    
    idx_t d = 10;
    idx_t n_lambda = 50;
    
    model_t *model = model_create(d, n_lambda);
    assert(model != NULL);
    assert(model->β != NULL);
    assert(model->β_path != NULL);
    assert(model->β0_path != NULL);
    assert(model->λ != NULL);
    assert(model->n_lambda == n_lambda);
    
    model_destroy(model);
    printf("  PASS\n");
    return 1;
}

/* ============================================================================
 * Test 4: Simple univariate Gaussian regression
 * ============================================================================ */

int test_fit_univariate_gaussian(void) {
    printf("Test 4: Fit univariate Gaussian\n");
    
    /* Create simple data: y = 2*X + noise */
    idx_t n = 50;
    idx_t p = 1;
    
    real_t *X = (real_t *)malloc((size_t)(n * p) * sizeof(real_t));
    real_t *y = (real_t *)malloc((size_t)n * sizeof(real_t));
    
    /* Generate data */
    for (idx_t i = 0; i < n; i++) {
        X[i] = (real_t)i / 50.0;  /* X from 0 to 1 */
        y[i] = 2.0 * X[i] + 0.01 * (i % 5) / 5.0;  /* y = 2*X + small noise */
    }
    
    /* Create basis and fit configs */
    basis_config_t *basis_cfg = basis_config_create(1);
    basis_config_add_section(basis_cfg, 0, (idx_t[]){0}, 1);
    
    fit_config_t *fit_cfg = fit_config_create();
    fit_cfg->n_lambda = 5;  /* Just a few lambdas for testing */
    
    /* Fit model */
    model_t *model = NULL;
    int status = fit_gaussian(X, n, p, y, basis_cfg, fit_cfg, &model);
    
    assert(status == FASTHAL_SUCCESS);
    assert(model != NULL);
    assert(model->n_lambda == 5);
    
    /* Check that intercept is roughly the mean of y */
    real_t y_mean = 0.0;
    for (idx_t i = 0; i < n; i++) y_mean += y[i];
    y_mean /= n;
    
    printf("  y_mean: %.4f, fitted β0: %.4f\n", y_mean, model->β0);
    ASSERT_NEAR(model->β0, y_mean, 0.2);  /* Loose tolerance for unregularized */
    
    /* Cleanup */
    model_destroy(model);
    basis_config_destroy(basis_cfg);
    fit_config_destroy(fit_cfg);
    free(X);
    free(y);
    
    printf("  PASS\n");
    return 1;
}

/* ============================================================================
 * Test 5: Multivariate Gaussian regression
 * ============================================================================ */

int test_fit_multivariate_gaussian(void) {
    printf("Test 5: Fit multivariate Gaussian\n");
    
    /* Create data: y = 1.5*X1 - 0.8*X2 + small noise */
    idx_t n = 100;
    idx_t p = 2;
    
    real_t *X = (real_t *)malloc((size_t)(n * p) * sizeof(real_t));
    real_t *y = (real_t *)malloc((size_t)n * sizeof(real_t));
    
    /* Generate data */
    for (idx_t i = 0; i < n; i++) {
        X[i + 0 * n] = (real_t)i / 100.0;
        X[i + 1 * n] = (real_t)(i % 50) / 50.0;
        y[i] = 1.5 * X[i + 0 * n] - 0.8 * X[i + 1 * n] + 0.01;
    }
    
    /* Configs */
    basis_config_t *basis_cfg = basis_config_create(1);
    idx_t cols[] = {0, 1};
    basis_config_add_section(basis_cfg, 0, cols, 2);
    
    fit_config_t *fit_cfg = fit_config_create();
    fit_cfg->n_lambda = 3;
    
    /* Fit */
    model_t *model = NULL;
    int status = fit_gaussian(X, n, p, y, basis_cfg, fit_cfg, &model);
    
    assert(status == FASTHAL_SUCCESS);
    assert(model != NULL);
    assert(model->β != NULL);
    
    printf("  Fitted β: [%.4f, %.4f]\n", model->β[0], model->β[1]);
    printf("  Fitted β0: %.4f\n", model->β0);
    
    /* Cleanup */
    model_destroy(model);
    basis_config_destroy(basis_cfg);
    fit_config_destroy(fit_cfg);
    free(X);
    free(y);
    
    printf("  PASS\n");
    return 1;
}

/* ============================================================================
 * Test 6: Lambda grid generation
 * ============================================================================ */

int test_lambda_grid_generation(void) {
    printf("Test 6: Lambda grid generation\n");
    
    idx_t n = 50;
    idx_t p = 2;
    
    real_t *X = (real_t *)malloc((size_t)(n * p) * sizeof(real_t));
    real_t *y = (real_t *)malloc((size_t)n * sizeof(real_t));
    
    /* Simple data */
    for (idx_t i = 0; i < n; i++) {
        X[i + 0 * n] = (real_t)i / 50.0;
        X[i + 1 * n] = (real_t)(n - i) / 50.0;
        y[i] = X[i + 0 * n] + X[i + 1 * n];
    }
    
    basis_config_t *basis_cfg = basis_config_create(1);
    idx_t cols[] = {0, 1};
    basis_config_add_section(basis_cfg, 0, cols, 2);
    
    fit_config_t *fit_cfg = fit_config_create();
    fit_cfg->n_lambda = 10;
    fit_cfg->lambda_min_ratio = 0.001;
    
    model_t *model = NULL;
    int status = fit_gaussian(X, n, p, y, basis_cfg, fit_cfg, &model);
    
    assert(status == FASTHAL_SUCCESS);
    assert(model->n_lambda == 10);
    
    /* Check lambda grid is monotonic decreasing */
    for (idx_t i = 0; i < model->n_lambda - 1; i++) {
        assert(model->λ[i] >= model->λ[i + 1]);
    }
    
    printf("  λ_max: %.6f, λ_min: %.6f\n", model->λ[0], model->λ[model->n_lambda - 1]);
    
    model_destroy(model);
    basis_config_destroy(basis_cfg);
    fit_config_destroy(fit_cfg);
    free(X);
    free(y);
    
    printf("  PASS\n");
    return 1;
}

/* ============================================================================
 * Test 7: Prediction on simple data
 * ============================================================================ */

int test_predict_gaussian(void) {
    printf("Test 7: Predict Gaussian\n");
    
    idx_t n_train = 30;
    idx_t n_test = 10;
    idx_t p = 2;
    
    real_t *X_train = (real_t *)malloc((size_t)(n_train * p) * sizeof(real_t));
    real_t *y_train = (real_t *)malloc((size_t)n_train * sizeof(real_t));
    real_t *X_test = (real_t *)malloc((size_t)(n_test * p) * sizeof(real_t));
    real_t *y_pred = (real_t *)malloc((size_t)n_test * sizeof(real_t));
    
    /* Create training data */
    for (idx_t i = 0; i < n_train; i++) {
        X_train[i + 0 * n_train] = (real_t)i / 30.0;
        X_train[i + 1 * n_train] = (real_t)(i % 15) / 15.0;
        y_train[i] = 2.0 * X_train[i + 0 * n_train] + 0.5 * X_train[i + 1 * n_train];
    }
    
    /* Fit model */
    basis_config_t *basis_cfg = basis_config_create(1);
    idx_t cols[] = {0, 1};
    basis_config_add_section(basis_cfg, 0, cols, 2);
    
    fit_config_t *fit_cfg = fit_config_create();
    fit_cfg->n_lambda = 3;
    
    model_t *model = NULL;
    fit_gaussian(X_train, n_train, p, y_train, basis_cfg, fit_cfg, &model);
    
    /* Create test data */
    for (idx_t i = 0; i < n_test; i++) {
        X_test[i + 0 * n_test] = (real_t)i / 10.0;
        X_test[i + 1 * n_test] = (real_t)(n_test - i) / 10.0;
    }
    
    /* Predict */
    int status = predict_gaussian(model, X_test, n_test, p, basis_cfg, y_pred);
    assert(status == FASTHAL_SUCCESS);
    
    /* Check predictions are not all zeros */
    int has_nonzero = 0;
    for (idx_t i = 0; i < n_test; i++) {
        if (fabs(y_pred[i]) > 1e-6) {
            has_nonzero = 1;
            break;
        }
    }
    assert(has_nonzero);
    
    printf("  Sample predictions: %.4f, %.4f, %.4f\n", y_pred[0], y_pred[n_test/2], y_pred[n_test-1]);
    
    /* Cleanup */
    model_destroy(model);
    basis_config_destroy(basis_cfg);
    fit_config_destroy(fit_cfg);
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_pred);
    
    printf("  PASS\n");
    return 1;
}

/* ============================================================================
 * Test 8: Binomial prediction (basic)
 * ============================================================================ */

int test_predict_binomial_basic(void) {
    printf("Test 8: Predict binomial basic\n");
    
    idx_t n = 20;
    idx_t p = 2;
    
    real_t *X = (real_t *)malloc((size_t)(n * p) * sizeof(real_t));
    real_t *probs = (real_t *)malloc((size_t)n * sizeof(real_t));
    
    for (idx_t i = 0; i < n; i++) {
        X[i + 0 * n] = (real_t)i / 20.0;
        X[i + 1 * n] = (real_t)(n - i) / 20.0;
    }
    
    basis_config_t *basis_cfg = basis_config_create(1);
    idx_t cols[] = {0, 1};
    basis_config_add_section(basis_cfg, 0, cols, 2);
    
    /* Create a simple model */
    model_t *model = model_create(p, 1);
    model->β[0] = 1.0;
    model->β[1] = -1.0;
    model->β0 = 0.0;
    
    /* Predict */
    int status = predict_binomial(model, X, n, p, basis_cfg, probs);
    assert(status == FASTHAL_SUCCESS);
    
    /* Check all probs are in [0,1] */
    for (idx_t i = 0; i < n; i++) {
        assert(probs[i] >= 0.0 && probs[i] <= 1.0);
    }
    
    printf("  Sample probabilities: %.4f, %.4f, %.4f\n", probs[0], probs[n/2], probs[n-1]);
    
    model_destroy(model);
    basis_config_destroy(basis_cfg);
    free(X);
    free(probs);
    
    printf("  PASS\n");
    return 1;
}

/* ============================================================================
 * Main test runner
 * ============================================================================ */

int main(void) {
    printf("\n");
    printf("===========================================================================\n");
    printf("Phase 3: High-level Fitting Tests\n");
    printf("===========================================================================\n\n");
    
    test_func_t tests[] = {
        test_basis_config_creation,
        test_fit_config_creation,
        test_model_creation,
        test_fit_univariate_gaussian,
        test_fit_multivariate_gaussian,
        test_lambda_grid_generation,
        test_predict_gaussian,
        test_predict_binomial_basic,
    };
    
    int n_tests = sizeof(tests) / sizeof(tests[0]);
    int n_passed = 0;
    
    for (int i = 0; i < n_tests; i++) {
        if (tests[i]()) {
            n_passed++;
        } else {
            printf("  ^^ Test %d FAILED\n\n", i + 1);
        }
    }
    
    printf("\n");
    printf("===========================================================================\n");
    printf("Results: %d / %d tests passed\n", n_passed, n_tests);
    printf("===========================================================================\n\n");
    
    return (n_passed == n_tests) ? 0 : 1;
}
