#include "test_fixtures.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

/* Simple seeded random number generator for reproducibility */
static uint32_t seed = 12345;

static real_t randn(void) {
    /* Box-Muller transform for standard normal */
    const real_t PI = 3.14159265358979323846;
    seed = seed * 1103515245 + 12345;
    real_t u1 = (real_t)(seed / 65536 % 32768) / 32768.0;
    seed = seed * 1103515245 + 12345;
    real_t u2 = (real_t)(seed / 65536 % 32768) / 32768.0;
    return sqrt(-2.0 * log(u1 + 1e-10)) * cos(2.0 * PI * u2);
}

static real_t unifrand(void) {
    seed = seed * 1103515245 + 12345;
    return (real_t)(seed / 65536 % 32768) / 32768.0;
}

test_dataset_t* test_dataset_create_small(void) {
    test_dataset_t *ds = (test_dataset_t *)malloc(sizeof(test_dataset_t));
    if (!ds) return NULL;
    
    ds->n = 100;
    ds->p = 5;
    
    ds->X = (real_t *)malloc((size_t)(ds->n * ds->p) * sizeof(real_t));
    ds->y = (real_t *)malloc((size_t)ds->n * sizeof(real_t));
    ds->y_binomial = (real_t *)malloc((size_t)ds->n * sizeof(real_t));
    
    if (!ds->X || !ds->y || !ds->y_binomial) {
        if (ds->X) free(ds->X);
        if (ds->y) free(ds->y);
        if (ds->y_binomial) free(ds->y_binomial);
        free(ds);
        return NULL;
    }
    
    /* Generate random data */
    seed = 12345;
    
    /* X ~ N(0,1) clipped to [0, 1] */
    for (idx_t j = 0; j < ds->p; j++) {
        for (idx_t i = 0; i < ds->n; i++) {
            real_t val = unifrand();
            ds->X[j * ds->n + i] = val;
        }
    }
    
    /* y ~ sin(2*pi*X1) + sin(2*pi*X2) + X1*X2 + noise */
    const real_t PI = 3.14159265358979323846;
    for (idx_t i = 0; i < ds->n; i++) {
        real_t x1 = ds->X[i];
        real_t x2 = ds->X[ds->n + i];
        real_t mu = sin(2 * PI * x1) + sin(2 * PI * x2) + x1 * x2;
        ds->y[i] = mu + 0.1 * randn();
        ds->y_binomial[i] = (mu > 0) ? 1.0 : 0.0;
    }
    
    return ds;
}

void test_dataset_destroy(test_dataset_t *ds) {
    if (!ds) return;
    if (ds->X) free(ds->X);
    if (ds->y) free(ds->y);
    if (ds->y_binomial) free(ds->y_binomial);
    free(ds);
}

bool doubles_approx_equal(real_t a, real_t b, real_t tol) {
    real_t diff = fabs(a - b);
    real_t scale = fmax(fabs(a), fabs(b));
    if (scale < 1e-10) {
        return diff < tol;
    }
    return diff / scale < tol;
}

bool vectors_approx_equal(const real_t *v1, const real_t *v2, idx_t n, real_t tol) {
    if (!v1 || !v2) return false;
    for (idx_t i = 0; i < n; i++) {
        if (!doubles_approx_equal(v1[i], v2[i], tol)) {
            return false;
        }
    }
    return true;
}

bool ivectors_equal(const idx_t *v1, const idx_t *v2, idx_t n) {
    if (!v1 || !v2) return false;
    for (idx_t i = 0; i < n; i++) {
        if (v1[i] != v2[i]) return false;
    }
    return true;
}

void vector_print(const char *name, const real_t *v, idx_t n) {
    printf("%s = [", name);
    for (idx_t i = 0; i < MIN(n, 10); i++) {
        printf("%.6f", v[i]);
        if (i < MIN(n, 10) - 1) printf(", ");
    }
    if (n > 10) printf(", ... (%ld more)", (long)(n - 10));
    printf("]\n");
}

void ivector_print(const char *name, const idx_t *v, idx_t n) {
    printf("%s = [", name);
    for (idx_t i = 0; i < MIN(n, 10); i++) {
        printf("%d", (int)v[i]);
        if (i < MIN(n, 10) - 1) printf(", ");
    }
    if (n > 10) printf(", ... (%ld more)", (long)(n - 10));
    printf("]\n");
}

void matrix_print(const char *name, const real_t *X, idx_t nrows, idx_t ncols) {
    printf("%s (shape %d x %d, column-major):\n", name, (int)nrows, (int)ncols);
    idx_t max_rows = MIN(nrows, 5);
    idx_t max_cols = MIN(ncols, 5);
    for (idx_t i = 0; i < max_rows; i++) {
        for (idx_t j = 0; j < max_cols; j++) {
            printf("%8.4f", X[j * nrows + i]);
        }
        if (ncols > 5) printf(" ...");
        printf("\n");
    }
    if (nrows > 5) printf("... (%ld more rows)\n", (long)(nrows - 5));
}
