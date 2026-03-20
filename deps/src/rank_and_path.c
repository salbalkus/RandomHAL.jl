#include "rank_and_path.h"
#include <stdlib.h>

/* Comparison function for qsort (pairs of value and index) */
typedef struct {
    real_t value;
    idx_t original_index;
} rank_pair_t;

static int compare_rank_pairs(const void *a, const void *b) {
    real_t diff = ((rank_pair_t *)a)->value - ((rank_pair_t *)b)->value;
    if (diff < 0) return -1;
    if (diff > 0) return 1;
    return 0;
}

/* Compute competitive rank for a single column */
int rank_column(const real_t *col, idx_t n, idx_t *ranks_out) {
    if (!col || !ranks_out || n <= 0) return FASTHAL_ERROR_INVALID_ARGS;
    
    /* Create pairs of (value, index) */
    rank_pair_t *pairs = (rank_pair_t *)malloc((size_t)n * sizeof(rank_pair_t));
    if (!pairs) return FASTHAL_ERROR_ALLOCATION;
    
    for (idx_t i = 0; i < n; i++) {
        pairs[i].value = col[i];
        pairs[i].original_index = i;
    }
    
    /* Sort by value */
    qsort(pairs, (size_t)n, sizeof(rank_pair_t), compare_rank_pairs);
    
    /* Assign competitive ranks (handle ties) */
    for (idx_t i = 0; i < n; i++) {
        idx_t rank = i + 1;  /* 1-indexed rank */
        /* Check for ties with next element */
        if (i > 0 && fabs(pairs[i].value - pairs[i-1].value) < FTOL) {
            rank = ranks_out[pairs[i-1].original_index];
        }
        ranks_out[pairs[i].original_index] = rank;
    }
    
    free(pairs);
    return FASTHAL_SUCCESS;
}

/* Compute ranks for all columns */
int rank_all_columns(const real_t *X, idx_t n, idx_t p, idx_t *ranks_matrix) {
    if (!X || !ranks_matrix || n <= 0 || p <= 0) {
        return FASTHAL_ERROR_INVALID_ARGS;
    }
    
    for (idx_t j = 0; j < p; j++) {
        /* Extract column j from X (column-major) */
        const real_t *col = X + j * n;
        /* Output to column j of ranks_matrix */
        idx_t *rank_col = ranks_matrix + j * n;
        
        int status = rank_column(col, n, rank_col);
        if (status != FASTHAL_SUCCESS) return status;
    }
    
    return FASTHAL_SUCCESS;
}

/* Check if observation i1 dominates observation i2 in all dimensions of section */
static bool dominates(const idx_t *ranks_section, idx_t n, idx_t section_size,
                      idx_t i1, idx_t i2) {
    for (idx_t s = 0; s < section_size; s++) {
        idx_t ranks_idx_1 = s * n + i1;  /* Column-major for ranks_section */
        idx_t ranks_idx_2 = s * n + i2;
        if (ranks_section[ranks_idx_1] < ranks_section[ranks_idx_2]) {
            return false;
        }
    }
    return true;
}

/* Sample nested path */
int path_sample(const idx_t *ranks_section, idx_t n, idx_t section_size,
                idx_t *path_out, idx_t *path_len_out) {
    if (!ranks_section || !path_out || !path_len_out || n <= 0) {
        return FASTHAL_ERROR_INVALID_ARGS;
    }
    
    /* Create array of (max_rank, index) for sorting */
    typedef struct {
        idx_t max_rank;
        idx_t index;
    } rank_max_pair_t;
    
    rank_max_pair_t *rank_indices = (rank_max_pair_t *)malloc((size_t)n * sizeof(rank_max_pair_t));
    if (!rank_indices) return FASTHAL_ERROR_ALLOCATION;
    
    /* Compute max rank for each observation */
    for (idx_t i = 0; i < n; i++) {
        idx_t max_rank = 0;
        for (idx_t s = 0; s < section_size; s++) {
            idx_t rank = ranks_section[s * n + i];
            if (rank > max_rank) max_rank = rank;
        }
        rank_indices[i].max_rank = max_rank;
        rank_indices[i].index = i;
    }
    
    /* Sort by max rank */
    qsort(rank_indices, (size_t)n, sizeof(rank_max_pair_t),
          (int (*)(const void *, const void *))
          (int (*)(const rank_max_pair_t *, const rank_max_pair_t *))
          compare_rank_pairs);
    
    /* Greedy path sampling */
    path_out[0] = 0;  /* Start at first observation (index 0 in rank-sorted order) */
    idx_t path_len = 1;
    idx_t i = 0, k = 1;
    
    while (i + k < n) {
        idx_t candidate_idx = rank_indices[i + k].index;
        idx_t current_idx = rank_indices[i].index;
        
        if (dominates(ranks_section, n, section_size, candidate_idx, current_idx)) {
            path_out[path_len++] = i + k;
            i = i + k;
            k = 1;
        } else {
            k++;
        }
    }
    
    /* Convert sorted indices back to original indices */
    for (idx_t j = 0; j < path_len; j++) {
        path_out[j] = rank_indices[path_out[j]].index;
    }
    
    *path_len_out = path_len;
    free(rank_indices);
    return FASTHAL_SUCCESS;
}

/* Binary bin search */
int binary_bin_search(const real_t *X, idx_t n,
                      const real_t *X_bins, const idx_t *path,
                      const idx_t *section, idx_t section_size,
                      idx_t num_bins,
                      idx_t *order_out) {
    if (!X || !X_bins || !path || !section || !order_out ||
        n <= 0 || section_size <= 0 || num_bins <= 0) {
        return FASTHAL_ERROR_INVALID_ARGS;
    }
    
    /* For each observation, binary search to find which bin it falls into */
    for (idx_t i = 0; i < n; i++) {
        idx_t left = 0;
        idx_t right = num_bins + 1;  /* Virtual infinity bin */
        
        /* Binary search loop */
        while (left + 1 < right) {
            idx_t mid = left + (right - left) / 2;
            
            /* Check if X[i, :] < X_bins[path[mid-1], :] in section */
            bool x_less = false;
            
            if (mid <= num_bins) {
                idx_t knot_idx = path[mid - 1];  /* Bin indices are 1-indexed */
                for (idx_t s = 0; s < section_size; s++) {
                    idx_t col_s = section[s];
                    real_t x_val = X[col_s * n + i];
                    real_t bin_val = X_bins[col_s * n + knot_idx];
                    if (x_val < bin_val - FTOL) {
                        x_less = true;
                        break;
                    }
                }
            } else {
                /* mid > num_bins: treat as X < infinity, always true */
                x_less = true;
            }
            
            if (x_less) {
                right = mid;
            } else {
                left = mid;
            }
        }
        
        order_out[i] = left;
    }
    
    return FASTHAL_SUCCESS;
}
