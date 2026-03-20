#ifndef FASTHAL_RANK_AND_PATH_H
#define FASTHAL_RANK_AND_PATH_H

#include "common.h"
#include "memory_pool.h"

/* Competitive ranking and nested path sampling for HAL basis. */

/* Compute competitive rank of each element in a column (0-indexed).
 * This ranks observations by their value, with ties getting the same rank.
 * The result is the rank of each observation (1-indexed for compatibility with Julia).
 * 
 * Args:
 *   col: Column of data to rank
 *   n: Length of column
 *   ranks_out: Output array of ranks (must be allocated, length n)
 * Returns: FASTHAL_SUCCESS or error code
 */
int rank_column(const real_t *col, idx_t n, idx_t *ranks_out);

/* Compute competitive ranks for all columns simultaneously.
 * Stores results in ranks_matrix in column-major order.
 * 
 * Args:
 *   X: Input matrix (n x p), column-major
 *   n: Number of rows
 *   p: Number of columns
 *   ranks_matrix: Output matrix (n x p), column-major, pre-allocated
 * Returns: FASTHAL_SUCCESS or error code
 */
int rank_all_columns(const real_t *X, idx_t n, idx_t p, idx_t *ranks_matrix);

/* Sample a nested path of knots given ranks for a section.
 * The nested path is a sequence of indices where each knot dominates all
 * previous ones in all dimensions of the section.
 *
 * Args:
 *   ranks_section: Rank matrix subset for this section (n x len_section)
 *   n: Number of observations
 *   section_size: Length of section (len(S))
 *   path_out: Output path indices (pre-allocated, worst case n elements)
 *   path_len_out: Output length of path
 * Returns: FASTHAL_SUCCESS or error code
 */
int path_sample(const idx_t *ranks_section, idx_t n, idx_t section_size,
                idx_t *path_out, idx_t *path_len_out);

/* Binary bin search: for each observation, find which bin of the nested path it falls into.
 * The bins are defined by the nested path knots.
 *
 * Args:
 *   X: Full data matrix (n x p), column-major
 *   X_bins: Data at knot indices (num_bins x p), column-major
 *   section: Indices of columns to use for comparison (len_section elements)
 *   section_size: Length of section
 *   num_bins: Number of bins (length of path)
 *   order_out: Output array (n) with bin indices (1-indexed, 0 if before first bin)
 * Returns: FASTHAL_SUCCESS or error code
 */
int binary_bin_search(const real_t *X, idx_t n,
                      const real_t *X_bins, const idx_t *path,
                      const idx_t *section, idx_t section_size,
                      idx_t num_bins,
                      idx_t *order_out);

#endif /* FASTHAL_RANK_AND_PATH_H */
