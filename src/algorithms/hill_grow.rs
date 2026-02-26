use crate::fenwick::WeightedSampler;
use crate::grid::Grid;
use crate::operation::scale;
use super::make_rng;
use rand::Rng;

/// Returns an iterator over `(row, col, kernel_value)` for every in-bounds cell
/// covered by `kernel` centered at `(row, col)`.
fn kernel_cells(
    grid_rows: usize,
    grid_cols: usize,
    row: usize,
    col: usize,
    kernel: &[Vec<f64>],
) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
    let half = (kernel.len() as i32 - 1) / 2;
    let row_i = row as i32;
    let col_i = col as i32;
    kernel.iter().enumerate().flat_map(move |(ki, krow)| {
        let i = row_i - half + ki as i32;
        krow.iter().copied().enumerate().filter_map(move |(kj, kv)| {
            let j = col_i - half + kj as i32;
            if i >= 0 && i < grid_rows as i32 && j >= 0 && j < grid_cols as i32 {
                Some((i as usize, j as usize, kv))
            } else {
                None
            }
        })
    })
}

fn apply_kernel(grid: &mut Grid, row: usize, col: usize, kernel: &[Vec<f64>], factor: f64) {
    for (iu, ju, kv) in kernel_cells(grid.rows, grid.cols, row, col, kernel) {
        grid[iu][ju] = (grid[iu][ju] + kv * factor).max(0.);
    }
}

/// Like `apply_kernel` but records every `(flat_index, delta)` change for Fenwick tree updates.
fn apply_kernel_tracked(
    grid: &mut Grid,
    row: usize,
    col: usize,
    kernel: &[Vec<f64>],
    factor: f64,
    changes: &mut Vec<(usize, f64)>,
) {
    let cols = grid.cols;
    for (iu, ju, kv) in kernel_cells(grid.rows, cols, row, col, kernel) {
        let old = grid[iu][ju];
        grid[iu][ju] = (old + kv * factor).max(0.);
        let delta = grid[iu][ju] - old;
        if delta.abs() > f64::EPSILON {
            changes.push((iu * cols + ju, delta));
        }
    }
}

fn valid_kernel(kernel: &[Vec<f64>]) -> bool {
    !kernel.is_empty()
        && kernel.len() == kernel[0].len()
        && kernel.len() % 2 == 1
        && kernel[0].len() % 2 == 1
}

/// Returns a hill-grow NLM with values ranging [0, 1).
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n` - Number of iterations.
/// * `runaway` - If `true`, probability of selection is proportional to current cell height,
///   causing hills to grow in clusters. Uses a Fenwick tree for O(log n) sampling per step.
/// * `kernel` - Convolution kernel applied at each iteration. Must be square with odd dimensions.
///   Defaults to a simple 3×3 diamond kernel.
/// * `only_grow` - If `true`, the surface only accumulates (no shrinking steps).
/// * `seed` - Optional RNG seed for reproducible results.
pub fn hill_grow(
    rows: usize,
    cols: usize,
    n: usize,
    runaway: bool,
    kernel: Option<Vec<Vec<f64>>>,
    only_grow: bool,
    seed: Option<u64>,
) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }

    let mut grid = if only_grow {
        Grid::new(rows, cols)
    } else {
        Grid::filled(rows, cols, 0.5)
    };

    let default_kernel = vec![vec![0., 0.5, 0.], vec![0.5, 1., 0.5], vec![0., 0.5, 0.]];
    let k = kernel.unwrap_or(default_kernel);
    if !valid_kernel(&k) {
        return grid;
    }

    let mut rng = make_rng(seed);

    if runaway {
        // Fenwick tree for weighted random selection in O(log(rows*cols)) per step,
        // replacing the previous O(rows*cols) full Vec rebuild each iteration.
        let mut sampler = WeightedSampler::new(&grid.data);
        // Pre-allocate change buffer: at most kernel_area changes per step.
        let mut changes = Vec::with_capacity(k.len() * k[0].len());

        for _ in 0..n {
            let flat_idx = {
                let total = sampler.total();
                if total <= 0.0 {
                    // All weights are zero (e.g. only_grow=true, grid not yet grown):
                    // fall back to uniform random selection.
                    rng.gen_range(0..rows * cols)
                } else {
                    sampler.sample(&mut rng)
                }
            };
            let row = flat_idx / cols;
            let col = flat_idx % cols;

            let grow = only_grow || rng.gen_bool(0.5);
            let factor = if grow { 0.1 } else { -0.1 };

            changes.clear();
            apply_kernel_tracked(&mut grid, row, col, &k, factor, &mut changes);

            // Update the Fenwick tree for each modified cell — O(k * log n) total.
            for &(idx, delta) in &changes {
                sampler.update(idx, delta);
            }
        }
    } else {
        // Non-runaway: plain uniform random — no tree needed.
        for _ in 0..n {
            let row = rng.gen_range(0..rows);
            let col = rng.gen_range(0..cols);
            let grow = only_grow || rng.gen_bool(0.5);
            let factor = if grow { 0.1 } else { -0.1 };
            apply_kernel(&mut grid, row, col, &k, factor);
        }
    }

    scale(&mut grid);
    grid
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{nan_count, zero_to_one_count};
    use rstest::rstest;

    #[rstest]
    #[case(0, 0, 50000, true, None)]
    #[case(1, 1, 50000, true, None)]
    #[case(2, 1, 50000, true, None)]
    #[case(3, 2, 50000, true, None)]
    #[case(4, 3, 50000, true, None)]
    #[case(5, 5, 50000, true, None)]
    #[case(10, 10, 50000, true, None)]
    #[case(100, 100, 50000, true, None)]
    #[case(500, 1000, 50000, true, None)]
    #[case(1000, 500, 50000, true, None)]
    #[case(1000, 1000, 50000, true, None)]
    fn test_hill_grow(
        #[case] rows: usize,
        #[case] cols: usize,
        #[case] n: usize,
        #[case] runaway: bool,
        #[case] kernel: Option<Vec<Vec<f64>>>,
    ) {
        let grid = hill_grow(rows, cols, n, runaway, kernel, false, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }
}
