use crate::array::{diamond_square, rand_grid, rand_sub_grid};
use crate::grid::Grid;
use crate::operation::{interpolate, scale};
use super::make_rng;
use rand::Rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Returns a spatially random NLM with values ranging [0, 1).
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn random(rows: usize, cols: usize, seed: Option<u64>) -> Grid {
    let mut rng = make_rng(seed);
    rand_grid(rows, cols, &mut rng)
}

/// Returns a random element nearest-neighbour NLM with values ranging [0, 1).
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n` - Number of labelled seed elements to place.
/// * `seed` - Optional RNG seed for reproducible results.
///
/// Implementation ported from NLMpy.
pub fn random_element(rows: usize, cols: usize, n: f64, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }

    let mut rng = make_rng(seed);
    let mut grid = Grid::new(rows, cols);

    // Track the last successfully placed label to avoid an O(rows*cols) max() scan
    // every iteration (which would make the whole function O(n * rows * cols)).
    let mut last_label = 0.0f64;
    let mut i: f64 = 1.;
    while last_label < n && i < n {
        let row = rng.gen_range(0..rows);
        let col = rng.gen_range(0..cols);
        if grid[row][col] == 0. {
            grid[row][col] = i;
            last_label = i;
        }
        i += 1.;
    }

    interpolate(&mut grid);
    scale(&mut grid);

    grid
}

/// Returns a midpoint displacement NLM with values ranging [0, 1).
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `h` - Controls the spatial autocorrelation in element values.
/// * `seed` - Optional RNG seed for reproducible results.
///
/// Implementation ported from NLMpy.
pub fn midpoint_displacement(rows: usize, cols: usize, h: f64, seed: Option<u64>) -> Grid {
    let max_dim = rows.max(cols);
    if max_dim == 0 {
        return Grid::new(0, 0);
    }
    let n = ((max_dim - 1) as f64).log2().ceil() as u32;
    let dim = usize::pow(2, n) + 1;

    let mut rng = make_rng(seed);
    let mut surface = diamond_square(dim, h, &mut rng);
    surface = rand_sub_grid(surface, rows, cols, &mut rng);

    scale(&mut surface);
    surface
}

/// Returns a spatially correlated Gaussian random field NLM with values in [0, 1).
///
/// Generates white noise then applies a separable Gaussian blur, so nearby
/// cells are correlated over a distance roughly proportional to `sigma` cells.
/// `sigma` maps directly to ecological correlation length.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `sigma` - Standard deviation of the Gaussian kernel in cells.  Higher
///   values produce larger, smoother patches.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn gaussian_field(rows: usize, cols: usize, sigma: f64, seed: Option<u64>) -> Grid {
    let mut rng = make_rng(seed);

    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }

    let mut grid = rand_grid(rows, cols, &mut rng);

    if sigma <= 0.0 {
        scale(&mut grid);
        return grid;
    }

    // Build a 1-D normalised Gaussian kernel with radius = ceil(3σ).
    let radius = (3.0 * sigma).ceil() as usize;
    let kernel: Vec<f64> = {
        let size = 2 * radius + 1;
        let mut k = vec![0.0f64; size];
        let mut sum = 0.0f64;
        for i in 0..size {
            let x = i as f64 - radius as f64;
            k[i] = (-0.5 * (x / sigma).powi(2)).exp();
            sum += k[i];
        }
        k.iter_mut().for_each(|v| *v /= sum);
        k
    };

    // Horizontal pass — row-wise, cache-friendly.
    let mut row_conv = Grid::new(rows, cols);
    {
        let fill = |(i, out_row): (usize, &mut [f64])| {
            let src_row = &grid[i];
            for j in 0..cols {
                let mut val = 0.0f64;
                for (k, &w) in kernel.iter().enumerate() {
                    let jj = (j as i64 + k as i64 - radius as i64)
                        .clamp(0, cols as i64 - 1) as usize;
                    val += src_row[jj] * w;
                }
                out_row[j] = val;
            }
        };
        #[cfg(feature = "parallel")]
        row_conv.data.par_chunks_mut(cols).enumerate().for_each(fill);
        #[cfg(not(feature = "parallel"))]
        row_conv.data.chunks_mut(cols).enumerate().for_each(fill);
    }

    // Vertical pass.
    let mut result = Grid::new(rows, cols);
    {
        let fill = |(i, out_row): (usize, &mut [f64])| {
            for j in 0..cols {
                let mut val = 0.0f64;
                for (k, &w) in kernel.iter().enumerate() {
                    let ii = (i as i64 + k as i64 - radius as i64)
                        .clamp(0, rows as i64 - 1) as usize;
                    val += row_conv[ii][j] * w;
                }
                out_row[j] = val;
            }
        };
        #[cfg(feature = "parallel")]
        result.data.par_chunks_mut(cols).enumerate().for_each(fill);
        #[cfg(not(feature = "parallel"))]
        result.data.chunks_mut(cols).enumerate().for_each(fill);
    }

    scale(&mut result);
    result
}

/// Returns a random cluster NLM with values ranging [0, 1).
///
/// Applies `n` random fault-line cuts; each cut adds +1 to all cells on one
/// side and −1 on the other.  After all cuts the accumulated field is scaled
/// to [0, 1], producing spatially clustered landscapes with the linear
/// structural elements characteristic of geological fault patterns.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n` - Number of fault-line cuts (higher = finer-grained clustering).
/// * `seed` - Optional RNG seed for reproducible results.
pub fn random_cluster(rows: usize, cols: usize, n: usize, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }

    let mut rng = make_rng(seed);

    // Pre-generate all cuts so the per-cell inner loop has no mutable state.
    // Each cut: (px, py, sin θ, cos θ) where θ ∈ [0, π).
    let cuts: Vec<(f64, f64, f64, f64)> = (0..n)
        .map(|_| {
            let theta: f64 = rng.gen_range(0.0..std::f64::consts::PI);
            let px: f64 = rng.gen();
            let py: f64 = rng.gen();
            (px, py, theta.sin(), theta.cos())
        })
        .collect();

    let mut grid = Grid::new(rows, cols);
    let inv_rows = 1.0 / rows as f64;
    let inv_cols = 1.0 / cols as f64;

    let fill = |(k, v): (usize, &mut f64)| {
        let row = k / cols;
        let col = k % cols;
        let x = col as f64 * inv_cols;
        let y = row as f64 * inv_rows;
        *v = cuts
            .iter()
            .map(|&(px, py, sin_t, cos_t)| {
                if (x - px) * sin_t + (y - py) * cos_t > 0.0 { 1.0 } else { -1.0 }
            })
            .sum::<f64>();
    };
    #[cfg(feature = "parallel")]
    grid.data.par_iter_mut().enumerate().for_each(fill);
    #[cfg(not(feature = "parallel"))]
    grid.data.iter_mut().enumerate().for_each(fill);

    scale(&mut grid);
    grid
}

/// Returns a mosaic (discrete Voronoi) NLM with values ranging [0, 1).
///
/// Places `n` random seed cells each with a unique random float value, then
/// fills every remaining cell with the value of its nearest seed via BFS.
/// The result is a flat-coloured patch map — all cells within a patch share
/// the same value, unlike `random_element` where values grade between seeds.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n` - Number of Voronoi seed points to place.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn mosaic(rows: usize, cols: usize, n: usize, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }

    let mut rng = make_rng(seed);
    let mut grid = Grid::new(rows, cols);

    for _ in 0..n {
        let row = rng.gen_range(0..rows);
        let col = rng.gen_range(0..cols);
        if grid[row][col] == 0.0 {
            // Use (0, 1) range but avoid 0.0 (reserved for "unlabelled")
            grid[row][col] = rng.gen::<f64>() * 0.999 + 0.001;
        }
    }

    // BFS fills every unlabelled cell with the nearest seed's value.
    // Values are already in (0.001, 1.0) — no scale() needed.
    interpolate(&mut grid);
    grid
}

/// Returns a rectangular cluster NLM with values ranging [0, 1).
///
/// Places `n` random axis-aligned rectangles and accumulates +1 per cell for
/// each overlapping rectangle. The result is scaled to [0, 1], producing
/// patch-like landscapes with rectilinear boundaries.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n` - Number of rectangles to place.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn rectangular_cluster(rows: usize, cols: usize, n: usize, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }

    let mut rng = make_rng(seed);
    let mut grid = Grid::new(rows, cols);
    let max_side = (rows.max(cols) / 3).max(1);

    for _ in 0..n {
        let r0 = rng.gen_range(0..rows);
        let c0 = rng.gen_range(0..cols);
        let r1 = (r0 + rng.gen_range(1..=max_side)).min(rows);
        let c1 = (c0 + rng.gen_range(1..=max_side)).min(cols);
        for i in r0..r1 {
            for j in c0..c1 {
                grid[i][j] += 1.0;
            }
        }
    }

    scale(&mut grid);
    grid
}

/// Returns a binary percolation NLM with values in {0.0, 1.0}.
///
/// Each cell is independently set to 1.0 (habitat) with probability `p` and
/// 0.0 (matrix) with probability `1 - p`.  As `p` approaches the critical
/// percolation threshold (~0.593 for 4-connectivity) habitat clusters coalesce
/// and span the landscape, making this the canonical NLM for studying
/// percolation theory and habitat connectivity.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `p` - Probability that a cell is habitat (0.0–1.0).
/// * `seed` - Optional RNG seed for reproducible results.
///
/// Based on: Gardner et al. (1987). Neutral models for the analysis of
/// broad-scale landscape pattern. *Landscape Ecology* 1(1):19–28.
pub fn percolation(rows: usize, cols: usize, p: f64, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let mut rng = make_rng(seed);
    let data = (0..rows * cols)
        .map(|_| if rng.gen::<f64>() < p { 1.0 } else { 0.0 })
        .collect();
    Grid { data, rows, cols }
}

/// Returns a binary space partitioning (BSP) NLM with values in [0, 1).
///
/// Recursively splits the grid into non-overlapping axis-aligned rectangles.
/// At each step the largest remaining rectangle is split along its longest
/// dimension at a random position. Once `n` rectangles exist each is assigned
/// a unique random float value, producing a hierarchically-nested rectilinear
/// partition.  Unlike `rectangular_cluster` (overlapping accumulation), BSP
/// produces a complete, non-overlapping partition of the grid — ideal for
/// modelling human-dominated agricultural or urban landscapes.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n` - Number of rectangles in the final partition.
/// * `seed` - Optional RNG seed for reproducible results.
///
/// Based on: Etherington, Morgan & O'Sullivan (2022). Binary space
/// partitioning generates hierarchical and rectilinear neutral landscape
/// models suitable for human-dominated landscapes. *Landscape Ecology*
/// 37:1761–1769.
pub fn binary_space_partitioning(rows: usize, cols: usize, n: usize, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let mut rng = make_rng(seed);
    let mut grid = Grid::new(rows, cols);

    // (row_start, col_start, row_end, col_end)
    let mut rects: Vec<(usize, usize, usize, usize)> = vec![(0, 0, rows, cols)];

    while rects.len() < n.max(1) {
        // Always split the largest rectangle for more uniform patch sizes.
        let idx = rects
            .iter()
            .enumerate()
            .max_by_key(|(_, &(r0, c0, r1, c1))| (r1 - r0) * (c1 - c0))
            .map(|(i, _)| i)
            .unwrap();

        let (r0, c0, r1, c1) = rects[idx];
        let height = r1 - r0;
        let width = c1 - c0;

        if height <= 1 && width <= 1 {
            break; // All patches are single cells; can't split further.
        }

        if height >= width && height > 1 {
            let split = r0 + rng.gen_range(1..height);
            rects[idx] = (r0, c0, split, c1);
            rects.push((split, c0, r1, c1));
        } else {
            let split = c0 + rng.gen_range(1..width);
            rects[idx] = (r0, c0, r1, split);
            rects.push((r0, split, r1, c1));
        }
    }

    // Assign a unique random float to every leaf rectangle.
    for (r0, c0, r1, c1) in rects {
        let val: f64 = rng.gen();
        for r in r0..r1 {
            for c in c0..c1 {
                grid[r][c] = val;
            }
        }
    }

    grid
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{nan_count, zero_to_one_count};
    use rstest::rstest;

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(2, 1)]
    #[case(3, 2)]
    #[case(4, 3)]
    #[case(5, 5)]
    #[case(10, 10)]
    #[case(100, 100)]
    #[case(500, 1000)]
    #[case(1000, 500)]
    #[case(1000, 1000)]
    #[case(2000, 2000)]
    fn test_random(#[case] rows: usize, #[case] cols: usize) {
        let grid = random(rows, cols, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_random_seeded_determinism() {
        let a = random(50, 50, Some(42));
        let b = random(50, 50, Some(42));
        assert_eq!(a.data, b.data);
        let c = random(50, 50, Some(99));
        assert_ne!(a.data, c.data);
    }

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(2, 1)]
    #[case(3, 2)]
    #[case(4, 3)]
    #[case(5, 5)]
    #[case(10, 10)]
    #[case(100, 100)]
    #[case(500, 1000)]
    #[case(1000, 500)]
    #[case(1000, 1000)]
    #[case(2000, 2000)]
    fn test_random_element(#[case] rows: usize, #[case] cols: usize) {
        let grid = random_element(rows, cols, 900., None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(2, 1)]
    #[case(3, 2)]
    #[case(4, 3)]
    #[case(5, 5)]
    #[case(10, 10)]
    #[case(100, 100)]
    #[case(500, 1000)]
    #[case(1000, 500)]
    #[case(1000, 1000)]
    #[case(2000, 2000)]
    fn test_midpoint_displacement(#[case] rows: usize, #[case] cols: usize) {
        let grid = midpoint_displacement(rows, cols, 1., None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_midpoint_displacement_seeded_determinism() {
        let a = midpoint_displacement(100, 100, 1.0, Some(42));
        let b = midpoint_displacement(100, 100, 1.0, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── gaussian_field ────────────────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_gaussian_field(#[case] rows: usize, #[case] cols: usize) {
        let grid = gaussian_field(rows, cols, 5.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_gaussian_field_seeded_determinism() {
        let a = gaussian_field(50, 50, 5.0, Some(42));
        let b = gaussian_field(50, 50, 5.0, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── random_cluster ────────────────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    #[case(500, 500)]
    fn test_random_cluster(#[case] rows: usize, #[case] cols: usize) {
        let grid = random_cluster(rows, cols, 200, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_random_cluster_seeded_determinism() {
        let a = random_cluster(50, 50, 200, Some(42));
        let b = random_cluster(50, 50, 200, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── mosaic ────────────────────────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_mosaic(#[case] rows: usize, #[case] cols: usize) {
        let grid = mosaic(rows, cols, 50, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_mosaic_seeded_determinism() {
        let a = mosaic(50, 50, 50, Some(42));
        let b = mosaic(50, 50, 50, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── rectangular_cluster ───────────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_rectangular_cluster(#[case] rows: usize, #[case] cols: usize) {
        let grid = rectangular_cluster(rows, cols, 50, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_rectangular_cluster_seeded_determinism() {
        let a = rectangular_cluster(50, 50, 50, Some(42));
        let b = rectangular_cluster(50, 50, 50, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── percolation ───────────────────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_percolation(#[case] rows: usize, #[case] cols: usize) {
        let grid = percolation(rows, cols, 0.5, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        for &v in grid.iter() {
            assert!(v == 0.0 || v == 1.0, "unexpected value {v}");
        }
    }

    #[test]
    fn test_percolation_all_habitat() {
        let grid = percolation(20, 20, 1.0, Some(1));
        assert!(grid.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_percolation_no_habitat() {
        let grid = percolation(20, 20, 0.0, Some(1));
        assert!(grid.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_percolation_seeded_determinism() {
        let a = percolation(50, 50, 0.5, Some(42));
        let b = percolation(50, 50, 0.5, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── binary_space_partitioning ─────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_binary_space_partitioning(#[case] rows: usize, #[case] cols: usize) {
        let grid = binary_space_partitioning(rows, cols, 20, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_binary_space_partitioning_seeded_determinism() {
        let a = binary_space_partitioning(50, 50, 20, Some(42));
        let b = binary_space_partitioning(50, 50, 20, Some(42));
        assert_eq!(a.data, b.data);
    }
}
