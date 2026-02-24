pub mod export;
pub mod grid;
mod array;
pub mod operation;
mod fenwick;
#[cfg(feature = "python")]
mod python;

pub use grid::Grid;
pub use operation::{abs, add, add_value, invert, max, min, min_and_max, multiply, multiply_value, scale};

use crate::array::{diamond_square, rand_grid, rand_sub_grid, value_mask};
use crate::fenwick::WeightedSampler;
use crate::operation::{euclidean_distance_transform, interpolate};
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

fn make_rng(seed: Option<u64>) -> Box<dyn RngCore> {
    match seed {
        Some(s) => Box::new(StdRng::seed_from_u64(s)),
        None => Box::new(rand::thread_rng()),
    }
}

/// Converts an optional 64-bit seed to a 32-bit Perlin seed, generating a random
/// one from thread_rng when none is provided.
fn perlin_seed(seed: Option<u64>) -> u32 {
    seed.map(|s| s as u32).unwrap_or_else(|| rand::thread_rng().gen())
}

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
    let mut grid = Grid::filled(rows, cols, 1.0);

    // Track the last successfully placed label to avoid an O(rows*cols) max() scan
    // every iteration (which would make the whole function O(n * rows * cols)).
    let mut last_label = 1.0f64;
    let mut i: f64 = 1.;
    while last_label < n && i < n {
        let row = rng.gen_range(0..rows);
        let col = rng.gen_range(0..cols);
        if grid[row][col] == 1. {
            grid[row][col] = i;
            last_label = i;
        }
        i += 1.;
    }

    let mask = value_mask(&grid, 0.);
    interpolate(&mut grid, &mask, &mut rng);
    scale(&mut grid);

    grid
}

/// Returns a planar gradient NLM with values ranging [0, 1).
///
/// The gradient falls across the array in a direction given in degrees [0, 360).
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `direction` - Direction of the gradient in degrees. `None` picks a random direction.
/// * `seed` - Optional RNG seed for reproducible results.
///
/// Implementation ported from NLMpy.
pub fn planar_gradient(rows: usize, cols: usize, direction: Option<f64>, seed: Option<u64>) -> Grid {
    let mut rng = make_rng(seed);
    let d = direction.unwrap_or_else(|| rng.gen_range(0.0..360.0));
    let right = d.to_radians().sin();
    let down = -d.to_radians().cos();

    let mut grid = Grid::new(rows, cols);
    // Compute each cell's gradient value in parallel; no pre-allocated index arrays needed.
    let fill = |(k, v): (usize, &mut f64)| {
        *v = (k / cols) as f64 * down + (k % cols) as f64 * right;
    };
    #[cfg(feature = "parallel")]
    grid.data.par_iter_mut().enumerate().for_each(fill);
    #[cfg(not(feature = "parallel"))]
    grid.data.iter_mut().enumerate().for_each(fill);

    scale(&mut grid);
    grid
}

/// Returns an edge gradient NLM with values ranging [0, 1).
///
/// Zero at opposite ends, one at the midpoint between them. Direction in degrees [0, 360).
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `direction` - Direction of the gradient in degrees. `None` picks a random direction.
/// * `seed` - Optional RNG seed for reproducible results.
///
/// Implementation ported from NLMpy.
pub fn edge_gradient(rows: usize, cols: usize, direction: Option<f64>, seed: Option<u64>) -> Grid {
    let mut grid = planar_gradient(rows, cols, direction, seed);
    #[cfg(feature = "parallel")]
    grid.data.par_iter_mut().for_each(|v| *v = 1. - (2. * (*v - 0.5).abs()));
    #[cfg(not(feature = "parallel"))]
    grid.data.iter_mut().for_each(|v| *v = 1. - (2. * (*v - 0.5).abs()));
    scale(&mut grid);
    grid
}

/// Returns a distance gradient NLM with values ranging [0, 1).
///
/// Zero at a single random point, with the gradient emanating outward.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `seed` - Optional RNG seed for reproducible results.
///
/// Implementation ported from NLMpy.
pub fn distance_gradient(rows: usize, cols: usize, seed: Option<u64>) -> Grid {
    let mut rng = make_rng(seed);
    let mut grid = rand_grid(rows, cols, &mut rng);
    invert(&mut grid);
    euclidean_distance_transform(&mut grid);
    scale(&mut grid);
    grid
}

/// Returns a wave gradient NLM with values ranging [0, 1).
///
/// Cycles 0 → 1 → 0 repeatedly from one end of the array to the other.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `period` - Period of the wave function (smaller = larger wave).
/// * `direction` - Direction of the gradient in degrees. `None` picks a random direction.
/// * `seed` - Optional RNG seed for reproducible results.
///
/// Implementation ported from NLMpy.
pub fn wave_gradient(rows: usize, cols: usize, period: f64, direction: Option<f64>, seed: Option<u64>) -> Grid {
    let mut grid = planar_gradient(rows, cols, direction, seed);
    #[cfg(feature = "parallel")]
    grid.data.par_iter_mut().for_each(|v| *v = (*v * 2. * std::f64::consts::PI * period).sin());
    #[cfg(not(feature = "parallel"))]
    grid.data.iter_mut().for_each(|v| *v = (*v * 2. * std::f64::consts::PI * period).sin());
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

/// Returns a Perlin noise NLM with values ranging [0, 1).
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `scale_factor` - Frequency of the noise (higher = more features per unit).
/// * `seed` - Optional RNG seed for reproducible results.
pub fn perlin_noise(rows: usize, cols: usize, scale_factor: f64, seed: Option<u64>) -> Grid {
    use noise::{NoiseFn, Perlin};
    let seed_val = perlin_seed(seed);
    let perlin = Perlin::new(seed_val);

    let mut grid = Grid::new(rows, cols);
    // Each row is independent — parallelise over rows.
    let fill_row = |(i, row): (usize, &mut [f64])| {
        for (j, cell) in row.iter_mut().enumerate() {
            let nx = j as f64 / cols as f64 * scale_factor;
            let ny = i as f64 / rows as f64 * scale_factor;
            *cell = perlin.get([nx, ny]);
        }
    };
    #[cfg(feature = "parallel")]
    grid.data.par_chunks_mut(cols).enumerate().for_each(fill_row);
    #[cfg(not(feature = "parallel"))]
    grid.data.chunks_mut(cols).enumerate().for_each(fill_row);

    scale(&mut grid);
    grid
}

/// Returns a fractal Brownian motion (fBm) NLM with values ranging [0, 1).
///
/// Layers multiple octaves of Perlin noise for a more natural, detailed result.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `scale_factor` - Frequency of the base noise layer.
/// * `octaves` - Number of noise layers to combine (more = finer detail).
/// * `persistence` - Amplitude scaling per octave (0.5 = each octave half as strong).
/// * `lacunarity` - Frequency scaling per octave (2.0 = each octave twice the frequency).
/// * `seed` - Optional RNG seed for reproducible results.
pub fn fbm_noise(
    rows: usize,
    cols: usize,
    scale_factor: f64,
    octaves: usize,
    persistence: f64,
    lacunarity: f64,
    seed: Option<u64>,
) -> Grid {
    use noise::{NoiseFn, Perlin};
    let seed_val = perlin_seed(seed);
    // Each octave uses a separately seeded Perlin generator for independent noise.
    let generators: Vec<Perlin> = (0..octaves)
        .map(|o| Perlin::new(seed_val.wrapping_add(o as u32)))
        .collect();

    let mut grid = Grid::new(rows, cols);
    // Each row is independent — parallelise over rows.
    let fill_row = |(i, row): (usize, &mut [f64])| {
        for (j, cell) in row.iter_mut().enumerate() {
            let mut value = 0.0;
            let mut amplitude = 1.0;
            let mut frequency = scale_factor;
            let mut total_amplitude = 0.0;
            for gen in &generators {
                let nx = j as f64 / cols as f64 * frequency;
                let ny = i as f64 / rows as f64 * frequency;
                value += gen.get([nx, ny]) * amplitude;
                total_amplitude += amplitude;
                amplitude *= persistence;
                frequency *= lacunarity;
            }
            *cell = if total_amplitude > 0.0 { value / total_amplitude } else { 0.0 };
        }
    };
    #[cfg(feature = "parallel")]
    grid.data.par_chunks_mut(cols).enumerate().for_each(fill_row);
    #[cfg(not(feature = "parallel"))]
    grid.data.chunks_mut(cols).enumerate().for_each(fill_row);

    scale(&mut grid);
    grid
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn nan_count(grid: &Grid) -> usize {
        grid.iter().filter(|n| n.is_nan()).count()
    }

    fn zero_to_one_count(grid: &Grid) -> usize {
        grid.iter().filter(|&&n| n >= 0. && n <= 1.).count()
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
    fn test_planar_gradient(#[case] rows: usize, #[case] cols: usize) {
        let grid = planar_gradient(rows, cols, Some(90.), None);
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
    fn test_edge_gradient(#[case] rows: usize, #[case] cols: usize) {
        let grid = edge_gradient(rows, cols, Some(90.), None);
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
    fn test_distance_gradient(#[case] rows: usize, #[case] cols: usize) {
        let grid = distance_gradient(rows, cols, None);
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
    fn test_wave_gradient(#[case] rows: usize, #[case] cols: usize) {
        let grid = wave_gradient(rows, cols, 2.0, Some(90.), None);
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

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_perlin_noise(#[case] rows: usize, #[case] cols: usize) {
        let grid = perlin_noise(rows, cols, 4.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_perlin_seeded_determinism() {
        let a = perlin_noise(50, 50, 4.0, Some(42));
        let b = perlin_noise(50, 50, 4.0, Some(42));
        assert_eq!(a.data, b.data);
    }

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_fbm_noise(#[case] rows: usize, #[case] cols: usize) {
        let grid = fbm_noise(rows, cols, 4.0, 6, 0.5, 2.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }
}
