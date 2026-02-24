pub mod export;
pub mod grid;
mod array;
pub mod operation;
mod fenwick;
#[cfg(feature = "python")]
mod python;

pub use grid::Grid;
pub use operation::{abs, add, add_value, classify, invert, max, min, min_and_max, multiply, multiply_value, scale, threshold};

use crate::array::{diamond_square, rand_grid, rand_sub_grid};
use crate::fenwick::WeightedSampler;
use crate::operation::{euclidean_distance_transform, interpolate};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

fn make_rng(seed: Option<u64>) -> StdRng {
    StdRng::seed_from_u64(seed.unwrap_or_else(|| rand::thread_rng().gen()))
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
    let inv_rows = 1.0 / rows as f64;
    let inv_cols = 1.0 / cols as f64;
    // Each row is independent — parallelise over rows.
    let fill_row = |(i, row): (usize, &mut [f64])| {
        let ny = i as f64 * inv_rows * scale_factor;
        for (j, cell) in row.iter_mut().enumerate() {
            let nx = j as f64 * inv_cols * scale_factor;
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

    // Precompute per-octave (frequency, amplitude) pairs and the normalisation
    // factor — these are identical for every pixel and must not be recomputed
    // inside the hot loop.
    let mut freq_amp: Vec<(f64, f64)> = Vec::with_capacity(octaves);
    {
        let mut amp = 1.0f64;
        let mut freq = scale_factor;
        let mut total = 0.0f64;
        for _ in 0..octaves {
            freq_amp.push((freq, amp));
            total += amp;
            amp *= persistence;
            freq *= lacunarity;
        }
        // Store the reciprocal so the inner loop multiplies instead of divides.
        let inv_total = if total > 0.0 { 1.0 / total } else { 0.0 };
        // Re-use the amplitude slot to bake the normalisation in.
        for (_, a) in &mut freq_amp {
            *a *= inv_total;
        }
    }

    let mut grid = Grid::new(rows, cols);
    let inv_rows = 1.0 / rows as f64;
    let inv_cols = 1.0 / cols as f64;
    // Each row is independent — parallelise over rows.
    let fill_row = |(i, row): (usize, &mut [f64])| {
        // Precompute ny for each octave — constant across all columns of this row.
        let nys: Vec<f64> = freq_amp.iter().map(|&(freq, _)| i as f64 * inv_rows * freq).collect();
        for (j, cell) in row.iter_mut().enumerate() {
            let x = j as f64 * inv_cols;
            let mut value = 0.0;
            for (k, gen) in generators.iter().enumerate() {
                let (freq, amp) = freq_amp[k];
                value += gen.get([x * freq, nys[k]]) * amp;
            }
            *cell = value;
        }
    };
    #[cfg(feature = "parallel")]
    grid.data.par_chunks_mut(cols).enumerate().for_each(fill_row);
    #[cfg(not(feature = "parallel"))]
    grid.data.chunks_mut(cols).enumerate().for_each(fill_row);

    scale(&mut grid);
    grid
}

/// Returns a ridged multifractal NLM with values ranging [0, 1).
///
/// Produces sharp ridges and peak-like terrain.  Similar in structure to fBm
/// but each octave's value is folded (`1 - |x|`) so high-frequency details
/// accumulate into pronounced ridges rather than smooth hills.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `scale_factor` - Base noise frequency (higher = more features per unit).
/// * `octaves` - Number of noise layers to combine.
/// * `persistence` - Amplitude scaling per octave.
/// * `lacunarity` - Frequency scaling per octave (2.0 = each octave twice as dense).
/// * `seed` - Optional RNG seed for reproducible results.
pub fn ridged_noise(
    rows: usize,
    cols: usize,
    scale_factor: f64,
    octaves: usize,
    persistence: f64,
    lacunarity: f64,
    seed: Option<u64>,
) -> Grid {
    use noise::{MultiFractal, NoiseFn, Perlin, RidgedMulti};
    let seed_val = perlin_seed(seed);
    let ridged = RidgedMulti::<Perlin>::new(seed_val)
        .set_octaves(octaves)
        .set_persistence(persistence)
        .set_lacunarity(lacunarity);

    let mut grid = Grid::new(rows, cols);
    let inv_rows = 1.0 / rows as f64;
    let inv_cols = 1.0 / cols as f64;
    let fill_row = |(i, row): (usize, &mut [f64])| {
        let ny = i as f64 * inv_rows * scale_factor;
        for (j, cell) in row.iter_mut().enumerate() {
            let nx = j as f64 * inv_cols * scale_factor;
            *cell = ridged.get([nx, ny]);
        }
    };
    #[cfg(feature = "parallel")]
    grid.data.par_chunks_mut(cols).enumerate().for_each(fill_row);
    #[cfg(not(feature = "parallel"))]
    grid.data.chunks_mut(cols).enumerate().for_each(fill_row);

    scale(&mut grid);
    grid
}

/// Returns a billow NLM with values ranging [0, 1).
///
/// Billow noise applies an absolute-value fold to each octave of Perlin noise,
/// producing rounded, cloud- and hill-like patterns.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `scale_factor` - Base noise frequency.
/// * `octaves` - Number of noise layers to combine.
/// * `persistence` - Amplitude scaling per octave.
/// * `lacunarity` - Frequency scaling per octave.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn billow_noise(
    rows: usize,
    cols: usize,
    scale_factor: f64,
    octaves: usize,
    persistence: f64,
    lacunarity: f64,
    seed: Option<u64>,
) -> Grid {
    use noise::{Billow, MultiFractal, NoiseFn, Perlin};
    let seed_val = perlin_seed(seed);
    let billow = Billow::<Perlin>::new(seed_val)
        .set_octaves(octaves)
        .set_persistence(persistence)
        .set_lacunarity(lacunarity);

    let mut grid = Grid::new(rows, cols);
    let inv_rows = 1.0 / rows as f64;
    let inv_cols = 1.0 / cols as f64;
    let fill_row = |(i, row): (usize, &mut [f64])| {
        let ny = i as f64 * inv_rows * scale_factor;
        for (j, cell) in row.iter_mut().enumerate() {
            let nx = j as f64 * inv_cols * scale_factor;
            *cell = billow.get([nx, ny]);
        }
    };
    #[cfg(feature = "parallel")]
    grid.data.par_chunks_mut(cols).enumerate().for_each(fill_row);
    #[cfg(not(feature = "parallel"))]
    grid.data.chunks_mut(cols).enumerate().for_each(fill_row);

    scale(&mut grid);
    grid
}

/// Returns a Worley (cellular) noise NLM with values ranging [0, 1).
///
/// Each cell value is proportional to its distance to the nearest of a set of
/// randomly scattered Voronoi seed points, producing cellular / territory-like
/// patch patterns.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `scale_factor` - Frequency of the seed points (higher = smaller cells).
/// * `seed` - Optional RNG seed for reproducible results.
pub fn worley_noise(rows: usize, cols: usize, scale_factor: f64, seed: Option<u64>) -> Grid {
    use noise::{NoiseFn, Worley};
    let seed_val = perlin_seed(seed);

    let mut grid = Grid::new(rows, cols);
    let inv_rows = 1.0 / rows as f64;
    let inv_cols = 1.0 / cols as f64;
    // Worley contains Rc<dyn Fn> and is not Sync, so each row constructs its
    // own instance.  The permutation table creation is O(256) — negligible.
    let fill_row = |(i, row): (usize, &mut [f64])| {
        let worley = Worley::new(seed_val);
        let ny = i as f64 * inv_rows * scale_factor;
        for (j, cell) in row.iter_mut().enumerate() {
            let nx = j as f64 * inv_cols * scale_factor;
            *cell = worley.get([nx, ny]);
        }
    };
    #[cfg(feature = "parallel")]
    grid.data.par_chunks_mut(cols).enumerate().for_each(fill_row);
    #[cfg(not(feature = "parallel"))]
    grid.data.chunks_mut(cols).enumerate().for_each(fill_row);

    scale(&mut grid);
    grid
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

    // ── classify ─────────────────────────────────────────────────────────────

    #[test]
    fn test_classify_two_classes() {
        // With n=2 every output value must be exactly 0.0 or 1.0.
        let mut grid = random(50, 50, Some(1));
        classify(&mut grid, 2);
        for &v in grid.iter() {
            assert!(v == 0.0 || v == 1.0, "unexpected value {v}");
        }
    }

    #[test]
    fn test_classify_four_classes() {
        // With n=4 the only legal output values are 0.0, 1/3, 2/3, 1.0.
        let mut grid = random(50, 50, Some(2));
        classify(&mut grid, 4);
        let allowed = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
        for &v in grid.iter() {
            assert!(
                allowed.iter().any(|&a| (v - a).abs() < 1e-10),
                "unexpected value {v}"
            );
        }
    }

    #[test]
    fn test_classify_one_class() {
        let mut grid = random(10, 10, Some(3));
        classify(&mut grid, 1);
        for &v in grid.iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_classify_boundary_at_one() {
        // A cell with value 1.0 must land in the highest class.
        let mut grid = Grid::filled(1, 1, 1.0);
        classify(&mut grid, 3);
        assert_eq!(grid[0][0], 1.0);
    }

    // ── threshold ────────────────────────────────────────────────────────────

    #[test]
    fn test_threshold_binary() {
        let mut grid = random(50, 50, Some(4));
        let original = grid.clone();
        threshold(&mut grid, 0.5);
        for (i, (&orig, &new)) in original.iter().zip(grid.iter()).enumerate() {
            let expected = if orig < 0.5 { 0.0 } else { 1.0 };
            assert_eq!(new, expected, "cell {i}: original {orig}");
        }
    }

    #[test]
    fn test_threshold_all_ones() {
        // t = 0.0 → everything becomes 1.0 (all values >= 0.0).
        let mut grid = random(10, 10, Some(5));
        threshold(&mut grid, 0.0);
        for &v in grid.iter() {
            assert_eq!(v, 1.0);
        }
    }

    #[test]
    fn test_threshold_all_zeros() {
        // t > 1.0 → everything becomes 0.0.
        let mut grid = random(10, 10, Some(6));
        threshold(&mut grid, 1.1);
        for &v in grid.iter() {
            assert_eq!(v, 0.0);
        }
    }

    // ── ridged_noise ──────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_ridged_noise(#[case] rows: usize, #[case] cols: usize) {
        let grid = ridged_noise(rows, cols, 4.0, 6, 0.5, 2.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_ridged_noise_seeded_determinism() {
        let a = ridged_noise(50, 50, 4.0, 6, 0.5, 2.0, Some(42));
        let b = ridged_noise(50, 50, 4.0, 6, 0.5, 2.0, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── billow_noise ──────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_billow_noise(#[case] rows: usize, #[case] cols: usize) {
        let grid = billow_noise(rows, cols, 4.0, 6, 0.5, 2.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_billow_noise_seeded_determinism() {
        let a = billow_noise(50, 50, 4.0, 6, 0.5, 2.0, Some(42));
        let b = billow_noise(50, 50, 4.0, 6, 0.5, 2.0, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── worley_noise ──────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_worley_noise(#[case] rows: usize, #[case] cols: usize) {
        let grid = worley_noise(rows, cols, 4.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_worley_noise_seeded_determinism() {
        let a = worley_noise(50, 50, 4.0, Some(42));
        let b = worley_noise(50, 50, 4.0, Some(42));
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
}
