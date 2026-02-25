use extendr_api::prelude::*;
use nlmrs::Grid;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Convert a row-major Grid into a column-major R matrix.
///
/// R matrices are stored column-major; Grid.data is row-major.
/// `RMatrix::new_matrix` fills the R SEXP in column-major order, calling
/// `f(row, col)` for each position, so reading `grid.data[r * cols + c]`
/// gives the correct row-major element.
fn grid_to_rmatrix(grid: Grid) -> RMatrix<f64> {
    let rows = grid.rows;
    let cols = grid.cols;
    RMatrix::new_matrix(rows, cols, |r, c| grid.data[r * cols + c])
}

/// Map an R numeric-or-NULL seed to `Option<u64>`.
///
/// R has no unsigned integer type; seeds arrive as `f64` doubles.
fn seed_from_r(seed: Nullable<f64>) -> Option<u64> {
    match seed {
        Nullable::NotNull(v) => Some(v as u64),
        Nullable::Null => None,
    }
}

/// Map an R numeric-or-NULL direction to `Option<f64>`.
fn direction_from_r(dir: Nullable<f64>) -> Option<f64> {
    match dir {
        Nullable::NotNull(v) => Some(v),
        Nullable::Null => None,
    }
}

// ── Generators ───────────────────────────────────────────────────────────────

/// Spatially random NLM. Values in [0, 1).
#[extendr]
fn r_random(rows: i32, cols: i32, seed: Nullable<f64>) -> RMatrix<f64> {
    let grid = nlmrs::random(rows as usize, cols as usize, seed_from_r(seed));
    grid_to_rmatrix(grid)
}

/// Random element nearest-neighbour NLM. Values in [0, 1).
#[extendr]
fn r_random_element(rows: i32, cols: i32, n: f64, seed: Nullable<f64>) -> RMatrix<f64> {
    let grid = nlmrs::random_element(rows as usize, cols as usize, n, seed_from_r(seed));
    grid_to_rmatrix(grid)
}

/// Linear planar gradient. Values in [0, 1).
#[extendr]
fn r_planar_gradient(
    rows: i32,
    cols: i32,
    direction: Nullable<f64>,
    seed: Nullable<f64>,
) -> RMatrix<f64> {
    let grid = nlmrs::planar_gradient(
        rows as usize,
        cols as usize,
        direction_from_r(direction),
        seed_from_r(seed),
    );
    grid_to_rmatrix(grid)
}

/// Symmetric edge gradient — zero at both edges, peak in the middle. Values in [0, 1).
#[extendr]
fn r_edge_gradient(
    rows: i32,
    cols: i32,
    direction: Nullable<f64>,
    seed: Nullable<f64>,
) -> RMatrix<f64> {
    let grid = nlmrs::edge_gradient(
        rows as usize,
        cols as usize,
        direction_from_r(direction),
        seed_from_r(seed),
    );
    grid_to_rmatrix(grid)
}

/// Radial distance gradient from a random centre point. Values in [0, 1).
#[extendr]
fn r_distance_gradient(rows: i32, cols: i32, seed: Nullable<f64>) -> RMatrix<f64> {
    let grid = nlmrs::distance_gradient(rows as usize, cols as usize, seed_from_r(seed));
    grid_to_rmatrix(grid)
}

/// Sinusoidal wave gradient. Values in [0, 1).
#[extendr]
fn r_wave_gradient(
    rows: i32,
    cols: i32,
    period: f64,
    direction: Nullable<f64>,
    seed: Nullable<f64>,
) -> RMatrix<f64> {
    let grid = nlmrs::wave_gradient(
        rows as usize,
        cols as usize,
        period,
        direction_from_r(direction),
        seed_from_r(seed),
    );
    grid_to_rmatrix(grid)
}

/// Diamond-square (midpoint displacement) fractal terrain. Values in [0, 1).
#[extendr]
fn r_midpoint_displacement(rows: i32, cols: i32, h: f64, seed: Nullable<f64>) -> RMatrix<f64> {
    let grid = nlmrs::midpoint_displacement(rows as usize, cols as usize, h, seed_from_r(seed));
    grid_to_rmatrix(grid)
}

/// Hill-grow NLM. Values in [0, 1).
///
/// `kernel` must be an R list of equal-length numeric vectors forming a square
/// odd-dimensioned matrix, or NULL to use the default 3×3 diamond kernel.
#[extendr]
fn r_hill_grow(
    rows: i32,
    cols: i32,
    n: i32,
    runaway: bool,
    kernel: Nullable<List>,
    only_grow: bool,
    seed: Nullable<f64>,
) -> RMatrix<f64> {
    let kernel_opt: Option<Vec<Vec<f64>>> = match kernel {
        Nullable::NotNull(list) => {
            let k: Vec<Vec<f64>> = list
                .iter()
                .filter_map(|(_, obj)| obj.as_real_vector())
                .collect();
            if k.is_empty() { None } else { Some(k) }
        }
        Nullable::Null => None,
    };
    let grid = nlmrs::hill_grow(
        rows as usize,
        cols as usize,
        n as usize,
        runaway,
        kernel_opt,
        only_grow,
        seed_from_r(seed),
    );
    grid_to_rmatrix(grid)
}

/// Single-layer Perlin noise. Values in [0, 1).
#[extendr]
fn r_perlin_noise(rows: i32, cols: i32, scale_factor: f64, seed: Nullable<f64>) -> RMatrix<f64> {
    let grid = nlmrs::perlin_noise(rows as usize, cols as usize, scale_factor, seed_from_r(seed));
    grid_to_rmatrix(grid)
}

/// Fractal Brownian motion — layered Perlin noise. Values in [0, 1).
#[extendr]
fn r_fbm_noise(
    rows: i32,
    cols: i32,
    scale_factor: f64,
    octaves: i32,
    persistence: f64,
    lacunarity: f64,
    seed: Nullable<f64>,
) -> RMatrix<f64> {
    let grid = nlmrs::fbm_noise(
        rows as usize,
        cols as usize,
        scale_factor,
        octaves as usize,
        persistence,
        lacunarity,
        seed_from_r(seed),
    );
    grid_to_rmatrix(grid)
}

/// Ridged multifractal noise. Values in [0, 1).
#[extendr]
fn r_ridged_noise(
    rows: i32,
    cols: i32,
    scale_factor: f64,
    octaves: i32,
    persistence: f64,
    lacunarity: f64,
    seed: Nullable<f64>,
) -> RMatrix<f64> {
    let grid = nlmrs::ridged_noise(
        rows as usize,
        cols as usize,
        scale_factor,
        octaves as usize,
        persistence,
        lacunarity,
        seed_from_r(seed),
    );
    grid_to_rmatrix(grid)
}

/// Billow noise — rounded cloud- and hill-like patterns. Values in [0, 1).
#[extendr]
fn r_billow_noise(
    rows: i32,
    cols: i32,
    scale_factor: f64,
    octaves: i32,
    persistence: f64,
    lacunarity: f64,
    seed: Nullable<f64>,
) -> RMatrix<f64> {
    let grid = nlmrs::billow_noise(
        rows as usize,
        cols as usize,
        scale_factor,
        octaves as usize,
        persistence,
        lacunarity,
        seed_from_r(seed),
    );
    grid_to_rmatrix(grid)
}

/// Worley (cellular) noise — territory / patch patterns. Values in [0, 1).
#[extendr]
fn r_worley_noise(rows: i32, cols: i32, scale_factor: f64, seed: Nullable<f64>) -> RMatrix<f64> {
    let grid = nlmrs::worley_noise(rows as usize, cols as usize, scale_factor, seed_from_r(seed));
    grid_to_rmatrix(grid)
}

/// Gaussian random field — spatially correlated noise. Values in [0, 1).
#[extendr]
fn r_gaussian_field(rows: i32, cols: i32, sigma: f64, seed: Nullable<f64>) -> RMatrix<f64> {
    let grid = nlmrs::gaussian_field(rows as usize, cols as usize, sigma, seed_from_r(seed));
    grid_to_rmatrix(grid)
}

/// Random cluster NLM via fault-line cuts. Values in [0, 1).
#[extendr]
fn r_random_cluster(rows: i32, cols: i32, n: i32, seed: Nullable<f64>) -> RMatrix<f64> {
    let grid = nlmrs::random_cluster(rows as usize, cols as usize, n as usize, seed_from_r(seed));
    grid_to_rmatrix(grid)
}

/// Hybrid multifractal noise. Values in [0, 1).
#[extendr]
fn r_hybrid_noise(
    rows: i32,
    cols: i32,
    scale_factor: f64,
    octaves: i32,
    persistence: f64,
    lacunarity: f64,
    seed: Nullable<f64>,
) -> RMatrix<f64> {
    let grid = nlmrs::hybrid_noise(
        rows as usize,
        cols as usize,
        scale_factor,
        octaves as usize,
        persistence,
        lacunarity,
        seed_from_r(seed),
    );
    grid_to_rmatrix(grid)
}

/// Value noise — interpolated lattice noise. Values in [0, 1).
#[extendr]
fn r_value_noise(rows: i32, cols: i32, scale_factor: f64, seed: Nullable<f64>) -> RMatrix<f64> {
    let grid = nlmrs::value_noise(rows as usize, cols as usize, scale_factor, seed_from_r(seed));
    grid_to_rmatrix(grid)
}

/// Turbulence — fBm with absolute-value fold per octave. Values in [0, 1).
#[extendr]
fn r_turbulence(
    rows: i32,
    cols: i32,
    scale_factor: f64,
    octaves: i32,
    persistence: f64,
    lacunarity: f64,
    seed: Nullable<f64>,
) -> RMatrix<f64> {
    let grid = nlmrs::turbulence(
        rows as usize,
        cols as usize,
        scale_factor,
        octaves as usize,
        persistence,
        lacunarity,
        seed_from_r(seed),
    );
    grid_to_rmatrix(grid)
}

/// Domain-warped Perlin noise — organic, swirling patterns. Values in [0, 1).
#[extendr]
fn r_domain_warp(
    rows: i32,
    cols: i32,
    scale_factor: f64,
    warp_strength: f64,
    seed: Nullable<f64>,
) -> RMatrix<f64> {
    let grid = nlmrs::domain_warp(
        rows as usize,
        cols as usize,
        scale_factor,
        warp_strength,
        seed_from_r(seed),
    );
    grid_to_rmatrix(grid)
}

/// Mosaic NLM — discrete Voronoi patch map. Values in [0, 1).
#[extendr]
fn r_mosaic(rows: i32, cols: i32, n: i32, seed: Nullable<f64>) -> RMatrix<f64> {
    let grid = nlmrs::mosaic(rows as usize, cols as usize, n as usize, seed_from_r(seed));
    grid_to_rmatrix(grid)
}

/// Rectangular cluster NLM — overlapping random rectangles. Values in [0, 1).
#[extendr]
fn r_rectangular_cluster(rows: i32, cols: i32, n: i32, seed: Nullable<f64>) -> RMatrix<f64> {
    let grid =
        nlmrs::rectangular_cluster(rows as usize, cols as usize, n as usize, seed_from_r(seed));
    grid_to_rmatrix(grid)
}

// ── Post-processing ───────────────────────────────────────────────────────────

/// Convert a column-major R matrix to a row-major Grid.
fn rmatrix_to_grid(m: &RMatrix<f64>) -> Grid {
    let rows = m.nrows();
    let cols = m.ncols();
    // R matrix data is column-major: element [r, c] lives at col_major[c * rows + r].
    let col_major = m.data();
    let mut data = vec![0.0f64; rows * cols];
    for c in 0..cols {
        for r in 0..rows {
            data[r * cols + c] = col_major[c * rows + r];
        }
    }
    Grid { data, rows, cols }
}

/// Quantise a landscape matrix into `n` equal-width classes.
///
/// Class `k` (0-indexed) is assigned output value `k / (n - 1)`,
/// evenly spacing the classes across [0, 1].
#[extendr]
fn r_classify(m: &RMatrix<f64>, n: i32) -> RMatrix<f64> {
    let mut grid = rmatrix_to_grid(m);
    nlmrs::classify(&mut grid, n as usize);
    grid_to_rmatrix(grid)
}

/// Apply a binary threshold to a landscape matrix.
///
/// Values strictly below `t` become 0.0; values at or above become 1.0.
#[extendr]
fn r_threshold(m: &RMatrix<f64>, t: f64) -> RMatrix<f64> {
    let mut grid = rmatrix_to_grid(m);
    nlmrs::threshold(&mut grid, t);
    grid_to_rmatrix(grid)
}

/// Binary percolation NLM. Values in {0.0, 1.0}.
#[extendr]
fn r_percolation(rows: i32, cols: i32, p: f64, seed: Nullable<f64>) -> RMatrix<f64> {
    let grid = nlmrs::percolation(rows as usize, cols as usize, p, seed_from_r(seed));
    grid_to_rmatrix(grid)
}

/// Binary space partitioning NLM — hierarchical rectilinear partition. Values in [0, 1).
#[extendr]
fn r_binary_space_partitioning(rows: i32, cols: i32, n: i32, seed: Nullable<f64>) -> RMatrix<f64> {
    let grid = nlmrs::binary_space_partitioning(
        rows as usize,
        cols as usize,
        n as usize,
        seed_from_r(seed),
    );
    grid_to_rmatrix(grid)
}

// ── Module registration ───────────────────────────────────────────────────────

// `mod nlmrs` matches the R package name, so the generated C symbol is
// `R_init_nlmrs_extendr`, which entrypoint.c bridges to `R_init_nlmrs`.
extendr_module! {
    mod nlmrs;
    fn r_random;
    fn r_random_element;
    fn r_planar_gradient;
    fn r_edge_gradient;
    fn r_distance_gradient;
    fn r_wave_gradient;
    fn r_midpoint_displacement;
    fn r_hill_grow;
    fn r_perlin_noise;
    fn r_fbm_noise;
    fn r_ridged_noise;
    fn r_billow_noise;
    fn r_worley_noise;
    fn r_gaussian_field;
    fn r_random_cluster;
    fn r_hybrid_noise;
    fn r_value_noise;
    fn r_turbulence;
    fn r_domain_warp;
    fn r_mosaic;
    fn r_rectangular_cluster;
    fn r_percolation;
    fn r_binary_space_partitioning;
    fn r_classify;
    fn r_threshold;
}
