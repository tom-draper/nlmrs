//! C bindings for nlmrs.
//!
//! Build:
//!   cargo build --release
//!     → target/release/libnlmrs_c.so   (Linux shared)
//!     → target/release/libnlmrs_c.a    (Linux static)
//!
//! The generated header is written to include/nlmrs.h during the build.

use nlmrs::Grid;

// ── Return type ───────────────────────────────────────────────────────────────

/// A 2-D grid returned by every NLM generator.
///
/// `data` points to a heap-allocated row-major array of `rows * cols` doubles
/// with values in [0, 1]. Call `nlmrs_free` exactly once when you are done.
#[repr(C)]
pub struct NlmGrid {
    /// Heap-allocated row-major data, length = rows * cols.
    pub data: *mut f64,
    pub rows: usize,
    pub cols: usize,
}

impl NlmGrid {
    fn from_grid(grid: Grid) -> Self {
        let rows = grid.rows;
        let cols = grid.cols;
        let mut vec = grid.data;
        vec.shrink_to_fit(); // ensure capacity == len
        let ptr = vec.as_mut_ptr();
        std::mem::forget(vec);
        NlmGrid { data: ptr, rows, cols }
    }
}

// Safety: NlmGrid is an owned pointer; the caller controls lifetime.
unsafe impl Send for NlmGrid {}

// ── Memory management ─────────────────────────────────────────────────────────

/// Free a grid returned by any `nlmrs_*` function.
///
/// Must be called exactly once per grid. Behaviour is undefined if called on a
/// zeroed struct or a grid that has already been freed.
#[no_mangle]
pub extern "C" fn nlmrs_free(grid: NlmGrid) {
    if !grid.data.is_null() {
        let len = grid.rows * grid.cols;
        unsafe { drop(Vec::from_raw_parts(grid.data, len, len)) };
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[inline]
fn opt_seed(seed: *const u64) -> Option<u64> {
    if seed.is_null() { None } else { Some(unsafe { *seed }) }
}

#[inline]
fn opt_f64(ptr: *const f64) -> Option<f64> {
    if ptr.is_null() { None } else { Some(unsafe { *ptr }) }
}

#[inline]
fn opt_kernel(data: *const f64, size: usize) -> Option<Vec<Vec<f64>>> {
    if data.is_null() {
        None
    } else {
        let flat = unsafe { std::slice::from_raw_parts(data, size * size) };
        Some(flat.chunks(size).map(|row| row.to_vec()).collect())
    }
}

// ── Generators ───────────────────────────────────────────────────────────────

/// Spatially random NLM. Values in [0, 1).
///
/// @param seed  Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_random(rows: usize, cols: usize, seed: *const u64) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::random(rows, cols, opt_seed(seed)))
}

/// Random element nearest-neighbour NLM. Values in [0, 1).
///
/// @param n     Number of labelled seed elements to place.
/// @param seed  Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_random_element(
    rows: usize,
    cols: usize,
    n: f64,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::random_element(rows, cols, n, opt_seed(seed)))
}

/// Linear planar gradient. Values in [0, 1).
///
/// @param direction  Pointer to gradient direction in degrees [0, 360), or NULL for random.
/// @param seed       Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_planar_gradient(
    rows: usize,
    cols: usize,
    direction: *const f64,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::planar_gradient(rows, cols, opt_f64(direction), opt_seed(seed)))
}

/// Symmetric edge gradient — zero at both edges, peak in the middle. Values in [0, 1).
///
/// @param direction  Pointer to gradient direction in degrees [0, 360), or NULL for random.
/// @param seed       Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_edge_gradient(
    rows: usize,
    cols: usize,
    direction: *const f64,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::edge_gradient(rows, cols, opt_f64(direction), opt_seed(seed)))
}

/// Radial distance gradient from a random centre point. Values in [0, 1).
///
/// @param seed  Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_distance_gradient(
    rows: usize,
    cols: usize,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::distance_gradient(rows, cols, opt_seed(seed)))
}

/// Sinusoidal wave gradient. Values in [0, 1).
///
/// @param period     Wave period — smaller values produce larger waves.
/// @param direction  Pointer to wave direction in degrees [0, 360), or NULL for random.
/// @param seed       Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_wave_gradient(
    rows: usize,
    cols: usize,
    period: f64,
    direction: *const f64,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::wave_gradient(rows, cols, period, opt_f64(direction), opt_seed(seed)))
}

/// Diamond-square (midpoint displacement) fractal terrain. Values in [0, 1).
///
/// @param h     Spatial autocorrelation — 0 = rough, 1 = smooth.
/// @param seed  Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_midpoint_displacement(
    rows: usize,
    cols: usize,
    h: f64,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::midpoint_displacement(rows, cols, h, opt_seed(seed)))
}

/// Hill-grow NLM. Values in [0, 1).
///
/// @param n            Number of iterations.
/// @param runaway      If non-zero, hills cluster via weighted sampling.
/// @param kernel_data  Flat row-major convolution kernel, or NULL for the default 3×3 diamond.
/// @param kernel_size  Side length of the square kernel (ignored when kernel_data is NULL).
/// @param only_grow    If non-zero the surface only accumulates, never shrinks.
/// @param seed         Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_hill_grow(
    rows: usize,
    cols: usize,
    n: usize,
    runaway: bool,
    kernel_data: *const f64,
    kernel_size: usize,
    only_grow: bool,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::hill_grow(
        rows, cols, n, runaway,
        opt_kernel(kernel_data, kernel_size),
        only_grow, opt_seed(seed),
    ))
}

/// Single-layer Perlin noise. Values in [0, 1).
///
/// @param scale  Noise frequency — higher values produce more features per unit area.
/// @param seed   Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_perlin_noise(
    rows: usize,
    cols: usize,
    scale: f64,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::perlin_noise(rows, cols, scale, opt_seed(seed)))
}

/// Fractal Brownian motion — layered Perlin noise. Values in [0, 1).
///
/// @param scale       Base noise frequency.
/// @param octaves     Number of noise layers to combine.
/// @param persistence Amplitude scaling per octave.
/// @param lacunarity  Frequency scaling per octave.
/// @param seed        Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_fbm_noise(
    rows: usize,
    cols: usize,
    scale: f64,
    octaves: usize,
    persistence: f64,
    lacunarity: f64,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::fbm_noise(rows, cols, scale, octaves, persistence, lacunarity, opt_seed(seed)))
}

/// Ridged multifractal noise. Values in [0, 1).
///
/// @param scale       Base noise frequency.
/// @param octaves     Number of noise layers to combine.
/// @param persistence Amplitude scaling per octave.
/// @param lacunarity  Frequency scaling per octave.
/// @param seed        Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_ridged_noise(
    rows: usize,
    cols: usize,
    scale: f64,
    octaves: usize,
    persistence: f64,
    lacunarity: f64,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::ridged_noise(rows, cols, scale, octaves, persistence, lacunarity, opt_seed(seed)))
}

/// Billow noise — rounded cloud- and hill-like patterns. Values in [0, 1).
///
/// @param scale       Base noise frequency.
/// @param octaves     Number of noise layers to combine.
/// @param persistence Amplitude scaling per octave.
/// @param lacunarity  Frequency scaling per octave.
/// @param seed        Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_billow_noise(
    rows: usize,
    cols: usize,
    scale: f64,
    octaves: usize,
    persistence: f64,
    lacunarity: f64,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::billow_noise(rows, cols, scale, octaves, persistence, lacunarity, opt_seed(seed)))
}

/// Worley (cellular) noise — territory / patch patterns. Values in [0, 1).
///
/// @param scale  Seed-point frequency; higher values produce smaller cells.
/// @param seed   Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_worley_noise(
    rows: usize,
    cols: usize,
    scale: f64,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::worley_noise(rows, cols, scale, opt_seed(seed)))
}

/// Gaussian random field — spatially correlated noise. Values in [0, 1).
///
/// @param sigma  Gaussian kernel standard deviation in cells (controls correlation length).
/// @param seed   Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_gaussian_field(
    rows: usize,
    cols: usize,
    sigma: f64,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::gaussian_field(rows, cols, sigma, opt_seed(seed)))
}

/// Random cluster NLM via fault-line cuts. Values in [0, 1).
///
/// @param n     Number of fault-line cuts. Higher values produce finer-grained clusters.
/// @param seed  Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_random_cluster(
    rows: usize,
    cols: usize,
    n: usize,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::random_cluster(rows, cols, n, opt_seed(seed)))
}

/// Hybrid multifractal noise. Values in [0, 1).
///
/// @param scale       Base noise frequency.
/// @param octaves     Number of noise layers to combine.
/// @param persistence Amplitude scaling per octave.
/// @param lacunarity  Frequency scaling per octave.
/// @param seed        Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_hybrid_noise(
    rows: usize,
    cols: usize,
    scale: f64,
    octaves: usize,
    persistence: f64,
    lacunarity: f64,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::hybrid_noise(rows, cols, scale, octaves, persistence, lacunarity, opt_seed(seed)))
}

/// Value noise — interpolated lattice noise. Values in [0, 1).
///
/// @param scale  Noise frequency — higher values produce more features per unit area.
/// @param seed   Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_value_noise(
    rows: usize,
    cols: usize,
    scale: f64,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::value_noise(rows, cols, scale, opt_seed(seed)))
}

/// Turbulence — fBm with absolute-value fold per octave. Values in [0, 1).
///
/// @param scale       Base noise frequency.
/// @param octaves     Number of noise layers to combine.
/// @param persistence Amplitude scaling per octave.
/// @param lacunarity  Frequency scaling per octave.
/// @param seed        Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_turbulence(
    rows: usize,
    cols: usize,
    scale: f64,
    octaves: usize,
    persistence: f64,
    lacunarity: f64,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::turbulence(rows, cols, scale, octaves, persistence, lacunarity, opt_seed(seed)))
}

/// Domain-warped Perlin noise — organic, swirling patterns. Values in [0, 1).
///
/// @param scale         Coordinate frequency.
/// @param warp_strength Displacement magnitude applied to sample coordinates.
/// @param seed          Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_domain_warp(
    rows: usize,
    cols: usize,
    scale: f64,
    warp_strength: f64,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::domain_warp(rows, cols, scale, warp_strength, opt_seed(seed)))
}

/// Mosaic NLM — discrete Voronoi patch map with flat-coloured regions. Values in [0, 1).
///
/// @param n     Number of Voronoi seed points to place.
/// @param seed  Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_mosaic(
    rows: usize,
    cols: usize,
    n: usize,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::mosaic(rows, cols, n, opt_seed(seed)))
}

/// Binary percolation NLM. Values in {0.0, 1.0}.
///
/// Each cell is independently habitat (1.0) with probability `p`.
/// The critical percolation threshold for 4-connectivity is ~0.593.
///
/// @param p     Habitat probability (0.0–1.0).
/// @param seed  Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_percolation(
    rows: usize,
    cols: usize,
    p: f64,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::percolation(rows, cols, p, opt_seed(seed)))
}

/// Binary space partitioning NLM — hierarchical rectilinear partition. Values in [0, 1).
///
/// @param n     Number of rectangles in the final partition.
/// @param seed  Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_binary_space_partitioning(
    rows: usize,
    cols: usize,
    n: usize,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::binary_space_partitioning(rows, cols, n, opt_seed(seed)))
}

/// Rectangular cluster NLM — overlapping random axis-aligned rectangles. Values in [0, 1).
///
/// @param n     Number of rectangles to place.
/// @param seed  Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_rectangular_cluster(
    rows: usize,
    cols: usize,
    n: usize,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::rectangular_cluster(rows, cols, n, opt_seed(seed)))
}

/// Cellular automaton NLM — binary cave-like patterns from birth/survival rules. Values in {0.0, 1.0}.
///
/// @param p                  Initial alive probability.
/// @param iterations         Number of rule applications.
/// @param birth_threshold    Min live neighbours to birth a dead cell.
/// @param survival_threshold Min live neighbours for a live cell to survive.
/// @param seed               Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_cellular_automaton(
    rows: usize,
    cols: usize,
    p: f64,
    iterations: usize,
    birth_threshold: usize,
    survival_threshold: usize,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::cellular_automaton(
        rows, cols, p, iterations, birth_threshold, survival_threshold, opt_seed(seed),
    ))
}

/// Neighbourhood clustering NLM — iterative majority-vote patch clustering. Values in [0, 1).
///
/// @param k          Number of distinct patch classes (>= 2).
/// @param iterations Number of majority-vote passes.
/// @param seed       Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_neighbourhood_clustering(
    rows: usize,
    cols: usize,
    k: usize,
    iterations: usize,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::neighbourhood_clustering(rows, cols, k, iterations, opt_seed(seed)))
}

/// Spectral synthesis NLM — 1/f^beta noise generated in the frequency domain. Values in [0, 1).
///
/// @param beta  Spectral exponent: 0 = white noise, 1 = pink, 2 = brown/natural terrain.
/// @param seed  Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_spectral_synthesis(
    rows: usize,
    cols: usize,
    beta: f64,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::spectral_synthesis(rows, cols, beta, opt_seed(seed)))
}

/// Gray-Scott reaction-diffusion NLM — Turing-pattern spots, stripes and labyrinths. Values in [0, 1).
///
/// @param iterations Number of simulation steps.
/// @param feed       Feed rate for chemical A (controls pattern type).
/// @param kill       Kill rate for chemical B (controls pattern type).
/// @param seed       Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_reaction_diffusion(
    rows: usize, cols: usize, iterations: usize, feed: f64, kill: f64, seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::reaction_diffusion(rows, cols, iterations, feed, kill, opt_seed(seed)))
}

/// Eden growth model NLM — compact fractal blob grown from the centre. Values in {0.0, 1.0}.
///
/// @param n     Number of cells to add to the cluster.
/// @param seed  Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_eden_growth(
    rows: usize, cols: usize, n: usize, seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::eden_growth(rows, cols, n, opt_seed(seed)))
}

/// Fractal Brownian surface NLM — parameterised by Hurst exponent. Values in [0, 1).
///
/// @param h     Hurst exponent in (0, 1): 0 = rough, 1 = smooth.
/// @param seed  Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_fractal_brownian_surface(
    rows: usize, cols: usize, h: f64, seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::fractal_brownian_surface(rows, cols, h, opt_seed(seed)))
}

/// Elliptical landscape gradient centred at the grid midpoint. Values in [0, 1).
///
/// @param direction  Pointer to major-axis orientation in degrees [0, 360), or NULL for random.
/// @param aspect     Major-to-minor axis ratio (≥ 1.0). 1.0 = circular.
/// @param seed       Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_landscape_gradient(
    rows: usize, cols: usize, direction: *const f64, aspect: f64, seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::landscape_gradient(rows, cols, opt_f64(direction), aspect, opt_seed(seed)))
}

/// Diffusion-limited aggregation NLM — branching fractal cluster grown from the centre. Values in {0.0, 1.0}.
///
/// @param n     Number of particles to release.
/// @param seed  Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_diffusion_limited_aggregation(
    rows: usize,
    cols: usize,
    n: usize,
    seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::diffusion_limited_aggregation(rows, cols, n, opt_seed(seed)))
}

/// OpenSimplex noise NLM. Values in [0, 1).
///
/// @param scale  Coordinate frequency.
/// @param seed   Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_simplex_noise(
    rows: usize, cols: usize, scale: f64, seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::simplex_noise(rows, cols, scale, opt_seed(seed)))
}

/// Invasion percolation NLM — lowest-weight boundary growth from centre. Values in {0.0, 1.0}.
///
/// @param n    Number of cells to invade.
/// @param seed Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_invasion_percolation(
    rows: usize, cols: usize, n: usize, seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::invasion_percolation(rows, cols, n, opt_seed(seed)))
}

/// Sum of random Gaussian blob kernels. Values in [0, 1).
///
/// @param n     Number of blob centres.
/// @param sigma Gaussian width in cells.
/// @param seed  Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_gaussian_blobs(
    rows: usize, cols: usize, n: usize, sigma: f64, seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::gaussian_blobs(rows, cols, n, sigma, opt_seed(seed)))
}

/// Ising model via Glauber dynamics. Binary values {0.0, 1.0}.
///
/// @param beta       Inverse temperature (near 0.44 = critical point).
/// @param iterations Number of sweeps (each = rows × cols spin-flip attempts).
/// @param seed       Pointer to a u64 seed, or NULL for a random seed.
#[no_mangle]
pub extern "C" fn nlmrs_ising_model(
    rows: usize, cols: usize, beta: f64, iterations: usize, seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::ising_model(rows, cols, beta, iterations, opt_seed(seed)))
}

#[no_mangle]
pub extern "C" fn nlmrs_voronoi_distance(
    rows: usize, cols: usize, n: usize, seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::voronoi_distance(rows, cols, n, opt_seed(seed)))
}

#[no_mangle]
pub extern "C" fn nlmrs_sine_composite(
    rows: usize, cols: usize, waves: usize, seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::sine_composite(rows, cols, waves, opt_seed(seed)))
}

#[no_mangle]
pub extern "C" fn nlmrs_curl_noise(
    rows: usize, cols: usize, scale: f64, seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::curl_noise(rows, cols, scale, opt_seed(seed)))
}

#[no_mangle]
pub extern "C" fn nlmrs_hydraulic_erosion(
    rows: usize, cols: usize, n: usize, seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::hydraulic_erosion(rows, cols, n, opt_seed(seed)))
}

#[no_mangle]
pub extern "C" fn nlmrs_levy_flight(
    rows: usize, cols: usize, n: usize, seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::levy_flight(rows, cols, n, opt_seed(seed)))
}

#[no_mangle]
pub extern "C" fn nlmrs_poisson_disk(
    rows: usize, cols: usize, min_dist: f64, seed: *const u64,
) -> NlmGrid {
    NlmGrid::from_grid(nlmrs::poisson_disk(rows, cols, min_dist, opt_seed(seed)))
}
