use wasm_bindgen::prelude::*;
use nlmrs::Grid;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn grid_to_wasm(grid: Grid) -> WasmGrid {
    WasmGrid {
        rows: grid.rows as u32,
        cols: grid.cols as u32,
        data: grid.data,
    }
}

/// Map a JS number-or-undefined seed to `Option<u64>`.
///
/// Seeds are passed as `Option<u32>` to avoid JS BigInt (which `u64` requires).
fn seed_from_js(seed: Option<u32>) -> Option<u64> {
    seed.map(|s| s as u64)
}

// ── Return type ───────────────────────────────────────────────────────────────

/// A 2-D grid returned by every NLM generator.
///
/// - `rows` and `cols` are public JS properties.
/// - `data` is a getter returning a `Float64Array` (copy) in **row-major** order,
///   with values in \[0, 1\].
///
/// ```js
/// const grid = nlmrs.midpoint_displacement(200, 200, 0.8, 42);
/// const flat = grid.data;          // Float64Array, length rows*cols
/// const val  = flat[r * grid.cols + c]; // value at (r, c)
/// grid.free();                     // release Rust memory
/// ```
#[wasm_bindgen]
pub struct WasmGrid {
    pub rows: u32,
    pub cols: u32,
    data: Vec<f64>,
}

#[wasm_bindgen]
impl WasmGrid {
    /// Returns a copy of the flat row-major grid data as a `Float64Array`.
    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Vec<f64> {
        self.data.clone()
    }
}

// ── Generators ───────────────────────────────────────────────────────────────

/// Spatially random NLM. Values in \[0, 1\).
///
/// @param rows - Number of rows.
/// @param cols - Number of columns.
/// @param seed - Optional integer seed for reproducibility.
#[wasm_bindgen]
pub fn random(rows: u32, cols: u32, seed: Option<u32>) -> WasmGrid {
    let grid = nlmrs::random(rows as usize, cols as usize, seed_from_js(seed));
    grid_to_wasm(grid)
}

/// Random element nearest-neighbour NLM. Values in \[0, 1\).
///
/// @param rows - Number of rows.
/// @param cols - Number of columns.
/// @param n    - Number of labelled seed elements to place (default 50 000).
/// @param seed - Optional integer seed.
#[wasm_bindgen]
pub fn random_element(rows: u32, cols: u32, n: f64, seed: Option<u32>) -> WasmGrid {
    let grid = nlmrs::random_element(rows as usize, cols as usize, n, seed_from_js(seed));
    grid_to_wasm(grid)
}

/// Linear planar gradient. Values in \[0, 1\).
///
/// @param rows      - Number of rows.
/// @param cols      - Number of columns.
/// @param direction - Gradient direction in degrees \[0, 360). Omit for random.
/// @param seed      - Optional integer seed.
#[wasm_bindgen]
pub fn planar_gradient(rows: u32, cols: u32, direction: Option<f64>, seed: Option<u32>) -> WasmGrid {
    let grid = nlmrs::planar_gradient(rows as usize, cols as usize, direction, seed_from_js(seed));
    grid_to_wasm(grid)
}

/// Symmetric edge gradient — zero at both edges, peak in the middle.
/// Values in \[0, 1\).
///
/// @param rows      - Number of rows.
/// @param cols      - Number of columns.
/// @param direction - Gradient direction in degrees \[0, 360). Omit for random.
/// @param seed      - Optional integer seed.
#[wasm_bindgen]
pub fn edge_gradient(rows: u32, cols: u32, direction: Option<f64>, seed: Option<u32>) -> WasmGrid {
    let grid = nlmrs::edge_gradient(rows as usize, cols as usize, direction, seed_from_js(seed));
    grid_to_wasm(grid)
}

/// Radial distance gradient from a random centre point. Values in \[0, 1\).
///
/// @param rows - Number of rows.
/// @param cols - Number of columns.
/// @param seed - Optional integer seed.
#[wasm_bindgen]
pub fn distance_gradient(rows: u32, cols: u32, seed: Option<u32>) -> WasmGrid {
    let grid = nlmrs::distance_gradient(rows as usize, cols as usize, seed_from_js(seed));
    grid_to_wasm(grid)
}

/// Sinusoidal wave gradient. Values in \[0, 1\).
///
/// @param rows      - Number of rows.
/// @param cols      - Number of columns.
/// @param period    - Wave period; smaller = larger waves (default 2.5).
/// @param direction - Wave direction in degrees \[0, 360). Omit for random.
/// @param seed      - Optional integer seed.
#[wasm_bindgen]
pub fn wave_gradient(
    rows: u32,
    cols: u32,
    period: f64,
    direction: Option<f64>,
    seed: Option<u32>,
) -> WasmGrid {
    let grid = nlmrs::wave_gradient(rows as usize, cols as usize, period, direction, seed_from_js(seed));
    grid_to_wasm(grid)
}

/// Diamond-square (midpoint displacement) fractal terrain. Values in \[0, 1\).
///
/// @param rows - Number of rows.
/// @param cols - Number of columns.
/// @param h    - Hurst exponent: 0 = rough, 1 = smooth (default 1.0).
/// @param seed - Optional integer seed.
#[wasm_bindgen]
pub fn midpoint_displacement(rows: u32, cols: u32, h: f64, seed: Option<u32>) -> WasmGrid {
    let grid = nlmrs::midpoint_displacement(rows as usize, cols as usize, h, seed_from_js(seed));
    grid_to_wasm(grid)
}

/// Hill-grow NLM. Values in \[0, 1\).
///
/// @param rows        - Number of rows.
/// @param cols        - Number of columns.
/// @param n           - Number of iterations (default 10 000).
/// @param runaway     - If true, selection is weighted by cell height (default true).
/// @param kernel_flat - Flat row-major kernel as `Float64Array`, or empty / omitted for
///                      the default 3×3 diamond kernel.
/// @param kernel_size - Side length of the kernel (e.g. 3 for a 3×3 kernel).
///                      Ignored when `kernel_flat` is empty.
/// @param only_grow   - If true, surface only accumulates (default false).
/// @param seed        - Optional integer seed.
#[wasm_bindgen]
pub fn hill_grow(
    rows: u32,
    cols: u32,
    n: u32,
    runaway: bool,
    kernel_flat: Vec<f64>,
    kernel_size: u32,
    only_grow: bool,
    seed: Option<u32>,
) -> WasmGrid {
    let kernel: Option<Vec<Vec<f64>>> = if kernel_size > 0 && !kernel_flat.is_empty() {
        let sz = kernel_size as usize;
        Some(kernel_flat.chunks(sz).map(|row| row.to_vec()).collect())
    } else {
        None
    };
    let grid = nlmrs::hill_grow(
        rows as usize,
        cols as usize,
        n as usize,
        runaway,
        kernel,
        only_grow,
        seed_from_js(seed),
    );
    grid_to_wasm(grid)
}

/// Single-layer Perlin noise. Values in \[0, 1\).
///
/// @param rows         - Number of rows.
/// @param cols         - Number of columns.
/// @param scale_factor - Noise frequency; higher = more features (default 4.0).
/// @param seed         - Optional integer seed.
#[wasm_bindgen]
pub fn perlin_noise(rows: u32, cols: u32, scale_factor: f64, seed: Option<u32>) -> WasmGrid {
    let grid = nlmrs::perlin_noise(rows as usize, cols as usize, scale_factor, seed_from_js(seed));
    grid_to_wasm(grid)
}

/// Fractal Brownian motion — layered Perlin noise. Values in \[0, 1\).
///
/// @param rows         - Number of rows.
/// @param cols         - Number of columns.
/// @param scale_factor - Base noise frequency (default 4.0).
/// @param octaves      - Number of noise layers to combine (default 6).
/// @param persistence  - Amplitude scaling per octave, e.g. 0.5 (default 0.5).
/// @param lacunarity   - Frequency scaling per octave, e.g. 2.0 (default 2.0).
/// @param seed         - Optional integer seed.
#[wasm_bindgen]
pub fn fbm_noise(
    rows: u32,
    cols: u32,
    scale_factor: f64,
    octaves: u32,
    persistence: f64,
    lacunarity: f64,
    seed: Option<u32>,
) -> WasmGrid {
    let grid = nlmrs::fbm_noise(
        rows as usize,
        cols as usize,
        scale_factor,
        octaves as usize,
        persistence,
        lacunarity,
        seed_from_js(seed),
    );
    grid_to_wasm(grid)
}

/// Ridged multifractal noise. Values in \[0, 1\).
///
/// @param rows        - Number of rows.
/// @param cols        - Number of columns.
/// @param scale_factor - Base noise frequency (default 4.0).
/// @param octaves     - Number of noise layers (default 6).
/// @param persistence - Amplitude scaling per octave (default 0.5).
/// @param lacunarity  - Frequency scaling per octave (default 2.0).
/// @param seed        - Optional integer seed.
#[wasm_bindgen]
pub fn ridged_noise(
    rows: u32,
    cols: u32,
    scale_factor: f64,
    octaves: u32,
    persistence: f64,
    lacunarity: f64,
    seed: Option<u32>,
) -> WasmGrid {
    let grid = nlmrs::ridged_noise(
        rows as usize,
        cols as usize,
        scale_factor,
        octaves as usize,
        persistence,
        lacunarity,
        seed_from_js(seed),
    );
    grid_to_wasm(grid)
}

/// Billow noise — rounded cloud- and hill-like patterns. Values in \[0, 1\).
///
/// @param rows        - Number of rows.
/// @param cols        - Number of columns.
/// @param scale_factor - Base noise frequency (default 4.0).
/// @param octaves     - Number of noise layers (default 6).
/// @param persistence - Amplitude scaling per octave (default 0.5).
/// @param lacunarity  - Frequency scaling per octave (default 2.0).
/// @param seed        - Optional integer seed.
#[wasm_bindgen]
pub fn billow_noise(
    rows: u32,
    cols: u32,
    scale_factor: f64,
    octaves: u32,
    persistence: f64,
    lacunarity: f64,
    seed: Option<u32>,
) -> WasmGrid {
    let grid = nlmrs::billow_noise(
        rows as usize,
        cols as usize,
        scale_factor,
        octaves as usize,
        persistence,
        lacunarity,
        seed_from_js(seed),
    );
    grid_to_wasm(grid)
}

/// Worley (cellular) noise — territory / patch patterns. Values in \[0, 1\).
///
/// @param rows         - Number of rows.
/// @param cols         - Number of columns.
/// @param scale_factor - Seed-point frequency; higher = smaller cells (default 4.0).
/// @param seed         - Optional integer seed.
#[wasm_bindgen]
pub fn worley_noise(rows: u32, cols: u32, scale_factor: f64, seed: Option<u32>) -> WasmGrid {
    let grid = nlmrs::worley_noise(rows as usize, cols as usize, scale_factor, seed_from_js(seed));
    grid_to_wasm(grid)
}

/// Gaussian random field — spatially correlated noise. Values in \[0, 1\).
///
/// @param rows  - Number of rows.
/// @param cols  - Number of columns.
/// @param sigma - Gaussian kernel standard deviation in cells (default 10.0).
/// @param seed  - Optional integer seed.
#[wasm_bindgen]
pub fn gaussian_field(rows: u32, cols: u32, sigma: f64, seed: Option<u32>) -> WasmGrid {
    let grid = nlmrs::gaussian_field(rows as usize, cols as usize, sigma, seed_from_js(seed));
    grid_to_wasm(grid)
}

/// Random cluster NLM via fault-line cuts. Values in \[0, 1\).
///
/// @param rows - Number of rows.
/// @param cols - Number of columns.
/// @param n    - Number of fault-line cuts (default 200).
/// @param seed - Optional integer seed.
#[wasm_bindgen]
pub fn random_cluster(rows: u32, cols: u32, n: u32, seed: Option<u32>) -> WasmGrid {
    let grid = nlmrs::random_cluster(rows as usize, cols as usize, n as usize, seed_from_js(seed));
    grid_to_wasm(grid)
}

/// Hybrid multifractal noise. Values in \[0, 1\).
///
/// @param rows         - Number of rows.
/// @param cols         - Number of columns.
/// @param scale_factor - Base noise frequency (default 4.0).
/// @param octaves      - Number of noise layers (default 6).
/// @param persistence  - Amplitude scaling per octave (default 0.5).
/// @param lacunarity   - Frequency scaling per octave (default 2.0).
/// @param seed         - Optional integer seed.
#[wasm_bindgen]
pub fn hybrid_noise(
    rows: u32,
    cols: u32,
    scale_factor: f64,
    octaves: u32,
    persistence: f64,
    lacunarity: f64,
    seed: Option<u32>,
) -> WasmGrid {
    let grid = nlmrs::hybrid_noise(
        rows as usize,
        cols as usize,
        scale_factor,
        octaves as usize,
        persistence,
        lacunarity,
        seed_from_js(seed),
    );
    grid_to_wasm(grid)
}

/// Value noise — interpolated lattice noise. Values in \[0, 1\).
///
/// @param rows         - Number of rows.
/// @param cols         - Number of columns.
/// @param scale_factor - Noise frequency; higher = more features (default 4.0).
/// @param seed         - Optional integer seed.
#[wasm_bindgen]
pub fn value_noise(rows: u32, cols: u32, scale_factor: f64, seed: Option<u32>) -> WasmGrid {
    let grid = nlmrs::value_noise(rows as usize, cols as usize, scale_factor, seed_from_js(seed));
    grid_to_wasm(grid)
}

/// Turbulence — fBm with absolute-value fold per octave. Values in \[0, 1\).
///
/// @param rows         - Number of rows.
/// @param cols         - Number of columns.
/// @param scale_factor - Base noise frequency (default 4.0).
/// @param octaves      - Number of noise layers (default 6).
/// @param persistence  - Amplitude scaling per octave (default 0.5).
/// @param lacunarity   - Frequency scaling per octave (default 2.0).
/// @param seed         - Optional integer seed.
#[wasm_bindgen]
pub fn turbulence(
    rows: u32,
    cols: u32,
    scale_factor: f64,
    octaves: u32,
    persistence: f64,
    lacunarity: f64,
    seed: Option<u32>,
) -> WasmGrid {
    let grid = nlmrs::turbulence(
        rows as usize,
        cols as usize,
        scale_factor,
        octaves as usize,
        persistence,
        lacunarity,
        seed_from_js(seed),
    );
    grid_to_wasm(grid)
}

/// Domain-warped Perlin noise — organic, swirling patterns. Values in \[0, 1\).
///
/// @param rows          - Number of rows.
/// @param cols          - Number of columns.
/// @param scale_factor  - Coordinate frequency (default 4.0).
/// @param warp_strength - Displacement magnitude (default 1.0).
/// @param seed          - Optional integer seed.
#[wasm_bindgen]
pub fn domain_warp(
    rows: u32,
    cols: u32,
    scale_factor: f64,
    warp_strength: f64,
    seed: Option<u32>,
) -> WasmGrid {
    let grid = nlmrs::domain_warp(
        rows as usize,
        cols as usize,
        scale_factor,
        warp_strength,
        seed_from_js(seed),
    );
    grid_to_wasm(grid)
}

/// Mosaic NLM — discrete Voronoi patch map. Values in \[0, 1\).
///
/// @param rows - Number of rows.
/// @param cols - Number of columns.
/// @param n    - Number of Voronoi seed points (default 200).
/// @param seed - Optional integer seed.
#[wasm_bindgen]
pub fn mosaic(rows: u32, cols: u32, n: u32, seed: Option<u32>) -> WasmGrid {
    let grid = nlmrs::mosaic(rows as usize, cols as usize, n as usize, seed_from_js(seed));
    grid_to_wasm(grid)
}

/// Rectangular cluster NLM — overlapping random rectangles. Values in \[0, 1\).
///
/// @param rows - Number of rows.
/// @param cols - Number of columns.
/// @param n    - Number of rectangles (default 200).
/// @param seed - Optional integer seed.
#[wasm_bindgen]
pub fn rectangular_cluster(rows: u32, cols: u32, n: u32, seed: Option<u32>) -> WasmGrid {
    let grid =
        nlmrs::rectangular_cluster(rows as usize, cols as usize, n as usize, seed_from_js(seed));
    grid_to_wasm(grid)
}

/// Binary percolation NLM. Values in {0.0, 1.0}.
///
/// @param rows - Number of rows.
/// @param cols - Number of columns.
/// @param p    - Habitat probability (0.0–1.0). Critical threshold ~0.593.
/// @param seed - Optional integer seed.
#[wasm_bindgen]
pub fn percolation(rows: u32, cols: u32, p: f64, seed: Option<u32>) -> WasmGrid {
    let grid = nlmrs::percolation(rows as usize, cols as usize, p, seed_from_js(seed));
    grid_to_wasm(grid)
}

/// Binary space partitioning NLM — hierarchical rectilinear partition. Values in [0, 1).
///
/// @param rows - Number of rows.
/// @param cols - Number of columns.
/// @param n    - Number of rectangles in the final partition (default 100).
/// @param seed - Optional integer seed.
#[wasm_bindgen]
pub fn binary_space_partitioning(rows: u32, cols: u32, n: u32, seed: Option<u32>) -> WasmGrid {
    let grid = nlmrs::binary_space_partitioning(
        rows as usize,
        cols as usize,
        n as usize,
        seed_from_js(seed),
    );
    grid_to_wasm(grid)
}

/// Cellular automaton NLM — binary cave-like patterns from birth/survival rules. Values in {0.0, 1.0}.
///
/// @param rows               - Number of rows.
/// @param cols               - Number of columns.
/// @param p                  - Initial alive probability (default 0.45).
/// @param iterations         - Number of rule applications (default 5).
/// @param birth_threshold    - Min live neighbours to birth a dead cell (default 5).
/// @param survival_threshold - Min live neighbours for a live cell to survive (default 4).
/// @param seed               - Optional integer seed.
#[wasm_bindgen]
pub fn cellular_automaton(
    rows: u32, cols: u32, p: f64,
    iterations: u32, birth_threshold: u32, survival_threshold: u32,
    seed: Option<u32>,
) -> WasmGrid {
    let grid = nlmrs::cellular_automaton(
        rows as usize, cols as usize, p,
        iterations as usize, birth_threshold as usize, survival_threshold as usize,
        seed_from_js(seed),
    );
    grid_to_wasm(grid)
}

/// Neighbourhood clustering NLM — iterative majority-vote patch clustering. Values in [0, 1).
///
/// @param rows       - Number of rows.
/// @param cols       - Number of columns.
/// @param k          - Number of distinct patch classes (default 5).
/// @param iterations - Number of majority-vote passes (default 10).
/// @param seed       - Optional integer seed.
#[wasm_bindgen]
pub fn neighbourhood_clustering(rows: u32, cols: u32, k: u32, iterations: u32, seed: Option<u32>) -> WasmGrid {
    let grid = nlmrs::neighbourhood_clustering(
        rows as usize, cols as usize, k as usize, iterations as usize, seed_from_js(seed),
    );
    grid_to_wasm(grid)
}

/// Spectral synthesis NLM — 1/f^beta noise generated in the frequency domain. Values in [0, 1).
///
/// @param rows - Number of rows.
/// @param cols - Number of columns.
/// @param beta - Spectral exponent: 0 = white noise, 1 = pink, 2 = brown/natural terrain (default 2.0).
/// @param seed - Optional integer seed.
#[wasm_bindgen]
pub fn spectral_synthesis(rows: u32, cols: u32, beta: f64, seed: Option<u32>) -> WasmGrid {
    let grid = nlmrs::spectral_synthesis(rows as usize, cols as usize, beta, seed_from_js(seed));
    grid_to_wasm(grid)
}

/// Gray-Scott reaction-diffusion NLM — Turing-pattern spots, stripes and labyrinths. Values in [0, 1).
///
/// @param rows       - Number of rows.
/// @param cols       - Number of columns.
/// @param iterations - Number of simulation steps (default 1000).
/// @param feed       - Feed rate for chemical A (default 0.055).
/// @param kill       - Kill rate for chemical B (default 0.062).
/// @param seed       - Optional integer seed.
#[wasm_bindgen]
pub fn reaction_diffusion(
    rows: u32, cols: u32, iterations: u32, feed: f64, kill: f64, seed: Option<u32>,
) -> WasmGrid {
    let grid = nlmrs::reaction_diffusion(
        rows as usize, cols as usize, iterations as usize, feed, kill, seed_from_js(seed),
    );
    grid_to_wasm(grid)
}

/// Eden growth model NLM — compact fractal blob grown from the centre. Values in {0.0, 1.0}.
///
/// @param rows - Number of rows.
/// @param cols - Number of columns.
/// @param n    - Number of cells to add to the cluster (default 2000).
/// @param seed - Optional integer seed.
#[wasm_bindgen]
pub fn eden_growth(rows: u32, cols: u32, n: u32, seed: Option<u32>) -> WasmGrid {
    let grid = nlmrs::eden_growth(rows as usize, cols as usize, n as usize, seed_from_js(seed));
    grid_to_wasm(grid)
}

/// Fractal Brownian surface NLM — parameterised by Hurst exponent. Values in [0, 1).
///
/// @param rows - Number of rows.
/// @param cols - Number of columns.
/// @param h    - Hurst exponent in (0, 1): 0 = rough, 1 = smooth (default 0.5).
/// @param seed - Optional integer seed.
#[wasm_bindgen]
pub fn fractal_brownian_surface(rows: u32, cols: u32, h: f64, seed: Option<u32>) -> WasmGrid {
    let grid = nlmrs::fractal_brownian_surface(rows as usize, cols as usize, h, seed_from_js(seed));
    grid_to_wasm(grid)
}

/// Elliptical landscape gradient centred at the grid midpoint. Values in [0, 1).
///
/// @param rows      - Number of rows.
/// @param cols      - Number of columns.
/// @param direction - Major-axis orientation in degrees [0, 360). Omit for random.
/// @param aspect    - Major-to-minor axis ratio (≥ 1.0). 1.0 = circular (default 1.0).
/// @param seed      - Optional integer seed.
#[wasm_bindgen]
pub fn landscape_gradient(
    rows: u32, cols: u32, direction: Option<f64>, aspect: f64, seed: Option<u32>,
) -> WasmGrid {
    let grid = nlmrs::landscape_gradient(
        rows as usize, cols as usize, direction, aspect, seed_from_js(seed),
    );
    grid_to_wasm(grid)
}

/// Diffusion-limited aggregation NLM — branching fractal cluster grown from the centre. Values in {0.0, 1.0}.
///
/// @param rows - Number of rows.
/// @param cols - Number of columns.
/// @param n    - Number of particles to release (default 2000).
/// @param seed - Optional integer seed.
#[wasm_bindgen]
pub fn diffusion_limited_aggregation(rows: u32, cols: u32, n: u32, seed: Option<u32>) -> WasmGrid {
    let grid = nlmrs::diffusion_limited_aggregation(
        rows as usize, cols as usize, n as usize, seed_from_js(seed),
    );
    grid_to_wasm(grid)
}

/// OpenSimplex noise NLM. Values in [0, 1).
///
/// @param rows  - Number of rows.
/// @param cols  - Number of columns.
/// @param scale - Coordinate frequency (default 4.0).
/// @param seed  - Optional integer seed.
#[wasm_bindgen]
pub fn simplex_noise(rows: u32, cols: u32, scale: f64, seed: Option<u32>) -> WasmGrid {
    let grid = nlmrs::simplex_noise(rows as usize, cols as usize, scale, seed_from_js(seed));
    grid_to_wasm(grid)
}

/// Invasion percolation NLM — lowest-weight boundary growth from centre. Values in {0.0, 1.0}.
///
/// @param rows - Number of rows.
/// @param cols - Number of columns.
/// @param n    - Number of cells to invade (default 2000).
/// @param seed - Optional integer seed.
#[wasm_bindgen]
pub fn invasion_percolation(rows: u32, cols: u32, n: u32, seed: Option<u32>) -> WasmGrid {
    let grid = nlmrs::invasion_percolation(
        rows as usize, cols as usize, n as usize, seed_from_js(seed),
    );
    grid_to_wasm(grid)
}

/// Sum of random Gaussian blob kernels. Values in [0, 1).
///
/// @param rows  - Number of rows.
/// @param cols  - Number of columns.
/// @param n     - Number of blob centres (default 50).
/// @param sigma - Gaussian width in cells (default 5.0).
/// @param seed  - Optional integer seed.
#[wasm_bindgen]
pub fn gaussian_blobs(rows: u32, cols: u32, n: u32, sigma: f64, seed: Option<u32>) -> WasmGrid {
    let grid = nlmrs::gaussian_blobs(
        rows as usize, cols as usize, n as usize, sigma, seed_from_js(seed),
    );
    grid_to_wasm(grid)
}

/// Ising model via Glauber dynamics. Binary values {0.0, 1.0}.
///
/// @param rows       - Number of rows.
/// @param cols       - Number of columns.
/// @param beta       - Inverse temperature (near 0.44 = critical point, default 0.4).
/// @param iterations - Number of sweeps (default 1000).
/// @param seed       - Optional integer seed.
#[wasm_bindgen]
pub fn ising_model(rows: u32, cols: u32, beta: f64, iterations: u32, seed: Option<u32>) -> WasmGrid {
    let grid = nlmrs::ising_model(
        rows as usize, cols as usize, beta, iterations as usize, seed_from_js(seed),
    );
    grid_to_wasm(grid)
}

// ── Post-processing ───────────────────────────────────────────────────────────

/// Quantise a grid into `n` equal-width classes.
///
/// Class `k` (0-indexed) is assigned output value `k / (n - 1)`,
/// evenly spacing the classes across \[0, 1\].
///
/// @param grid - A grid returned by a generator function.
/// @param n    - Number of classes (>= 1).
#[wasm_bindgen]
pub fn classify(grid: WasmGrid, n: u32) -> WasmGrid {
    let mut g = Grid { data: grid.data, rows: grid.rows as usize, cols: grid.cols as usize };
    nlmrs::classify(&mut g, n as usize);
    WasmGrid { rows: g.rows as u32, cols: g.cols as u32, data: g.data }
}

/// Apply a binary threshold to a grid.
///
/// Values strictly below `t` become 0.0; values at or above become 1.0.
///
/// @param grid - A grid returned by a generator function.
/// @param t    - Threshold value in \[0, 1\].
#[wasm_bindgen]
pub fn threshold(grid: WasmGrid, t: f64) -> WasmGrid {
    let mut g = Grid { data: grid.data, rows: grid.rows as usize, cols: grid.cols as usize };
    nlmrs::threshold(&mut g, t);
    WasmGrid { rows: g.rows as u32, cols: g.cols as u32, data: g.data }
}
