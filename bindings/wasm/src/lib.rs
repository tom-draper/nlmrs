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
