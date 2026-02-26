use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::Grid;

/// Convert a Grid into a 2-D numpy array of shape (rows, cols).
fn to_numpy<'py>(py: Python<'py>, grid: Grid) -> Bound<'py, PyArray2<f64>> {
    let rows = grid.rows;
    let cols = grid.cols;
    Array2::from_shape_vec((rows, cols), grid.data)
        .expect("grid shape mismatch")
        .into_pyarray_bound(py)
}

// ── Generators ──────────────────────────────────────────────────────────────

/// Spatially random NLM. Values in [0, 1).
#[pyfunction]
#[pyo3(signature = (rows, cols, seed=None))]
fn random(py: Python<'_>, rows: usize, cols: usize, seed: Option<u64>) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::random(rows, cols, seed));
    to_numpy(py, grid)
}

/// Random element nearest-neighbour NLM. Values in [0, 1).
///
/// Parameters
/// ----------
/// n : float
///     Number of seed elements to place.
#[pyfunction]
#[pyo3(signature = (rows, cols, n=50000.0, seed=None))]
fn random_element(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    n: f64,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::random_element(rows, cols, n, seed));
    to_numpy(py, grid)
}

/// Linear planar gradient. Values in [0, 1).
///
/// Parameters
/// ----------
/// direction : float, optional
///     Gradient direction in degrees [0, 360). Random if not supplied.
#[pyfunction]
#[pyo3(signature = (rows, cols, direction=None, seed=None))]
fn planar_gradient(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    direction: Option<f64>,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::planar_gradient(rows, cols, direction, seed));
    to_numpy(py, grid)
}

/// Symmetric edge gradient — zero at both edges, peak in the middle. Values in [0, 1).
///
/// Parameters
/// ----------
/// direction : float, optional
///     Gradient direction in degrees [0, 360). Random if not supplied.
#[pyfunction]
#[pyo3(signature = (rows, cols, direction=None, seed=None))]
fn edge_gradient(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    direction: Option<f64>,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::edge_gradient(rows, cols, direction, seed));
    to_numpy(py, grid)
}

/// Radial distance gradient from a random centre point. Values in [0, 1).
#[pyfunction]
#[pyo3(signature = (rows, cols, seed=None))]
fn distance_gradient(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::distance_gradient(rows, cols, seed));
    to_numpy(py, grid)
}

/// Sinusoidal wave gradient. Values in [0, 1).
///
/// Parameters
/// ----------
/// period : float
///     Wave period — smaller values produce larger waves.
/// direction : float, optional
///     Wave direction in degrees [0, 360). Random if not supplied.
#[pyfunction]
#[pyo3(signature = (rows, cols, period=2.5, direction=None, seed=None))]
fn wave_gradient(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    period: f64,
    direction: Option<f64>,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::wave_gradient(rows, cols, period, direction, seed));
    to_numpy(py, grid)
}

/// Diamond-square (midpoint displacement) fractal terrain. Values in [0, 1).
///
/// Parameters
/// ----------
/// h : float
///     Spatial autocorrelation — 0 = rough, 1 = smooth.
#[pyfunction]
#[pyo3(signature = (rows, cols, h=1.0, seed=None))]
fn midpoint_displacement(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    h: f64,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::midpoint_displacement(rows, cols, h, seed));
    to_numpy(py, grid)
}

/// Hill-grow NLM. Values in [0, 1).
///
/// Parameters
/// ----------
/// n : int
///     Number of iterations.
/// runaway : bool
///     If True, selection probability is proportional to current cell height,
///     causing hills to cluster. Uses a Fenwick tree for O(log n) sampling.
/// kernel : list[list[float]], optional
///     Convolution kernel (square, odd-sized). Defaults to a 3×3 diamond kernel.
/// only_grow : bool
///     If True the surface only accumulates, never shrinks.
#[pyfunction]
#[pyo3(signature = (rows, cols, n=10000, runaway=true, kernel=None, only_grow=false, seed=None))]
fn hill_grow(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    n: usize,
    runaway: bool,
    kernel: Option<Vec<Vec<f64>>>,
    only_grow: bool,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::hill_grow(rows, cols, n, runaway, kernel, only_grow, seed));
    to_numpy(py, grid)
}

/// Single-layer Perlin noise. Values in [0, 1).
///
/// Parameters
/// ----------
/// scale : float
///     Noise frequency — higher values produce more features per unit area.
#[pyfunction]
#[pyo3(signature = (rows, cols, scale=4.0, seed=None))]
fn perlin_noise(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    scale: f64,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::perlin_noise(rows, cols, scale, seed));
    to_numpy(py, grid)
}

/// Fractal Brownian motion — layered Perlin noise. Values in [0, 1).
///
/// Parameters
/// ----------
/// scale : float
///     Base noise frequency.
/// octaves : int
///     Number of noise layers to combine. More octaves = finer detail.
/// persistence : float
///     Amplitude scaling per octave (0.5 = each octave half as loud).
/// lacunarity : float
///     Frequency scaling per octave (2.0 = each octave twice as dense).
#[pyfunction]
#[pyo3(signature = (rows, cols, scale=4.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None))]
fn fbm_noise(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    scale: f64,
    octaves: usize,
    persistence: f64,
    lacunarity: f64,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid =
        py.allow_threads(|| crate::fbm_noise(rows, cols, scale, octaves, persistence, lacunarity, seed));
    to_numpy(py, grid)
}

/// Ridged multifractal noise. Values in [0, 1).
///
/// Parameters
/// ----------
/// scale : float
///     Base noise frequency.
/// octaves : int
///     Number of noise layers to combine.
/// persistence : float
///     Amplitude scaling per octave.
/// lacunarity : float
///     Frequency scaling per octave.
#[pyfunction]
#[pyo3(signature = (rows, cols, scale=4.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None))]
fn ridged_noise(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    scale: f64,
    octaves: usize,
    persistence: f64,
    lacunarity: f64,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid =
        py.allow_threads(|| crate::ridged_noise(rows, cols, scale, octaves, persistence, lacunarity, seed));
    to_numpy(py, grid)
}

/// Billow noise — rounded cloud- and hill-like patterns. Values in [0, 1).
///
/// Parameters
/// ----------
/// scale : float
///     Base noise frequency.
/// octaves : int
///     Number of noise layers to combine.
/// persistence : float
///     Amplitude scaling per octave.
/// lacunarity : float
///     Frequency scaling per octave.
#[pyfunction]
#[pyo3(signature = (rows, cols, scale=4.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None))]
fn billow_noise(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    scale: f64,
    octaves: usize,
    persistence: f64,
    lacunarity: f64,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid =
        py.allow_threads(|| crate::billow_noise(rows, cols, scale, octaves, persistence, lacunarity, seed));
    to_numpy(py, grid)
}

/// Worley (cellular) noise — territory / patch patterns. Values in [0, 1).
///
/// Parameters
/// ----------
/// scale : float
///     Seed-point frequency; higher values produce smaller cells.
#[pyfunction]
#[pyo3(signature = (rows, cols, scale=4.0, seed=None))]
fn worley_noise(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    scale: f64,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::worley_noise(rows, cols, scale, seed));
    to_numpy(py, grid)
}

/// Gaussian random field — spatially correlated noise. Values in [0, 1).
///
/// Parameters
/// ----------
/// sigma : float
///     Gaussian kernel standard deviation in cells (controls correlation length).
#[pyfunction]
#[pyo3(signature = (rows, cols, sigma=10.0, seed=None))]
fn gaussian_field(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    sigma: f64,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::gaussian_field(rows, cols, sigma, seed));
    to_numpy(py, grid)
}

/// Random cluster NLM via fault-line cuts. Values in [0, 1).
///
/// Parameters
/// ----------
/// n : int
///     Number of fault-line cuts. Higher values produce finer-grained clusters.
#[pyfunction]
#[pyo3(signature = (rows, cols, n=200, seed=None))]
fn random_cluster(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    n: usize,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::random_cluster(rows, cols, n, seed));
    to_numpy(py, grid)
}

/// Hybrid multifractal noise. Values in [0, 1).
///
/// Parameters
/// ----------
/// scale : float
///     Base noise frequency.
/// octaves : int
///     Number of noise layers to combine.
/// persistence : float
///     Amplitude scaling per octave.
/// lacunarity : float
///     Frequency scaling per octave.
#[pyfunction]
#[pyo3(signature = (rows, cols, scale=4.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None))]
fn hybrid_noise(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    scale: f64,
    octaves: usize,
    persistence: f64,
    lacunarity: f64,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid =
        py.allow_threads(|| crate::hybrid_noise(rows, cols, scale, octaves, persistence, lacunarity, seed));
    to_numpy(py, grid)
}

/// Value noise — interpolated lattice noise. Values in [0, 1).
///
/// Parameters
/// ----------
/// scale : float
///     Noise frequency — higher values produce more features per unit area.
#[pyfunction]
#[pyo3(signature = (rows, cols, scale=4.0, seed=None))]
fn value_noise(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    scale: f64,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::value_noise(rows, cols, scale, seed));
    to_numpy(py, grid)
}

/// Turbulence — fBm with absolute-value fold per octave. Values in [0, 1).
///
/// Parameters
/// ----------
/// scale : float
///     Base noise frequency.
/// octaves : int
///     Number of noise layers to combine.
/// persistence : float
///     Amplitude scaling per octave.
/// lacunarity : float
///     Frequency scaling per octave.
#[pyfunction]
#[pyo3(signature = (rows, cols, scale=4.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None))]
fn turbulence(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    scale: f64,
    octaves: usize,
    persistence: f64,
    lacunarity: f64,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid =
        py.allow_threads(|| crate::turbulence(rows, cols, scale, octaves, persistence, lacunarity, seed));
    to_numpy(py, grid)
}

/// Domain-warped Perlin noise — organic, swirling patterns. Values in [0, 1).
///
/// Parameters
/// ----------
/// scale : float
///     Coordinate frequency — higher values produce more features per unit area.
/// warp_strength : float
///     Displacement magnitude applied to sample coordinates (default 1.0).
#[pyfunction]
#[pyo3(signature = (rows, cols, scale=4.0, warp_strength=1.0, seed=None))]
fn domain_warp(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    scale: f64,
    warp_strength: f64,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::domain_warp(rows, cols, scale, warp_strength, seed));
    to_numpy(py, grid)
}

/// Mosaic NLM — discrete Voronoi patch map with flat-coloured regions. Values in [0, 1).
///
/// Parameters
/// ----------
/// n : int
///     Number of Voronoi seed points to place (default 200).
#[pyfunction]
#[pyo3(signature = (rows, cols, n=200, seed=None))]
fn mosaic(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    n: usize,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::mosaic(rows, cols, n, seed));
    to_numpy(py, grid)
}

/// Rectangular cluster NLM — overlapping random axis-aligned rectangles. Values in [0, 1).
///
/// Parameters
/// ----------
/// n : int
///     Number of rectangles to place (default 200).
#[pyfunction]
#[pyo3(signature = (rows, cols, n=200, seed=None))]
fn rectangular_cluster(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    n: usize,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::rectangular_cluster(rows, cols, n, seed));
    to_numpy(py, grid)
}

/// Binary percolation NLM. Values in {0.0, 1.0}.
///
/// Parameters
/// ----------
/// p : float
///     Probability a cell is habitat (0.0–1.0). The critical threshold for
///     spanning habitat clusters is ~0.593 (4-connectivity).
#[pyfunction]
#[pyo3(signature = (rows, cols, p=0.5, seed=None))]
fn percolation(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    p: f64,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::percolation(rows, cols, p, seed));
    to_numpy(py, grid)
}

/// Binary space partitioning NLM — hierarchical rectilinear partition. Values in [0, 1).
///
/// Parameters
/// ----------
/// n : int
///     Number of rectangles in the final partition (default 100).
/// Cellular automaton NLM. Binary values {0.0, 1.0}.
///
/// Parameters
/// ----------
/// rows : int
/// cols : int
/// p : float
///     Initial probability of a cell being alive (default 0.45).
/// iterations : int
///     Number of rule applications (default 5).
/// birth_threshold : int
///     Min live neighbours for a dead cell to become alive (default 5).
/// survival_threshold : int
///     Min live neighbours for a live cell to stay alive (default 4).
/// seed : int, optional
#[pyfunction]
#[pyo3(signature = (rows, cols, p=0.45, iterations=5, birth_threshold=5, survival_threshold=4, seed=None))]
fn cellular_automaton(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    p: f64,
    iterations: usize,
    birth_threshold: usize,
    survival_threshold: usize,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| {
        crate::cellular_automaton(rows, cols, p, iterations, birth_threshold, survival_threshold, seed)
    });
    to_numpy(py, grid)
}

#[pyfunction]
#[pyo3(signature = (rows, cols, n=100, seed=None))]
fn binary_space_partitioning(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    n: usize,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::binary_space_partitioning(rows, cols, n, seed));
    to_numpy(py, grid)
}

/// Neighbourhood clustering NLM. Values in [0, 1).
///
/// Parameters
/// ----------
/// rows : int
/// cols : int
/// k : int
///     Number of distinct patch classes (default 5).
/// iterations : int
///     Number of majority-vote passes (default 10). More iterations produce
///     larger, smoother patches.
/// seed : int, optional
#[pyfunction]
#[pyo3(signature = (rows, cols, k=5, iterations=10, seed=None))]
fn neighbourhood_clustering(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    k: usize,
    iterations: usize,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::neighbourhood_clustering(rows, cols, k, iterations, seed));
    to_numpy(py, grid)
}

/// Spectral synthesis NLM. Values in [0, 1).
///
/// Parameters
/// ----------
/// rows : int
/// cols : int
/// beta : float
///     Spectral exponent. 0 = white noise, 1 = pink noise,
///     2 = red/brown noise (natural terrain), higher = smoother.
/// seed : int, optional
#[pyfunction]
#[pyo3(signature = (rows, cols, beta=2.0, seed=None))]
fn spectral_synthesis(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    beta: f64,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::spectral_synthesis(rows, cols, beta, seed));
    to_numpy(py, grid)
}

/// Gray-Scott reaction-diffusion NLM. Values in [0, 1).
///
/// Parameters
/// ----------
/// iterations : int
///     Number of simulation steps (default 1000).
/// feed : float
///     Feed rate for chemical A. Controls pattern type (default 0.055).
/// kill : float
///     Kill rate for chemical B. Controls pattern type (default 0.062).
#[pyfunction]
#[pyo3(signature = (rows, cols, iterations=1000, feed=0.055, kill=0.062, seed=None))]
fn reaction_diffusion(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    iterations: usize,
    feed: f64,
    kill: f64,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::reaction_diffusion(rows, cols, iterations, feed, kill, seed));
    to_numpy(py, grid)
}

/// Eden growth model NLM. Binary values {0.0, 1.0}.
///
/// Parameters
/// ----------
/// n : int
///     Number of cells to add to the cluster (default 2000).
#[pyfunction]
#[pyo3(signature = (rows, cols, n=2000, seed=None))]
fn eden_growth(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    n: usize,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::eden_growth(rows, cols, n, seed));
    to_numpy(py, grid)
}

/// Fractal Brownian surface NLM parameterised by the Hurst exponent. Values in [0, 1).
///
/// Parameters
/// ----------
/// h : float
///     Hurst exponent in (0, 1). 0 = rough, 1 = smooth (default 0.5).
#[pyfunction]
#[pyo3(signature = (rows, cols, h=0.5, seed=None))]
fn fractal_brownian_surface(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    h: f64,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::fractal_brownian_surface(rows, cols, h, seed));
    to_numpy(py, grid)
}

/// Elliptical landscape gradient centred at the grid midpoint. Values in [0, 1).
///
/// Parameters
/// ----------
/// direction : float, optional
///     Major-axis orientation in degrees [0, 360). Random if not supplied.
/// aspect : float
///     Major-to-minor axis ratio (≥ 1.0). 1.0 = circular (default 1.0).
#[pyfunction]
#[pyo3(signature = (rows, cols, direction=None, aspect=1.0, seed=None))]
fn landscape_gradient(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    direction: Option<f64>,
    aspect: f64,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::landscape_gradient(rows, cols, direction, aspect, seed));
    to_numpy(py, grid)
}

/// Diffusion-limited aggregation NLM. Binary values {0.0, 1.0}.
///
/// Parameters
/// ----------
/// n : int
///     Number of particles to release (default 2000). More particles produce
///     a denser, more branching fractal cluster.
#[pyfunction]
#[pyo3(signature = (rows, cols, n=2000, seed=None))]
fn diffusion_limited_aggregation(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    n: usize,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::diffusion_limited_aggregation(rows, cols, n, seed));
    to_numpy(py, grid)
}

/// OpenSimplex noise NLM. Values in [0, 1).
///
/// Parameters
/// ----------
/// scale : float
///     Coordinate frequency (higher = more features per unit, default 4.0).
#[pyfunction]
#[pyo3(signature = (rows, cols, scale=4.0, seed=None))]
fn simplex_noise(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    scale: f64,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::simplex_noise(rows, cols, scale, seed));
    to_numpy(py, grid)
}

/// Invasion percolation NLM. Binary values {0.0, 1.0}.
///
/// Parameters
/// ----------
/// n : int
///     Number of cells to invade from the centre (default 2000).
#[pyfunction]
#[pyo3(signature = (rows, cols, n=2000, seed=None))]
fn invasion_percolation(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    n: usize,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::invasion_percolation(rows, cols, n, seed));
    to_numpy(py, grid)
}

/// Sum of random Gaussian blob kernels. Values in [0, 1).
///
/// Parameters
/// ----------
/// n : int
///     Number of blob centres (default 50).
/// sigma : float
///     Gaussian width in cells (default 5.0).
#[pyfunction]
#[pyo3(signature = (rows, cols, n=50, sigma=5.0, seed=None))]
fn gaussian_blobs(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    n: usize,
    sigma: f64,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::gaussian_blobs(rows, cols, n, sigma, seed));
    to_numpy(py, grid)
}

/// Ising model via Glauber dynamics. Binary values {0.0, 1.0}.
///
/// Parameters
/// ----------
/// beta : float
///     Inverse temperature. Near 0.44 = critical point (default 0.4).
/// iterations : int
///     Number of sweeps; each sweep = rows × cols spin-flip attempts (default 1000).
#[pyfunction]
#[pyo3(signature = (rows, cols, beta=0.4, iterations=1000, seed=None))]
fn ising_model(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    beta: f64,
    iterations: usize,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::ising_model(rows, cols, beta, iterations, seed));
    to_numpy(py, grid)
}

/// Voronoi distance field from random feature points. Values in [0, 1).
///
/// Parameters
/// ----------
/// n : int
///     Number of feature points to scatter (default 50).
#[pyfunction]
#[pyo3(signature = (rows, cols, n=50, seed=None))]
fn voronoi_distance(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    n: usize,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::voronoi_distance(rows, cols, n, seed));
    to_numpy(py, grid)
}

/// Superposition of sinusoidal plane waves. Values in [0, 1).
///
/// Parameters
/// ----------
/// waves : int
///     Number of sinusoidal waves to superpose (default 8).
#[pyfunction]
#[pyo3(signature = (rows, cols, waves=8, seed=None))]
fn sine_composite(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    waves: usize,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::sine_composite(rows, cols, waves, seed));
    to_numpy(py, grid)
}

/// Divergence-free curl-warped Perlin noise NLM. Values in [0, 1).
///
/// Parameters
/// ----------
/// scale : float
///     Coordinate frequency (higher = more features per unit, default 4.0).
#[pyfunction]
#[pyo3(signature = (rows, cols, scale=4.0, seed=None))]
fn curl_noise(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    scale: f64,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::curl_noise(rows, cols, scale, seed));
    to_numpy(py, grid)
}

/// Hydraulic erosion simulation on a random heightmap. Values in [0, 1).
///
/// Parameters
/// ----------
/// n : int
///     Number of erosion droplets to simulate (default 500).
#[pyfunction]
#[pyo3(signature = (rows, cols, n=500, seed=None))]
fn hydraulic_erosion(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    n: usize,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::hydraulic_erosion(rows, cols, n, seed));
    to_numpy(py, grid)
}

/// Levy flight random walk density map. Values in [0, 1).
///
/// Parameters
/// ----------
/// n : int
///     Number of flight steps (default 1000).
#[pyfunction]
#[pyo3(signature = (rows, cols, n=1000, seed=None))]
fn levy_flight(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    n: usize,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::levy_flight(rows, cols, n, seed));
    to_numpy(py, grid)
}

/// Poisson disk sampling inhibition pattern. Binary values {0.0, 1.0}.
///
/// Parameters
/// ----------
/// min_dist : float
///     Minimum distance in cells between any two sample points (default 5.0).
#[pyfunction]
#[pyo3(signature = (rows, cols, min_dist=5.0, seed=None))]
fn poisson_disk(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    min_dist: f64,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::poisson_disk(rows, cols, min_dist, seed));
    to_numpy(py, grid)
}

/// Gabor noise NLM. Values in [0, 1).
///
/// Parameters
/// ----------
/// scale : float
///     Controls carrier frequency and envelope width (higher = finer features, default 4.0).
/// n : int
///     Number of Gabor kernels to place (more = smoother result, default 500).
#[pyfunction]
#[pyo3(signature = (rows, cols, scale=4.0, n=500, seed=None))]
fn gabor_noise(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    scale: f64,
    n: usize,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::gabor_noise(rows, cols, scale, n, seed));
    to_numpy(py, grid)
}

/// Spot noise NLM — random oriented elliptical Gaussian blobs. Values in [0, 1).
///
/// Parameters
/// ----------
/// n : int
///     Number of spots to place (default 200).
#[pyfunction]
#[pyo3(signature = (rows, cols, n=200, seed=None))]
fn spot_noise(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    n: usize,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::spot_noise(rows, cols, n, seed));
    to_numpy(py, grid)
}

/// Anisotropic fBm NLM — noise stretched along a dominant axis. Values in [0, 1).
///
/// Parameters
/// ----------
/// scale : float
///     Base noise frequency along the primary axis (default 4.0).
/// octaves : int
///     Number of noise layers to combine (default 6).
/// direction : float
///     Orientation of elongation in degrees [0, 360) (default 45.0).
/// stretch : float
///     Compression ratio for the perpendicular axis ≥ 1.0 (default 4.0).
#[pyfunction]
#[pyo3(signature = (rows, cols, scale=4.0, octaves=6, direction=45.0, stretch=4.0, seed=None))]
fn anisotropic_noise(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    scale: f64,
    octaves: usize,
    direction: f64,
    stretch: f64,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::anisotropic_noise(rows, cols, scale, octaves, direction, stretch, seed));
    to_numpy(py, grid)
}

/// Seamlessly tileable Perlin noise NLM. Values in [0, 1).
///
/// Parameters
/// ----------
/// scale : float
///     Number of noise cycles per tile (higher = more features, default 4.0).
#[pyfunction]
#[pyo3(signature = (rows, cols, scale=4.0, seed=None))]
fn tiled_noise(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    scale: f64,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::tiled_noise(rows, cols, scale, seed));
    to_numpy(py, grid)
}

/// Brownian motion (Gaussian random walk) density NLM. Values in [0, 1).
///
/// Parameters
/// ----------
/// n : int
///     Number of walk steps (default 5000).
#[pyfunction]
#[pyo3(signature = (rows, cols, n=5000, seed=None))]
fn brownian_motion(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    n: usize,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::brownian_motion(rows, cols, n, seed));
    to_numpy(py, grid)
}

/// Forest fire NLM — Drossel-Schwabl CA burn-history map. Values in [0, 1).
///
/// Parameters
/// ----------
/// p_tree : float
///     Per-step probability an empty cell becomes a tree (default 0.02).
/// p_lightning : float
///     Per-step probability a tree ignites spontaneously (default 0.001).
/// iterations : int
///     Number of simulation steps (default 500).
#[pyfunction]
#[pyo3(signature = (rows, cols, p_tree=0.02, p_lightning=0.001, iterations=500, seed=None))]
fn forest_fire(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    p_tree: f64,
    p_lightning: f64,
    iterations: usize,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::forest_fire(rows, cols, p_tree, p_lightning, iterations, seed));
    to_numpy(py, grid)
}

/// River network NLM — D8 flow accumulation on fBm terrain. Values in [0, 1).
#[pyfunction]
#[pyo3(signature = (rows, cols, seed=None))]
fn river_network(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::river_network(rows, cols, seed));
    to_numpy(py, grid)
}

/// Hexagonal Voronoi NLM — BFS mosaic from a regular hexagonal seed lattice. Values in (0, 1].
///
/// Parameters
/// ----------
/// n : int
///     Approximate number of hexagonal cells (default 50).
#[pyfunction]
#[pyo3(signature = (rows, cols, n=50, seed=None))]
fn hexagonal_voronoi(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    n: usize,
    seed: Option<u64>,
) -> Bound<'_, PyArray2<f64>> {
    let grid = py.allow_threads(|| crate::hexagonal_voronoi(rows, cols, n, seed));
    to_numpy(py, grid)
}

// ── Post-processing ──────────────────────────────────────────────────────────

/// Quantise a grid into `n` equal-width classes.
///
/// Parameters
/// ----------
/// arr : numpy.ndarray
///     2-D float64 array with values in [0, 1], as returned by any generator.
/// n : int
///     Number of classes (>= 1). Class `k` maps to output value `k / (n - 1)`.
#[pyfunction]
fn classify<'py>(py: Python<'py>, arr: &Bound<'py, PyArray2<f64>>, n: usize) -> Bound<'py, PyArray2<f64>> {
    let (rows, cols, data) = {
        let ro = arr.readonly();
        let view = ro.as_array();
        let (r, c) = view.dim();
        (r, c, view.to_owned().into_raw_vec_and_offset().0)
    };
    let grid = py.allow_threads(|| {
        let mut g = Grid { data, rows, cols };
        crate::classify(&mut g, n);
        g
    });
    to_numpy(py, grid)
}

/// Apply a binary threshold to a grid.
///
/// Parameters
/// ----------
/// arr : numpy.ndarray
///     2-D float64 array with values in [0, 1].
/// t : float
///     Values strictly below `t` become 0.0; values at or above become 1.0.
#[pyfunction]
fn threshold<'py>(py: Python<'py>, arr: &Bound<'py, PyArray2<f64>>, t: f64) -> Bound<'py, PyArray2<f64>> {
    let (rows, cols, data) = {
        let ro = arr.readonly();
        let view = ro.as_array();
        let (r, c) = view.dim();
        (r, c, view.to_owned().into_raw_vec_and_offset().0)
    };
    let grid = py.allow_threads(|| {
        let mut g = Grid { data, rows, cols };
        crate::threshold(&mut g, t);
        g
    });
    to_numpy(py, grid)
}

// ── Module ───────────────────────────────────────────────────────────────────

/// Fast Neutral Landscape Model generation.
///
/// All functions return a 2-D numpy array of float64 with values in [0, 1].
/// Pass `seed` (int) for reproducible output; omit or pass `None` for random.
///
/// Example
/// -------
/// >>> import nlmrs
/// >>> import matplotlib.pyplot as plt
/// >>> grid = nlmrs.midpoint_displacement(200, 200, h=0.8, seed=42)
/// >>> plt.imshow(grid, cmap="terrain")
/// >>> plt.show()
#[pymodule]
fn nlmrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(random, m)?)?;
    m.add_function(wrap_pyfunction!(random_element, m)?)?;
    m.add_function(wrap_pyfunction!(planar_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(edge_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(distance_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(wave_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(midpoint_displacement, m)?)?;
    m.add_function(wrap_pyfunction!(hill_grow, m)?)?;
    m.add_function(wrap_pyfunction!(perlin_noise, m)?)?;
    m.add_function(wrap_pyfunction!(fbm_noise, m)?)?;
    m.add_function(wrap_pyfunction!(ridged_noise, m)?)?;
    m.add_function(wrap_pyfunction!(billow_noise, m)?)?;
    m.add_function(wrap_pyfunction!(worley_noise, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_field, m)?)?;
    m.add_function(wrap_pyfunction!(random_cluster, m)?)?;
    m.add_function(wrap_pyfunction!(hybrid_noise, m)?)?;
    m.add_function(wrap_pyfunction!(value_noise, m)?)?;
    m.add_function(wrap_pyfunction!(turbulence, m)?)?;
    m.add_function(wrap_pyfunction!(domain_warp, m)?)?;
    m.add_function(wrap_pyfunction!(mosaic, m)?)?;
    m.add_function(wrap_pyfunction!(rectangular_cluster, m)?)?;
    m.add_function(wrap_pyfunction!(percolation, m)?)?;
    m.add_function(wrap_pyfunction!(cellular_automaton, m)?)?;
    m.add_function(wrap_pyfunction!(binary_space_partitioning, m)?)?;
    m.add_function(wrap_pyfunction!(neighbourhood_clustering, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_synthesis, m)?)?;
    m.add_function(wrap_pyfunction!(diffusion_limited_aggregation, m)?)?;
    m.add_function(wrap_pyfunction!(reaction_diffusion, m)?)?;
    m.add_function(wrap_pyfunction!(eden_growth, m)?)?;
    m.add_function(wrap_pyfunction!(fractal_brownian_surface, m)?)?;
    m.add_function(wrap_pyfunction!(landscape_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(simplex_noise, m)?)?;
    m.add_function(wrap_pyfunction!(invasion_percolation, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_blobs, m)?)?;
    m.add_function(wrap_pyfunction!(ising_model, m)?)?;
    m.add_function(wrap_pyfunction!(voronoi_distance, m)?)?;
    m.add_function(wrap_pyfunction!(sine_composite, m)?)?;
    m.add_function(wrap_pyfunction!(curl_noise, m)?)?;
    m.add_function(wrap_pyfunction!(hydraulic_erosion, m)?)?;
    m.add_function(wrap_pyfunction!(levy_flight, m)?)?;
    m.add_function(wrap_pyfunction!(poisson_disk, m)?)?;
    m.add_function(wrap_pyfunction!(gabor_noise, m)?)?;
    m.add_function(wrap_pyfunction!(spot_noise, m)?)?;
    m.add_function(wrap_pyfunction!(anisotropic_noise, m)?)?;
    m.add_function(wrap_pyfunction!(tiled_noise, m)?)?;
    m.add_function(wrap_pyfunction!(brownian_motion, m)?)?;
    m.add_function(wrap_pyfunction!(forest_fire, m)?)?;
    m.add_function(wrap_pyfunction!(river_network, m)?)?;
    m.add_function(wrap_pyfunction!(hexagonal_voronoi, m)?)?;
    m.add_function(wrap_pyfunction!(classify, m)?)?;
    m.add_function(wrap_pyfunction!(threshold, m)?)?;
    Ok(())
}
