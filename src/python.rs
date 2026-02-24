use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
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
    Ok(())
}
