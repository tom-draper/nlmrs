use crate::grid::Grid;
use crate::operation::{euclidean_distance_transform, invert, scale};
use super::make_rng;
use rand::Rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
    if rows == 0 || cols == 0 {
        return Grid::new(rows, cols);
    }
    let mut rng = make_rng(seed);
    // All cells start as non-seed (infinity); one random cell is the source (0.0).
    let data = vec![f64::INFINITY; rows * cols];
    let mut grid = Grid { data, rows, cols };
    let r = rng.gen_range(0..rows);
    let c = rng.gen_range(0..cols);
    grid[r][c] = 0.0;
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
pub fn wave_gradient(
    rows: usize,
    cols: usize,
    period: f64,
    direction: Option<f64>,
    seed: Option<u64>,
) -> Grid {
    let mut grid = planar_gradient(rows, cols, direction, seed);
    #[cfg(feature = "parallel")]
    grid.data.par_iter_mut().for_each(|v| *v = (*v * 2. * std::f64::consts::PI * period).sin());
    #[cfg(not(feature = "parallel"))]
    grid.data.iter_mut().for_each(|v| *v = (*v * 2. * std::f64::consts::PI * period).sin());
    scale(&mut grid);
    grid
}

/// Returns an elliptical landscape gradient centred at the grid midpoint. Values in [0, 1).
///
/// The gradient decreases radially from the centre with an elliptical falloff.
/// The `direction` angle orients the major axis of the ellipse, and `aspect`
/// controls how elongated it is.
///
/// # Arguments
///
/// * `rows`      - Number of rows.
/// * `cols`      - Number of columns.
/// * `direction` - Orientation of the major axis in degrees [0, 360). `None` picks a random direction.
/// * `aspect`    - Ratio of major to minor axis length (≥ 1.0). 1.0 = circular.
/// * `seed`      - Optional RNG seed (used when `direction` is `None`).
pub fn landscape_gradient(
    rows: usize,
    cols: usize,
    direction: Option<f64>,
    aspect: f64,
    seed: Option<u64>,
) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let mut rng = make_rng(seed);
    let angle = direction.unwrap_or_else(|| rng.gen::<f64>() * 360.0).to_radians();
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let aspect = aspect.max(1.0);

    let row_scale = if rows > 1 { (rows - 1) as f64 } else { 1.0 };
    let col_scale = if cols > 1 { (cols - 1) as f64 } else { 1.0 };

    let fill = |(idx, v): (usize, &mut f64)| {
        let i = idx / cols;
        let j = idx % cols;
        let ny = i as f64 / row_scale - 0.5;
        let nx = j as f64 / col_scale - 0.5;
        // Rotate into ellipse axes
        let rx = nx * cos_a + ny * sin_a;
        let ry = -nx * sin_a + ny * cos_a;
        // Major axis (rx) is `aspect` times wider than minor axis (ry)
        *v = ((rx / aspect).powi(2) + ry.powi(2)).sqrt();
    };

    let mut data = vec![0.0f64; rows * cols];
    #[cfg(feature = "parallel")]
    data.par_iter_mut().enumerate().for_each(fill);
    #[cfg(not(feature = "parallel"))]
    data.iter_mut().enumerate().for_each(fill);

    let mut grid = Grid { data, rows, cols };
    scale(&mut grid);
    invert(&mut grid);
    grid
}

/// Returns a concentric rings NLM with values ranging [0, 1).
///
/// Concentric sinusoidal bands radiate outward from the grid centre.
/// `frequency` controls how many full rings span the grid radius.
///
/// # Arguments
///
/// * `rows`      - Number of rows.
/// * `cols`      - Number of columns.
/// * `frequency` - Number of concentric ring pairs across the grid radius.
/// * `seed`      - Optional RNG seed (unused; provided for API consistency).
pub fn concentric_rings(rows: usize, cols: usize, frequency: f64, _seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let cx = (cols as f64 - 1.0) / 2.0;
    let cy = (rows as f64 - 1.0) / 2.0;
    let max_r = cx.hypot(cy).max(1.0);

    let data: Vec<f64> = (0..rows * cols)
        .map(|idx| {
            let dx = (idx % cols) as f64 - cx;
            let dy = (idx / cols) as f64 - cy;
            let r = dx.hypot(dy) / max_r;
            (r * frequency * std::f64::consts::PI).sin() * 0.5 + 0.5
        })
        .collect();

    let mut result = Grid { data, rows, cols };
    scale(&mut result);
    result
}

/// Returns a checkerboard NLM with binary values {0.0, 1.0}.
///
/// Deterministic alternating pattern of axis-aligned squares with side
/// length `scale` cells. A canonical control landscape for ecological
/// studies and spatial autocorrelation analysis.
///
/// # Arguments
///
/// * `rows`  - Number of rows.
/// * `cols`  - Number of columns.
/// * `scale` - Side length of each square in cells (≥ 1).
/// * `seed`  - Optional RNG seed (unused; provided for API consistency).
pub fn checkerboard(rows: usize, cols: usize, scale: usize, _seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let s = scale.max(1);
    let data: Vec<f64> = (0..rows * cols)
        .map(|idx| if ((idx / cols) / s + (idx % cols) / s) % 2 == 0 { 0.0 } else { 1.0 })
        .collect();
    Grid { data, rows, cols }
}

/// Returns a spiral gradient NLM with values ranging [0, 1).
///
/// Values increase along an Archimedean spiral radiating outward from the
/// centre of the grid. `turns` controls how many full rotations span the
/// grid radius — higher values produce tighter, more closely-wound spirals.
///
/// # Arguments
///
/// * `rows`  - Number of rows.
/// * `cols`  - Number of columns.
/// * `turns` - Number of spiral rotations across the grid radius.
/// * `seed`  - Optional RNG seed (unused; provided for API consistency).
pub fn spiral_gradient(rows: usize, cols: usize, turns: f64, _seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let cx = (cols as f64 - 1.0) / 2.0;
    let cy = (rows as f64 - 1.0) / 2.0;
    let max_r = cx.hypot(cy).max(1.0);

    let data: Vec<f64> = (0..rows * cols)
        .map(|idx| {
            let i = idx / cols;
            let j = idx % cols;
            let dx = j as f64 - cx;
            let dy = i as f64 - cy;
            let r = dx.hypot(dy) / max_r;
            let theta = dy.atan2(dx); // [-π, π]
            let theta_norm = theta / (2.0 * std::f64::consts::PI) + 0.5; // [0, 1]
            ((r * turns + theta_norm) % 1.0 + 1.0) % 1.0
        })
        .collect();

    let mut result = Grid { data, rows, cols };
    scale(&mut result);
    result
}

/// Returns a radial sweep NLM with values ranging [0, 1).
///
/// Each cell's value is the normalised clockwise angle from the grid centre,
/// mapped from `atan2(dy, dx)` into `[0, 1)`. Produces a smooth rotation
/// field useful as a directional covariate or combined with radial algorithms.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `seed` - Optional RNG seed (unused; provided for API consistency).
pub fn radial_sweep(rows: usize, cols: usize, _seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let cx = (cols as f64 - 1.0) / 2.0;
    let cy = (rows as f64 - 1.0) / 2.0;

    let data: Vec<f64> = (0..rows * cols)
        .map(|idx| {
            let dx = (idx % cols) as f64 - cx;
            let dy = (idx / cols) as f64 - cy;
            (dy.atan2(dx) + std::f64::consts::PI) / (2.0 * std::f64::consts::PI)
        })
        .collect();

    let mut result = Grid { data, rows, cols };
    scale(&mut result);
    result
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

    // ── landscape_gradient ────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(50, 50)]
    fn test_landscape_gradient(#[case] rows: usize, #[case] cols: usize) {
        let grid = landscape_gradient(rows, cols, Some(0.0), 1.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_landscape_gradient_seeded_determinism() {
        let a = landscape_gradient(50, 50, None, 2.0, Some(42));
        let b = landscape_gradient(50, 50, None, 2.0, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── concentric_rings ──────────────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_concentric_rings(#[case] rows: usize, #[case] cols: usize) {
        let grid = concentric_rings(rows, cols, 4.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    // ── checkerboard ──────────────────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_checkerboard(#[case] rows: usize, #[case] cols: usize) {
        let grid = checkerboard(rows, cols, 5, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    // ── spiral_gradient ───────────────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(2, 1)]
    #[case(3, 2)]
    #[case(5, 5)]
    #[case(10, 10)]
    #[case(100, 100)]
    #[case(500, 1000)]
    fn test_spiral_gradient(#[case] rows: usize, #[case] cols: usize) {
        let grid = spiral_gradient(rows, cols, 3.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    // ── radial_sweep ──────────────────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_radial_sweep(#[case] rows: usize, #[case] cols: usize) {
        let grid = radial_sweep(rows, cols, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }
}
