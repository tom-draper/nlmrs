use crate::array::rand_grid;
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
}
