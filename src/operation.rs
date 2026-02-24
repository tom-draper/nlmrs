use rand::Rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::grid::Grid;

pub fn max(grid: &Grid) -> f64 {
    min_and_max(grid).1
}

pub fn min(grid: &Grid) -> f64 {
    min_and_max(grid).0
}

pub fn min_and_max(grid: &Grid) -> (f64, f64) {
    grid.data
        .iter()
        .copied()
        .fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(mn, mx), v| (mn.min(v), mx.max(v)),
        )
}

/// Returns the value of a randomly chosen neighbour of (row, col).
fn nearest_neighbour(grid: &Grid, row: usize, col: usize, rng: &mut impl Rng) -> f64 {
    let mut options = [0.0f64; 4];
    let mut count = 0usize;
    if row + 1 < grid.rows { options[count] = grid[row + 1][col]; count += 1; }
    if row > 0             { options[count] = grid[row - 1][col]; count += 1; }
    if col + 1 < grid.cols { options[count] = grid[row][col + 1]; count += 1; }
    if col > 0             { options[count] = grid[row][col - 1]; count += 1; }
    options[rng.gen_range(0..count)]
}

pub fn interpolate(grid: &mut Grid, mask: &[usize], rng: &mut impl Rng) {
    let replacements: Vec<(usize, usize, f64)> = mask
        .iter()
        .map(|&idx| {
            let row = idx / grid.cols;
            let col = idx % grid.cols;
            (row, col, nearest_neighbour(grid, row, col, rng))
        })
        .collect();
    for (row, col, val) in replacements {
        grid[row][col] = val;
    }
}

pub fn scale(grid: &mut Grid) {
    let (min, max) = min_and_max(grid);
    let range = max - min;
    let scale_v = |v: &mut f64| {
        *v = if range == 0.0 { 0.5 } else { (*v - min) / range };
    };
    #[cfg(feature = "parallel")]
    grid.data.par_iter_mut().for_each(scale_v);
    #[cfg(not(feature = "parallel"))]
    grid.data.iter_mut().for_each(scale_v);
}

/// Euclidean distance transform using the separable Meijster algorithm — O(rows*cols).
///
/// Phase 1 (column passes) is sequential — column-major access into a row-major array
/// is non-contiguous and not easily parallelised without transposition overhead.
/// Phase 2 (row passes) is parallelised via rayon when the `parallel` feature is enabled.
pub fn euclidean_distance_transform(grid: &mut Grid) {
    let rows = grid.rows;
    let cols = grid.cols;

    if rows == 0 || cols == 0 {
        return;
    }

    let inf = f64::INFINITY;
    let mut g = vec![inf; rows * cols];

    // Phase 1: per-column vertical squared-distance to nearest zero — sequential.
    for j in 0..cols {
        g[j] = if grid[0][j] == 0.0 { 0.0 } else { inf };
        for i in 1..rows {
            g[i * cols + j] = if grid[i][j] == 0.0 {
                0.0
            } else if g[(i - 1) * cols + j] < inf {
                g[(i - 1) * cols + j] + 1.0
            } else {
                inf
            };
        }
        for i in (0..rows - 1).rev() {
            let below = g[(i + 1) * cols + j];
            if below < g[i * cols + j] {
                g[i * cols + j] = below + 1.0;
            }
        }
        for i in 0..rows {
            let v = g[i * cols + j];
            g[i * cols + j] = if v < inf { v * v } else { inf };
        }
    }

    // Phase 2: per-row parabola DT — rows are fully independent, parallelise with rayon.
    // Each row needs its own v/z scratch buffers.
    #[cfg(feature = "parallel")]
    let chunks = grid.data.par_chunks_mut(cols);
    #[cfg(not(feature = "parallel"))]
    let chunks = grid.data.chunks_mut(cols);
    chunks.enumerate().for_each(|(i, row_data)| {
            let row_g: &[f64] = &g[i * cols..(i + 1) * cols];

            let mut v = vec![0usize; cols];
            let mut z = vec![0.0f64; cols + 1];

            let mut k: usize = 0;
            v[0] = 0;
            z[0] = f64::NEG_INFINITY;
            z[1] = f64::INFINITY;

            for q in 1..cols {
                let gq = row_g[q];
                loop {
                    let vk = v[k];
                    let gvk = row_g[vk];
                    let s = ((gq + (q * q) as f64) - (gvk + (vk * vk) as f64))
                        / (2.0 * (q as f64 - vk as f64));
                    if s > z[k] {
                        k += 1;
                        v[k] = q;
                        z[k] = s;
                        z[k + 1] = f64::INFINITY;
                        break;
                    }
                    if k == 0 {
                        v[0] = q;
                        z[0] = f64::NEG_INFINITY;
                        z[1] = f64::INFINITY;
                        break;
                    }
                    k -= 1;
                }
            }

            k = 0;
            for q in 0..cols {
                while z[k + 1] < q as f64 {
                    k += 1;
                }
                let vk = v[k];
                let dist_sq = (q as f64 - vk as f64).powi(2) + row_g[vk];
                row_data[q] = if dist_sq < inf { dist_sq.sqrt() } else { 0.0 };
            }
        });
}

pub fn invert(grid: &mut Grid) {
    #[cfg(feature = "parallel")]
    grid.data.par_iter_mut().for_each(|v| *v = 1.0 - *v);
    #[cfg(not(feature = "parallel"))]
    grid.data.iter_mut().for_each(|v| *v = 1.0 - *v);
}

pub fn multiply(grid: &mut Grid, other: &Grid) {
    #[cfg(feature = "parallel")]
    grid.data.par_iter_mut().zip(other.data.par_iter()).for_each(|(v, &o)| *v *= o);
    #[cfg(not(feature = "parallel"))]
    grid.data.iter_mut().zip(other.data.iter()).for_each(|(v, &o)| *v *= o);
}

pub fn multiply_value(grid: &mut Grid, value: f64) {
    #[cfg(feature = "parallel")]
    grid.data.par_iter_mut().for_each(|v| *v *= value);
    #[cfg(not(feature = "parallel"))]
    grid.data.iter_mut().for_each(|v| *v *= value);
}

pub fn add(grid: &mut Grid, other: &Grid) {
    #[cfg(feature = "parallel")]
    grid.data.par_iter_mut().zip(other.data.par_iter()).for_each(|(v, &o)| *v += o);
    #[cfg(not(feature = "parallel"))]
    grid.data.iter_mut().zip(other.data.iter()).for_each(|(v, &o)| *v += o);
}

pub fn add_value(grid: &mut Grid, value: f64) {
    #[cfg(feature = "parallel")]
    grid.data.par_iter_mut().for_each(|v| *v += value);
    #[cfg(not(feature = "parallel"))]
    grid.data.iter_mut().for_each(|v| *v += value);
}

pub fn abs(grid: &mut Grid) {
    #[cfg(feature = "parallel")]
    grid.data.par_iter_mut().for_each(|v| *v = v.abs());
    #[cfg(not(feature = "parallel"))]
    grid.data.iter_mut().for_each(|v| *v = v.abs());
}
