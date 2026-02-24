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

/// Multi-source BFS nearest-neighbour fill (Manhattan-metric Voronoi).
///
/// Every cell with a non-zero value is treated as a labelled seed; all zero
/// cells are flooded in BFS order so each inherits the label of the nearest
/// seed.  Running time is O(rows × cols).
pub fn interpolate(grid: &mut Grid) {
    use std::collections::VecDeque;
    let cols = grid.cols;
    let rows = grid.rows;

    // Seed the queue with every labelled cell.
    let mut queue: VecDeque<usize> = grid
        .data
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v != 0.0 { Some(i) } else { None })
        .collect();

    while let Some(idx) = queue.pop_front() {
        let val = grid.data[idx];
        let row = idx / cols;
        let col = idx % cols;

        // Up
        if row > 0 {
            let ni = idx - cols;
            if grid.data[ni] == 0.0 {
                grid.data[ni] = val;
                queue.push_back(ni);
            }
        }
        // Down
        if row + 1 < rows {
            let ni = idx + cols;
            if grid.data[ni] == 0.0 {
                grid.data[ni] = val;
                queue.push_back(ni);
            }
        }
        // Left
        if col > 0 {
            let ni = idx - 1;
            if grid.data[ni] == 0.0 {
                grid.data[ni] = val;
                queue.push_back(ni);
            }
        }
        // Right
        if col + 1 < cols {
            let ni = idx + 1;
            if grid.data[ni] == 0.0 {
                grid.data[ni] = val;
                queue.push_back(ni);
            }
        }
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

/// Quantises each cell into one of `n` equal-width classes over [0, 1].
///
/// Class `k` (0-indexed) is assigned the output value `k / (n − 1)`,
/// evenly spacing the `n` classes across [0, 1]. Panics if `n == 0`.
pub fn classify(grid: &mut Grid, n: usize) {
    assert!(n >= 1, "n must be at least 1");
    let n_f = n as f64;
    let max_class = (n - 1) as f64;
    let op = |v: &mut f64| {
        let class = (*v * n_f).floor().min(max_class);
        *v = if n == 1 { 0.0 } else { class / max_class };
    };
    #[cfg(feature = "parallel")]
    grid.data.par_iter_mut().for_each(op);
    #[cfg(not(feature = "parallel"))]
    grid.data.iter_mut().for_each(op);
}

/// Maps every cell to `0.0` if its value is strictly below `t`, or `1.0` otherwise.
pub fn threshold(grid: &mut Grid, t: f64) {
    let op = |v: &mut f64| {
        *v = if *v < t { 0.0 } else { 1.0 };
    };
    #[cfg(feature = "parallel")]
    grid.data.par_iter_mut().for_each(op);
    #[cfg(not(feature = "parallel"))]
    grid.data.iter_mut().for_each(op);
}
