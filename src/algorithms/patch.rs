use crate::array::{diamond_square, rand_grid, rand_sub_grid};
use crate::grid::Grid;
use crate::operation::{interpolate, scale};
use super::{make_rng, perlin_seed};
use rand::Rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ── Private helpers ──────────────────────────────────────────────────────────

/// 4-point Laplacian with periodic (toroidal) boundary conditions.
fn laplacian_periodic(v: &[f64], i: usize, j: usize, rows: usize, cols: usize) -> f64 {
    let get = |ri: isize, ci: isize| -> f64 {
        v[ri.rem_euclid(rows as isize) as usize * cols
            + ci.rem_euclid(cols as isize) as usize]
    };
    let ri = i as isize;
    let ci = j as isize;
    get(ri - 1, ci) + get(ri + 1, ci) + get(ri, ci - 1) + get(ri, ci + 1)
        - 4.0 * get(ri, ci)
}

/// True if point `d` lies inside the circumcircle of counter-clockwise triangle (a, b, c).
fn circumcircle_contains(
    ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64,
    dx: f64, dy: f64,
) -> bool {
    let ax = ax - dx; let ay = ay - dy;
    let bx = bx - dx; let by = by - dy;
    let cx = cx - dx; let cy = cy - dy;
    let det = ax * (by * (cx * cx + cy * cy) - cy * (bx * bx + by * by))
            - ay * (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by))
            + (ax * ax + ay * ay) * (bx * cy - by * cx);
    det > 0.0
}

/// True if point `p` lies inside or on the edge of triangle (a, b, c).
fn point_in_tri(
    px: f64, py: f64,
    ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64,
) -> bool {
    let sign = |p1x: f64, p1y: f64, p2x: f64, p2y: f64, p3x: f64, p3y: f64| -> f64 {
        (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y)
    };
    let d1 = sign(px, py, ax, ay, bx, by);
    let d2 = sign(px, py, bx, by, cx, cy);
    let d3 = sign(px, py, cx, cy, ax, ay);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

/// Bowyer-Watson incremental Delaunay triangulation.
/// Returns triangles as index triples into `pts`.
fn bowyer_watson(pts: &[(f64, f64)]) -> Vec<[usize; 3]> {
    let n = pts.len();
    if n < 3 { return vec![]; }

    // Extend point list with super-triangle vertices.
    let mut all: Vec<(f64, f64)> = pts.to_vec();
    all.push((-10.0, -10.0));    // n
    all.push((10.0,  -10.0));    // n+1
    all.push((0.0,   10.0));     // n+2

    let mut tris: Vec<[usize; 3]> = vec![[n, n + 1, n + 2]];

    for (pi, &(px, py)) in pts.iter().enumerate() {
        let (bad, good): (Vec<[usize; 3]>, Vec<[usize; 3]>) = tris.into_iter().partition(|&[a, b, c]| {
            let (ax, ay) = all[a]; let (bx, by) = all[b]; let (cx, cy) = all[c];
            circumcircle_contains(ax, ay, bx, by, cx, cy, px, py)
        });

        // Boundary edges of the polygonal hole.
        let mut boundary: Vec<[usize; 2]> = Vec::new();
        for &tri in &bad {
            let edges = [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]];
            for edge in edges {
                let shared = bad.iter().any(|other| {
                    *other != tri && other.contains(&edge[0]) && other.contains(&edge[1])
                });
                if !shared { boundary.push(edge); }
            }
        }

        tris = good;
        for edge in boundary {
            tris.push([edge[0], edge[1], pi]);
        }
    }

    tris.retain(|&[a, b, c]| a < n && b < n && c < n);
    tris
}

/// Returns a spatially random NLM with values ranging [0, 1).
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn random(rows: usize, cols: usize, seed: Option<u64>) -> Grid {
    let mut rng = make_rng(seed);
    rand_grid(rows, cols, &mut rng)
}

/// Returns a random element nearest-neighbour NLM with values ranging [0, 1).
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n` - Number of labelled seed elements to place.
/// * `seed` - Optional RNG seed for reproducible results.
///
/// Implementation ported from NLMpy.
pub fn random_element(rows: usize, cols: usize, n: f64, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }

    let mut rng = make_rng(seed);
    let mut grid = Grid::new(rows, cols);

    // Track the last successfully placed label to avoid an O(rows*cols) max() scan
    // every iteration (which would make the whole function O(n * rows * cols)).
    let mut last_label = 0.0f64;
    let mut i: f64 = 1.;
    while last_label < n && i < n {
        let row = rng.gen_range(0..rows);
        let col = rng.gen_range(0..cols);
        if grid[row][col] == 0. {
            grid[row][col] = i;
            last_label = i;
        }
        i += 1.;
    }

    interpolate(&mut grid);
    scale(&mut grid);

    grid
}

/// Returns a midpoint displacement NLM with values ranging [0, 1).
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `h` - Controls the spatial autocorrelation in element values.
/// * `seed` - Optional RNG seed for reproducible results.
///
/// Implementation ported from NLMpy.
pub fn midpoint_displacement(rows: usize, cols: usize, h: f64, seed: Option<u64>) -> Grid {
    let max_dim = rows.max(cols);
    if max_dim == 0 {
        return Grid::new(0, 0);
    }
    let n = ((max_dim - 1) as f64).log2().ceil() as u32;
    let dim = usize::pow(2, n) + 1;

    let mut rng = make_rng(seed);
    let mut surface = diamond_square(dim, h, &mut rng);
    surface = rand_sub_grid(surface, rows, cols, &mut rng);

    scale(&mut surface);
    surface
}

/// Returns a spatially correlated Gaussian random field NLM with values in [0, 1).
///
/// Generates white noise then applies a separable Gaussian blur, so nearby
/// cells are correlated over a distance roughly proportional to `sigma` cells.
/// `sigma` maps directly to ecological correlation length.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `sigma` - Standard deviation of the Gaussian kernel in cells.  Higher
///   values produce larger, smoother patches.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn gaussian_field(rows: usize, cols: usize, sigma: f64, seed: Option<u64>) -> Grid {
    let mut rng = make_rng(seed);

    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }

    let mut grid = rand_grid(rows, cols, &mut rng);

    if sigma <= 0.0 {
        scale(&mut grid);
        return grid;
    }

    // Build a 1-D normalised Gaussian kernel with radius = ceil(3σ).
    let radius = (3.0 * sigma).ceil() as usize;
    let kernel: Vec<f64> = {
        let size = 2 * radius + 1;
        let mut k = vec![0.0f64; size];
        let mut sum = 0.0f64;
        for i in 0..size {
            let x = i as f64 - radius as f64;
            k[i] = (-0.5 * (x / sigma).powi(2)).exp();
            sum += k[i];
        }
        k.iter_mut().for_each(|v| *v /= sum);
        k
    };

    // Horizontal pass — row-wise, cache-friendly.
    let mut row_conv = Grid::new(rows, cols);
    {
        let fill = |(i, out_row): (usize, &mut [f64])| {
            let src_row = &grid[i];
            for j in 0..cols {
                let mut val = 0.0f64;
                for (k, &w) in kernel.iter().enumerate() {
                    let jj = (j as i64 + k as i64 - radius as i64)
                        .clamp(0, cols as i64 - 1) as usize;
                    val += src_row[jj] * w;
                }
                out_row[j] = val;
            }
        };
        #[cfg(feature = "parallel")]
        row_conv.data.par_chunks_mut(cols).enumerate().for_each(fill);
        #[cfg(not(feature = "parallel"))]
        row_conv.data.chunks_mut(cols).enumerate().for_each(fill);
    }

    // Vertical pass.
    let mut result = Grid::new(rows, cols);
    {
        let fill = |(i, out_row): (usize, &mut [f64])| {
            for j in 0..cols {
                let mut val = 0.0f64;
                for (k, &w) in kernel.iter().enumerate() {
                    let ii = (i as i64 + k as i64 - radius as i64)
                        .clamp(0, rows as i64 - 1) as usize;
                    val += row_conv[ii][j] * w;
                }
                out_row[j] = val;
            }
        };
        #[cfg(feature = "parallel")]
        result.data.par_chunks_mut(cols).enumerate().for_each(fill);
        #[cfg(not(feature = "parallel"))]
        result.data.chunks_mut(cols).enumerate().for_each(fill);
    }

    scale(&mut result);
    result
}

/// Returns a random cluster NLM with values ranging [0, 1).
///
/// Applies `n` random fault-line cuts; each cut adds +1 to all cells on one
/// side and −1 on the other.  After all cuts the accumulated field is scaled
/// to [0, 1], producing spatially clustered landscapes with the linear
/// structural elements characteristic of geological fault patterns.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n` - Number of fault-line cuts (higher = finer-grained clustering).
/// * `seed` - Optional RNG seed for reproducible results.
pub fn random_cluster(rows: usize, cols: usize, n: usize, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }

    let mut rng = make_rng(seed);

    // Pre-generate all cuts so the per-cell inner loop has no mutable state.
    // Each cut: (px, py, sin θ, cos θ) where θ ∈ [0, π).
    let cuts: Vec<(f64, f64, f64, f64)> = (0..n)
        .map(|_| {
            let theta: f64 = rng.gen_range(0.0..std::f64::consts::PI);
            let px: f64 = rng.gen();
            let py: f64 = rng.gen();
            (px, py, theta.sin(), theta.cos())
        })
        .collect();

    let mut grid = Grid::new(rows, cols);
    let inv_rows = 1.0 / rows as f64;
    let inv_cols = 1.0 / cols as f64;

    let fill = |(k, v): (usize, &mut f64)| {
        let row = k / cols;
        let col = k % cols;
        let x = col as f64 * inv_cols;
        let y = row as f64 * inv_rows;
        *v = cuts
            .iter()
            .map(|&(px, py, sin_t, cos_t)| {
                if (x - px) * sin_t + (y - py) * cos_t > 0.0 { 1.0 } else { -1.0 }
            })
            .sum::<f64>();
    };
    #[cfg(feature = "parallel")]
    grid.data.par_iter_mut().enumerate().for_each(fill);
    #[cfg(not(feature = "parallel"))]
    grid.data.iter_mut().enumerate().for_each(fill);

    scale(&mut grid);
    grid
}

/// Returns a mosaic (discrete Voronoi) NLM with values ranging [0, 1).
///
/// Places `n` random seed cells each with a unique random float value, then
/// fills every remaining cell with the value of its nearest seed via BFS.
/// The result is a flat-coloured patch map — all cells within a patch share
/// the same value, unlike `random_element` where values grade between seeds.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n` - Number of Voronoi seed points to place.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn mosaic(rows: usize, cols: usize, n: usize, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }

    let mut rng = make_rng(seed);
    let mut grid = Grid::new(rows, cols);

    for _ in 0..n {
        let row = rng.gen_range(0..rows);
        let col = rng.gen_range(0..cols);
        if grid[row][col] == 0.0 {
            // Use (0, 1) range but avoid 0.0 (reserved for "unlabelled")
            grid[row][col] = rng.gen::<f64>() * 0.999 + 0.001;
        }
    }

    // BFS fills every unlabelled cell with the nearest seed's value.
    // Values are already in (0.001, 1.0) — no scale() needed.
    interpolate(&mut grid);
    grid
}

/// Returns a rectangular cluster NLM with values ranging [0, 1).
///
/// Places `n` random axis-aligned rectangles and accumulates +1 per cell for
/// each overlapping rectangle. The result is scaled to [0, 1], producing
/// patch-like landscapes with rectilinear boundaries.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n` - Number of rectangles to place.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn rectangular_cluster(rows: usize, cols: usize, n: usize, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }

    let mut rng = make_rng(seed);
    let mut grid = Grid::new(rows, cols);
    let max_side = (rows.max(cols) / 3).max(1);

    for _ in 0..n {
        let r0 = rng.gen_range(0..rows);
        let c0 = rng.gen_range(0..cols);
        let r1 = (r0 + rng.gen_range(1..=max_side)).min(rows);
        let c1 = (c0 + rng.gen_range(1..=max_side)).min(cols);
        for i in r0..r1 {
            for j in c0..c1 {
                grid[i][j] += 1.0;
            }
        }
    }

    scale(&mut grid);
    grid
}

/// Returns a binary percolation NLM with values in {0.0, 1.0}.
///
/// Each cell is independently set to 1.0 (habitat) with probability `p` and
/// 0.0 (matrix) with probability `1 - p`.  As `p` approaches the critical
/// percolation threshold (~0.593 for 4-connectivity) habitat clusters coalesce
/// and span the landscape, making this the canonical NLM for studying
/// percolation theory and habitat connectivity.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `p` - Probability that a cell is habitat (0.0–1.0).
/// * `seed` - Optional RNG seed for reproducible results.
///
/// Based on: Gardner et al. (1987). Neutral models for the analysis of
/// broad-scale landscape pattern. *Landscape Ecology* 1(1):19–28.
pub fn percolation(rows: usize, cols: usize, p: f64, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let mut rng = make_rng(seed);
    let data = (0..rows * cols)
        .map(|_| if rng.gen::<f64>() < p { 1.0 } else { 0.0 })
        .collect();
    Grid { data, rows, cols }
}

/// Returns a binary space partitioning (BSP) NLM with values in [0, 1).
///
/// Recursively splits the grid into non-overlapping axis-aligned rectangles.
/// At each step the largest remaining rectangle is split along its longest
/// dimension at a random position. Once `n` rectangles exist each is assigned
/// a unique random float value, producing a hierarchically-nested rectilinear
/// partition.  Unlike `rectangular_cluster` (overlapping accumulation), BSP
/// produces a complete, non-overlapping partition of the grid — ideal for
/// modelling human-dominated agricultural or urban landscapes.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n` - Number of rectangles in the final partition.
/// * `seed` - Optional RNG seed for reproducible results.
///
/// Based on: Etherington, Morgan & O'Sullivan (2022). Binary space
/// partitioning generates hierarchical and rectilinear neutral landscape
/// models suitable for human-dominated landscapes. *Landscape Ecology*
/// 37:1761–1769.
pub fn binary_space_partitioning(rows: usize, cols: usize, n: usize, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let mut rng = make_rng(seed);
    let mut grid = Grid::new(rows, cols);

    // (row_start, col_start, row_end, col_end)
    let mut rects: Vec<(usize, usize, usize, usize)> = vec![(0, 0, rows, cols)];

    while rects.len() < n.max(1) {
        // Always split the largest rectangle for more uniform patch sizes.
        let idx = rects
            .iter()
            .enumerate()
            .max_by_key(|(_, &(r0, c0, r1, c1))| (r1 - r0) * (c1 - c0))
            .map(|(i, _)| i)
            .unwrap();

        let (r0, c0, r1, c1) = rects[idx];
        let height = r1 - r0;
        let width = c1 - c0;

        if height <= 1 && width <= 1 {
            break; // All patches are single cells; can't split further.
        }

        if height >= width && height > 1 {
            let split = r0 + rng.gen_range(1..height);
            rects[idx] = (r0, c0, split, c1);
            rects.push((split, c0, r1, c1));
        } else {
            let split = c0 + rng.gen_range(1..width);
            rects[idx] = (r0, c0, r1, split);
            rects.push((r0, split, r1, c1));
        }
    }

    // Assign a unique random float to every leaf rectangle.
    for (r0, c0, r1, c1) in rects {
        let val: f64 = rng.gen();
        for r in r0..r1 {
            for c in c0..c1 {
                grid[r][c] = val;
            }
        }
    }

    grid
}

/// Returns a neighbourhood clustering NLM with values ranging [0, 1).
///
/// Initialises a grid with `k` randomly assigned classes then repeatedly
/// applies a majority-vote rule: each cell adopts the most common class among
/// its 3×3 Moore neighbourhood. After `iterations` passes the discrete classes
/// are mapped to evenly spaced values in [0, 1), producing smooth organic
/// patch regions.
///
/// # Arguments
///
/// * `rows`       - Number of rows.
/// * `cols`       - Number of columns.
/// * `k`          - Number of distinct patch classes (≥ 2).
/// * `iterations` - Number of majority-vote passes (more = larger, smoother patches).
/// * `seed`       - Optional RNG seed for reproducible results.
pub fn neighbourhood_clustering(
    rows: usize,
    cols: usize,
    k: usize,
    iterations: usize,
    seed: Option<u64>,
) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }

    let k = k.max(2);
    let mut rng = make_rng(seed);

    let mut classes: Vec<usize> = (0..rows * cols).map(|_| rng.gen_range(0..k)).collect();
    let mut next = vec![0usize; rows * cols];

    for _ in 0..iterations {
        let fill = |(idx, out): (usize, &mut usize)| {
            let i = idx / cols;
            let j = idx % cols;
            let mut counts = vec![0usize; k];
            for di in -1i64..=1 {
                let ni = (i as i64 + di).clamp(0, rows as i64 - 1) as usize;
                for dj in -1i64..=1 {
                    let nj = (j as i64 + dj).clamp(0, cols as i64 - 1) as usize;
                    counts[classes[ni * cols + nj]] += 1;
                }
            }
            *out = counts.iter().enumerate().max_by_key(|&(_, &c)| c).map(|(i, _)| i).unwrap_or(0);
        };
        #[cfg(feature = "parallel")]
        next.par_iter_mut().enumerate().for_each(fill);
        #[cfg(not(feature = "parallel"))]
        next.iter_mut().enumerate().for_each(fill);
        std::mem::swap(&mut classes, &mut next);
    }

    let inv = 1.0 / k as f64;
    let data: Vec<f64> = classes.iter().map(|&c| c as f64 * inv).collect();
    Grid { data, rows, cols }
}

/// Returns a cellular automaton NLM with binary values {0.0, 1.0}.
///
/// Initialises a random binary grid where each cell is alive with probability
/// `p`, then applies Conway-style birth/survival rules for `iterations` steps.
/// Out-of-bounds neighbours are treated as dead, producing natural cave walls
/// at grid edges. Typical cave settings: `p ≈ 0.45`, `birth ≥ 5`, `survival ≥ 4`.
///
/// # Arguments
///
/// * `rows`               - Number of rows.
/// * `cols`               - Number of columns.
/// * `p`                  - Initial probability of a cell being alive.
/// * `iterations`         - Number of CA rule applications.
/// * `birth_threshold`    - Minimum live neighbours for a dead cell to become alive.
/// * `survival_threshold` - Minimum live neighbours for a live cell to stay alive.
/// * `seed`               - Optional RNG seed for reproducible results.
pub fn cellular_automaton(
    rows: usize,
    cols: usize,
    p: f64,
    iterations: usize,
    birth_threshold: usize,
    survival_threshold: usize,
    seed: Option<u64>,
) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }

    let mut rng = make_rng(seed);
    let mut state: Vec<bool> = (0..rows * cols).map(|_| rng.gen::<f64>() < p).collect();
    let mut next = vec![false; rows * cols];

    for _ in 0..iterations {
        let fill = |(idx, out): (usize, &mut bool)| {
            let i = idx / cols;
            let j = idx % cols;
            let mut count = 0usize;
            for di in -1i64..=1 {
                let ni = i as i64 + di;
                if ni < 0 || ni >= rows as i64 { continue; }
                let ni = ni as usize;
                for dj in -1i64..=1 {
                    if di == 0 && dj == 0 { continue; }
                    let nj = j as i64 + dj;
                    if nj < 0 || nj >= cols as i64 { continue; }
                    if state[ni * cols + nj as usize] { count += 1; }
                }
            }
            *out = if state[idx] { count >= survival_threshold } else { count >= birth_threshold };
        };
        #[cfg(feature = "parallel")]
        next.par_iter_mut().enumerate().for_each(fill);
        #[cfg(not(feature = "parallel"))]
        next.iter_mut().enumerate().for_each(fill);
        std::mem::swap(&mut state, &mut next);
    }

    let data: Vec<f64> = state.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
    Grid { data, rows, cols }
}

/// Returns a diffusion-limited aggregation (DLA) NLM with binary values {0.0, 1.0}.
///
/// Seeds a single cluster at the grid centre then releases `n` particles one at
/// a time. Each particle spawns on a circle slightly outside the current cluster
/// radius and performs a 4-connected random walk until it touches the cluster,
/// at which point it sticks. Particles that wander too far are respawned.
/// Produces branching fractal tree-like structures.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n`    - Number of particles to release.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn diffusion_limited_aggregation(rows: usize, cols: usize, n: usize, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }

    let mut rng = make_rng(seed);
    let mut cluster = vec![false; rows * cols];

    let cr = rows / 2;
    let cc = cols / 2;
    cluster[cr * cols + cc] = true;
    let mut cluster_r: f64 = 1.0;

    let half = (rows.min(cols) as f64 / 2.0) - 1.0;

    for _ in 0..n {
        let spawn_r = (cluster_r + 5.0).min(half);
        let kill_r  = (spawn_r * 2.0).min(half);

        let angle = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
        let mut pr = (cr as f64 + spawn_r * angle.sin())
            .round().clamp(0.0, rows as f64 - 1.0) as usize;
        let mut pc = (cc as f64 + spawn_r * angle.cos())
            .round().clamp(0.0, cols as f64 - 1.0) as usize;

        if cluster[pr * cols + pc] {
            continue;
        }

        loop {
            // Stick if adjacent to cluster (4-connected).
            let stuck = [
                (pr.wrapping_sub(1), pc),
                (pr + 1, pc),
                (pr, pc.wrapping_sub(1)),
                (pr, pc + 1),
            ]
            .iter()
            .any(|&(nr, nc)| nr < rows && nc < cols && cluster[nr * cols + nc]);

            if stuck {
                cluster[pr * cols + pc] = true;
                let d = ((pr as f64 - cr as f64).powi(2) + (pc as f64 - cc as f64).powi(2)).sqrt();
                if d > cluster_r { cluster_r = d; }
                break;
            }

            // Respawn if the particle has wandered too far.
            let d = ((pr as f64 - cr as f64).powi(2) + (pc as f64 - cc as f64).powi(2)).sqrt();
            if d > kill_r {
                let angle = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
                pr = (cr as f64 + spawn_r * angle.sin())
                    .round().clamp(0.0, rows as f64 - 1.0) as usize;
                pc = (cc as f64 + spawn_r * angle.cos())
                    .round().clamp(0.0, cols as f64 - 1.0) as usize;
                continue;
            }

            // 4-connected random walk step.
            match rng.gen_range(0..4u8) {
                0 => { if pr > 0 { pr -= 1; } }
                1 => { if pr + 1 < rows { pr += 1; } }
                2 => { if pc > 0 { pc -= 1; } }
                _ => { if pc + 1 < cols { pc += 1; } }
            }
        }
    }

    let data: Vec<f64> = cluster.iter().map(|&c| if c { 1.0 } else { 0.0 }).collect();
    Grid { data, rows, cols }
}

/// Returns a Gray-Scott reaction-diffusion NLM. Values in [0, 1).
///
/// Two virtual chemicals (A and B) diffuse and react across the grid.
/// Different `feed`/`kill` parameter combinations produce spots, stripes,
/// labyrinths, and other Turing-pattern morphologies.
///
/// # Arguments
///
/// * `rows`       - Number of rows.
/// * `cols`       - Number of columns.
/// * `iterations` - Number of simulation steps (more = more developed patterns).
/// * `feed`       - Feed rate for chemical A. Controls pattern type.
/// * `kill`       - Kill rate for chemical B. Controls pattern type.
/// * `seed`       - Optional RNG seed for reproducible initial conditions.
pub fn reaction_diffusion(
    rows: usize,
    cols: usize,
    iterations: usize,
    feed: f64,
    kill: f64,
    seed: Option<u64>,
) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let mut rng = make_rng(seed);

    const DA: f64 = 0.2;
    const DB: f64 = 0.1;

    let mut a = vec![1.0f64; rows * cols];
    let mut b = vec![0.0f64; rows * cols];

    // Seed random 3×3 squares of B (with periodic wrap)
    let n_seeds = (((rows * cols) as f64 * 0.01).ceil() as usize).max(1);
    for _ in 0..n_seeds {
        let r = rng.gen_range(0..rows);
        let c = rng.gen_range(0..cols);
        for dr in -1i64..=1 {
            for dc in -1i64..=1 {
                let nr = ((r as i64 + dr).rem_euclid(rows as i64)) as usize;
                let nc = ((c as i64 + dc).rem_euclid(cols as i64)) as usize;
                b[nr * cols + nc] = 1.0;
                a[nr * cols + nc] = 0.0;
            }
        }
    }

    let mut na = vec![0.0f64; rows * cols];
    let mut nb = vec![0.0f64; rows * cols];

    for _ in 0..iterations {
        let iter_fn = |(idx, (out_a, out_b)): (usize, (&mut f64, &mut f64))| {
            let i = idx / cols;
            let j = idx % cols;
            let ip = if i + 1 < rows { i + 1 } else { 0 };
            let im = if i > 0 { i - 1 } else { rows - 1 };
            let jp = if j + 1 < cols { j + 1 } else { 0 };
            let jm = if j > 0 { j - 1 } else { cols - 1 };

            let av = a[idx];
            let bv = b[idx];
            let lap_a = a[im * cols + j] + a[ip * cols + j]
                      + a[i * cols + jm] + a[i * cols + jp]
                      - 4.0 * av;
            let lap_b = b[im * cols + j] + b[ip * cols + j]
                      + b[i * cols + jm] + b[i * cols + jp]
                      - 4.0 * bv;

            let reaction = av * bv * bv;
            *out_a = (av + DA * lap_a - reaction + feed * (1.0 - av)).clamp(0.0, 1.0);
            *out_b = (bv + DB * lap_b + reaction - (kill + feed) * bv).clamp(0.0, 1.0);
        };

        #[cfg(feature = "parallel")]
        na.par_iter_mut().zip(nb.par_iter_mut()).enumerate().for_each(iter_fn);
        #[cfg(not(feature = "parallel"))]
        na.iter_mut().zip(nb.iter_mut()).enumerate().for_each(iter_fn);

        std::mem::swap(&mut a, &mut na);
        std::mem::swap(&mut b, &mut nb);
    }

    let mut grid = Grid { data: b, rows, cols };
    scale(&mut grid);
    grid
}

/// Returns an Eden growth model NLM. Binary values {0.0, 1.0}.
///
/// Grows a compact cluster from the grid centre by repeatedly selecting a
/// random cell on the current cluster boundary and adding it. Produces
/// irregular blob shapes with fractal perimeters.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n`    - Number of cells to add to the cluster (beyond the initial seed).
/// * `seed` - Optional RNG seed for reproducible results.
pub fn eden_growth(rows: usize, cols: usize, n: usize, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let mut rng = make_rng(seed);

    let mut cluster = vec![false; rows * cols];
    let cr = rows / 2;
    let cc = cols / 2;
    cluster[cr * cols + cc] = true;

    let dirs: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    let mut perimeter: Vec<(usize, usize)> = Vec::new();
    for (dr, dc) in dirs {
        let nr = cr as i64 + dr;
        let nc = cc as i64 + dc;
        if nr >= 0 && nr < rows as i64 && nc >= 0 && nc < cols as i64 {
            perimeter.push((nr as usize, nc as usize));
        }
    }

    let n_cells = n.min(rows * cols - 1);
    for _ in 0..n_cells {
        if perimeter.is_empty() { break; }
        let idx = rng.gen_range(0..perimeter.len());
        let (r, c) = perimeter.swap_remove(idx);
        if cluster[r * cols + c] { continue; }
        cluster[r * cols + c] = true;
        for (dr, dc) in dirs {
            let nr = r as i64 + dr;
            let nc = c as i64 + dc;
            if nr >= 0 && nr < rows as i64 && nc >= 0 && nc < cols as i64 {
                let nr = nr as usize;
                let nc = nc as usize;
                if !cluster[nr * cols + nc] {
                    perimeter.push((nr, nc));
                }
            }
        }
    }

    let data: Vec<f64> = cluster.iter().map(|&c| if c { 1.0 } else { 0.0 }).collect();
    Grid { data, rows, cols }
}

/// Returns an invasion percolation NLM. Binary values {0.0, 1.0}.
///
/// Each cell is assigned a random weight. Growing from the centre, the algorithm
/// always invades the boundary cell with the lowest weight first, producing
/// dendritic, river-like fractal cluster patterns.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n`    - Number of cells to invade (beyond the initial seed).
/// * `seed` - Optional RNG seed for reproducible results.
pub fn invasion_percolation(rows: usize, cols: usize, n: usize, seed: Option<u64>) -> Grid {
    use std::collections::BinaryHeap;
    use std::cmp::Reverse;

    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let mut rng = make_rng(seed);

    let weights: Vec<f64> = (0..rows * cols).map(|_| rng.gen::<f64>()).collect();
    let mut invaded = vec![false; rows * cols];
    let cr = rows / 2;
    let cc = cols / 2;
    invaded[cr * cols + cc] = true;

    let dirs: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    // Min-heap keyed on weight bits (IEEE 754 positive floats sort correctly as u64)
    let mut heap: BinaryHeap<Reverse<(u64, usize, usize)>> = BinaryHeap::new();
    for (dr, dc) in dirs {
        let nr = cr as i64 + dr;
        let nc = cc as i64 + dc;
        if nr >= 0 && nr < rows as i64 && nc >= 0 && nc < cols as i64 {
            let nr = nr as usize;
            let nc = nc as usize;
            heap.push(Reverse((weights[nr * cols + nc].to_bits(), nr, nc)));
        }
    }

    let n_cells = n.min(rows * cols - 1);
    for _ in 0..n_cells {
        if heap.is_empty() { break; }
        let Reverse((_, r, c)) = heap.pop().unwrap();
        if invaded[r * cols + c] { continue; }
        invaded[r * cols + c] = true;
        for (dr, dc) in dirs {
            let nr = r as i64 + dr;
            let nc = c as i64 + dc;
            if nr >= 0 && nr < rows as i64 && nc >= 0 && nc < cols as i64 {
                let nr = nr as usize;
                let nc = nc as usize;
                if !invaded[nr * cols + nc] {
                    heap.push(Reverse((weights[nr * cols + nc].to_bits(), nr, nc)));
                }
            }
        }
    }

    let data: Vec<f64> = invaded.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
    Grid { data, rows, cols }
}

/// Returns a Gaussian blobs NLM with values in [0, 1).
///
/// Places `n` Gaussian kernels at random positions and accumulates their
/// contributions. Produces soft, overlapping circular patches resembling
/// habitat fragments or dispersal kernels.
///
/// # Arguments
///
/// * `rows`  - Number of rows.
/// * `cols`  - Number of columns.
/// * `n`     - Number of Gaussian blobs to place.
/// * `sigma` - Standard deviation (radius) of each blob in cells.
/// * `seed`  - Optional RNG seed for reproducible results.
pub fn gaussian_blobs(rows: usize, cols: usize, n: usize, sigma: f64, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let mut rng = make_rng(seed);

    let centers: Vec<(f64, f64)> = (0..n)
        .map(|_| (rng.gen::<f64>() * rows as f64, rng.gen::<f64>() * cols as f64))
        .collect();
    let inv2s2 = 1.0 / (2.0 * sigma * sigma);

    let fill = |(idx, v): (usize, &mut f64)| {
        let i = idx / cols;
        let j = idx % cols;
        *v = centers.iter().map(|&(cr, cc)| {
            let dr = i as f64 - cr;
            let dc = j as f64 - cc;
            (-(dr * dr + dc * dc) * inv2s2).exp()
        }).sum::<f64>();
    };

    let mut data = vec![0.0f64; rows * cols];
    #[cfg(feature = "parallel")]
    data.par_iter_mut().enumerate().for_each(fill);
    #[cfg(not(feature = "parallel"))]
    data.iter_mut().enumerate().for_each(fill);

    let mut grid = Grid { data, rows, cols };
    scale(&mut grid);
    grid
}

/// Returns an Ising model NLM. Binary values {0.0, 1.0}.
///
/// Simulates a 2D ferromagnetic Ising model using Glauber dynamics. Spins start
/// random and relax under neighbourhood energy rules. The inverse temperature
/// `beta` controls cluster size: values near 0 produce random binary noise;
/// values near and above the critical point (~0.44) produce large clusters.
///
/// # Arguments
///
/// * `rows`       - Number of rows.
/// * `cols`       - Number of columns.
/// * `beta`       - Inverse temperature. Higher = larger, more ordered clusters.
/// * `iterations` - Number of full-grid update sweeps.
/// * `seed`       - Optional RNG seed for reproducible results.
pub fn ising_model(
    rows: usize,
    cols: usize,
    beta: f64,
    iterations: usize,
    seed: Option<u64>,
) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let mut rng = make_rng(seed);

    let mut spins: Vec<i8> = (0..rows * cols)
        .map(|_| if rng.gen::<bool>() { 1i8 } else { -1i8 })
        .collect();

    for _ in 0..iterations {
        // One sweep = rows*cols random single-spin updates (Glauber dynamics)
        for _ in 0..rows * cols {
            let idx = rng.gen_range(0..rows * cols);
            let i = idx / cols;
            let j = idx % cols;

            let s = spins[idx] as i32;
            let sum = spins[if i > 0 { (i - 1) * cols + j } else { (rows - 1) * cols + j }] as i32
                    + spins[if i + 1 < rows { (i + 1) * cols + j } else { j }] as i32
                    + spins[if j > 0 { i * cols + j - 1 } else { i * cols + cols - 1 }] as i32
                    + spins[if j + 1 < cols { i * cols + j + 1 } else { i * cols }] as i32;

            // Glauber flip probability: P(flip) = 1 / (1 + exp(β * ΔE)), ΔE = 2 * s * Σ
            let delta_e = 2.0 * s as f64 * sum as f64;
            if rng.gen::<f64>() < 1.0 / (1.0 + (beta * delta_e).exp()) {
                spins[idx] = -spins[idx];
            }
        }
    }

    let data: Vec<f64> = spins.iter().map(|&s| if s > 0 { 1.0 } else { 0.0 }).collect();
    Grid { data, rows, cols }
}

/// Returns a Levy flight NLM with values ranging [0, 1).
///
/// Simulates a Levy flight: a random walk where step lengths follow a
/// power-law (heavy-tailed) distribution. Starting from a random cell, the
/// walker takes `n` steps on a toroidal grid, incrementing the visit count at
/// each landing cell. The normalised density map produces clustered hotspots
/// with occasional long-range jumps, modelling dispersal or foraging patterns.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n`    - Number of flight steps.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn levy_flight(rows: usize, cols: usize, n: usize, seed: Option<u64>) -> Grid {
    use std::f64::consts::PI;
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let mut rng = make_rng(seed);
    let mut grid = Grid::new(rows, cols);

    let mut row = rng.gen_range(0..rows);
    let mut col = rng.gen_range(0..cols);
    grid.data[row * cols + col] += 1.0;

    let alpha = 1.5_f64;
    let max_step = (rows + cols) as f64;
    for _ in 0..n {
        // Power-law step length via inverse CDF: step ~ U^(-1/alpha).
        let u = rng.gen::<f64>().max(1e-10);
        let step = u.powf(-1.0 / alpha).min(max_step);
        let angle = rng.gen::<f64>() * 2.0 * PI;
        let dr = (step * angle.sin()).round() as i64;
        let dc = (step * angle.cos()).round() as i64;
        // Toroidal wrap so the walker never leaves the grid.
        row = ((row as i64 + dr).rem_euclid(rows as i64)) as usize;
        col = ((col as i64 + dc).rem_euclid(cols as i64)) as usize;
        grid.data[row * cols + col] += 1.0;
    }

    scale(&mut grid);
    grid
}

/// Returns a hydraulic erosion NLM with values ranging [0, 1).
///
/// Generates a random initial heightmap, then simulates `n` water droplets
/// flowing downhill. Each droplet carries sediment, eroding steeper terrain
/// and depositing on flatter areas. The result resembles naturally worn
/// terrain with drainage channels, alluvial fans, and rounded ridges.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n`    - Number of erosion droplets to simulate.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn hydraulic_erosion(rows: usize, cols: usize, n: usize, seed: Option<u64>) -> Grid {
    if rows < 2 || cols < 2 {
        return Grid::new(rows, cols);
    }
    let mut rng = make_rng(seed);
    let mut grid = rand_grid(rows, cols, &mut rng);

    let inertia = 0.3_f64;
    let capacity_factor = 8.0_f64;
    let erosion_rate = 0.1_f64;
    let deposition_rate = 0.1_f64;
    let evaporation_rate = 0.02_f64;
    let min_slope = 0.01_f64;

    for _ in 0..n {
        let mut x = rng.gen::<f64>() * (cols - 1) as f64;
        let mut y = rng.gen::<f64>() * (rows - 1) as f64;
        let mut vx = 0.0_f64;
        let mut vy = 0.0_f64;
        let mut water = 1.0_f64;
        let mut sediment = 0.0_f64;

        for _ in 0..30 {
            let ix = x as usize;
            let iy = y as usize;
            if ix + 1 >= cols || iy + 1 >= rows {
                break;
            }

            let fx = x - ix as f64;
            let fy = y - iy as f64;
            let h00 = grid.data[iy * cols + ix];
            let h10 = grid.data[iy * cols + ix + 1];
            let h01 = grid.data[(iy + 1) * cols + ix];
            let h11 = grid.data[(iy + 1) * cols + ix + 1];
            let h = h00 * (1.0 - fx) * (1.0 - fy)
                + h10 * fx * (1.0 - fy)
                + h01 * (1.0 - fx) * fy
                + h11 * fx * fy;
            // Bilinear gradient
            let gx = (h10 - h00) * (1.0 - fy) + (h11 - h01) * fy;
            let gy = (h01 - h00) * (1.0 - fx) + (h11 - h10) * fx;

            vx = vx * inertia - gx * (1.0 - inertia);
            vy = vy * inertia - gy * (1.0 - inertia);
            let speed = (vx * vx + vy * vy).sqrt();
            if speed < 1e-6 {
                break;
            }

            let new_x = (x + vx / speed).clamp(0.0, (cols - 1) as f64 - 1e-6);
            let new_y = (y + vy / speed).clamp(0.0, (rows - 1) as f64 - 1e-6);
            let ix2 = new_x as usize;
            let iy2 = new_y as usize;
            let fx2 = new_x - ix2 as f64;
            let fy2 = new_y - iy2 as f64;
            let h_new = grid.data[iy2 * cols + ix2] * (1.0 - fx2) * (1.0 - fy2)
                + grid.data[iy2 * cols + (ix2 + 1).min(cols - 1)] * fx2 * (1.0 - fy2)
                + grid.data[(iy2 + 1).min(rows - 1) * cols + ix2] * (1.0 - fx2) * fy2
                + grid.data[(iy2 + 1).min(rows - 1) * cols + (ix2 + 1).min(cols - 1)]
                    * fx2
                    * fy2;

            let dh = h_new - h;
            let capacity = ((-dh).max(min_slope) * speed * water * capacity_factor).max(0.0);
            let (deposit, erode) = if sediment > capacity {
                ((sediment - capacity) * deposition_rate, 0.0)
            } else {
                (0.0, ((capacity - sediment) * erosion_rate).min(0.1))
            };
            let delta = deposit - erode;
            grid.data[iy * cols + ix] += delta * (1.0 - fx) * (1.0 - fy);
            grid.data[iy * cols + ix + 1] += delta * fx * (1.0 - fy);
            grid.data[(iy + 1) * cols + ix] += delta * (1.0 - fx) * fy;
            grid.data[(iy + 1) * cols + ix + 1] += delta * fx * fy;

            sediment += erode - deposit;
            x = new_x;
            y = new_y;
            water *= 1.0 - evaporation_rate;
            if water < 0.01 {
                break;
            }
        }
    }

    scale(&mut grid);
    grid
}

/// Returns a Poisson disk sampling NLM. Binary values {0.0, 1.0}.
///
/// Uses Bridson's algorithm to place points such that no two are closer than
/// `min_dist`. Cells at sampling locations are set to 1.0; all others 0.0.
/// The resulting pattern has regular, inhibition-driven spacing, modelling
/// processes such as territorial behaviour or tree canopy competition.
///
/// # Arguments
///
/// * `rows`     - Number of rows.
/// * `cols`     - Number of columns.
/// * `min_dist` - Minimum distance in cells between any two sample points.
/// * `seed`     - Optional RNG seed for reproducible results.
pub fn poisson_disk(rows: usize, cols: usize, min_dist: f64, seed: Option<u64>) -> Grid {
    use std::f64::consts::PI;
    let mut grid = Grid::new(rows, cols);
    if rows == 0 || cols == 0 || min_dist <= 0.0 {
        return grid;
    }
    let mut rng = make_rng(seed);

    // Background acceleration grid; cell size = min_dist / sqrt(2).
    let cell = (min_dist / 2.0_f64.sqrt()).max(1.0);
    let gcols = (cols as f64 / cell).ceil() as usize + 1;
    let grows = (rows as f64 / cell).ceil() as usize + 1;
    // -1 = empty; otherwise stores the index into `samples`.
    let mut spatial: Vec<i32> = vec![-1i32; grows * gcols];
    let mut samples: Vec<(f64, f64)> = Vec::new();
    let mut active: Vec<usize> = Vec::new();

    // Seed the algorithm with one random point.
    let r0 = rng.gen::<f64>() * rows as f64;
    let c0 = rng.gen::<f64>() * cols as f64;
    let gi = (r0 / cell) as usize;
    let gj = (c0 / cell) as usize;
    spatial[gi * gcols + gj] = 0;
    samples.push((r0, c0));
    active.push(0);

    while !active.is_empty() {
        let pick = rng.gen_range(0..active.len());
        let (pr, pc) = samples[active[pick]];
        let mut found = false;
        for _ in 0..30 {
            // Sample in annulus [min_dist, 2 * min_dist].
            let r = min_dist * (1.0 + rng.gen::<f64>());
            let angle = rng.gen::<f64>() * 2.0 * PI;
            let nr = pr + r * angle.sin();
            let nc = pc + r * angle.cos();
            if nr < 0.0 || nr >= rows as f64 || nc < 0.0 || nc >= cols as f64 {
                continue;
            }
            let gi = (nr / cell) as usize;
            let gj = (nc / cell) as usize;
            let mut ok = true;
            'check: for di in 0..5usize {
                for dj in 0..5usize {
                    let ni = gi.wrapping_add(di).wrapping_sub(2);
                    let nj = gj.wrapping_add(dj).wrapping_sub(2);
                    if ni >= grows || nj >= gcols {
                        continue;
                    }
                    let s = spatial[ni * gcols + nj];
                    if s >= 0 {
                        let (sr, sc) = samples[s as usize];
                        let dr = nr - sr;
                        let dc = nc - sc;
                        if dr * dr + dc * dc < min_dist * min_dist {
                            ok = false;
                            break 'check;
                        }
                    }
                }
            }
            if ok {
                let new_idx = samples.len();
                spatial[gi * gcols + gj] = new_idx as i32;
                samples.push((nr, nc));
                active.push(new_idx);
                found = true;
                break;
            }
        }
        if !found {
            active.swap_remove(pick);
        }
    }

    for (r, c) in &samples {
        let ri = r.round() as usize;
        let ci = c.round() as usize;
        if ri < rows && ci < cols {
            grid.data[ri * cols + ci] = 1.0;
        }
    }

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
    fn test_random(#[case] rows: usize, #[case] cols: usize) {
        let grid = random(rows, cols, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_random_seeded_determinism() {
        let a = random(50, 50, Some(42));
        let b = random(50, 50, Some(42));
        assert_eq!(a.data, b.data);
        let c = random(50, 50, Some(99));
        assert_ne!(a.data, c.data);
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
    fn test_random_element(#[case] rows: usize, #[case] cols: usize) {
        let grid = random_element(rows, cols, 900., None);
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
    fn test_midpoint_displacement(#[case] rows: usize, #[case] cols: usize) {
        let grid = midpoint_displacement(rows, cols, 1., None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_midpoint_displacement_seeded_determinism() {
        let a = midpoint_displacement(100, 100, 1.0, Some(42));
        let b = midpoint_displacement(100, 100, 1.0, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── gaussian_field ────────────────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_gaussian_field(#[case] rows: usize, #[case] cols: usize) {
        let grid = gaussian_field(rows, cols, 5.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_gaussian_field_seeded_determinism() {
        let a = gaussian_field(50, 50, 5.0, Some(42));
        let b = gaussian_field(50, 50, 5.0, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── random_cluster ────────────────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    #[case(500, 500)]
    fn test_random_cluster(#[case] rows: usize, #[case] cols: usize) {
        let grid = random_cluster(rows, cols, 200, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_random_cluster_seeded_determinism() {
        let a = random_cluster(50, 50, 200, Some(42));
        let b = random_cluster(50, 50, 200, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── mosaic ────────────────────────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_mosaic(#[case] rows: usize, #[case] cols: usize) {
        let grid = mosaic(rows, cols, 50, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_mosaic_seeded_determinism() {
        let a = mosaic(50, 50, 50, Some(42));
        let b = mosaic(50, 50, 50, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── rectangular_cluster ───────────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_rectangular_cluster(#[case] rows: usize, #[case] cols: usize) {
        let grid = rectangular_cluster(rows, cols, 50, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_rectangular_cluster_seeded_determinism() {
        let a = rectangular_cluster(50, 50, 50, Some(42));
        let b = rectangular_cluster(50, 50, 50, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── percolation ───────────────────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_percolation(#[case] rows: usize, #[case] cols: usize) {
        let grid = percolation(rows, cols, 0.5, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        for &v in grid.iter() {
            assert!(v == 0.0 || v == 1.0, "unexpected value {v}");
        }
    }

    #[test]
    fn test_percolation_all_habitat() {
        let grid = percolation(20, 20, 1.0, Some(1));
        assert!(grid.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_percolation_no_habitat() {
        let grid = percolation(20, 20, 0.0, Some(1));
        assert!(grid.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_percolation_seeded_determinism() {
        let a = percolation(50, 50, 0.5, Some(42));
        let b = percolation(50, 50, 0.5, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── binary_space_partitioning ─────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_binary_space_partitioning(#[case] rows: usize, #[case] cols: usize) {
        let grid = binary_space_partitioning(rows, cols, 20, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_binary_space_partitioning_seeded_determinism() {
        let a = binary_space_partitioning(50, 50, 20, Some(42));
        let b = binary_space_partitioning(50, 50, 20, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── neighbourhood_clustering ──────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_neighbourhood_clustering(#[case] rows: usize, #[case] cols: usize) {
        let grid = neighbourhood_clustering(rows, cols, 5, 10, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_neighbourhood_clustering_seeded_determinism() {
        let a = neighbourhood_clustering(50, 50, 5, 10, Some(42));
        let b = neighbourhood_clustering(50, 50, 5, 10, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── cellular_automaton ────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_cellular_automaton(#[case] rows: usize, #[case] cols: usize) {
        let grid = cellular_automaton(rows, cols, 0.45, 5, 5, 4, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        // Binary output: every value must be 0.0 or 1.0
        assert!(grid.data.iter().all(|&v| v == 0.0 || v == 1.0));
    }

    #[test]
    fn test_cellular_automaton_seeded_determinism() {
        let a = cellular_automaton(50, 50, 0.45, 5, 5, 4, Some(42));
        let b = cellular_automaton(50, 50, 0.45, 5, 5, 4, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── diffusion_limited_aggregation ─────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(50, 50)]
    fn test_diffusion_limited_aggregation(#[case] rows: usize, #[case] cols: usize) {
        let grid = diffusion_limited_aggregation(rows, cols, 50, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert!(grid.data.iter().all(|&v| v == 0.0 || v == 1.0));
    }

    #[test]
    fn test_diffusion_limited_aggregation_seeded_determinism() {
        let a = diffusion_limited_aggregation(50, 50, 200, Some(42));
        let b = diffusion_limited_aggregation(50, 50, 200, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── reaction_diffusion ────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(50, 50)]
    fn test_reaction_diffusion(#[case] rows: usize, #[case] cols: usize) {
        let grid = reaction_diffusion(rows, cols, 100, 0.055, 0.062, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_reaction_diffusion_seeded_determinism() {
        let a = reaction_diffusion(30, 30, 100, 0.055, 0.062, Some(42));
        let b = reaction_diffusion(30, 30, 100, 0.055, 0.062, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── eden_growth ───────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(50, 50)]
    fn test_eden_growth(#[case] rows: usize, #[case] cols: usize) {
        let grid = eden_growth(rows, cols, 50, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert!(grid.data.iter().all(|&v| v == 0.0 || v == 1.0));
    }

    #[test]
    fn test_eden_growth_seeded_determinism() {
        let a = eden_growth(50, 50, 200, Some(42));
        let b = eden_growth(50, 50, 200, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── invasion_percolation ──────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(50, 50)]
    fn test_invasion_percolation(#[case] rows: usize, #[case] cols: usize) {
        let grid = invasion_percolation(rows, cols, 50, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert!(grid.data.iter().all(|&v| v == 0.0 || v == 1.0));
    }

    #[test]
    fn test_invasion_percolation_seeded_determinism() {
        let a = invasion_percolation(50, 50, 200, Some(42));
        let b = invasion_percolation(50, 50, 200, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── gaussian_blobs ────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_gaussian_blobs(#[case] rows: usize, #[case] cols: usize) {
        let grid = gaussian_blobs(rows, cols, 10, 5.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_gaussian_blobs_seeded_determinism() {
        let a = gaussian_blobs(50, 50, 10, 5.0, Some(42));
        let b = gaussian_blobs(50, 50, 10, 5.0, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── ising_model ───────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(50, 50)]
    fn test_ising_model(#[case] rows: usize, #[case] cols: usize) {
        let grid = ising_model(rows, cols, 0.4, 100, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert!(grid.data.iter().all(|&v| v == 0.0 || v == 1.0));
    }

    #[test]
    fn test_ising_model_seeded_determinism() {
        let a = ising_model(50, 50, 0.4, 100, Some(42));
        let b = ising_model(50, 50, 0.4, 100, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── levy_flight ───────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_levy_flight(#[case] rows: usize, #[case] cols: usize) {
        let grid = levy_flight(rows, cols, 100, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_levy_flight_seeded_determinism() {
        let a = levy_flight(50, 50, 100, Some(42));
        let b = levy_flight(50, 50, 100, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── hydraulic_erosion ─────────────────────────────────────────────────────

    #[rstest]
    #[case(2, 2)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_hydraulic_erosion(#[case] rows: usize, #[case] cols: usize) {
        let grid = hydraulic_erosion(rows, cols, 50, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_hydraulic_erosion_seeded_determinism() {
        let a = hydraulic_erosion(50, 50, 50, Some(42));
        let b = hydraulic_erosion(50, 50, 50, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── poisson_disk ──────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_poisson_disk(#[case] rows: usize, #[case] cols: usize) {
        let grid = poisson_disk(rows, cols, 5.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert!(grid.data.iter().all(|&v| v == 0.0 || v == 1.0));
    }

    #[test]
    fn test_poisson_disk_seeded_determinism() {
        let a = poisson_disk(50, 50, 5.0, Some(42));
        let b = poisson_disk(50, 50, 5.0, Some(42));
        assert_eq!(a.data, b.data);
    }
}

// ── New algorithms ────────────────────────────────────────────────────────────

/// Returns a Brownian motion (Gaussian random walk) density NLM with values ranging [0, 1).
///
/// A single particle starts at a random cell and takes `n` isotropic Gaussian
/// steps (standard deviation ~1 cell) wrapping toroidally.  Each cell accumulates
/// the number of visits; the result is normalised to [0, 1].  Unlike Lévy flight,
/// Gaussian steps produce locally concentrated, diffusive paths without long jumps.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n`    - Number of walk steps (more steps = smoother, denser density field).
/// * `seed` - Optional RNG seed for reproducible results.
pub fn brownian_motion(rows: usize, cols: usize, n: usize, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(rows, cols);
    }
    let mut rng = make_rng(seed);
    let mut grid = Grid::new(rows, cols);
    let mut row = rng.gen_range(0..rows) as f64;
    let mut col = rng.gen_range(0..cols) as f64;

    for _ in 0..n {
        // Box-Muller transform: isotropic Gaussian step with σ = 1 cell.
        let u1: f64 = rng.gen::<f64>().max(f64::EPSILON);
        let u2: f64 = rng.gen::<f64>();
        let mag = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        row = (row + mag * angle.sin()).rem_euclid(rows as f64);
        col = (col + mag * angle.cos()).rem_euclid(cols as f64);
        grid[row as usize][col as usize] += 1.0;
    }

    scale(&mut grid);
    grid
}

/// Returns a forest fire NLM with values ranging [0, 1).
///
/// Simulates the Drossel-Schwabl forest fire cellular automaton for `iterations`
/// steps, tracking cumulative burn counts per cell.  Empty cells grow trees with
/// probability `p_tree`; trees ignite from 4-connected burning neighbours or with
/// spontaneous lightning probability `p_lightning`.  The normalised burn map encodes
/// heterogeneous landscape structure driven by fire disturbance.
///
/// # Arguments
///
/// * `rows`        - Number of rows.
/// * `cols`        - Number of columns.
/// * `p_tree`      - Per-step probability an empty cell becomes a tree (default 0.02).
/// * `p_lightning` - Per-step probability a tree ignites spontaneously (default 0.001).
/// * `iterations`  - Number of simulation steps (default 500).
/// * `seed`        - Optional RNG seed for reproducible results.
pub fn forest_fire(
    rows: usize,
    cols: usize,
    p_tree: f64,
    p_lightning: f64,
    iterations: usize,
    seed: Option<u64>,
) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(rows, cols);
    }
    let mut rng = make_rng(seed);
    // 0 = empty, 1 = tree, 2 = fire
    let mut state = vec![0u8; rows * cols];
    for s in state.iter_mut() {
        if rng.gen::<f64>() < p_tree { *s = 1; }
    }
    let mut burn_count = vec![0u32; rows * cols];

    for _ in 0..iterations {
        let mut next = state.clone();
        for i in 0..rows {
            for j in 0..cols {
                match state[i * cols + j] {
                    0 => {
                        if rng.gen::<f64>() < p_tree { next[i * cols + j] = 1; }
                    }
                    1 => {
                        let nbr_fire =
                            (i > 0 && state[(i - 1) * cols + j] == 2)
                            || (i + 1 < rows && state[(i + 1) * cols + j] == 2)
                            || (j > 0 && state[i * cols + j - 1] == 2)
                            || (j + 1 < cols && state[i * cols + j + 1] == 2);
                        if nbr_fire || rng.gen::<f64>() < p_lightning {
                            next[i * cols + j] = 2;
                            burn_count[i * cols + j] += 1;
                        }
                    }
                    _ => { next[i * cols + j] = 0; } // fire → empty (burned out)
                }
            }
        }
        state = next;
    }

    let mut grid = Grid::new(rows, cols);
    for (cell, &count) in grid.data.iter_mut().zip(burn_count.iter()) {
        *cell = count as f64;
    }
    scale(&mut grid);
    grid
}

/// Returns a river network NLM with values ranging [0, 1).
///
/// Generates a random fBm elevation model, routes flow downhill via the D8
/// algorithm (steepest of 8 neighbours), and accumulates drainage area by
/// topological sort.  The log-transformed flow accumulation is normalised to
/// [0, 1]: high values correspond to main channels; low values to headwater ridges.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn river_network(rows: usize, cols: usize, seed: Option<u64>) -> Grid {
    use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
    use std::collections::VecDeque;
    if rows == 0 || cols == 0 {
        return Grid::new(rows, cols);
    }

    // Generate fBm elevation model inline.
    let seed_val = perlin_seed(seed);
    let fbm = Fbm::<Perlin>::new(seed_val)
        .set_octaves(6)
        .set_persistence(0.5)
        .set_lacunarity(2.0);
    let inv_r = 3.0 / rows as f64;
    let inv_c = 3.0 / cols as f64;
    let mut elev: Vec<f64> = (0..rows * cols)
        .map(|idx| {
            let i = idx / cols;
            let j = idx % cols;
            fbm.get([j as f64 * inv_c, i as f64 * inv_r])
        })
        .collect();

    // Normalise elevation to [0, 1].
    let (mn, mx) = elev.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(a, b), &v| (a.min(v), b.max(v)),
    );
    let range = (mx - mn).max(1e-10);
    elev.iter_mut().for_each(|v| *v = (*v - mn) / range);

    // D8 flow direction: index of steepest-descent neighbour, or usize::MAX at sinks.
    let mut flow_dir = vec![usize::MAX; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let z = elev[i * cols + j];
            let mut best_z = z;
            let mut best_nb = usize::MAX;
            for di in -1i32..=1 {
                for dj in -1i32..=1 {
                    if di == 0 && dj == 0 { continue; }
                    let ni = i as i32 + di;
                    let nj = j as i32 + dj;
                    if ni < 0 || ni >= rows as i32 || nj < 0 || nj >= cols as i32 { continue; }
                    let nz = elev[ni as usize * cols + nj as usize];
                    if nz < best_z { best_z = nz; best_nb = ni as usize * cols + nj as usize; }
                }
            }
            flow_dir[i * cols + j] = best_nb;
        }
    }

    // Flow accumulation via topological sort (Kahn's algorithm).
    let mut in_degree = vec![0u32; rows * cols];
    for &dst in &flow_dir {
        if dst != usize::MAX { in_degree[dst] += 1; }
    }
    let mut queue: VecDeque<usize> =
        (0..rows * cols).filter(|&i| in_degree[i] == 0).collect();
    let mut accum = vec![1u32; rows * cols];
    while let Some(idx) = queue.pop_front() {
        if flow_dir[idx] != usize::MAX {
            let dst = flow_dir[idx];
            accum[dst] += accum[idx];
            in_degree[dst] -= 1;
            if in_degree[dst] == 0 { queue.push_back(dst); }
        }
    }

    let mut grid = Grid::new(rows, cols);
    for (cell, &acc) in grid.data.iter_mut().zip(accum.iter()) {
        *cell = (acc as f64).ln();
    }
    scale(&mut grid);
    grid
}

/// Returns a hexagonal Voronoi NLM with values ranging in (0, 1].
///
/// Seeds are placed on a slightly jittered regular hexagonal lattice, sized to
/// produce approximately `n` cells.  BFS nearest-seed fill assigns every cell the
/// value of its closest seed.  Compared to `mosaic` (purely random seeds) the
/// result has more regular, honeycomb-like patch sizes.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n`    - Approximate number of hexagonal cells.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn hexagonal_voronoi(rows: usize, cols: usize, n: usize, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(rows, cols);
    }
    let mut rng = make_rng(seed);
    let n = n.max(1);

    // Hex lattice spacing for approximately n cells.
    // Area of a regular hexagon with circumradius r: A = (3√3/2) r²
    let hex_r = ((rows * cols) as f64 / (n as f64 * 3.0 * 3_f64.sqrt() / 2.0))
        .sqrt()
        .max(1.0);
    let hex_w = 3_f64.sqrt() * hex_r; // column stride
    let hex_h = 1.5 * hex_r;          // row stride

    let mut grid = Grid::new(rows, cols);
    let mut row_f = 0.0f64;
    let mut row_idx = 0usize;
    while row_f < rows as f64 + hex_h {
        let offset = if row_idx % 2 == 0 { 0.0 } else { hex_w / 2.0 };
        let mut col_f = -hex_w / 2.0 + offset;
        while col_f < cols as f64 + hex_w {
            let jr = rng.gen_range(-hex_r * 0.1..hex_r * 0.1);
            let jc = rng.gen_range(-hex_w * 0.1..hex_w * 0.1);
            let ri = ((row_f + jr) as isize).clamp(0, rows as isize - 1) as usize;
            let ci = ((col_f + jc) as isize).clamp(0, cols as isize - 1) as usize;
            if grid[ri][ci] == 0.0 {
                grid[ri][ci] = rng.gen::<f64>() * 0.999 + 0.001;
            }
            col_f += hex_w;
        }
        row_f += hex_h;
        row_idx += 1;
    }
    interpolate(&mut grid);
    grid
}

/// Returns a fault uplift terrain NLM with values ranging [0, 1).
///
/// Generates `n` random geological fault lines. Along each fault the
/// terrain forms a smooth scarp: elevation rises sharply at the fault
/// line and decays exponentially away from it on both sides, producing
/// a network of linear ridges with realistic topographic relief.
/// Unlike `random_cluster` (which creates broad plateau/basin regions),
/// this generates narrow ridge features aligned along fault traces.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n`    - Number of fault lines.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn fault_uplift(rows: usize, cols: usize, n: usize, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let mut rng = make_rng(seed);
    let mut data = vec![0.0f64; rows * cols];

    let inv_r = 1.0 / rows as f64;
    let inv_c = 1.0 / cols as f64;
    // Ridge half-width in normalised units (~5% of grid).
    let width = 0.05f64;

    for _ in 0..n {
        let px: f64 = rng.gen();
        let py: f64 = rng.gen();
        let theta: f64 = rng.gen_range(0.0..std::f64::consts::PI);
        let (nnx, nny) = (theta.cos(), theta.sin());

        for idx in 0..rows * cols {
            let cy = (idx / cols) as f64 * inv_r;
            let cx = (idx % cols) as f64 * inv_c;
            // Perpendicular distance to the fault line.
            let d = ((cx - px) * nnx + (cy - py) * nny).abs();
            data[idx] += (-d / width).exp();
        }
    }

    let mut grid = Grid { data, rows, cols };
    scale(&mut grid);
    grid
}

/// Returns a triangular tessellation NLM with values ranging [0, 1).
///
/// Scatters `n` random points and computes their Delaunay triangulation
/// via the Bowyer-Watson algorithm. Each triangle is then assigned a
/// uniform random value, producing a mosaic of flat-shaded triangles.
/// Unlike `mosaic` (Voronoi polygons) or `hexagonal_voronoi`, all
/// regions have exactly three straight edges.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n`    - Number of seed points (triangles ≈ 2n).
/// * `seed` - Optional RNG seed for reproducible results.
pub fn triangular_tessellation(rows: usize, cols: usize, n: usize, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let mut rng = make_rng(seed);
    let n = n.max(3);

    // Points in [0, 1] × [0, 1] normalised space.
    let pts: Vec<(f64, f64)> = (0..n).map(|_| (rng.gen::<f64>(), rng.gen::<f64>())).collect();
    let tris = bowyer_watson(&pts);
    let values: Vec<f64> = (0..tris.len()).map(|_| rng.gen::<f64>()).collect();

    let inv_r = 1.0 / rows as f64;
    let inv_c = 1.0 / cols as f64;

    let data: Vec<f64> = (0..rows * cols)
        .map(|idx| {
            let py = (idx / cols) as f64 * inv_r;
            let px = (idx % cols) as f64 * inv_c;

            // Find which triangle contains this pixel.
            if let Some(ti) = tris.iter().position(|&[a, b, c]| {
                let (ax, ay) = pts[a]; let (bx, by) = pts[b]; let (cx, cy) = pts[c];
                point_in_tri(px, py, ax, ay, bx, by, cx, cy)
            }) {
                return values[ti];
            }

            // Fallback: colour by nearest point's first triangle.
            let nearest = pts
                .iter()
                .enumerate()
                .min_by(|&(_, &(ax, ay)), &(_, &(bx, by))| {
                    (px - ax).hypot(py - ay)
                        .partial_cmp(&((px - bx).hypot(py - by)))
                        .unwrap()
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            tris.iter()
                .position(|t| t.contains(&nearest))
                .map(|ti| values[ti])
                .unwrap_or(0.0)
        })
        .collect();

    Grid { data, rows, cols }
}

/// Returns a Physarum (slime mould) network NLM with values ranging [0, 1).
///
/// Simulates the *Physarum polycephalum* transport-network algorithm
/// (Jones 2010): `n` agents deposit a chemo-attractant trail, sense
/// concentrations ahead-left and ahead-right, and steer toward the
/// stronger signal. After each step the trail is blurred and decayed.
/// The accumulated trail map reveals the characteristic vein-like
/// network patterns used to model ecological corridors and road networks.
///
/// # Arguments
///
/// * `rows`       - Number of rows.
/// * `cols`       - Number of columns.
/// * `n`          - Number of agents.
/// * `iterations` - Number of simulation steps.
/// * `seed`       - Optional RNG seed for reproducible results.
pub fn physarum(
    rows: usize,
    cols: usize,
    n: usize,
    iterations: usize,
    seed: Option<u64>,
) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let mut rng = make_rng(seed);
    let mut trail = vec![0.0f64; rows * cols];

    let mut pos: Vec<(f64, f64)> = (0..n)
        .map(|_| (rng.gen_range(0.0..rows as f64), rng.gen_range(0.0..cols as f64)))
        .collect();
    let mut heading: Vec<f64> = (0..n)
        .map(|_| rng.gen_range(0.0..2.0 * std::f64::consts::PI))
        .collect();

    let sensor_dist = (rows.min(cols) as f64 * 0.08).max(2.0);
    let sensor_angle = std::f64::consts::FRAC_PI_4;
    let step_size = 1.0f64;
    let deposit = 1.0f64;
    let decay = 0.92f64;
    let turn_speed = std::f64::consts::FRAC_PI_4;
    let row_max = (rows - 1) as f64;
    let col_max = (cols - 1) as f64;

    let fold = |x: f64, max: f64| -> f64 {
        let x = x.rem_euclid(2.0 * max);
        if x > max { 2.0 * max - x } else { x }
    };
    let sample = |trail: &[f64], r: f64, c: f64| -> f64 {
        let ri = (r as isize).clamp(0, rows as isize - 1) as usize;
        let ci = (c as isize).clamp(0, cols as isize - 1) as usize;
        trail[ri * cols + ci]
    };

    let mut new_pos = pos.clone();
    let mut new_heading = heading.clone();

    for _ in 0..iterations {
        // Sense and rotate.
        for i in 0..n {
            let (r, c) = pos[i];
            let h = heading[i];
            let fl = sample(&trail, r + (h + sensor_angle).sin() * sensor_dist,
                                   c + (h + sensor_angle).cos() * sensor_dist);
            let fc = sample(&trail, r + h.sin() * sensor_dist,
                                   c + h.cos() * sensor_dist);
            let fr = sample(&trail, r + (h - sensor_angle).sin() * sensor_dist,
                                   c + (h - sensor_angle).cos() * sensor_dist);
            new_heading[i] = if fc >= fl && fc >= fr {
                h
            } else if fl > fr {
                h + turn_speed
            } else if fr > fl {
                h - turn_speed
            } else {
                h + if rng.gen_bool(0.5) { turn_speed } else { -turn_speed }
            };
        }

        // Move and deposit.
        for i in 0..n {
            let (r, c) = pos[i];
            let h = new_heading[i];
            new_pos[i] = (fold(r + h.sin() * step_size, row_max),
                          fold(c + h.cos() * step_size, col_max));
            let ri = new_pos[i].0 as usize;
            let ci = new_pos[i].1 as usize;
            if ri < rows && ci < cols {
                trail[ri * cols + ci] += deposit;
            }
        }

        // Diffuse and decay.
        let old = trail.clone();
        for idx in 0..rows * cols {
            let ri = idx / cols;
            let ci = idx % cols;
            let mut sum = 0.0;
            let mut cnt = 0u32;
            for dr in -1i64..=1 {
                let nr = (ri as i64 + dr).clamp(0, rows as i64 - 1) as usize;
                for dc in -1i64..=1 {
                    let nc = (ci as i64 + dc).clamp(0, cols as i64 - 1) as usize;
                    sum += old[nr * cols + nc];
                    cnt += 1;
                }
            }
            trail[idx] = (sum / cnt as f64) * decay;
        }

        std::mem::swap(&mut pos, &mut new_pos);
        std::mem::swap(&mut heading, &mut new_heading);
    }

    let mut result = Grid { data: trail, rows, cols };
    scale(&mut result);
    result
}

/// Returns a Cahn-Hilliard phase-separation NLM with values ranging [0, 1).
///
/// Numerically integrates the Cahn-Hilliard PDE, which models spinodal
/// decomposition: a uniform mixture separates into two pure phases (e.g.,
/// two habitat types, oil and water). The resulting spatial pattern
/// transitions from fine-grained mixed to coarse blob-and-matrix as
/// `iterations` increases — a smooth, two-phase field distinct from
/// `reaction_diffusion`.
///
/// # Arguments
///
/// * `rows`       - Number of rows.
/// * `cols`       - Number of columns.
/// * `iterations` - Number of integration steps.
/// * `seed`       - Optional RNG seed for reproducible results.
pub fn cahn_hilliard(rows: usize, cols: usize, iterations: usize, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let mut rng = make_rng(seed);

    // Small random perturbations around zero (the unstable mixed state).
    let mut u: Vec<f64> = (0..rows * cols).map(|_| rng.gen_range(-0.05..0.05)).collect();
    let mut mu = vec![0.0f64; rows * cols];
    let mut u_next = vec![0.0f64; rows * cols];

    let eps2 = 0.01f64; // interface width parameter
    let dt = 0.05f64;

    for _ in 0..iterations {
        // Chemical potential: μ = u³ - u - ε²∇²u
        for idx in 0..rows * cols {
            let i = idx / cols;
            let j = idx % cols;
            let ui = u[idx];
            mu[idx] = ui * ui * ui - ui - eps2 * laplacian_periodic(&u, i, j, rows, cols);
        }
        // Update: u += dt * ∇²μ
        for idx in 0..rows * cols {
            let i = idx / cols;
            let j = idx % cols;
            u_next[idx] = u[idx] + dt * laplacian_periodic(&mu, i, j, rows, cols);
        }
        std::mem::swap(&mut u, &mut u_next);
    }

    let mut grid = Grid { data: u, rows, cols };
    scale(&mut grid);
    grid
}

/// Returns a crystal growth NLM with values ranging [0, 1).
///
/// Simulates Reiter's (1996) snowflake cellular automaton on a
/// hexagonal lattice. A single frozen seed at the centre grows by
/// absorbing water vapour from surrounding cells; receptive cells (those
/// adjacent to the crystal) accumulate vapour at rate `gamma` and freeze
/// when their level reaches 1.  The background vapour density `beta`
/// and the number of `iterations` control the final crystal complexity
/// and size.
///
/// # Arguments
///
/// * `rows`       - Number of rows.
/// * `cols`       - Number of columns.
/// * `iterations` - Number of growth steps.
/// * `seed`       - Optional RNG seed (unused; growth is deterministic).
pub fn crystal_growth(rows: usize, cols: usize, iterations: usize, _seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let beta = 0.4f64;    // background vapour density
    let alpha = 0.9f64;   // diffusion weight for non-receptive cells
    let gamma = 0.001f64; // vapour added to receptive cells per step

    let mut u = vec![beta; rows * cols];
    let mut frozen = vec![false; rows * cols];

    let cr = rows / 2;
    let cc = cols / 2;
    frozen[cr * cols + cc] = true;
    u[cr * cols + cc] = 0.0;

    // 6-connectivity using offset-row hex grid.
    let hex_nb = |i: usize, j: usize| -> [(isize, isize); 6] {
        if i % 2 == 0 {
            [(-1,-1),(-1,0),(0,-1),(0,1),(1,-1),(1,0)]
        } else {
            [(-1,0),(-1,1),(0,-1),(0,1),(1,0),(1,1)]
        }
        .map(|(dr, dc)| (i as isize + dr, j as isize + dc))
    };

    let mut u_next = u.clone();
    for _ in 0..iterations {
        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                if frozen[idx] { continue; }

                let nb: Vec<(usize, usize)> = hex_nb(i, j)
                    .iter()
                    .filter(|&&(ni, nj)| ni >= 0 && ni < rows as isize && nj >= 0 && nj < cols as isize)
                    .map(|&(ni, nj)| (ni as usize, nj as usize))
                    .collect();

                let has_frozen_nb = nb.iter().any(|&(ni, nj)| frozen[ni * cols + nj]);

                if has_frozen_nb {
                    u_next[idx] = u[idx] + gamma;
                    if u_next[idx] >= 1.0 {
                        frozen[idx] = true;
                        u_next[idx] = 0.0;
                    }
                } else {
                    let non_frozen: Vec<_> = nb.iter()
                        .filter(|&&(ni, nj)| !frozen[ni * cols + nj])
                        .copied()
                        .collect();
                    let avg = if non_frozen.is_empty() {
                        u[idx]
                    } else {
                        non_frozen.iter().map(|&(ni, nj)| u[ni * cols + nj]).sum::<f64>()
                            / non_frozen.len() as f64
                    };
                    u_next[idx] = (1.0 - alpha) * u[idx] + alpha * avg;
                }
            }
        }
        std::mem::swap(&mut u, &mut u_next);
    }

    let data: Vec<f64> = (0..rows * cols)
        .map(|idx| if frozen[idx] { 1.0 } else { u[idx] / beta })
        .collect();
    let mut grid = Grid { data, rows, cols };
    scale(&mut grid);
    grid
}

/// Returns a predator-prey (Lotka-Volterra) NLM with values ranging [0, 1).
///
/// Numerically integrates the spatially explicit Lotka-Volterra system
/// with diffusion. Prey (u) grows logistically and is consumed by
/// predators (v); predators grow from predation and die at a constant
/// rate.  Diffusion disperses both populations across the grid.  Near
/// Turing-instability parameter regimes the system spontaneously
/// develops travelling waves and spatial spirals, producing dynamical
/// spatial patterns useful as ecological null models.
///
/// # Arguments
///
/// * `rows`       - Number of rows.
/// * `cols`       - Number of columns.
/// * `iterations` - Number of integration steps.
/// * `seed`       - Optional RNG seed for reproducible results.
pub fn predator_prey(rows: usize, cols: usize, iterations: usize, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let mut rng = make_rng(seed);

    // Lotka-Volterra parameters.
    let r = 1.0f64;    // prey growth rate
    let k = 1.0f64;    // prey carrying capacity
    let a = 1.0f64;    // predation rate
    let b = 0.5f64;    // predator growth efficiency
    let d = 0.3f64;    // predator death rate
    let du = 0.05f64;  // prey diffusivity
    let dv = 0.5f64;   // predator diffusivity
    let dt = 0.05f64;

    // Equilibrium: u* = d/b, v* = r/a*(1 - u*/k)
    let u_eq = d / b;
    let v_eq = (r / a) * (1.0 - u_eq / k);

    let mut u: Vec<f64> = (0..rows * cols)
        .map(|_| (u_eq + rng.gen_range(-0.1..0.1)).max(0.0))
        .collect();
    let mut v: Vec<f64> = (0..rows * cols)
        .map(|_| (v_eq + rng.gen_range(-0.1..0.1)).max(0.0))
        .collect();
    let mut u_next = u.clone();
    let mut v_next = v.clone();

    for _ in 0..iterations {
        for idx in 0..rows * cols {
            let i = idx / cols;
            let j = idx % cols;
            let ui = u[idx].max(0.0);
            let vi = v[idx].max(0.0);
            let lap_u = laplacian_periodic(&u, i, j, rows, cols);
            let lap_v = laplacian_periodic(&v, i, j, rows, cols);
            u_next[idx] = (ui + dt * (r * ui * (1.0 - ui / k) - a * ui * vi + du * lap_u)).max(0.0);
            v_next[idx] = (vi + dt * (b * ui * vi - d * vi + dv * lap_v)).max(0.0);
        }
        std::mem::swap(&mut u, &mut u_next);
        std::mem::swap(&mut v, &mut v_next);
    }

    let mut grid = Grid { data: u, rows, cols };
    scale(&mut grid);
    grid
}

/// Returns a Bak-Tang-Wiesenfeld sandpile NLM with values ranging [0, 1).
///
/// Adds `n` grains one at a time to random grid cells. When a cell
/// accumulates 4 or more grains it "topples": it loses 4 grains and each
/// of its 4 cardinal neighbours gains 1. Grains that fall off the grid
/// boundary are lost. Toppling propagates until the grid is stable before
/// the next grain is dropped. The resulting grain-count map, scaled to
/// [0, 1), reveals self-similar avalanche structure characteristic of
/// self-organized criticality.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n`    - Number of grains to drop.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn sandpile(rows: usize, cols: usize, n: usize, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let mut rng = make_rng(seed);
    let mut grains: Vec<i64> = vec![0; rows * cols];
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();

    for _ in 0..n {
        let r = rng.gen_range(0..rows);
        let c = rng.gen_range(0..cols);
        let idx = r * cols + c;
        grains[idx] += 1;
        if grains[idx] >= 4 {
            queue.push_back(idx);
        }
        while let Some(idx) = queue.pop_front() {
            if grains[idx] < 4 {
                continue;
            }
            grains[idx] -= 4;
            let r = idx / cols;
            let c = idx % cols;
            let neighbors: [Option<usize>; 4] = [
                if r > 0 { Some((r - 1) * cols + c) } else { None },
                if r + 1 < rows { Some((r + 1) * cols + c) } else { None },
                if c > 0 { Some(r * cols + c - 1) } else { None },
                if c + 1 < cols { Some(r * cols + c + 1) } else { None },
            ];
            for nb in neighbors.into_iter().flatten() {
                grains[nb] += 1;
                if grains[nb] >= 4 {
                    queue.push_back(nb);
                }
            }
        }
    }

    let data: Vec<f64> = grains.iter().map(|&v| v as f64).collect();
    let mut result = Grid { data, rows, cols };
    scale(&mut result);
    result
}

/// Returns a correlated random walk density NLM with values ranging [0, 1).
///
/// A single walker takes `n` steps. At each step the new heading is drawn
/// from a wrapped normal distribution centred on the previous heading with
/// standard deviation `1 / sqrt(kappa + 1)`. Setting `kappa = 0` gives a
/// classic Brownian motion; increasing `kappa` produces straighter, more
/// directional trajectories. The grid accumulates visit counts, which are
/// scaled to [0, 1). Boundaries are reflecting.
///
/// # Arguments
///
/// * `rows`  - Number of rows.
/// * `cols`  - Number of columns.
/// * `n`     - Number of walk steps.
/// * `kappa` - Directional persistence (0 = isotropic, higher = straighter).
/// * `seed`  - Optional RNG seed for reproducible results.
pub fn correlated_walk(
    rows: usize,
    cols: usize,
    n: usize,
    kappa: f64,
    seed: Option<u64>,
) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let mut rng = make_rng(seed);
    let mut data = vec![0.0f64; rows * cols];

    let mut r = rng.gen_range(0..rows) as f64;
    let mut c = rng.gen_range(0..cols) as f64;
    let mut theta = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
    let sigma = 1.0 / (kappa + 1.0).sqrt();

    let row_max = (rows - 1) as f64;
    let col_max = (cols - 1) as f64;

    for _ in 0..n {
        let ri = r as usize;
        let ci = c as usize;
        data[ri * cols + ci] += 1.0;

        // Box-Muller transform for a N(0, sigma) turn angle.
        let u1 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
        let u2 = rng.gen::<f64>();
        let turn =
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * sigma;
        theta += turn;

        let nr = r + theta.sin();
        let nc = c + theta.cos();

        // Reflecting boundary: fold back into [0, max].
        let fold = |x: f64, max: f64| -> f64 {
            let x = x.rem_euclid(2.0 * max);
            if x > max { 2.0 * max - x } else { x }
        };
        r = fold(nr, row_max);
        c = fold(nc, col_max);
    }

    let mut result = Grid { data, rows, cols };
    scale(&mut result);
    result
}

/// Returns a Schelling segregation NLM with values in {0.0, 0.5, 1.0}.
///
/// Initialises a grid where ~45 % of cells are type A (0.0), ~45 % are
/// type B (1.0), and ~10 % are empty (0.5). In each iteration every
/// unhappy cell — one where fewer than `tolerance` of its (up to 8)
/// non-empty Moore neighbours share its type — is collected and then
/// relocated to a random empty cell. The process runs for `iterations`
/// sweeps, producing spatially segregated clusters typical of Schelling's
/// model of residential segregation.
///
/// # Arguments
///
/// * `rows`       - Number of rows.
/// * `cols`       - Number of columns.
/// * `tolerance`  - Minimum fraction of same-type neighbours for happiness (0–1).
/// * `iterations` - Number of relocation sweeps.
/// * `seed`       - Optional RNG seed for reproducible results.
pub fn schelling(
    rows: usize,
    cols: usize,
    tolerance: f64,
    iterations: usize,
    seed: Option<u64>,
) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let mut rng = make_rng(seed);
    let n = rows * cols;

    // -1 = empty, 0 = type A, 1 = type B
    let mut state: Vec<i8> = (0..n)
        .map(|_| {
            let r: f64 = rng.gen();
            if r < 0.45 { 0 } else if r < 0.90 { 1 } else { -1 }
        })
        .collect();

    // Collect empty cells into an index list for O(1) random-empty lookup.
    let mut empties: Vec<usize> = state
        .iter()
        .enumerate()
        .filter(|(_, &v)| v == -1)
        .map(|(i, _)| i)
        .collect();

    for _ in 0..iterations {
        let mut unhappy: Vec<usize> = Vec::new();
        for idx in 0..n {
            let kind = state[idx];
            if kind == -1 {
                continue;
            }
            let r = idx / cols;
            let c = idx % cols;
            let mut same = 0usize;
            let mut total = 0usize;
            for dr in -1i64..=1 {
                let nr = r as i64 + dr;
                if nr < 0 || nr >= rows as i64 {
                    continue;
                }
                for dc in -1i64..=1 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let nc = c as i64 + dc;
                    if nc < 0 || nc >= cols as i64 {
                        continue;
                    }
                    let nb = state[nr as usize * cols + nc as usize];
                    if nb != -1 {
                        total += 1;
                        if nb == kind {
                            same += 1;
                        }
                    }
                }
            }
            let frac = if total == 0 { 1.0 } else { same as f64 / total as f64 };
            if frac < tolerance {
                unhappy.push(idx);
            }
        }

        if unhappy.is_empty() || empties.is_empty() {
            break;
        }

        for &idx in &unhappy {
            if empties.is_empty() {
                break;
            }
            let pick = rng.gen_range(0..empties.len());
            let dest = empties[pick];
            // Swap agent into the empty slot; old position becomes empty.
            state[dest] = state[idx];
            state[idx] = -1;
            empties[pick] = idx; // old position is now empty
        }
    }

    let data: Vec<f64> = state
        .iter()
        .map(|&v| match v {
            0 => 0.0,
            1 => 1.0,
            _ => 0.5,
        })
        .collect();
    Grid { data, rows, cols }
}

/// Returns a spatial SIR epidemic NLM with values ranging [0, 1).
///
/// Numerically integrates the spatially-explicit SIR reaction-diffusion PDE:
///
///   dS/dt = -β·S·I + D_S·∇²S
///   dI/dt =  β·S·I - γ·I + D_I·∇²I
///   dR/dt =  γ·I
///
/// The grid is initialised with S≈1 everywhere and a small infection seed
/// I=0.1 at the centre. Periodic boundaries are used. The output is the
/// recovered field R, which maps the spatial footprint of the epidemic wave.
///
/// # Arguments
///
/// * `rows`       - Number of rows.
/// * `cols`       - Number of columns.
/// * `beta`       - Transmission rate.
/// * `gamma`      - Recovery rate.
/// * `iterations` - Number of forward-Euler integration steps.
/// * `seed`       - Optional RNG seed (unused; provided for API consistency).
pub fn sir_epidemic(
    rows: usize,
    cols: usize,
    beta: f64,
    gamma: f64,
    iterations: usize,
    _seed: Option<u64>,
) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let n = rows * cols;
    let dt  = 0.5f64;
    let d_s = 0.1f64;
    let d_i = 0.5f64;

    let mut s = vec![1.0f64; n];
    let mut inf = vec![0.0f64; n];
    let mut r = vec![0.0f64; n];

    let centre = (rows / 2) * cols + (cols / 2);
    inf[centre] = 0.1;
    s[centre]   = 0.9;

    let mut s_next = s.clone();
    let mut i_next = inf.clone();
    let mut r_next = r.clone();

    for _ in 0..iterations {
        for idx in 0..n {
            let row = idx / cols;
            let col = idx % cols;
            let si = s[idx].max(0.0);
            let ii = inf[idx].max(0.0);
            let ri = r[idx];
            let lap_s = laplacian_periodic(&s, row, col, rows, cols);
            let lap_i = laplacian_periodic(&inf, row, col, rows, cols);
            let infection = beta * si * ii;
            s_next[idx] = (si + dt * (-infection + d_s * lap_s)).max(0.0);
            i_next[idx] = (ii + dt * (infection - gamma * ii + d_i * lap_i)).max(0.0);
            r_next[idx] = (ri + dt * gamma * ii).clamp(0.0, 1.0);
        }
        std::mem::swap(&mut s, &mut s_next);
        std::mem::swap(&mut inf, &mut i_next);
        std::mem::swap(&mut r, &mut r_next);
    }

    let mut grid = Grid { data: r, rows, cols };
    scale(&mut grid);
    grid
}

/// Returns a thermally eroded heightmap NLM with values ranging [0, 1).
///
/// Initialises a multi-octave Perlin heightmap and applies `n` passes of
/// talus-slope erosion: any cell whose height exceeds a neighbour's by more
/// than the talus angle donates a fraction of the surplus, simulating rock-fall
/// and scree formation. Uses four-connectivity with non-periodic boundaries.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n`    - Number of erosion passes.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn thermal_erosion(rows: usize, cols: usize, n: usize, seed: Option<u64>) -> Grid {
    use noise::{NoiseFn, Perlin};

    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }

    let seed_val = perlin_seed(seed);
    let inv_rows = 1.0 / rows as f64;
    let inv_cols = 1.0 / cols as f64;
    let scale_f  = 4.0f64;
    let lacunarity = 2.0f64;
    let persistence = 0.5f64;
    let n_oct = 6usize;

    let generators: Vec<(Perlin, f64, f64)> = {
        let mut v = Vec::new();
        let mut freq = scale_f;
        let mut amp  = 1.0f64;
        let mut total = 0.0f64;
        for o in 0..n_oct {
            v.push((Perlin::new(seed_val.wrapping_add(o as u32)), freq, amp));
            total += amp;
            amp  *= persistence;
            freq *= lacunarity;
        }
        let inv = 1.0 / total;
        v.into_iter().map(|(g, f, a)| (g, f, a * inv)).collect()
    };

    let mut height: Vec<f64> = (0..rows * cols)
        .map(|idx| {
            let nx = (idx % cols) as f64 * inv_cols;
            let ny = (idx / cols) as f64 * inv_rows;
            generators.iter().map(|(g, f, a)| g.get([nx * f, ny * f]) * a).sum::<f64>()
        })
        .collect();

    let talus    = 0.4 / rows.max(cols) as f64;
    let fraction = 0.5f64;
    let mut next = height.clone();

    for _ in 0..n {
        next.copy_from_slice(&height);
        for idx in 0..rows * cols {
            let r = idx / cols;
            let c = idx % cols;
            let h = height[idx];
            let neighbours: [Option<usize>; 4] = [
                if r > 0        { Some((r - 1) * cols + c) } else { None },
                if r + 1 < rows { Some((r + 1) * cols + c) } else { None },
                if c > 0        { Some(r * cols + c - 1)   } else { None },
                if c + 1 < cols { Some(r * cols + c + 1)   } else { None },
            ];
            for ni in neighbours.into_iter().flatten() {
                let diff = h - height[ni];
                if diff > talus {
                    let transfer = fraction * (diff - talus);
                    next[idx] -= transfer;
                    next[ni]  += transfer;
                }
            }
        }
        std::mem::swap(&mut height, &mut next);
    }

    let mut grid = Grid { data: height, rows, cols };
    scale(&mut grid);
    grid
}

/// Returns a space colonisation vascular network NLM with values ranging [0, 1).
///
/// Models plant vascular development: `n` auxin (attractor) points are placed
/// in the upper 80% of the grid. A root node at the bottom centre grows
/// step-by-step toward nearby auxin points, consuming them when close enough.
/// New nodes branch when they are attracted by multiple auxin simultaneously.
/// The output encodes the visit density of the resulting branching network.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n`    - Number of auxin attractor points.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn space_colonization(rows: usize, cols: usize, n: usize, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let mut rng = make_rng(seed);

    let aux_row_max = (rows as f64 * 0.8).max(1.0);
    let mut auxin: Vec<(f64, f64)> = (0..n)
        .map(|_| (rng.gen_range(0.0..aux_row_max), rng.gen_range(0.0..cols as f64)))
        .collect();

    let mut nodes: Vec<(f64, f64)> = vec![(rows as f64 - 1.0, cols as f64 / 2.0)];
    let mut data = vec![0.0f64; rows * cols];

    let influence_r = (rows.max(cols) as f64) * 0.4;
    let kill_dist   = (rows.max(cols) as f64) * 0.04;
    let step_size   = 1.5f64;
    let max_iter    = n * 20 + 100;
    let max_nodes   = rows * cols;

    let mark = |data: &mut Vec<f64>, r: f64, c: f64| {
        let ri = (r as usize).min(rows - 1);
        let ci = (c as usize).min(cols - 1);
        data[ri * cols + ci] += 1.0;
    };
    mark(&mut data, nodes[0].0, nodes[0].1);

    'outer: for _ in 0..max_iter {
        if auxin.is_empty() || nodes.len() >= max_nodes {
            break;
        }
        let n_nodes = nodes.len();
        let mut new_nodes: Vec<(f64, f64)> = Vec::new();

        for ni in 0..n_nodes {
            let (nr, nc) = nodes[ni];
            let mut dir_r = 0.0f64;
            let mut dir_c = 0.0f64;
            let mut count = 0usize;

            for &(ar, ac) in &auxin {
                let dr = ar - nr;
                let dc = ac - nc;
                let dist = (dr * dr + dc * dc).sqrt();
                if dist < influence_r && dist > 0.0 {
                    dir_r += dr / dist;
                    dir_c += dc / dist;
                    count += 1;
                }
            }
            if count == 0 {
                continue;
            }

            let len = (dir_r * dir_r + dir_c * dir_c).sqrt().max(1e-12);
            let new_r = (nr + step_size * dir_r / len).clamp(0.0, rows as f64 - 1.0);
            let new_c = (nc + step_size * dir_c / len).clamp(0.0, cols as f64 - 1.0);
            new_nodes.push((new_r, new_c));
        }

        if new_nodes.is_empty() {
            break;
        }

        for &(nr, nc) in &new_nodes {
            mark(&mut data, nr, nc);
            nodes.push((nr, nc));
            auxin.retain(|&(ar, ac)| {
                let dr = ar - nr;
                let dc = ac - nc;
                (dr * dr + dc * dc).sqrt() >= kill_dist
            });
            if auxin.is_empty() {
                break 'outer;
            }
            if nodes.len() >= max_nodes {
                break 'outer;
            }
        }
    }

    let mut grid = Grid { data, rows, cols };
    scale(&mut grid);
    grid
}

/// Returns a crack-propagation substrate NLM with values ranging [0, 1).
///
/// Simulates Jared Tarbell's Substrate algorithm: `n` crack seeds propagate
/// across the grid in their initial random directions, deviating slightly at
/// each step. When a crack reaches an occupied cell it stops, potentially
/// spawning a perpendicular child. Active cracks also occasionally branch.
/// The output encodes the visit density of all crack paths.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n`    - Number of initial crack seeds.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn substrate(rows: usize, cols: usize, n: usize, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let mut rng = make_rng(seed);
    let mut data     = vec![0.0f64; rows * cols];
    let mut occupied = vec![false;  rows * cols];

    let mut cracks: Vec<(f64, f64, f64)> = (0..n.max(1))
        .map(|_| {
            let r = rng.gen_range(0.0..rows as f64);
            let c = rng.gen_range(0.0..cols as f64);
            let a = rng.gen_range(0.0..std::f64::consts::TAU);
            (r, c, a)
        })
        .collect();

    let p_spawn  = 0.002f64;
    let max_iter = rows * cols * 2 + 1000;
    let max_cracks = 2000usize;

    for _ in 0..max_iter {
        if cracks.is_empty() {
            break;
        }
        let mut new_cracks: Vec<(f64, f64, f64)> = Vec::new();
        let mut to_remove: Vec<usize> = Vec::new();
        let num_cracks = cracks.len();

        for (idx, (r, c, angle)) in cracks.iter_mut().enumerate() {
            // Small random Gaussian angular deviation (Box-Muller).
            let u1 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
            let u2 = rng.gen::<f64>();
            *angle += (-2.0 * u1.ln()).sqrt()
                * (2.0 * std::f64::consts::PI * u2).cos()
                * 0.05;

            *r += angle.sin();
            *c += angle.cos();

            if *r < 0.0 || *r >= rows as f64 || *c < 0.0 || *c >= cols as f64 {
                to_remove.push(idx);
                continue;
            }

            let ri = *r as usize;
            let ci = *c as usize;
            let cell = ri * cols + ci;

            if occupied[cell] {
                to_remove.push(idx);
                if num_cracks + new_cracks.len() < max_cracks {
                    new_cracks.push((*r, *c, *angle + std::f64::consts::FRAC_PI_2));
                }
                continue;
            }

            occupied[cell] = true;
            data[cell] += 1.0;

            if rng.gen::<f64>() < p_spawn && num_cracks + new_cracks.len() < max_cracks {
                new_cracks.push((*r, *c, *angle + std::f64::consts::FRAC_PI_2));
            }
        }

        for &i in to_remove.iter().rev() {
            cracks.swap_remove(i);
        }
        cracks.extend(new_cracks);
    }

    let mut grid = Grid { data, rows, cols };
    scale(&mut grid);
    grid
}

/// Returns a Conway's Game of Life activity-density NLM with values ranging [0, 1).
///
/// Cells are randomly initialised alive with probability 0.45 and evolved
/// for `iterations` steps under the standard B3/S23 rules. The output is
/// the proportion of steps each cell spent alive, normalised to [0, 1).
/// Toroidal (periodic) boundary conditions are used.
///
/// # Arguments
///
/// * `rows`       - Number of rows.
/// * `cols`       - Number of columns.
/// * `iterations` - Number of simulation steps.
/// * `seed`       - Optional RNG seed for reproducible results.
pub fn game_of_life(rows: usize, cols: usize, iterations: usize, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let mut rng = make_rng(seed);

    let mut state: Vec<bool> = (0..rows * cols).map(|_| rng.gen::<f64>() < 0.45).collect();
    let mut next  = state.clone();
    let mut visits: Vec<f64> = state.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();

    for _ in 0..iterations {
        for idx in 0..rows * cols {
            let r = (idx / cols) as i64;
            let c = (idx % cols) as i64;
            let mut alive_nb = 0u8;
            for dr in -1i64..=1 {
                for dc in -1i64..=1 {
                    if dr == 0 && dc == 0 { continue; }
                    let nr = (r + dr).rem_euclid(rows as i64) as usize;
                    let nc = (c + dc).rem_euclid(cols as i64) as usize;
                    if state[nr * cols + nc] { alive_nb += 1; }
                }
            }
            next[idx] = matches!((state[idx], alive_nb), (true, 2) | (true, 3) | (false, 3));
        }
        std::mem::swap(&mut state, &mut next);
        for (v, &alive) in visits.iter_mut().zip(state.iter()) {
            if alive { *v += 1.0; }
        }
    }

    let mut grid = Grid { data: visits, rows, cols };
    scale(&mut grid);
    grid
}

#[cfg(test)]
mod new_tests {
    use super::*;
    use super::super::{nan_count, zero_to_one_count};
    use rstest::rstest;

    // ── brownian_motion ───────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_brownian_motion(#[case] rows: usize, #[case] cols: usize) {
        let grid = brownian_motion(rows, cols, 500, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_brownian_motion_seeded_determinism() {
        let a = brownian_motion(50, 50, 500, Some(42));
        let b = brownian_motion(50, 50, 500, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── forest_fire ───────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_forest_fire(#[case] rows: usize, #[case] cols: usize) {
        let grid = forest_fire(rows, cols, 0.05, 0.01, 100, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_forest_fire_seeded_determinism() {
        let a = forest_fire(50, 50, 0.05, 0.01, 100, Some(42));
        let b = forest_fire(50, 50, 0.05, 0.01, 100, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── river_network ─────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_river_network(#[case] rows: usize, #[case] cols: usize) {
        let grid = river_network(rows, cols, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_river_network_seeded_determinism() {
        let a = river_network(50, 50, Some(42));
        let b = river_network(50, 50, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── hexagonal_voronoi ─────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_hexagonal_voronoi(#[case] rows: usize, #[case] cols: usize) {
        let grid = hexagonal_voronoi(rows, cols, 20, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_hexagonal_voronoi_seeded_determinism() {
        let a = hexagonal_voronoi(50, 50, 20, Some(42));
        let b = hexagonal_voronoi(50, 50, 20, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── sandpile ──────────────────────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_sandpile(#[case] rows: usize, #[case] cols: usize) {
        let grid = sandpile(rows, cols, 500, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_sandpile_seeded_determinism() {
        let a = sandpile(50, 50, 500, Some(42));
        let b = sandpile(50, 50, 500, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── correlated_walk ───────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_correlated_walk(#[case] rows: usize, #[case] cols: usize) {
        let grid = correlated_walk(rows, cols, 500, 2.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_correlated_walk_seeded_determinism() {
        let a = correlated_walk(50, 50, 500, 2.0, Some(42));
        let b = correlated_walk(50, 50, 500, 2.0, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── schelling ─────────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_schelling(#[case] rows: usize, #[case] cols: usize) {
        let grid = schelling(rows, cols, 0.5, 20, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_schelling_seeded_determinism() {
        let a = schelling(50, 50, 0.5, 20, Some(42));
        let b = schelling(50, 50, 0.5, 20, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── fault_uplift ──────────────────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_fault_uplift(#[case] rows: usize, #[case] cols: usize) {
        let grid = fault_uplift(rows, cols, 20, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_fault_uplift_seeded_determinism() {
        let a = fault_uplift(50, 50, 20, Some(42));
        let b = fault_uplift(50, 50, 20, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── triangular_tessellation ───────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_triangular_tessellation(#[case] rows: usize, #[case] cols: usize) {
        let grid = triangular_tessellation(rows, cols, 20, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_triangular_tessellation_seeded_determinism() {
        let a = triangular_tessellation(50, 50, 20, Some(42));
        let b = triangular_tessellation(50, 50, 20, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── physarum ─────────────────────────────────────────────────────────────

    #[rstest]
    #[case(10, 10)]
    #[case(50, 50)]
    #[case(100, 100)]
    fn test_physarum(#[case] rows: usize, #[case] cols: usize) {
        let grid = physarum(rows, cols, 100, 50, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_physarum_seeded_determinism() {
        let a = physarum(50, 50, 100, 50, Some(42));
        let b = physarum(50, 50, 100, 50, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── cahn_hilliard ─────────────────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_cahn_hilliard(#[case] rows: usize, #[case] cols: usize) {
        let grid = cahn_hilliard(rows, cols, 50, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_cahn_hilliard_seeded_determinism() {
        let a = cahn_hilliard(50, 50, 50, Some(42));
        let b = cahn_hilliard(50, 50, 50, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── crystal_growth ────────────────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_crystal_growth(#[case] rows: usize, #[case] cols: usize) {
        let grid = crystal_growth(rows, cols, 50, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    // ── predator_prey ─────────────────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_predator_prey(#[case] rows: usize, #[case] cols: usize) {
        let grid = predator_prey(rows, cols, 50, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_predator_prey_seeded_determinism() {
        let a = predator_prey(50, 50, 50, Some(42));
        let b = predator_prey(50, 50, 50, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── sir_epidemic ──────────────────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_sir_epidemic(#[case] rows: usize, #[case] cols: usize) {
        let grid = sir_epidemic(rows, cols, 0.3, 0.1, 50, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_sir_epidemic_determinism() {
        let a = sir_epidemic(50, 50, 0.3, 0.1, 50, None);
        let b = sir_epidemic(50, 50, 0.3, 0.1, 50, None);
        assert_eq!(a.data, b.data);
    }

    // ── thermal_erosion ───────────────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_thermal_erosion(#[case] rows: usize, #[case] cols: usize) {
        let grid = thermal_erosion(rows, cols, 10, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_thermal_erosion_seeded_determinism() {
        let a = thermal_erosion(50, 50, 10, Some(42));
        let b = thermal_erosion(50, 50, 10, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── space_colonization ────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_space_colonization(#[case] rows: usize, #[case] cols: usize) {
        let grid = space_colonization(rows, cols, 50, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_space_colonization_seeded_determinism() {
        let a = space_colonization(50, 50, 50, Some(42));
        let b = space_colonization(50, 50, 50, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── substrate ─────────────────────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_substrate(#[case] rows: usize, #[case] cols: usize) {
        let grid = substrate(rows, cols, 5, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_substrate_seeded_determinism() {
        let a = substrate(50, 50, 5, Some(42));
        let b = substrate(50, 50, 5, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── game_of_life ──────────────────────────────────────────────────────────

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_game_of_life(#[case] rows: usize, #[case] cols: usize) {
        let grid = game_of_life(rows, cols, 50, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_game_of_life_seeded_determinism() {
        let a = game_of_life(50, 50, 50, Some(42));
        let b = game_of_life(50, 50, 50, Some(42));
        assert_eq!(a.data, b.data);
    }
}
