use rand::Rng;

use crate::grid::Grid;

/// Returns a grid of size (rows x cols) containing zeros.
pub fn zeros(rows: usize, cols: usize) -> Grid {
    Grid::new(rows, cols)
}

/// Returns a grid of size (rows x cols) containing ones.
pub fn ones(rows: usize, cols: usize) -> Grid {
    Grid::filled(rows, cols, 1.0)
}

/// Returns a grid of size (rows x cols) containing a given value.
pub fn filled(rows: usize, cols: usize, value: f64) -> Grid {
    Grid::filled(rows, cols, value)
}

/// Returns a grid of size (rows x cols) containing uniform random [0, 1) values.
pub fn rand_grid(rows: usize, cols: usize, rng: &mut impl Rng) -> Grid {
    let data = (0..rows * cols).map(|_| rng.gen()).collect();
    Grid { data, rows, cols }
}

/// Returns a grid of size (rows x cols) containing uniform random binary [0, 1) values.
pub fn binary_rand_grid(rows: usize, cols: usize, rng: &mut impl Rng) -> Grid {
    let data = (0..rows * cols)
        .map(|_| rng.gen_range(0.0..1.0))
        .collect();
    Grid { data, rows, cols }
}

/// Returns a flat boolean mask where `true` marks cells equal to `value`.
pub fn value_mask(grid: &Grid, value: f64) -> Vec<bool> {
    grid.iter().map(|&x| x == value).collect()
}


fn random_displace(disheight: f64, r: f64) -> f64 {
    r * disheight - 0.5 * disheight
}

fn displace_vals(vals: &[f64], disheight: f64, r: (f64, f64)) -> Option<f64> {
    if vals.len() == 4 {
        let sum: f64 = vals.iter().sum();
        Some(sum * 0.25 + random_displace(disheight, r.0))
    } else if vals.len() == 3 {
        let sum: f64 = vals.iter().sum();
        Some(sum / 3. + random_displace(disheight, r.1))
    } else {
        None
    }
}

fn check_diamond_coords(diax: i32, diay: i32, dim: i32, i2: i32) -> Vec<(i32, i32)> {
    if diax < 0 || diax > dim || diay < 0 || diay > dim {
        return vec![];
    } else if diax - i2 < 0 {
        return vec![(diax + i2, diay), (diax, diay - i2), (diax, diay + i2)];
    } else if diax + i2 >= dim {
        return vec![(diax - i2, diay), (diax, diay - i2), (diax, diay + i2)];
    } else if diay - i2 < 0 {
        return vec![(diax + i2, diay), (diax - i2, diay), (diax, diay + i2)];
    } else if diay + i2 >= dim {
        return vec![(diax + i2, diay), (diax - i2, diay), (diax, diay - i2)];
    }
    vec![
        (diax + i2, diay),
        (diax - i2, diay),
        (diax, diay - i2),
        (diax, diay + i2),
    ]
}

fn diamond_step(
    surface: &mut Grid,
    disheight: f64,
    mid: usize,
    dim: usize,
    rng: &mut impl Rng,
    diax: usize,
    diay: usize,
) {
    let diaco = check_diamond_coords(diax as i32, diay as i32, dim as i32, mid as i32);
    let diavals: Vec<f64> = diaco
        .iter()
        .map(|&(x, y)| surface[x as usize][y as usize])
        .collect();
    let r = rng.gen();
    surface[diax][diay] = displace_vals(&diavals, disheight, r).unwrap();
}

/// Returns a diamond-square fractal surface of size (dim x dim).
///
/// Implementation ported from NLMpy.
pub fn diamond_square(dim: usize, h: f64, rng: &mut impl Rng) -> Grid {
    let mut disheight = 2.;
    let mut surface = rand_grid(dim, dim, rng);
    for val in surface.iter_mut() {
        *val = *val * disheight - 0.5 * disheight;
    }

    let mut inc = dim - 1;
    while inc > 1 {
        let mid = inc / 2;

        // Square step
        for i in (0..dim - 1).step_by(inc) {
            for j in (0..dim - 1).step_by(inc) {
                let mut arr = vec![surface[i][j]];
                if i + inc < dim {
                    arr.push(surface[i + inc][j]);
                    if j + inc < dim {
                        arr.push(surface[i + inc][j + inc]);
                    }
                } else if j + inc < dim {
                    arr.push(surface[i][j + inc]);
                }
                let r = rng.gen();
                surface[i + mid][j + mid] = displace_vals(&arr, disheight, r).unwrap();
            }
        }

        // Diamond step
        for i in (0..dim - 1).step_by(inc) {
            for j in (0..dim - 1).step_by(inc) {
                diamond_step(&mut surface, disheight, mid, dim, rng, i + mid, j);
                diamond_step(&mut surface, disheight, mid, dim, rng, i, j + mid);
                diamond_step(&mut surface, disheight, mid, dim, rng, i + inc, j + mid);
                diamond_step(&mut surface, disheight, mid, dim, rng, i + mid, j + inc);
            }
        }

        disheight = disheight.powf(-h) * 2.;
        inc /= 2;
    }
    surface
}

/// Selects a random subgrid of size (rows x cols) from the grid.
pub fn rand_sub_grid(grid: Grid, rows: usize, cols: usize, rng: &mut impl Rng) -> Grid {
    if rows >= grid.rows && cols >= grid.cols {
        return grid;
    }

    let row_start = if grid.rows != rows {
        rng.gen_range(0..(grid.rows - rows))
    } else {
        0
    };
    let col_start = if grid.cols != cols {
        rng.gen_range(0..(grid.cols - cols))
    } else {
        0
    };

    let mut sub = Grid::new(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            sub[i][j] = grid[row_start + i][col_start + j];
        }
    }
    sub
}
