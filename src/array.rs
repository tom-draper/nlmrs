use rand::Rng;

use crate::grid::Grid;

/// Returns a grid of size (rows x cols) containing uniform random [0, 1) values.
pub fn rand_grid(rows: usize, cols: usize, rng: &mut impl Rng) -> Grid {
    let data = (0..rows * cols).map(|_| rng.gen()).collect();
    Grid { data, rows, cols }
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

fn check_diamond_coords(diax: i32, diay: i32, dim: i32, i2: i32) -> ([(i32, i32); 4], usize) {
    let mut coords = [(0i32, 0i32); 4];
    if diax < 0 || diax > dim || diay < 0 || diay > dim {
        return (coords, 0);
    } else if diax - i2 < 0 {
        coords[0] = (diax + i2, diay);
        coords[1] = (diax, diay - i2);
        coords[2] = (diax, diay + i2);
        return (coords, 3);
    } else if diax + i2 >= dim {
        coords[0] = (diax - i2, diay);
        coords[1] = (diax, diay - i2);
        coords[2] = (diax, diay + i2);
        return (coords, 3);
    } else if diay - i2 < 0 {
        coords[0] = (diax + i2, diay);
        coords[1] = (diax - i2, diay);
        coords[2] = (diax, diay + i2);
        return (coords, 3);
    } else if diay + i2 >= dim {
        coords[0] = (diax + i2, diay);
        coords[1] = (diax - i2, diay);
        coords[2] = (diax, diay - i2);
        return (coords, 3);
    }
    coords[0] = (diax + i2, diay);
    coords[1] = (diax - i2, diay);
    coords[2] = (diax, diay - i2);
    coords[3] = (diax, diay + i2);
    (coords, 4)
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
    let (diaco, n) = check_diamond_coords(diax as i32, diay as i32, dim as i32, mid as i32);
    let mut diavals = [0.0f64; 4];
    for k in 0..n {
        let (x, y) = diaco[k];
        diavals[k] = surface[x as usize][y as usize];
    }
    let r = rng.gen();
    surface[diax][diay] = displace_vals(&diavals[..n], disheight, r).unwrap();
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
                let mut arr = [surface[i][j], 0.0, 0.0, 0.0];
                let mut arr_n = 1usize;
                if i + inc < dim {
                    arr[arr_n] = surface[i + inc][j];
                    arr_n += 1;
                    if j + inc < dim {
                        arr[arr_n] = surface[i + inc][j + inc];
                        arr_n += 1;
                    }
                } else if j + inc < dim {
                    arr[arr_n] = surface[i][j + inc];
                    arr_n += 1;
                }
                let r = rng.gen();
                surface[i + mid][j + mid] = displace_vals(&arr[..arr_n], disheight, r).unwrap();
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
        let src_start = (row_start + i) * grid.cols + col_start;
        sub.data[i * cols..(i + 1) * cols].copy_from_slice(&grid.data[src_start..src_start + cols]);
    }
    sub
}
