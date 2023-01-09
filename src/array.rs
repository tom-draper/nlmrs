use rand::prelude::*;

/// Returns a 2D array of size (rows x cols) containing zeros.
pub fn zeros_arr(rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut vec: Vec<Vec<f64>> = Vec::with_capacity(rows);
    for _ in 0..rows {
        let row = vec![0f64; cols];
        vec.push(row);
    }
    vec
}

/// Returns a 2D array of size (rows x cols) containing ones.
pub fn ones_arr(rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut vec: Vec<Vec<f64>> = Vec::with_capacity(rows);
    for _ in 0..rows {
        let row = vec![1f64; cols];
        vec.push(row);
    }
    vec
}

/// Returns a 2D array of size (rows x cols) containing uniform random [0, 1) values.
pub fn rand_arr(rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    let mut vec: Vec<Vec<f64>> = Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for _ in 0..cols {
            row.push(rng.gen());
        }
        vec.push(row);
    }
    vec
}

/// Returns a 2D array of size (rows x cols) containing uniform random [0, 1) values.
pub fn fill_rand(arr: &mut Vec<Vec<f64>>) {
    let mut rng = rand::thread_rng();
    for i in 0..arr.len() {
        for j in 0..arr[i].len() {
            arr[i][j] = rng.gen();
        }
    }
}

/// Returns a 2D array of size (rows x cols) containing uniform random [0, 1) values.
pub fn value_mask(arr: &Vec<Vec<f64>>, value: f64) -> Vec<Vec<bool>> {
    let rows = arr.len();
    let mut mask: Vec<Vec<bool>> = Vec::with_capacity(rows);
    for i in 0..rows {
        let cols = arr[i].len();
        let mut row: Vec<bool> = Vec::with_capacity(cols);
        for j in 0..cols {
            row.push(arr[i][j] == value);
        }
        mask.push(row);
    }
    mask
}

/// Returns two 2D indices arrays of size (rows x cols).
///
/// # Examples
///
/// ```
/// let (row_idx, cols_idx) = indices_arr(3, 3);
/// assert_eq!(rows_idx, [[0, 0, 0], [1, 1, 1], [2, 2, 2]]);
/// assert_eq!(cols_idx, [[0, 1, 2], [0, 1, 2], [0, 1, 2]]);
/// ```
pub fn indices_arr(rows: usize, cols: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut row_idx = zeros_arr(rows, cols);
    let mut cols_idx = zeros_arr(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            row_idx[i][j] = i as f64;
            cols_idx[i][j] = j as f64;
        }
    }
    (row_idx, cols_idx)
}

fn random_displace(disheight: f64, r: f64) -> f64 {
    r * disheight - 0.5 * disheight
}

fn displace_vals(arr: &mut Vec<f64>, disheight: f64, r: (f64, f64)) -> Option<f64> {
    if arr.len() == 4 {
        let sum: f64 = arr.iter().sum();
        return Some(sum * 0.25 + random_displace(disheight, r.0));
    } else if arr.len() == 3 {
        let sum: f64 = arr.iter().sum();
        return Some(sum / 3. + random_displace(disheight, r.1));
    }
    return None
}

fn check_diamond_coords(diax: i32, diay: i32, dim: i32, i2: i32) -> Vec<(i32, i32)> {
    if diax < 0 || diax > dim || diay < 0 || diay > dim {
        return vec![]
    } else if diax - i2 < 0 {
        return vec![(diax+i2, diay), (diax, diay-i2), (diax, diay+i2)]
    } else if diax + i2 >= dim {
        return vec![(diax-i2, diay), (diax, diay-i2), (diax, diay+i2)]
    } else if diay - i2 < 0 {
        return vec![(diax+i2, diay), (diax-i2, diay), (diax, diay+i2)]
    } else if diay + i2 >= dim {
        return vec![(diax+i2, diay), (diax-i2, diay), (diax, diay-i2)]
    }
    return vec![(diax+i2, diay), (diax-i2, diay), (diax, diay-i2), (diax, diay+i2)]
}

fn diamond_step(surface: &mut Vec<Vec<f64>>, disheight: f64, mid: usize, dim: usize, rng: &mut ThreadRng, diax: usize, diay: usize) {
    let diaco = check_diamond_coords(diax as i32, diay as i32, dim as i32, mid as i32);
    let mut diavals = vec![0f64; diaco.len()];
    for c in 0..diavals.len() {
        diavals[c] = surface[diaco[c].0 as usize][diaco[c].1 as usize];
    }
    let r = rng.gen();
    surface[diax][diay] = displace_vals(&mut diavals, disheight, r).unwrap();

}

/// Returns two 2D indices arrays of size (dim x dim).
///
/// Implementation ported from NLMpy.
pub fn diamond_square(dim: usize, h: f64) -> Vec<Vec<f64>> {
    let mut disheight: f64 = 2.;
    let mut surface = rand_arr(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            surface[i][j] *= disheight;
            surface[i][j] -= 0.5 * disheight;
        }
    }

    let mut inc = dim - 1;
    let mut rng = rand::thread_rng();
    while inc > 1 {
        let mid = inc/2;  // Centre point
        
        // Square
        for i in (0..dim-1).step_by(inc) {
            for j in (0..dim-1).step_by(inc) {
                let mut vec = vec![surface[i][j]];
                if i + inc < dim {
                    vec.push(surface[i+inc][j]);
                    if j + inc < dim {
                        vec.push(surface[i+inc][j+inc]);
                    }
                } else if j + inc < dim {
                    vec.push(surface[i+inc][j+inc]);
                }
                let r = rng.gen();
                surface[i+mid][j+mid] = displace_vals(&mut vec, disheight, r).unwrap();
            }
        }

        // Diamond
        for i in (0..dim-1).step_by(inc) {
            for j in (0..dim-1).step_by(inc) {
                diamond_step(&mut surface, disheight, mid, dim, &mut rng, i+mid, j);
                diamond_step(&mut surface, disheight, mid, dim, &mut rng, i, j+mid);
                diamond_step(&mut surface, disheight, mid, dim, &mut rng, i+inc, j+mid);
                diamond_step(&mut surface, disheight, mid, dim, &mut rng, i+mid, j+inc);
            }
        }

        disheight = disheight.powf(-h) * 2.;
        inc /= 2;
    }
    surface
}


/// Selects a random subarray of size( rows x cols) from the arr.
pub fn rand_sub_arr(arr: Vec<Vec<f64>>, rows: usize, cols: usize) -> Vec<Vec<f64>> {
    // If original array larger than target size, return original
    if rows >= arr.len() || (arr.len() > 0 && cols >= arr[0].len()) {
        return arr;
    }

    let mut rng = rand::thread_rng();
    let row_start = rng.gen_range(0..(arr.len() - rows));
    let col_start = rng.gen_range(0..(arr[0].len() - cols));

    let mut sub_arr = vec![vec![0f64; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            sub_arr[i][j] = arr[row_start + i][col_start + j];
        }
    }
    sub_arr
}
