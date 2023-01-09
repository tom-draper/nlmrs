use rand::prelude::*;

/// Returns a 2D array of size (rows x cols) containing zeros.
pub fn zeros_arr(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let mut vec: Vec<Vec<f32>> = Vec::with_capacity(rows);
    for _ in 0..rows {
        let row = vec![0f32; cols];
        vec.push(row);
    }
    vec
}

/// Returns a 2D array of size (rows x cols) containing ones.
pub fn ones_arr(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let mut vec: Vec<Vec<f32>> = Vec::with_capacity(rows);
    for _ in 0..rows {
        let row = vec![1f32; cols];
        vec.push(row);
    }
    vec
}

/// Returns a 2D array of size (rows x cols) containing uniform random [0, 1) values.
pub fn rand_arr(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let mut vec: Vec<Vec<f32>> = Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for _ in 0..cols {
            let value: f32 = rng.gen();
            row.push(value);
        }
        vec.push(row);
    }
    vec
}

/// Returns a 2D array of size (rows x cols) containing uniform random [0, 1) values.
pub fn value_mask(arr: &Vec<Vec<f32>>, value: f32) -> Vec<Vec<bool>> {
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
pub fn indices_arr(rows: usize, cols: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut row_idx = zeros_arr(rows, cols);
    let mut cols_idx = zeros_arr(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            row_idx[i][j] = i as f32;
            cols_idx[i][j] = j as f32;
        }
    }
    (row_idx, cols_idx)
}

fn random_displace(disheight: f32, r: f32) -> f32 {
    r * disheight - 0.5 * disheight
}

fn displace_vals(arr: &mut Vec<f32>, disheight: f32, r: (f32, f32)) -> Option<f32> {
    if arr.len() == 4 {
        let sum: f32 = arr.iter().sum();
        return Some(sum * 0.25 + random_displace(disheight, r.0));
    } else if arr.len() == 3 {
        let sum: f32 = arr.iter().sum();
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

fn diamond_step(surface: &mut Vec<Vec<f32>>, disheight: f32, mid: usize, dim: usize, rng: &mut ThreadRng, diax: usize, diay: usize) {
    let diaco = check_diamond_coords(diax as i32, diay as i32, dim as i32, mid as i32);
    let mut diavals = vec![0f32; diaco.len()];
    for c in 0..diavals.len() {
        diavals[c] = surface[diaco[c].0 as usize][diaco[c].1 as usize];
    }
    let r = rng.gen();
    surface[diax][diay] = displace_vals(&mut diavals, disheight, r).unwrap();

}

/// Returns two 2D indices arrays of size (dim x dim).
pub fn diamond_square(dim: usize, h: f32) -> Vec<Vec<f32>> {
    let mut disheight: f32 = 2.;
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
        
        println!("{} {}", inc, mid);
        
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

        disheight = (disheight * 2.).powf(-h);
        inc /= 2;
    }
    surface
}