use rand::prelude::*;

/// Returns a 2D array of size rows x cols containing zeros
pub fn zeros_arr(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let mut vec: Vec<Vec<f32>> = Vec::with_capacity(rows);
    for _ in 0..rows {
        let row = vec![0f32; cols];
        vec.push(row);
    }
    vec
}

/// Returns a 2D array of size rows x cols containing ones
pub fn ones_arr(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let mut vec: Vec<Vec<f32>> = Vec::with_capacity(rows);
    for _ in 0..rows {
        let row = vec![1f32; cols];
        vec.push(row);
    }
    vec
}

/// Returns a 2D array of size rows x cols containing uniform random [0, 1) values
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

/// Returns a 2D array of size rows x cols containing uniform random [0, 1) values
pub fn value_mask(arr: &Vec<Vec<f32>>, value: f32) -> Vec<Vec<bool>> {
    let rows = arr.len();
    let mut mask: Vec<Vec<bool>> = Vec::with_capacity(rows);
    for i in 0..rows{
        let cols = arr[i].len();
        let mut row: Vec<bool> = Vec::with_capacity(cols);
        for j in 0..cols {
            row.push(arr[i][j] == value);
        }
        mask.push(row);
    }
    mask
}

/// Returns two 2D indicies arrays of size rows x cols containing
/// 
/// # Examples
///
/// ```
/// let (row_idx, cols_idx) = indicies_arr(3, 3);
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