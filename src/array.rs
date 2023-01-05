use rand::prelude::*;

/// Returns a 2D array of size rows x cols containing zeros
#[allow(dead_code)]
pub fn zeros_arr(rows: usize, cols: usize) -> Vec<Vec<i32>> {
    let mut vec: Vec<Vec<i32>> = Vec::with_capacity(rows);
    for _ in 0..rows {
        let row = vec![0; cols];
        vec.push(row);
    }
    vec
}

/// Returns a 2D array of size rows x cols containing uniform random [0, 1) values
#[allow(dead_code)]
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