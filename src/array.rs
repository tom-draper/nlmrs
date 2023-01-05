use rand::prelude::*;

pub fn zeros_arr(rows: usize, cols: usize) -> Vec<Vec<i32>> {
    let mut vec: Vec<Vec<i32>> = Vec::with_capacity(rows);
    for _ in 0..rows {
        let row = Vec::with_capacity(cols);
        vec.push(row);
    }
    vec
}

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