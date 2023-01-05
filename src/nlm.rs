use rand::Rng;
use crate::array::{rand_arr, ones_arr, value_mask};
use crate::operation::{max, interpolate, scale};

/// Returns a spatially random NLM with values ranging [0, 1).
///
/// # Arguments
///
/// * `rows` - Number of rows in the array.
/// * `cols` - Number of columns in the array.
#[allow(dead_code)]
pub fn random(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    rand_arr(rows, cols)
}

/// Returns a random cluster nearest-neighbour NLM with values ranging [0, 1).
///
/// # Arguments
///
/// * `rows` - Number of rows in the array.
/// * `cols` - Number of columns in the array.
#[allow(dead_code)]
pub fn random_cluster(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let arr = rand_arr(rows, cols);
    arr
}

/// Returns a random element nearest-neighbour NLM with values ranging [0, 1).
///
/// # Arguments
///
/// * `rows` - Number of rows in the array.
/// * `cols` - Number of columns in the array.
#[allow(dead_code)]
pub fn random_element(rows: usize, cols: usize, n: u32) -> Vec<Vec<f32>> {
    let mut arr = ones_arr(rows, cols);

    let mut rng = rand::thread_rng();
    let mut i: f32 = 1.0;
    while max(&arr) < n as f32 {
        let row = rng.gen_range(0..rows);
        let col = rng.gen_range(0..cols);
        if arr[row][col] == 1.0 {
            arr[row][col] = i;
        }
        i += 1.0;
    }

    let mask = value_mask(&arr, 1.0 as f32);
    interpolate(&mut arr, mask);
    scale(&mut arr);

    arr
}