use crate::array::{indices_arr, ones_arr, rand_arr, value_mask};
use crate::operation::{add, interpolate, max, multiply_value, scale};
use rand::Rng;

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
    // TODO
    arr
}

/// Returns a random element nearest-neighbour NLM with values ranging [0, 1).
///
/// # Arguments
///
/// * `rows` - Number of rows in the array.
/// * `cols` - Number of columns in the array.
#[allow(dead_code)]
pub fn random_element(rows: usize, cols: usize, n: f32) -> Vec<Vec<f32>> {
    let mut arr = ones_arr(rows, cols);

    let mut rng = rand::thread_rng();
    let mut i: f32 = 1.0;
    while max(&arr) < n && i < n {
        let row = rng.gen_range(0..rows);
        let col = rng.gen_range(0..cols);
        if arr[row][col] == 1.0 {
            arr[row][col] = i;
        }
        i += 1.0;
    }

    let mask = value_mask(&arr, 0.0);
    interpolate(&mut arr, mask);

    scale(&mut arr);
    arr
}

/// Returns a planar gradient NLM with values ranging [0, 1).
///
/// A planar gradient with 0 and 1 values on directly opposite ends of the 2D
/// array. The gradient falls across the array in a random direction.
///
/// # Arguments
///
/// * `rows` - Number of rows in the array.
/// * `cols` - Number of columns in the array.
#[allow(dead_code)]
pub fn planar_gradient(rows: usize, cols: usize, direction: Option<f32>) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let d = direction.unwrap_or(rng.gen_range(0.0..360.0));
    let right = d.sin() as f32;
    let down = -d.cos() as f32;
    // Two 2D indices arrays each of size (rows x cols)
    let (mut row_idx, mut col_idx) = indices_arr(rows, cols);

    // Build gradient array
    multiply_value(&mut row_idx, down);
    multiply_value(&mut col_idx, right);
    // Combine two directions to create final gradient array
    add(&mut row_idx, col_idx);
    let mut gradient = row_idx;

    scale(&mut gradient);
    gradient
}

/// Returns an edge gradient NLM with values ranging [0, 1).
///
/// A edge gradient with 0 values on directly opposite ends of the 2D array,
/// with 1 in the midpoint between the two. The gradient falls across the array
/// in a random direction.
///
/// # Arguments
///
/// * `rows` - Number of rows in the array.
/// * `cols` - Number of columns in the array.
#[allow(dead_code)]
pub fn edge_gradient(rows: usize, cols: usize, direction: Option<f32>) -> Vec<Vec<f32>> {
    let mut arr = planar_gradient(rows, cols, direction);
    for i in 0..arr.len() {
        for j in 0..arr[i].len() {
            arr[i][j] = -(2.0 * (arr[i][j] - 0.5).abs()) + 1.0;
        }
    }
    scale(&mut arr);
    arr
}

/// Returns an wave gradient NLM with values ranging [0, 1).
///
/// A wave gradient cycles through 0->1->0... repeatedly from one end of the
/// array to the other. The gradient falls across the array in a random direction.
///
/// # Arguments
///
/// * `rows` - Number of rows in the array.
/// * `cols` - Number of columns in the array.
/// * `period` - Period of the wave function. Smaller = larger wave.
#[allow(dead_code)]
pub fn wave_gradient(rows: usize, cols: usize, period: f32, direction: Option<f32>) -> Vec<Vec<f32>> {
    let mut arr = planar_gradient(rows, cols, direction);
    for i in 0..arr.len() {
        for j in 0..arr[i].len() {
            arr[i][j] = (arr[i][j] * 2.0 * std::f32::consts::PI * period).sin();
        }
    }
    scale(&mut arr);
    arr
}
