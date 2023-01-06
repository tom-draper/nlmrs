use crate::array::{indices_arr, ones_arr, rand_arr, value_mask, zeros_arr, diamond_square};
use crate::operation::{interpolate, max, scale, euclidean_distance_transform, invert};
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
/// 
/// Implementation ported from NLMpy.
#[allow(dead_code)]
pub fn random_element(rows: usize, cols: usize, n: f32) -> Vec<Vec<f32>> {
    let mut arr = ones_arr(rows, cols);

    let mut rng = rand::thread_rng();
    let mut i: f32 = 1.;
    while max(&arr) < n && i < n {
        let row = rng.gen_range(0..rows);
        let col = rng.gen_range(0..cols);
        if arr[row][col] == 1. {
            arr[row][col] = i;
        }
        i += 1.;
    }

    let mask = value_mask(&arr, 0.);
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
/// * `direction` - Direction of the gradient in degrees [0, 360).
/// 
/// Implementation ported from NLMpy.
#[allow(dead_code)]
pub fn planar_gradient(rows: usize, cols: usize, direction: Option<f32>) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let d = direction.unwrap_or(rng.gen_range(0.0..360.0));
    let right = d.sin();
    let down = -d.cos();
    // Two 2D indices arrays each of size (rows x cols)
    let (row_idx, col_idx) = indices_arr(rows, cols);

    // Build gradient array
    let mut gradient = zeros_arr(rows, cols);
    for i in 0..rows {
        for j in 0..cols{
            gradient[i][j] = (row_idx[i][j] * down) + (col_idx[i][j] * right)
        }
    }

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
/// * `direction` - Direction of the gradient in degrees [0.0, 360.0).
/// 
/// Implementation ported from NLMpy.
#[allow(dead_code)]
pub fn edge_gradient(rows: usize, cols: usize, direction: Option<f32>) -> Vec<Vec<f32>> {
    let mut arr = planar_gradient(rows, cols, direction);
    for i in 0..arr.len() {
        for j in 0..arr[i].len() {
            arr[i][j] = -(2. * (arr[i][j] - 0.5).abs()) + 1.;
        }
    }
    scale(&mut arr);
    arr
}

/// Returns a distance gradient NLM with values ranging [0, 1).
///
/// A distance gradient with a 0 value at a single point, with the gradient
/// emanating from this point.
///
/// # Arguments
///
/// * `rows` - Number of rows in the array.
/// * `cols` - Number of columns in the array.
/// 
/// Implementation ported from NLMpy.
#[allow(dead_code)]
pub fn distance_gradient(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let mut arr = rand_arr(rows, cols);
    invert(&mut arr);
    euclidean_distance_transform(&mut arr);
    scale(&mut arr);
    arr
}

/// Returns a wave gradient NLM with values ranging [0, 1).
///
/// A wave gradient cycles through 0->1->0... repeatedly from one end of the
/// array to the other. The gradient falls across the array in a random direction.
///
/// # Arguments
///
/// * `rows` - Number of rows in the array.
/// * `cols` - Number of columns in the array.
/// * `period` - Period of the wave function (smaller = larger wave).
/// * `direction` - Direction of the gradient in degrees [0, 360).
/// 
/// Implementation ported from NLMpy.
#[allow(dead_code)]
pub fn wave_gradient(rows: usize, cols: usize, period: f32, direction: Option<f32>) -> Vec<Vec<f32>> {
    let mut arr = planar_gradient(rows, cols, direction);
    for i in 0..arr.len() {
        for j in 0..arr[i].len() {
            arr[i][j] = (arr[i][j] * 2. * std::f32::consts::PI * period).sin();
        }
    }
    scale(&mut arr);
    arr
}

/// Returns a midpoint displacement NLM with values ranging [0, 1).
///
///
/// # Arguments
///
/// * `rows` - Number of rows in the array.
/// * `cols` - Number of columns in the array.
/// * `h` - Controls the spatial autocorrelation in element values.
/// 
/// Implementation ported from NLMpy.
#[allow(dead_code)]
pub fn midpoint_displacement(rows: usize, cols: usize, h: f32) -> Vec<Vec<f32>> {
    let max_dim = std::cmp::max(rows, cols);
    let n = ((max_dim - 1) as f32).log2().ceil() as usize;
    let dim = n.pow(2) + 1;

    let mut surface = diamond_square(dim, h);

    scale(&mut surface);
    surface
}
