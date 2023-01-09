mod array;
mod operation;
pub mod export;
use rand::Rng;
use crate::operation::{interpolate, max, scale, euclidean_distance_transform, invert};
use crate::array::{indices_arr, ones_arr, rand_arr, value_mask, zeros_arr, diamond_square, rand_sub_arr};

/// Returns a spatially random NLM with values ranging [0, 1).
///
/// # Arguments
///
/// * `rows` - Number of rows in the array.
/// * `cols` - Number of columns in the array.
#[allow(dead_code)]
pub fn random(rows: usize, cols: usize) -> Vec<Vec<f64>> {
    rand_arr(rows, cols)
}

/// Returns a spatially random NLM with values ranging [0, 1).
///
/// # Arguments
///
/// * `rows` - Number of rows in the array.
/// * `cols` - Number of columns in the array.
#[allow(dead_code)]
pub fn random_arr<'a>(arr: &mut[&mut[f64]], rows: usize, cols: usize){
    let mut rng = rand::thread_rng();
    for i in 0..rows {
        for j in 0..cols {
            arr[i][j] = rng.gen();
        }
    }
}

/// Returns a random cluster nearest-neighbour NLM with values ranging [0, 1).
///
/// # Arguments
///
/// * `rows` - Number of rows in the array.
/// * `cols` - Number of columns in the array.
#[allow(dead_code)]
pub fn random_cluster(rows: usize, cols: usize) -> Vec<Vec<f64>> {
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
pub fn random_element(rows: usize, cols: usize, n: f64) -> Vec<Vec<f64>> {
    if rows == 0 {
        return vec![]
    }

    let mut arr = ones_arr(rows, cols);

    let mut rng = rand::thread_rng();
    let mut i: f64 = 1.;
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
pub fn planar_gradient(rows: usize, cols: usize, direction: Option<f64>) -> Vec<Vec<f64>> {
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
pub fn edge_gradient(rows: usize, cols: usize, direction: Option<f64>) -> Vec<Vec<f64>> {
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
pub fn distance_gradient(rows: usize, cols: usize) -> Vec<Vec<f64>> {
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
pub fn wave_gradient(rows: usize, cols: usize, period: f64, direction: Option<f64>) -> Vec<Vec<f64>> {
    let mut arr = planar_gradient(rows, cols, direction);
    for i in 0..arr.len() {
        for j in 0..arr[i].len() {
            arr[i][j] = (arr[i][j] * 2. * std::f64::consts::PI * period).sin();
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
pub fn midpoint_displacement(rows: usize, cols: usize, h: f64) -> Vec<Vec<f64>> {
    let max_dim = std::cmp::max(rows, cols);
    if max_dim == 0 {
        return vec![]
    }
    let n = ((max_dim - 1) as f64).log2().ceil() as u32;
    let dim = usize::pow(2, n) + 1;
    
    let mut surface = diamond_square(dim, h);

    // Take random (row x cols) sub array
    surface = rand_sub_arr(surface, rows, cols);

    scale(&mut surface);
    surface
}


#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    
    fn nan_count(arr: &Vec<Vec<f64>>) -> usize {
        let mut count = 0;
        for i in 0..arr.len() {
            count += arr[i].iter().filter(|n| n.is_nan()).count();
        }
        count
    }

    fn zero_count(arr: &Vec<Vec<f64>>) -> usize {
        let mut count = 0;
        for i in 0..arr.len() {
            count += arr[i].iter().filter(|n| **n == 0.).count();
        }
        count
    }

    fn one_count(arr: &Vec<Vec<f64>>) -> usize {
        let mut count = 0;
        for i in 0..arr.len() {
            count += arr[i].iter().filter(|n| **n == 1.).count();
        }
        count
    }

    fn zero_to_one_count(arr: &Vec<Vec<f64>>) -> usize {
        let mut count = 0;
        for i in 0..arr.len() {
            count += arr[i].iter().filter(|n| **n >= 0. && **n <= 1.).count();
        }
        count
    }

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(2, 1)]
    #[case(3, 2)]
    #[case(4, 3)]
    #[case(5, 5)]
    #[case(10, 10)]
    #[case(100, 100)]
    #[case(500, 1000)]
    #[case(1000, 500)]
    #[case(1000, 1000)]
    #[case(2000, 2000)]
    fn test_random(#[case] rows: usize, #[case] cols: usize) {
        let arr = random(rows, cols);
        assert_eq!(arr.len(), rows);
        if arr.len() > 0 {
            assert_eq!(arr[0].len(), cols);
        }
        assert_eq!(nan_count(&arr), 0);
        assert_eq!(zero_to_one_count(&arr), rows*cols);
    }
    
    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(2, 1)]
    #[case(3, 2)]
    #[case(4, 3)]
    #[case(5, 5)]
    #[case(10, 10)]
    #[case(100, 100)]
    #[case(500, 1000)]
    #[case(1000, 500)]
    #[case(1000, 1000)]
    #[case(2000, 2000)]
    fn test_random_element(#[case] rows: usize, #[case] cols: usize) {
        let arr = random_element(rows, cols, 900.);
        assert_eq!(arr.len(), rows);
        if arr.len() > 0 {
            assert_eq!(arr[0].len(), cols);
        }
        assert_eq!(nan_count(&arr), 0);
        assert_eq!(zero_to_one_count(&arr), rows*cols);
    }
    
    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(2, 1)]
    #[case(3, 2)]
    #[case(4, 3)]
    #[case(5, 5)]
    #[case(10, 10)]
    #[case(100, 100)]
    #[case(500, 1000)]
    #[case(1000, 500)]
    #[case(1000, 1000)]
    #[case(2000, 2000)]
    fn test_planar_gradient(#[case] rows: usize, #[case] cols: usize) {
        let arr = planar_gradient(rows, cols, Some(90.));
        assert_eq!(arr.len(), rows);
        if arr.len() > 0 {
            assert_eq!(arr[0].len(), cols);
        }
        assert_eq!(nan_count(&arr), 0);
        assert_eq!(zero_to_one_count(&arr), rows*cols);
    }
    
    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(2, 1)]
    #[case(3, 2)]
    #[case(4, 3)]
    #[case(5, 5)]
    #[case(10, 10)]
    #[case(100, 100)]
    #[case(500, 1000)]
    #[case(1000, 500)]
    #[case(1000, 1000)]
    #[case(2000, 2000)]
    fn test_edge_gradient(#[case] rows: usize, #[case] cols: usize) {
        let arr = edge_gradient(rows, cols, Some(90.));
        assert_eq!(arr.len(), rows);
        if arr.len() > 0 {
            assert_eq!(arr[0].len(), cols);
        }
        assert_eq!(nan_count(&arr), 0);
        assert_eq!(zero_to_one_count(&arr), rows*cols);
    }
    
    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(2, 1)]
    #[case(3, 2)]
    #[case(4, 3)]
    #[case(5, 5)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_distance_gradient(#[case] rows: usize, #[case] cols: usize) {
        let arr = distance_gradient(rows, cols);
        assert_eq!(arr.len(), rows);
        if arr.len() > 0 {
            assert_eq!(arr[0].len(), cols);
        }
        assert_eq!(nan_count(&arr), 0);
        assert_eq!(zero_to_one_count(&arr), rows*cols);
    }
    
    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(2, 1)]
    #[case(3, 2)]
    #[case(4, 3)]
    #[case(5, 5)]
    #[case(10, 10)]
    #[case(100, 100)]
    #[case(500, 1000)]
    #[case(1000, 500)]
    #[case(1000, 1000)]
    #[case(2000, 2000)]
    fn test_wave_gradient(#[case] rows: usize, #[case] cols: usize) {
        let arr = wave_gradient(rows, cols, 2.0, Some(90.));
        assert_eq!(arr.len(), rows);
        if arr.len() > 0 {
            assert_eq!(arr[0].len(), cols);
        }
        assert_eq!(nan_count(&arr), 0);
        assert_eq!(zero_to_one_count(&arr), rows*cols);
        if rows > 0 && cols > 0 && (rows > 1 || cols > 1) {
            assert_eq!(one_count(&arr), 1);
            assert_eq!(zero_count(&arr), 1);
        }
    }
    
    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(2, 1)]
    #[case(3, 2)]
    #[case(4, 3)]
    #[case(5, 5)]
    #[case(10, 10)]
    #[case(100, 100)]
    #[case(500, 1000)]
    #[case(1000, 500)]
    #[case(1000, 1000)]
    #[case(2000, 2000)]
    fn test_midpoint_displacement(#[case] rows: usize, #[case] cols: usize) {
        let arr = midpoint_displacement(rows, cols, 1.);
        assert_eq!(arr.len(), rows);
        if arr.len() > 0 {
            assert_eq!(arr[0].len(), cols);
        }
        assert_eq!(nan_count(&arr), 0);
        assert_eq!(zero_to_one_count(&arr), rows*cols);
    }
    
    #[rstest]
    #[case(0, 0, 2.)]
    #[case(1, 1, 1.5)]
    #[case(2, 1, 1.75)]
    #[case(3, 2, 3.5)]
    #[case(4, 3, 15.)]
    #[case(5, 5, 0.125)]
    #[case(10, 10, 0.1)]
    #[case(100, 100, 0.3)]
    #[case(500, 1000, 0.4)]
    #[case(1000, 500, 0.5)]
    #[case(1000, 1000, 0.1)]
    #[case(2000, 2000, 0.2)]
    fn test_diamond_square(#[case] rows: usize, #[case] cols: usize, #[case] h: f64) {
        let max_dim = std::cmp::max(rows, cols);
        if max_dim == 0 {
            return
        }
        let n = ((max_dim - 1) as f64).log2().ceil() as u32;
        let dim = usize::pow(2, n) + 1;

        let arr = diamond_square(dim, h);
        assert_eq!(arr.len(), rows);
        if arr.len() > 0 {
            assert_eq!(arr[0].len(), cols);
        }
        assert_eq!(nan_count(&arr), 0);
    }

    #[test]
    fn test_write_to_csv() {
        // let arr = random(100, 100);
        // let arr = random_element(100, 100, 50000.);
        // let arr = planar_gradient(100, 100, Some(60.));
        // let arr = edge_gradient(100, 100, Some(140.));
        // let arr = distance_gradient(100, 100);
        // let arr = wave_gradient(100, 100, 2.5, Some(90.));
        let arr = midpoint_displacement(100, 100, 1.);
        export::write_to_csv(arr);
    }
}