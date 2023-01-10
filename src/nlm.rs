mod array;
mod operation;
pub mod export;
use array::{binary_rand_arr, value_arr};
use rand::prelude::*;
use rand::Rng;
use rand::distributions::WeightedIndex;
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
pub fn random_arr(arr: &mut[&mut[f64]], rows: usize, cols: usize) {
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
    let mut arr = binary_rand_arr(rows, cols);
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

fn apply_kernel(arr: &mut Vec<Vec<f64>>, row: usize, col: usize, kernel: &Vec<Vec<f64>>, factor: Option<f64>) {
    let f = factor.unwrap_or(0.1);

    let half = (kernel.len() as i32 - 1) / 2;
    let row_i = row as i32;
    let col_i = col as i32;

    // println!("{} {}", row, row_i);
    // println!("{} {}", col, col_i);

    for i in row_i-half..row_i+half+1 {
        for j in col_i-half..col_i+half+1 {
            if i >= 0 && j >= 0 {
                let iu = i as usize;
                let ju = j as usize;
                if iu < arr.len() && ju < arr[iu].len() {
                    // println!("-- {} {} {} {} {} {}", i, j, row_i-half, col_i-half, i - (row_i-half), j - (col_i-half));
                    arr[iu][ju] += kernel[(i - (row_i - half)) as usize][(j - (col_i - half)) as usize] * f
                }
            }
        }
    }

    // arr[row][col] += f;
    // if row > 0 {
    //     arr[row-1][col] += f * 0.5;
    // }
    // if col > 0 {
    //     arr[row][col-1] += f * 0.5;
    // }
    // if row < arr.len()-1 {
    //     arr[row+1][col] += f * 0.5;
    // }
    // if col < arr[0].len()-1 {
    //     arr[row][col+1] += f * 0.5;
    // }
}

fn hill_grow_next_point(arr: &Vec<Vec<f64>>, runaway: bool) -> (usize, usize) {
    let mut rng = rand::thread_rng();
    if runaway {
        let mut points= Vec::new();
        let mut weights = Vec::new();
        for i in 0..arr.len() {
            for j in 0..arr[i].len() {
                points.push((i, j));
                weights.push(arr[i][j]);
            }
        }

        let dist = WeightedIndex::new(&weights).unwrap();
        dist.sample(&mut rng);
        return points[dist.sample(&mut rng)];
    } else {
        let r = rng.gen_range(0..arr.len());
        let mut c = 0;
        if arr.len() > 0 {
            c = rng.gen_range(0..arr[0].len());
        }
        return (r, c)
    }
}

fn valid_kernel<T>(kernel: &Vec<Vec<T>>) -> bool {
    kernel.len() > 0 && kernel.len() == kernel[0].len() && kernel.len() % 2 == 1 && kernel[0].len() % 2 == 1
}

/// Returns a hill-grow NLM with values ranging [0, 1).
///
/// # Arguments
///
/// * `rows` - Number of rows in the array.
/// * `cols` - Number of columns in the array.
/// * `n` - Number of iterations.
/// * `runaway` - Whether probability of selection for hill location is proportional to 
/// the current height of each location. A value of `true` makes a new hills more likely to be grown further.
#[allow(dead_code)]
pub fn hill_grow(rows: usize, cols: usize, n: usize, runaway: bool, kernel: Option<&Vec<Vec<f64>>>) -> Vec<Vec<f64>> {
    let mut arr = value_arr(rows, cols, 0.5);
    
    let default_kernel = vec![vec![0., 0.5, 0.], vec![0.5, 1., 0.5], vec![0., 0.5, 0.]];
    let k = kernel.unwrap_or(&default_kernel);
    if !valid_kernel(k) {
        return arr
    }

    let mut rng = rand::thread_rng();
    let factor = 0.1;
    for _ in 0..n {
        let (r, c) = hill_grow_next_point(&arr, runaway);
        let grow = rng.gen_bool(0.5);
        if grow {
            apply_kernel(&mut arr, r, c,  k, Some(factor));
        } else {
            // Shrink
            apply_kernel(&mut arr, r, c,  k, Some(-factor));
        }
    };

    scale(&mut arr);
    arr
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
    #[case(0, 0, 50000, true, None)]
    #[case(1, 1, 50000, true, None)]
    #[case(2, 1, 50000, true, None)]
    #[case(3, 2, 50000, true, None)]
    #[case(4, 3, 50000, true, None)]
    #[case(5, 5, 50000, true, None)]
    #[case(10, 10, 50000, true, None)]
    #[case(100, 100, 50000, true, None)]
    #[case(500, 1000, 50000, true, None)]
    #[case(1000, 500, 50000, true, None)]
    #[case(1000, 1000, 50000, true, None)]
    #[case(2000, 2000, 50000, true, None)]
    fn test_hill_grow(#[case] rows: usize, #[case] cols: usize, #[case] n: usize, #[case] runaway: bool, #[case] kernel: Option<&Vec<Vec<f64>>>) {
        let arr = hill_grow(rows, cols, n, runaway, kernel);
        assert_eq!(arr.len(), rows);
        if arr.len() > 0 {
            assert_eq!(arr[0].len(), cols);
        }
        assert_eq!(nan_count(&arr), 0);
        assert_eq!(zero_to_one_count(&arr), rows*cols);
    }
}