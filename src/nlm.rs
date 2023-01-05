use crate::array::rand_arr;

/// Returns a spatially random NLM with values ranging [0, 1).
///
/// # Arguments
///
/// * `rows` - Number of rows in the array.
/// * `cols` - Number of columns in the array.
pub fn nlm_random(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    rand_arr(rows, cols)
}
