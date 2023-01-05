use rand::Rng;

pub fn max(arr: &Vec<Vec<f32>>) -> f32 {
    let mut max: f32 = 0.0;
    for row in arr.iter() {
        for val in row.iter() {
            if *val > max {
                max = *val;
            }
        }
    }
    max
}

pub fn min(arr: &Vec<Vec<f32>>) -> f32 {
    let mut min: f32 = 0.0;
    for row in arr.iter() {
        for val in row.iter() {
            if *val < min {
                min = *val;
            }
        }
    }
    min
}

fn nearest_neighbour(arr: &Vec<Vec<f32>>, row: usize, col: usize) -> f32 {
    let mut rng = rand::thread_rng();
    let mut options: Vec<f32> = Vec::new();
    if row < arr.len() - 1 {
        options.push(arr[row + 1][col]);
    } else if row > 0 {
        options.push(arr[row - 1][col]);
    } else if col < arr[0].len() - 1 {
        options.push(arr[row][col + 1]);
    } else if col > 0 {
        options.push(arr[row][col - 1]);
    }
    options[rng.gen_range(0..options.len())]
}

pub fn interpolate(arr: &mut Vec<Vec<f32>>, mask: Vec<Vec<bool>>) {
    for i in 0..arr.len() {
        for j in 0..arr[i].len() {
            if mask[i][j] {
                // Replace with nearest neighbour value
                arr[i][j] = nearest_neighbour(arr, i, j);
            }
        }
    }
}

pub fn scale(arr: &mut Vec<Vec<f32>>) {
    let max = max(&arr);
    let min = min(&arr);
    for i in 0..arr.len() {
        for j in 0..arr[i].len() {
            arr[i][j] = (arr[i][j] - min) / (max - min);
        }
    }
}

pub fn multiply_value(arr: &mut Vec<Vec<f32>>, value: f32) {
    for i in 0..arr.len() {
        for j in 0..arr[i].len() {
            arr[i][j] *= value;
        }
    }
}

pub fn multiply(arr: &mut Vec<Vec<f32>>, arr2: Vec<Vec<f32>>) {
    for i in 0..arr.len() {
        for j in 0..arr[i].len() {
            arr[i][j] *= arr2[i][j];
        }
    }
}

pub fn add(arr: &mut Vec<Vec<f32>>, arr2: Vec<Vec<f32>>) {
    for i in 0..arr.len() {
        for j in 0..arr[i].len() {
            arr[i][j] += arr2[i][j];
        }
    }
}

pub fn add_value(arr: &mut Vec<Vec<f32>>, value: f32) {
    for i in 0..arr.len() {
        for j in 0..arr[i].len() {
            arr[i][j] += value;
        }
    }
}

pub fn abs(arr: &mut Vec<Vec<f32>>) {
    for i in 0..arr.len() {
        for j in 0..arr[i].len() {
            if arr[i][j] < 0.0 {
                arr[i][j] *= -1.0;
            }
        }
    }
}
