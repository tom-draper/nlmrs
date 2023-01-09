use rand::Rng;

pub fn max(arr: &Vec<Vec<f64>>) -> f64 {
    let mut max: f64 = 0.0;
    for row in arr.iter() {
        for val in row.iter() {
            if *val > max {
                max = *val;
            }
        }
    }
    max
}

pub fn min(arr: &Vec<Vec<f64>>) -> f64 {
    let mut min: f64 = std::f64::INFINITY;
    for row in arr.iter() {
        for val in row.iter() {
            if *val < min {
                min = *val;
            }
        }
    }
    min
}

pub fn min_and_max(arr: &Vec<Vec<f64>>) -> (f64, f64) {
    let mut min: f64 = std::f64::INFINITY;
    let mut max: f64 = 0.0;
    for row in arr.iter() {
        for val in row.iter() {
            if *val < min {
                min = *val;
            }
            if *val > max {
                max = *val;
            }
        }
    }
    (min, max)
}

fn nearest_neighbour(arr: &Vec<Vec<f64>>, row: usize, col: usize) -> f64 {
    let mut options: Vec<f64> = Vec::new();
    if row < arr.len() - 1 {
        options.push(arr[row + 1][col]);
    } else if row > 0 {
        options.push(arr[row - 1][col]);
    } else if col < arr[0].len() - 1 {
        options.push(arr[row][col + 1]);
    } else if col > 0 {
        options.push(arr[row][col - 1]);
    }

    let mut rng = rand::thread_rng();
    options[rng.gen_range(0..options.len())]
}

pub fn interpolate(arr: &mut Vec<Vec<f64>>, mask: Vec<Vec<bool>>) {
    for i in 0..arr.len() {
        for j in 0..arr[i].len() {
            if mask[i][j] {
                // Replace with nearest non-zero neighbour value
                arr[i][j] = nearest_neighbour(arr, i, j);
            }
        }
    }
}

pub fn scale(arr: &mut Vec<Vec<f64>>) {
    let (min, max) = min_and_max(arr);
    let range = max - min;
    for i in 0..arr.len() {
        for j in 0..arr[i].len() {
            if range == 0. {
                arr[i][j] = 0.5;
            } else {
                arr[i][j] = (arr[i][j] - min) / range;
            }
        }
    }
}

fn euclidean_distance(x1: i32, y1: i32, x2: i32, y2: i32) -> f64 {
    let y = ((x2 - x1).pow(2) + (y2 - y1).pow(2)) as f64;
    y.sqrt()
}

fn nearest_non_zero(arr: &Vec<Vec<f64>>, row: usize, col: usize) -> (usize, usize) {
    let mut best = (std::f64::INFINITY, 0, 0);
    for i in 0..arr.len() {
        for j in 0..arr[i].len() {
            if row != i || col != j {
                let d = euclidean_distance(row as i32, col as i32, i as i32, j as i32);
                if d < best.0 {
                    best = (d, i, j);
                }
            }
        }
    }
    (best.1, best.2)
}

pub fn euclidean_distance_transform(arr: &mut Vec<Vec<f64>>) {
    for i in 0..arr.len() {
        for j in 0..arr[i].len() {
            if arr[i][j] != 0.0 {
                let (row, col) = nearest_non_zero(arr, i, j);
                arr[i][j] = euclidean_distance(row as i32, col as i32, i as i32, j as i32) as f64;
            }
        }
    }
}

pub fn invert(arr: &mut Vec<Vec<f64>>) {
    for i in 0..arr.len() {
        for j in 0..arr[i].len() {
            arr[i][j] = 1.0 - arr[i][j]
        }
    }
}

pub fn multiply_value(arr: &mut Vec<Vec<f64>>, value: f64) {
    for i in 0..arr.len() {
        for j in 0..arr[i].len() {
            arr[i][j] *= value;
        }
    }
}

pub fn multiply(arr: &mut Vec<Vec<f64>>, arr2: Vec<Vec<f64>>) {
    for i in 0..arr.len() {
        for j in 0..arr[i].len() {
            arr[i][j] *= arr2[i][j];
        }
    }
}

pub fn add(arr: &mut Vec<Vec<f64>>, arr2: Vec<Vec<f64>>) {
    for i in 0..arr.len() {
        for j in 0..arr[i].len() {
            arr[i][j] += arr2[i][j];
        }
    }
}

pub fn add_value(arr: &mut Vec<Vec<f64>>, value: f64) {
    for i in 0..arr.len() {
        for j in 0..arr[i].len() {
            arr[i][j] += value;
        }
    }
}

pub fn abs(arr: &mut Vec<Vec<f64>>) {
    for i in 0..arr.len() {
        for j in 0..arr[i].len() {
            if arr[i][j] < 0.0 {
                arr[i][j] *= -1.0;
            }
        }
    }
}
