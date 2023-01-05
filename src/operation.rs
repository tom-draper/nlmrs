use rand::Rng;


pub fn max(vec: &Vec<Vec<f32>>) -> f32 {
    let mut max: f32 = 0.0;
    for row in vec.iter() {
        for val in row.iter() {
            if *val > max {
                max = *val;
            }
        }
    }
    max
}

pub fn min(vec: &Vec<Vec<f32>>) -> f32 {
    let mut min: f32 = 0.0;
    for row in vec.iter() {
        for val in row.iter() {
            if *val < min {
                min = *val;
            }
        }
    }
    min
}

fn nearest_neighbour(vec: &Vec<Vec<f32>>, row: usize, col: usize) -> f32 {
    let mut rng = rand::thread_rng();
    let n = rng.gen_range(0..4);
    if n == 0 {
        return vec[row+1][col];
    } else if n == 1 {
        return vec[row-1][col];
    } else if n == 2 {
        return vec[row][col+1];
    } else  {
        return vec[row][col-1];
    }
}

pub fn interpolate(vec: &mut Vec<Vec<f32>>, mask: Vec<Vec<bool>>) {
    for i in 0..vec.len() {
        for j in 0..vec[i].len() {
            if mask[i][j] {
                // Replace with nearest neighbour
                vec[i][j] = nearest_neighbour(vec, i, j);
            }
        }
    }
}

pub fn scale(vec: &mut Vec<Vec<f32>>) {
    let max = max(&vec);
    let min = min(&vec);
    for i in 0..vec.len() {
        for j in 0..vec[i].len() {
            vec[i][j] = (vec[i][j] - min)/(max - min);
        }
    }
}