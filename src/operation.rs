
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

pub fn interpolate(vec: &mut Vec<Vec<f32>>, mask: Vec<Vec<bool>>) {
    
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