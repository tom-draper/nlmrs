use csv::Writer;

#[allow(dead_code)]
pub fn display_arr(arr: Vec<Vec<f32>>) {
    for i in 0..arr.len() {
        println!("{:?}", arr[i]);
    }
}

#[allow(dead_code)]
pub fn write_to_csv(arr: Vec<Vec<f32>>) {
    let mut wtr = Writer::from_path("data/data.csv").unwrap();
    for row in arr {
        let _ = wtr.serialize(row);
    }
}
