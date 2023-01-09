use csv::Writer;

pub fn display_arr(arr: Vec<Vec<f64>>) {
    for i in 0..arr.len() {
        println!("{:?}", arr[i]);
    }
}

pub fn write_to_csv(arr: Vec<Vec<f64>>) {
    let mut wtr = Writer::from_path("data/data.csv").unwrap();
    for row in arr.iter() {
        let _ = wtr.serialize(row);
    }
}
