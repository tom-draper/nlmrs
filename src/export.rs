use serde_json;
use std::fs::File;
use std::io::{BufWriter, Write, Result};

use csv::Writer;

pub fn display_arr(arr: Vec<Vec<f64>>) {
    for i in 0..arr.len() {
        println!("{:?}", arr[i]);
    }
}

pub fn write_to_csv(arr: Vec<Vec<f64>>) -> Result<()> {
    let mut wtr = Writer::from_path("data/data.csv").unwrap();
    for row in arr.iter() {
        let _ = wtr.serialize(row);
    }
    Ok(())
}

pub fn write_to_json(arr: Vec<Vec<f64>>) -> Result<()> {
    let file = File::create("data/data.json")?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &arr)?;
    writer.flush()?;
    Ok(())
}
