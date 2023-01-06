mod array;
mod export;
mod nlm;
mod operation;
use std::time::Instant;

use crate::{export::write_to_csv};

fn main() {
    let start = Instant::now();
    let arr = nlm::midpoint_displacement(200, 200, 0.2);
    let elapsed = start.elapsed();
    println!("elapsed: {:?}", elapsed);
    write_to_csv(arr);
}
