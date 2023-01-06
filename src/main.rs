mod array;
mod export;
mod nlm;
mod operation;
use std::time::Instant;

use crate::export::write_to_csv;

fn main() {
    let start = Instant::now();
    let arr = nlm::wave_gradient(200, 200, 2.0, None);
    let elapsed = start.elapsed();
    println!("elapsed: {:?}", elapsed);
    write_to_csv(arr);
}
