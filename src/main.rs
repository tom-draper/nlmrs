mod nlm;
mod export;
mod array;
mod operation;
use crate::export::{write_to_csv};

fn main() {
    let arr = nlm::nlm_random(100, 100);
    write_to_csv(arr);
}