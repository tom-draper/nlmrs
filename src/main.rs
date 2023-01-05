mod nlm;
mod export;
mod array;
mod operation;
use crate::export::{write_to_csv};

fn main() {
    let arr = nlm::random_element(200, 200, 100000);
    write_to_csv(arr);
}