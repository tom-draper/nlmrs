mod nlm;
mod output;
mod array;
mod operation;
use crate::output::{write_to_csv};

fn main() {
    let arr = nlm::nlm_random(100, 100);
    write_to_csv(arr);
}