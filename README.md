# NLMrs

A Rust crate for building <b>Neutral Landscape Models</b>.

<img src="https://user-images.githubusercontent.com/41476809/211358340-8e4d68de-fdc4-4e75-846b-b7b5c6105bfb.png" alt="" />

Inspired by [nlmpy](https://pypi.org/project/nlmpy/) and [nlmr](https://github.com/ropensci/NLMR).

## Installation

```bash
cargo add nlmrs
```

## Example

```rs
use nlmrs;

fn main() {
    let arr: Vec<Vec<f64>> = nlmrs::midpoint_displacement(10, 10, 1.);
    println!("{:?}", arr);
}
```

### Export

The `export` module holds a collection of user-friendly functions to export your 2D NLM vector.

```rs
use nlmrs::{distance_gradient, export::write_to_csv};

fn main() {
    let arr: Vec<Vec<f64>> = distance_gradient(50, 50);
    write_to_csv(arr, "./data/data.csv");
}
```

## Algorithms

### Random

### Random Element

### Planar Gradient

### Edge Gradient

### Distance Gradient

### Wave Gradient

### Midpoint Displacement

## Visualisation

Running `script/vis.py` will read any contents of `data/data.csv` and render them in a matplotlib plot.

## Contributions

Contributions, issues and feature requests are welcome.

- Fork it (https://github.com/tom-draper/nlmrs)
- Create your feature branch (`git checkout -b my-new-feature`)
- Commit your changes (`git commit -am 'Add some feature'`)
- Push to the branch (`git push origin my-new-feature`)
- Create a new Pull Request
