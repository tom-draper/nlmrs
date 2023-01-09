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

<img src="https://user-images.githubusercontent.com/41476809/211366758-60cd60fa-bc7d-4bfb-b514-1aca1870dfa8.png" alt="" width=300 />

### Random Element

<img src="https://user-images.githubusercontent.com/41476809/211366837-8912ddc8-a541-4060-b948-abb81aeb4c27.png" alt="" width=300 />

### Planar Gradient

<img src="https://user-images.githubusercontent.com/41476809/211367190-9ed6604f-7352-4fb8-a75a-957bd733e01a.png" alt="" width=300 />

### Edge Gradient

<img src="https://user-images.githubusercontent.com/41476809/211367676-e731dcc4-3da3-48d6-b7bf-d90b08a3645f.png" alt="" width=300 />

### Distance Gradient

### Wave Gradient

<img src="https://user-images.githubusercontent.com/41476809/211368695-61bc245b-214c-4e7d-9f74-dcfc8dde0087.png" alt="" width=300 />

### Midpoint Displacement

<img src="https://user-images.githubusercontent.com/41476809/211368739-3bfd4026-f47b-4b15-92ab-38e48c963d87.png" alt="" width=300 />

## Visualization

Running `script/viz.py` will read any contents of `data/data.csv` and render them in a matplotlib plot.

## Contributions

Contributions, issues and feature requests are welcome.

- Fork it (https://github.com/tom-draper/nlmrs)
- Create your feature branch (`git checkout -b my-new-feature`)
- Commit your changes (`git commit -am 'Add some feature'`)
- Push to the branch (`git push origin my-new-feature`)
- Create a new Pull Request
