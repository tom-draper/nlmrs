# NLMrs

A Rust crate for building <b>Neutral Landscape Models</b>.

<img src="https://user-images.githubusercontent.com/41476809/211358340-8e4d68de-fdc4-4e75-846b-b7b5c6105bfb.png" alt="" />

Inspired by [nlmpy](https://pypi.org/project/nlmpy/) and [nlmr](https://github.com/ropensci/NLMR).

## Installation

```bash
cargo add nlmrs
```

## Usage

```rs
use nlmrs;

fn main() {
    // All functions accept an optional seed for reproducible output.
    let grid = nlmrs::midpoint_displacement(100, 100, 1.0, Some(42));
    println!("{:?}", grid.data);
}
```

Pass `None` as the seed for a random result each run:

```rs
let grid = nlmrs::midpoint_displacement(100, 100, 1.0, None);
```

### Export

The `export` module provides functions to save a grid to disk.

```rs
use nlmrs::{midpoint_displacement, export};

fn main() {
    let grid = midpoint_displacement(200, 200, 0.8, Some(42));

    export::write_to_png(&grid, "examples/terrain.png").unwrap();           // terrain colormap
    export::write_to_png_grayscale(&grid, "examples/terrain_gray.png").unwrap();
    export::write_to_csv(&grid, "examples/terrain.csv").unwrap();
    export::write_to_json(&grid, "examples/terrain.json").unwrap();
}
```

Output format is determined by the file extension. The PNG terrain colormap maps values from deep water (low) through sand, grass, and rock to snow (high).

### Visualization

`scripts/visualize.py` reads a CSV or JSON file and renders it as a matplotlib heatmap. It defaults to `examples/example.csv` but accepts any path as a command-line argument:

```bash
python scripts/visualize.py                        # uses examples/example.csv
python scripts/visualize.py examples/terrain.png   # any supported file
```

### CLI

A command-line binary is included. Output format is inferred from the file extension (`.png`, `.csv`, `.json`).

```bash
cargo install nlmrs

nlmrs midpoint-displacement 200 200 --h 0.8 --seed 42 --output terrain.png
nlmrs fbm 300 300 --scale 6.0 --octaves 8 --seed 99 --output landscape.png
nlmrs hill-grow 200 200 --n 20000 --runaway --output hills.csv
nlmrs perlin 500 500 --scale 4.0 --grayscale --output noise.png

nlmrs --help   # list all subcommands and options
```

## Python bindings

NLMrs is available as a Python package. Every function returns a 2-D **numpy array**, so it slots directly into matplotlib, rasterio, or any scientific Python workflow.

### Install

```bash
pip install nlmrs
```

Or build from source (requires Rust and [maturin](https://github.com/PyO3/maturin)):

```bash
pip install maturin
maturin develop --features python   # editable install into the active venv
```

### Usage

```python
import nlmrs
import matplotlib.pyplot as plt

# All functions accept an optional seed for reproducible output.
grid = nlmrs.midpoint_displacement(200, 200, h=0.8, seed=42)  # numpy array (200, 200)

plt.imshow(grid, cmap="terrain")
plt.axis("off")
plt.show()
```

All parameters are keyword-friendly with sensible defaults:

```python
nlmrs.random(200, 200)
nlmrs.random_element(200, 200, n=50000.0)
nlmrs.planar_gradient(200, 200, direction=45.0)
nlmrs.edge_gradient(200, 200)
nlmrs.distance_gradient(200, 200)
nlmrs.wave_gradient(200, 200, period=2.5, direction=90.0)
nlmrs.midpoint_displacement(200, 200, h=1.0)
nlmrs.hill_grow(200, 200, n=10000, runaway=True)
nlmrs.perlin_noise(200, 200, scale=4.0)
nlmrs.fbm_noise(200, 200, scale=4.0, octaves=6, persistence=0.5, lacunarity=2.0)
```

The GIL is released during computation, so rayon uses all available cores even when called from Python threads. Compared to [nlmpy](https://pypi.org/project/nlmpy/) on CPython, NLMrs is typically an order of magnitude faster.

## Algorithms

All functions share the signature pattern:

```
algorithm(rows, cols, [...params], seed: Option<u64>) -> Grid
```

### Random

`random(rows: 100, cols: 100, seed: None)`

<img src="https://user-images.githubusercontent.com/41476809/211366758-60cd60fa-bc7d-4bfb-b514-1aca1870dfa8.png" alt="" width=300 />

### Random Element

`random_element(rows: 100, cols: 100, n: 50000., seed: None)`

<img src="https://user-images.githubusercontent.com/41476809/211366837-8912ddc8-a541-4060-b948-abb81aeb4c27.png" alt="" width=300 />

### Planar Gradient

`planar_gradient(rows: 100, cols: 100, direction: Some(60.), seed: None)`

<img src="https://user-images.githubusercontent.com/41476809/211367190-9ed6604f-7352-4fb8-a75a-957bd733e01a.png" alt="" width=300 />

### Edge Gradient

`edge_gradient(rows: 100, cols: 100, direction: Some(140.), seed: None)`

<img src="https://user-images.githubusercontent.com/41476809/211367676-e731dcc4-3da3-48d6-b7bf-d90b08a3645f.png" alt="" width=300 />

### Distance Gradient

`distance_gradient(rows: 100, cols: 100, seed: None)`

<img src="https://user-images.githubusercontent.com/41476809/211400760-8558426c-330b-4a01-9024-246b4432819a.png" alt="" width=300 />

### Wave Gradient

`wave_gradient(rows: 100, cols: 100, period: 2.5, direction: Some(90.), seed: None)`

<img src="https://user-images.githubusercontent.com/41476809/211368695-61bc245b-214c-4e7d-9f74-dcfc8dde0087.png" alt="" width=300 />

### Midpoint Displacement

`midpoint_displacement(rows: 100, cols: 100, h: 1., seed: None)`

<img src="https://user-images.githubusercontent.com/41476809/211368739-3bfd4026-f47b-4b15-92ab-38e48c963d87.png" alt="" width=300 />

### Hill Grow

`hill_grow(rows: 100, cols: 100, n: 10000, runaway: true, kernel: None, only_grow: false, seed: None)`

<img src="https://user-images.githubusercontent.com/41476809/211400491-79c4767e-3caf-44ca-a58f-859fbbf2a8c7.png" alt="" width=300 />

### Perlin Noise

`perlin_noise(rows: 100, cols: 100, scale: 4.0, seed: None)`

### fBm Noise

`fbm_noise(rows: 100, cols: 100, scale: 4.0, octaves: 6, persistence: 0.5, lacunarity: 2.0, seed: None)`

Fractal Brownian motion layers multiple octaves of Perlin noise for more natural-looking terrain detail.

### Ridged Noise

`ridged_noise(rows: 300, cols: 300, scale_factor: 4.0, octaves: 6, persistence: 0.5, lacunarity: 2.0, seed: Some(42))`

### Billow Noise

`billow_noise(rows: 300, cols: 300, scale_factor: 4.0, octaves: 6, persistence: 0.5, lacunarity: 2.0, seed: Some(42))`

### Worley Noise

`worley_noise(rows: 300, cols: 300, scale_factor: 4.0, seed: Some(42))`

### Gaussian Field

`gaussian_field(rows: 300, cols: 300, sigma: 15.0, seed: Some(42))`

### Random Cluster

`random_cluster(rows: 300, cols: 300, n: 200, seed: Some(42))`

## Grid operations

The `operation` module exposes combinators for building composite NLMs:

```rs
use nlmrs::{midpoint_displacement, planar_gradient, operation};

fn main() {
    let mut terrain = midpoint_displacement(200, 200, 0.8, Some(1));
    let gradient   = planar_gradient(200, 200, Some(90.), Some(2));

    operation::multiply(&mut terrain, &gradient);
    operation::scale(&mut terrain);
}
```

Available operations: `add`, `add_value`, `multiply`, `multiply_value`, `invert`, `abs`, `scale`, `min`, `max`, `min_and_max`.

## Contributions

Contributions, issues and feature requests are welcome.

- Fork it (https://github.com/tom-draper/nlmrs)
- Create your feature branch (`git checkout -b my-new-feature`)
- Commit your changes (`git commit -am 'Add some feature'`)
- Push to the branch (`git push origin my-new-feature`)
- Create a new Pull Request
