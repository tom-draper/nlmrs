# NLMrs

A Rust crate for building **Neutral Landscape Models**.

<img src="https://user-images.githubusercontent.com/41476809/211358340-8e4d68de-fdc4-4e75-846b-b7b5c6105bfb.png" alt="" />

Inspired by [nlmpy](https://pypi.org/project/nlmpy/) and [nlmr](https://github.com/ropensci/NLMR).

## Usage

`nlmrs` can be installed as a Rust crate, but language bindings also exist for Python, R, WASM and C. 

```bash
cargo add nlmrs
```

```rs
use nlmrs;

fn main() {
    // All functions accept an optional seed for reproducible output.
    let grid = nlmrs::midpoint_displacement(100, 100, 1.0, Some(42));
    println!("{:?}", grid.data);
}
```

### Export

The `export` module provides functions to save a grid to disk.

```rs
use nlmrs::{midpoint_displacement, export};

fn main() {
    let grid = midpoint_displacement(100, 100, 0.8, Some(42));

    export::write_to_png(&grid, "terrain.png").unwrap();
    export::write_to_png_grayscale(&grid, "terrain_gray.png").unwrap();
    export::write_to_tiff(&grid, "terrain.tif").unwrap();
    export::write_to_csv(&grid, "terrain.csv").unwrap();
    export::write_to_json(&grid, "terrain.json").unwrap();
    export::write_to_ascii_grid(&grid, "terrain.asc").unwrap();
}
```

### CLI

A command-line binary is included. Output format is inferred from the file extension (`.png`, `.csv`, `.json`, `.tif`, `.asc`).

```bash
cargo install nlmrs

nlmrs midpoint-displacement 200 200 --h 0.8 --seed 42 --output terrain.png
nlmrs fbm 300 300 --scale 6.0 --octaves 8 --seed 99 --output landscape.png
nlmrs hill-grow 200 200 --n 20000 --runaway --output hills.csv
nlmrs perlin 500 500 --scale 4.0 --grayscale --output noise.png

nlmrs --help   # list all subcommands and options
```


### Random

`random(rows, cols, seed)`

<img src="examples/random.png" alt="" width=300 />

### Random Element

`random_element(rows, cols, n, seed)`

<img src="examples/random_element.png" alt="" width=300 />

*Source: [Etherington, Holland & O'Sullivan (2015)](https://doi.org/10.1111/2041-210X.12308)*

### Planar Gradient

`planar_gradient(rows, cols, direction, seed)`

<img src="examples/planar_gradient.png" alt="" width=300 />

### Edge Gradient

`edge_gradient(rows, cols, direction, seed)`

<img src="examples/edge_gradient.png" alt="" width=300 />

### Distance Gradient

`distance_gradient(rows, cols, seed)`

<img src="examples/distance_gradient.png" alt="" width=300 />

### Wave Gradient

`wave_gradient(rows, cols, period, direction, seed)`

<img src="examples/wave_gradient.png" alt="" width=300 />

### Midpoint Displacement

`midpoint_displacement(rows, cols, h, seed)`

<img src="examples/midpoint_displacement.png" alt="" width=300 />

*Source: [Fournier, Fussell & Carpenter (1982)](https://doi.org/10.1145/358523.358553)*

### Hill Grow

`hill_grow(rows, cols, n, runaway, kernel, only_grow, seed)`

<img src="examples/hill_grow.png" alt="" width=300 />

*Source: [Etherington, Holland & O'Sullivan (2015)](https://doi.org/10.1111/2041-210X.12308)*

### Perlin Noise

`perlin_noise(rows, cols, scale, seed)`

<img src="examples/perlin.png" alt="" width=300 />

*Source: [Perlin (1985)](https://doi.org/10.1145/325165.325247)*

### fBm Noise

`fbm_noise(rows, cols, scale, octaves, persistence, lacunarity, seed)`

Fractal Brownian motion layers multiple octaves of Perlin noise for more natural-looking terrain detail.

<img src="examples/fbm.png" alt="" width=300 />

*Source: [Mandelbrot & Van Ness (1968)](https://doi.org/10.1137/1010093); [Voss (1985)](https://doi.org/10.1007/978-3-642-84574-1_34)*

### Ridged Noise

`ridged_noise(rows, cols, scale, octaves, persistence, lacunarity, seed)`

<img src="examples/ridged.png" alt="" width=300 />

*Source: [Musgrave, Kolb & Mace (1989)](https://doi.org/10.1145/74334.74337)*

### Billow Noise

`billow_noise(rows, cols, scale, octaves, persistence, lacunarity, seed)`

<img src="examples/billow.png" alt="" width=300 />

*Source: Ebert et al. — Texturing and Modeling: A Procedural Approach (2002)*

### Worley Noise

`worley_noise(rows, cols, scale, seed)`

<img src="examples/worley.png" alt="" width=300 />

*Source: [Worley (1996)](https://doi.org/10.1145/237170.237267)*

### Gaussian Field

`gaussian_field(rows, cols, sigma, seed)`

<img src="examples/gaussian_field.png" alt="" width=300 />

### Random Cluster

`random_cluster(rows, cols, n, seed)`

<img src="examples/random_cluster.png" alt="" width=300 />

*Source: [Saura & Martínez-Millán (2000)](https://doi.org/10.1023/A:1008107902848)*

### Hybrid Noise

`hybrid_noise(rows, cols, scale, octaves, persistence, lacunarity, seed)`

Hybrid multifractal noise combines fBm-style layering with a multiplicative weighting that amplifies high-frequency detail near peaks.

<img src="examples/hybrid_noise.png" alt="" width=300 />

*Source: [Musgrave, Kolb & Mace (1989)](https://doi.org/10.1145/74334.74337)*

### Value Noise

`value_noise(rows, cols, scale, seed)`

Interpolated lattice noise — smoother and more rounded than Perlin noise.

<img src="examples/value_noise.png" alt="" width=300 />

### Turbulence

`turbulence(rows, cols, scale, octaves, persistence, lacunarity, seed)`

fBm with absolute-value folding per octave, producing sharp ridges and a storm-cloud appearance.

<img src="examples/turbulence.png" alt="" width=300 />

*Source: [Perlin (1985)](https://doi.org/10.1145/325165.325247)*

### Domain Warp

`domain_warp(rows, cols, scale, warp_strength, seed)`

Perlin noise sampled at coordinates displaced by a second Perlin field, producing organic swirling patterns.

<img src="examples/domain_warp.png" alt="" width=300 />

*Source: [Quilez (2002)](https://iquilezles.org/articles/warp/)*

### Mosaic

`mosaic(rows, cols, n, seed)`

Discrete Voronoi map — each region is a flat colour determined by its nearest seed point, producing a stained-glass or territory effect.

<img src="examples/mosaic.png" alt="" width=300 />

### Rectangular Cluster

`rectangular_cluster(rows, cols, n, seed)`

Overlapping random axis-aligned rectangles accumulated and scaled, producing blocky clustered patches.

<img src="examples/rectangular_cluster.png" alt="" width=300 />

## Python bindings

`nlmrs` is available as a Python package. Every function returns a 2D numpy array.

### Install

```bash
pip install nlmrs
```

Or build from source (requires Rust and [maturin](https://github.com/PyO3/maturin)):

```bash
maturin develop --features python   # editable install into the active venv
```

### Usage

```python
import nlmrs
import matplotlib.pyplot as plt

# All functions accept an optional seed for reproducible output.
grid = nlmrs.midpoint_displacement(100, 100, h=0.8, seed=42)  # numpy array (100, 100)

plt.imshow(grid, cmap="terrain")
plt.axis("off")
plt.show()
```

All parameters are keyword-friendly with sensible defaults:

```python
# Patch-based
nlmrs.random(100, 100)
nlmrs.random_element(100, 100, n=50000.0)
nlmrs.hill_grow(100, 100, n=10000, runaway=True)
nlmrs.midpoint_displacement(100, 100, h=1.0)
nlmrs.gaussian_field(100, 100, sigma=10.0)
nlmrs.random_cluster(100, 100, n=200)
nlmrs.mosaic(100, 100, n=200)
nlmrs.rectangular_cluster(100, 100, n=200)

# Gradient
nlmrs.planar_gradient(100, 100, direction=45.0)
nlmrs.edge_gradient(100, 100)
nlmrs.distance_gradient(100, 100)
nlmrs.wave_gradient(100, 100, period=2.5, direction=90.0)

# Noise
nlmrs.perlin_noise(100, 100, scale=4.0)
nlmrs.fbm_noise(100, 100, scale=4.0, octaves=6, persistence=0.5, lacunarity=2.0)
nlmrs.ridged_noise(100, 100, scale=4.0, octaves=6)
nlmrs.billow_noise(100, 100, scale=4.0, octaves=6)
nlmrs.worley_noise(100, 100, scale=4.0)
nlmrs.hybrid_noise(100, 100, scale=4.0, octaves=6)
nlmrs.value_noise(100, 100, scale=4.0)
nlmrs.turbulence(100, 100, scale=4.0, octaves=6)
nlmrs.domain_warp(100, 100, scale=4.0, warp_strength=1.0)
```

Post-processing functions are also available:

```python
grid = nlmrs.fbm_noise(100, 100, scale=4.0)
nlmrs.classify(grid, n=5)    # quantise into n equal-width classes
nlmrs.threshold(grid, t=0.5) # binarise at threshold t
```

## R bindings

`nlmrs` is available as an R package via the [extendr](https://extendr.github.io/) framework. Every function returns a numeric matrix.

### Install

```r
# Install from source (requires Rust)
remotes::install_github("tom-draper/nlmrs", subdir = "bindings/r")
```

### Usage

```r
library(nlmrs)

# All functions accept an optional integer seed.
m <- nlm_midpoint_displacement(100, 100, h = 0.8, seed = 42L)
image(m, col = terrain.colors(256))
```

All 21 algorithms are available with the `nlm_` prefix:

```r
nlm_random(100, 100)
nlm_random_element(100, 100, n = 50000)
nlm_planar_gradient(100, 100, direction = 45)
nlm_edge_gradient(100, 100)
nlm_distance_gradient(100, 100)
nlm_wave_gradient(100, 100, period = 2.5)
nlm_midpoint_displacement(100, 100, h = 1.0)
nlm_hill_grow(100, 100, n = 10000L, runaway = TRUE)
nlm_perlin_noise(100, 100, scale = 4.0)
nlm_fbm_noise(100, 100, scale = 4.0, octaves = 6L)
nlm_ridged_noise(100, 100, scale = 4.0, octaves = 6L)
nlm_billow_noise(100, 100, scale = 4.0, octaves = 6L)
nlm_worley_noise(100, 100, scale = 4.0)
nlm_gaussian_field(100, 100, sigma = 10.0)
nlm_random_cluster(100, 100, n = 200L)
nlm_hybrid_noise(100, 100, scale = 4.0, octaves = 6L)
nlm_value_noise(100, 100, scale = 4.0)
nlm_turbulence(100, 100, scale = 4.0, octaves = 6L)
nlm_domain_warp(100, 100, scale = 4.0, warp_strength = 1.0)
nlm_mosaic(100, 100, n = 200L)
nlm_rectangular_cluster(100, 100, n = 200L)
```

## WASM bindings

`nlmrs` can run in the browser or Node.js via WebAssembly.

### Build

```bash
cd bindings/wasm
wasm-pack build --target web
```

### Usage

```js
import init, * as nlmrs from "./pkg/nlmrs_wasm.js";

await init();

const grid = nlmrs.midpoint_displacement(100, 100, 0.8, 42);
console.log(grid.rows, grid.cols);   // 100 100

// Flat Float64Array in row-major order
const flat = grid.data;
const value = flat[r * grid.cols + c];

grid.free();  // release Rust memory
```

All 21 algorithms are available. Seeds are passed as plain integers (not BigInt). Omit the seed argument for random output.

## Grid operations

The `operation` module exposes combinators for building composite NLMs:

```rs
use nlmrs::{midpoint_displacement, planar_gradient, operation};

fn main() {
    let mut terrain = midpoint_displacement(100, 100, 0.8, Some(1));
    let gradient   = planar_gradient(100, 100, Some(90.), Some(2));

    operation::multiply(&mut terrain, &gradient);
    operation::scale(&mut terrain);
}
```

Available operations: `add`, `add_value`, `multiply`, `multiply_value`, `invert`, `abs`, `scale`, `min`, `max`, `min_and_max`, `classify`, `threshold`.

## Contributions

Contributions, issues and feature requests are welcome.

- Fork it (https://github.com/tom-draper/nlmrs)
- Create your feature branch (`git checkout -b my-new-feature`)
- Commit your changes (`git commit -am 'Add some feature')
- Push to the branch (`git push origin my-new-feature`)
- Create a new Pull Request
