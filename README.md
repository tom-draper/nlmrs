# NLMrs

A Rust crate for building **Neutral Landscape Models**.

<img src="https://user-images.githubusercontent.com/41476809/211358340-8e4d68de-fdc4-4e75-846b-b7b5c6105bfb.png" alt="" />

Inspired by [nlmpy](https://pypi.org/project/nlmpy/) and [nlmr](https://github.com/ropensci/NLMR).

NLMrs is available as a Rust crate or as a CLI tool. Language bindings are also provided for Python, R, Julia, WASM, and C.

## Installation

```bash
cargo add nlmrs
```

## Example

```rs
use nlmrs;

fn main() {
    let arr = nlmrs::midpoint_displacement(10, 10, 1.);
    println!("{:?}", arr);
}
```

### Export

The `export` module holds a collection of user-friendly functions to export your 2D NLM vector.

```rs
use nlmrs::{distance_gradient, export};

fn main() {
    let arr = distance_gradient(50, 50);
    export::write_to_csv(arr, "./data/data.csv");
}
```

## Algorithms

### Gradient

Deterministic spatial fields derived from direction, distance, or position.

#### Planar Gradient

`planar_gradient(rows: 100, cols: 100, direction: 45.0, seed: 42)`

Linear ramp at a given `direction` angle, increasing uniformly from 0 to 1 across the grid.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/planar_gradient.png" alt="" width=300 />

#### Edge Gradient

`edge_gradient(rows: 100, cols: 100, direction: 45.0, seed: 42)`

Symmetric version of the planar gradient: values peak at 1.0 along the central axis and fall to 0.0 at both edges.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/edge_gradient.png" alt="" width=300 />

#### Checkerboard

`checkerboard(rows: 100, cols: 100, scale: 10, seed: 42)`

Deterministic alternating binary pattern of axis-aligned squares with side length `scale` cells. Canonical control landscape for spatial autocorrelation analysis.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/checkerboard.png" alt="" width=300 />

#### Distance Gradient

`distance_gradient(rows: 100, cols: 100, seed: 42)`

Euclidean distance transform from random seed cells, producing a smooth radial falloff from 0 at the seeds outward to 1 at the most distant point.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/distance_gradient.png" alt="" width=300 />

#### Landscape Gradient

`landscape_gradient(rows: 100, cols: 100, direction: 45.0, aspect: 2.0, seed: 42)`

Elliptical gradient centred at the grid midpoint. `direction` orients the major axis; `aspect` controls elongation (1.0 = circular). More flexible than `distance_gradient`.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/landscape_gradient.png" alt="" width=300 />

#### Concentric Rings

`concentric_rings(rows: 100, cols: 100, frequency: 5.0, seed: 42)`

Sinusoidal rings centred at the grid midpoint. `frequency` controls how many oscillations span the grid radius; higher values produce tighter, more closely-spaced rings.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/concentric_rings.png" alt="" width=300 />

#### Wave Gradient

`wave_gradient(rows: 100, cols: 100, period: 3.0, seed: 42)`

Sinusoidal wave oriented at a given `direction` angle, cycling repeatedly from 0 to 1 and back at the specified `period`.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/wave_gradient.png" alt="" width=300 />

#### Spiral Gradient

`spiral_gradient(rows: 100, cols: 100, turns: 4.0, seed: 42)`

Archimedean spiral radiating outward from the grid centre. Values increase along the spiral arms; `turns` controls how many full rotations span the grid radius.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/spiral_gradient.png" alt="" width=300 />

### Noise

Continuous stochastic fields, from single-layer lattice noise to multi-octave fractal composites.

#### Perlin Noise

`perlin_noise(rows: 100, cols: 100, scale: 4.0, seed: 42)`

Smooth gradient noise built from dot products of random gradient vectors at lattice points, producing continuous, natural-looking variation.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/perlin.png" alt="" width=300 />

*Source: [Perlin (1985)](https://doi.org/10.1145/325165.325247)*

#### Value Noise

`value_noise(rows: 100, cols: 100, scale: 4.0, seed: 42)`

Interpolated lattice noise that is smoother and more rounded than Perlin noise.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/value_noise.png" alt="" width=300 />

#### Worley Noise

`worley_noise(rows: 100, cols: 100, scale: 4.0, seed: 42)`

Cell noise built from distances to random feature points, producing cellular, cracked-earth, or mosaic-like patterns.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/worley.png" alt="" width=300 />

*Source: [Worley (1996)](https://doi.org/10.1145/237170.237267)*

#### Voronoi Crease

`voronoi_crease(rows: 100, cols: 100, n: 30, seed: 42)`

F2-F1 Worley noise: the difference between distances to the two nearest seed points. Values peak at Voronoi cell boundaries and fall off toward cell centres, highlighting the skeleton of the Voronoi diagram.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/voronoi_crease.png" alt="" width=300 />

#### Perlin-Worley Noise

`perlin_worley(rows: 100, cols: 100, scale: 4.0, seed: 42)`

Hybrid of Perlin and Worley noise computed on the same cell grid. Perlin values are blended with inverted Worley distances, producing fluffy, cloud-like structures with soft cellular interiors.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/perlin_worley.png" alt="" width=300 />

#### fBm Noise

`fbm_noise(rows: 100, cols: 100, scale: 4.0, octaves: 6, seed: 42)`

Fractal Brownian motion layers multiple octaves of Perlin noise for more natural-looking terrain detail.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/fbm.png" alt="" width=300 />

*Source: [Mandelbrot & Van Ness (1968)](https://doi.org/10.1137/1010093); [Voss (1985)](https://doi.org/10.1007/978-3-642-84574-1_34)*

#### Ridged Noise

`ridged_noise(rows: 100, cols: 100, scale: 4.0, octaves: 6, seed: 42)`

Multi-octave noise where each octave is inverted and folded, producing sharp mountain ridges and valleys.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/ridged.png" alt="" width=300 />

*Source: [Musgrave, Kolb & Mace (1989)](https://doi.org/10.1145/74334.74337)*

#### Billow Noise

`billow_noise(rows: 100, cols: 100, scale: 4.0, octaves: 6, seed: 42)`

Multi-octave noise with absolute-value folding applied before accumulation, producing rounded billowing clouds or rolling dune shapes.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/billow.png" alt="" width=300 />

*Source: Ebert et al., Texturing and Modeling: A Procedural Approach (2002)*

#### Hybrid Noise

`hybrid_noise(rows: 100, cols: 100, scale: 4.0, octaves: 6, seed: 42)`

Hybrid multifractal noise combines fBm-style layering with a multiplicative weighting that amplifies high-frequency detail near peaks.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/hybrid_noise.png" alt="" width=300 />

*Source: [Musgrave, Kolb & Mace (1989)](https://doi.org/10.1145/74334.74337)*

#### Turbulence

`turbulence(rows: 100, cols: 100, scale: 4.0, octaves: 6, seed: 42)`

fBm with absolute-value folding per octave, producing sharp ridges and a storm-cloud appearance.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/turbulence.png" alt="" width=300 />

*Source: [Perlin (1985)](https://doi.org/10.1145/325165.325247)*

#### Domain Warp

`domain_warp(rows: 100, cols: 100, scale: 4.0, warp_strength: 1.0, seed: 42)`

Perlin noise sampled at coordinates displaced by a second Perlin field, producing organic swirling patterns.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/domain_warp.png" alt="" width=300 />

*Source: [Quilez (2002)](https://iquilezles.org/articles/warp/)*

#### Spectral Synthesis

`spectral_synthesis(rows: 100, cols: 100, beta: 2.0, seed: 42)`

Generates correlated noise in the frequency domain by scaling each component's amplitude by `f^(-beta/2)`, giving a power spectrum proportional to `1/f^beta`. Higher `beta` produces smoother, more spatially correlated landscapes.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/spectral_synthesis.png" alt="" width=300 />

*Source: [Peitgen & Saupe (1988)](https://link.springer.com/book/9780387966694)*

#### Fractal Brownian Surface

`fractal_brownian_surface(rows: 100, cols: 100, h: 0.5, seed: 42)`

Spectral synthesis parameterised by the Hurst exponent `h` ∈ (0, 1), which has direct ecological meaning. `h` near 0 is rough; `h` near 1 is smooth. Related to `spectral_synthesis` by β = 2h + 2.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/fractal_brownian_surface.png" alt="" width=300 />

#### Simplex Noise

`simplex_noise(rows: 100, cols: 100, scale: 4.0, seed: 42)`

An open-source alternative to Perlin noise with fewer directional artefacts.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/simplex_noise.png" alt="" width=300 />

#### Voronoi Distance

`voronoi_distance(rows: 100, cols: 100, n: 50, seed: 42)`

Scatters `n` random feature points across the grid and fills each cell with the Euclidean distance to the nearest point, producing smooth conical gradients centred on each point.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/voronoi_distance.png" alt="" width=300 />

#### Sine Composite

`sine_composite(rows: 100, cols: 100, waves: 8, seed: 42)`

Superposes `waves` sinusoidal plane waves, each with a random orientation, frequency, and phase. The interference of multiple waves produces standing-wave patterns whose complexity grows with the number of waves.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/sine_composite.png" alt="" width=300 />

#### Curl Noise

`curl_noise(rows: 100, cols: 100, scale: 4.0, seed: 42)`

Computes the curl (gradient rotated 90 degrees) of a Perlin potential field using finite differences, producing a divergence-free velocity field. Sample coordinates of a second Perlin generator are warped by this field, yielding swirling, flow-aligned patterns without directional clumping.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/curl_noise.png" alt="" width=300 />

#### Gabor Noise

`gabor_noise(rows: 100, cols: 100, scale: 4.0, n: 500, seed: 42)`

Sums `n` random Gabor kernels (Gaussian-windowed sinusoidal patches) over the grid. Each kernel has a random centre, orientation, and phase; the result is a spatially correlated texture whose dominant frequency is controlled by `scale`.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/gabor_noise.png" alt="" width=300 />

#### Spot Noise

`spot_noise(rows: 100, cols: 100, n: 200, seed: 42)`

Places `n` random elliptical Gaussian spots on the grid, each with an independent orientation, major radius, and aspect ratio. The superimposed spots produce an anisotropic, texture-like field.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/spot_noise.png" alt="" width=300 />

#### Anisotropic Noise

`anisotropic_noise(rows: 100, cols: 100, scale: 4.0, octaves: 6, direction: 45.0, stretch: 4.0, seed: 42)`

Fractal Brownian motion computed in a rotated and stretched coordinate frame, producing directionally biased textures. `direction` sets the elongation axis; `stretch` controls the anisotropy ratio.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/anisotropic_noise.png" alt="" width=300 />

#### Tiled Noise

`tiled_noise(rows: 100, cols: 100, scale: 4.0, seed: 42)`

Seamlessly tileable Perlin noise generated by embedding the grid on a 4D torus, ensuring that the left/right and top/bottom edges match perfectly.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/tiled_noise.png" alt="" width=300 />

#### Lognormal Field

`lognormal_field(rows: 100, cols: 100, sigma: 10.0, seed: 42)`

Gaussian random field transformed by an exponential, giving a right-skewed distribution with rare intense hotspots embedded in a low-value background, typical of ecological quantities such as resource concentrations and population densities.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/lognormal_field.png" alt="" width=300 />

### Patch

Discrete spatial patterns built from random processes, clustering, or hierarchical partitioning.

#### Random

`random(rows: 100, cols: 100, seed: 42)`

Independent uniform random values at each cell, with no spatial structure.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/random.png" alt="" width=300 />

#### Percolation

`percolation(rows: 100, cols: 100, p: 0.55, seed: 42)`

Binary Bernoulli lattice where each cell is independently set to 1 with probability `p`, producing binary habitat maps. The critical percolation threshold for 4-connectivity is approximately 0.593.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/percolation.png" alt="" width=300 />

*Source: [Gardner et al. (1987)](https://doi.org/10.1007/BF02275052)*

#### Random Element

`random_element(rows: 100, cols: 100, n: 5000, seed: 42)`

Places `n` labelled seed cells at random positions, then fills all remaining cells with the value of the nearest seed using nearest-neighbour interpolation.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/random_element.png" alt="" width=300 />

*Source: [Etherington, Holland & O'Sullivan (2015)](https://doi.org/10.1111/2041-210X.12308)*

#### Mosaic

`mosaic(rows: 100, cols: 100, n: 300, seed: 42)`

Discrete Voronoi map where each region is a flat colour determined by its nearest seed point, producing a stained-glass or territory effect.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/mosaic.png" alt="" width=300 />

#### Random Cluster

`random_cluster(rows: 100, cols: 100, n: 200, seed: 42)`

Applies `n` random fault-line cuts across the grid, accumulating the field on each side, then rescales. Produces spatially clustered landscapes with the linear structural elements characteristic of geological fault patterns.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/random_cluster.png" alt="" width=300 />

*Source: [Saura & Martínez-Millán (2000)](https://doi.org/10.1023/A:1008107902848)*

#### Rectangular Cluster

`rectangular_cluster(rows: 100, cols: 100, n: 300, seed: 42)`

Overlapping random axis-aligned rectangles accumulated and scaled, producing blocky clustered patches.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/rectangular_cluster.png" alt="" width=300 />

#### Binary Space Partitioning

`binary_space_partitioning(rows: 100, cols: 100, n: 200, seed: 42)`

Hierarchical rectilinear partition: the largest rectangle is repeatedly split along its longest dimension until `n` leaf regions remain, each assigned a random value. Produces structured blocky landscapes.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/binary_space_partitioning.png" alt="" width=300 />

*Source: [Etherington, Morgan & O'Sullivan (2022)](https://doi.org/10.1007/s10980-022-01452-6)*

#### Neighbourhood Clustering

`neighbourhood_clustering(rows: 100, cols: 100, k: 5, iterations: 10, seed: 42)`

Initialises a grid with `k` random classes then repeatedly applies a majority-vote rule: each cell adopts the most common class in its 3×3 Moore neighbourhood. More iterations produce larger, smoother organic patches.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/neighbourhood_clustering.png" alt="" width=300 />

#### Cellular Automaton

`cellular_automaton(rows: 100, cols: 100, p: 0.45, iterations: 5, seed: 42)`

Random binary grid evolved by Conway-style birth/survival rules: a dead cell is born if it has at least `birth_threshold` live neighbours; a live cell survives if it has at least `survival_threshold`. Produces cave-like binary landscapes.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/cellular_automaton.png" alt="" width=300 />

#### Reaction-Diffusion

`reaction_diffusion(rows: 100, cols: 100, iterations: 1000, feed: 0.055, kill: 0.062, seed: 42)`

Gray-Scott reaction-diffusion model where two chemicals (A and B) diffuse and react across the grid. Different `feed`/`kill` combinations produce spots, stripes, labyrinths, and other Turing-pattern morphologies.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/reaction_diffusion.png" alt="" width=300 />

#### Eden Growth

`eden_growth(rows: 100, cols: 100, n: 2000, seed: 42)`

Compact cluster grown from the grid centre by randomly selecting a boundary cell at each step. Produces irregular blob shapes with fractal perimeters.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/eden_growth.png" alt="" width=300 />

#### Diffusion-Limited Aggregation

`diffusion_limited_aggregation(rows: 100, cols: 100, n: 2000, seed: 42)`

Random-walking particles released from a spawn ring stick when adjacent to the growing cluster, producing intricate branching fractal structures.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/diffusion_limited_aggregation.png" alt="" width=300 />

#### Invasion Percolation

`invasion_percolation(rows: 100, cols: 100, n: 2000, seed: 42)`

Grows a cluster from the grid centre by always invading the boundary cell with the lowest random weight, producing fractal-like connected binary patches.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/invasion_percolation.png" alt="" width=300 />

#### Gaussian Blobs

`gaussian_blobs(rows: 100, cols: 100, n: 50, sigma: 5.0, seed: 42)`

Places random Gaussian kernel centres and accumulates their contributions across the grid, then rescales to [0, 1]. Produces smooth blob-like elevation fields.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/gaussian_blobs.png" alt="" width=300 />

#### Ising Model

`ising_model(rows: 100, cols: 100, beta: 0.4, iterations: 1000, seed: 42)`

Simulates a 2D Ising spin lattice via Glauber dynamics. Near the critical inverse temperature (β ≈ 0.44) the model produces scale-free, patchy binary patterns reminiscent of habitat mosaics.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/ising_model.png" alt="" width=300 />

#### Schelling

`schelling(rows: 100, cols: 100, tolerance: 0.5, iterations: 50, seed: 42)`

Schelling's model of residential segregation: cells of two types relocate to random empty slots whenever fewer than `tolerance` of their neighbours share their type. Even mild thresholds produce strong spatial segregation, generating sharp-edged binary cluster patterns.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/schelling.png" alt="" width=300 />

*Source: [Schelling (1971)](https://doi.org/10.1080/0022250X.1971.9989794)*

#### Sandpile

`sandpile(rows: 100, cols: 100, n: 5000, seed: 42)`

Bak-Tang-Wiesenfeld sandpile model: `n` grains are dropped at random cells; any cell accumulating 4 or more grains topples, distributing one grain to each of its four neighbours. The resulting grain-count map reveals self-similar avalanche structure characteristic of self-organized criticality.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/sandpile.png" alt="" width=300 />

*Source: [Bak, Tang & Wiesenfeld (1987)](https://doi.org/10.1103/PhysRevLett.59.381)*

#### Hydraulic Erosion

`hydraulic_erosion(rows: 100, cols: 100, n: 500, seed: 42)`

Generates a random initial heightmap, then simulates `n` water droplets flowing downhill. Each droplet carries sediment, eroding steeper terrain and depositing on flatter areas. The result resembles naturally worn terrain with drainage channels, alluvial fans, and rounded ridges.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/hydraulic_erosion.png" alt="" width=300 />

#### Levy Flight

`levy_flight(rows: 100, cols: 100, n: 1000, seed: 42)`

Simulates a Levy flight: a random walk where step lengths follow a power-law (heavy-tailed) distribution. The resulting visit-density map has clustered hotspots with occasional long-range jumps, modelling dispersal or foraging patterns.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/levy_flight.png" alt="" width=300 />

#### Correlated Walk

`correlated_walk(rows: 100, cols: 100, n: 5000, kappa: 2.0, seed: 42)`

A random walk where each step direction is drawn from a wrapped normal distribution centred on the previous heading. Higher `kappa` values produce straighter, more directional trajectories; `kappa = 0` reduces to isotropic Brownian motion. The visit-density map models correlated animal movement or dispersal paths.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/correlated_walk.png" alt="" width=300 />

#### Poisson Disk

`poisson_disk(rows: 100, cols: 100, min_dist: 5.0, seed: 42)`

Uses Bridson's algorithm to place points such that no two are closer than `min_dist`. The resulting inhibition pattern has regular, even spacing compared to random point placement, modelling processes such as territorial behaviour or tree canopy competition.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/poisson_disk.png" alt="" width=300 />

#### Hill Grow

`hill_grow(rows: 100, cols: 100, n: 20000, seed: 42)`

Iteratively stamps a smooth convolution kernel at randomly selected cells, building up hill-like mounds. With `runaway=True`, taller cells attract more growth, causing hills to cluster into ridges.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/hill_grow.png" alt="" width=300 />

*Source: [Etherington, Holland & O'Sullivan (2015)](https://doi.org/10.1111/2041-210X.12308)*

#### Midpoint Displacement

`midpoint_displacement(rows: 100, cols: 100, h: 0.8, seed: 42)`

Recursive fractal terrain generation: grid midpoints are displaced by decreasing random amounts at each subdivision step, with `h` controlling the roughness (0 = rough, 1 = smooth).

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/midpoint_displacement.png" alt="" width=300 />

*Source: [Fournier, Fussell & Carpenter (1982)](https://doi.org/10.1145/358523.358553)*

#### Gaussian Field

`gaussian_field(rows: 100, cols: 100, sigma: 10.0, seed: 42)`

White noise smoothed by a Gaussian blur kernel with standard deviation `sigma`, producing spatially correlated fields where patch size scales directly with `sigma`.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/gaussian_field.png" alt="" width=300 />

#### Brownian Motion

`brownian_motion(rows: 100, cols: 100, n: 5000, seed: 42)`

Simulates `n` steps of a 2D Brownian random walk on a toroidal grid and accumulates visit counts, producing a visit-density map with dense hotspots along the walker's path.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/brownian_motion.png" alt="" width=300 />

#### Forest Fire

`forest_fire(rows: 100, cols: 100, p_tree: 0.02, p_lightning: 0.001, iterations: 500, seed: 42)`

Drossel-Schwabl forest fire cellular automaton: empty cells grow trees with probability `p_tree`; trees ignite from lightning with probability `p_lightning` and spread to neighbouring trees. The cumulative burn count, scaled to [0, 1], captures spatial fire-scar patterns.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/forest_fire.png" alt="" width=300 />

#### River Network

`river_network(rows: 100, cols: 100, seed: 42)`

Generates a fractal Brownian terrain, computes D8 steepest-descent flow directions, then accumulates upstream drainage area. Log-scaling the flow accumulation reveals branching river networks with realistic headwater-to-trunk gradients.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/river_network.png" alt="" width=300 />

#### Hexagonal Voronoi

`hexagonal_voronoi(rows: 100, cols: 100, n: 50, seed: 42)`

Places seed points on a slightly jittered hexagonal lattice and uses nearest-neighbour BFS to fill each Voronoi cell with a random flat value, producing a regular honeycomb-like mosaic.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/hexagonal_voronoi.png" alt="" width=300 />

#### Fault Uplift

`fault_uplift(rows: 100, cols: 100, n: 50, seed: 42)`

Places `n` random fault lines across the grid and adds an exponential ridge of elevation along each one. Superimposing many ridges produces a crumpled, tectonically-inspired terrain with narrow elevated seams.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/fault_uplift.png" alt="" width=300 />

#### Triangular Tessellation

`triangular_tessellation(rows: 100, cols: 100, n: 30, seed: 42)`

Scatters `n` random points and computes their Delaunay triangulation via the Bowyer-Watson algorithm. Each triangle is assigned a uniform random value, producing a mosaic of flat-shaded triangles with straight edges.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/triangular_tessellation.png" alt="" width=300 />

#### Physarum

`physarum(rows: 100, cols: 100, n: 1000, iterations: 300, seed: 42)`

Agent-based simulation of _Physarum polycephalum_ (slime mould). Agents sense a chemical trail at ±45° ahead and steer toward the strongest signal, then deposit trail at their new position. Trail diffuses and decays at each step. The emergent network of reinforced paths resembles efficient transport networks.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/physarum.png" alt="" width=300 />

*Source: [Jones (2010)](https://doi.org/10.1142/S0219525910002640)*

#### Cahn-Hilliard

`cahn_hilliard(rows: 100, cols: 100, iterations: 2000, seed: 42)`

Numerically integrates the Cahn-Hilliard PDE for spinodal decomposition: ∂u/∂t = ∇²(u³ - u - ε²∇²u). Starting from small random perturbations, a binary mixture spontaneously phase-separates into smooth, rounded domains that coarsen over time.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/cahn_hilliard.png" alt="" width=300 />

#### Crystal Growth

`crystal_growth(rows: 100, cols: 100, iterations: 300, seed: 42)`

Reiter's (1996) hexagonal cellular automaton model of snowflake growth. A single frozen seed at the centre absorbs water vapour from receptive cells (those adjacent to the crystal), which freeze when their accumulated level reaches 1. The result is a six-fold symmetric crystal with intricate branching arms.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/crystal_growth.png" alt="" width=300 />

*Source: [Reiter (1996)](https://doi.org/10.1016/0097-8493(95)00104-2)*

#### Predator-Prey

`predator_prey(rows: 100, cols: 100, iterations: 500, seed: 42)`

Spatial Lotka-Volterra PDE with diffusion. Two fields, prey (`u`) and predators (`v`), interact via classic predation kinetics while diffusing across the grid. Initialised near the coexistence equilibrium with small noise, the system self-organises into travelling waves and spiral patterns.

<img src="https://raw.githubusercontent.com/tom-draper/nlmrs/main/examples/predator_prey.png" alt="" width=300 />


## Usage

NLMrs is available as a Rust crate or as a CLI tool. Language bindings are also provided for Python, R, Julia, WASM, and C.

```bash
cargo add nlmrs
```

```rs
use nlmrs;

fn main() {
    let grid = nlmrs::midpoint_displacement(100, 100, 1.0, seed: Some(42));
    println!("{:?}", grid.data);
}
```

### Export

The `export` module provides functions to save a grid to disk.

```rs
use nlmrs::{midpoint_displacement, export};

fn main() {
    let grid = midpoint_displacement(rows: 100, cols: 100, h: 0.8, Some(42));

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

### Grid operations

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

### Python bindings

NLMrs is available as a Python package. Every function returns a 2D numpy array.

#### Install

```bash
pip install nlmrs
```

Or build from source (requires Rust and [maturin](https://github.com/PyO3/maturin)):

```bash
maturin develop --features python   # editable install into the active venv
```

#### Usage

```python
import nlmrs
import matplotlib.pyplot as plt

# All functions accept an optional seed for reproducible output.
grid = nlmrs.midpoint_displacement(100, 100, h=0.8, seed=42)  # numpy array (100, 100)

plt.imshow(grid, cmap="terrain")
plt.axis("off")
plt.show()
```

Post-processing functions are also available:

```python
grid = nlmrs.fbm_noise(100, 100, scale=4.0)
nlmrs.classify(grid, n=5)    # quantise into n equal-width classes
nlmrs.threshold(grid, t=0.5) # binarise at threshold t
```

### R bindings

NLMrs is available as an R package via the [extendr](https://extendr.github.io/) framework. Every function returns a numeric matrix.

#### Install

```r
# Install from source (requires Rust)
remotes::install_github("tom-draper/nlmrs", subdir = "bindings/r")
```

#### Usage

```r
library(nlmrs)

# All functions accept an optional integer seed.
m <- nlm_midpoint_displacement(100, 100, h = 0.8, seed = 42L)
image(m, col = terrain.colors(256))
```

### Julia bindings

NLMrs is available as a Julia package. Every function returns a `Matrix{Float64}`.

#### Install

```julia
# Requires Rust (https://rustup.rs) - builds the C library automatically.
import Pkg
Pkg.add(url="https://github.com/tom-draper/nlmrs", subdir="bindings/julia")
```

#### Usage

```julia
using NLMrs

# All functions accept an optional seed keyword for reproducible output.
m = midpoint_displacement(100, 100, h=0.8, seed=42)  # Matrix{Float64} (100, 100)
```

### C bindings

NLMrs exposes a C-compatible shared/static library, making it usable from any language with C FFI support (C++, Go, MATLAB, Fortran, etc.).

#### Build

```bash
cd bindings/c
cargo build --release
# → ../../target/release/libnlmrs_c.so   (Linux shared)
# → ../../target/release/libnlmrs_c.a    (Linux static)
# → include/nlmrs.h                       (generated header)
```

#### Usage

```c
#include "nlmrs.h"
#include <stdio.h>

int main(void) {
    uint64_t seed = 42;

    // Generate a 200×200 midpoint displacement grid.
    NlmGrid grid = nlmrs_midpoint_displacement(200, 200, 0.8, &seed);

    printf("rows=%zu cols=%zu\n", grid.rows, grid.cols);

    // Access row-major data: value at (r, c) = grid.data[r * grid.cols + c]
    printf("value at (0,0): %f\n", grid.data[0]);

    nlmrs_free(grid);   // release Rust-owned memory
    return 0;
}
```

Compile and link against the shared library:

```bash
gcc example.c -I bindings/c/include -L target/release -lnlmrs_c -o example
```

#### Optional parameters

Seeds and optional floats (e.g. gradient `direction`) are passed as pointers. Pass `NULL` to use the default (random seed / random direction):

```c
// Random seed
NlmGrid g1 = nlmrs_perlin_noise(200, 200, 4.0, NULL);

// Fixed direction, random seed
double dir = 45.0;
NlmGrid g2 = nlmrs_planar_gradient(200, 200, &dir, NULL);

nlmrs_free(g1);
nlmrs_free(g2);
```

All 49 algorithms are available as `nlmrs_<name>`. The header `include/nlmrs.h` is generated automatically by `cbindgen` during the build.

### WASM bindings

NLMrs can run in the browser or Node.js via WebAssembly.

#### Build

```bash
cd bindings/wasm
wasm-pack build --target web
```

#### Usage

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

All 49 algorithms are available. Seeds are passed as plain integers. Omit the seed argument for random output.

## Contributions

Contributions, issues and feature requests are welcome.

- Fork it (https://github.com/tom-draper/nlmrs)
- Create your feature branch (`git checkout -b my-new-feature`)
- Commit your changes (`git commit -am 'Add some feature')
- Push to the branch (`git push origin my-new-feature`)
- Create a new Pull Request
