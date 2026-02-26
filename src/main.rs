use clap::{Parser, Subcommand};
use nlmrs::export;

#[derive(Parser)]
#[command(
    name = "nlmrs",
    about = "Generate Neutral Landscape Models",
    long_about = "Generate 2D spatial grids using various NLM algorithms.\nOutput format is inferred from the file extension (.png, .csv, .json)."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Output file path (extension determines format: .png, .csv, .json)
    #[arg(long, short, default_value = "output.png", global = true)]
    output: String,

    /// RNG seed for reproducible output
    #[arg(long, global = true)]
    seed: Option<u64>,

    /// Use grayscale instead of terrain colormap (PNG only)
    #[arg(long, global = true)]
    grayscale: bool,

    /// Classify output into N equal-width classes
    #[arg(long, global = true)]
    classify: Option<usize>,

    /// Threshold output at T: values below T → 0.0, at or above → 1.0
    #[arg(long, global = true)]
    threshold: Option<f64>,
}

#[derive(Subcommand)]
enum Commands {
    /// Spatially random noise
    Random {
        /// Number of rows
        rows: usize,
        /// Number of columns
        cols: usize,
    },
    /// Random element nearest-neighbour interpolation
    RandomElement {
        rows: usize,
        cols: usize,
        /// Number of seed elements to place
        #[arg(long, default_value = "50000")]
        n: f64,
    },
    /// Linear gradient at a given angle
    PlanarGradient {
        rows: usize,
        cols: usize,
        /// Gradient direction in degrees [0, 360)
        #[arg(long)]
        direction: Option<f64>,
    },
    /// Symmetric gradient (zero at edges, peak in middle)
    EdgeGradient {
        rows: usize,
        cols: usize,
        /// Gradient direction in degrees [0, 360)
        #[arg(long)]
        direction: Option<f64>,
    },
    /// Radial gradient from a random centre point
    DistanceGradient {
        rows: usize,
        cols: usize,
    },
    /// Sinusoidal wave gradient
    WaveGradient {
        rows: usize,
        cols: usize,
        /// Wave period (smaller = larger waves)
        #[arg(long, default_value = "2.5")]
        period: f64,
        /// Wave direction in degrees [0, 360)
        #[arg(long)]
        direction: Option<f64>,
    },
    /// Diamond-square fractal terrain (midpoint displacement)
    MidpointDisplacement {
        rows: usize,
        cols: usize,
        /// Spatial autocorrelation (0 = rough, 1 = smooth)
        #[arg(long, default_value = "1.0")]
        h: f64,
    },
    /// Hill-grow algorithm
    HillGrow {
        rows: usize,
        cols: usize,
        /// Number of iterations
        #[arg(long, default_value = "10000")]
        n: usize,
        /// Hills grow in clusters (weighted random selection)
        #[arg(long)]
        runaway: bool,
        /// Surface only grows, never shrinks
        #[arg(long)]
        only_grow: bool,
    },
    /// Perlin noise
    Perlin {
        rows: usize,
        cols: usize,
        /// Noise frequency (higher = more features)
        #[arg(long, default_value = "4.0")]
        scale: f64,
    },
    /// Fractal Brownian motion (layered Perlin noise)
    Fbm {
        rows: usize,
        cols: usize,
        /// Base noise frequency
        #[arg(long, default_value = "4.0")]
        scale: f64,
        /// Number of octaves
        #[arg(long, default_value = "6")]
        octaves: usize,
        /// Amplitude scaling per octave
        #[arg(long, default_value = "0.5")]
        persistence: f64,
        /// Frequency scaling per octave
        #[arg(long, default_value = "2.0")]
        lacunarity: f64,
    },
    /// Ridged multifractal noise — sharp ridges and mountain-like terrain
    Ridged {
        rows: usize,
        cols: usize,
        /// Base noise frequency
        #[arg(long, default_value = "4.0")]
        scale: f64,
        /// Number of octaves
        #[arg(long, default_value = "6")]
        octaves: usize,
        /// Amplitude scaling per octave
        #[arg(long, default_value = "0.5")]
        persistence: f64,
        /// Frequency scaling per octave
        #[arg(long, default_value = "2.0")]
        lacunarity: f64,
    },
    /// Billow noise — rounded cloud- and hill-like patterns
    Billow {
        rows: usize,
        cols: usize,
        /// Base noise frequency
        #[arg(long, default_value = "4.0")]
        scale: f64,
        /// Number of octaves
        #[arg(long, default_value = "6")]
        octaves: usize,
        /// Amplitude scaling per octave
        #[arg(long, default_value = "0.5")]
        persistence: f64,
        /// Frequency scaling per octave
        #[arg(long, default_value = "2.0")]
        lacunarity: f64,
    },
    /// Worley (cellular) noise — territory / patch patterns
    Worley {
        rows: usize,
        cols: usize,
        /// Seed-point frequency (higher = smaller cells)
        #[arg(long, default_value = "4.0")]
        scale: f64,
    },
    /// Gaussian random field — spatially correlated noise
    GaussianField {
        rows: usize,
        cols: usize,
        /// Gaussian kernel standard deviation in cells (correlation length)
        #[arg(long, default_value = "10.0")]
        sigma: f64,
    },
    /// Random cluster via fault-line cuts
    RandomCluster {
        rows: usize,
        cols: usize,
        /// Number of fault-line cuts
        #[arg(long, default_value = "200")]
        n: usize,
    },
    /// Hybrid multifractal noise — blends smooth and ridged characteristics
    HybridNoise {
        rows: usize,
        cols: usize,
        /// Base noise frequency
        #[arg(long, default_value = "4.0")]
        scale: f64,
        /// Number of octaves
        #[arg(long, default_value = "6")]
        octaves: usize,
        /// Amplitude scaling per octave
        #[arg(long, default_value = "0.5")]
        persistence: f64,
        /// Frequency scaling per octave
        #[arg(long, default_value = "2.0")]
        lacunarity: f64,
    },
    /// Value noise — interpolated lattice noise
    ValueNoise {
        rows: usize,
        cols: usize,
        /// Noise frequency (higher = more features)
        #[arg(long, default_value = "4.0")]
        scale: f64,
    },
    /// Turbulence — fBm with absolute-value fold per octave
    Turbulence {
        rows: usize,
        cols: usize,
        /// Base noise frequency
        #[arg(long, default_value = "4.0")]
        scale: f64,
        /// Number of octaves
        #[arg(long, default_value = "6")]
        octaves: usize,
        /// Amplitude scaling per octave
        #[arg(long, default_value = "0.5")]
        persistence: f64,
        /// Frequency scaling per octave
        #[arg(long, default_value = "2.0")]
        lacunarity: f64,
    },
    /// Domain-warped Perlin noise — organic, swirling patterns
    DomainWarp {
        rows: usize,
        cols: usize,
        /// Coordinate frequency (higher = more features)
        #[arg(long, default_value = "4.0")]
        scale: f64,
        /// Displacement magnitude applied to sample coordinates
        #[arg(long, default_value = "1.0")]
        warp_strength: f64,
    },
    /// Mosaic — discrete Voronoi patch map with flat-coloured regions
    Mosaic {
        rows: usize,
        cols: usize,
        /// Number of Voronoi seed points
        #[arg(long, default_value = "200")]
        n: usize,
    },
    /// Rectangular cluster — overlapping random axis-aligned rectangles
    RectangularCluster {
        rows: usize,
        cols: usize,
        /// Number of rectangles to place
        #[arg(long, default_value = "200")]
        n: usize,
    },
    /// Percolation — binary Bernoulli lattice (Gardner 1987)
    Percolation {
        rows: usize,
        cols: usize,
        /// Probability a cell is habitat (0.0–1.0)
        #[arg(long, default_value = "0.5")]
        p: f64,
    },
    /// Binary space partitioning — hierarchical rectilinear partition
    BinarySpacePartitioning {
        rows: usize,
        cols: usize,
        /// Number of rectangles in the final partition
        #[arg(long, default_value = "100")]
        n: usize,
    },
    /// Cellular automaton — binary cave-like patterns from birth/survival rules
    CellularAutomaton {
        rows: usize,
        cols: usize,
        /// Initial probability of a cell being alive
        #[arg(long, default_value = "0.45")]
        p: f64,
        /// Number of rule iterations
        #[arg(long, default_value = "5")]
        iterations: usize,
        /// Min live neighbours for a dead cell to become alive
        #[arg(long, default_value = "5")]
        birth_threshold: usize,
        /// Min live neighbours for a live cell to stay alive
        #[arg(long, default_value = "4")]
        survival_threshold: usize,
    },
    /// Neighbourhood clustering — iterative majority-vote patch clustering
    NeighbourhoodClustering {
        rows: usize,
        cols: usize,
        /// Number of distinct patch classes
        #[arg(long, default_value = "5")]
        k: usize,
        /// Number of majority-vote iterations
        #[arg(long, default_value = "10")]
        iterations: usize,
    },
    /// Spectral synthesis — 1/f^beta noise generated in the frequency domain
    SpectralSynthesis {
        rows: usize,
        cols: usize,
        /// Spectral exponent: 0 = white noise, 1 = pink, 2 = brown/natural terrain
        #[arg(long, default_value = "2.0")]
        beta: f64,
    },
    /// Diffusion-limited aggregation — branching fractal cluster grown from centre
    DiffusionLimitedAggregation {
        rows: usize,
        cols: usize,
        /// Number of particles to release
        #[arg(long, default_value = "2000")]
        n: usize,
    },
    /// Gray-Scott reaction-diffusion — Turing-pattern spots, stripes and labyrinths
    ReactionDiffusion {
        rows: usize,
        cols: usize,
        /// Number of simulation steps
        #[arg(long, default_value = "1000")]
        iterations: usize,
        /// Feed rate for chemical A (controls pattern type)
        #[arg(long, default_value = "0.055")]
        feed: f64,
        /// Kill rate for chemical B (controls pattern type)
        #[arg(long, default_value = "0.062")]
        kill: f64,
    },
    /// Eden growth model — compact fractal blob grown from the centre
    EdenGrowth {
        rows: usize,
        cols: usize,
        /// Number of cells to add to the cluster
        #[arg(long, default_value = "2000")]
        n: usize,
    },
    /// Fractal Brownian surface parameterised by the Hurst exponent
    FractalBrownianSurface {
        rows: usize,
        cols: usize,
        /// Hurst exponent in (0, 1): 0 = rough, 1 = smooth
        #[arg(long, default_value = "0.5")]
        h: f64,
    },
    /// Elliptical landscape gradient centred at the grid midpoint
    LandscapeGradient {
        rows: usize,
        cols: usize,
        /// Major-axis orientation in degrees [0, 360). Random if omitted.
        #[arg(long)]
        direction: Option<f64>,
        /// Major-to-minor axis ratio (≥ 1.0). 1.0 = circular.
        #[arg(long, default_value = "1.0")]
        aspect: f64,
    },
    /// OpenSimplex noise
    SimplexNoise {
        rows: usize,
        cols: usize,
        /// Coordinate frequency (higher = more features per unit)
        #[arg(long, default_value = "4.0")]
        scale: f64,
    },
    /// Invasion percolation (lowest-weight boundary growth from centre)
    InvasionPercolation {
        rows: usize,
        cols: usize,
        /// Number of cells to invade
        #[arg(long, default_value = "2000")]
        n: usize,
    },
    /// Sum of random Gaussian blob kernels
    GaussianBlobs {
        rows: usize,
        cols: usize,
        /// Number of blob centres
        #[arg(long, default_value = "50")]
        n: usize,
        /// Gaussian width (in cells)
        #[arg(long, default_value = "5.0")]
        sigma: f64,
    },
    /// Ising model via Glauber dynamics (binary spin lattice)
    IsingModel {
        rows: usize,
        cols: usize,
        /// Inverse temperature (near 0.44 = critical point)
        #[arg(long, default_value = "0.4")]
        beta: f64,
        /// Number of sweeps (each sweep = rows × cols spin-flip attempts)
        #[arg(long, default_value = "1000")]
        iterations: usize,
    },
    /// Voronoi distance field from random feature points
    VoronoiDistance {
        rows: usize,
        cols: usize,
        /// Number of feature points
        #[arg(long, default_value = "50")]
        n: usize,
    },
    /// Superposition of sinusoidal plane waves
    SineComposite {
        rows: usize,
        cols: usize,
        /// Number of sinusoidal waves to superpose
        #[arg(long, default_value = "8")]
        waves: usize,
    },
    /// Divergence-free curl-warped Perlin noise
    CurlNoise {
        rows: usize,
        cols: usize,
        /// Coordinate frequency (higher = more features per unit)
        #[arg(long, default_value = "4.0")]
        scale: f64,
    },
    /// Hydraulic erosion simulation on a random heightmap
    HydraulicErosion {
        rows: usize,
        cols: usize,
        /// Number of erosion droplets to simulate
        #[arg(long, default_value = "500")]
        n: usize,
    },
    /// Levy flight random walk density map
    LevyFlight {
        rows: usize,
        cols: usize,
        /// Number of flight steps
        #[arg(long, default_value = "1000")]
        n: usize,
    },
    /// Poisson disk sampling (binary inhibition pattern)
    PoissonDisk {
        rows: usize,
        cols: usize,
        /// Minimum distance in cells between any two sample points
        #[arg(long, default_value = "5.0")]
        min_dist: f64,
    },
    /// Gabor noise — oriented sinusoidal kernel superposition
    GaborNoise {
        rows: usize,
        cols: usize,
        /// Controls carrier frequency and envelope width (higher = finer features)
        #[arg(long, default_value = "4.0")]
        scale: f64,
        /// Number of Gabor kernels to place
        #[arg(long, default_value = "500")]
        n: usize,
    },
    /// Spot noise — random anisotropic elliptical Gaussian blobs
    SpotNoise {
        rows: usize,
        cols: usize,
        /// Number of spots to place
        #[arg(long, default_value = "200")]
        n: usize,
    },
    /// Anisotropic noise — fBm stretched along a dominant axis
    AnisotropicNoise {
        rows: usize,
        cols: usize,
        /// Base noise frequency along the primary axis
        #[arg(long, default_value = "4.0")]
        scale: f64,
        /// Number of noise layers to combine
        #[arg(long, default_value = "6")]
        octaves: usize,
        /// Orientation of elongation in degrees [0, 360)
        #[arg(long, default_value = "45.0")]
        direction: f64,
        /// Compression ratio for the perpendicular axis (≥ 1.0)
        #[arg(long, default_value = "4.0")]
        stretch: f64,
    },
    /// Tiled noise — seamlessly repeating Perlin noise via 4-D torus mapping
    TiledNoise {
        rows: usize,
        cols: usize,
        /// Number of noise cycles per tile (higher = more features)
        #[arg(long, default_value = "4.0")]
        scale: f64,
    },
    /// Brownian motion — Gaussian random-walk visit density
    BrownianMotion {
        rows: usize,
        cols: usize,
        /// Number of walk steps
        #[arg(long, default_value = "5000")]
        n: usize,
    },
    /// Forest fire — Drossel-Schwabl cellular automaton burn-history map
    ForestFire {
        rows: usize,
        cols: usize,
        /// Per-step probability an empty cell becomes a tree
        #[arg(long, default_value = "0.02")]
        p_tree: f64,
        /// Per-step probability a tree ignites spontaneously
        #[arg(long, default_value = "0.001")]
        p_lightning: f64,
        /// Number of simulation steps
        #[arg(long, default_value = "500")]
        iterations: usize,
    },
    /// River network — D8 flow accumulation on fBm terrain
    RiverNetwork {
        rows: usize,
        cols: usize,
    },
    /// Hexagonal Voronoi — BFS mosaic from a regular hexagonal seed lattice
    HexagonalVoronoi {
        rows: usize,
        cols: usize,
        /// Approximate number of hexagonal cells
        #[arg(long, default_value = "50")]
        n: usize,
    },
}

fn main() {
    let cli = Cli::parse();
    let seed = cli.seed;

    let grid = match cli.command {
        Commands::Random { rows, cols } => nlmrs::random(rows, cols, seed),
        Commands::RandomElement { rows, cols, n } => nlmrs::random_element(rows, cols, n, seed),
        Commands::PlanarGradient { rows, cols, direction } => {
            nlmrs::planar_gradient(rows, cols, direction, seed)
        }
        Commands::EdgeGradient { rows, cols, direction } => {
            nlmrs::edge_gradient(rows, cols, direction, seed)
        }
        Commands::DistanceGradient { rows, cols } => nlmrs::distance_gradient(rows, cols, seed),
        Commands::WaveGradient { rows, cols, period, direction } => {
            nlmrs::wave_gradient(rows, cols, period, direction, seed)
        }
        Commands::MidpointDisplacement { rows, cols, h } => {
            nlmrs::midpoint_displacement(rows, cols, h, seed)
        }
        Commands::HillGrow { rows, cols, n, runaway, only_grow } => {
            nlmrs::hill_grow(rows, cols, n, runaway, None, only_grow, seed)
        }
        Commands::Perlin { rows, cols, scale } => nlmrs::perlin_noise(rows, cols, scale, seed),
        Commands::Fbm { rows, cols, scale, octaves, persistence, lacunarity } => {
            nlmrs::fbm_noise(rows, cols, scale, octaves, persistence, lacunarity, seed)
        }
        Commands::Ridged { rows, cols, scale, octaves, persistence, lacunarity } => {
            nlmrs::ridged_noise(rows, cols, scale, octaves, persistence, lacunarity, seed)
        }
        Commands::Billow { rows, cols, scale, octaves, persistence, lacunarity } => {
            nlmrs::billow_noise(rows, cols, scale, octaves, persistence, lacunarity, seed)
        }
        Commands::Worley { rows, cols, scale } => nlmrs::worley_noise(rows, cols, scale, seed),
        Commands::GaussianField { rows, cols, sigma } => {
            nlmrs::gaussian_field(rows, cols, sigma, seed)
        }
        Commands::RandomCluster { rows, cols, n } => nlmrs::random_cluster(rows, cols, n, seed),
        Commands::HybridNoise { rows, cols, scale, octaves, persistence, lacunarity } => {
            nlmrs::hybrid_noise(rows, cols, scale, octaves, persistence, lacunarity, seed)
        }
        Commands::ValueNoise { rows, cols, scale } => nlmrs::value_noise(rows, cols, scale, seed),
        Commands::Turbulence { rows, cols, scale, octaves, persistence, lacunarity } => {
            nlmrs::turbulence(rows, cols, scale, octaves, persistence, lacunarity, seed)
        }
        Commands::DomainWarp { rows, cols, scale, warp_strength } => {
            nlmrs::domain_warp(rows, cols, scale, warp_strength, seed)
        }
        Commands::Mosaic { rows, cols, n } => nlmrs::mosaic(rows, cols, n, seed),
        Commands::RectangularCluster { rows, cols, n } => {
            nlmrs::rectangular_cluster(rows, cols, n, seed)
        }
        Commands::Percolation { rows, cols, p } => nlmrs::percolation(rows, cols, p, seed),
        Commands::BinarySpacePartitioning { rows, cols, n } => {
            nlmrs::binary_space_partitioning(rows, cols, n, seed)
        }
        Commands::CellularAutomaton { rows, cols, p, iterations, birth_threshold, survival_threshold } => {
            nlmrs::cellular_automaton(rows, cols, p, iterations, birth_threshold, survival_threshold, seed)
        }
        Commands::NeighbourhoodClustering { rows, cols, k, iterations } => {
            nlmrs::neighbourhood_clustering(rows, cols, k, iterations, seed)
        }
        Commands::SpectralSynthesis { rows, cols, beta } => {
            nlmrs::spectral_synthesis(rows, cols, beta, seed)
        }
        Commands::DiffusionLimitedAggregation { rows, cols, n } => {
            nlmrs::diffusion_limited_aggregation(rows, cols, n, seed)
        }
        Commands::ReactionDiffusion { rows, cols, iterations, feed, kill } => {
            nlmrs::reaction_diffusion(rows, cols, iterations, feed, kill, seed)
        }
        Commands::EdenGrowth { rows, cols, n } => nlmrs::eden_growth(rows, cols, n, seed),
        Commands::FractalBrownianSurface { rows, cols, h } => {
            nlmrs::fractal_brownian_surface(rows, cols, h, seed)
        }
        Commands::LandscapeGradient { rows, cols, direction, aspect } => {
            nlmrs::landscape_gradient(rows, cols, direction, aspect, seed)
        }
        Commands::SimplexNoise { rows, cols, scale } => {
            nlmrs::simplex_noise(rows, cols, scale, seed)
        }
        Commands::InvasionPercolation { rows, cols, n } => {
            nlmrs::invasion_percolation(rows, cols, n, seed)
        }
        Commands::GaussianBlobs { rows, cols, n, sigma } => {
            nlmrs::gaussian_blobs(rows, cols, n, sigma, seed)
        }
        Commands::IsingModel { rows, cols, beta, iterations } => {
            nlmrs::ising_model(rows, cols, beta, iterations, seed)
        }
        Commands::VoronoiDistance { rows, cols, n } => {
            nlmrs::voronoi_distance(rows, cols, n, seed)
        }
        Commands::SineComposite { rows, cols, waves } => {
            nlmrs::sine_composite(rows, cols, waves, seed)
        }
        Commands::CurlNoise { rows, cols, scale } => nlmrs::curl_noise(rows, cols, scale, seed),
        Commands::HydraulicErosion { rows, cols, n } => {
            nlmrs::hydraulic_erosion(rows, cols, n, seed)
        }
        Commands::LevyFlight { rows, cols, n } => nlmrs::levy_flight(rows, cols, n, seed),
        Commands::PoissonDisk { rows, cols, min_dist } => {
            nlmrs::poisson_disk(rows, cols, min_dist, seed)
        }
        Commands::GaborNoise { rows, cols, scale, n } => {
            nlmrs::gabor_noise(rows, cols, scale, n, seed)
        }
        Commands::SpotNoise { rows, cols, n } => nlmrs::spot_noise(rows, cols, n, seed),
        Commands::AnisotropicNoise { rows, cols, scale, octaves, direction, stretch } => {
            nlmrs::anisotropic_noise(rows, cols, scale, octaves, direction, stretch, seed)
        }
        Commands::TiledNoise { rows, cols, scale } => nlmrs::tiled_noise(rows, cols, scale, seed),
        Commands::BrownianMotion { rows, cols, n } => nlmrs::brownian_motion(rows, cols, n, seed),
        Commands::ForestFire { rows, cols, p_tree, p_lightning, iterations } => {
            nlmrs::forest_fire(rows, cols, p_tree, p_lightning, iterations, seed)
        }
        Commands::RiverNetwork { rows, cols } => nlmrs::river_network(rows, cols, seed),
        Commands::HexagonalVoronoi { rows, cols, n } => {
            nlmrs::hexagonal_voronoi(rows, cols, n, seed)
        }
    };

    let mut grid = grid;
    if let Some(n) = cli.classify {
        nlmrs::classify(&mut grid, n);
    }
    if let Some(t) = cli.threshold {
        nlmrs::threshold(&mut grid, t);
    }

    let path = &cli.output;
    let ext = path.rsplit('.').next().unwrap_or("png");

    let result = match ext {
        "csv" => export::write_to_csv(&grid, path),
        "json" => export::write_to_json(&grid, path),
        "asc" => export::write_to_ascii_grid(&grid, path),
        "tif" | "tiff" => export::write_to_tiff(&grid, path),
        "png" if cli.grayscale => export::write_to_png_grayscale(&grid, path),
        "png" | _ => export::write_to_png(&grid, path),
    };

    if let Err(e) = result {
        eprintln!("Error writing output: {e}");
        std::process::exit(1);
    }

    println!("Written {}×{} grid to {}", grid.rows, grid.cols, path);
}
