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
        Commands::NeighbourhoodClustering { rows, cols, k, iterations } => {
            nlmrs::neighbourhood_clustering(rows, cols, k, iterations, seed)
        }
        Commands::SpectralSynthesis { rows, cols, beta } => {
            nlmrs::spectral_synthesis(rows, cols, beta, seed)
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
