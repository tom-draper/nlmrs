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
    };

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
