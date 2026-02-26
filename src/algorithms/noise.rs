use crate::grid::Grid;
use crate::operation::scale;
use super::{make_rng, perlin_seed};
use rand::Rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Returns a Perlin noise NLM with values ranging [0, 1).
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `scale_factor` - Frequency of the noise (higher = more features per unit).
/// * `seed` - Optional RNG seed for reproducible results.
pub fn perlin_noise(rows: usize, cols: usize, scale_factor: f64, seed: Option<u64>) -> Grid {
    use noise::{NoiseFn, Perlin};
    let seed_val = perlin_seed(seed);
    let perlin = Perlin::new(seed_val);

    let mut grid = Grid::new(rows, cols);
    let inv_rows = 1.0 / rows as f64;
    let inv_cols = 1.0 / cols as f64;
    let fill_row = |(i, row): (usize, &mut [f64])| {
        let ny = i as f64 * inv_rows * scale_factor;
        for (j, cell) in row.iter_mut().enumerate() {
            let nx = j as f64 * inv_cols * scale_factor;
            *cell = perlin.get([nx, ny]);
        }
    };
    #[cfg(feature = "parallel")]
    grid.data.par_chunks_mut(cols).enumerate().for_each(fill_row);
    #[cfg(not(feature = "parallel"))]
    grid.data.chunks_mut(cols).enumerate().for_each(fill_row);

    scale(&mut grid);
    grid
}

/// Returns a fractal Brownian motion (fBm) NLM with values ranging [0, 1).
///
/// Layers multiple octaves of Perlin noise for a more natural, detailed result.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `scale_factor` - Frequency of the base noise layer.
/// * `octaves` - Number of noise layers to combine (more = finer detail).
/// * `persistence` - Amplitude scaling per octave (0.5 = each octave half as strong).
/// * `lacunarity` - Frequency scaling per octave (2.0 = each octave twice the frequency).
/// * `seed` - Optional RNG seed for reproducible results.
pub fn fbm_noise(
    rows: usize,
    cols: usize,
    scale_factor: f64,
    octaves: usize,
    persistence: f64,
    lacunarity: f64,
    seed: Option<u64>,
) -> Grid {
    use noise::{NoiseFn, Perlin};
    let seed_val = perlin_seed(seed);
    let generators: Vec<Perlin> = (0..octaves)
        .map(|o| Perlin::new(seed_val.wrapping_add(o as u32)))
        .collect();

    let mut freq_amp: Vec<(f64, f64)> = Vec::with_capacity(octaves);
    {
        let mut amp = 1.0f64;
        let mut freq = scale_factor;
        let mut total = 0.0f64;
        for _ in 0..octaves {
            freq_amp.push((freq, amp));
            total += amp;
            amp *= persistence;
            freq *= lacunarity;
        }
        let inv_total = if total > 0.0 { 1.0 / total } else { 0.0 };
        for (_, a) in &mut freq_amp {
            *a *= inv_total;
        }
    }

    let mut grid = Grid::new(rows, cols);
    let inv_rows = 1.0 / rows as f64;
    let inv_cols = 1.0 / cols as f64;
    let fill_row = |(i, row): (usize, &mut [f64])| {
        let nys: Vec<f64> =
            freq_amp.iter().map(|&(freq, _)| i as f64 * inv_rows * freq).collect();
        for (j, cell) in row.iter_mut().enumerate() {
            let x = j as f64 * inv_cols;
            let mut value = 0.0;
            for (k, gen) in generators.iter().enumerate() {
                let (freq, amp) = freq_amp[k];
                value += gen.get([x * freq, nys[k]]) * amp;
            }
            *cell = value;
        }
    };
    #[cfg(feature = "parallel")]
    grid.data.par_chunks_mut(cols).enumerate().for_each(fill_row);
    #[cfg(not(feature = "parallel"))]
    grid.data.chunks_mut(cols).enumerate().for_each(fill_row);

    scale(&mut grid);
    grid
}

/// Returns a ridged multifractal NLM with values ranging [0, 1).
///
/// Produces sharp ridges and peak-like terrain.  Similar in structure to fBm
/// but each octave's value is folded (`1 - |x|`) so high-frequency details
/// accumulate into pronounced ridges rather than smooth hills.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `scale_factor` - Base noise frequency (higher = more features per unit).
/// * `octaves` - Number of noise layers to combine.
/// * `persistence` - Amplitude scaling per octave.
/// * `lacunarity` - Frequency scaling per octave (2.0 = each octave twice as dense).
/// * `seed` - Optional RNG seed for reproducible results.
pub fn ridged_noise(
    rows: usize,
    cols: usize,
    scale_factor: f64,
    octaves: usize,
    persistence: f64,
    lacunarity: f64,
    seed: Option<u64>,
) -> Grid {
    use noise::{MultiFractal, NoiseFn, Perlin, RidgedMulti};
    let seed_val = perlin_seed(seed);
    let ridged = RidgedMulti::<Perlin>::new(seed_val)
        .set_octaves(octaves)
        .set_persistence(persistence)
        .set_lacunarity(lacunarity);

    let mut grid = Grid::new(rows, cols);
    let inv_rows = 1.0 / rows as f64;
    let inv_cols = 1.0 / cols as f64;
    let fill_row = |(i, row): (usize, &mut [f64])| {
        let ny = i as f64 * inv_rows * scale_factor;
        for (j, cell) in row.iter_mut().enumerate() {
            let nx = j as f64 * inv_cols * scale_factor;
            *cell = ridged.get([nx, ny]);
        }
    };
    #[cfg(feature = "parallel")]
    grid.data.par_chunks_mut(cols).enumerate().for_each(fill_row);
    #[cfg(not(feature = "parallel"))]
    grid.data.chunks_mut(cols).enumerate().for_each(fill_row);

    scale(&mut grid);
    grid
}

/// Returns a billow NLM with values ranging [0, 1).
///
/// Billow noise applies an absolute-value fold to each octave of Perlin noise,
/// producing rounded, cloud- and hill-like patterns.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `scale_factor` - Base noise frequency.
/// * `octaves` - Number of noise layers to combine.
/// * `persistence` - Amplitude scaling per octave.
/// * `lacunarity` - Frequency scaling per octave.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn billow_noise(
    rows: usize,
    cols: usize,
    scale_factor: f64,
    octaves: usize,
    persistence: f64,
    lacunarity: f64,
    seed: Option<u64>,
) -> Grid {
    use noise::{Billow, MultiFractal, NoiseFn, Perlin};
    let seed_val = perlin_seed(seed);
    let billow = Billow::<Perlin>::new(seed_val)
        .set_octaves(octaves)
        .set_persistence(persistence)
        .set_lacunarity(lacunarity);

    let mut grid = Grid::new(rows, cols);
    let inv_rows = 1.0 / rows as f64;
    let inv_cols = 1.0 / cols as f64;
    let fill_row = |(i, row): (usize, &mut [f64])| {
        let ny = i as f64 * inv_rows * scale_factor;
        for (j, cell) in row.iter_mut().enumerate() {
            let nx = j as f64 * inv_cols * scale_factor;
            *cell = billow.get([nx, ny]);
        }
    };
    #[cfg(feature = "parallel")]
    grid.data.par_chunks_mut(cols).enumerate().for_each(fill_row);
    #[cfg(not(feature = "parallel"))]
    grid.data.chunks_mut(cols).enumerate().for_each(fill_row);

    scale(&mut grid);
    grid
}

/// Returns a Worley (cellular) noise NLM with values ranging [0, 1).
///
/// Each cell value is proportional to its distance to the nearest of a set of
/// randomly scattered Voronoi seed points, producing cellular / territory-like
/// patch patterns.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `scale_factor` - Frequency of the seed points (higher = smaller cells).
/// * `seed` - Optional RNG seed for reproducible results.
pub fn worley_noise(rows: usize, cols: usize, scale_factor: f64, seed: Option<u64>) -> Grid {
    use noise::{NoiseFn, Worley};
    let seed_val = perlin_seed(seed);

    let mut grid = Grid::new(rows, cols);
    let inv_rows = 1.0 / rows as f64;
    let inv_cols = 1.0 / cols as f64;
    // Worley contains Rc<dyn Fn> and is not Sync, so each row constructs its
    // own instance.  The permutation table creation is O(256) — negligible.
    let fill_row = |(i, row): (usize, &mut [f64])| {
        let worley = Worley::new(seed_val);
        let ny = i as f64 * inv_rows * scale_factor;
        for (j, cell) in row.iter_mut().enumerate() {
            let nx = j as f64 * inv_cols * scale_factor;
            *cell = worley.get([nx, ny]);
        }
    };
    #[cfg(feature = "parallel")]
    grid.data.par_chunks_mut(cols).enumerate().for_each(fill_row);
    #[cfg(not(feature = "parallel"))]
    grid.data.chunks_mut(cols).enumerate().for_each(fill_row);

    scale(&mut grid);
    grid
}

/// Returns a hybrid multifractal NLM with values ranging [0, 1).
///
/// Similar to ridged noise but uses the HybridMulti combiner, which blends
/// spectral characteristics between smooth and ridged noise.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `scale_factor` - Base noise frequency (higher = more features per unit).
/// * `octaves` - Number of noise layers to combine.
/// * `persistence` - Amplitude scaling per octave.
/// * `lacunarity` - Frequency scaling per octave.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn hybrid_noise(
    rows: usize,
    cols: usize,
    scale_factor: f64,
    octaves: usize,
    persistence: f64,
    lacunarity: f64,
    seed: Option<u64>,
) -> Grid {
    use noise::{HybridMulti, MultiFractal, NoiseFn, Perlin};
    let seed_val = perlin_seed(seed);
    let hybrid = HybridMulti::<Perlin>::new(seed_val)
        .set_octaves(octaves)
        .set_persistence(persistence)
        .set_lacunarity(lacunarity);

    let mut grid = Grid::new(rows, cols);
    let inv_rows = 1.0 / rows as f64;
    let inv_cols = 1.0 / cols as f64;
    let fill_row = |(i, row): (usize, &mut [f64])| {
        let ny = i as f64 * inv_rows * scale_factor;
        for (j, cell) in row.iter_mut().enumerate() {
            let nx = j as f64 * inv_cols * scale_factor;
            *cell = hybrid.get([nx, ny]);
        }
    };
    #[cfg(feature = "parallel")]
    grid.data.par_chunks_mut(cols).enumerate().for_each(fill_row);
    #[cfg(not(feature = "parallel"))]
    grid.data.chunks_mut(cols).enumerate().for_each(fill_row);

    scale(&mut grid);
    grid
}

/// Returns a value noise NLM with values ranging [0, 1).
///
/// Value noise interpolates random values assigned to a grid of lattice points,
/// producing blocky, low-frequency patterns compared to Perlin noise.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `scale_factor` - Frequency of the noise (higher = more features per unit).
/// * `seed` - Optional RNG seed for reproducible results.
pub fn value_noise(rows: usize, cols: usize, scale_factor: f64, seed: Option<u64>) -> Grid {
    use noise::{NoiseFn, Value};
    let seed_val = perlin_seed(seed);
    let v = Value::new(seed_val);

    let mut grid = Grid::new(rows, cols);
    let inv_rows = 1.0 / rows as f64;
    let inv_cols = 1.0 / cols as f64;
    let fill_row = |(i, row): (usize, &mut [f64])| {
        let ny = i as f64 * inv_rows * scale_factor;
        for (j, cell) in row.iter_mut().enumerate() {
            let nx = j as f64 * inv_cols * scale_factor;
            *cell = v.get([nx, ny]);
        }
    };
    #[cfg(feature = "parallel")]
    grid.data.par_chunks_mut(cols).enumerate().for_each(fill_row);
    #[cfg(not(feature = "parallel"))]
    grid.data.chunks_mut(cols).enumerate().for_each(fill_row);

    scale(&mut grid);
    grid
}

/// Returns a turbulence NLM with values ranging [0, 1).
///
/// Same as fBm but accumulates the absolute value of each octave's contribution,
/// producing sharper, more chaotic textures.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `scale_factor` - Frequency of the base noise layer.
/// * `octaves` - Number of noise layers to combine.
/// * `persistence` - Amplitude scaling per octave.
/// * `lacunarity` - Frequency scaling per octave.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn turbulence(
    rows: usize,
    cols: usize,
    scale_factor: f64,
    octaves: usize,
    persistence: f64,
    lacunarity: f64,
    seed: Option<u64>,
) -> Grid {
    use noise::{NoiseFn, Perlin};
    let seed_val = perlin_seed(seed);
    let generators: Vec<Perlin> = (0..octaves)
        .map(|o| Perlin::new(seed_val.wrapping_add(o as u32)))
        .collect();

    let mut freq_amp: Vec<(f64, f64)> = Vec::with_capacity(octaves);
    {
        let mut amp = 1.0f64;
        let mut freq = scale_factor;
        let mut total = 0.0f64;
        for _ in 0..octaves {
            freq_amp.push((freq, amp));
            total += amp;
            amp *= persistence;
            freq *= lacunarity;
        }
        let inv_total = if total > 0.0 { 1.0 / total } else { 0.0 };
        for (_, a) in &mut freq_amp {
            *a *= inv_total;
        }
    }

    let mut grid = Grid::new(rows, cols);
    let inv_rows = 1.0 / rows as f64;
    let inv_cols = 1.0 / cols as f64;
    let fill_row = |(i, row): (usize, &mut [f64])| {
        let nys: Vec<f64> =
            freq_amp.iter().map(|&(freq, _)| i as f64 * inv_rows * freq).collect();
        for (j, cell) in row.iter_mut().enumerate() {
            let x = j as f64 * inv_cols;
            let mut value = 0.0;
            for (k, gen) in generators.iter().enumerate() {
                let (freq, amp) = freq_amp[k];
                value += gen.get([x * freq, nys[k]]).abs() * amp;
            }
            *cell = value;
        }
    };
    #[cfg(feature = "parallel")]
    grid.data.par_chunks_mut(cols).enumerate().for_each(fill_row);
    #[cfg(not(feature = "parallel"))]
    grid.data.chunks_mut(cols).enumerate().for_each(fill_row);

    scale(&mut grid);
    grid
}

/// Returns a domain-warped Perlin noise NLM with values ranging [0, 1).
///
/// Displaces the sample coordinates of a base Perlin generator using a
/// secondary warp generator, producing highly organic, swirling patterns.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `scale_factor` - Coordinate frequency (higher = more features per unit).
/// * `warp_strength` - Displacement magnitude applied to sample coordinates.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn domain_warp(
    rows: usize,
    cols: usize,
    scale_factor: f64,
    warp_strength: f64,
    seed: Option<u64>,
) -> Grid {
    use noise::{NoiseFn, Perlin};
    let seed_val = perlin_seed(seed);
    let warp = Perlin::new(seed_val);
    let base = Perlin::new(seed_val.wrapping_add(1));

    let mut grid = Grid::new(rows, cols);
    let inv_rows = 1.0 / rows as f64;
    let inv_cols = 1.0 / cols as f64;
    // Perlin is Sync (no Rc), so sharing across rayon threads is safe.
    let fill_row = |(i, row): (usize, &mut [f64])| {
        let ny = i as f64 * inv_rows * scale_factor;
        for (j, cell) in row.iter_mut().enumerate() {
            let nx = j as f64 * inv_cols * scale_factor;
            let wx = warp.get([nx, ny]) * warp_strength;
            // Offset by irrational amounts to prevent correlation on the axes.
            let wy = warp.get([nx + 3.7, ny + 1.9]) * warp_strength;
            *cell = base.get([nx + wx, ny + wy]);
        }
    };
    #[cfg(feature = "parallel")]
    grid.data.par_chunks_mut(cols).enumerate().for_each(fill_row);
    #[cfg(not(feature = "parallel"))]
    grid.data.chunks_mut(cols).enumerate().for_each(fill_row);

    scale(&mut grid);
    grid
}

/// Returns a spectral synthesis NLM with values ranging [0, 1).
///
/// Generates correlated noise in the frequency domain: each complex frequency
/// component is assigned a random phase and an amplitude proportional to
/// `f^(-beta/2)`, giving a power spectrum ∝ `1/f^beta` (1/f noise).
/// The result is the real part of the 2-D inverse FFT, normalised to [0, 1).
///
/// # Arguments
///
/// * `rows`  - Number of rows.
/// * `cols`  - Number of columns.
/// * `beta`  - Spectral exponent. 0 = white noise, 1 = pink noise,
///             2 = red/brown noise (Brownian landscape), higher = smoother.
/// * `seed`  - Optional RNG seed for reproducible results.
pub fn spectral_synthesis(rows: usize, cols: usize, beta: f64, seed: Option<u64>) -> Grid {
    use rustfft::{FftPlanner, num_complex::Complex};
    use super::make_rng;
    use rand::Rng;

    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }

    let mut rng = make_rng(seed);

    // Build frequency-domain grid: random phase, amplitude ∝ f^(-beta/2).
    let mut freq: Vec<Complex<f64>> = (0..rows * cols)
        .map(|_| Complex::new(0.0f64, 0.0))
        .collect();

    for i in 0..rows {
        // Map to centred frequency coordinates.
        let fi = if i <= rows / 2 { i as f64 } else { i as f64 - rows as f64 };
        for j in 0..cols {
            if i == 0 && j == 0 {
                continue; // DC component stays 0 → zero mean
            }
            let fj = if j <= cols / 2 { j as f64 } else { j as f64 - cols as f64 };
            let f = (fi * fi + fj * fj).sqrt();
            let amplitude = f.powf(-beta / 2.0);
            let phase = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
            freq[i * cols + j] = Complex::new(amplitude * phase.cos(), amplitude * phase.sin());
        }
    }

    // 2-D IFFT: IFFT each row, then each column.
    let mut planner = FftPlanner::<f64>::new();
    let ifft_row = planner.plan_fft_inverse(cols);
    let ifft_col = planner.plan_fft_inverse(rows);

    for row in freq.chunks_mut(cols) {
        ifft_row.process(row);
    }

    let mut col_buf = vec![Complex::new(0.0f64, 0.0); rows];
    for j in 0..cols {
        for i in 0..rows {
            col_buf[i] = freq[i * cols + j];
        }
        ifft_col.process(&mut col_buf);
        for i in 0..rows {
            freq[i * cols + j] = col_buf[i];
        }
    }

    let data: Vec<f64> = freq.iter().map(|c| c.re).collect();
    let mut grid = Grid { data, rows, cols };
    scale(&mut grid);
    grid
}

/// Returns a fractal Brownian surface NLM parameterised by the Hurst exponent. Values in [0, 1).
///
/// Uses the spectral synthesis method. The Hurst exponent `h` directly controls
/// the fractal dimension of the surface and has direct ecological meaning:
/// `h` near 0 produces rough, fine-grained textures; `h` near 1 produces
/// smooth, long-range correlated surfaces resembling natural terrain.
///
/// The relationship to the spectral exponent β used internally is β = 2h + 2.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `h`    - Hurst exponent in (0, 1). Default 0.5.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn fractal_brownian_surface(rows: usize, cols: usize, h: f64, seed: Option<u64>) -> Grid {
    let beta = 2.0 * h.clamp(1e-6, 1.0 - 1e-6) + 2.0;
    spectral_synthesis(rows, cols, beta, seed)
}

/// Returns an OpenSimplex noise NLM with values ranging [0, 1).
///
/// OpenSimplex is a patent-free alternative to Perlin noise with a rounder
/// feature shape, fewer directional artefacts, and a different visual character.
///
/// # Arguments
///
/// * `rows`         - Number of rows.
/// * `cols`         - Number of columns.
/// * `scale_factor` - Noise frequency (higher = more features per unit area).
/// * `seed`         - Optional RNG seed for reproducible results.
pub fn simplex_noise(rows: usize, cols: usize, scale_factor: f64, seed: Option<u64>) -> Grid {
    use noise::{NoiseFn, OpenSimplex};
    let seed_val = perlin_seed(seed);
    let gen = OpenSimplex::new(seed_val);

    let mut grid = Grid::new(rows, cols);
    let inv_rows = 1.0 / rows as f64;
    let inv_cols = 1.0 / cols as f64;
    let fill_row = |(i, row): (usize, &mut [f64])| {
        let ny = i as f64 * inv_rows * scale_factor;
        for (j, cell) in row.iter_mut().enumerate() {
            let nx = j as f64 * inv_cols * scale_factor;
            *cell = gen.get([nx, ny]);
        }
    };
    #[cfg(feature = "parallel")]
    grid.data.par_chunks_mut(cols).enumerate().for_each(fill_row);
    #[cfg(not(feature = "parallel"))]
    grid.data.chunks_mut(cols).enumerate().for_each(fill_row);

    scale(&mut grid);
    grid
}

/// Returns a Voronoi distance NLM with values ranging [0, 1).
///
/// Scatters `n` random feature points across the grid, then fills each cell
/// with the Euclidean distance to the nearest point. Produces a smooth field
/// of conical gradients centred on each feature point.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n`    - Number of feature points to scatter.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn voronoi_distance(rows: usize, cols: usize, n: usize, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(0, 0);
    }
    let mut rng = make_rng(seed);
    let points: Vec<(f64, f64)> = (0..n.max(1))
        .map(|_| (rng.gen::<f64>() * rows as f64, rng.gen::<f64>() * cols as f64))
        .collect();

    let mut grid = Grid::new(rows, cols);
    let fill_row = |(i, row): (usize, &mut [f64])| {
        let pr = i as f64 + 0.5;
        for (j, cell) in row.iter_mut().enumerate() {
            let pc = j as f64 + 0.5;
            *cell = points
                .iter()
                .map(|&(sr, sc)| {
                    let dr = pr - sr;
                    let dc = pc - sc;
                    (dr * dr + dc * dc).sqrt()
                })
                .fold(f64::INFINITY, f64::min);
        }
    };
    #[cfg(feature = "parallel")]
    grid.data.par_chunks_mut(cols).enumerate().for_each(fill_row);
    #[cfg(not(feature = "parallel"))]
    grid.data.chunks_mut(cols).enumerate().for_each(fill_row);

    scale(&mut grid);
    grid
}

/// Returns a sine composite NLM with values ranging [0, 1).
///
/// Superposes `waves` sinusoidal plane waves, each with a random orientation,
/// frequency, and phase. The superposition produces standing-wave interference
/// patterns whose complexity scales with the number of waves.
///
/// # Arguments
///
/// * `rows`  - Number of rows.
/// * `cols`  - Number of columns.
/// * `waves` - Number of sinusoidal waves to superpose.
/// * `seed`  - Optional RNG seed for reproducible results.
pub fn sine_composite(rows: usize, cols: usize, waves: usize, seed: Option<u64>) -> Grid {
    use std::f64::consts::PI;
    let mut rng = make_rng(seed);

    let n = waves.max(1);
    // (cos_angle, sin_angle, frequency, phase)
    let params: Vec<(f64, f64, f64, f64)> = (0..n)
        .map(|_| {
            let angle = rng.gen::<f64>() * 2.0 * PI;
            let freq = rng.gen::<f64>() * 3.5 + 0.5;
            let phase = rng.gen::<f64>() * 2.0 * PI;
            (angle.cos(), angle.sin(), freq, phase)
        })
        .collect();

    let inv_rows = 1.0 / rows as f64;
    let inv_cols = 1.0 / cols as f64;
    let inv_n = 1.0 / n as f64;

    let mut grid = Grid::new(rows, cols);
    let fill_row = |(i, row): (usize, &mut [f64])| {
        let ny = i as f64 * inv_rows;
        for (j, cell) in row.iter_mut().enumerate() {
            let nx = j as f64 * inv_cols;
            *cell = params
                .iter()
                .map(|&(ca, sa, freq, phase)| {
                    (2.0 * PI * freq * (ca * nx + sa * ny) + phase).sin()
                })
                .sum::<f64>()
                * inv_n;
        }
    };
    #[cfg(feature = "parallel")]
    grid.data.par_chunks_mut(cols).enumerate().for_each(fill_row);
    #[cfg(not(feature = "parallel"))]
    grid.data.chunks_mut(cols).enumerate().for_each(fill_row);

    scale(&mut grid);
    grid
}

/// Returns a curl noise NLM with values ranging [0, 1).
///
/// Computes the curl (gradient rotated 90°) of a Perlin potential field using
/// finite differences, producing a divergence-free velocity field. The velocity
/// is then used to warp the sample coordinates of a second Perlin generator,
/// yielding swirling, flow-aligned patterns without directional clumping.
///
/// # Arguments
///
/// * `rows`         - Number of rows.
/// * `cols`         - Number of columns.
/// * `scale_factor` - Coordinate frequency (higher = more features per unit).
/// * `seed`         - Optional RNG seed for reproducible results.
pub fn curl_noise(rows: usize, cols: usize, scale_factor: f64, seed: Option<u64>) -> Grid {
    use noise::{NoiseFn, Perlin};
    let seed_val = perlin_seed(seed);
    let potential = Perlin::new(seed_val);
    let base = Perlin::new(seed_val.wrapping_add(1));

    let mut grid = Grid::new(rows, cols);
    let inv_rows = 1.0 / rows as f64;
    let inv_cols = 1.0 / cols as f64;
    // Finite-difference step: half a scaled cell width.
    let eps = inv_rows.min(inv_cols) * scale_factor * 0.5;

    // Curl of potential: vx = dP/dy, vy = -dP/dx (divergence-free).
    // Warp base-field sample coordinates by the curl velocity vector.
    let fill_row = |(i, row): (usize, &mut [f64])| {
        let ny = i as f64 * inv_rows * scale_factor;
        for (j, cell) in row.iter_mut().enumerate() {
            let nx = j as f64 * inv_cols * scale_factor;
            let vx =
                (potential.get([nx, ny + eps]) - potential.get([nx, ny - eps])) / (2.0 * eps);
            let vy =
                -(potential.get([nx + eps, ny]) - potential.get([nx - eps, ny])) / (2.0 * eps);
            *cell = base.get([nx + vx, ny + vy]);
        }
    };
    #[cfg(feature = "parallel")]
    grid.data.par_chunks_mut(cols).enumerate().for_each(fill_row);
    #[cfg(not(feature = "parallel"))]
    grid.data.chunks_mut(cols).enumerate().for_each(fill_row);

    scale(&mut grid);
    grid
}

/// Returns a Gabor noise NLM with values ranging [0, 1).
///
/// Sums `n` randomly placed, randomly oriented Gabor kernels — each the product
/// of a Gaussian envelope and a cosine carrier — producing coherent, anisotropic
/// textures with a controllable spatial frequency.  More kernels yield a smoother
/// result (central-limit theorem averaging).
///
/// # Arguments
///
/// * `rows`         - Number of rows.
/// * `cols`         - Number of columns.
/// * `scale_factor` - Controls carrier frequency and envelope width (higher = finer features).
/// * `n`            - Number of kernels to place.
/// * `seed`         - Optional RNG seed for reproducible results.
pub fn gabor_noise(rows: usize, cols: usize, scale_factor: f64, n: usize, seed: Option<u64>) -> Grid {
    use std::f64::consts::TAU;
    if rows == 0 || cols == 0 {
        return Grid::new(rows, cols);
    }
    let mut rng = make_rng(seed);
    let max_dim = rows.max(cols) as f64;
    let sigma = max_dim / (2.0 * scale_factor.max(0.1));
    let freq = scale_factor / max_dim;
    let sigma_sq = sigma * sigma;
    let support_r_sq = (3.0 * sigma).powi(2);

    // Precompute (cx, cy, cos_θ, sin_θ, phase) for every kernel.
    let kernels: Vec<(f64, f64, f64, f64, f64)> = (0..n.max(1))
        .map(|_| {
            let cx = rng.gen::<f64>() * cols as f64;
            let cy = rng.gen::<f64>() * rows as f64;
            let theta = rng.gen::<f64>() * std::f64::consts::PI;
            let phase = rng.gen::<f64>() * TAU;
            (cx, cy, theta.cos(), theta.sin(), phase)
        })
        .collect();

    let mut data = vec![0.0f64; rows * cols];
    let fill = |(idx, cell): (usize, &mut f64)| {
        let x = (idx % cols) as f64;
        let y = (idx / cols) as f64;
        let mut val = 0.0f64;
        for &(cx, cy, cos_t, sin_t, phase) in &kernels {
            let dx = x - cx;
            let dy = y - cy;
            if dx * dx + dy * dy > support_r_sq {
                continue;
            }
            let xp = dx * cos_t + dy * sin_t;
            let yp = -dx * sin_t + dy * cos_t;
            val += (-(xp * xp + yp * yp) / (2.0 * sigma_sq)).exp()
                * (TAU * freq * xp + phase).cos();
        }
        *cell = val;
    };
    #[cfg(feature = "parallel")]
    data.par_iter_mut().enumerate().for_each(fill);
    #[cfg(not(feature = "parallel"))]
    data.iter_mut().enumerate().for_each(fill);

    let mut grid = Grid { data, rows, cols };
    scale(&mut grid);
    grid
}

/// Returns a spot noise NLM with values ranging [0, 1).
///
/// Places `n` randomly oriented elliptical Gaussian spots at random positions
/// and accumulates their contributions.  Unlike `gaussian_blobs` (circular,
/// uniform width), each spot has an independent orientation and aspect ratio,
/// producing anisotropic brush-stroke-like textures.
///
/// # Arguments
///
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `n`    - Number of spots to place.
/// * `seed` - Optional RNG seed for reproducible results.
pub fn spot_noise(rows: usize, cols: usize, n: usize, seed: Option<u64>) -> Grid {
    if rows == 0 || cols == 0 {
        return Grid::new(rows, cols);
    }
    let mut rng = make_rng(seed);
    let base = rows.max(cols) as f64 / 8.0;

    // (cx, cy, cos_θ, sin_θ, 1/(2σ_maj²), 1/(2σ_min²))
    let spots: Vec<(f64, f64, f64, f64, f64, f64)> = (0..n.max(1))
        .map(|_| {
            let cx = rng.gen::<f64>() * cols as f64;
            let cy = rng.gen::<f64>() * rows as f64;
            let theta = rng.gen::<f64>() * std::f64::consts::PI;
            let s_maj = base * rng.gen_range(0.5f64..2.0);
            let s_min = s_maj * rng.gen_range(0.2f64..0.6);
            (cx, cy, theta.cos(), theta.sin(),
             1.0 / (2.0 * s_maj * s_maj),
             1.0 / (2.0 * s_min * s_min))
        })
        .collect();

    let mut data = vec![0.0f64; rows * cols];
    let fill = |(idx, cell): (usize, &mut f64)| {
        let x = (idx % cols) as f64;
        let y = (idx / cols) as f64;
        let mut val = 0.0f64;
        for &(cx, cy, cos_t, sin_t, inv2sx, inv2sy) in &spots {
            let dx = x - cx;
            let dy = y - cy;
            let xp = dx * cos_t + dy * sin_t;
            let yp = -dx * sin_t + dy * cos_t;
            val += (-(xp * xp * inv2sx + yp * yp * inv2sy)).exp();
        }
        *cell = val;
    };
    #[cfg(feature = "parallel")]
    data.par_iter_mut().enumerate().for_each(fill);
    #[cfg(not(feature = "parallel"))]
    data.iter_mut().enumerate().for_each(fill);

    let mut grid = Grid { data, rows, cols };
    scale(&mut grid);
    grid
}

/// Returns an anisotropic fBm NLM with values ranging [0, 1).
///
/// Applies a directional stretch to the noise coordinate space before sampling
/// fBm, creating elongated features aligned with `direction`.  A `stretch` of
/// 4.0 compresses the perpendicular axis 4× relative to the primary axis,
/// producing ridge-like or streak-like textures.
///
/// # Arguments
///
/// * `rows`         - Number of rows.
/// * `cols`         - Number of columns.
/// * `scale_factor` - Base noise frequency along the primary axis.
/// * `octaves`      - Number of noise layers to combine.
/// * `direction`    - Orientation of elongation in degrees [0, 360).
/// * `stretch`      - Compression ratio for the perpendicular axis (≥ 1.0).
/// * `seed`         - Optional RNG seed for reproducible results.
pub fn anisotropic_noise(
    rows: usize,
    cols: usize,
    scale_factor: f64,
    octaves: usize,
    direction: f64,
    stretch: f64,
    seed: Option<u64>,
) -> Grid {
    use noise::{NoiseFn, Perlin};
    let seed_val = perlin_seed(seed);
    let oct = octaves.max(1);
    let generators: Vec<Perlin> = (0..oct)
        .map(|o| Perlin::new(seed_val.wrapping_add(o as u32)))
        .collect();

    let angle = direction.to_radians();
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let stretch = stretch.max(1.0);

    // Build normalised (freq, amp) pairs using standard FBM parameters.
    let freq_amp: Vec<(f64, f64)> = {
        let mut v = Vec::with_capacity(oct);
        let mut amp = 1.0f64;
        let mut freq = scale_factor;
        let mut total = 0.0f64;
        for _ in 0..oct {
            v.push((freq, amp));
            total += amp;
            amp *= 0.5;
            freq *= 2.0;
        }
        let inv = if total > 0.0 { 1.0 / total } else { 0.0 };
        for (_, a) in &mut v { *a *= inv; }
        v
    };

    let inv_rows = 1.0 / rows as f64;
    let inv_cols = 1.0 / cols as f64;
    let mut grid = Grid::new(rows, cols);
    let fill_row = |(i, row): (usize, &mut [f64])| {
        let ny = i as f64 * inv_rows;
        for (j, cell) in row.iter_mut().enumerate() {
            let nx = j as f64 * inv_cols;
            // Rotate into the anisotropy frame, then scale with stretch.
            let xr =  nx * cos_a + ny * sin_a;
            let yr = -nx * sin_a + ny * cos_a;
            let mut value = 0.0f64;
            for (k, gen) in generators.iter().enumerate() {
                let (freq, amp) = freq_amp[k];
                value += gen.get([xr * freq, yr * freq * stretch]) * amp;
            }
            *cell = value;
        }
    };
    #[cfg(feature = "parallel")]
    grid.data.par_chunks_mut(cols).enumerate().for_each(fill_row);
    #[cfg(not(feature = "parallel"))]
    grid.data.chunks_mut(cols).enumerate().for_each(fill_row);

    scale(&mut grid);
    grid
}

/// Returns a seamlessly tileable Perlin noise NLM with values ranging [0, 1).
///
/// Maps the 2-D grid coordinates onto a torus in 4-D via the standard
/// `(cos θ, sin θ)` embedding, then samples 4-D Perlin noise.  The result
/// tiles perfectly: the value at the left edge matches the right edge, and
/// the top matches the bottom.
///
/// # Arguments
///
/// * `rows`         - Number of rows.
/// * `cols`         - Number of columns.
/// * `scale_factor` - Number of noise cycles per tile (higher = more features).
/// * `seed`         - Optional RNG seed for reproducible results.
pub fn tiled_noise(rows: usize, cols: usize, scale_factor: f64, seed: Option<u64>) -> Grid {
    use noise::{NoiseFn, Perlin};
    use std::f64::consts::TAU;
    if rows == 0 || cols == 0 {
        return Grid::new(rows, cols);
    }
    let seed_val = perlin_seed(seed);
    let perlin = Perlin::new(seed_val);
    let r = scale_factor;
    let inv_cols = TAU / cols as f64;
    let inv_rows = TAU / rows as f64;

    let mut data = vec![0.0f64; rows * cols];
    let fill = |(idx, cell): (usize, &mut f64)| {
        let tx = (idx % cols) as f64 * inv_cols;
        let ty = (idx / cols) as f64 * inv_rows;
        *cell = perlin.get([tx.cos() * r, tx.sin() * r, ty.cos() * r, ty.sin() * r]);
    };
    #[cfg(feature = "parallel")]
    data.par_iter_mut().enumerate().for_each(fill);
    #[cfg(not(feature = "parallel"))]
    data.iter_mut().enumerate().for_each(fill);

    let mut grid = Grid { data, rows, cols };
    scale(&mut grid);
    grid
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{nan_count, zero_to_one_count};
    use rstest::rstest;

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_perlin_noise(#[case] rows: usize, #[case] cols: usize) {
        let grid = perlin_noise(rows, cols, 4.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_perlin_seeded_determinism() {
        let a = perlin_noise(50, 50, 4.0, Some(42));
        let b = perlin_noise(50, 50, 4.0, Some(42));
        assert_eq!(a.data, b.data);
    }

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_fbm_noise(#[case] rows: usize, #[case] cols: usize) {
        let grid = fbm_noise(rows, cols, 4.0, 6, 0.5, 2.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    // ── ridged_noise ──────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_ridged_noise(#[case] rows: usize, #[case] cols: usize) {
        let grid = ridged_noise(rows, cols, 4.0, 6, 0.5, 2.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_ridged_noise_seeded_determinism() {
        let a = ridged_noise(50, 50, 4.0, 6, 0.5, 2.0, Some(42));
        let b = ridged_noise(50, 50, 4.0, 6, 0.5, 2.0, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── billow_noise ──────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_billow_noise(#[case] rows: usize, #[case] cols: usize) {
        let grid = billow_noise(rows, cols, 4.0, 6, 0.5, 2.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_billow_noise_seeded_determinism() {
        let a = billow_noise(50, 50, 4.0, 6, 0.5, 2.0, Some(42));
        let b = billow_noise(50, 50, 4.0, 6, 0.5, 2.0, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── worley_noise ──────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_worley_noise(#[case] rows: usize, #[case] cols: usize) {
        let grid = worley_noise(rows, cols, 4.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_worley_noise_seeded_determinism() {
        let a = worley_noise(50, 50, 4.0, Some(42));
        let b = worley_noise(50, 50, 4.0, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── hybrid_noise ──────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_hybrid_noise(#[case] rows: usize, #[case] cols: usize) {
        let grid = hybrid_noise(rows, cols, 4.0, 6, 0.5, 2.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_hybrid_noise_seeded_determinism() {
        let a = hybrid_noise(50, 50, 4.0, 6, 0.5, 2.0, Some(42));
        let b = hybrid_noise(50, 50, 4.0, 6, 0.5, 2.0, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── value_noise ───────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_value_noise(#[case] rows: usize, #[case] cols: usize) {
        let grid = value_noise(rows, cols, 4.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_value_noise_seeded_determinism() {
        let a = value_noise(50, 50, 4.0, Some(42));
        let b = value_noise(50, 50, 4.0, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── turbulence ────────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_turbulence(#[case] rows: usize, #[case] cols: usize) {
        let grid = turbulence(rows, cols, 4.0, 6, 0.5, 2.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_turbulence_seeded_determinism() {
        let a = turbulence(50, 50, 4.0, 6, 0.5, 2.0, Some(42));
        let b = turbulence(50, 50, 4.0, 6, 0.5, 2.0, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── domain_warp ───────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_domain_warp(#[case] rows: usize, #[case] cols: usize) {
        let grid = domain_warp(rows, cols, 4.0, 1.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_domain_warp_seeded_determinism() {
        let a = domain_warp(50, 50, 4.0, 1.0, Some(42));
        let b = domain_warp(50, 50, 4.0, 1.0, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── spectral_synthesis ────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_spectral_synthesis(#[case] rows: usize, #[case] cols: usize) {
        let grid = spectral_synthesis(rows, cols, 2.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_spectral_synthesis_seeded_determinism() {
        let a = spectral_synthesis(50, 50, 2.0, Some(42));
        let b = spectral_synthesis(50, 50, 2.0, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── fractal_brownian_surface ───────────────────────────────────────────────

    #[rstest]
    #[case(2, 2)]
    #[case(10, 10)]
    #[case(50, 50)]
    fn test_fractal_brownian_surface(#[case] rows: usize, #[case] cols: usize) {
        let grid = fractal_brownian_surface(rows, cols, 0.5, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_fractal_brownian_surface_seeded_determinism() {
        let a = fractal_brownian_surface(50, 50, 0.5, Some(42));
        let b = fractal_brownian_surface(50, 50, 0.5, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── simplex_noise ─────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_simplex_noise(#[case] rows: usize, #[case] cols: usize) {
        let grid = simplex_noise(rows, cols, 4.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_simplex_noise_seeded_determinism() {
        let a = simplex_noise(50, 50, 4.0, Some(42));
        let b = simplex_noise(50, 50, 4.0, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── voronoi_distance ──────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_voronoi_distance(#[case] rows: usize, #[case] cols: usize) {
        let grid = voronoi_distance(rows, cols, 20, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_voronoi_distance_seeded_determinism() {
        let a = voronoi_distance(50, 50, 20, Some(42));
        let b = voronoi_distance(50, 50, 20, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── sine_composite ────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_sine_composite(#[case] rows: usize, #[case] cols: usize) {
        let grid = sine_composite(rows, cols, 8, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_sine_composite_seeded_determinism() {
        let a = sine_composite(50, 50, 8, Some(42));
        let b = sine_composite(50, 50, 8, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── curl_noise ────────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_curl_noise(#[case] rows: usize, #[case] cols: usize) {
        let grid = curl_noise(rows, cols, 4.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_curl_noise_seeded_determinism() {
        let a = curl_noise(50, 50, 4.0, Some(42));
        let b = curl_noise(50, 50, 4.0, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── gabor_noise ───────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_gabor_noise(#[case] rows: usize, #[case] cols: usize) {
        let grid = gabor_noise(rows, cols, 4.0, 200, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_gabor_noise_seeded_determinism() {
        let a = gabor_noise(50, 50, 4.0, 200, Some(42));
        let b = gabor_noise(50, 50, 4.0, 200, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── spot_noise ────────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_spot_noise(#[case] rows: usize, #[case] cols: usize) {
        let grid = spot_noise(rows, cols, 100, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_spot_noise_seeded_determinism() {
        let a = spot_noise(50, 50, 100, Some(42));
        let b = spot_noise(50, 50, 100, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── anisotropic_noise ─────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_anisotropic_noise(#[case] rows: usize, #[case] cols: usize) {
        let grid = anisotropic_noise(rows, cols, 4.0, 6, 45.0, 4.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_anisotropic_noise_seeded_determinism() {
        let a = anisotropic_noise(50, 50, 4.0, 6, 45.0, 4.0, Some(42));
        let b = anisotropic_noise(50, 50, 4.0, 6, 45.0, 4.0, Some(42));
        assert_eq!(a.data, b.data);
    }

    // ── tiled_noise ───────────────────────────────────────────────────────────

    #[rstest]
    #[case(1, 1)]
    #[case(10, 10)]
    #[case(100, 100)]
    fn test_tiled_noise(#[case] rows: usize, #[case] cols: usize) {
        let grid = tiled_noise(rows, cols, 4.0, None);
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);
        assert_eq!(nan_count(&grid), 0);
        assert_eq!(zero_to_one_count(&grid), rows * cols);
    }

    #[test]
    fn test_tiled_noise_seeded_determinism() {
        let a = tiled_noise(50, 50, 4.0, Some(42));
        let b = tiled_noise(50, 50, 4.0, Some(42));
        assert_eq!(a.data, b.data);
    }

}
