use crate::grid::Grid;
use crate::operation::scale;
use super::perlin_seed;
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
}
