use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub mod gradient;
pub mod hill_grow;
pub mod noise;
pub mod patch;

pub use gradient::*;
pub use hill_grow::*;
pub use noise::*;
pub use patch::*;

pub(crate) fn make_rng(seed: Option<u64>) -> StdRng {
    StdRng::seed_from_u64(seed.unwrap_or_else(|| rand::thread_rng().gen()))
}

/// Converts an optional 64-bit seed to a 32-bit Perlin seed, generating a random
/// one from thread_rng when none is provided.
pub(crate) fn perlin_seed(seed: Option<u64>) -> u32 {
    seed.map(|s| s as u32).unwrap_or_else(|| rand::thread_rng().gen())
}

#[cfg(test)]
pub(crate) fn nan_count(grid: &crate::Grid) -> usize {
    grid.iter().filter(|n| n.is_nan()).count()
}

#[cfg(test)]
pub(crate) fn zero_to_one_count(grid: &crate::Grid) -> usize {
    grid.iter().filter(|&&n| n >= 0. && n <= 1.).count()
}

#[cfg(test)]
mod tests {
    use crate::{classify, random, threshold, Grid};

    // ── classify ─────────────────────────────────────────────────────────────

    #[test]
    fn test_classify_two_classes() {
        let mut grid = random(50, 50, Some(1));
        classify(&mut grid, 2);
        for &v in grid.iter() {
            assert!(v == 0.0 || v == 1.0, "unexpected value {v}");
        }
    }

    #[test]
    fn test_classify_four_classes() {
        let mut grid = random(50, 50, Some(2));
        classify(&mut grid, 4);
        let allowed = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
        for &v in grid.iter() {
            assert!(
                allowed.iter().any(|&a| (v - a).abs() < 1e-10),
                "unexpected value {v}"
            );
        }
    }

    #[test]
    fn test_classify_one_class() {
        let mut grid = random(10, 10, Some(3));
        classify(&mut grid, 1);
        for &v in grid.iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_classify_boundary_at_one() {
        let mut grid = Grid::filled(1, 1, 1.0);
        classify(&mut grid, 3);
        assert_eq!(grid[0][0], 1.0);
    }

    // ── threshold ────────────────────────────────────────────────────────────

    #[test]
    fn test_threshold_binary() {
        let mut grid = random(50, 50, Some(4));
        let original = grid.clone();
        threshold(&mut grid, 0.5);
        for (i, (&orig, &new)) in original.iter().zip(grid.iter()).enumerate() {
            let expected = if orig < 0.5 { 0.0 } else { 1.0 };
            assert_eq!(new, expected, "cell {i}: original {orig}");
        }
    }

    #[test]
    fn test_threshold_all_ones() {
        let mut grid = random(10, 10, Some(5));
        threshold(&mut grid, 0.0);
        for &v in grid.iter() {
            assert_eq!(v, 1.0);
        }
    }

    #[test]
    fn test_threshold_all_zeros() {
        let mut grid = random(10, 10, Some(6));
        threshold(&mut grid, 1.1);
        for &v in grid.iter() {
            assert_eq!(v, 0.0);
        }
    }

}
