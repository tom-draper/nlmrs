use rand::Rng;

/// A Fenwick tree (Binary Indexed Tree) for O(log n) weighted random sampling.
///
/// Replaces the O(n) Vec rebuild + WeightedIndex pattern in hot loops.
/// All weights must be non-negative.
pub struct WeightedSampler {
    /// 1-indexed internal tree of prefix sums.
    tree: Vec<f64>,
    n: usize,
}

impl WeightedSampler {
    /// Builds the tree from a slice of non-negative weights in O(n log n).
    pub fn new(weights: &[f64]) -> Self {
        let n = weights.len();
        let mut tree = vec![0.0f64; n + 1];
        for (i, &w) in weights.iter().enumerate() {
            let mut j = i + 1;
            while j <= n {
                tree[j] += w;
                j += j & j.wrapping_neg();
            }
        }
        WeightedSampler { tree, n }
    }

    /// Adds `delta` to the weight at 0-indexed position `i` in O(log n).
    pub fn update(&mut self, i: usize, delta: f64) {
        let mut j = i + 1;
        while j <= self.n {
            self.tree[j] += delta;
            j += j & j.wrapping_neg();
        }
    }

    /// Returns the sum of all weights.
    pub fn total(&self) -> f64 {
        // prefix_sum over all n elements
        let mut sum = 0.0;
        let mut j = self.n;
        while j > 0 {
            sum += self.tree[j];
            j -= j & j.wrapping_neg();
        }
        sum
    }

    /// Samples a 0-indexed element with probability proportional to its weight in O(log n).
    ///
    /// Panics if the total weight is zero or negative.
    pub fn sample(&self, rng: &mut impl Rng) -> usize {
        let total = self.total();
        debug_assert!(total > 0.0, "Cannot sample from zero-weight distribution");
        let mut target = rng.gen::<f64>() * total;

        // Walk the tree from the highest bit down, finding the leftmost
        // position where the cumulative prefix sum exceeds target.
        let mut pos = 0usize;
        let mut bit = self.n.next_power_of_two();
        while bit > 0 {
            let next = pos + bit;
            if next <= self.n && self.tree[next] <= target {
                target -= self.tree[next];
                pos = next;
            }
            bit >>= 1;
        }
        // pos is now the 0-indexed result
        pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_total() {
        let s = WeightedSampler::new(&[1.0, 2.0, 3.0, 4.0]);
        assert!((s.total() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_update() {
        let mut s = WeightedSampler::new(&[1.0, 2.0, 3.0]);
        s.update(1, 5.0); // weight[1] becomes 7.0
        assert!((s.total() - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_distribution() {
        // weights [0, 1, 0, 0] → always sample index 1
        let s = WeightedSampler::new(&[0.0, 1.0, 0.0, 0.0]);
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            assert_eq!(s.sample(&mut rng), 1);
        }
    }

    #[test]
    fn test_sample_uniform() {
        // Equal weights → roughly uniform distribution
        let n = 4;
        let s = WeightedSampler::new(&vec![1.0; n]);
        let mut rng = StdRng::seed_from_u64(42);
        let mut counts = vec![0usize; n];
        for _ in 0..10000 {
            counts[s.sample(&mut rng)] += 1;
        }
        // Each bucket should be ~2500 ± some variance
        for &c in &counts {
            assert!(c > 2000 && c < 3000, "Bucket count {c} out of expected range");
        }
    }

    #[test]
    fn test_sample_weighted() {
        // weights [1, 3] → index 1 sampled ~75% of the time
        let s = WeightedSampler::new(&[1.0, 3.0]);
        let mut rng = StdRng::seed_from_u64(42);
        let count_1: usize = (0..10000).filter(|_| s.sample(&mut rng) == 1).count();
        // Should be ~7500
        assert!(count_1 > 7000 && count_1 < 8000);
    }
}
