//! Small deterministic PRNG for tests, benchmarks and (later) k-means seeding.
//! Not cryptographic.

#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in [lo, hi).
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    /// Uniform in [0, n). Slight modulo bias; fine for test data.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "below(0)");
        self.next_u64() % n
    }

    /// Standard normal sample (Box-Muller).
    pub fn gaussian(&mut self) -> f32 {
        // 1 - u keeps the log argument in (0, 1].
        let u1 = 1.0 - (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }

    /// `count` distinct indices from `0..n` in random order (partial
    /// Fisher-Yates). Panics if `count > n`.
    pub fn sample_indices(&mut self, n: usize, count: usize) -> Vec<usize> {
        assert!(count <= n, "sample_indices: count {count} > n {n}");
        let mut idx: Vec<usize> = (0..n).collect();
        for i in 0..count {
            let j = i + self.below((n - i) as u64) as usize;
            idx.swap(i, j);
        }
        idx.truncate(count);
        idx
    }
}
