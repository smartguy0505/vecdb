//! Distance kernels. Every metric returns a distance: smaller means closer.
//!
//! These are portable scalar loops shaped for LLVM's auto-vectorizer
//! (fixed-width chunks, independent accumulators). Measured on an i7-8700,
//! 128-dim L2 plus top-k with in-cache data: 22 ns per vector with the default
//! x86-64 target, 15.5 ns with `-C target-cpu=native`. Explicit SIMD with
//! runtime CPU detection (AVX2/FMA, AVX-512, NEON) is roadmap step 2; these
//! stay as the reference implementation it is tested against.

use crate::error::{Error, Result};

/// Number of independent accumulators. 16 f32 = two AVX2 registers or four
/// SSE/NEON registers, enough to hide add latency.
const LANES: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Metric {
    /// Squared Euclidean distance (same ordering as L2, no sqrt).
    L2,
    /// Negative inner product: `-dot(a, b)`. For maximum-inner-product search.
    InnerProduct,
    /// Cosine distance `1 - cos(a, b)`. Vectors are normalized when inserted
    /// and when queried, so the kernel is a dot product.
    Cosine,
}

impl Metric {
    pub fn name(self) -> &'static str {
        match self {
            Metric::L2 => "l2",
            Metric::InnerProduct => "ip",
            Metric::Cosine => "cosine",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "l2" | "euclidean" => Some(Metric::L2),
            "ip" | "dot" | "inner_product" => Some(Metric::InnerProduct),
            "cosine" | "cos" => Some(Metric::Cosine),
            _ => None,
        }
    }

    /// Stable on-disk code. Never renumber.
    pub fn to_code(self) -> u32 {
        match self {
            Metric::L2 => 0,
            Metric::InnerProduct => 1,
            Metric::Cosine => 2,
        }
    }

    pub fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Metric::L2),
            1 => Some(Metric::InnerProduct),
            2 => Some(Metric::Cosine),
            _ => None,
        }
    }

    /// Whether vectors must be normalized before storage and search.
    pub fn needs_normalization(self) -> bool {
        self == Metric::Cosine
    }

    /// Distance between two *prepared* vectors (normalized for cosine).
    #[inline]
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::L2 => l2_squared(a, b),
            Metric::InnerProduct => -dot(a, b),
            Metric::Cosine => 1.0 - dot(a, b),
        }
    }
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let (ac, ar) = a.as_chunks::<LANES>();
    let (bc, br) = b.as_chunks::<LANES>();
    let mut acc = [0.0f32; LANES];
    for (x, y) in ac.iter().zip(bc) {
        for i in 0..LANES {
            acc[i] += x[i] * y[i];
        }
    }
    let mut tail = 0.0f32;
    for (x, y) in ar.iter().zip(br) {
        tail += x * y;
    }
    reduce(&acc) + tail
}

#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let (ac, ar) = a.as_chunks::<LANES>();
    let (bc, br) = b.as_chunks::<LANES>();
    let mut acc = [0.0f32; LANES];
    for (x, y) in ac.iter().zip(bc) {
        for i in 0..LANES {
            let d = x[i] - y[i];
            acc[i] += d * d;
        }
    }
    let mut tail = 0.0f32;
    for (x, y) in ar.iter().zip(br) {
        let d = x - y;
        tail += d * d;
    }
    reduce(&acc) + tail
}

#[inline]
fn reduce(acc: &[f32; LANES]) -> f32 {
    // Pairwise sum: a little more accurate than a left fold.
    let mut v = *acc;
    let mut width = LANES;
    while width > 1 {
        width /= 2;
        for i in 0..width {
            v[i] += v[i + width];
        }
    }
    v[0]
}

/// Scales `v` to unit length in place. A zero vector stays zero.
/// The norm is computed in f64 so large f32 values do not overflow to infinity.
pub fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x = (*x as f64 * inv) as f32;
        }
    }
}

pub fn check_finite(v: &[f32]) -> Result<()> {
    if v.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(Error::NonFiniteValue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    fn naive_dot(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
    }

    fn naive_l2(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| {
                let d = x as f64 - y as f64;
                d * d
            })
            .sum()
    }

    fn close(got: f32, want: f64, scale: f64) -> bool {
        (got as f64 - want).abs() <= 1e-5 * scale.max(1.0)
    }

    #[test]
    fn kernels_match_naive_for_all_lengths() {
        let mut rng = SplitMix64::new(7);
        // Covers empty, shorter than one chunk, exact chunks and remainders.
        for len in 0..=200 {
            let a: Vec<f32> = (0..len).map(|_| rng.uniform(-2.0, 2.0)).collect();
            let b: Vec<f32> = (0..len).map(|_| rng.uniform(-2.0, 2.0)).collect();
            let scale = len as f64 * 4.0;
            assert!(close(dot(&a, &b), naive_dot(&a, &b), scale), "dot len {len}");
            assert!(close(l2_squared(&a, &b), naive_l2(&a, &b), scale), "l2 len {len}");
        }
    }

    #[test]
    fn metric_distances() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        assert_eq!(Metric::L2.distance(&a, &b), 2.0);
        assert_eq!(Metric::InnerProduct.distance(&a, &a), -1.0);
        assert_eq!(Metric::Cosine.distance(&a, &b), 1.0);
        assert_eq!(Metric::Cosine.distance(&a, &a), 0.0);
    }

    #[test]
    fn normalize_handles_zero_and_huge() {
        let mut z = [0.0f32; 5];
        normalize(&mut z);
        assert_eq!(z, [0.0; 5]);

        let mut big = [3.0e38f32, 3.0e38];
        normalize(&mut big);
        let n = (big[0] as f64).hypot(big[1] as f64);
        assert!((n - 1.0).abs() < 1e-6, "norm {n}");
    }

    #[test]
    fn metric_codes_round_trip() {
        for m in [Metric::L2, Metric::InnerProduct, Metric::Cosine] {
            assert_eq!(Metric::from_code(m.to_code()), Some(m));
            assert_eq!(Metric::parse(m.name()), Some(m));
        }
        assert_eq!(Metric::from_code(99), None);
    }

    #[test]
    fn check_finite_rejects_nan_and_inf() {
        assert!(check_finite(&[1.0, 2.0]).is_ok());
        assert!(check_finite(&[1.0, f32::NAN]).is_err());
        assert!(check_finite(&[f32::NEG_INFINITY]).is_err());
    }
}
