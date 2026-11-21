//! Metrics and vector helpers. Every metric returns a distance: smaller means
//! closer. The arithmetic lives in [`crate::kernels`], which picks the fastest
//! implementation for the CPU at runtime.

use crate::error::{Error, Result};
use crate::kernels;

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

/// Dot product using the process-wide kernel set ([`kernels::best`]).
/// Panics if the lengths differ. For hot loops, use the batch kernels directly.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    (kernels::best().dot)(a, b)
}

/// Squared Euclidean distance using the process-wide kernel set.
/// Panics if the lengths differ.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    (kernels::best().l2_squared)(a, b)
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
