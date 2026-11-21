//! f32 distance kernels with runtime CPU dispatch.
//!
//! A [`Kernels`] value is one implementation of the distance primitives. All
//! sets compute the same functions; they differ only in speed and in float
//! summation order, so results can differ in the last bits between sets.
//!
//! - `scalar`: portable loops shaped for auto-vectorization. It is the
//!   reference implementation. On aarch64 it is expected to auto-vectorize to
//!   NEON; that has not been built or tested yet.
//! - `avx2-fma`: x86_64 with AVX2 and FMA, detected at runtime.
//!
//! AVX-512 and hand-written NEON kernels are planned. They are not here yet
//! because they cannot be tested on the current development machine.
//!
//! Hot loops should use the batch functions: the row loop then runs inside the
//! feature-enabled code, and the kernel is inlined instead of called per row.

use std::sync::OnceLock;

mod scalar;
#[cfg(target_arch = "x86_64")]
mod x86;

/// Distance primitives over `f32` slices.
///
/// Single-vector functions panic if the two slices differ in length. Batch
/// functions compute `out[i] = op(query, rows[i*d..(i+1)*d])` with
/// `d = query.len()`, and panic unless `d > 0` and
/// `rows.len() == out.len() * d`.
pub struct Kernels {
    pub name: &'static str,
    pub dot: fn(&[f32], &[f32]) -> f32,
    pub l2_squared: fn(&[f32], &[f32]) -> f32,
    pub dot_batch: fn(&[f32], &[f32], &mut [f32]),
    pub l2_squared_batch: fn(&[f32], &[f32], &mut [f32]),
}

impl std::fmt::Debug for Kernels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernels").field("name", &self.name).finish()
    }
}

pub static SCALAR: Kernels = Kernels {
    name: "scalar",
    dot: scalar::dot,
    l2_squared: scalar::l2_squared,
    dot_batch: scalar::dot_batch,
    l2_squared_batch: scalar::l2_squared_batch,
};

/// Every kernel set this CPU can run, slowest first.
pub fn available() -> Vec<&'static Kernels> {
    #[allow(unused_mut)]
    let mut sets = vec![&SCALAR];
    #[cfg(target_arch = "x86_64")]
    if let Some(k) = x86::avx2_fma() {
        sets.push(k);
    }
    sets
}

/// Name of the environment variable that forces a kernel set (for A/B tests
/// and debugging), e.g. `VECDB_KERNELS=scalar`. Unknown or unsupported names
/// fall back to the fastest available set.
pub const ENV_OVERRIDE: &str = "VECDB_KERNELS";

/// The kernel set used by default. Chosen once per process: the fastest
/// available, unless `VECDB_KERNELS` names another available set.
pub fn best() -> &'static Kernels {
    static CHOSEN: OnceLock<&'static Kernels> = OnceLock::new();
    CHOSEN.get_or_init(|| {
        let sets = available();
        let forced = std::env::var(ENV_OVERRIDE).ok();
        forced
            .and_then(|name| sets.iter().copied().find(|k| k.name == name))
            .unwrap_or_else(|| sets[sets.len() - 1])
    })
}

#[inline]
fn check_batch(query: &[f32], rows: &[f32], out: &[f32]) {
    assert!(!query.is_empty(), "batch kernel: empty query");
    assert_eq!(
        Some(rows.len()),
        out.len().checked_mul(query.len()),
        "batch kernel: rows do not match out.len() x query.len()"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    fn random(rng: &mut SplitMix64, n: usize) -> Vec<f32> {
        (0..n).map(|_| rng.uniform(-2.0, 2.0)).collect()
    }

    /// f64 reference value and the sum of |terms|, which bounds rounding error.
    fn reference(a: &[f32], b: &[f32], l2: bool) -> (f64, f64) {
        let terms = a.iter().zip(b).map(|(&x, &y)| {
            let (x, y) = (x as f64, y as f64);
            if l2 { (x - y) * (x - y) } else { x * y }
        });
        terms.fold((0.0, 0.0), |(s, abs), t| (s + t, abs + t.abs()))
    }

    fn lengths() -> Vec<usize> {
        // Every small length (all remainders mod 8 and 32), then large ones
        // around multiples of 8 and 32 up to 8192.
        let mut v: Vec<usize> = (0..=130).collect();
        v.extend([255, 256, 257, 767, 768, 769, 1023, 1024, 1025, 1536, 4096, 8191, 8192]);
        v
    }

    #[test]
    fn every_set_matches_f64_reference() {
        let mut rng = SplitMix64::new(11);
        for k in available() {
            for len in lengths() {
                let a = random(&mut rng, len);
                let b = random(&mut rng, len);
                for (l2, got) in [(false, (k.dot)(&a, &b)), (true, (k.l2_squared)(&a, &b))] {
                    let (want, abs) = reference(&a, &b, l2);
                    let tol = 1e-5 * abs + 1e-6;
                    assert!(
                        (got as f64 - want).abs() <= tol,
                        "{} {} len {len}: got {got}, want {want}",
                        k.name,
                        if l2 { "l2" } else { "dot" }
                    );
                }
            }
        }
    }

    #[test]
    fn batch_is_bit_identical_to_single() {
        let mut rng = SplitMix64::new(12);
        for k in available() {
            for dim in [1usize, 7, 8, 31, 32, 33, 100, 128, 1024, 8192] {
                for rows in [0usize, 1, 5] {
                    let q = random(&mut rng, dim);
                    let data = random(&mut rng, rows * dim);
                    let mut out = vec![f32::NAN; rows];
                    (k.dot_batch)(&q, &data, &mut out);
                    for (i, row) in data.chunks_exact(dim).enumerate() {
                        assert_eq!(out[i].to_bits(), (k.dot)(&q, row).to_bits(), "{} dot dim {dim}", k.name);
                    }
                    (k.l2_squared_batch)(&q, &data, &mut out);
                    for (i, row) in data.chunks_exact(dim).enumerate() {
                        assert_eq!(
                            out[i].to_bits(),
                            (k.l2_squared)(&q, row).to_bits(),
                            "{} l2 dim {dim}",
                            k.name
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn length_mismatch_panics_in_every_set() {
        for k in available() {
            let r = std::panic::catch_unwind(|| (k.dot)(&[1.0; 9], &[1.0; 8]));
            assert!(r.is_err(), "{} dot accepted mismatched lengths", k.name);
            let r = std::panic::catch_unwind(|| (k.l2_squared)(&[1.0; 8], &[1.0; 40]));
            assert!(r.is_err(), "{} l2 accepted mismatched lengths", k.name);
            let r = std::panic::catch_unwind(|| {
                let mut out = [0.0; 2];
                (k.dot_batch)(&[1.0; 8], &[1.0; 15], &mut out)
            });
            assert!(r.is_err(), "{} batch accepted a short block", k.name);
        }
    }

    #[test]
    fn best_is_available() {
        let names: Vec<&str> = available().iter().map(|k| k.name).collect();
        assert!(names.contains(&best().name));
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            assert!(names.contains(&"avx2-fma"));
        }
    }
}
