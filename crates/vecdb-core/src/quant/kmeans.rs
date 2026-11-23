//! Lloyd's k-means with random initialization, used to train PQ codebooks
//! (and later IVF partitions).

use crate::error::{Error, Result};
use crate::kernels;
use crate::par;
use crate::rng::SplitMix64;

#[derive(Clone, Debug)]
pub struct KMeansParams {
    pub k: usize,
    pub iters: usize,
    pub seed: u64,
    /// 0 = one thread per CPU.
    pub threads: usize,
}

/// Relative size of the nudge that separates a split centroid from its source.
const SPLIT_EPS: f32 = 1.0 / 1024.0;

/// Trains `k` centroids on `data` (row-major, `dim` per row) and returns them
/// row-major (`k * dim`). Needs at least `k` rows. Empty clusters are
/// refilled by splitting the largest cluster, so every centroid stays used.
pub fn train(data: &[f32], dim: usize, p: &KMeansParams) -> Result<Vec<f32>> {
    if dim == 0 || data.len() % dim != 0 {
        return Err(Error::InvalidArgument("k-means: data is not whole rows".into()));
    }
    let n = data.len() / dim;
    if p.k == 0 || n < p.k {
        return Err(Error::InvalidArgument(format!("k-means: need at least k = {} rows, have {n}", p.k)));
    }
    let k = p.k;
    let mut rng = SplitMix64::new(p.seed);
    let mut centroids = Vec::with_capacity(k * dim);
    for i in rng.sample_indices(n, k) {
        centroids.extend_from_slice(&data[i * dim..(i + 1) * dim]);
    }

    for _ in 0..p.iters {
        // Assign each row to its nearest centroid; accumulate sums per thread.
        let parts = par::map_ranges(n, p.threads, |range| {
            let kern = kernels::best();
            let mut sums = vec![0.0f64; k * dim];
            let mut counts = vec![0usize; k];
            let mut dists = vec![0.0f32; k];
            for i in range {
                let x = &data[i * dim..(i + 1) * dim];
                (kern.l2_squared_batch)(x, &centroids, &mut dists);
                let j = argmin(&dists);
                counts[j] += 1;
                for (s, &v) in sums[j * dim..(j + 1) * dim].iter_mut().zip(x) {
                    *s += v as f64;
                }
            }
            (sums, counts)
        });
        let mut sums = vec![0.0f64; k * dim];
        let mut counts = vec![0usize; k];
        for (s, c) in parts {
            for (a, b) in sums.iter_mut().zip(&s) {
                *a += b;
            }
            for (a, b) in counts.iter_mut().zip(&c) {
                *a += b;
            }
        }
        for j in 0..k {
            if counts[j] > 0 {
                let inv = 1.0 / counts[j] as f64;
                for d in 0..dim {
                    centroids[j * dim + d] = (sums[j * dim + d] * inv) as f32;
                }
            }
        }
        split_empty_clusters(&mut centroids, &mut counts, dim);
    }
    Ok(centroids)
}

/// Index of the smallest value; ties go to the lowest index.
#[inline]
pub fn argmin(v: &[f32]) -> usize {
    let mut best = 0;
    let mut best_d = f32::INFINITY;
    for (j, &d) in v.iter().enumerate() {
        if d < best_d {
            best_d = d;
            best = j;
        }
    }
    best
}

fn split_empty_clusters(centroids: &mut [f32], counts: &mut [usize], dim: usize) {
    for j in 0..counts.len() {
        if counts[j] > 0 {
            continue;
        }
        let Some(big) = (0..counts.len()).max_by_key(|&c| counts[c]).filter(|&c| counts[c] > 1) else {
            return; // nothing left to split
        };
        for d in 0..dim {
            let v = centroids[big * dim + d];
            let delta = SPLIT_EPS * v.abs().max(1e-6);
            let sign = if d % 2 == 0 { 1.0 } else { -1.0 };
            centroids[j * dim + d] = v + sign * delta;
            centroids[big * dim + d] = v - sign * delta;
        }
        counts[j] = counts[big] / 2;
        counts[big] -= counts[j];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_well_separated_clusters() {
        let mut rng = SplitMix64::new(1);
        let dim = 4;
        let centers = [[0.0f32, 0.0, 0.0, 0.0], [10.0, 10.0, 0.0, 0.0], [0.0, 10.0, 10.0, 10.0]];
        let mut data = Vec::new();
        for i in 0..900 {
            let c = centers[i % 3];
            data.extend(c.iter().map(|&x| x + 0.1 * rng.gaussian()));
        }
        let p = KMeansParams { k: 3, iters: 20, seed: 7, threads: 2 };
        let got = train(&data, dim, &p).unwrap();
        for c in centers {
            let nearest = got
                .chunks_exact(dim)
                .map(|g| g.iter().zip(&c).map(|(a, b)| (a - b).powi(2)).sum::<f32>())
                .fold(f32::INFINITY, f32::min);
            assert!(nearest < 0.01, "center {c:?} not found: {got:?}");
        }
    }

    #[test]
    fn duplicate_points_do_not_produce_nan() {
        let data = vec![1.5f32; 2 * 300];
        let p = KMeansParams { k: 16, iters: 5, seed: 3, threads: 3 };
        let got = train(&data, 2, &p).unwrap();
        assert_eq!(got.len(), 32);
        assert!(got.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn deterministic_for_same_seed_and_threads() {
        let mut rng = SplitMix64::new(2);
        let data: Vec<f32> = (0..3 * 500).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let p = KMeansParams { k: 8, iters: 10, seed: 11, threads: 2 };
        assert_eq!(train(&data, 3, &p).unwrap(), train(&data, 3, &p).unwrap());
    }

    #[test]
    fn rejects_too_few_rows() {
        let p = KMeansParams { k: 10, iters: 1, seed: 0, threads: 1 };
        assert!(train(&[0.0; 9], 1, &p).is_err());
    }
}
