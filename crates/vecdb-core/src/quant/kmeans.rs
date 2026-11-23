//! Lloyd's k-means with random initialization, used to train PQ codebooks
//! and the partitions of a partitioned SSD index build. The centroids depend
//! on the seed but not on the thread count.

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

    let row = |i: usize| &data[i * dim..(i + 1) * dim];
    for _ in 0..p.iters {
        // Assign each row to its nearest centroid.
        let labels: Vec<usize> = par::map_ranges(n, p.threads, |range| {
            let kern = kernels::best();
            let mut dists = vec![0.0f32; k];
            range
                .map(|i| {
                    (kern.l2_squared_batch)(row(i), &centroids, &mut dists);
                    argmin(&dists)
                })
                .collect::<Vec<_>>()
        })
        .concat();
        // Group the rows by cluster, in row order within each cluster.
        let mut start = vec![0usize; k + 1];
        for &j in &labels {
            start[j + 1] += 1;
        }
        for j in 0..k {
            start[j + 1] += start[j];
        }
        let mut members = vec![0usize; n];
        let mut fill = start.clone();
        for (i, &j) in labels.iter().enumerate() {
            members[fill[j]] = i;
            fill[j] += 1;
        }
        drop(labels);
        // One thread sums each cluster, in row order, so the centroids do
        // not depend on the thread count.
        let means = par::map_ranges(k, p.threads, |range| {
            range
                .map(|j| {
                    let rows = &members[start[j]..start[j + 1]];
                    if rows.is_empty() {
                        return None;
                    }
                    let mut sum = vec![0.0f64; dim];
                    for &i in rows {
                        for (s, &v) in sum.iter_mut().zip(row(i)) {
                            *s += v as f64;
                        }
                    }
                    let inv = 1.0 / rows.len() as f64;
                    Some(sum.into_iter().map(|s| (s * inv) as f32).collect::<Vec<f32>>())
                })
                .collect::<Vec<_>>()
        })
        .concat();
        let mut counts: Vec<usize> = (0..k).map(|j| start[j + 1] - start[j]).collect();
        for (j, mean) in means.iter().enumerate() {
            if let Some(m) = mean {
                centroids[j * dim..(j + 1) * dim].copy_from_slice(m);
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
    fn same_centroids_for_any_thread_count() {
        let mut rng = SplitMix64::new(4);
        let data: Vec<f32> = (0..5 * 3000).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let at = |threads| train(&data, 5, &KMeansParams { k: 16, iters: 8, seed: 5, threads }).unwrap();
        let one = at(1);
        assert_eq!(one, at(3));
        assert_eq!(one, at(8));
    }

    #[test]
    fn rejects_too_few_rows() {
        let p = KMeansParams { k: 10, iters: 1, seed: 0, threads: 1 };
        assert!(train(&[0.0; 9], 1, &p).is_err());
    }
}
