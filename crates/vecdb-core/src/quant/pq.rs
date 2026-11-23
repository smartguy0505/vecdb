//! Product quantization (PQ).
//!
//! A vector is split into `m` subvectors; each subvector is replaced by the
//! index of its nearest centroid among 256 trained for that subspace. A
//! vector therefore costs `m` bytes, whatever its dimension. `m` is the
//! per-collection size/recall knob: more bytes, better recall.
//!
//! Search uses asymmetric distance computation (ADC): the query stays exact,
//! a table of query-to-centroid distances is built once per query (`m x 256`
//! floats), and each code's distance is `m` table lookups.
//!
//! Dimensions need not divide evenly by `m`: the first `dim % m` subspaces
//! get one extra dimension.

use crate::distance::Metric;
use crate::error::{Error, Result};
use crate::kernels;
use crate::par;
use crate::quant::kmeans::{self, KMeansParams, argmin};
use crate::rng::SplitMix64;
use crate::store::check_dim;

/// Centroids per subspace (8-bit codes).
pub const KSUB: usize = 256;

#[derive(Clone, Debug)]
pub struct PqParams {
    /// Subspaces = bytes per vector. 1 <= m <= dim.
    pub m: usize,
    /// k-means iterations per subspace.
    pub iters: usize,
    /// Training uses at most this many rows (a random sample).
    pub max_train: usize,
    pub seed: u64,
    /// 0 = one thread per CPU.
    pub threads: usize,
}

impl PqParams {
    pub fn new(m: usize) -> Self {
        Self { m, iters: 25, max_train: 65_536, seed: 0x5eed, threads: 0 }
    }
}

#[derive(Clone, Debug)]
pub struct ProductQuantizer {
    dim: usize,
    /// Subspace `i` covers dimensions `offsets[i]..offsets[i + 1]`.
    offsets: Vec<usize>,
    /// Subspace `i`'s 256 centroids, row-major, starting at `KSUB * offsets[i]`.
    codebooks: Vec<f32>,
}

impl ProductQuantizer {
    /// Trains on `data` (row-major, `dim` per row; at least 256 rows).
    /// Vectors must already be prepared for the metric (normalized for cosine).
    pub fn train(dim: usize, data: &[f32], p: &PqParams) -> Result<Self> {
        check_dim(dim)?;
        if p.m == 0 || p.m > dim {
            return Err(Error::InvalidArgument(format!("PQ m = {} must be in 1..={dim}", p.m)));
        }
        if data.len() % dim != 0 {
            return Err(Error::InvalidArgument("PQ training data is not whole rows".into()));
        }
        let n = data.len() / dim;
        if n < KSUB {
            return Err(Error::InvalidArgument(format!("PQ training needs at least {KSUB} vectors, have {n}")));
        }
        let mut rng = SplitMix64::new(p.seed);
        let rows: Vec<usize> = if n > p.max_train.max(KSUB) {
            let mut r = rng.sample_indices(n, p.max_train.max(KSUB));
            r.sort_unstable(); // sequential reads while gathering
            r
        } else {
            (0..n).collect()
        };

        let offsets = subspace_offsets(dim, p.m);
        let mut codebooks = vec![0.0f32; KSUB * dim];
        for i in 0..p.m {
            let (lo, hi) = (offsets[i], offsets[i + 1]);
            let dsub = hi - lo;
            let mut sub = Vec::with_capacity(rows.len() * dsub);
            for &r in &rows {
                sub.extend_from_slice(&data[r * dim + lo..r * dim + hi]);
            }
            let km = KMeansParams { k: KSUB, iters: p.iters, seed: p.seed.wrapping_add(i as u64 + 1), threads: p.threads };
            let centroids = kmeans::train(&sub, dsub, &km)?;
            codebooks[KSUB * lo..KSUB * hi].copy_from_slice(&centroids);
        }
        Ok(Self { dim, offsets, codebooks })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Bytes per encoded vector (= number of subspaces).
    pub fn code_size(&self) -> usize {
        self.offsets.len() - 1
    }

    #[inline]
    fn codebook(&self, i: usize) -> &[f32] {
        &self.codebooks[KSUB * self.offsets[i]..KSUB * self.offsets[i + 1]]
    }

    /// Encodes one vector into `code` (`code_size()` bytes).
    pub fn encode(&self, v: &[f32], code: &mut [u8]) {
        assert_eq!(v.len(), self.dim, "PQ encode: wrong dimension");
        assert_eq!(code.len(), self.code_size(), "PQ encode: wrong code length");
        let kern = kernels::best();
        let mut dists = [0.0f32; KSUB];
        for (i, c) in code.iter_mut().enumerate() {
            let sub = &v[self.offsets[i]..self.offsets[i + 1]];
            (kern.l2_squared_batch)(sub, self.codebook(i), &mut dists);
            *c = argmin(&dists) as u8;
        }
    }

    /// Encodes every row of `data`; returns `rows * code_size()` bytes.
    pub fn encode_all(&self, data: &[f32], threads: usize) -> Vec<u8> {
        assert_eq!(data.len() % self.dim, 0, "PQ encode_all: data is not whole rows");
        let n = data.len() / self.dim;
        let m = self.code_size();
        let parts = par::map_ranges(n, threads, |range| {
            let mut out = vec![0u8; range.len() * m];
            for (j, row) in range.enumerate() {
                self.encode(&data[row * self.dim..(row + 1) * self.dim], &mut out[j * m..(j + 1) * m]);
            }
            out
        });
        parts.concat()
    }

    /// Reconstructs the approximate vector for `code`.
    pub fn decode(&self, code: &[u8], out: &mut [f32]) {
        assert_eq!(code.len(), self.code_size(), "PQ decode: wrong code length");
        assert_eq!(out.len(), self.dim, "PQ decode: wrong dimension");
        for (i, &c) in code.iter().enumerate() {
            let (lo, hi) = (self.offsets[i], self.offsets[i + 1]);
            let dsub = hi - lo;
            let cb = self.codebook(i);
            out[lo..hi].copy_from_slice(&cb[c as usize * dsub..(c as usize + 1) * dsub]);
        }
    }

    /// Builds the ADC lookup table for `query`, which must be prepared for the
    /// metric (normalized for cosine).
    pub fn distance_table(&self, metric: Metric, query: &[f32]) -> DistanceTable {
        assert_eq!(query.len(), self.dim, "PQ distance_table: wrong dimension");
        let kern = kernels::best();
        let m = self.code_size();
        let mut tables = vec![[0.0f32; KSUB]; m];
        for (i, t) in tables.iter_mut().enumerate() {
            let sub = &query[self.offsets[i]..self.offsets[i + 1]];
            match metric {
                Metric::L2 => (kern.l2_squared_batch)(sub, self.codebook(i), t),
                Metric::InnerProduct | Metric::Cosine => {
                    (kern.dot_batch)(sub, self.codebook(i), t);
                    for x in t.iter_mut() {
                        *x = -*x;
                    }
                }
            }
        }
        let bias = if metric == Metric::Cosine { 1.0 } else { 0.0 };
        DistanceTable { tables, bias }
    }
}

/// Splits `dim` into `m` near-equal contiguous ranges; returns `m + 1` offsets.
fn subspace_offsets(dim: usize, m: usize) -> Vec<usize> {
    let (base, extra) = (dim / m, dim % m);
    let mut offsets = Vec::with_capacity(m + 1);
    offsets.push(0);
    for i in 0..m {
        offsets.push(offsets[i] + base + usize::from(i < extra));
    }
    offsets
}

/// Per-query ADC table: distance(code) = bias + sum of one entry per subspace.
#[derive(Clone, Debug)]
pub struct DistanceTable {
    tables: Vec<[f32; KSUB]>,
    bias: f32,
}

impl DistanceTable {
    /// Approximate distance from the query to the vector encoded as `code`.
    #[inline]
    pub fn distance(&self, code: &[u8]) -> f32 {
        debug_assert_eq!(code.len(), self.tables.len());
        // Four accumulators break the add dependency chain.
        let (tc, tr) = self.tables.as_chunks::<4>();
        let (cc, cr) = code.as_chunks::<4>();
        let mut acc = [0.0f32; 4];
        for (t4, c4) in tc.iter().zip(cc) {
            for j in 0..4 {
                acc[j] += t4[j][c4[j] as usize];
            }
        }
        let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        for (t, &c) in tr.iter().zip(cr) {
            sum += t[c as usize];
        }
        self.bias + sum
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Data with low intrinsic dimension (like real embeddings): x = A z + noise.
    pub(crate) fn structured(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        let r = 8;
        let a: Vec<f32> = (0..dim * r).map(|_| rng.gaussian()).collect();
        let mut out = Vec::with_capacity(n * dim);
        for _ in 0..n {
            let z: Vec<f32> = (0..r).map(|_| rng.gaussian()).collect();
            for d in 0..dim {
                let v: f32 = (0..r).map(|j| a[d * r + j] * z[j]).sum();
                out.push(v / (r as f32).sqrt() + 0.01 * rng.gaussian());
            }
        }
        out
    }

    fn params(m: usize) -> PqParams {
        PqParams { m, iters: 10, max_train: 4096, seed: 1, threads: 2 }
    }

    #[test]
    fn offsets_split_evenly() {
        assert_eq!(subspace_offsets(10, 3), vec![0, 4, 7, 10]);
        assert_eq!(subspace_offsets(8, 8), vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(subspace_offsets(128, 16).len(), 17);
    }

    #[test]
    fn more_bytes_means_less_reconstruction_error() {
        let dim = 32;
        let data = structured(2000, dim, 3);
        let energy: f64 = data.iter().map(|&x| (x as f64).powi(2)).sum();
        let mut last = f64::INFINITY;
        for m in [2, 4, 8, 16] {
            let pq = ProductQuantizer::train(dim, &data, &params(m)).unwrap();
            assert_eq!(pq.code_size(), m);
            let codes = pq.encode_all(&data, 2);
            let mut out = vec![0.0f32; dim];
            let mut err = 0.0f64;
            for (row, code) in codes.chunks_exact(m).enumerate() {
                pq.decode(code, &mut out);
                err += out.iter().zip(&data[row * dim..]).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>();
            }
            let rel = err / energy;
            assert!(rel < last, "m {m}: relative error {rel} not below {last}");
            last = rel;
        }
        assert!(last < 0.05, "m 16 relative error {last}");
    }

    #[test]
    fn adc_equals_distance_to_decoded_vector() {
        let dim = 10; // uneven split with m = 3
        let data = structured(600, dim, 4);
        let pq = ProductQuantizer::train(dim, &data, &params(3)).unwrap();
        let q: Vec<f32> = structured(1, dim, 99);
        let mut decoded = vec![0.0f32; dim];
        let mut code = vec![0u8; 3];
        for metric in [Metric::L2, Metric::InnerProduct, Metric::Cosine] {
            let table = pq.distance_table(metric, &q);
            for row in data.chunks_exact(dim).take(50) {
                pq.encode(row, &mut code);
                pq.decode(&code, &mut decoded);
                let want = metric.distance(&q, &decoded);
                let got = table.distance(&code);
                assert!((got - want).abs() <= 1e-4 * (1.0 + want.abs()), "{metric:?}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn encode_all_matches_single_encode_across_threads() {
        let dim = 16;
        let data = structured(700, dim, 5);
        let pq = ProductQuantizer::train(dim, &data, &params(4)).unwrap();
        let one = pq.encode_all(&data, 1);
        assert_eq!(one, pq.encode_all(&data, 3));
        let mut code = [0u8; 4];
        pq.encode(&data[5 * dim..6 * dim], &mut code);
        assert_eq!(&one[5 * 4..6 * 4], &code);
    }

    #[test]
    fn rejects_bad_parameters() {
        let data = structured(300, 8, 6);
        assert!(ProductQuantizer::train(8, &data, &params(0)).is_err());
        assert!(ProductQuantizer::train(8, &data, &params(9)).is_err());
        assert!(ProductQuantizer::train(8, &data[..8 * 100], &params(2)).is_err()); // < 256 rows
        assert!(ProductQuantizer::train(8, &data, &params(8)).is_ok()); // m = dim
    }
}
