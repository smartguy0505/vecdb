//! Exact (brute-force) index. It is the ground truth every approximate index
//! is measured against, and the search path for small or still-growing segments.

use crate::distance::{Metric, check_finite, dot, l2_squared, normalize};
use crate::error::{Error, Result};
use crate::store::{InMemoryStore, VectorStore};
use crate::topk::{SearchResult, TopK};

#[derive(Debug, Clone)]
pub struct FlatIndex<S = InMemoryStore> {
    metric: Metric,
    store: S,
}

impl FlatIndex<InMemoryStore> {
    pub fn new(dim: usize, metric: Metric) -> Result<Self> {
        Self::with_capacity(dim, metric, 0)
    }

    pub fn with_capacity(dim: usize, metric: Metric, rows: usize) -> Result<Self> {
        Ok(Self { metric, store: InMemoryStore::with_capacity(dim, rows)? })
    }

    pub fn add(&mut self, id: u64, vector: &[f32]) -> Result<()> {
        if self.metric.needs_normalization() {
            let mut v = vector.to_vec();
            normalize(&mut v);
            self.store.push(id, &v)?;
        } else {
            self.store.push(id, vector)?;
        }
        Ok(())
    }
}

impl<S: VectorStore> FlatIndex<S> {
    /// Wraps an existing store. Its vectors must already be prepared for
    /// `metric` (normalized for cosine), as they are in segments we wrote.
    pub fn from_store(store: S, metric: Metric) -> Self {
        Self { metric, store }
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn dim(&self) -> usize {
        self.store.dim()
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    pub fn store(&self) -> &S {
        &self.store
    }

    /// Returns the `k` nearest vectors, closest first. Returns fewer than `k`
    /// when the index holds fewer vectors. Ties break toward the smaller id.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dim() {
            return Err(Error::DimensionMismatch { expected: self.dim(), got: query.len() });
        }
        check_finite(query)?;
        if k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }
        let mut top = TopK::new(k.min(self.len()));
        // Match on the metric once, outside the loop, so each arm is a
        // monomorphized scan with the kernel inlined.
        match self.metric {
            Metric::L2 => self.scan(query, &mut top, l2_squared),
            Metric::InnerProduct => self.scan(query, &mut top, |a, b| -dot(a, b)),
            Metric::Cosine => {
                let mut q = query.to_vec();
                normalize(&mut q);
                self.scan(&q, &mut top, |a, b| 1.0 - dot(a, b));
            }
        }
        Ok(top.into_sorted())
    }

    #[inline]
    fn scan(&self, query: &[f32], top: &mut TopK, dist: impl Fn(&[f32], &[f32]) -> f32) {
        for row in 0..self.store.len() {
            top.push(self.store.id(row), dist(query, self.store.vector(row)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    /// Independent reference distances in f64, indexed by id (ids are 0..n).
    fn reference_distances(metric: Metric, data: &[Vec<f32>], q: &[f32]) -> Vec<f64> {
        let unit = |v: &[f32]| -> Vec<f64> {
            let n = v.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
            v.iter().map(|&x| if n > 0.0 { x as f64 / n } else { 0.0 }).collect()
        };
        let qu = unit(q);
        data.iter()
            .map(|v| match metric {
                Metric::L2 => v.iter().zip(q).map(|(&a, &b)| (a as f64 - b as f64).powi(2)).sum(),
                Metric::InnerProduct => -v.iter().zip(q).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>(),
                Metric::Cosine => 1.0 - unit(v).iter().zip(&qu).map(|(a, b)| a * b).sum::<f64>(),
            })
            .collect()
    }

    #[test]
    fn matches_reference_on_random_data() {
        let mut rng = SplitMix64::new(3);
        let k = 10;
        for metric in [Metric::L2, Metric::InnerProduct, Metric::Cosine] {
            for &dim in &[1usize, 7, 16, 32, 33, 128, 1024, 8192] {
                let n = if dim > 1024 { 100 } else { 300 };
                // f32 rounding error grows with dimension; scale the tolerance.
                let tol = 1e-4 * (dim as f64 / 128.0).max(1.0);
                let data: Vec<Vec<f32>> =
                    (0..n).map(|_| (0..dim).map(|_| rng.uniform(-1.0, 1.0)).collect()).collect();
                let mut index = FlatIndex::new(dim, metric).unwrap();
                for (i, v) in data.iter().enumerate() {
                    index.add(i as u64, v).unwrap();
                }
                for _ in 0..20 {
                    let q: Vec<f32> = (0..dim).map(|_| rng.uniform(-1.0, 1.0)).collect();
                    let dist = reference_distances(metric, &data, &q);
                    let mut want = dist.clone();
                    want.sort_by(f64::total_cmp);
                    let got = index.search(&q, k).unwrap();
                    assert_eq!(got.len(), k);
                    let mut ids: Vec<u64> = got.iter().map(|r| r.id).collect();
                    ids.sort_unstable();
                    ids.dedup();
                    assert_eq!(ids.len(), k, "duplicate ids in result");
                    // Compare the i-th result's true distance with the i-th best
                    // true distance; tolerant of f32 rounding at near-ties.
                    for (i, r) in got.iter().enumerate() {
                        let d = dist[r.id as usize];
                        assert!(
                            (d - want[i]).abs() <= tol * (1.0 + want[i].abs()),
                            "{metric:?} dim {dim} rank {i}: got {d}, want {}",
                            want[i]
                        );
                        assert!(((r.distance as f64) - d).abs() <= tol * (1.0 + d.abs()));
                    }
                }
            }
        }
    }

    #[test]
    fn edge_cases() {
        let mut index = FlatIndex::new(2, Metric::L2).unwrap();
        assert!(index.search(&[0.0, 0.0], 5).unwrap().is_empty());
        index.add(10, &[0.0, 0.0]).unwrap();
        index.add(11, &[1.0, 0.0]).unwrap();
        assert!(index.search(&[0.0, 0.0], 0).unwrap().is_empty());
        assert_eq!(index.search(&[0.0, 0.0], 100).unwrap().len(), 2);
        assert!(matches!(index.search(&[0.0], 1), Err(Error::DimensionMismatch { .. })));
        assert!(matches!(index.search(&[f32::NAN, 0.0], 1), Err(Error::NonFiniteValue)));
    }

    #[test]
    fn exact_ties_break_on_id() {
        let mut index = FlatIndex::new(1, Metric::L2).unwrap();
        for id in [5u64, 3, 9, 1] {
            index.add(id, &[2.0]).unwrap();
        }
        let ids: Vec<u64> = index.search(&[0.0], 3).unwrap().iter().map(|r| r.id).collect();
        assert_eq!(ids, vec![1, 3, 5]);
    }

    #[test]
    fn cosine_ignores_scale_and_handles_zero_vectors() {
        let mut index = FlatIndex::new(2, Metric::Cosine).unwrap();
        index.add(1, &[10.0, 0.0]).unwrap();
        index.add(2, &[0.0, 0.5]).unwrap();
        index.add(3, &[0.0, 0.0]).unwrap();
        let r = index.search(&[0.001, 0.0], 3).unwrap();
        assert_eq!(r[0].id, 1);
        assert!(r[0].distance.abs() < 1e-6);
        // Zero vectors (stored or queried) sit at cosine distance 1.
        let z = r.iter().find(|r| r.id == 3).unwrap();
        assert_eq!(z.distance, 1.0);
        let all_one = index.search(&[0.0, 0.0], 3).unwrap();
        assert!(all_one.iter().all(|r| r.distance == 1.0));
    }
}
