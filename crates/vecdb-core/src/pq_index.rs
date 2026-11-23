//! PQ index: compact codes scanned with ADC, then optional exact re-ranking.
//!
//! Search scans every code (`m` bytes per vector) with the query's ADC table,
//! keeps the best `rerank` candidates, and recomputes their distances from
//! full vectors (stored as f32, f16, bf16 or sq8). Re-ranking is what turns
//! PQ's approximate order into high recall.
//!
//! The code scan is linear in the number of vectors. That is enough to
//! measure quantization quality; the graph index (roadmap step 4) uses the
//! same codes to visit only a small part of the collection.

use std::borrow::Cow;
use std::collections::HashSet;

use crate::distance::{Metric, check_finite, normalize};
use crate::error::{Error, Result};
use crate::kernels;
use crate::quant::{Codec, EncodedVectors, PqParams, ProductQuantizer};
use crate::store::check_dim;
use crate::topk::{SearchResult, TopK};

#[derive(Debug, Clone)]
pub struct PqIndex {
    metric: Metric,
    pq: ProductQuantizer,
    ids: Vec<u64>,
    codes: Vec<u8>,
    full: Option<EncodedVectors>,
}

/// Validates rows and returns them prepared for `metric` (normalized for cosine).
fn prepare<'a>(metric: Metric, dim: usize, rows: usize, data: &'a [f32]) -> Result<Cow<'a, [f32]>> {
    if Some(data.len()) != rows.checked_mul(dim) {
        return Err(Error::InvalidArgument(format!(
            "{} values do not form {rows} rows of dimension {dim}",
            data.len()
        )));
    }
    check_finite(data)?;
    if metric.needs_normalization() {
        let mut v = data.to_vec();
        for row in v.chunks_exact_mut(dim) {
            normalize(row);
        }
        Ok(Cow::Owned(v))
    } else {
        Ok(Cow::Borrowed(data))
    }
}

impl PqIndex {
    /// Trains PQ on a sample of `data`, encodes every row, and (if `full` is
    /// given) keeps a copy of the full vectors in that codec for re-ranking.
    pub fn build(
        metric: Metric,
        dim: usize,
        ids: &[u64],
        data: &[f32],
        params: &PqParams,
        full: Option<Codec>,
    ) -> Result<Self> {
        check_dim(dim)?;
        let mut seen = HashSet::with_capacity(ids.len());
        for &id in ids {
            if !seen.insert(id) {
                return Err(Error::DuplicateId(id));
            }
        }
        let prepared = prepare(metric, dim, ids.len(), data)?;
        let pq = ProductQuantizer::train(dim, &prepared, params)?;
        let codes = pq.encode_all(&prepared, params.threads);
        let full = full.map(|c| EncodedVectors::encode(c, dim, &prepared)).transpose()?;
        Ok(Self { metric, pq, ids: ids.to_vec(), codes, full })
    }

    /// Replaces the re-ranking copy (e.g. to compare codecs on one trained
    /// index). `data` must be the same rows, in the same order, as at build.
    pub fn set_full_vectors(&mut self, data: &[f32], codec: Option<Codec>) -> Result<()> {
        let prepared = prepare(self.metric, self.dim(), self.len(), data)?;
        self.full = codec.map(|c| EncodedVectors::encode(c, self.dim(), &prepared)).transpose()?;
        Ok(())
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn dim(&self) -> usize {
        self.pq.dim()
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn quantizer(&self) -> &ProductQuantizer {
        &self.pq
    }

    /// PQ bytes per vector (what must stay in RAM).
    pub fn code_size(&self) -> usize {
        self.pq.code_size()
    }

    /// Codec and bytes per vector of the re-ranking copy, if any.
    pub fn full_vectors(&self) -> Option<(Codec, usize)> {
        self.full.as_ref().map(|f| (f.codec(), f.bytes_per_vector()))
    }

    /// Returns up to `k` nearest vectors, closest first.
    ///
    /// `rerank = 0` returns PQ (approximate) distances. `rerank = r > 0`
    /// re-scores the best `max(r, k)` PQ candidates with full vectors and
    /// returns their exact distances (up to codec rounding).
    pub fn search(&self, query: &[f32], k: usize, rerank: usize) -> Result<Vec<SearchResult>> {
        let dim = self.dim();
        if query.len() != dim {
            return Err(Error::DimensionMismatch { expected: dim, got: query.len() });
        }
        check_finite(query)?;
        if rerank > 0 && self.full.is_none() {
            return Err(Error::InvalidArgument("re-ranking needs full vectors; build with a codec".into()));
        }
        if k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }
        let mut q = query.to_vec();
        if self.metric.needs_normalization() {
            normalize(&mut q);
        }

        // Stage 1: ADC scan over all codes. Candidate ids are row numbers.
        let table = self.pq.distance_table(self.metric, &q);
        let shortlist = if rerank > 0 { rerank.max(k) } else { k }.min(self.len());
        let mut top = TopK::new(shortlist);
        for (row, code) in self.codes.chunks_exact(self.code_size()).enumerate() {
            top.push(row as u64, table.distance(code));
        }
        let candidates = top.into_sorted();

        let Some(full) = self.full.as_ref().filter(|_| rerank > 0) else {
            return Ok(candidates
                .into_iter()
                .map(|c| SearchResult { id: self.ids[c.id as usize], distance: c.distance })
                .collect());
        };

        // Stage 2: exact re-ranking from the full vectors.
        let kern = kernels::best();
        let mut buf = vec![0.0f32; dim];
        let mut best = TopK::new(k.min(self.len()));
        for c in candidates {
            let row = c.id as usize;
            let v = full.get(row, &mut buf);
            let d = match self.metric {
                Metric::L2 => (kern.l2_squared)(&q, v),
                Metric::InnerProduct => -(kern.dot)(&q, v),
                Metric::Cosine => 1.0 - (kern.dot)(&q, v),
            };
            best.push(self.ids[row], d);
        }
        Ok(best.into_sorted())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat::FlatIndex;
    use crate::quant::pq::tests::structured;

    fn params(m: usize) -> PqParams {
        PqParams { m, iters: 10, max_train: 4096, seed: 2, threads: 2 }
    }

    #[test]
    fn reranking_everything_equals_flat_search() {
        let (dim, n) = (24, 1500);
        let data = structured(n, dim, 7);
        let ids: Vec<u64> = (0..n as u64).map(|i| i * 3 + 1).collect();
        let queries = structured(20, dim, 8);
        for metric in [Metric::L2, Metric::InnerProduct, Metric::Cosine] {
            let pq = PqIndex::build(metric, dim, &ids, &data, &params(6), Some(Codec::F32)).unwrap();
            let mut flat = FlatIndex::new(dim, metric).unwrap();
            for (row, id) in ids.iter().enumerate() {
                flat.add(*id, &data[row * dim..(row + 1) * dim]).unwrap();
            }
            for q in queries.chunks_exact(dim) {
                assert_eq!(pq.search(q, 10, n).unwrap(), flat.search(q, 10).unwrap(), "{metric:?}");
            }
        }
    }

    #[test]
    fn reranking_improves_recall() {
        let (dim, n, k) = (32, 3000, 10);
        let data = structured(n, dim, 9);
        let ids: Vec<u64> = (0..n as u64).collect();
        let mut flat = FlatIndex::new(dim, Metric::L2).unwrap();
        for (row, id) in ids.iter().enumerate() {
            flat.add(*id, &data[row * dim..(row + 1) * dim]).unwrap();
        }
        let mut pq = PqIndex::build(Metric::L2, dim, &ids, &data, &params(4), Some(Codec::F32)).unwrap();
        let queries = structured(50, dim, 10);
        let recall = |pq: &PqIndex, rerank: usize| -> f64 {
            let mut hits = 0;
            for q in queries.chunks_exact(dim) {
                let truth: Vec<u64> = flat.search(q, k).unwrap().iter().map(|r| r.id).collect();
                hits += pq.search(q, k, rerank).unwrap().iter().filter(|r| truth.contains(&r.id)).count();
            }
            hits as f64 / (50 * k) as f64
        };
        let r0 = recall(&pq, 0);
        let r100 = recall(&pq, 100);
        assert!(r100 > r0, "rerank 100 ({r100}) should beat PQ alone ({r0})");
        assert!(r100 > 0.9, "rerank 100 recall {r100}");
        for codec in [Codec::F16, Codec::BF16, Codec::SQ8] {
            pq.set_full_vectors(&data, Some(codec)).unwrap();
            let r = recall(&pq, 100);
            assert!(r > r100 - 0.05, "{}: recall {r} vs f32 {r100}", codec.name());
        }
    }

    #[test]
    fn validates_input() {
        let dim = 8;
        let data = structured(300, dim, 11);
        let ids: Vec<u64> = (0..300).collect();
        let mut dup = ids.clone();
        dup[5] = 0;
        assert!(matches!(
            PqIndex::build(Metric::L2, dim, &dup, &data, &params(2), None),
            Err(Error::DuplicateId(0))
        ));
        assert!(PqIndex::build(Metric::L2, dim, &ids[..299], &data, &params(2), None).is_err());
        let pq = PqIndex::build(Metric::L2, dim, &ids, &data, &params(2), None).unwrap();
        assert_eq!((pq.len(), pq.code_size(), pq.full_vectors()), (300, 2, None));
        assert!(pq.search(&data[..dim], 5, 50).is_err(), "rerank without full vectors");
        assert_eq!(pq.search(&data[..dim], 5, 0).unwrap().len(), 5);
        assert!(pq.search(&data[..dim - 1], 5, 0).is_err());
        assert!(pq.search(&[f32::NAN; 8], 5, 0).is_err());
        assert!(pq.search(&data[..dim], 0, 0).unwrap().is_empty());
    }
}
