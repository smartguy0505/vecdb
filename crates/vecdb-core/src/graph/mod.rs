//! In-memory graph index: a Vamana graph (the graph DiskANN uses) over full
//! vectors, searched with a greedy beam search.
//!
//! The search-time list size `l` is the recall/latency knob: a longer list
//! expands more nodes and computes more distances.
//!
//! Two ways to navigate the same graph:
//! - [`Navigation::Exact`]: full vectors in memory steer the search.
//! - [`Navigation::Pq`]: PQ codes steer it, and every expanded node is
//!   re-scored with its full vector. This is the disk index's search (codes
//!   in RAM, graph and full vectors on SSD, one read per expanded node), run
//!   in memory so its recall can be measured before any I/O exists.
//!
//! Identical vectors share one graph node. Without that, copies of a vector
//! prune each other (distance 0) and most of them become unreachable.
//!
//! Scope for now: batch build only, no inserts or deletes. A built graph is
//! saved as an SSD index with [`DiskIndex::write`](crate::disk::DiskIndex::write).

pub(crate) mod build;
pub(crate) mod search;
pub(crate) mod visited;

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use crate::cancel::Cancel;
use crate::distance::{Metric, check_finite, normalize};
use crate::error::{Error, Result};
use crate::kernels::{self, Kernels};
use crate::quant::{PqParams, ProductQuantizer};
use crate::store::check_dim;
use crate::topk::{SearchResult, TopK};

pub use search::SearchStats;
use search::{Scratch, beam_search};

/// Largest supported out-degree.
pub const MAX_DEGREE: usize = 1024;

#[derive(Clone, Debug)]
pub struct GraphParams {
    /// R: maximum out-degree. The graph costs 4 bytes per edge slot.
    pub max_degree: usize,
    /// L used while building: larger gives a better graph and a slower build.
    pub build_list: usize,
    /// α >= 1: larger keeps more long-range edges (fewer hops, more edges).
    pub alpha: f32,
    /// 1 = one pass with α. 2 = a pass with α = 1, then one with α.
    pub passes: usize,
    pub seed: u64,
    /// 0 = one thread per CPU. The graph does not depend on it.
    pub threads: usize,
}

impl Default for GraphParams {
    fn default() -> Self {
        Self { max_degree: 64, build_list: 100, alpha: 1.2, passes: 1, seed: 0x5eed, threads: 0 }
    }
}

impl GraphParams {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.max_degree == 0 || self.max_degree > MAX_DEGREE {
            return Err(Error::InvalidArgument(format!("max_degree must be in 1..={MAX_DEGREE}")));
        }
        if self.build_list == 0 {
            return Err(Error::InvalidArgument("build_list must be at least 1".into()));
        }
        if !(self.alpha >= 1.0 && self.alpha.is_finite()) {
            return Err(Error::InvalidArgument(format!("alpha must be finite and >= 1, got {}", self.alpha)));
        }
        if self.passes == 0 {
            return Err(Error::InvalidArgument("passes must be at least 1".into()));
        }
        Ok(())
    }
}

/// What steers a graph search. See the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Navigation {
    Exact,
    Pq,
}

#[derive(Clone, Debug)]
pub struct GraphIndex {
    metric: Metric,
    dim: usize,
    /// One row per node, prepared for the metric (normalized for cosine).
    data: Vec<f32>,
    /// Node `i` holds `ids[id_start[i]..id_start[i + 1]]`.
    id_start: Vec<usize>,
    ids: Vec<u64>,
    max_degree: usize,
    /// Stride `max_degree + 1` per node: degree, then the neighbors.
    adjacency: Vec<u32>,
    entry: u32,
    pq: Option<(ProductQuantizer, Vec<u8>)>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::new());
}

impl GraphIndex {
    /// Builds the graph over `data` (row-major, `dim` per row; row `i` has
    /// id `ids[i]`). The index keeps `data` as its full vectors, so large
    /// collections are not copied.
    pub fn build(metric: Metric, dim: usize, ids: &[u64], mut data: Vec<f32>, params: &GraphParams) -> Result<Self> {
        check_dim(dim)?;
        params.validate()?;
        if Some(data.len()) != ids.len().checked_mul(dim) {
            return Err(Error::InvalidArgument(format!(
                "{} values do not form {} rows of dimension {dim}",
                data.len(),
                ids.len()
            )));
        }
        if ids.len() >= u32::MAX as usize {
            return Err(Error::InvalidArgument("a graph holds fewer than 2^32 - 1 vectors".into()));
        }
        let mut seen = HashSet::with_capacity(ids.len());
        if let Some(&dup) = ids.iter().find(|&&id| !seen.insert(id)) {
            return Err(Error::DuplicateId(dup));
        }
        drop(seen);
        check_finite(&data)?;
        if metric.needs_normalization() {
            data.chunks_exact_mut(dim).for_each(normalize);
        }
        let (data, id_start, node_ids) = dedup(data, dim, ids);
        let points = build::Points::new(&data, dim, metric, params.threads);
        let (lists, entry) = build::build(&points, params, &Cancel::default())?;
        drop(points);

        let stride = params.max_degree + 1;
        let mut adjacency = vec![0u32; lists.len() * stride];
        for (slot, list) in adjacency.chunks_exact_mut(stride).zip(&lists) {
            slot[0] = list.len() as u32;
            slot[1..=list.len()].copy_from_slice(list);
        }
        Ok(Self {
            metric,
            dim,
            data,
            id_start,
            ids: node_ids,
            max_degree: params.max_degree,
            adjacency,
            entry,
            pq: None,
        })
    }

    /// Trains PQ on the stored vectors and encodes them, for [`Navigation::Pq`].
    pub fn train_pq(&mut self, params: &PqParams) -> Result<()> {
        let pq = ProductQuantizer::train(self.dim, &self.data, params)?;
        let codes = pq.encode_all(&self.data, params.threads);
        self.pq = Some((pq, codes));
        Ok(())
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of vectors (ids).
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Number of graph nodes (distinct vectors).
    pub fn nodes(&self) -> usize {
        self.id_start.len() - 1
    }

    pub fn max_degree(&self) -> usize {
        self.max_degree
    }

    pub fn entry(&self) -> u32 {
        self.entry
    }

    /// Out-neighbors of `node`.
    #[inline]
    pub fn neighbors(&self, node: u32) -> &[u32] {
        let slot = &self.adjacency[node as usize * (self.max_degree + 1)..][..self.max_degree + 1];
        &slot[1..=slot[0] as usize]
    }

    /// Ids stored at `node`.
    pub fn ids_of(&self, node: u32) -> &[u64] {
        &self.ids[self.id_start[node as usize]..self.id_start[node as usize + 1]]
    }

    /// Mean out-degree.
    pub fn mean_degree(&self) -> f64 {
        let edges: usize = (0..self.nodes() as u32).map(|v| self.neighbors(v).len()).sum();
        edges as f64 / self.nodes().max(1) as f64
    }

    /// Number of nodes reachable from the entry point. Nodes that are not
    /// reachable can never be returned.
    pub fn reachable(&self) -> usize {
        if self.nodes() == 0 {
            return 0;
        }
        let mut seen = vec![false; self.nodes()];
        let mut stack = vec![self.entry];
        seen[self.entry as usize] = true;
        let mut count = 0;
        while let Some(v) = stack.pop() {
            count += 1;
            for &nb in self.neighbors(v) {
                if !std::mem::replace(&mut seen[nb as usize], true) {
                    stack.push(nb);
                }
            }
        }
        count
    }

    /// Bytes used by the graph, the full vectors and the PQ codes.
    pub fn memory(&self) -> (usize, usize, usize) {
        let pq = self.pq.as_ref().map_or(0, |(_, c)| c.len());
        (self.adjacency.len() * 4, self.data.len() * 4, pq)
    }

    /// PQ bytes per vector, if PQ is trained.
    /// The trained quantizer and the codes of every node, if PQ is trained.
    pub(crate) fn pq_parts(&self) -> Option<(&ProductQuantizer, &[u8])> {
        self.pq.as_ref().map(|(pq, codes)| (pq, codes.as_slice()))
    }

    pub fn pq_code_size(&self) -> Option<usize> {
        self.pq.as_ref().map(|(pq, _)| pq.code_size())
    }

    /// The `k` nearest vectors found with a candidate list of `l` (raised to
    /// at least `k`), closest first, with exact distances.
    pub fn search(&self, query: &[f32], k: usize, l: usize) -> Result<Vec<SearchResult>> {
        Ok(self.search_with(query, k, l, Navigation::Exact)?.0)
    }

    /// Like [`search`](Self::search), with a choice of navigation and the
    /// search counters. Returned distances are always exact.
    pub fn search_with(
        &self,
        query: &[f32],
        k: usize,
        l: usize,
        nav: Navigation,
    ) -> Result<(Vec<SearchResult>, SearchStats)> {
        if query.len() != self.dim {
            return Err(Error::DimensionMismatch { expected: self.dim, got: query.len() });
        }
        check_finite(query)?;
        if nav == Navigation::Pq && self.pq.is_none() {
            return Err(Error::InvalidArgument("PQ navigation needs train_pq first".into()));
        }
        if k == 0 || self.is_empty() {
            return Ok((Vec::new(), SearchStats::default()));
        }
        let mut q = query.to_vec();
        if self.metric.needs_normalization() {
            normalize(&mut q);
        }
        let l = l.max(k);
        let kern = kernels::best();
        let neighbors = move |v: u32| self.neighbors(v);
        SCRATCH.with_borrow_mut(|s| {
            let mut top = TopK::new(k);
            let stats = match &self.pq {
                Some((pq, codes)) if nav == Navigation::Pq => {
                    let table = pq.distance_table(self.metric, &q);
                    let cs = pq.code_size();
                    let code = |v: u32| &codes[v as usize * cs..][..cs];
                    let stats = beam_search(
                        s,
                        self.entry,
                        l,
                        |v| table.distance(code(v)),
                        neighbors,
                        |v| kernels::prefetch(code(v)),
                    );
                    for &(_, node) in &s.expanded {
                        let d = self.distance(kern, &q, node);
                        self.ids_of(node).iter().for_each(|&id| top.push(id, d));
                    }
                    stats
                }
                _ => {
                    let stats = beam_search(
                        s,
                        self.entry,
                        l,
                        |v| self.distance(kern, &q, v),
                        neighbors,
                        |v| kernels::prefetch(self.row(v)),
                    );
                    for c in &s.list {
                        self.ids_of(c.node).iter().for_each(|&id| top.push(id, c.dist));
                    }
                    stats
                }
            };
            Ok((top.into_sorted(), stats))
        })
    }

    #[inline]
    pub(crate) fn row(&self, node: u32) -> &[f32] {
        &self.data[node as usize * self.dim..][..self.dim]
    }

    /// Exact distance from a prepared query to `node`, as the flat index computes it.
    #[inline]
    fn distance(&self, kern: &Kernels, q: &[f32], node: u32) -> f32 {
        let v = self.row(node);
        match self.metric {
            Metric::L2 => (kern.l2_squared)(q, v),
            Metric::InnerProduct => -(kern.dot)(q, v),
            Metric::Cosine => 1.0 - (kern.dot)(q, v),
        }
    }
}

/// Hash of a row for finding identical rows. Rows equal by `==` hash alike
/// (-0.0 is hashed as 0.0).
pub(crate) fn row_hash(row: &[f32]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    row.iter().for_each(|&x| (x + 0.0).to_bits().hash(&mut h));
    h.finish()
}

/// Groups identical rows. Returns one row per distinct vector (in order of
/// first appearance), the id ranges per node, and the ids in node order.
fn dedup(data: Vec<f32>, dim: usize, ids: &[u64]) -> (Vec<f32>, Vec<usize>, Vec<u64>) {
    use std::collections::hash_map::Entry;
    let n = ids.len();
    let row = |i: usize| &data[i * dim..(i + 1) * dim];
    // rep[i]: the first row equal to row i (`==`, so 0.0 equals -0.0; the
    // hash canonicalizes -0.0 to match).
    let mut first: HashMap<u64, usize> = HashMap::with_capacity(n);
    let mut collisions: HashMap<u64, Vec<usize>> = HashMap::new();
    let mut rep = Vec::with_capacity(n);
    for i in 0..n {
        let h = row_hash(row(i));
        let r = match first.entry(h) {
            Entry::Vacant(e) => *e.insert(i),
            Entry::Occupied(e) if row(*e.get()) == row(i) => *e.get(),
            Entry::Occupied(_) => {
                // Different vectors with the same 64-bit hash.
                let list = collisions.entry(h).or_default();
                match list.iter().find(|&&f| row(f) == row(i)) {
                    Some(&f) => f,
                    None => {
                        list.push(i);
                        i
                    }
                }
            }
        };
        rep.push(r);
    }
    drop((first, collisions));
    let distinct = (0..n).filter(|&i| rep[i] == i).count();
    if distinct == n {
        return (data, (0..=n).collect(), ids.to_vec());
    }
    let mut node_of = vec![usize::MAX; n];
    let mut nodes_data = Vec::with_capacity(distinct * dim);
    let mut id_start = vec![0usize; distinct + 1];
    for i in 0..n {
        if rep[i] == i {
            node_of[i] = nodes_data.len() / dim;
            nodes_data.extend_from_slice(row(i));
        }
        id_start[node_of[rep[i]] + 1] += 1;
    }
    for v in 1..=distinct {
        id_start[v] += id_start[v - 1];
    }
    let mut fill = id_start.clone();
    let mut node_ids = vec![0u64; n];
    for i in 0..n {
        let node = node_of[rep[i]];
        node_ids[fill[node]] = ids[i];
        fill[node] += 1;
    }
    (nodes_data, id_start, node_ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat::FlatIndex;
    use crate::quant::pq::tests::structured;

    const METRICS: [Metric; 3] = [Metric::L2, Metric::InnerProduct, Metric::Cosine];

    fn params(r: usize, l: usize, threads: usize) -> GraphParams {
        GraphParams { max_degree: r, build_list: l, alpha: 1.2, passes: 1, seed: 9, threads }
    }

    fn flat(metric: Metric, dim: usize, ids: &[u64], data: &[f32]) -> FlatIndex {
        let mut f = FlatIndex::new(dim, metric).unwrap();
        for (row, &id) in ids.iter().enumerate() {
            f.add(id, &data[row * dim..(row + 1) * dim]).unwrap();
        }
        f
    }

    fn check_invariants(g: &GraphIndex) {
        for v in 0..g.nodes() as u32 {
            let nb = g.neighbors(v);
            assert!(nb.len() <= g.max_degree(), "node {v} degree {}", nb.len());
            assert!(!nb.contains(&v), "self loop at {v}");
            let mut sorted = nb.to_vec();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), nb.len(), "duplicate neighbor at {v}");
            assert!(nb.iter().all(|&x| (x as usize) < g.nodes()));
        }
        assert_eq!(g.reachable(), g.nodes(), "unreachable nodes");
    }

    #[test]
    fn full_list_equals_flat_search() {
        let (dim, n) = (24, 1500);
        let data = structured(n, dim, 1);
        let ids: Vec<u64> = (0..n as u64).map(|i| i * 7 + 3).collect();
        let queries = structured(20, dim, 2);
        for metric in METRICS {
            let mut g = GraphIndex::build(metric, dim, &ids, data.clone(), &params(16, 40, 2)).unwrap();
            check_invariants(&g);
            g.train_pq(&PqParams { m: 4, iters: 5, max_train: 2000, seed: 1, threads: 2 }).unwrap();
            let f = flat(metric, dim, &ids, &data);
            for q in queries.chunks_exact(dim) {
                let want = f.search(q, 10).unwrap();
                assert_eq!(g.search(q, 10, n).unwrap(), want, "{metric:?}");
                // Every node gets expanded, so every node is re-scored exactly.
                assert_eq!(g.search_with(q, 10, n, Navigation::Pq).unwrap().0, want, "{metric:?} pq");
            }
        }
    }

    #[test]
    fn recall_at_moderate_list_size() {
        let (dim, n, k, nq) = (32, 4000, 10, 50);
        let data = structured(n, dim, 3);
        let ids: Vec<u64> = (0..n as u64).collect();
        let queries = structured(nq, dim, 4);
        for metric in METRICS {
            let mut g = GraphIndex::build(metric, dim, &ids, data.clone(), &params(24, 60, 3)).unwrap();
            check_invariants(&g);
            g.train_pq(&PqParams { m: 8, iters: 8, max_train: 4000, seed: 1, threads: 2 }).unwrap();
            let f = flat(metric, dim, &ids, &data);
            let (mut hits, mut hits_pq, mut dists) = (0, 0, 0);
            for q in queries.chunks_exact(dim) {
                let truth: Vec<u64> = f.search(q, k).unwrap().iter().map(|r| r.id).collect();
                let (res, stats) = g.search_with(q, k, 40, Navigation::Exact).unwrap();
                hits += res.iter().filter(|r| truth.contains(&r.id)).count();
                dists += stats.distances;
                let (res, _) = g.search_with(q, k, 40, Navigation::Pq).unwrap();
                hits_pq += res.iter().filter(|r| truth.contains(&r.id)).count();
            }
            let recall = hits as f64 / (nq * k) as f64;
            let recall_pq = hits_pq as f64 / (nq * k) as f64;
            let mean_dists = dists / nq;
            // Inner product is the weak metric for graphs (see build.rs and
            // the README); on this small 32-dim set it gets about 0.90.
            let floor = if metric == Metric::InnerProduct { 0.85 } else { 0.95 };
            assert!(recall >= floor, "{metric:?}: recall {recall}");
            assert!(recall_pq >= 0.85, "{metric:?}: PQ-guided recall {recall_pq}");
            assert!(mean_dists < n / 3, "{metric:?}: {mean_dists} distances per query is close to a scan");
        }
    }

    #[test]
    fn same_graph_for_any_thread_count() {
        let (dim, n) = (16, 2000);
        let data = structured(n, dim, 5);
        let ids: Vec<u64> = (0..n as u64).collect();
        for metric in METRICS {
            let a = GraphIndex::build(metric, dim, &ids, data.clone(), &params(12, 30, 1)).unwrap();
            let b = GraphIndex::build(metric, dim, &ids, data.clone(), &params(12, 30, 4)).unwrap();
            assert_eq!((a.entry, &a.adjacency), (b.entry, &b.adjacency), "{metric:?}");
        }
        let two = GraphParams { passes: 2, ..params(12, 30, 3) };
        let g = GraphIndex::build(Metric::L2, dim, &ids, data.clone(), &two).unwrap();
        check_invariants(&g);
    }

    #[test]
    fn tiny_collections() {
        let dim = 4;
        for n in [0usize, 1, 2, 3, 10, 17] {
            let data = structured(n, dim, 6);
            let ids: Vec<u64> = (0..n as u64).rev().collect();
            for metric in METRICS {
                let g = GraphIndex::build(metric, dim, &ids, data.clone(), &params(8, 10, 2)).unwrap();
                assert_eq!(g.len(), n);
                if n > 0 {
                    check_invariants(&g);
                }
                let f = flat(metric, dim, &ids, &data);
                let q = [0.3, -0.2, 0.9, 0.1];
                // k > n returns everything; l is raised to k.
                assert_eq!(g.search(&q, 20, 1).unwrap(), f.search(&q, 20).unwrap(), "{metric:?} n {n}");
            }
        }
    }

    #[test]
    fn identical_vectors_share_a_node() {
        let dim = 8;
        // 300 copies of one vector, then 50 copies of another plus noise rows.
        let mut data = [0.5f32; 8].repeat(300);
        let mut noise = structured(500, dim, 7);
        for row in noise.chunks_exact_mut(dim).take(50) {
            row.copy_from_slice(&[-1.0, 2.0, 0.0, -0.0, 1.0, 1.0, 3.0, 0.25]);
        }
        data.extend(noise);
        let n = data.len() / dim;
        let ids: Vec<u64> = (0..n as u64).map(|i| 1000 - i).collect();
        let g = GraphIndex::build(Metric::L2, dim, &ids, data.clone(), &params(8, 20, 2)).unwrap();
        assert_eq!(g.nodes(), n - 299 - 49);
        check_invariants(&g);
        let f = flat(Metric::L2, dim, &ids, &data);
        for q in [[0.5f32; 8], [-1.0, 2.0, -0.0, 0.0, 1.0, 1.0, 3.0, 0.25]] {
            // All 50 results sit at distance 0 on one node.
            assert_eq!(g.search(&q, 50, 10).unwrap(), f.search(&q, 50).unwrap());
            assert_eq!(g.search(&q, 60, n).unwrap(), f.search(&q, 60).unwrap());
        }
    }

    #[test]
    fn validates_input() {
        let dim = 4;
        let data = structured(300, dim, 8);
        let ids: Vec<u64> = (0..300).collect();
        let mut dup = ids.clone();
        dup[9] = 0;
        let p = params(8, 16, 1);
        assert!(matches!(GraphIndex::build(Metric::L2, dim, &dup, data.clone(), &p), Err(Error::DuplicateId(0))));
        assert!(GraphIndex::build(Metric::L2, dim, &ids[..299], data.clone(), &p).is_err());
        for bad in [
            GraphParams { max_degree: 0, ..p.clone() },
            GraphParams { max_degree: MAX_DEGREE + 1, ..p.clone() },
            GraphParams { build_list: 0, ..p.clone() },
            GraphParams { alpha: 0.9, ..p.clone() },
            GraphParams { alpha: f32::NAN, ..p.clone() },
            GraphParams { passes: 0, ..p.clone() },
        ] {
            assert!(GraphIndex::build(Metric::L2, dim, &ids, data.clone(), &bad).is_err(), "{bad:?}");
        }
        let mut nan = data.clone();
        nan[5] = f32::NAN;
        assert!(GraphIndex::build(Metric::L2, dim, &ids, nan.clone(), &p).is_err());

        let g = GraphIndex::build(Metric::L2, dim, &ids, data.clone(), &p).unwrap();
        assert!(g.search(&data[..3], 5, 10).is_err());
        assert!(g.search(&[f32::INFINITY, 0.0, 0.0, 0.0], 5, 10).is_err());
        assert!(g.search(&data[..dim], 0, 10).unwrap().is_empty());
        assert!(g.search_with(&data[..dim], 5, 10, Navigation::Pq).is_err(), "PQ not trained");
    }
}
