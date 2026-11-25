//! Vamana graph construction (DiskANN: Subramanya et al., NeurIPS 2019).
//!
//! Points are inserted in random order. Each insert searches the current
//! graph from the entry point, prunes the nodes it expanded down to at most
//! R out-edges with RobustPrune(α), and adds the reverse edges.
//!
//! Inserts run in batches (sizes 1, 2, 4, ... up to 2% of the collection,
//! as in ParlayANN). Within a batch, searches and prunes run in parallel
//! against a read-only graph; then out-lists are installed, reverse edges are
//! grouped by target, and over-full targets are re-pruned in parallel. Batch
//! boundaries depend only on the collection size, so the graph is identical
//! for any thread count.

use crate::cancel::Cancel;
use crate::distance::Metric;
use crate::error::Result;
use crate::kernels::{self, Kernels};
use crate::par;
use crate::rng::SplitMix64;

use super::GraphParams;
use super::search::{Scratch, beam_search};

/// A reverse edge may grow a list to `SLACK * R` before it is pruned back
/// to R (DiskANN's graph slack): fewer prunes, same final degree bound.
const SLACK: f32 = 1.3;
/// Largest batch, as a fraction of the collection.
const MAX_BATCH_FRACTION: f64 = 0.02;
/// RobustPrune considers at most this many candidates, nearest first.
const MAX_POOL: usize = 750;

/// The vectors being indexed, with the distance used to build the graph.
///
/// Build distances are always non-negative and metric-like, so the
/// α-pruning rule makes sense:
/// - L2: squared L2.
/// - Cosine: 1 - dot on normalized rows (proportional to squared L2).
/// - Inner product: the cosine distance between rows (the graph links
///   vectors by direction); searches still rank by -dot.
///
/// For inner product we also tried the usual reduction to L2 (append
/// sqrt(M² - |x|²) to each row). It puts every query far from every point,
/// and on 1024-dim data it reached recall 0.63 where the angular graph
/// reached 0.93 (0.09 against 0.70 with skewed norms).
pub(crate) struct Points<'a> {
    pub data: &'a [f32],
    pub dim: usize,
    pub metric: Metric,
    /// 1 / |row| for inner product (0 for a zero row); empty otherwise.
    inv_norm: Vec<f32>,
}

impl<'a> Points<'a> {
    pub fn new(data: &'a [f32], dim: usize, metric: Metric, threads: usize) -> Self {
        let mut p = Self { data, dim, metric, inv_norm: Vec::new() };
        if metric == Metric::InnerProduct {
            p.inv_norm = par::map_ranges(p.len(), threads, |range| {
                range.map(|i| inv_norm(p.row(i as u32))).collect::<Vec<f32>>()
            })
            .concat();
        }
        p
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.dim
    }

    #[inline]
    pub fn row(&self, i: u32) -> &'a [f32] {
        let i = i as usize;
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    #[inline]
    pub fn inv_norm_of(&self, i: u32) -> f32 {
        self.inv_norm.get(i as usize).copied().unwrap_or(0.0)
    }

    /// Build distance from a point (row `x`, with `x_inv_norm` = 1 / |x|,
    /// used only for inner product) to node `b`.
    #[inline]
    pub fn dist_to(&self, kern: &Kernels, x: &[f32], x_inv_norm: f32, b: u32) -> f32 {
        let y = self.row(b);
        match self.metric {
            Metric::L2 => (kern.l2_squared)(x, y),
            Metric::Cosine => (1.0 - (kern.dot)(x, y)).max(0.0),
            Metric::InnerProduct => (1.0 - (kern.dot)(x, y) * x_inv_norm * self.inv_norm_of(b)).max(0.0),
        }
    }

    #[inline]
    pub fn dist(&self, kern: &Kernels, a: u32, b: u32) -> f32 {
        self.dist_to(kern, self.row(a), self.inv_norm_of(a), b)
    }
}

/// Builds the graph. Returns the out-lists (each at most `max_degree` long)
/// and the entry node (the point nearest the mean).
pub(crate) fn build(points: &Points, p: &GraphParams, cancel: &Cancel) -> Result<(Vec<Vec<u32>>, u32)> {
    let n = points.len();
    let mut graph: Vec<Vec<u32>> = vec![Vec::new(); n];
    if n == 0 {
        return Ok((graph, 0));
    }
    let entry = medoid(points, cancel.threads(p.threads));
    let max_batch = ((n as f64 * MAX_BATCH_FRACTION).ceil() as usize).max(1);
    for pass in 0..p.passes {
        // With two or more passes, all but the last use α = 1 (the paper's schedule).
        let alpha = if pass + 1 < p.passes { 1.0 } else { p.alpha };
        let order = permutation(n, p.seed.wrapping_add(pass as u64));
        let mut start = 0;
        let mut size = if pass == 0 { 1 } else { max_batch };
        while start < n {
            let end = (start + size).min(n);
            insert_batch(points, &mut graph, &order[start..end], entry, alpha, pass > 0, p, cancel);
            cancel.check()?;
            start = end;
            size = (size * 2).min(max_batch);
        }
    }
    // Reverse edges may have left lists between R and SLACK * R.
    let over: Vec<u32> = (0..n as u32).filter(|&i| graph[i as usize].len() > p.max_degree).collect();
    let g = &graph;
    let pruned = par::map_ranges(over.len(), cancel.threads(p.threads), |range| {
        let kern = kernels::best();
        let mut occl = Vec::new();
        range
            .map(|i| {
                let node = over[i];
                let mut pool = g[node as usize].iter().map(|&c| (points.dist(kern, node, c), c)).collect();
                (node, robust_prune(points, kern, node, &mut pool, p.alpha, p.max_degree, &mut occl))
            })
            .collect::<Vec<_>>()
    });
    for (node, list) in pruned.into_iter().flatten() {
        graph[node as usize] = list;
    }
    connect_unreachable(points, &mut graph, entry, p);
    Ok((graph, entry))
}

/// Vamana does not guarantee that every node is reachable from the entry:
/// pruning can remove all of a node's in-edges (0.25% of GIST1M's nodes).
/// Search can never return such a node. For each one, in node order, this
/// searches for it from the entry and adds an edge from the nearest reachable
/// node with a free slot. Returns the number of edges added.
fn connect_unreachable(points: &Points, graph: &mut [Vec<u32>], entry: u32, p: &GraphParams) -> usize {
    let n = graph.len();
    let mut reached = vec![false; n];
    mark_reachable(graph, entry, &mut reached);
    let kern = kernels::best();
    let mut scratch = Scratch::new();
    let mut added = 0;
    for u in 0..n as u32 {
        if reached[u as usize] {
            continue;
        }
        let (x, xn) = (points.row(u), points.inv_norm_of(u));
        let g: &[Vec<u32>] = graph;
        beam_search(
            &mut scratch,
            entry,
            p.build_list,
            |v| points.dist_to(kern, x, xn, v),
            move |v| &g[v as usize],
            |v| kernels::prefetch(points.row(v)),
        );
        // Every node the search found is reachable. If all of them are full,
        // take any reachable node with room; if there is none, give up on u.
        let has_room = |v: u32| g[v as usize].len() < p.max_degree;
        let from = scratch
            .list
            .iter()
            .map(|c| c.node)
            .find(|&v| has_room(v))
            .or_else(|| (0..n as u32).find(|&v| reached[v as usize] && has_room(v)));
        if let Some(from) = from {
            graph[from as usize].push(u);
            added += 1;
            mark_reachable(graph, u, &mut reached);
        }
    }
    added
}

/// Marks every node reachable from `start` (which is marked too).
fn mark_reachable(graph: &[Vec<u32>], start: u32, reached: &mut [bool]) {
    if std::mem::replace(&mut reached[start as usize], true) {
        return;
    }
    let mut stack = vec![start];
    while let Some(v) = stack.pop() {
        for &nb in &graph[v as usize] {
            if !std::mem::replace(&mut reached[nb as usize], true) {
                stack.push(nb);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn insert_batch(
    points: &Points,
    graph: &mut [Vec<u32>],
    batch: &[u32],
    entry: u32,
    alpha: f32,
    keep_existing: bool,
    p: &GraphParams,
    cancel: &Cancel,
) {
    let r = p.max_degree;
    let slack = ((r as f32 * SLACK).ceil() as usize).max(r);

    // 1. Search and prune every point of the batch against the current graph.
    let g: &[Vec<u32>] = graph;
    let lists = par::map_ranges(batch.len(), cancel.threads(p.threads), |range| {
        let kern = kernels::best();
        let mut scratch = Scratch::new();
        let mut occl = Vec::new();
        range
            // (Cancelled, the graph is dropped: what is in it does not matter.)
            .take_while(|_| !cancel.is_cancelled())
            .map(|i| {
                let node = batch[i];
                let (x, xn) = (points.row(node), points.inv_norm_of(node));
                beam_search(
                    &mut scratch,
                    entry,
                    p.build_list,
                    |v| points.dist_to(kern, x, xn, v),
                    move |v| &g[v as usize],
                    |v| kernels::prefetch(points.row(v)),
                );
                let mut pool = std::mem::take(&mut scratch.expanded);
                if keep_existing {
                    pool.extend(g[node as usize].iter().map(|&c| (points.dist(kern, node, c), c)));
                }
                let out = robust_prune(points, kern, node, &mut pool, alpha, r, &mut occl);
                scratch.expanded = pool; // hand the buffer back for reuse
                (node, out)
            })
            .collect::<Vec<_>>()
    });

    // 2. Install the out-lists and collect reverse edges (target, source).
    let mut reverse: Vec<(u32, u32)> = Vec::new();
    for (node, out) in lists.into_iter().flatten() {
        reverse.extend(out.iter().map(|&t| (t, node)));
        graph[node as usize] = out;
    }
    reverse.sort_unstable();
    reverse.dedup();
    let mut groups = Vec::new(); // ranges of `reverse` sharing a target
    let mut start = 0;
    while start < reverse.len() {
        let target = reverse[start].0;
        let end = start + reverse[start..].partition_point(|e| e.0 == target);
        groups.push(start..end);
        start = end;
    }

    // 3. Add reverse edges; re-prune targets whose lists grew past the slack.
    let g: &[Vec<u32>] = graph;
    let updates = par::map_ranges(groups.len(), cancel.threads(p.threads), |range| {
        let kern = kernels::best();
        let mut occl = Vec::new();
        range
            .filter_map(|gi| {
                let edges = &reverse[groups[gi].clone()];
                let target = edges[0].0;
                let current = &g[target as usize];
                let mut list = current.clone();
                for &(_, src) in edges {
                    if src != target && !list.contains(&src) {
                        list.push(src);
                    }
                }
                if list.len() == current.len() {
                    return None;
                }
                if list.len() > slack {
                    let mut pool = list.iter().map(|&c| (points.dist(kern, target, c), c)).collect();
                    list = robust_prune(points, kern, target, &mut pool, alpha, r, &mut occl);
                }
                Some((target, list))
            })
            .collect::<Vec<_>>()
    });
    for (target, list) in updates.into_iter().flatten() {
        graph[target as usize] = list;
    }
}

/// RobustPrune(node, pool, α, R). `pool` holds (build distance to `node`,
/// candidate). Picks candidates nearest first; each pick p* removes every
/// remaining p' with α·d(p*, p') <= d(node, p'). Like DiskANN, it first runs
/// with α = 1 and then raises α (×1.2 per round) while slots are free.
pub(crate) fn robust_prune(
    points: &Points,
    kern: &Kernels,
    node: u32,
    pool: &mut Vec<(f32, u32)>,
    alpha: f32,
    r: usize,
    occl: &mut Vec<f32>,
) -> Vec<u32> {
    pool.retain(|&(_, c)| c != node);
    pool.sort_unstable_by(|a, b| a.1.cmp(&b.1).then(a.0.total_cmp(&b.0)));
    pool.dedup_by_key(|e| e.1);
    pool.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    pool.truncate(MAX_POOL);

    // occl[j]: largest d(node, j) / d(p*, j) over picks p* so far;
    // infinity once j is picked or a pick duplicates it.
    occl.clear();
    occl.resize(pool.len(), 0.0);
    let mut out = Vec::with_capacity(r.min(pool.len()));
    let mut cur = 1.0f32;
    loop {
        for i in 0..pool.len() {
            if out.len() >= r {
                break;
            }
            if occl[i] > cur {
                continue;
            }
            occl[i] = f32::INFINITY;
            let pick = pool[i].1;
            out.push(pick);
            let (x, xn) = (points.row(pick), points.inv_norm_of(pick));
            for j in i + 1..pool.len() {
                if occl[j] > alpha {
                    continue;
                }
                let d = points.dist_to(kern, x, xn, pool[j].1);
                occl[j] = if d > 0.0 { occl[j].max(pool[j].0 / d) } else { f32::INFINITY };
            }
        }
        if out.len() >= r || cur >= alpha {
            break;
        }
        cur = (cur * 1.2).min(alpha);
    }
    out
}

/// Rows per chunk when summing for the mean: see [`chunk_sum`].
pub(crate) const MEAN_CHUNK: usize = 4096;

/// Sum of one chunk of rows (at most [`MEAN_CHUNK`]), in f64. The mean adds
/// the sums of consecutive chunks in order ([`mean_of`]), so its rounding,
/// and so the medoid, does not depend on the thread count or on how a
/// streaming build batches its reads.
pub(crate) fn chunk_sum<'a>(rows: impl Iterator<Item = &'a [f32]>, dim: usize) -> Vec<f64> {
    let mut s = vec![0.0f64; dim];
    for row in rows {
        for (acc, &x) in s.iter_mut().zip(row) {
            *acc += x as f64;
        }
    }
    s
}

/// Mean of `n` rows from the sums of their consecutive chunks.
pub(crate) fn mean_of<'a>(chunk_sums: impl Iterator<Item = &'a Vec<f64>>, n: usize, dim: usize) -> Vec<f32> {
    let mut mean = vec![0.0f64; dim];
    for s in chunk_sums {
        for (m, v) in mean.iter_mut().zip(s) {
            *m += v;
        }
    }
    mean.iter().map(|&m| (m / n as f64) as f32).collect()
}

/// The point nearest the mean of all points (by build distance).
fn medoid(points: &Points, threads: usize) -> u32 {
    let (n, dim) = (points.len(), points.dim);
    let chunks: Vec<Vec<f64>> = par::map_ranges(n.div_ceil(MEAN_CHUNK), threads, |range| {
        range
            .map(|c| chunk_sum((c * MEAN_CHUNK..((c + 1) * MEAN_CHUNK).min(n)).map(|i| points.row(i as u32)), dim))
            .collect::<Vec<_>>()
    })
    .concat();
    let mean = mean_of(chunks.iter(), n, dim);
    let mean_inv_norm = inv_norm(&mean);
    let best = par::map_ranges(n, threads, |range| {
        let kern = kernels::best();
        range
            .map(|i| (points.dist_to(kern, &mean, mean_inv_norm, i as u32), i as u32))
            .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
    });
    best.into_iter().flatten().min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))).map_or(0, |b| b.1)
}

/// 1 / |v| (computed in f64), or 0 for a zero vector.
pub(crate) fn inv_norm(v: &[f32]) -> f32 {
    let n = v.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
    if n > 0.0 { (1.0 / n) as f32 } else { 0.0 }
}

/// Random permutation of 0..n (Fisher-Yates).
fn permutation(n: usize, seed: u64) -> Vec<u32> {
    let mut rng = SplitMix64::new(seed);
    let mut v: Vec<u32> = (0..n as u32).collect();
    for i in (1..n).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        v.swap(i, j);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prune_keeps_diverse_neighbors() {
        // Node 0 at the origin; 1 and 2 close together on +x, 3 on +y.
        let data = [0.0, 0.0, 1.0, 0.0, 1.1, 0.0, 0.0, 1.5];
        let points = Points::new(&data, 2, Metric::L2, 1);
        let kern = kernels::best();
        let mut pool: Vec<(f32, u32)> = (1..4).map(|c| (points.dist(kern, 0, c), c)).collect();
        let out = robust_prune(&points, kern, 0, &mut pool, 1.2, 8, &mut Vec::new());
        // 2 is occluded by 1 (d(1,2) = 0.01 << d(0,2) = 1.21); 3 is not.
        assert_eq!(out, vec![1, 3]);
        let mut pool: Vec<(f32, u32)> = (1..4).map(|c| (points.dist(kern, 0, c), c)).collect();
        assert_eq!(robust_prune(&points, kern, 0, &mut pool, 1.2, 1, &mut Vec::new()), vec![1]);
    }

    #[test]
    fn inner_product_builds_by_angle() {
        let data = [3.0, 0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0];
        let points = Points::new(&data, 2, Metric::InnerProduct, 1);
        let kern = kernels::best();
        // Same direction: distance 0 whatever the norms.
        assert_eq!(points.dist(kern, 0, 1), 0.0);
        // Orthogonal: 1. At 45 degrees: 1 - cos 45.
        assert!((points.dist(kern, 0, 2) - 1.0).abs() < 1e-6);
        assert!((points.dist(kern, 0, 3) - (1.0 - 0.5f32.sqrt())).abs() < 1e-6);
    }

    #[test]
    fn unreachable_nodes_get_an_in_edge() {
        // Six points on a line; node 0 is the entry. Nodes 3, 4 and 5 have
        // no in-edges from the reachable part {0, 1, 2}; 5 also has none at all.
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let points = Points::new(&data, 1, Metric::L2, 1);
        let mut graph = vec![vec![1], vec![2], vec![1], vec![4], vec![3], vec![]];
        let p = GraphParams { max_degree: 2, build_list: 4, ..GraphParams::default() };
        let added = connect_unreachable(&points, &mut graph, 0, &p);
        let mut reached = vec![false; 6];
        mark_reachable(&graph, 0, &mut reached);
        assert!(reached.iter().all(|&r| r), "{graph:?}");
        // 3 is reached from 2 (its nearest reachable node); 4 then comes
        // through 3, and 5 needs its own edge.
        assert_eq!(added, 2, "{graph:?}");
        assert_eq!(graph[2], vec![1, 3]);
    }

    #[test]
    fn permutation_is_a_permutation() {
        let mut p = permutation(1000, 3);
        assert_ne!(p, (0..1000).collect::<Vec<u32>>());
        p.sort();
        assert_eq!(p, (0..1000).collect::<Vec<u32>>());
    }
}
