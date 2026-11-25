//! Partitioned build: an SSD index for a collection whose vectors and graph
//! do not fit in RAM together (the merged-shard build of DiskANN).
//!
//! 1. Scan: read every row; check it, prepare it for the metric and hash it.
//!    Identical rows share one node. Nodes are numbered in row order, as in
//!    the in-memory build.
//! 2. Train: PQ, plus k-means centroids that split the nodes into
//!    partitions, both on one random sample of nodes. With a memory budget,
//!    the partition count grows until the largest partition (estimated on
//!    the sample) fits.
//! 3. Encode: read every node again. PQ-encode it, add it to the mean (for
//!    the entry point), and assign it to its `overlap` nearest centroids, so
//!    neighboring partitions share nodes and their graphs join up.
//! 4. Partitions, one at a time: read the partition's vectors, build a
//!    Vamana graph over them in memory, and write the out-lists, with their
//!    exact distances, to a scratch file in node order.
//! 5. Merge: join each node's out-lists from its partitions and cut them
//!    back to R ([`MergePrune`]). Then walk the merged graph from the entry
//!    and give an in-edge to every node the walk cannot reach.
//! 6. Write: read every node a third time and stream it, with its merged
//!    out-list, into the index file.
//!
//! RAM: one partition's vectors and graph (`memory_budget`), plus each
//! node's PQ code and partition numbers, plus read buffers of up to 32 MB
//! per thread. Before that, the scan needs 16 bytes per row; after it, the
//! reachability walk needs up to 4 bytes per node. The plan sizes the
//! partitions from a sample with a 10% margin; a partition that still comes
//! out over the budget is built anyway, with a warning in the progress log.
//! Scratch disk: the partition out-lists and the merged graph, both removed
//! when the build ends.
//!
//! With one partition, the file is byte for byte the one that
//! [`GraphIndex::build`](crate::GraphIndex::build), `train_pq` and
//! [`DiskIndex::write`](super::DiskIndex::write) produce.

use std::borrow::Cow;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::write::{DiskWriteOptions, DiskWriteReport, IndexWriter, Shape, header};
use super::{free_space, read_exact_at};
use crate::cancel::Cancel;
use crate::dataset::VectorSource;
use crate::distance::{Metric, check_finite, normalize};
use crate::error::{Error, Result};
use crate::graph::build::{MEAN_CHUNK, Points, build as build_graph, chunk_sum, inv_norm, robust_prune};
use crate::graph::{GraphParams, row_hash};
use crate::kernels::{self, Kernels};
use crate::par;
use crate::quant::kmeans::{self, KMeansParams};
use crate::quant::pq::KSUB;
use crate::quant::{PqParams, ProductQuantizer};
use crate::rng::SplitMix64;
use crate::store::check_dim;

/// Rows per streaming read: about this many bytes of `f32`.
const READ_BYTES: usize = 64 << 20;
/// A gather reads through gaps of up to this many bytes rather than issuing
/// another read. Larger gaps made each of GIST1M's 23 partitions read most of
/// the file (64 KB is 17 rows there, and a partition holds every 12th row).
const GAP_BYTES: usize = 16 << 10;
/// Largest single gather read, in `f32` bytes.
const RUN_BYTES: usize = 16 << 20;
/// Nodes merged (and partition records written) per parallel batch.
const BATCH: usize = 1 << 16;
/// k-means iterations for the partition centroids.
const PARTITION_ITERS: usize = 10;
/// Partition counts tried before giving up on a memory budget.
const PLAN_TRIES: usize = 40;
/// Margin on the partition size estimated from the sample.
const SIZE_MARGIN: f64 = 1.1;

/// How a node's out-lists from its partitions are cut back to R.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergePrune {
    /// RobustPrune(α) over the joined list. Distances from the node are exact
    /// (kept from the partition builds); distances between candidates come
    /// from their PQ codes, since their vectors are not in RAM.
    Pq,
    /// Shuffle the joined list and keep R (DiskANN's merge), seeded per
    /// node. Needs no codes, but gave lower recall than `Pq` on SIFT1M.
    Random,
}

impl MergePrune {
    pub fn name(self) -> &'static str {
        match self {
            MergePrune::Pq => "pq",
            MergePrune::Random => "random",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "pq" => Some(MergePrune::Pq),
            "random" => Some(MergePrune::Random),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PartitionedBuildParams {
    /// R (the final out-degree), build L, α, passes, seed, threads.
    pub graph: GraphParams,
    pub pq: PqParams,
    pub write: DiskWriteOptions,
    /// Bytes one partition's build may use: its vectors and graph. Sets the
    /// number of partitions when `partitions` is 0.
    pub memory_budget: u64,
    /// Number of partitions; 0 = as few as `memory_budget` allows.
    pub partitions: usize,
    /// Partitions per node (its nearest centroids). At least 2 for the
    /// partition graphs to join up.
    pub overlap: usize,
    /// Out-degree inside a partition, at most R; 0 = R.
    pub partition_degree: usize,
    pub merge: MergePrune,
    /// Where the scratch directory goes; `None` = next to the index.
    pub scratch_dir: Option<PathBuf>,
    /// Called with a line of progress after each step.
    pub progress: Option<fn(&str)>,
    /// Stops the build early, leaving no files.
    pub cancel: Cancel,
}

impl Default for PartitionedBuildParams {
    fn default() -> Self {
        Self {
            graph: GraphParams::default(),
            pq: PqParams::new(32),
            write: DiskWriteOptions::default(),
            memory_budget: 8 << 30,
            partitions: 0,
            overlap: 2,
            partition_degree: 0,
            merge: MergePrune::Pq,
            scratch_dir: None,
            progress: None,
            cancel: Cancel::default(),
        }
    }
}

/// What [`DiskIndex::build_partitioned`](super::DiskIndex::build_partitioned) did.
#[derive(Clone, Debug)]
pub struct PartitionedBuildReport {
    pub write: DiskWriteReport,
    pub rows: usize,
    /// Distinct vectors.
    pub nodes: usize,
    pub partitions: usize,
    pub overlap: usize,
    pub largest_partition: usize,
    /// Most nodes one partition may hold under `memory_budget` (`None` when
    /// the partition count was given).
    pub budget_nodes: Option<usize>,
    pub mean_degree: f64,
    /// Nodes the merged graph did not reach from the entry; all but
    /// `unrepaired` of them then got an in-edge.
    pub unreachable: usize,
    pub unrepaired: usize,
    /// Peak scratch disk use.
    pub scratch_bytes: u64,
    /// Seconds per step.
    pub steps: Vec<(&'static str, f64)>,
}

/// Bytes one node of a partition costs while its graph is built: the vector,
/// its out-list (which grows past R before pruning), and allocator overhead.
pub fn partition_node_bytes(dim: usize, degree: usize) -> u64 {
    (dim * 4 + 8 * degree + 96) as u64
}

/// Rows to nodes. A row identical to an earlier row shares that row's node.
struct NodeMap {
    rows: usize,
    /// (row, earlier identical row), by row.
    repeats: Vec<(u32, u32)>,
    /// The same pairs swapped, by first row.
    by_first: Vec<(u32, u32)>,
    /// For `repeats[j]`: the nodes before it (its row - j).
    nodes_before: Vec<u32>,
}

impl NodeMap {
    fn new(rows: usize, mut repeats: Vec<(u32, u32)>) -> Self {
        repeats.sort_unstable();
        let nodes_before = repeats.iter().enumerate().map(|(j, r)| r.0 - j as u32).collect();
        let mut by_first: Vec<(u32, u32)> = repeats.iter().map(|&(r, f)| (f, r)).collect();
        by_first.sort_unstable();
        Self { rows, repeats, by_first, nodes_before }
    }

    fn nodes(&self) -> usize {
        self.rows - self.repeats.len()
    }

    /// The first row holding `node`'s vector.
    fn row_of(&self, node: u32) -> usize {
        node as usize + self.nodes_before.partition_point(|&b| b <= node)
    }

    /// Later rows identical to `first`, in order.
    fn repeats_of(&self, first: u32) -> impl Iterator<Item = u32> + '_ {
        let start = self.by_first.partition_point(|e| e.0 < first);
        self.by_first[start..].iter().take_while(move |e| e.0 == first).map(|e| e.1)
    }
}

/// Checks rows and prepares them for the metric, as `GraphIndex::build` does.
/// Checks rows and prepares them for the metric (unit length for cosine),
/// unless the source says they already are.
fn prepare(src: &dyn VectorSource, rows: &mut [f32], dim: usize, metric: Metric, threads: usize) -> Result<()> {
    let norm = metric.needs_normalization() && !src.prepared();
    par::map_rows_mut(rows, dim, threads, |_, rows| {
        check_finite(rows)?;
        if norm {
            rows.chunks_exact_mut(dim).for_each(normalize);
        }
        Ok(())
    })
    .into_iter()
    .collect()
}

/// Reads the vectors of `nodes` (ascending) into `out`, prepared. Each
/// thread reads its share, so the many small reads of a sparse partition
/// overlap.
fn read_nodes(
    src: &dyn VectorSource,
    map: &NodeMap,
    metric: Metric,
    nodes: &[u32],
    threads: usize,
    out: &mut Vec<f32>,
) -> Result<()> {
    let dim = src.dim();
    out.clear();
    out.resize(nodes.len() * dim, 0.0);
    par::map_rows_mut(out, dim, threads, |first, rows| gather(src, map, &nodes[first..first + rows.len() / dim], rows))
        .into_iter()
        .collect::<Result<()>>()?;
    prepare(src, out, dim, metric, threads)
}

/// Reads the rows of `nodes` (ascending) into `out`: one read per run of
/// rows whose gaps are at most [`GAP_BYTES`].
fn gather(src: &dyn VectorSource, map: &NodeMap, nodes: &[u32], out: &mut [f32]) -> Result<()> {
    let dim = src.dim();
    let gap = GAP_BYTES / src.row_bytes().max(1);
    let max_run = (RUN_BYTES / (dim * 4)).max(1);
    let mut buf = Vec::new();
    let mut i = 0;
    while i < nodes.len() {
        let first = map.row_of(nodes[i]);
        let (mut last, mut j) = (first, i + 1);
        while j < nodes.len() {
            let r = map.row_of(nodes[j]);
            if r > last + gap + 1 || r + 1 - first > max_run {
                break;
            }
            last = r;
            j += 1;
        }
        if last + 1 - first == j - i {
            // No gaps: read straight into place.
            src.read_rows(first, &mut out[i * dim..j * dim])?;
        } else {
            buf.resize((last + 1 - first) * dim, 0.0);
            src.read_rows(first, &mut buf)?;
            for k in i..j {
                let r = map.row_of(nodes[k]) - first;
                out[k * dim..(k + 1) * dim].copy_from_slice(&buf[r * dim..(r + 1) * dim]);
            }
        }
        i = j;
    }
    Ok(())
}

/// Step 1: finds identical rows.
fn scan(src: &dyn VectorSource, metric: Metric, threads: usize, cancel: &Cancel) -> Result<NodeMap> {
    let (n, dim) = (src.len(), src.dim());
    let batch = (READ_BYTES / (dim * 4)).max(1);
    let mut keys: Vec<(u64, u32)> = Vec::with_capacity(n);
    let mut buf = Vec::new();
    for first in (0..n).step_by(batch) {
        cancel.check()?;
        let count = batch.min(n - first);
        buf.resize(count * dim, 0.0);
        src.read_rows(first, &mut buf)?;
        prepare(src, &mut buf, dim, metric, threads)?;
        let hashes = par::map_ranges(count, threads, |range| {
            range.map(|i| row_hash(&buf[i * dim..(i + 1) * dim])).collect::<Vec<_>>()
        });
        keys.extend(hashes.into_iter().flatten().enumerate().map(|(i, h)| (h, (first + i) as u32)));
    }
    keys.sort_unstable();
    // Rows with equal hashes: compare the vectors. Within a run rows are in
    // order, so each group's first row is its earliest.
    let mut repeats = Vec::new();
    let mut row = vec![0.0f32; dim];
    let mut i = 0;
    while i < n {
        let j = i + keys[i..].partition_point(|k| k.0 == keys[i].0);
        if j > i + 1 {
            let mut firsts: Vec<(u32, Vec<f32>)> = Vec::new();
            for &(_, r) in &keys[i..j] {
                src.read_rows(r as usize, &mut row)?;
                prepare(src, &mut row, dim, metric, 1)?;
                match firsts.iter().find(|(_, v)| *v == row) {
                    Some(&(f, _)) => repeats.push((r, f)),
                    None => firsts.push((r, row.clone())),
                }
            }
        }
        i = j;
    }
    Ok(NodeMap::new(n, repeats))
}

/// Rows as the partitioning sees them: by direction for inner product (the
/// graph links by angle), as they are otherwise.
fn partition_space(metric: Metric, rows: &[f32], dim: usize) -> Cow<'_, [f32]> {
    if metric == Metric::InnerProduct {
        let mut v = rows.to_vec();
        v.chunks_exact_mut(dim).for_each(normalize);
        Cow::Owned(v)
    } else {
        Cow::Borrowed(rows)
    }
}

/// The `out.len()` nearest centroids to `x`, nearest first (ties to the
/// lower index).
fn nearest_centroids(kern: &Kernels, x: &[f32], centroids: &[f32], dists: &mut [f32], out: &mut [u16]) {
    (kern.l2_squared_batch)(x, centroids, dists);
    for slot in 0..out.len() {
        let mut best = None;
        for (j, &d) in dists.iter().enumerate() {
            if out[..slot].contains(&(j as u16)) {
                continue;
            }
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((j, d));
            }
        }
        out[slot] = best.map_or(0, |(j, _)| j as u16);
    }
}

/// Step 2 for the partitions: their number and centroids.
fn plan_partitions(
    sample: &[f32],
    dim: usize,
    metric: Metric,
    nodes: usize,
    budget_nodes: Option<usize>,
    p: &PartitionedBuildParams,
) -> Result<(usize, Vec<f32>)> {
    let s = sample.len() / dim;
    let mut k = match budget_nodes {
        None => p.partitions,
        Some(b) if nodes <= b => 1,
        Some(b) => (p.overlap * nodes).div_ceil(b).max(p.overlap + 1),
    };
    if k == 1 {
        return Ok((1, Vec::new()));
    }
    let space = partition_space(metric, sample, dim);
    let kern = kernels::best();
    for _ in 0..PLAN_TRIES {
        if k > u16::MAX as usize || k * 8 > s {
            return Err(Error::InvalidArgument(format!(
                "{k} partitions need a sample of at least {} vectors (have {s}): raise the memory budget or PQ max_train",
                k * 8
            )));
        }
        let threads = p.cancel.threads(p.graph.threads);
        let km = KMeansParams { k, iters: PARTITION_ITERS, seed: p.graph.seed ^ 0x9a27_17c3, threads };
        let centroids = kmeans::train(&space, dim, &km)?;
        let Some(budget) = budget_nodes else {
            return Ok((k, centroids));
        };
        // Largest partition, estimated from the sample.
        let counts = par::map_ranges(s, threads, |range| {
            let mut counts = vec![0usize; k];
            let mut dists = vec![0.0f32; k];
            let mut near = vec![0u16; p.overlap];
            for i in range {
                nearest_centroids(kern, &space[i * dim..(i + 1) * dim], &centroids, &mut dists, &mut near);
                near.iter().for_each(|&c| counts[c as usize] += 1);
            }
            counts
        });
        let largest = (0..k).map(|c| counts.iter().map(|v| v[c]).sum::<usize>()).max().unwrap_or(0);
        let estimate = (largest as f64 / s as f64 * nodes as f64 * SIZE_MARGIN).ceil() as usize;
        if estimate <= budget {
            return Ok((k, centroids));
        }
        // Grow in steps of at most 25%: the clusters' imbalance changes with
        // k, so scaling k by the overshoot can ask for far more partitions
        // than needed.
        k = ((k as f64 * (estimate as f64 / budget as f64).min(1.25)).ceil() as usize).max(k + 1);
    }
    Err(Error::InvalidArgument("could not split the collection to fit the memory budget".into()))
}

/// Reads one partition's scratch records back in node order.
struct PartReader {
    r: BufReader<File>,
    word: [u8; 4],
}

impl PartReader {
    fn open(path: &Path) -> Result<Self> {
        Ok(Self { r: BufReader::with_capacity(1 << 18, File::open(path)?), word: [0; 4] })
    }

    fn u32(&mut self) -> Result<u32> {
        self.r.read_exact(&mut self.word)?;
        Ok(u32::from_le_bytes(self.word))
    }

    /// Appends node `node`'s (distance, neighbor) pairs to `out`.
    fn read(&mut self, node: u32, max_degree: usize, nodes: usize, out: &mut Vec<(f32, u32)>) -> Result<()> {
        let (got, deg) = (self.u32()?, self.u32()? as usize);
        if got != node || deg > max_degree {
            return Err(Error::Corrupt(format!("partition scratch file: node {got} (expected {node}), degree {deg}")));
        }
        let start = out.len();
        for _ in 0..deg {
            let nb = self.u32()?;
            if nb as usize >= nodes {
                return Err(Error::Corrupt(format!("partition scratch file: neighbor {nb}")));
            }
            out.push((0.0, nb));
        }
        for e in &mut out[start..] {
            e.0 = f32::from_bits(self.u32()?);
        }
        Ok(())
    }
}

/// Context shared by the merge of every node.
struct Merge<'a> {
    mode: MergePrune,
    r: usize,
    alpha: f32,
    seed: u64,
    metric: Metric,
    dim: usize,
    pq: &'a ProductQuantizer,
    codes: &'a [u8],
}

impl Merge<'_> {
    /// Cuts `node`'s joined out-lists (`(distance, neighbor)`, in partition
    /// order) to at most R. A single list is kept as it is.
    fn one(&self, node: u32, joined: &[(f32, u32)], lists: usize, kern: &Kernels, buf: &mut Vec<f32>) -> Vec<u32> {
        if lists == 1 {
            return joined.iter().map(|e| e.1).collect();
        }
        // Drop repeated neighbors, keeping the first; keep the order.
        let mut tagged: Vec<(u32, usize)> = joined.iter().enumerate().map(|(i, e)| (e.1, i)).collect();
        tagged.sort_unstable();
        tagged.dedup_by_key(|t| t.0);
        tagged.sort_unstable_by_key(|t| t.1);
        let mut union: Vec<(f32, u32)> =
            tagged.iter().map(|t| joined[t.1]).filter(|e| e.1 != node).collect();
        if union.len() <= self.r {
            return union.iter().map(|e| e.1).collect();
        }
        let by_distance = |a: &(f32, u32), b: &(f32, u32)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
        match self.mode {
            MergePrune::Random => {
                let mut rng = SplitMix64::new(self.seed ^ (node as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                for i in (1..union.len()).rev() {
                    union.swap(i, rng.below(i as u64 + 1) as usize);
                }
                union.truncate(self.r);
                union.sort_unstable_by(by_distance);
                union.iter().map(|e| e.1).collect()
            }
            MergePrune::Pq => {
                let (dim, m) = (self.dim, self.pq.code_size());
                buf.resize(union.len() * dim, 0.0);
                for (row, &(_, nb)) in buf.chunks_exact_mut(dim).zip(&union) {
                    self.pq.decode(&self.codes[nb as usize * m..][..m], row);
                    if self.metric.needs_normalization() {
                        normalize(row);
                    }
                }
                let points = Points::new(&buf[..], dim, self.metric, 1);
                let mut pool: Vec<(f32, u32)> = union.iter().enumerate().map(|(j, e)| (e.0, j as u32)).collect();
                let picked = robust_prune(&points, kern, u32::MAX, &mut pool, self.alpha, self.r, &mut Vec::new());
                picked.iter().map(|&j| union[j as usize].1).collect()
            }
        }
    }
}

/// The merged graph on scratch disk: per node, its degree then R slots.
struct Adjacency {
    file: File,
    r: usize,
}

impl Adjacency {
    fn stride(&self) -> usize {
        (self.r + 1) * 4
    }

    fn read(&self, node: u32, rec: &mut [u8], list: &mut Vec<u32>) -> Result<()> {
        read_exact_at(&self.file, rec, node as u64 * self.stride() as u64)?;
        parse_adjacency(rec, list);
        Ok(())
    }

    fn write(&self, node: u32, list: &[u32], rec: &mut [u8]) -> Result<()> {
        encode_adjacency(list, rec);
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.write_all_at(rec, node as u64 * self.stride() as u64)?;
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::FileExt;
            let (mut done, at) = (0, node as u64 * self.stride() as u64);
            while done < rec.len() {
                done += self.file.seek_write(&rec[done..], at + done as u64)?;
            }
        }
        Ok(())
    }
}

fn encode_adjacency(list: &[u32], rec: &mut [u8]) {
    rec.fill(0);
    rec[..4].copy_from_slice(&(list.len() as u32).to_le_bytes());
    for (slot, &nb) in rec[4..].chunks_exact_mut(4).zip(list) {
        slot.copy_from_slice(&nb.to_le_bytes());
    }
}

fn parse_adjacency(rec: &[u8], list: &mut Vec<u32>) {
    let (words, _) = rec.as_chunks::<4>();
    let deg = (u32::from_le_bytes(words[0]) as usize).min(words.len() - 1);
    list.clear();
    list.extend(words[1..=deg].iter().map(|w| u32::from_le_bytes(*w)));
}

/// Step 5b: every node the merged graph does not reach from `entry` gets an
/// in-edge, from its nearest reachable out-neighbor with a free slot, or
/// else from the first reachable node with one. Returns the nodes not
/// reached at first, the edges added, and the nodes left unreachable.
fn connect_unreachable(adj: &Adjacency, nodes: usize, entry: u32, cancel: &Cancel) -> Result<(usize, usize, usize)> {
    let mut reached = vec![0u64; nodes.div_ceil(64)];
    let is = |reached: &[u64], v: u32| reached[v as usize / 64] >> (v % 64) & 1 == 1;
    let mut rec = vec![0u8; adj.stride()];
    let mut list = Vec::new();
    let mut stack = Vec::new();
    let mut mark = |reached: &mut [u64], start: u32, rec: &mut [u8], list: &mut Vec<u32>| -> Result<usize> {
        let mut count = 0;
        if is(reached, start) {
            return Ok(0);
        }
        reached[start as usize / 64] |= 1 << (start % 64);
        stack.push(start);
        while let Some(v) = stack.pop() {
            count += 1;
            if count % 4096 == 0 {
                cancel.check()?;
            }
            adj.read(v, rec, list)?;
            for &nb in list.iter() {
                if !is(reached, nb) {
                    reached[nb as usize / 64] |= 1 << (nb % 64);
                    stack.push(nb);
                }
            }
        }
        Ok(count)
    };
    let found = mark(&mut reached, entry, &mut rec, &mut list)?;
    let unreachable = nodes - found;
    let (mut added, mut unrepaired) = (0, 0);
    let mut cursor = 0u32;
    let mut other = Vec::new();
    for u in 0..nodes as u32 {
        if unreachable == 0 {
            break;
        }
        if is(&reached, u) {
            continue;
        }
        adj.read(u, &mut rec, &mut list)?;
        let mut from = None;
        for &nb in &list {
            if is(&reached, nb) {
                adj.read(nb, &mut rec, &mut other)?;
                if other.len() < adj.r {
                    from = Some(nb);
                    break;
                }
            }
        }
        if from.is_none() {
            while (cursor as usize) < nodes {
                if is(&reached, cursor) {
                    adj.read(cursor, &mut rec, &mut other)?;
                    if other.len() < adj.r {
                        from = Some(cursor);
                        break;
                    }
                }
                cursor += 1;
            }
        }
        let Some(from) = from else {
            unrepaired += 1;
            continue;
        };
        adj.read(from, &mut rec, &mut other)?;
        other.push(u);
        adj.write(from, &other, &mut rec)?;
        added += 1;
        mark(&mut reached, u, &mut rec, &mut list)?;
    }
    Ok((unreachable, added, unrepaired))
}

/// Removes the scratch directory when the build ends, however it ends.
struct ScratchDir(PathBuf);

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub(crate) fn build(
    src: &dyn VectorSource,
    ids: Option<&[u64]>,
    metric: Metric,
    path: &Path,
    p: &PartitionedBuildParams,
) -> Result<PartitionedBuildReport> {
    let (n, dim) = (src.len(), src.dim());
    let threads = || p.cancel.threads(p.graph.threads);
    let r = p.graph.max_degree;
    let part_r = if p.partition_degree == 0 { r } else { p.partition_degree };
    check_dim(dim)?;
    p.graph.validate()?;
    if p.overlap == 0 {
        return Err(Error::InvalidArgument("overlap must be at least 1".into()));
    }
    if part_r > r {
        return Err(Error::InvalidArgument(format!("partition degree {part_r} is above R = {r}")));
    }
    if p.partitions == 0 && p.memory_budget == 0 {
        return Err(Error::InvalidArgument("give a memory budget or a number of partitions".into()));
    }
    if n == 0 {
        return Err(Error::InvalidArgument("cannot write an empty SSD index".into()));
    }
    if n >= u32::MAX as usize {
        return Err(Error::InvalidArgument("a graph holds fewer than 2^32 - 1 vectors".into()));
    }
    if let Some(ids) = ids {
        if ids.len() != n {
            return Err(Error::InvalidArgument(format!("{} ids for {n} vectors", ids.len())));
        }
        let mut sorted = ids.to_vec();
        sorted.sort_unstable();
        if let Some(w) = sorted.windows(2).find(|w| w[0] == w[1]) {
            return Err(Error::DuplicateId(w[0]));
        }
    }
    let log = |msg: String| {
        if let Some(f) = p.progress {
            f(&msg);
        }
    };
    let mut steps = Vec::new();
    let mut t = Instant::now();
    let mut step = |name: &'static str, t: &mut Instant| {
        steps.push((name, t.elapsed().as_secs_f64()));
        *t = Instant::now();
    };

    // 1. Scan.
    let map = scan(src, metric, threads(), &p.cancel)?;
    let nodes = map.nodes();
    log(format!("scan: {n} rows, {nodes} distinct"));
    step("scan", &mut t);

    // 2. Train PQ and the partitions on one sample (the rows PQ training
    //    would pick from all nodes, so one partition matches the in-memory build).
    let sample_size = p.pq.max_train.max(KSUB);
    let sample_nodes: Vec<u32> = if nodes > sample_size {
        let mut s = SplitMix64::new(p.pq.seed).sample_indices(nodes, sample_size);
        s.sort_unstable();
        s.into_iter().map(|v| v as u32).collect()
    } else {
        (0..nodes as u32).collect()
    };
    let mut sample = Vec::new();
    read_nodes(src, &map, metric, &sample_nodes, threads(), &mut sample)?;
    drop(sample_nodes);
    let pq = ProductQuantizer::train_until(dim, &sample, &p.pq, &p.cancel)?;
    let budget_nodes = (p.partitions == 0).then(|| (p.memory_budget / partition_node_bytes(dim, part_r)) as usize);
    if budget_nodes.is_some_and(|b| b < 1000) {
        return Err(Error::InvalidArgument(format!(
            "a memory budget of {} bytes holds under 1000 vectors of dimension {dim}",
            p.memory_budget
        )));
    }
    let (k, centroids) = plan_partitions(&sample, dim, metric, nodes, budget_nodes, p)?;
    drop(sample);
    let overlap = p.overlap.min(k);
    log(format!("train: PQ {} B/vector; {k} partitions, overlap {overlap}", pq.code_size()));
    step("train", &mut t);

    // Scratch space: partition out-lists (at most), the merged graph, and the index.
    let shape = Shape { metric, dim, max_degree: r, nodes, ids: n, pq_m: pq.code_size() };
    let (_, index_bytes) = header(&shape, &p.write)?;
    let scratch_need = (overlap * nodes) as u64 * (8 + 8 * part_r as u64) + nodes as u64 * (r as u64 + 1) * 4;
    let out_dir = match path.parent() {
        Some(d) if !d.as_os_str().is_empty() => d.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let scratch_root = p.scratch_dir.clone().unwrap_or_else(|| out_dir.clone());
    let same_fs = same_device(&scratch_root, &out_dir);
    for (dir, need) in [
        (&scratch_root, scratch_need + if same_fs { index_bytes } else { 0 }),
        (&out_dir, index_bytes),
    ] {
        if let Some(free) = free_space(dir) {
            if free < need {
                return Err(Error::InvalidArgument(format!(
                    "{} has {:.1} GB free; the build needs up to {:.1} GB there",
                    dir.display(),
                    free as f64 / 1e9,
                    need as f64 / 1e9
                )));
            }
        }
    }
    let name = path.file_name().map_or("index".into(), |f| f.to_string_lossy().into_owned());
    let scratch = ScratchDir(scratch_root.join(format!("{name}.build-{}", std::process::id())));
    let _ = fs::remove_dir_all(&scratch.0);
    fs::create_dir_all(&scratch.0)?;
    let part_path = |s: usize| scratch.0.join(format!("part-{s}"));

    // 3. Encode: codes, mean, partitions.
    let m = pq.code_size();
    let mut codes = vec![0u8; nodes * m];
    let mut assign = vec![0u16; if k > 1 { nodes * overlap } else { 0 }];
    let mut mean_sum = vec![0.0f64; dim];
    let batch = ((READ_BYTES / (dim * 4)) / MEAN_CHUNK).max(1) * MEAN_CHUNK;
    let mut buf = Vec::new();
    let mut batch_nodes: Vec<u32> = Vec::with_capacity(batch);
    for v0 in (0..nodes).step_by(batch) {
        p.cancel.check()?;
        let v1 = (v0 + batch).min(nodes);
        batch_nodes.clear();
        batch_nodes.extend(v0 as u32..v1 as u32);
        read_nodes(src, &map, metric, &batch_nodes, threads(), &mut buf)?;
        let rows = &buf;
        let sums = par::map_ranges((v1 - v0).div_ceil(MEAN_CHUNK), threads(), |range| {
            range
                .map(|c| {
                    let (a, b) = (c * MEAN_CHUNK, ((c + 1) * MEAN_CHUNK).min(v1 - v0));
                    chunk_sum(rows[a * dim..b * dim].chunks_exact(dim), dim)
                })
                .collect::<Vec<_>>()
        });
        for s in sums.iter().flatten() {
            for (acc, v) in mean_sum.iter_mut().zip(s) {
                *acc += v;
            }
        }
        par::map_rows_mut(&mut codes[v0 * m..v1 * m], m, threads(), |first, out| {
            for (i, code) in out.chunks_exact_mut(m).enumerate() {
                pq.encode(&rows[(first + i) * dim..(first + i + 1) * dim], code);
            }
        });
        if k > 1 {
            par::map_rows_mut(&mut assign[v0 * overlap..v1 * overlap], overlap, threads(), |first, out| {
                let kern = kernels::best();
                let mut dists = vec![0.0f32; k];
                for (i, near) in out.chunks_exact_mut(overlap).enumerate() {
                    let row = &rows[(first + i) * dim..(first + i + 1) * dim];
                    let x = partition_space(metric, row, dim);
                    nearest_centroids(kern, &x, &centroids, &mut dists, near);
                }
            });
        }
    }
    let mean: Vec<f32> = mean_sum.iter().map(|&s| (s / nodes as f64) as f32).collect();
    let mean_inv_norm = inv_norm(&mean);
    log(format!("encode: {nodes} nodes"));
    step("encode", &mut t);

    // 4. Partition graphs. The entry is the node nearest the mean, as in the
    //    in-memory build; each partition checks its own nodes.
    let mut entry: Option<(f32, u32)> = None;
    let mut largest = 0;
    let mut part_bytes = 0u64;
    let by_distance = |a: &(f32, u32), b: &(f32, u32)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
    for s in 0..k {
        p.cancel.check()?;
        let ts = Instant::now();
        let members: Vec<u32> = if k == 1 {
            (0..nodes as u32).collect()
        } else {
            (0..nodes as u32).filter(|&v| assign[v as usize * overlap..][..overlap].contains(&(s as u16))).collect()
        };
        largest = largest.max(members.len());
        if let Some(b) = budget_nodes.filter(|&b| members.len() > b) {
            // The plan estimates sizes from a sample: say when it missed.
            log(format!("partition {}/{k}: {} nodes, over the budget of {b} nodes", s + 1, members.len()));
        }
        let mut w = BufWriter::with_capacity(1 << 20, File::create(part_path(s))?);
        if !members.is_empty() {
            read_nodes(src, &map, metric, &members, threads(), &mut buf)?;
            let points = Points::new(&buf, dim, metric, threads());
            let gp = GraphParams { max_degree: part_r, seed: p.graph.seed.wrapping_add(s as u64), ..p.graph.clone() };
            let (lists, _) = build_graph(&points, &gp, &p.cancel)?;
            let best = par::map_ranges(members.len(), threads(), |range| {
                let kern = kernels::best();
                range.map(|i| (points.dist_to(kern, &mean, mean_inv_norm, i as u32), members[i])).min_by(by_distance)
            });
            entry = entry.into_iter().chain(best.into_iter().flatten()).min_by(by_distance);
            for b0 in (0..members.len()).step_by(BATCH) {
                let b1 = (b0 + BATCH).min(members.len());
                let chunks = par::map_ranges(b1 - b0, threads(), |range| {
                    let kern = kernels::best();
                    let mut bytes = Vec::new();
                    for u in range.map(|i| b0 + i) {
                        let list = &lists[u];
                        bytes.extend(members[u].to_le_bytes());
                        bytes.extend((list.len() as u32).to_le_bytes());
                        for &nb in list {
                            bytes.extend(members[nb as usize].to_le_bytes());
                        }
                        for &nb in list {
                            bytes.extend(points.dist(kern, u as u32, nb).to_bits().to_le_bytes());
                        }
                    }
                    bytes
                });
                for c in &chunks {
                    w.write_all(c)?;
                    part_bytes += c.len() as u64;
                }
            }
        }
        w.into_inner().map_err(|e| e.into_error())?;
        log(format!("partition {}/{k}: {} nodes in {:.1} s", s + 1, members.len(), ts.elapsed().as_secs_f64()));
    }
    let entry = entry.map_or(0, |e| e.1);
    step("partitions", &mut t);

    // 5. Merge into the scratch adjacency file, then repair reachability.
    let adj_path = scratch.0.join("merged");
    let merge = Merge {
        mode: p.merge,
        r,
        alpha: p.graph.alpha,
        seed: p.graph.seed,
        metric,
        dim,
        pq: &pq,
        codes: &codes,
    };
    let mut readers = (0..k).map(|s| PartReader::open(&part_path(s))).collect::<Result<Vec<_>>>()?;
    let adj_file = fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&adj_path)?;
    let mut adj_out = BufWriter::with_capacity(1 << 20, adj_file);
    let mut edges = 0u64;
    let mut joined: Vec<Vec<(f32, u32)>> = Vec::new();
    for v0 in (0..nodes).step_by(BATCH) {
        p.cancel.check()?;
        let v1 = (v0 + BATCH).min(nodes);
        joined.resize_with(v1 - v0, Vec::new);
        for v in v0..v1 {
            let out = &mut joined[v - v0];
            out.clear();
            let parts: &[u16] = if k == 1 { &[0] } else { &assign[v * overlap..(v + 1) * overlap] };
            for &s in parts {
                readers[s as usize].read(v as u32, part_r, nodes, out)?;
            }
        }
        let lists = par::map_ranges(v1 - v0, threads(), |range| {
            let kern = kernels::best();
            let mut buf = Vec::new();
            range
                .map(|i| merge.one((v0 + i) as u32, &joined[i], if k == 1 { 1 } else { overlap }, kern, &mut buf))
                .collect::<Vec<_>>()
        });
        let mut rec = vec![0u8; (r + 1) * 4];
        for list in lists.iter().flatten() {
            edges += list.len() as u64;
            encode_adjacency(list, &mut rec);
            adj_out.write_all(&rec)?;
        }
    }
    drop(joined);
    let adj_file = adj_out.into_inner().map_err(|e| e.into_error())?;
    drop(readers);
    for s in 0..k {
        fs::remove_file(part_path(s))?;
    }
    let adj = Adjacency { file: adj_file, r };
    let (unreachable, added, unrepaired) = connect_unreachable(&adj, nodes, entry, &p.cancel)?;
    edges += added as u64;
    log(format!(
        "merge: mean degree {:.1}; {unreachable} nodes unreachable, {added} edges added, {unrepaired} still unreachable",
        edges as f64 / nodes as f64
    ));
    step("merge", &mut t);

    // 6. Write the index.
    let mut w = IndexWriter::create(path, &shape, &p.write)?;
    let mut adj_in = BufReader::with_capacity(1 << 20, File::open(&adj_path)?);
    let mut rec = vec![0u8; adj.stride()];
    let mut list = Vec::with_capacity(r);
    for v0 in (0..nodes).step_by(batch) {
        p.cancel.check()?;
        let v1 = (v0 + batch).min(nodes);
        batch_nodes.clear();
        batch_nodes.extend(v0 as u32..v1 as u32);
        read_nodes(src, &map, metric, &batch_nodes, threads(), &mut buf)?;
        for row in buf.chunks_exact(dim) {
            adj_in.read_exact(&mut rec)?;
            parse_adjacency(&rec, &mut list);
            w.push(row, &list)?;
        }
    }
    let id = |row: u32| ids.map_or(row as u64, |ids| ids[row as usize]);
    let report = w.finish(entry, pq.codebooks(), &codes, &mut |node, out| {
        let row = map.row_of(node) as u32;
        out.push(id(row));
        out.extend(map.repeats_of(row).map(id));
    })?;
    step("write", &mut t);
    drop(scratch);

    Ok(PartitionedBuildReport {
        write: report,
        rows: n,
        nodes,
        partitions: k,
        overlap,
        largest_partition: largest,
        budget_nodes,
        mean_degree: edges as f64 / nodes as f64,
        unreachable,
        unrepaired,
        scratch_bytes: part_bytes + nodes as u64 * adj.stride() as u64,
        steps,
    })
}

/// Whether two directories are on the same file system (true if unknown).
fn same_device(a: &Path, b: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        match (fs::metadata(a), fs::metadata(b)) {
            (Ok(x), Ok(y)) => x.dev() == y.dev(),
            _ => true,
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (a, b);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adjacency(dir: &Path, r: usize, lists: &[&[u32]]) -> Adjacency {
        let path = dir.join("adj");
        let file = fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path).unwrap();
        let adj = Adjacency { file, r };
        let mut rec = vec![0u8; adj.stride()];
        for (v, list) in lists.iter().enumerate() {
            adj.write(v as u32, list, &mut rec).unwrap();
        }
        adj
    }

    fn list(adj: &Adjacency, v: u32) -> Vec<u32> {
        let (mut rec, mut out) = (vec![0u8; adj.stride()], Vec::new());
        adj.read(v, &mut rec, &mut out).unwrap();
        out
    }

    #[test]
    fn unreachable_nodes_get_an_in_edge() {
        let dir = super::super::test_dir("repair");
        // The entry 0 reaches {0, 1, 2}. Node 3 points at 2, which has a free
        // slot; 4 is reached through 3; 5 has no out-edges and takes the
        // first reachable node with room (0).
        let adj = adjacency(&dir, 2, &[&[1], &[2], &[1], &[4, 2], &[3], &[]]);
        assert_eq!(connect_unreachable(&adj, 6, 0, &Cancel::default()).unwrap(), (3, 2, 0));
        assert_eq!(list(&adj, 2), vec![1, 3]);
        assert_eq!(list(&adj, 0), vec![1, 5]);
        assert_eq!(list(&adj, 4), vec![3]);
        // Every reachable node full: node 2 cannot get an in-edge.
        let adj = adjacency(&dir, 1, &[&[1], &[0], &[]]);
        assert_eq!(connect_unreachable(&adj, 3, 0, &Cancel::default()).unwrap(), (1, 0, 1));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn node_map_skips_repeated_rows() {
        // Rows 2, 3 and 6 repeat rows 0, 0 and 4.
        let map = NodeMap::new(8, vec![(6, 4), (2, 0), (3, 0)]);
        assert_eq!(map.nodes(), 5);
        let rows: Vec<usize> = (0..5).map(|v| map.row_of(v)).collect();
        assert_eq!(rows, vec![0, 1, 4, 5, 7]);
        assert_eq!(map.repeats_of(0).collect::<Vec<_>>(), vec![2, 3]);
        assert_eq!(map.repeats_of(4).collect::<Vec<_>>(), vec![6]);
        assert_eq!(map.repeats_of(1).count(), 0);
    }
}
