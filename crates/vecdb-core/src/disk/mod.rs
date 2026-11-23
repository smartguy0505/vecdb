//! SSD index (the DiskANN design): PQ codes in RAM steer a beam search over
//! a Vamana graph stored on SSD, and full vectors give the exact distances.
//!
//! RAM holds only the PQ codebooks, the PQ codes (`m` bytes per vector) and
//! the caller ids (8 bytes per vector), plus an optional cache of the nodes
//! nearest the entry point. Everything else is read on demand with direct
//! I/O, one block per expanded node.
//!
//! Search ([`DiskIndex::search_with`]):
//! 1. Keep a candidate list of the `l` best nodes by PQ distance.
//! 2. Each round, take up to `beam_width` unexpanded candidates from the
//!    front of the list and read their blocks in one batch (io_uring keeps
//!    them in flight together, so a round costs about one SSD latency).
//! 3. Add each block's neighbors to the list by PQ distance.
//! 4. Stop when every candidate in the list has been expanded.
//!
//! The layout ([`Layout`]) decides where exact distances come from:
//! - **Inline**: each block also holds the node's full vector, so every
//!   expanded node is scored exactly at no extra I/O. With `beam_width = 1`
//!   this is exactly [`Navigation::Pq`](crate::graph::Navigation::Pq) of
//!   the in-memory graph.
//! - **Separate**: blocks hold only neighbor lists (15 per sector at
//!   degree 64), and the `rerank` best expanded nodes by PQ distance get
//!   their vectors read at the end. Far fewer bytes per hop, which pays off
//!   once vectors span several sectors; see [`INLINE_MAX_RECORD`].
//!
//! Every record read is checked against its CRC-32C and its neighbor ids
//! against the node count, so a damaged file gives [`Error::Corrupt`]
//! rather than a panic or a silent wrong answer.

mod build;
mod format;
mod io;
mod write;

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::crc32c::{Crc32c, crc32c};
use crate::distance::{Metric, check_finite, normalize};
use crate::error::{Error, Result};
use crate::graph::GraphIndex;
use crate::graph::search::{Candidate, order};
use crate::graph::visited::VisitedSet;
use crate::kernels::{self, Kernels};
use crate::quant::pq::DistanceTable;
use crate::quant::{Codec, ProductQuantizer};
use crate::topk::{SearchResult, TopK};
use format::{CODEBOOKS, CODES, GRAPH, Geometry, Header, ID_START, IDS, VECTORS, decode_vector};
use io::{DEPTH, DataFile, Reader};
pub(crate) use format::codec_from_code;
pub(crate) use format::codec_code;
pub(crate) use io::{drop_cache, free_space, read_exact_at};

pub use format::{INLINE_MAX_RECORD, Layout, SECTOR};
pub use io::IoMode;
pub use build::{MergePrune, PartitionedBuildParams, PartitionedBuildReport, partition_node_bytes};
pub use write::{DiskWriteOptions, DiskWriteReport};

/// Largest `beam_width`: reads one search keeps in flight together.
pub const MAX_BEAM_WIDTH: usize = DEPTH;

/// How to open an SSD index.
#[derive(Clone, Debug)]
pub struct DiskOpenOptions {
    /// Bypass the page cache (`O_DIRECT`) where the filesystem allows it.
    /// [`DiskIndex::is_direct`] reports whether it took effect.
    pub direct: bool,
    pub io: IoMode,
}

impl Default for DiskOpenOptions {
    fn default() -> Self {
        Self { direct: true, io: IoMode::Auto }
    }
}

/// Search-time knobs.
#[derive(Clone, Debug)]
pub struct DiskSearchParams {
    /// Candidate list size: the recall/latency knob, as in the in-memory
    /// graph. Raised to at least `k`.
    pub l: usize,
    /// Reads issued together per round, 1..=[`MAX_BEAM_WIDTH`]. Wider beams
    /// finish in fewer rounds (lower latency) but expand more nodes.
    pub beam_width: usize,
    /// Separate layout: how many of the best expanded nodes (by PQ distance)
    /// get their full vectors read; 0 = all of them. Raised to at least `k`.
    /// Ignored by the inline layout, which scores every expanded node.
    pub rerank: usize,
}

impl Default for DiskSearchParams {
    fn default() -> Self {
        Self { l: 100, beam_width: 4, rerank: 0 }
    }
}

/// Counters for one search.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiskSearchStats {
    /// Nodes whose neighbor lists were used (cached or read).
    pub expanded: usize,
    /// Expanded nodes served from the node cache.
    pub cache_hits: usize,
    /// Blocks read from the file (graph blocks plus re-rank vector blocks).
    pub reads: usize,
    pub read_bytes: usize,
    /// Batches waited on. Each costs about one SSD read latency.
    pub rounds: usize,
    pub pq_distances: usize,
    pub exact_distances: usize,
    /// Expanded nodes with at least one id the filter admits (all of them
    /// in an unfiltered search).
    pub admitted: usize,
}

/// Per-thread search state: the I/O ring, read buffers and candidate lists.
/// Create one per thread with [`DiskIndex::context`] and reuse it.
pub struct SearchContext {
    reader: Reader,
    mode: IoMode,
    visited: VisitedSet,
    list: Vec<Candidate>,
    /// This round's nodes, with their PQ distances.
    picked: Vec<(f32, u32)>,
    /// Slot of each picked node in the reader, or `usize::MAX` if cached.
    slots: Vec<usize>,
    reqs: Vec<(u64, usize)>,
    fresh: Vec<u32>,
    expanded: Vec<Expanded>,
    vector: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
struct Expanded {
    pq: f32,
    node: u32,
    /// Inline layout: the exact distance. Separate: unused.
    exact: f32,
    /// Separate layout: CRC-32C of the node's vector block.
    vector_crc: u32,
}

impl SearchContext {
    pub fn new(mode: IoMode) -> Result<Self> {
        Ok(Self {
            reader: Reader::new(mode)?,
            mode,
            visited: VisitedSet::new(),
            list: Vec::new(),
            picked: Vec::new(),
            slots: Vec::new(),
            reqs: Vec::new(),
            fresh: Vec::new(),
            expanded: Vec::new(),
            vector: Vec::new(),
        })
    }

    /// The I/O backend in use: "io_uring" or "pread".
    pub fn io_backend(&self) -> &'static str {
        self.reader.kind()
    }
}

thread_local! {
    static CONTEXT: RefCell<Option<SearchContext>> = const { RefCell::new(None) };
}

/// An SSD index opened for search. Immutable and `Sync`: share one across
/// threads, with one [`SearchContext`] per thread.
pub struct DiskIndex {
    path: PathBuf,
    header: Header,
    geo: Geometry,
    file: DataFile,
    file_bytes: u64,
    pq: ProductQuantizer,
    codes: Vec<u8>,
    /// Node `i` holds `ids[id_start[i]..id_start[i + 1]]`; empty when every
    /// node has exactly one id (then node `i` holds `ids[i]`).
    id_start: Vec<u64>,
    ids: Vec<u64>,
    /// Records of hot nodes, already verified.
    cache: HashMap<u32, Box<[u8]>>,
    io_mode: IoMode,
}

/// A record split into its fields (checksum and bounds already checked).
struct Record<'a> {
    vector: &'a [u8],
    neighbors: &'a [u8],
    vector_crc: u32,
}

impl Record<'_> {
    fn neighbors(&self) -> impl Iterator<Item = u32> + '_ {
        self.neighbors.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b))
    }
}

impl DiskIndex {
    /// Writes `graph` (with PQ trained) as an SSD index at `path`, replacing
    /// any file there atomically (write to `<path>.tmp`, fsync, rename).
    pub fn write(graph: &GraphIndex, path: impl AsRef<Path>, opts: &DiskWriteOptions) -> Result<DiskWriteReport> {
        write::write(graph, path.as_ref(), opts)
    }

    /// Builds an SSD index straight from `source` without holding all its
    /// vectors in RAM: the collection is split into overlapping partitions
    /// whose graphs are built one at a time and merged (see
    /// [`PartitionedBuildParams`]). Row `i` gets id `ids[i]`, or `i` when
    /// `ids` is `None`.
    pub fn build_partitioned(
        source: &dyn crate::dataset::VectorSource,
        ids: Option<&[u64]>,
        metric: Metric,
        path: impl AsRef<Path>,
        params: &PartitionedBuildParams,
    ) -> Result<PartitionedBuildReport> {
        build::build(source, ids, metric, path.as_ref(), params)
    }

    /// Opens an index: validates the header, loads and checks the RAM
    /// sections, and opens the file for block reads.
    pub fn open(path: impl AsRef<Path>, opts: &DiskOpenOptions) -> Result<Self> {
        let path = path.as_ref();
        let meta = File::open(path)?;
        let file_bytes = meta.metadata()?.len();
        let mut head = vec![0u8; (file_bytes as usize).min(SECTOR)];
        read_exact_at(&meta, &mut head, 0)?;
        let header = Header::decode(&head, file_bytes)?;
        let geo = header.geometry();

        let codebook_bytes = read_section(&meta, &header, CODEBOOKS)?;
        let codebooks = codebook_bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        drop(codebook_bytes);
        let pq = ProductQuantizer::from_parts(header.dim, header.pq_m, codebooks)
            .map_err(|e| Error::Corrupt(format!("SSD index PQ codebooks: {e}")))?;
        let codes = read_section(&meta, &header, CODES)?;
        let id_start = read_u64s(&meta, &header, ID_START)?;
        let ids = read_u64s(&meta, &header, IDS)?;
        if !id_start.is_empty() {
            let ok = id_start[0] == 0
                && id_start.last() == Some(&header.ids)
                && id_start.windows(2).all(|w| w[0] < w[1]);
            if !ok {
                return Err(Error::Corrupt("SSD index id table is inconsistent".into()));
            }
        }
        drop_cache(&meta);
        drop(meta);

        let file = DataFile::open(path, opts.direct)?;
        Ok(Self {
            path: path.to_path_buf(),
            header,
            geo,
            file,
            file_bytes,
            pq,
            codes,
            id_start,
            ids,
            cache: HashMap::new(),
            io_mode: opts.io,
        })
    }

    pub fn metric(&self) -> Metric {
        self.header.metric
    }

    pub fn dim(&self) -> usize {
        self.header.dim
    }

    pub fn layout(&self) -> Layout {
        self.header.layout
    }

    pub fn codec(&self) -> Codec {
        self.header.codec
    }

    /// Number of stored vectors (ids).
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Graph nodes (identical vectors share one).
    pub fn nodes(&self) -> usize {
        self.header.nodes as usize
    }

    pub fn max_degree(&self) -> usize {
        self.header.max_degree
    }

    pub fn pq_code_size(&self) -> usize {
        self.header.pq_m
    }

    /// Whether block reads bypass the page cache.
    pub fn is_direct(&self) -> bool {
        self.file.is_direct()
    }

    pub fn io_mode(&self) -> IoMode {
        self.io_mode
    }

    pub fn file_bytes(&self) -> u64 {
        self.file_bytes
    }

    /// Bytes of one graph block read per expanded node.
    pub fn block_bytes(&self) -> usize {
        self.geo.block_bytes
    }

    /// Bytes of one vector block read per re-ranked node (separate layout; 0 inline).
    pub fn vector_block_bytes(&self) -> usize {
        self.geo.vector_block_bytes
    }

    /// RAM held by the index: (codebooks + codes + ids, node cache).
    pub fn ram_bytes(&self) -> (usize, usize) {
        let base = self.pq.codebooks().len() * 4 + self.codes.len() + (self.ids.len() + self.id_start.len()) * 8;
        let cache = self.cache.values().map(|r| r.len() + 32).sum();
        (base, cache)
    }

    /// Nodes currently in the cache.
    pub fn cached_nodes(&self) -> usize {
        self.cache.len()
    }

    /// The ids stored at `node`.
    pub fn ids_of(&self, node: u32) -> &[u64] {
        &self.ids[self.positions_of(node)]
    }

    /// Positions of `node`'s ids in the id table, `0..len()`. Every id has
    /// its own position, also when identical vectors share a node, so
    /// deletes and filters are kept per position.
    pub fn positions_of(&self, node: u32) -> std::ops::Range<usize> {
        let n = node as usize;
        if self.id_start.is_empty() { n..n + 1 } else { self.id_start[n] as usize..self.id_start[n + 1] as usize }
    }

    /// The id at position `pos`.
    pub fn id_at(&self, pos: usize) -> u64 {
        self.ids[pos]
    }

    /// The node that holds position `pos`.
    pub fn node_of(&self, pos: usize) -> u32 {
        if self.id_start.is_empty() { pos as u32 } else { (self.id_start.partition_point(|&s| s as usize <= pos) - 1) as u32 }
    }

    /// Reads the stored vectors of nodes `first..first + out.len() / dim()`
    /// from `file` (this index's file, opened for buffered reads), checking
    /// each record. Vectors come back decoded, as prepared at build time
    /// (unit length for cosine, rounded for f16 / bf16).
    pub fn read_node_vectors(&self, file: &File, first: u32, out: &mut [f32]) -> Result<()> {
        let dim = self.dim();
        let count = out.len() / dim;
        if out.len() % dim != 0 || first as usize + count > self.nodes() {
            return Err(Error::InvalidArgument("node range out of bounds".into()));
        }
        let (rpb, bb) = (self.geo.records_per_block, self.geo.block_bytes);
        let mut buf = Vec::new();
        let mut vbuf = Vec::new();
        let mut done = 0;
        while done < count {
            let node = first as usize + done;
            // Whole blocks covering up to about 1 MB of records.
            let block = node / rpb;
            let last_block = (node + count - done - 1) / rpb;
            let blocks = (last_block + 1 - block).min(((1 << 20) / bb).max(1));
            buf.resize(blocks * bb, 0);
            read_exact_at(file, &mut buf, self.header.sections[GRAPH].offset + (block * bb) as u64)?;
            let end = ((block + blocks) * rpb).min(first as usize + count);
            if self.layout() == Layout::Separate {
                let vbb = self.geo.vector_block_bytes;
                vbuf.resize((end - node) * vbb, 0);
                read_exact_at(file, &mut vbuf, self.header.sections[VECTORS].offset + (node * vbb) as u64)?;
            }
            for v in node..end {
                let at = (v / rpb - block) * bb + (v % rpb) * self.geo.record_len;
                let r = self.parse(v as u32, &buf[at..at + self.geo.record_len])?;
                let dst = &mut out[(v - first as usize) * dim..][..dim];
                match self.layout() {
                    Layout::Inline => decode_vector(self.codec(), r.vector, dst),
                    Layout::Separate => {
                        let vbb = self.geo.vector_block_bytes;
                        let bytes = &vbuf[(v - node) * vbb..][..self.geo.vector_bytes];
                        if crc32c(bytes) != r.vector_crc {
                            return Err(Error::Corrupt(format!("vector of node {v} failed its checksum")));
                        }
                        decode_vector(self.codec(), bytes, dst);
                    }
                }
            }
            done = end - first as usize;
        }
        Ok(())
    }

    /// A search context for this index's I/O mode.
    pub fn context(&self) -> Result<SearchContext> {
        SearchContext::new(self.io_mode)
    }

    /// Keeps the records of `count` nodes in RAM, chosen breadth-first from
    /// the entry point: every search starts there, so its first rounds then
    /// need no reads. Replaces any previous cache; returns the nodes cached.
    pub fn cache_hot_nodes(&mut self, count: usize) -> Result<usize> {
        self.cache.clear();
        let count = count.min(self.nodes());
        let mut ctx = self.context()?;
        ctx.reader.reserve(self.geo.block_bytes);
        let mut cache = HashMap::with_capacity(count);
        let mut seen = HashSet::from([self.header.entry]);
        let mut queue = VecDeque::from([self.header.entry]);
        let mut batch = Vec::with_capacity(DEPTH);
        while cache.len() < count && !queue.is_empty() {
            batch.clear();
            while batch.len() < DEPTH.min(count - cache.len()) {
                match queue.pop_front() {
                    Some(v) => batch.push(v),
                    None => break,
                }
            }
            ctx.reqs.clear();
            ctx.reqs.extend(batch.iter().map(|&v| (self.block_offset(v), self.geo.block_bytes)));
            ctx.reader.read(&self.file, &ctx.reqs)?;
            for (j, &node) in batch.iter().enumerate() {
                let rec = self.record_in(ctx.reader.slot(j), node);
                for nb in self.parse(node, rec)?.neighbors() {
                    if seen.insert(nb) {
                        queue.push_back(nb);
                    }
                }
                cache.insert(node, rec.into());
            }
        }
        self.cache = cache;
        Ok(self.cache.len())
    }

    /// Like [`search_with`](Self::search_with), with a context kept per thread.
    pub fn search(&self, query: &[f32], k: usize, params: &DiskSearchParams) -> Result<Vec<SearchResult>> {
        Ok(self.search_filtered(query, k, params, &|_| true)?.0)
    }

    /// Like [`search_filtered_with`](Self::search_filtered_with), with a
    /// context kept per thread. One context serves every index the thread
    /// searches: its buffers grow to the largest.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        params: &DiskSearchParams,
        admit: &dyn Fn(usize) -> bool,
    ) -> Result<(Vec<SearchResult>, DiskSearchStats)> {
        CONTEXT.with_borrow_mut(|slot| {
            if slot.as_ref().is_none_or(|c| c.mode != self.io_mode) {
                *slot = Some(self.context()?);
            }
            let ctx = slot.as_mut().expect("set above");
            let result = self.search_filtered_with(ctx, query, k, params, admit);
            if matches!(result, Err(Error::Io(_))) {
                // Start the next search with a fresh ring and buffers.
                *slot = None;
            }
            result
        })
    }

    /// The `k` nearest vectors found, closest first, with exact distances
    /// (exact for the stored vectors; f16/bf16 storage rounds them).
    pub fn search_with(
        &self,
        ctx: &mut SearchContext,
        query: &[f32],
        k: usize,
        params: &DiskSearchParams,
    ) -> Result<(Vec<SearchResult>, DiskSearchStats)> {
        self.search_filtered_with(ctx, query, k, params, &|_| true)
    }

    /// Like [`search_with`](Self::search_with), returning only ids whose
    /// position (see [`positions_of`](Self::positions_of)) `admit` accepts.
    /// The walk still goes through nodes it rejects, so a filter that
    /// rejects many nodes needs a larger `l` to find `k` results.
    pub fn search_filtered_with(
        &self,
        ctx: &mut SearchContext,
        query: &[f32],
        k: usize,
        params: &DiskSearchParams,
        admit: &dyn Fn(usize) -> bool,
    ) -> Result<(Vec<SearchResult>, DiskSearchStats)> {
        if query.len() != self.dim() {
            return Err(Error::DimensionMismatch { expected: self.dim(), got: query.len() });
        }
        check_finite(query)?;
        if params.beam_width == 0 || params.beam_width > MAX_BEAM_WIDTH {
            return Err(Error::InvalidArgument(format!("beam width must be in 1..={MAX_BEAM_WIDTH}")));
        }
        if k == 0 {
            return Ok((Vec::new(), DiskSearchStats::default()));
        }
        let mut q = query.to_vec();
        if self.metric().needs_normalization() {
            normalize(&mut q);
        }
        let table = self.pq.distance_table(self.metric(), &q);
        let kern = kernels::best();
        ctx.reader.reserve(self.geo.max_read());
        ctx.vector.resize(self.dim(), 0.0);

        let mut stats = self.beam(ctx, kern, &table, &q, params.l.max(k), params.beam_width)?;
        let mut top = TopK::new(k);
        match self.layout() {
            Layout::Inline => {
                for e in &ctx.expanded {
                    let mut any = false;
                    for pos in self.positions_of(e.node) {
                        if admit(pos) {
                            top.push(self.ids[pos], e.exact);
                            any = true;
                        }
                    }
                    stats.admitted += any as usize;
                }
            }
            Layout::Separate => {
                // Drop rejected nodes before the re-rank cut, so its reads
                // go to nodes that can be returned.
                ctx.expanded.retain(|e| self.positions_of(e.node).any(admit));
                stats.admitted = ctx.expanded.len();
                let rerank = if params.rerank == 0 { usize::MAX } else { params.rerank.max(k) };
                self.rerank(ctx, kern, &q, rerank, admit, &mut top, &mut stats)?;
            }
        }
        Ok((top.into_sorted(), stats))
    }

    fn beam(
        &self,
        ctx: &mut SearchContext,
        kern: &Kernels,
        table: &DistanceTable,
        q: &[f32],
        l: usize,
        width: usize,
    ) -> Result<DiskSearchStats> {
        let SearchContext { reader, visited, list, picked, slots, reqs, fresh, expanded, vector, .. } = ctx;
        visited.clear();
        list.clear();
        expanded.clear();
        let entry = self.header.entry;
        visited.insert(entry);
        list.push(Candidate { dist: table.distance(self.code(entry)), node: entry, expanded: false });
        let mut st = DiskSearchStats { pq_distances: 1, ..Default::default() };

        let mut cursor = 0; // first unexpanded candidate
        loop {
            picked.clear();
            let mut i = cursor;
            while i < list.len() && picked.len() < width {
                if !list[i].expanded {
                    list[i].expanded = true;
                    picked.push((list[i].dist, list[i].node));
                }
                i += 1;
            }
            if picked.is_empty() {
                break;
            }
            reqs.clear();
            slots.clear();
            for &(_, node) in picked.iter() {
                if !self.cache.is_empty() && self.cache.contains_key(&node) {
                    slots.push(usize::MAX);
                } else {
                    slots.push(reqs.len());
                    reqs.push((self.block_offset(node), self.geo.block_bytes));
                }
            }
            if !reqs.is_empty() {
                reader.read(&self.file, reqs)?;
                st.rounds += 1;
                st.reads += reqs.len();
                st.read_bytes += reqs.len() * self.geo.block_bytes;
            }
            st.expanded += picked.len();
            st.cache_hits += picked.len() - reqs.len();

            let mut min_pos = usize::MAX;
            for (&(pq, node), &slot) in picked.iter().zip(slots.iter()) {
                let rec = match slot {
                    usize::MAX => &self.cache[&node][..],
                    s => self.record_in(reader.slot(s), node),
                };
                let r = self.parse(node, rec)?;
                let mut e = Expanded { pq, node, exact: f32::NAN, vector_crc: r.vector_crc };
                if self.layout() == Layout::Inline {
                    decode_vector(self.codec(), r.vector, vector);
                    e.exact = self.exact(kern, q, vector);
                    st.exact_distances += 1;
                }
                expanded.push(e);

                // As in the in-memory search: mark, prefetch codes, then score.
                fresh.clear();
                fresh.extend(r.neighbors().filter(|&nb| visited.insert(nb)));
                fresh.iter().for_each(|&nb| kernels::prefetch(self.code(nb)));
                for &nb in fresh.iter() {
                    let d = table.distance(self.code(nb));
                    st.pq_distances += 1;
                    if list.len() == l {
                        let last = list[l - 1];
                        if order(d, nb, last.dist, last.node) != std::cmp::Ordering::Less {
                            continue;
                        }
                        list.pop();
                    }
                    let pos = list.partition_point(|c| order(c.dist, c.node, d, nb) == std::cmp::Ordering::Less);
                    list.insert(pos, Candidate { dist: d, node: nb, expanded: false });
                    min_pos = min_pos.min(pos);
                }
            }
            cursor = cursor.min(min_pos);
            while cursor < list.len() && list[cursor].expanded {
                cursor += 1;
            }
        }
        Ok(st)
    }

    /// Separate layout: reads the vectors of the `rerank` best expanded
    /// nodes (by PQ distance) and scores them exactly.
    fn rerank(
        &self,
        ctx: &mut SearchContext,
        kern: &Kernels,
        q: &[f32],
        rerank: usize,
        admit: &dyn Fn(usize) -> bool,
        top: &mut TopK,
        st: &mut DiskSearchStats,
    ) -> Result<()> {
        let SearchContext { reader, reqs, expanded, vector, .. } = ctx;
        expanded.sort_unstable_by(|a, b| order(a.pq, a.node, b.pq, b.node));
        let n = rerank.min(expanded.len());
        let vbb = self.geo.vector_block_bytes;
        let base = self.header.sections[VECTORS].offset;
        for chunk in expanded[..n].chunks(DEPTH) {
            reqs.clear();
            reqs.extend(chunk.iter().map(|e| (base + e.node as u64 * vbb as u64, vbb)));
            reader.read(&self.file, reqs)?;
            st.rounds += 1;
            st.reads += chunk.len();
            st.read_bytes += chunk.len() * vbb;
            for (j, e) in chunk.iter().enumerate() {
                let bytes = &reader.slot(j)[..self.geo.vector_bytes];
                if crc32c(bytes) != e.vector_crc {
                    return Err(Error::Corrupt(format!("vector of node {} failed its checksum", e.node)));
                }
                decode_vector(self.codec(), bytes, vector);
                let d = self.exact(kern, q, vector);
                st.exact_distances += 1;
                self.positions_of(e.node).filter(|&p| admit(p)).for_each(|p| top.push(self.ids[p], d));
            }
        }
        Ok(())
    }

    /// Reads every record and vector of the file at the path the index was
    /// opened from, and checks checksums and neighbor ids. Search checks
    /// what it reads anyway; this scans everything (after a copy, say).
    pub fn verify(&self) -> Result<()> {
        let f = File::open(&self.path)?;
        let result = self.verify_with(&f);
        drop_cache(&f);
        result
    }

    fn verify_with(&self, f: &File) -> Result<()> {
        let blocks_per_chunk = ((1 << 20) / self.geo.block_bytes).max(1);
        let mut buf = vec![0u8; blocks_per_chunk * self.geo.block_bytes];
        let mut vector_crcs = Vec::new();
        let rpb = self.geo.records_per_block;
        let blocks = self.nodes().div_ceil(rpb);
        let mut block = 0;
        while block < blocks {
            let nb = blocks_per_chunk.min(blocks - block);
            let chunk = &mut buf[..nb * self.geo.block_bytes];
            read_exact_at(f, chunk, self.header.sections[GRAPH].offset + (block * self.geo.block_bytes) as u64)?;
            for b in 0..nb {
                for j in 0..rpb {
                    let node = (block + b) * rpb + j;
                    if node >= self.nodes() {
                        break;
                    }
                    let rec = &chunk[b * self.geo.block_bytes + j * self.geo.record_len..][..self.geo.record_len];
                    let r = self.parse(node as u32, rec)?;
                    if self.layout() == Layout::Separate {
                        vector_crcs.push(r.vector_crc);
                    }
                }
            }
            block += nb;
        }
        let vbb = self.geo.vector_block_bytes;
        if vbb > 0 {
            let mut v = vec![0u8; vbb];
            for (node, &want) in vector_crcs.iter().enumerate() {
                read_exact_at(f, &mut v, self.header.sections[VECTORS].offset + (node * vbb) as u64)?;
                if crc32c(&v[..self.geo.vector_bytes]) != want {
                    return Err(Error::Corrupt(format!("vector of node {node} failed its checksum")));
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn code(&self, node: u32) -> &[u8] {
        let m = self.header.pq_m;
        &self.codes[node as usize * m..][..m]
    }

    #[inline]
    fn block_offset(&self, node: u32) -> u64 {
        self.header.sections[GRAPH].offset + self.geo.locate(node).0
    }

    #[inline]
    fn record_in<'b>(&self, block: &'b [u8], node: u32) -> &'b [u8] {
        let within = self.geo.locate(node).1;
        &block[within..within + self.geo.record_len]
    }

    /// Checks a record's checksum, degree and neighbor ids.
    fn parse<'b>(&self, node: u32, rec: &'b [u8]) -> Result<Record<'b>> {
        let corrupt = |what: &str| Error::Corrupt(format!("record of node {node}: {what}"));
        let body = rec.len() - 4;
        let stored = u32::from_le_bytes(rec[body..].try_into().unwrap());
        if crc32c(&rec[..body]) != stored {
            return Err(corrupt("checksum mismatch"));
        }
        let at = if self.layout() == Layout::Inline { self.geo.vector_bytes } else { 0 };
        let degree = u32::from_le_bytes(rec[at..at + 4].try_into().unwrap()) as usize;
        if degree > self.max_degree() {
            return Err(corrupt("degree out of range"));
        }
        let neighbors = &rec[at + 4..at + 4 + 4 * degree];
        let nodes = self.header.nodes;
        if neighbors.as_chunks::<4>().0.iter().any(|b| u32::from_le_bytes(*b) as u64 >= nodes) {
            return Err(corrupt("neighbor id out of range"));
        }
        let vector_crc = if self.layout() == Layout::Separate {
            let o = 4 + 4 * self.max_degree();
            u32::from_le_bytes(rec[o..o + 4].try_into().unwrap())
        } else {
            0
        };
        Ok(Record { vector: &rec[..at], neighbors, vector_crc })
    }

    /// Exact distance from a prepared query to a decoded vector, as the flat
    /// and graph indexes compute it.
    #[inline]
    fn exact(&self, kern: &Kernels, q: &[f32], v: &[f32]) -> f32 {
        match self.metric() {
            Metric::L2 => (kern.l2_squared)(q, v),
            Metric::InnerProduct => -(kern.dot)(q, v),
            Metric::Cosine => 1.0 - (kern.dot)(q, v),
        }
    }
}

/// Reads a RAM section and checks its CRC.
fn read_section(f: &File, h: &Header, i: usize) -> Result<Vec<u8>> {
    let s = h.sections[i];
    let mut out = vec![0u8; s.len as usize];
    let mut crc = Crc32c::new();
    for (j, chunk) in out.chunks_mut(64 << 20).enumerate() {
        read_exact_at(f, chunk, s.offset + (j as u64) * (64 << 20))?;
        crc.update(chunk);
    }
    if crc.finalize() != s.crc {
        return Err(Error::Corrupt(format!("SSD index section {i} failed its checksum")));
    }
    Ok(out)
}

fn read_u64s(f: &File, h: &Header, i: usize) -> Result<Vec<u64>> {
    Ok(read_section(f, h, i)?.as_chunks::<8>().0.iter().map(|b| u64::from_le_bytes(*b)).collect())
}

/// A fresh directory for one test's files (removed first if it exists).
#[cfg(test)]
pub(crate) fn test_dir(name: &str) -> PathBuf {
    let dir = io::tests::temp_path(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[cfg(test)]
mod build_tests;
#[cfg(test)]
mod tests;
