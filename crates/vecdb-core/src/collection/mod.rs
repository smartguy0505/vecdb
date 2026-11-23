//! Collections: the storage engine of one shard. A collection is a
//! directory of immutable segment files plus a write-ahead log, tied
//! together by a manifest.
//!
//! ```text
//! <dir>/MANIFEST                 which files make up the collection (atomic replace)
//!       LOCK                     held while a process has the collection open
//!       wal-<gen>.log            writes since the last flush
//!       seg-<id>.flat            sealed segment: ids + prepared vectors, searched exactly
//!       seg-<id>.vdx             sealed segment: SSD index (PQ in RAM, graph + vectors on SSD)
//!       seg-<id>.attrs           attributes per row (when the collection declares fields)
//!       seg-<id>.del-<gen>       tombstones: the segment's deleted rows
//! ```
//!
//! **Write path.** A batch of upserts and deletes is checked, appended to
//! the WAL as one record (fsynced under [`Durability::Sync`]), then applied:
//! an upsert deletes the id's current row, wherever it is, and appends a new
//! row to the growing segment. The growing segment lives in RAM and is
//! searched exactly. Once its vectors reach `seal_bytes`, [`Collection::flush`]
//! seals it: its live rows become a flat segment file, the tombstones of
//! every segment that gained deletes are written, a new WAL is started, and
//! a new manifest commits all of it.
//!
//! **Index builds and compaction** ([`Collection::maintain`]) replace sealed
//! segments with new ones built from their live rows, and commit the swap
//! with a manifest. Small flat segments are merged into an SSD index once
//! they hold `index_bytes` of vectors; SSD segments with many deleted rows
//! are rebuilt without them.
//!
//! **Read path.** A search runs on every segment and merges the top-k lists.
//! Every segment returns exact distances, so they compare directly.
//!
//! **Crash safety.** Files are written under new names, fsynced, then
//! committed by renaming the manifest into place; the old WAL is removed only
//! after that. Opening replays the WAL on top of the manifest's state and
//! removes files no manifest refers to.
//!
//! **Concurrency.** One writer: writes take `&mut self` and searches `&self`,
//! and nothing runs in the background. Sealed segments are shared through
//! `Arc` and only their tombstones change (atomically), so a later version
//! can search a snapshot of the segment list while a writer works.

mod bytes;
mod maintain;
mod manifest;
pub mod meta;
mod segment;
mod wal;

use std::fs::{self, File, OpenOptions as FsOpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::disk::{DiskIndex, DiskOpenOptions, DiskSearchParams, IoMode, Layout};
use crate::distance::{Metric, check_finite, normalize};
use crate::error::{Error, Result};
use crate::format::{read_flat_segment, sync_parent_dir, write_flat_segment};
use crate::graph::GraphParams;
use crate::kernels;
use crate::quant::Codec;
use crate::store::{VectorStore, check_dim};
use crate::topk::{SearchResult, TopK};
use manifest::{MANIFEST, Manifest, SegmentEntry, SegmentKind};
use segment::{AttrTable, Body, Growing, Sealed, Tombstones, attrs_name, del_name, disk_name, flat_name};
use wal::{LoggedOp, Wal, wal_name};

pub use maintain::{JobReport, MIN_SSD_ROWS};
pub use meta::{Attrs, Field, FieldType, NO_ATTRS, Value};
pub use wal::ReplayReport;

const LOCK: &str = "LOCK";

/// When a write counts as done.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Durability {
    /// After the WAL record is fsynced: survives power loss.
    Sync,
    /// After the WAL record is handed to the OS: survives a crash of the
    /// process, not of the machine.
    Buffered,
}

/// How the SSD indexes of a collection are built.
#[derive(Clone, Debug, PartialEq)]
pub struct IndexConfig {
    pub max_degree: usize,
    pub build_list: usize,
    pub alpha: f32,
    pub passes: usize,
    pub seed: u64,
    /// PQ bytes per vector.
    pub pq_m: usize,
    /// f32, f16 or bf16.
    pub codec: Codec,
    /// `None` picks by record size.
    pub layout: Option<Layout>,
}

/// What a collection is: fixed when it is created and kept in its manifest.
#[derive(Clone, Debug, PartialEq)]
pub struct CollectionConfig {
    pub dim: usize,
    pub metric: Metric,
    /// The growing segment is sealed once its vectors take this many bytes.
    pub seal_bytes: u64,
    /// Flat segments are merged into an SSD index once their live vectors
    /// take this many bytes.
    pub index_bytes: u64,
    /// An SSD segment is rebuilt without its deleted rows once this
    /// fraction of its rows is deleted.
    pub compact_deleted: f32,
    /// SSD segments are merged, smallest first, while there are more.
    pub max_indexed_segments: usize,
    pub index: IndexConfig,
    /// Attribute fields rows may carry.
    pub fields: Vec<Field>,
}

impl CollectionConfig {
    /// Defaults: seal at 32 MiB of vectors, index at 64 MiB, compact at
    /// 20% deleted, at most 8 SSD segments; R 64, build L 100, α 1.2, PQ
    /// with `dim / 4` bytes (8 to 256), f32 vectors.
    ///
    /// The growing and flat segments are scanned exactly on every search
    /// (about 30 ns per 128-dim row on one core, measured), so the two
    /// thresholds bound that cost: with `maintain` run after each flush, at
    /// most about 96 MiB of vectors are scanned.
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            dim,
            metric,
            seal_bytes: 32 << 20,
            index_bytes: 64 << 20,
            compact_deleted: 0.2,
            max_indexed_segments: 8,
            index: IndexConfig {
                max_degree: 64,
                build_list: 100,
                alpha: 1.2,
                passes: 1,
                seed: 0x5eed,
                pq_m: (dim / 4).clamp(dim.min(8), 256),
                codec: Codec::F32,
                layout: None,
            },
            fields: Vec::new(),
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        check_dim(self.dim)?;
        let bad = |m: &str| Err(Error::InvalidArgument(m.into()));
        if self.seal_bytes < self.dim as u64 * 4 {
            return bad("seal_bytes must hold at least one vector");
        }
        if !(self.compact_deleted > 0.0 && self.compact_deleted <= 1.0) {
            return bad("compact_deleted must be in (0, 1]");
        }
        if self.max_indexed_segments == 0 {
            return bad("max_indexed_segments must be at least 1");
        }
        self.graph_params(1).validate()?;
        if self.index.pq_m == 0 || self.index.pq_m > self.dim {
            return bad("pq_m must be in 1..=dim");
        }
        if crate::disk::codec_code(self.index.codec).is_none() {
            return bad("SSD indexes store f32, f16 or bf16 vectors");
        }
        meta::validate_fields(&self.fields)
    }

    fn graph_params(&self, threads: usize) -> GraphParams {
        let i = &self.index;
        GraphParams {
            max_degree: i.max_degree,
            build_list: i.build_list,
            alpha: i.alpha,
            passes: i.passes,
            seed: i.seed,
            threads,
        }
    }
}

/// How this process uses a collection (not stored).
#[derive(Clone, Debug)]
pub struct OpenOptions {
    pub durability: Durability,
    /// Threads for index builds; 0 = one per CPU.
    pub threads: usize,
    /// RAM an index build may use for one partition.
    pub build_memory: u64,
    /// Read SSD indexes with `O_DIRECT`.
    pub direct_io: bool,
    pub io: IoMode,
    /// Called with a line of progress during index builds.
    pub progress: Option<fn(&str)>,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            durability: Durability::Sync,
            threads: 0,
            build_memory: 2 << 30,
            direct_io: true,
            io: IoMode::Auto,
            progress: None,
        }
    }
}

/// One write in a batch.
#[derive(Clone, Copy, Debug)]
pub enum WriteOp<'a> {
    /// Inserts the row, replacing any row with the same id.
    Upsert { id: u64, vector: &'a [f32], attrs: &'a Attrs },
    Delete { id: u64 },
}

/// Search knobs, passed to every SSD segment.
#[derive(Clone, Debug)]
pub struct SearchOptions {
    /// Candidate list size per SSD segment (raised for segments with many
    /// deleted rows).
    pub l: usize,
    pub beam_width: usize,
    /// Separate layout: vectors re-ranked per segment; 0 = all expanded.
    pub rerank: usize,
    /// Scale `l` by 1 / (fraction of a segment's rows not deleted), up to
    /// 8x: deleted nodes still take places in the candidate list.
    pub adapt_l: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self { l: 100, beam_width: 4, rerank: 0, adapt_l: true }
    }
}

/// Counters for one search, summed over segments.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchStats {
    pub segments: usize,
    /// Rows scored exactly in RAM (growing and flat segments).
    pub scanned: usize,
    /// SSD segments: blocks read, rounds waited on, nodes expanded.
    pub reads: usize,
    pub rounds: usize,
    pub expanded: usize,
}

/// One segment, for [`Collection::stats`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentInfo {
    pub id: u64,
    /// "flat" or "ssd".
    pub kind: &'static str,
    pub rows: usize,
    pub deleted: usize,
    /// RAM for its tombstones and id lookup (beyond the index itself).
    pub lookup_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CollectionStats {
    /// Live rows.
    pub live: usize,
    pub growing_rows: usize,
    pub growing_live: usize,
    pub segments: Vec<SegmentInfo>,
    pub wal_bytes: u64,
    /// Sequence number the next write gets.
    pub next_seq: u64,
}

/// What [`Collection::open`] found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OpenReport {
    pub replay: ReplayReport,
    /// Files no manifest referred to (left by an interrupted flush or build).
    pub removed: Vec<String>,
}

/// The in-memory state: segments and the next sequence number.
struct State {
    config: CollectionConfig,
    growing: Growing,
    /// Oldest first, in manifest order.
    sealed: Vec<Arc<Sealed>>,
    next_seq: u64,
    scratch: Vec<f32>,
}

impl State {
    fn contains(&self, id: u64) -> bool {
        self.growing.row_of(id).is_some() || self.sealed.iter().any(|s| s.find(id).is_some())
    }

    /// Deletes the live row of `id`, if any.
    fn delete(&mut self, id: u64) -> bool {
        if self.growing.delete(id) {
            return true;
        }
        // Newest first: a re-inserted id is most likely in a recent segment.
        for s in self.sealed.iter().rev() {
            if let Some(pos) = s.find(id) {
                s.tomb.set(pos);
                return true;
            }
        }
        false
    }

    fn upsert(&mut self, id: u64, vector: &[f32], attrs: Option<Attrs>) {
        self.delete(id);
        let mut v = std::mem::take(&mut self.scratch);
        v.clear();
        v.extend_from_slice(vector);
        if self.config.metric.needs_normalization() {
            normalize(&mut v);
        }
        self.growing.push(id, &v, attrs);
        self.scratch = v;
    }

    fn apply(&mut self, op: &WriteOp) {
        match *op {
            WriteOp::Upsert { id, vector, attrs } => {
                let a = (!self.config.fields.is_empty()).then(|| attrs.clone());
                self.upsert(id, vector, a);
            }
            WriteOp::Delete { id } => {
                self.delete(id);
            }
        }
    }
}

pub struct Collection {
    dir: PathBuf,
    _lock: File,
    opts: OpenOptions,
    /// The manifest on disk.
    manifest: Manifest,
    wal: Wal,
    state: State,
}

impl Collection {
    /// Creates a collection in `dir` (created if missing; must not hold one).
    pub fn create(dir: impl AsRef<Path>, config: CollectionConfig, opts: OpenOptions) -> Result<Self> {
        config.validate()?;
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;
        let lock = lock_dir(dir)?;
        if dir.join(MANIFEST).exists() {
            return Err(Error::InvalidArgument(format!("{} already holds a collection", dir.display())));
        }
        // Leftovers of a create that did not finish are its WAL and manifest
        // temp file; any other file of ours means a collection lost its
        // manifest, which is not ours to delete.
        for entry in fs::read_dir(dir)? {
            let name = entry?.file_name().to_string_lossy().into_owned();
            if name == wal_name(1) || name == format!("{MANIFEST}.tmp") {
                fs::remove_file(dir.join(&name))?;
            } else if is_ours(&name) {
                return Err(Error::InvalidArgument(format!(
                    "{} holds collection files ({name}) but no manifest",
                    dir.display()
                )));
            }
        }
        let manifest = Manifest {
            config: config.clone(),
            generation: 1,
            flushed_seq: 0,
            wal_gen: 1,
            next_segment: 1,
            segments: Vec::new(),
        };
        let wal = Wal::create(dir, 1, config.dim)?;
        sync_parent_dir(wal.path())?;
        manifest.write(dir)?;
        let state = State { growing: Growing::new(config.dim), config, sealed: Vec::new(), next_seq: 1, scratch: Vec::new() };
        Ok(Self { dir: dir.to_path_buf(), _lock: lock, opts, manifest, wal, state })
    }

    /// Opens the collection in `dir`: loads its segments, replays the WAL
    /// and removes files the manifest does not refer to.
    pub fn open(dir: impl AsRef<Path>, opts: OpenOptions) -> Result<(Self, OpenReport)> {
        let dir = dir.as_ref();
        let lock = lock_dir(dir)?;
        let manifest = Manifest::read(dir)?;
        let config = manifest.config.clone();
        let mut sealed = Vec::with_capacity(manifest.segments.len());
        for e in &manifest.segments {
            sealed.push(Arc::new(load_segment(dir, &config, e, &opts)?));
        }
        let mut state = State {
            growing: Growing::new(config.dim),
            config,
            sealed,
            next_seq: manifest.flushed_seq + 1,
            scratch: Vec::new(),
        };
        let fields = state.config.fields.clone();
        let (wal, replay) =
            Wal::replay(dir, manifest.wal_gen, state.config.dim, state.next_seq, |seq, op| {
                match op {
                    LoggedOp::Upsert { id, vector, attrs } => {
                        let a = if fields.is_empty() { None } else { Some(Attrs::decode(attrs)?) };
                        state.upsert(id, vector, a);
                    }
                    LoggedOp::Delete { id } => {
                        state.delete(id);
                    }
                }
                state.next_seq = seq + 1;
                Ok(())
            })?;
        // Only once everything the manifest names has loaded.
        let removed = remove_unreferenced(dir, &manifest)?;
        let c = Self { dir: dir.to_path_buf(), _lock: lock, opts, manifest, wal, state };
        Ok((c, OpenReport { replay, removed }))
    }

    pub fn config(&self) -> &CollectionConfig {
        &self.state.config
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Live rows.
    pub fn len(&self) -> usize {
        self.state.growing.live() + self.state.sealed.iter().map(|s| s.live()).sum::<usize>()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, id: u64) -> bool {
        self.state.contains(id)
    }

    pub fn stats(&self) -> CollectionStats {
        let segments = self
            .state
            .sealed
            .iter()
            .map(|s| SegmentInfo {
                id: s.id,
                kind: match s.body {
                    Body::Flat(_) => "flat",
                    Body::Disk { .. } => "ssd",
                },
                rows: s.positions(),
                deleted: s.tomb.count(),
                lookup_bytes: s.lookup_bytes(),
            })
            .collect();
        CollectionStats {
            live: self.len(),
            growing_rows: self.state.growing.rows(),
            growing_live: self.state.growing.live(),
            segments,
            wal_bytes: self.wal.bytes(),
            next_seq: self.state.next_seq,
        }
    }

    pub fn upsert(&mut self, id: u64, vector: &[f32], attrs: &Attrs) -> Result<()> {
        self.write(&[WriteOp::Upsert { id, vector, attrs }])
    }

    /// Deletes `id`; returns whether it was there.
    pub fn delete(&mut self, id: u64) -> Result<bool> {
        let had = self.contains(id);
        self.write(&[WriteOp::Delete { id }])?;
        Ok(had)
    }

    /// Applies `ops` in order as one batch: logged as one WAL record, so a
    /// crash keeps all of them or none. Seals the growing segment when it
    /// is full; if that fails, the batch is still applied and logged.
    pub fn write(&mut self, ops: &[WriteOp]) -> Result<()> {
        if ops.is_empty() {
            return Ok(());
        }
        let cfg = &self.state.config;
        for op in ops {
            if let WriteOp::Upsert { vector, attrs, .. } = op {
                if vector.len() != cfg.dim {
                    return Err(Error::DimensionMismatch { expected: cfg.dim, got: vector.len() });
                }
                check_finite(vector)?;
                attrs.validate(&cfg.fields)?;
            }
        }
        if ops.len() > u32::MAX as usize || Wal::record_bytes(cfg.dim, ops) > wal::MAX_BATCH_BYTES {
            return Err(Error::InvalidArgument("write batch too large: split it".into()));
        }
        self.wal.append(self.state.next_seq, ops, self.opts.durability)?;
        for op in ops {
            self.state.apply(op);
        }
        self.state.next_seq += ops.len() as u64;
        // The WAL check bounds the log of a workload that mostly deletes.
        let seal = self.state.config.seal_bytes;
        if self.state.growing.bytes() >= seal || self.wal.bytes() >= seal.saturating_mul(2) {
            self.flush()?;
        }
        Ok(())
    }

    /// The stored vector and attributes of `id`. Vectors come back as
    /// stored: unit length for cosine, rounded by f16 / bf16 storage.
    pub fn get(&self, id: u64) -> Result<Option<(Vec<f32>, Attrs)>> {
        let g = &self.state.growing;
        if let Some(r) = g.row_of(id) {
            return Ok(Some((g.vector(r).to_vec(), g.attrs.get(r).cloned().unwrap_or_default())));
        }
        for s in self.state.sealed.iter().rev() {
            let Some(pos) = s.find(id) else { continue };
            let attrs = s.attrs.rows.get(pos).cloned().unwrap_or_default();
            let v = match &s.body {
                Body::Flat(store) => store.vector(pos).to_vec(),
                Body::Disk { index, .. } => {
                    let mut v = vec![0.0; index.dim()];
                    let f = File::open(&s.files[0])?;
                    index.read_node_vectors(&f, index.node_of(pos), &mut v)?;
                    v
                }
            };
            return Ok(Some((v, attrs)));
        }
        Ok(None)
    }

    pub fn search(&self, query: &[f32], k: usize, opts: &SearchOptions) -> Result<Vec<SearchResult>> {
        Ok(self.search_with_stats(query, k, opts)?.0)
    }

    /// The `k` nearest live rows over all segments, closest first.
    pub fn search_with_stats(
        &self,
        query: &[f32],
        k: usize,
        opts: &SearchOptions,
    ) -> Result<(Vec<SearchResult>, SearchStats)> {
        let cfg = &self.state.config;
        if query.len() != cfg.dim {
            return Err(Error::DimensionMismatch { expected: cfg.dim, got: query.len() });
        }
        check_finite(query)?;
        let mut st = SearchStats::default();
        if k == 0 {
            return Ok((Vec::new(), st));
        }
        let mut q = query.to_vec();
        if cfg.metric.needs_normalization() {
            normalize(&mut q);
        }
        let kern = kernels::best();
        let mut top = TopK::new(k);

        let g = &self.state.growing;
        if g.live() > 0 {
            st.segments += 1;
            st.scanned += g.live();
            segment::scan(kern, cfg.metric, &q, &g.data, |r| !g.is_dead(r), |r, d| top.push(g.ids[r], d));
        }
        for s in &self.state.sealed {
            if s.live() == 0 {
                continue;
            }
            st.segments += 1;
            match &s.body {
                Body::Flat(store) => {
                    st.scanned += s.live();
                    let admit = |p: usize| !s.tomb.is_set(p);
                    segment::scan(kern, cfg.metric, &q, store.raw_vectors(), admit, |p, d| top.push(store.id(p), d));
                }
                Body::Disk { index, .. } => {
                    let live = if opts.adapt_l { s.live() as f64 / s.positions() as f64 } else { 1.0 };
                    let params = DiskSearchParams {
                        l: boosted_l(opts.l.max(k), live),
                        beam_width: opts.beam_width,
                        rerank: opts.rerank,
                    };
                    let (res, ds) = index.search_filtered(query, k, &params, &|p| !s.tomb.is_set(p))?;
                    res.iter().for_each(|r| top.push(r.id, r.distance));
                    st.reads += ds.reads;
                    st.rounds += ds.rounds;
                    st.expanded += ds.expanded;
                }
            }
        }
        Ok((top.into_sorted(), st))
    }

    /// Seals the growing segment into a flat segment file and writes the
    /// tombstones of every segment that gained deletes, then commits both
    /// with a new manifest and starts a new WAL. Does nothing when there is
    /// nothing to write.
    pub fn flush(&mut self) -> Result<()> {
        let dirty: Vec<usize> = (0..self.state.sealed.len())
            .filter(|&i| self.state.sealed[i].tomb.count() as u64 != self.manifest.segments[i].deleted)
            .collect();
        if self.state.growing.rows() == 0 && dirty.is_empty() && self.wal.batches() == 0 {
            return Ok(());
        }
        let mut m = self.manifest.clone();
        m.generation += 1;
        let mut created: Vec<PathBuf> = Vec::new();
        let mut superseded: Vec<PathBuf> = Vec::new();
        let result = (|| -> Result<(Option<Sealed>, Wal)> {
            // 1. The growing segment's live rows, as a flat segment.
            let fresh = if self.state.growing.live() > 0 {
                let (store, attrs) = live_rows(&self.state.growing, self.state.config.dim)?;
                let id = m.next_segment;
                m.next_segment += 1;
                let path = self.dir.join(flat_name(id));
                created.push(path.clone());
                write_flat_segment(&path, self.state.config.metric, &store)?;
                let mut files = vec![path];
                if !self.state.config.fields.is_empty() {
                    let ap = self.dir.join(attrs_name(id));
                    created.push(ap.clone());
                    attrs.write(&ap)?;
                    files.push(ap);
                }
                m.segments.push(SegmentEntry {
                    id,
                    kind: SegmentKind::Flat,
                    positions: store.len() as u64,
                    deleted: 0,
                    del_gen: 0,
                });
                Some(Sealed::flat(id, store, attrs, files))
            } else {
                None
            };
            // 2. Tombstones that changed.
            for &i in &dirty {
                let s = &self.state.sealed[i];
                let path = self.dir.join(del_name(s.id, m.generation));
                created.push(path.clone());
                let count = s.tomb.write(&path, s.id)?;
                let e = &mut m.segments[i];
                if e.del_gen != 0 {
                    superseded.push(self.dir.join(del_name(e.id, e.del_gen)));
                }
                e.deleted = count as u64;
                e.del_gen = m.generation;
            }
            // 3. A new WAL for the writes after this flush.
            m.wal_gen += 1;
            let wal = Wal::create(&self.dir, m.wal_gen, self.state.config.dim)?;
            created.push(wal.path().to_path_buf());
            m.flushed_seq = self.state.next_seq - 1;
            // 4. Commit: the new files are fsynced, their names too, then
            // the manifest is renamed into place.
            sync_parent_dir(wal.path())?;
            m.install(&self.dir)?;
            Ok((fresh, wal))
        })();
        let (fresh, wal) = match result {
            Ok(r) => r,
            Err(e) => {
                created.iter().for_each(|p| {
                    let _ = fs::remove_file(p);
                });
                return Err(e);
            }
        };
        // The new manifest is in place, so the state follows it. Until the
        // directory fsync succeeds the rename may not survive a crash, so
        // the old files stay (the next open removes them).
        let durable = sync_parent_dir(&self.dir.join(MANIFEST));
        superseded.push(self.wal.path().to_path_buf());
        self.wal = wal;
        self.manifest = m;
        if let Some(s) = fresh {
            self.state.sealed.push(Arc::new(s));
        }
        self.state.growing = Growing::new(self.state.config.dim);
        if let Err(e) = durable {
            // A crash could bring back the old manifest, which does not know
            // the new WAL: take no writes that would go there.
            self.wal.poison();
            return Err(e);
        }
        for p in superseded {
            let _ = fs::remove_file(p);
        }
        Ok(())
    }
}

/// Candidate list size for an SSD segment where only `admitted` of the
/// rows can be returned: the walk passes through rejected nodes, so it
/// needs a longer list to expand as many returnable ones.
fn boosted_l(l: usize, admitted: f64) -> usize {
    const MAX_BOOST: f64 = 8.0;
    let boost = (1.0 / admitted.max(1e-9)).min(MAX_BOOST);
    (l as f64 * boost).ceil() as usize
}

/// The growing segment's live rows, copied (it stays intact if the flush
/// fails).
fn live_rows(g: &Growing, dim: usize) -> Result<(crate::store::InMemoryStore, AttrTable)> {
    let mut store = crate::store::InMemoryStore::with_capacity(dim, g.live())?;
    let mut attrs = AttrTable::default();
    for r in (0..g.rows()).filter(|&r| !g.is_dead(r)) {
        store.push(g.ids[r], g.vector(r))?;
        if let Some(a) = g.attrs.get(r) {
            attrs.rows.push(a.clone());
        }
    }
    Ok((store, attrs))
}

fn lock_dir(dir: &Path) -> Result<File> {
    let f = FsOpenOptions::new().create(true).truncate(false).write(true).open(dir.join(LOCK))?;
    match f.try_lock() {
        Ok(()) => Ok(f),
        Err(fs::TryLockError::WouldBlock) => {
            Err(Error::InvalidArgument(format!("collection {} is already open", dir.display())))
        }
        Err(fs::TryLockError::Error(e)) => Err(e.into()),
    }
}

fn load_segment(dir: &Path, cfg: &CollectionConfig, e: &SegmentEntry, opts: &OpenOptions) -> Result<Sealed> {
    let mismatch = |what: &str| Error::Corrupt(format!("segment {}: {what} does not match the collection", e.id));
    let mut s = match e.kind {
        SegmentKind::Flat => {
            let path = dir.join(flat_name(e.id));
            let (metric, store) = read_flat_segment(&path)?;
            if metric != cfg.metric || store.dim() != cfg.dim {
                return Err(mismatch("metric or dimension"));
            }
            Sealed::flat(e.id, store, AttrTable::default(), vec![path])
        }
        SegmentKind::Disk => {
            let path = dir.join(disk_name(e.id));
            let index = DiskIndex::open(&path, &DiskOpenOptions { direct: opts.direct_io, io: opts.io })?;
            if index.metric() != cfg.metric || index.dim() != cfg.dim {
                return Err(mismatch("metric or dimension"));
            }
            Sealed::disk(e.id, index, AttrTable::default(), vec![path])
        }
    };
    if s.positions() as u64 != e.positions {
        return Err(mismatch("row count"));
    }
    if !cfg.fields.is_empty() {
        let path = dir.join(attrs_name(e.id));
        s.attrs = AttrTable::read(&path, s.positions())?;
        s.files.push(path);
    }
    if e.del_gen != 0 {
        s.tomb = Tombstones::read(&dir.join(del_name(e.id, e.del_gen)), e.id, s.positions(), e.deleted as usize)?;
    }
    Ok(s)
}

/// The files of ours in `dir` that `m` does not refer to, removed. Files
/// with other names are left alone.
fn remove_unreferenced(dir: &Path, m: &Manifest) -> Result<Vec<String>> {
    let mut keep: Vec<String> = vec![wal_name(m.wal_gen)];
    for e in &m.segments {
        keep.push(match e.kind {
            SegmentKind::Flat => flat_name(e.id),
            SegmentKind::Disk => disk_name(e.id),
        });
        if !m.config.fields.is_empty() {
            keep.push(attrs_name(e.id));
        }
        if e.del_gen != 0 {
            keep.push(del_name(e.id, e.del_gen));
        }
    }
    let mut removed = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let Some(name) = entry.file_name().to_str().map(str::to_string) else { continue };
        if !is_ours(&name) || keep.contains(&name) {
            continue;
        }
        if entry.file_type()?.is_dir() {
            fs::remove_dir_all(entry.path())?;
        } else {
            fs::remove_file(entry.path())?;
        }
        removed.push(name);
    }
    removed.sort();
    Ok(removed)
}

/// Whether `name` is one of the names a collection writes: segment, WAL
/// and tombstone files, their `.tmp` files, index build scratch
/// directories, and `MANIFEST.tmp`.
fn is_ours(name: &str) -> bool {
    let digits = |s: &str| s.len() == 8 && s.bytes().all(|b| b.is_ascii_digit());
    let base = name.strip_suffix(".tmp").unwrap_or(name);
    if base == MANIFEST {
        return name != MANIFEST;
    }
    if let Some(rest) = base.strip_prefix("wal-") {
        return rest.strip_suffix(".log").is_some_and(digits);
    }
    let Some(rest) = base.strip_prefix("seg-") else { return false };
    let (id, ext) = rest.split_at(rest.len().min(8));
    if !digits(id) {
        return false;
    }
    match ext {
        ".flat" | ".vdx" | ".attrs" => true,
        _ => {
            if let Some(g) = ext.strip_prefix(".del-") {
                digits(g)
            } else if let Some(pid) = ext.strip_prefix(".vdx.build-") {
                !pid.is_empty() && pid.bytes().all(|b| b.is_ascii_digit())
            } else {
                false
            }
        }
    }
}

#[cfg(test)]
mod tests;
