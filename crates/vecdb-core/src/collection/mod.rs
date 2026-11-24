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
//!       seg-<id>.attrs           attributes: rows, and the column index filters read
//!                                (when the collection declares fields)
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
//! **Filters.** A search may carry a [`Filter`] on the declared fields.
//! Each segment keeps a column index of its attributes in RAM (keyword
//! postings, sorted integer values; see `columns`), and evaluates the
//! filter to a bitmap of matching positions, less its deleted ones. RAM
//! segments then score only those rows. An SSD segment either walks its
//! graph returning only matching rows, with the candidate list scaled by
//! 1 / (fraction matching), or scans: PQ distances to every matching row
//! from the codes in RAM, then the best `l` re-ranked from SSD
//! ([`FilterMode`]). `Auto` picks by the number of matching rows, and falls
//! back to the scan when a walk meets too few matching nodes, as it does
//! when the matches lie away from the query.
//!
//! **Crash safety.** Files are written under new names, fsynced, then
//! committed by renaming the manifest into place; the old WAL is removed only
//! after that. Opening replays the WAL on top of the manifest's state and
//! removes files no manifest refers to.
//!
//! **Concurrency.** A collection is shared by threads through `&self`.
//! Writes are applied one batch at a time, in sequence-number order, under
//! a writer lock; a batch becomes visible to searches all at once. A
//! search reads a snapshot: the segment list current when it starts (a
//! flush or a build replaces the list, never changes it) and the writes up
//! to the last one applied then. Later writes stay invisible to it, though
//! they change shared state: growing rows carry the sequence numbers of
//! the writes that inserted and deleted them, and deletes in sealed
//! segments keep their numbers until no open search predates them. So a
//! search never sees part of a batch, an id twice, or an upserted id
//! missing. [`Collection::get`] and [`Collection::contains`] read the same
//! way.
//!
//! **Group commit.** Writers that arrive while a batch is being logged
//! queue up; the next of them logs every waiting batch with one WAL write
//! and, under [`Durability::Sync`], one fsync. Each batch is still its own
//! WAL record, so a crash keeps or drops batches whole. Index builds run while searches and writes go on; only their
//! commit takes the writer lock (see `maintain`).

mod bitmap;
mod bytes;
mod columns;
pub mod filter;
mod maintain;
mod manifest;
pub mod meta;
mod segment;
mod wal;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::{self, File, OpenOptions as FsOpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};

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
use bitmap::Bitmap;
use columns::{AttrWriter, SegmentAttrs, decode_row, encode_row};
use filter::Compiled;
use segment::{AsOf, Body, Growing, Sealed, Tombstones, attrs_name, del_name, disk_name, flat_name};
use wal::{LoggedOp, Wal, wal_name};

pub use filter::Filter;
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
    /// deleted or filtered-out rows); also the rows re-ranked exactly when
    /// a filter scans a segment.
    pub l: usize,
    pub beam_width: usize,
    /// Separate layout: vectors re-ranked per segment; 0 = all expanded.
    pub rerank: usize,
    /// Scale `l` by 1 / (fraction of a segment's rows that can be
    /// returned: not deleted, and matching the filter), up to 8x for
    /// deletes and 16x under a filter: the other nodes still take places in
    /// the candidate list. Off, the walk keeps `l` (and `Auto` still picks
    /// between walk and scan as if it were on).
    pub adapt_l: bool,
    /// Only rows that match.
    pub filter: Option<Filter>,
    /// How SSD segments are searched under a filter.
    pub filter_mode: FilterMode,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self { l: 100, beam_width: 4, rerank: 0, adapt_l: true, filter: None, filter_mode: FilterMode::Auto }
    }
}

/// How an SSD segment is searched under a filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterMode {
    /// Whichever is expected to be cheaper, with a scan when the walk
    /// finds too few matching rows.
    Auto,
    /// The graph walk, returning only matching rows.
    Graph,
    /// PQ distances to every matching row (codes in RAM), then the best
    /// `l` re-ranked exactly from SSD.
    Scan,
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
    /// With a filter: live rows that match it.
    pub matched: usize,
    /// SSD segments searched by the graph walk and by a scan (a fallback
    /// counts as both).
    pub graph_segments: usize,
    pub scan_segments: usize,
    /// Walks that found too few matching rows and fell back to a scan.
    pub fallbacks: usize,
    /// Rows scored by PQ in scans.
    pub pq_scanned: usize,
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
    /// RAM for its attribute columns, and the values they hold.
    pub attr_bytes: usize,
    pub attr_values: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CollectionStats {
    /// Live rows.
    pub live: usize,
    pub growing_rows: usize,
    pub growing_live: usize,
    /// RAM for the growing segment's attributes and their index.
    pub growing_attr_bytes: usize,
    pub segments: Vec<SegmentInfo>,
    pub wal_bytes: u64,
    /// Sequence number the next write gets.
    pub next_seq: u64,
    /// Deletes kept with their sequence numbers for open searches.
    pub delete_records: usize,
    /// Why the last flush a write started failed, until one succeeds.
    pub flush_error: Option<String>,
    /// Since the collection was opened: write batches, the groups they
    /// were logged in (one WAL write each), and WAL fsyncs.
    pub write_batches: u64,
    pub write_groups: u64,
    pub wal_syncs: u64,
}

/// What [`Collection::open`] found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OpenReport {
    pub replay: ReplayReport,
    /// Files no manifest referred to (left by an interrupted flush or build).
    pub removed: Vec<String>,
}

/// The segments a search reads. Flushes and builds replace the view (they
/// never change one in place), so a search keeps a consistent list of
/// segments for as long as it runs.
pub(crate) struct View {
    /// Oldest first, in manifest order.
    sealed: Vec<Arc<Sealed>>,
    /// Appended to by the writer.
    growing: Arc<RwLock<Growing>>,
}

/// What only the writer touches.
struct Writer {
    /// The manifest on disk (its `segments` in the same order as the
    /// view's sealed segments).
    manifest: Manifest,
    wal: Wal,
    /// Sequence number the next write gets.
    next_seq: u64,
    /// The last flush that a write started and that failed, if the next
    /// one has not succeeded yet.
    flush_error: Option<String>,
    /// Since the collection was opened: batches logged, groups they were
    /// logged in, fsyncs.
    batches: u64,
    groups: u64,
    syncs: u64,
}

/// A write batch checked and encoded by its caller, ready to log and apply.
#[derive(Default)]
struct Prepared {
    /// Per op: the id, and whether it is an upsert.
    ops: Vec<(u64, bool)>,
    /// Upserts: prepared vectors and encoded attribute rows, in order.
    vectors: Vec<f32>,
    rows: Vec<u8>,
    row_end: Vec<usize>,
    /// The ops as the WAL records them.
    wal: Vec<u8>,
}

/// Write batches waiting for the writer lock (group commit).
#[derive(Default)]
struct Queue {
    /// In arrival order, with their tickets.
    pending: VecDeque<(u64, Prepared)>,
    /// Results of batches logged and applied, by ticket.
    done: HashMap<u64, Result<Vec<bool>>>,
    next_ticket: u64,
    /// A writer is logging a group.
    leading: bool,
}

/// Vector bytes of the growing segment a search scans per read lock.
const GROWING_CHUNK_BYTES: usize = 1 << 20;

/// Rows of the growing segment a search scans per read lock: whole blocks
/// of the scan, about `GROWING_CHUNK_BYTES` of vectors. Tests use 3 rows,
/// so their few growing rows still cross chunk boundaries.
fn growing_chunk_rows(dim: usize) -> usize {
    if cfg!(test) {
        return 3;
    }
    (GROWING_CHUNK_BYTES / (dim * 4)).max(256) / 256 * 256
}

/// Most WAL bytes logged as one group (a single larger batch goes alone).
const GROUP_BYTES: usize = 16 << 20;

/// Set while a writer logs a group: if it panics, the group's writers get
/// an error instead of waiting forever.
struct Leading<'a> {
    c: &'a Collection,
    tickets: Vec<u64>,
}

impl Drop for Leading<'_> {
    fn drop(&mut self) {
        if self.tickets.is_empty() {
            return;
        }
        let mut q = self.c.queue.lock().unwrap_or_else(|e| e.into_inner());
        for &t in &self.tickets {
            q.done.insert(t, Err(Error::InvalidArgument("the writer logging this batch panicked".into())));
        }
        q.leading = false;
        self.c.queue_cv.notify_all();
    }
}

/// A search's view of the collection: the segments, and the last write it
/// sees. Registered while it lives, so deletes it predates keep their
/// sequence numbers.
struct Snapshot<'a> {
    c: &'a Collection,
    view: Arc<View>,
    seq: u64,
}

impl Drop for Snapshot<'_> {
    fn drop(&mut self) {
        let mut open = self.c.snapshots.lock().unwrap();
        if let Some(n) = open.get_mut(&self.seq) {
            *n -= 1;
            if *n == 0 {
                open.remove(&self.seq);
            }
        }
    }
}

/// A collection: shared by any number of threads (`&self` everywhere).
/// Writes are applied one batch at a time, in order; each search reads a
/// snapshot (see the module docs).
pub struct Collection {
    dir: PathBuf,
    _lock: File,
    opts: OpenOptions,
    config: CollectionConfig,
    /// Held while a batch is logged and applied, and while a flush or an
    /// index build commits.
    writer: Mutex<Writer>,
    /// Batches waiting to be logged; one waiting writer at a time logs all
    /// that are waiting, with one WAL write and (under `Sync`) one fsync.
    queue: Mutex<Queue>,
    queue_cv: Condvar,
    /// The segments searches start from.
    view: RwLock<Arc<View>>,
    /// The last write applied: new snapshots read as of it.
    visible: AtomicU64,
    /// Open snapshots: sequence number -> how many.
    snapshots: Mutex<BTreeMap<u64, usize>>,
    /// One `maintain` or `optimize` at a time.
    maintaining: Mutex<()>,
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
        let growing = Growing::new(config.dim, &config.fields);
        Ok(Self::assemble(dir, lock, opts, config, manifest, wal, Vec::new(), growing, 1))
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
        let mut growing = Growing::new(config.dim, &config.fields);
        let mut next_seq = manifest.flushed_seq + 1;
        let (mut vector, mut row) = (Vec::new(), Vec::new());
        let (wal, replay) = Wal::replay(dir, manifest.wal_gen, config.dim, next_seq, |seq, op| {
            match op {
                LoggedOp::Upsert { id, vector: v, attrs } => {
                    let a = Attrs::decode(attrs)?;
                    a.validate(&config.fields).map_err(|e| Error::Corrupt(format!("WAL write {seq}: {e}")))?;
                    prepare_vector(config.metric, v, &mut vector);
                    row.clear();
                    if !config.fields.is_empty() {
                        encode_row(&a, &config.fields, &mut row);
                    }
                    delete_live(&sealed, &mut growing, id, seq);
                    growing.push(id, &vector, &row, seq);
                }
                LoggedOp::Delete { id } => {
                    delete_live(&sealed, &mut growing, id, seq);
                }
            }
            next_seq = seq + 1;
            Ok(())
        })?;
        // Only once everything the manifest names has loaded.
        let removed = remove_unreferenced(dir, &manifest)?;
        let c = Self::assemble(dir, lock, opts, config, manifest, wal, sealed, growing, next_seq);
        Ok((c, OpenReport { replay, removed }))
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble(
        dir: &Path,
        lock: File,
        opts: OpenOptions,
        config: CollectionConfig,
        manifest: Manifest,
        wal: Wal,
        sealed: Vec<Arc<Sealed>>,
        growing: Growing,
        next_seq: u64,
    ) -> Self {
        let view = View { sealed, growing: Arc::new(RwLock::new(growing)) };
        Self {
            dir: dir.to_path_buf(),
            _lock: lock,
            opts,
            config,
            writer: Mutex::new(Writer { manifest, wal, next_seq, flush_error: None, batches: 0, groups: 0, syncs: 0 }),
            queue: Mutex::new(Queue::default()),
            queue_cv: Condvar::new(),
            view: RwLock::new(Arc::new(view)),
            visible: AtomicU64::new(next_seq - 1),
            snapshots: Mutex::new(BTreeMap::new()),
            maintaining: Mutex::new(()),
        }
    }

    pub fn config(&self) -> &CollectionConfig {
        &self.config
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The current view (not registered as a snapshot).
    fn view(&self) -> Arc<View> {
        self.view.read().unwrap().clone()
    }

    /// The view and the last write applied, read together (no flush or
    /// build can swap the view in between), and registered.
    fn snapshot(&self) -> Snapshot<'_> {
        let mut open = self.snapshots.lock().unwrap();
        let (view, seq) = {
            let v = self.view.read().unwrap();
            (v.clone(), self.visible.load(Ordering::Acquire))
        };
        *open.entry(seq).or_insert(0) += 1;
        drop(open);
        Snapshot { c: self, view, seq }
    }

    /// Live rows now.
    pub fn len(&self) -> usize {
        let v = self.view();
        let g = v.growing.read().unwrap().live();
        g + v.sealed.iter().map(|s| s.live()).sum::<usize>()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether `id` has a live row (as of the last write applied).
    pub fn contains(&self, id: u64) -> bool {
        let snap = self.snapshot();
        locate_at(&snap.view, id, snap.seq).is_some()
    }

    pub fn stats(&self) -> CollectionStats {
        let (wal_bytes, next_seq, flush_error, writes) = {
            let w = self.writer.lock().unwrap();
            (w.wal.bytes(), w.next_seq, w.flush_error.clone(), (w.batches, w.groups, w.syncs))
        };
        let v = self.view();
        let segments = v
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
                attr_bytes: s.attr_ram(),
                attr_values: s.attrs.as_ref().map_or(0, |a| a.columns.values()),
            })
            .collect();
        let g = v.growing.read().unwrap();
        CollectionStats {
            live: g.live() + v.sealed.iter().map(|s| s.live()).sum::<usize>(),
            growing_rows: g.rows(),
            growing_live: g.live(),
            growing_attr_bytes: g.attr_ram(),
            segments,
            wal_bytes,
            next_seq,
            delete_records: v.sealed.iter().map(|s| s.tomb.recent_len()).sum(),
            flush_error,
            write_batches: writes.0,
            write_groups: writes.1,
            wal_syncs: writes.2,
        }
    }

    pub fn upsert(&self, id: u64, vector: &[f32], attrs: &Attrs) -> Result<()> {
        self.write(&[WriteOp::Upsert { id, vector, attrs }])
    }

    /// Deletes `id`; returns whether it had a live row.
    pub fn delete(&self, id: u64) -> Result<bool> {
        Ok(self.write_ops(&[WriteOp::Delete { id }])?[0])
    }

    /// Applies `ops` in order as one batch: logged as one WAL record, so a
    /// crash keeps all of them or none, and searches see all of them or
    /// none. Seals the growing segment when it is full; a failed seal does
    /// not fail the write (it is logged and applied), and is retried by
    /// the next write (see [`CollectionStats::flush_error`]).
    pub fn write(&self, ops: &[WriteOp]) -> Result<()> {
        self.write_ops(ops).map(|_| ())
    }

    /// As `write`; per op, whether its id had a live row.
    fn write_ops(&self, ops: &[WriteOp]) -> Result<Vec<bool>> {
        if ops.is_empty() {
            return Ok(Vec::new());
        }
        let p = self.prepare(ops)?;
        let mut q = self.queue.lock().unwrap();
        let ticket = q.next_ticket;
        q.next_ticket += 1;
        q.pending.push_back((ticket, p));
        loop {
            if let Some(r) = q.done.remove(&ticket) {
                return r;
            }
            if q.leading {
                q = self.queue_cv.wait(q).unwrap();
                continue;
            }
            // Log everything waiting (up to GROUP_BYTES), this batch
            // included unless older ones fill the group.
            q.leading = true;
            let mut group = Vec::new();
            let mut bytes = 0;
            while let Some((_, p)) = q.pending.front() {
                if !group.is_empty() && bytes + p.wal.len() > GROUP_BYTES {
                    break;
                }
                bytes += p.wal.len();
                group.push(q.pending.pop_front().unwrap());
            }
            drop(q);
            let mut leading = Leading { c: self, tickets: group.iter().map(|(t, _)| *t).collect() };
            let results = {
                let batches: Vec<&Prepared> = group.iter().map(|(_, p)| p).collect();
                let mut w = self.writer.lock().unwrap();
                self.commit(&mut w, &batches)
            };
            q = self.queue.lock().unwrap();
            assert_eq!(results.len(), leading.tickets.len(), "one result per batch");
            for (t, r) in leading.tickets.drain(..).zip(results) {
                q.done.insert(t, r);
            }
            q.leading = false;
            self.queue_cv.notify_all();
        }
    }

    /// Checks `ops` and encodes them, outside any lock.
    fn prepare(&self, ops: &[WriteOp]) -> Result<Prepared> {
        let cfg = &self.config;
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
        let mut p = Prepared::default();
        Wal::encode_ops(ops, &mut p.wal);
        let mut v = Vec::new();
        for op in ops {
            match *op {
                WriteOp::Upsert { id, vector, attrs } => {
                    p.ops.push((id, true));
                    prepare_vector(cfg.metric, vector, &mut v);
                    p.vectors.extend_from_slice(&v);
                    if !cfg.fields.is_empty() {
                        encode_row(attrs, &cfg.fields, &mut p.rows);
                    }
                    p.row_end.push(p.rows.len());
                }
                WriteOp::Delete { id } => p.ops.push((id, false)),
            }
        }
        Ok(p)
    }

    /// Logs `batches` (with one fsync under `Sync`), applies them in order
    /// and makes them visible; then seals the growing segment if it is
    /// full. Per batch: per op, whether its id had a live row.
    fn commit(&self, w: &mut Writer, batches: &[&Prepared]) -> Vec<Result<Vec<bool>>> {
        let first = w.next_seq;
        let mut records = Vec::with_capacity(batches.len());
        let mut seq = first;
        for p in batches {
            records.push((seq, p.ops.len() as u32, &p.wal[..]));
            seq += p.ops.len() as u64;
        }
        if let Err(e) = w.wal.append_group(&records, self.opts.durability) {
            return batches.iter().map(|_| Err(e.duplicate())).collect();
        }
        w.batches += batches.len() as u64;
        w.groups += 1;
        w.syncs += (self.opts.durability == Durability::Sync) as u64;
        let view = self.view();
        let dim = self.config.dim;
        let mut out = Vec::with_capacity(batches.len());
        {
            let mut g = view.growing.write().unwrap();
            let mut seq = first;
            for p in batches {
                let mut had = Vec::with_capacity(p.ops.len());
                let mut u = 0;
                for &(id, upsert) in &p.ops {
                    had.push(delete_live(&view.sealed, &mut g, id, seq));
                    if upsert {
                        let row = &p.rows[if u == 0 { 0 } else { p.row_end[u - 1] }..p.row_end[u]];
                        g.push(id, &p.vectors[u * dim..(u + 1) * dim], row, seq);
                        u += 1;
                    }
                    seq += 1;
                }
                out.push(Ok(had));
            }
        }
        w.next_seq = seq;
        self.visible.store(seq - 1, Ordering::Release);
        self.prune(&view);
        // The WAL check bounds the log of a workload that mostly deletes.
        let seal = self.config.seal_bytes;
        let full = view.growing.read().unwrap().bytes() >= seal || w.wal.bytes() >= seal.saturating_mul(2);
        if full {
            w.flush_error = self.flush_locked(w).err().map(|e| e.to_string());
        }
        out
    }

    /// Drops the sequence numbers of deletes that no open snapshot predates.
    fn prune(&self, view: &View) {
        let oldest = self.snapshots.lock().unwrap().keys().next().copied();
        let upto = oldest.unwrap_or_else(|| self.visible.load(Ordering::Acquire));
        view.sealed.iter().for_each(|s| s.tomb.prune(upto));
    }

    /// The stored vector and attributes of `id` (as of the last write
    /// applied). Vectors come back as stored: unit length for cosine,
    /// rounded by f16 / bf16 storage.
    pub fn get(&self, id: u64) -> Result<Option<(Vec<f32>, Attrs)>> {
        self.get_in(&self.snapshot(), id)
    }

    fn get_in(&self, snap: &Snapshot, id: u64) -> Result<Option<(Vec<f32>, Attrs)>> {
        let fields = &self.config.fields;
        match locate_at(&snap.view, id, snap.seq) {
            None => Ok(None),
            Some(Place::Growing(r)) => {
                let g = snap.view.growing.read().unwrap();
                let attrs = if fields.is_empty() { Attrs::default() } else { decode_row(g.attrs(r), fields)? };
                Ok(Some((g.vector(r).to_vec(), attrs)))
            }
            Some(Place::Sealed(i, pos)) => {
                let s = &snap.view.sealed[i];
                let attrs = match &s.attrs {
                    Some(a) => a.row(pos, fields)?,
                    None => Attrs::default(),
                };
                let v = match &s.body {
                    Body::Flat(store) => store.vector(pos).to_vec(),
                    Body::Disk { index, .. } => {
                        let mut v = vec![0.0; index.dim()];
                        index.read_node_vectors(s.raw.as_ref().expect("SSD segment"), index.node_of(pos), &mut v)?;
                        v
                    }
                };
                Ok(Some((v, attrs)))
            }
        }
    }

    pub fn search(&self, query: &[f32], k: usize, opts: &SearchOptions) -> Result<Vec<SearchResult>> {
        Ok(self.search_with_stats(query, k, opts)?.0)
    }

    /// The `k` nearest live rows over all segments, closest first, as of
    /// the last write applied when the search starts.
    pub fn search_with_stats(
        &self,
        query: &[f32],
        k: usize,
        opts: &SearchOptions,
    ) -> Result<(Vec<SearchResult>, SearchStats)> {
        self.search_in(&self.snapshot(), query, k, opts)
    }

    fn search_in(
        &self,
        snap: &Snapshot,
        query: &[f32],
        k: usize,
        opts: &SearchOptions,
    ) -> Result<(Vec<SearchResult>, SearchStats)> {
        let cfg = &self.config;
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
        let filter = match &opts.filter {
            None => None,
            Some(f) => match f.compile(&cfg.fields)? {
                Compiled::All => None,
                Compiled::Nothing => return Ok((Vec::new(), st)),
                c => Some(c),
            },
        };
        let seq = snap.seq;

        let n = snap.view.growing.read().unwrap().rows_at(seq);
        if n > 0 {
            st.segments += 1;
            let matching = filter.as_ref().map(|c| {
                c.eval(snap.view.growing.read().unwrap().cols.as_ref().expect("a filter with fields"))
            });
            // In chunks, each under its own read lock: a write waits for
            // one chunk, not a whole scan (and searches arriving meanwhile
            // wait for the write). Rows the writer adds or deletes between
            // chunks are newer than the snapshot.
            let chunk = growing_chunk_rows(cfg.dim);
            let mut start = 0;
            while start < n {
                let end = (start + chunk).min(n);
                let g = snap.view.growing.read().unwrap();
                let rows = &g.data[start * cfg.dim..end * cfg.dim];
                let push = |r: usize, d: f32| top.push(g.ids[start + r], d);
                let scanned = match &matching {
                    None => segment::scan(kern, cfg.metric, &q, rows, |r| g.live_at(start + r, seq), push),
                    Some(m) => {
                        let admit = |r: usize| m.get(start + r) && g.live_at(start + r, seq);
                        segment::scan(kern, cfg.metric, &q, rows, admit, push)
                    }
                };
                st.scanned += scanned;
                if matching.is_some() {
                    st.matched += scanned;
                }
                start = end;
            }
        }
        for s in &snap.view.sealed {
            st.segments += 1;
            let deleted = AsOf::new(&s.tomb, seq);
            let matching = filter.as_ref().map(|c| {
                let mut m = c.eval(&s.attrs.as_ref().expect("a filter with fields").columns);
                deleted.remove_from(&mut m);
                m
            });
            if let Some(m) = &matching {
                let n = m.count();
                st.matched += n;
                if n == 0 {
                    continue;
                }
            }
            match &s.body {
                Body::Flat(store) => {
                    let push = |p: usize, d: f32| top.push(store.id(p), d);
                    st.scanned += match &matching {
                        None => segment::scan(kern, cfg.metric, &q, store.raw_vectors(), |p| !deleted.deleted(p), push),
                        Some(m) => segment::scan(kern, cfg.metric, &q, store.raw_vectors(), |p| m.get(p), push),
                    };
                }
                Body::Disk { index, .. } => {
                    let res = match &matching {
                        None => {
                            let live = s.live() as f64 / s.positions() as f64;
                            self.search_graph(index, query, k, opts, live, &|p| !deleted.deleted(p), &mut st)?
                        }
                        Some(m) => self.search_disk_filtered(index, query, k, opts, m, &mut st)?,
                    };
                    res.iter().for_each(|r| top.push(r.id, r.distance));
                }
            }
        }
        Ok((top.into_sorted(), st))
    }

    /// Seals the growing segment into a flat segment file and writes the
    /// tombstones of every segment that gained deletes, then commits both
    /// with a new manifest and starts a new WAL. Does nothing when there is
    /// nothing to write. Searches go on meanwhile; writes wait.
    pub fn flush(&self) -> Result<()> {
        let mut w = self.writer.lock().unwrap();
        let r = self.flush_locked(&mut w);
        if r.is_ok() {
            w.flush_error = None;
        }
        r
    }

    fn flush_locked(&self, w: &mut Writer) -> Result<()> {
        let view = self.view();
        let g = view.growing.read().unwrap();
        let dirty: Vec<usize> = (0..view.sealed.len())
            .filter(|&i| view.sealed[i].tomb.count() as u64 != w.manifest.segments[i].deleted)
            .collect();
        if g.rows() == 0 && dirty.is_empty() && w.wal.batches() == 0 {
            return Ok(());
        }
        let cfg = &self.config;
        let mut m = w.manifest.clone();
        m.generation += 1;
        let mut created: Vec<PathBuf> = Vec::new();
        let mut superseded: Vec<PathBuf> = Vec::new();
        let result = (|| -> Result<(Option<Sealed>, Wal)> {
            // 1. The growing segment's live rows, as a flat segment.
            let fresh = if g.live() > 0 {
                let (store, live) = live_rows(&g, cfg.dim)?;
                let id = m.next_segment;
                m.next_segment += 1;
                let path = self.dir.join(flat_name(id));
                created.push(path.clone());
                write_flat_segment(&path, cfg.metric, &store)?;
                let mut files = vec![path];
                let mut attrs = None;
                if !cfg.fields.is_empty() {
                    let ap = self.dir.join(attrs_name(id));
                    created.push(ap.clone());
                    let mut aw = AttrWriter::create(&ap, &cfg.fields, live.len())?;
                    for (pos, &r) in live.iter().enumerate() {
                        aw.push(pos, g.attrs(r))?;
                    }
                    attrs = Some(aw.finish()?);
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
                let s = &view.sealed[i];
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
            let wal = Wal::create(&self.dir, m.wal_gen, cfg.dim)?;
            created.push(wal.path().to_path_buf());
            m.flushed_seq = w.next_seq - 1;
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
        drop(g);
        // The new manifest is in place, so the state follows it. Until the
        // directory fsync succeeds the rename may not survive a crash, so
        // the old files stay (the next open removes them).
        let durable = sync_parent_dir(&self.dir.join(MANIFEST));
        superseded.push(w.wal.path().to_path_buf());
        w.wal = wal;
        w.manifest = m;
        let mut sealed = view.sealed.clone();
        sealed.extend(fresh.map(Arc::new));
        let growing = Arc::new(RwLock::new(Growing::new(cfg.dim, &cfg.fields)));
        *self.view.write().unwrap() = Arc::new(View { sealed, growing });
        if let Err(e) = durable {
            // A crash could bring back the old manifest, which does not know
            // the new WAL: take no writes that would go there.
            w.wal.poison();
            return Err(e);
        }
        for p in superseded {
            let _ = fs::remove_file(p);
        }
        Ok(())
    }
}

/// Where a live row is.
enum Place {
    Growing(usize),
    /// Index into the view's sealed segments, and the position.
    Sealed(usize, usize),
}

/// The row of `id` live as of write `seq` in `view`. An id's rows are
/// newer in newer segments, so the newest that is not deleted is the one.
fn locate_at(view: &View, id: u64, seq: u64) -> Option<Place> {
    if let Some(r) = view.growing.read().unwrap().row_at(id, seq) {
        return Some(Place::Growing(r));
    }
    for (i, s) in view.sealed.iter().enumerate().rev() {
        if let Some(pos) = s.locate(id)
            && !s.tomb.deleted_as_of(pos, seq)
        {
            return Some(Place::Sealed(i, pos));
        }
    }
    None
}

/// Deletes the live row of `id`, wherever it is, by write `seq`. Only the
/// writer calls this, holding `g` (the growing segment of the view whose
/// sealed segments are `sealed`).
fn delete_live(sealed: &[Arc<Sealed>], g: &mut Growing, id: u64, seq: u64) -> bool {
    if g.delete(id, seq) {
        return true;
    }
    // Newest first: a re-inserted id is most likely in a recent segment.
    for s in sealed.iter().rev() {
        if let Some(pos) = s.find(id) {
            s.tomb.delete(pos, seq);
            return true;
        }
    }
    false
}

/// `v` as stored: normalized for the metrics that need it.
fn prepare_vector(metric: Metric, v: &[f32], out: &mut Vec<f32>) {
    out.clear();
    out.extend_from_slice(v);
    if metric.needs_normalization() {
        normalize(out);
    }
}

impl Collection {
    /// The graph walk over one SSD segment, `admitted` of whose rows can be
    /// returned.
    #[allow(clippy::too_many_arguments)]
    fn search_graph(
        &self,
        index: &DiskIndex,
        query: &[f32],
        k: usize,
        opts: &SearchOptions,
        admitted: f64,
        admit: &dyn Fn(usize) -> bool,
        st: &mut SearchStats,
    ) -> Result<Vec<SearchResult>> {
        let params = DiskSearchParams {
            l: if opts.adapt_l { boosted_l(opts.l.max(k), admitted, MAX_BOOST) } else { opts.l.max(k) },
            beam_width: opts.beam_width,
            rerank: opts.rerank,
        };
        let (res, ds) = index.search_filtered(query, k, &params, admit)?;
        st.reads += ds.reads;
        st.rounds += ds.rounds;
        st.expanded += ds.expanded;
        Ok(res)
    }

    /// One SSD segment under a filter that `matching` (live positions) holds.
    fn search_disk_filtered(
        &self,
        index: &DiskIndex,
        query: &[f32],
        k: usize,
        opts: &SearchOptions,
        matching: &Bitmap,
        st: &mut SearchStats,
    ) -> Result<Vec<SearchResult>> {
        let n = matching.count();
        let admit = |p: usize| matching.get(p);
        let l = opts.l.max(k);
        let scan = match opts.filter_mode {
            FilterMode::Scan => true,
            FilterMode::Graph => false,
            FilterMode::Auto => prefer_scan(n, index.len(), l, opts.beam_width, index.is_direct()),
        };
        if !scan {
            st.graph_segments += 1;
            let params = DiskSearchParams {
                l: if opts.adapt_l { boosted_l(l, n as f64 / index.len() as f64, MAX_FILTER_BOOST) } else { l },
                beam_width: opts.beam_width,
                rerank: opts.rerank,
            };
            let (res, ds) = index.search_filtered(query, k, &params, &admit)?;
            st.reads += ds.reads;
            st.rounds += ds.rounds;
            st.expanded += ds.expanded;
            // The walk should meet about `l` matching nodes; far fewer means
            // the matches lie away from its path (or the boost was capped).
            if opts.filter_mode == FilterMode::Graph || ds.admitted * 2 >= l.min(n) {
                return Ok(res);
            }
            st.fallbacks += 1;
        }
        st.scan_segments += 1;
        let (res, ds) = index.search_subset(query, k, l, &mut matching.ones(), &admit)?;
        st.reads += ds.reads;
        st.rounds += ds.rounds;
        st.pq_scanned += ds.pq_distances;
        Ok(res)
    }
}

/// Largest factor the candidate list of a walk is scaled by for deleted rows.
const MAX_BOOST: f64 = 8.0;
/// The same for a filter. `Auto` scans a segment where fewer than 1 / this
/// of the rows match, as the walk could not be scaled enough.
const MAX_FILTER_BOOST: f64 = 16.0;
/// Rows a filtered scan scores in the time of one round of a walk (one
/// read latency plus its CPU). Measured on SIFT1M (128 dimensions, 32-byte
/// PQ codes), 1 thread: a round took about 140 µs with direct I/O (reads
/// capped at 100,000 per second) and 16 µs from the page cache; a scan
/// about 40 ns per matching row (bitmap, PQ distance, code fetch).
const SCAN_ROWS_PER_ROUND_DIRECT: f64 = 3500.0;
const SCAN_ROWS_PER_ROUND_CACHED: f64 = 400.0;

/// Whether a filtered SSD segment with `matching` of its `positions` rows
/// is cheaper to scan than to walk with beam width `width`: the walk takes
/// about L / W + 4 rounds at its scaled L, the scan time grows with the
/// rows it scores.
fn prefer_scan(matching: usize, positions: usize, l: usize, width: usize, direct: bool) -> bool {
    let f = matching as f64 / positions as f64;
    if f < 1.0 / MAX_FILTER_BOOST {
        return true;
    }
    let rounds = boosted_l(l, f, MAX_FILTER_BOOST) as f64 / width.max(1) as f64 + 4.0;
    let per_round = if direct { SCAN_ROWS_PER_ROUND_DIRECT } else { SCAN_ROWS_PER_ROUND_CACHED };
    (matching as f64) < rounds * per_round
}

/// Candidate list size for an SSD segment where only `admitted` of the
/// rows can be returned: the walk passes through rejected nodes, so it
/// needs a longer list to expand as many returnable ones.
fn boosted_l(l: usize, admitted: f64, max_boost: f64) -> usize {
    let boost = (1.0 / admitted.max(1e-9)).min(max_boost);
    (l as f64 * boost).ceil() as usize
}

/// The growing segment's live rows, copied (it stays intact if the flush
/// fails), and their rows in the growing segment.
fn live_rows(g: &Growing, dim: usize) -> Result<(crate::store::InMemoryStore, Vec<usize>)> {
    let mut store = crate::store::InMemoryStore::with_capacity(dim, g.live())?;
    let live: Vec<usize> = (0..g.rows()).filter(|&r| !g.is_dead(r)).collect();
    for &r in &live {
        store.push(g.ids[r], g.vector(r))?;
    }
    Ok((store, live))
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
            Sealed::flat(e.id, store, None, vec![path])
        }
        SegmentKind::Disk => {
            let path = dir.join(disk_name(e.id));
            let index = DiskIndex::open(&path, &DiskOpenOptions { direct: opts.direct_io, io: opts.io })?;
            if index.metric() != cfg.metric || index.dim() != cfg.dim {
                return Err(mismatch("metric or dimension"));
            }
            Sealed::disk(e.id, index, None, vec![path])?
        }
    };
    if s.positions() as u64 != e.positions {
        return Err(mismatch("row count"));
    }
    if !cfg.fields.is_empty() {
        let path = dir.join(attrs_name(e.id));
        s.attrs = Some(SegmentAttrs::open(&path, &cfg.fields, s.positions())?);
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
impl Collection {
    pub(crate) fn wal_path(&self) -> PathBuf {
        self.writer.lock().unwrap().wal.path().to_path_buf()
    }

    pub(crate) fn wal_bytes(&self) -> u64 {
        self.writer.lock().unwrap().wal.bytes()
    }

    /// Makes every later WAL append fail, as a failed fsync does.
    pub(crate) fn poison_wal(&self) {
        self.writer.lock().unwrap().wal.poison();
    }

    pub(crate) fn sealed(&self) -> Vec<Arc<Sealed>> {
        self.view().sealed.clone()
    }

    /// Ids with a live row now, one entry per live row.
    pub(crate) fn live_ids(&self) -> Vec<u64> {
        let v = self.view();
        let mut out = Vec::new();
        for s in &v.sealed {
            out.extend((0..s.positions()).filter(|&p| !s.tomb.is_set(p)).map(|p| s.id_at(p)));
        }
        let g = v.growing.read().unwrap();
        out.extend((0..g.rows()).filter(|&r| !g.is_dead(r)).map(|r| g.ids[r]));
        out
    }
}

#[cfg(test)]
mod tests;
