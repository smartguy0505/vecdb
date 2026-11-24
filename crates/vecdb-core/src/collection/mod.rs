//! Collections: the storage engine of one shard. A collection is a
//! directory of immutable segment files plus a write-ahead log, tied
//! together by a manifest.
//!
//! ```text
//! <dir>/MANIFEST                 which files make up the collection (atomic replace)
//!       LOCK                     held while a process has the collection open
//!       wal-<gen>.log            writes not flushed yet: one log per frozen
//!                                segment, and the current one
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
//! searched exactly. Once its vectors reach `seal_bytes` it is frozen, and a
//! new growing segment and a new WAL start. A flush (see `flush`) then turns
//! the frozen segment's live rows into a flat segment file, writes the
//! tombstones of every segment that gained deletes, and commits all of it
//! with a new manifest, which no longer needs the frozen segment's WAL.
//!
//! **Index builds and compaction** ([`Collection::maintain`]) replace sealed
//! segments with new ones built from their live rows, and commit the swap
//! with a manifest. Small flat segments are merged into an SSD index once
//! they hold `index_bytes` of vectors; SSD segments with many deleted rows
//! are rebuilt without them.
//!
//! **Read path.** A search runs on every segment, on several at once if a
//! [`SearchPool`] has threads to spare (see `search`), and merges the top-k
//! lists. Every segment returns exact distances, so they compare
//! directly.
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
//! after that. Opening replays the logs on top of the manifest's state and
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
//! WAL record, so a crash keeps or drops batches whole.
//!
//! **Flushes and builds** run while searches and writes go on: they write
//! their files without locks, and take the writer lock only to replace the
//! segment list (see `flush` and `maintain`).

mod background;
mod bitmap;
pub(crate) mod bytes;
mod columns;
pub mod filter;
mod flush;
mod maintain;
mod manifest;
pub mod meta;
pub(crate) mod search;
mod segment;
mod wal;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::{self, File, OpenOptions as FsOpenOptions};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};

use crate::disk::{DiskIndex, DiskOpenOptions, IoMode, Layout};
use crate::distance::{Metric, check_finite, normalize};
use crate::error::{Error, Result};
use crate::format::{read_flat_segment, sync_parent_dir};
use crate::graph::GraphParams;
use crate::pool::SearchPool;
use crate::quant::Codec;
use crate::store::{VectorStore, check_dim};
use background::{Background, Room, Workers};
use columns::{SegmentAttrs, decode_row, encode_row};
use manifest::{MANIFEST, Manifest, SegmentEntry, SegmentKind};
use segment::{Body, Frozen, Growing, Sealed, Tombstones, attrs_name, del_name, disk_name, flat_name};
use wal::{Event, LoggedOp, Wal, replay_chain, wal_generation, wal_name};

pub use background::BuildLimit;
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
    /// The growing, frozen and flat segments are scanned exactly on every
    /// search (about 30 ns per live 128-dim row on one core, measured), so
    /// the two thresholds bound that cost: once flushes and builds have
    /// caught up, at most `seal_bytes` + `index_bytes` of live vectors are
    /// scanned, 96 MiB with the defaults.
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

    /// Whether a row can be in the collection.
    pub(crate) fn check_row(&self, vector: &[f32], attrs: &Attrs) -> Result<()> {
        if vector.len() != self.dim {
            return Err(Error::DimensionMismatch { expected: self.dim, got: vector.len() });
        }
        check_finite(vector)?;
        attrs.validate(&self.fields)
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
    /// Flush full segments and build indexes on background threads. Without,
    /// the write that fills a segment flushes it, and indexes are built by
    /// [`Engine::maintain`] only.
    pub background: bool,
    /// With `background`: writes wait while flat segments hold more than
    /// this many bytes of vectors (they are in RAM, and every search scans
    /// them) and an index build or a merge that will take them is due.
    /// Builds run one at a time, so that is after the one that is running,
    /// whatever its size, and that of other collections sharing
    /// [`builds`](Self::builds): minutes, if it is a compaction of a large
    /// segment. 0 = no limit.
    pub max_unindexed_bytes: u64,
    /// Background builds that may run at once, over all collections given
    /// the same limit. `None`: one per collection.
    pub builds: Option<Arc<BuildLimit>>,
    /// Threads for index builds. 0 = one per CPU, or one per two CPUs for
    /// background builds, which share the CPUs with searches.
    pub threads: usize,
    /// RAM an index build may use for one partition.
    pub build_memory: u64,
    /// Read SSD indexes with `O_DIRECT`.
    pub direct_io: bool,
    pub io: IoMode,
    /// Threads that help searches: a search then runs on several segments
    /// (and parts of those in RAM) at once, as far as there are threads and
    /// CPUs to spare. `None`: each search runs on its caller's thread.
    pub pool: Option<Arc<SearchPool>>,
    /// Called with a line of progress during index builds.
    pub progress: Option<fn(&str)>,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            durability: Durability::Sync,
            background: true,
            max_unindexed_bytes: 1 << 30,
            builds: None,
            threads: 0,
            build_memory: 2 << 30,
            direct_io: true,
            io: IoMode::Auto,
            pool: None,
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
    /// Segments frozen and not flushed yet, and their rows.
    pub frozen: usize,
    pub frozen_rows: usize,
    /// RAM for the attributes of the growing and frozen segments, and
    /// their index.
    pub growing_attr_bytes: usize,
    pub segments: Vec<SegmentInfo>,
    /// Bytes of every log still needed.
    pub wal_bytes: u64,
    /// Sequence number the next write gets.
    pub next_seq: u64,
    /// Deletes kept with their sequence numbers for open searches.
    pub delete_records: usize,
    /// Why the last flush failed, until one succeeds.
    pub flush_error: Option<String>,
    /// The same for background builds.
    pub build_error: Option<String>,
    /// Since the collection was opened: frozen segments flushed, and builds
    /// and compactions committed.
    pub flushes: u64,
    pub builds: u64,
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
    /// Waiting to be flushed, oldest first.
    frozen: Vec<Arc<Frozen>>,
    /// Appended to by the writer.
    growing: Arc<RwLock<Growing>>,
}

impl View {
    /// The segments in RAM, newest first.
    fn memory(&self) -> impl Iterator<Item = &Arc<RwLock<Growing>>> {
        [&self.growing].into_iter().chain(self.frozen.iter().rev().map(|f| &f.rows))
    }
}

/// What only the writer touches.
struct Writer {
    /// The log writes go to: the growing segment's.
    wal: Wal,
    /// Earlier logs still needed, oldest first: the frozen segments'.
    closed: Vec<Wal>,
    /// Sequence number the next write gets.
    next_seq: u64,
    /// Since the collection was opened: batches logged, groups they were
    /// logged in, fsyncs.
    batches: u64,
    groups: u64,
    syncs: u64,
}

impl Writer {
    fn new(wal: Wal, closed: Vec<Wal>, next_seq: u64) -> Self {
        Self { wal, closed, next_seq, batches: 0, groups: 0, syncs: 0 }
    }
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
    c: &'a Engine,
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
pub(crate) struct Snapshot<'a> {
    c: &'a Engine,
    view: Arc<View>,
    seq: u64,
}

impl Drop for Snapshot<'_> {
    fn drop(&mut self) {
        self.c.unregister(self.seq);
    }
}

/// An open collection: an [`Engine`], which has all the operations, and its
/// background threads (with [`OpenOptions::background`]), which stop when
/// it is dropped. What they had not flushed or built yet is done after the
/// next open.
pub struct Collection {
    // First, so the threads are gone before the lock is released.
    _workers: Workers,
    engine: Arc<Engine>,
}

impl Collection {
    /// Creates a collection in `dir` (created if missing; must not hold one).
    pub fn create(dir: impl AsRef<Path>, config: CollectionConfig, opts: OpenOptions) -> Result<Self> {
        Self::start(Engine::create(dir.as_ref(), config, opts)?)
    }

    /// Opens the collection in `dir`: loads its segments, replays the logs
    /// and removes files the manifest does not refer to.
    pub fn open(dir: impl AsRef<Path>, opts: OpenOptions) -> Result<(Self, OpenReport)> {
        let (engine, report) = Engine::open(dir.as_ref(), opts)?;
        Ok((Self::start(engine)?, report))
    }

    fn start(engine: Engine) -> Result<Self> {
        let engine = Arc::new(engine);
        let workers = Workers::start(&engine)?;
        Ok(Self { _workers: workers, engine })
    }
}

impl Deref for Collection {
    type Target = Engine;

    fn deref(&self) -> &Engine {
        &self.engine
    }
}

/// A collection's operations, for any number of threads at once (`&self`
/// everywhere). Writes are applied one batch at a time, in order; each
/// search reads a snapshot (see the module docs).
pub struct Engine {
    dir: PathBuf,
    _lock: File,
    opts: OpenOptions,
    config: Arc<CollectionConfig>,
    /// The manifest on disk (its `segments` in the order of the view's
    /// sealed segments). Held while a flush or an index build commits;
    /// taken before `writer`.
    manifest: Mutex<Manifest>,
    /// Held while a batch is logged and applied, and while the view is
    /// replaced.
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
    /// One flush at a time, and one `maintain` or `optimize`.
    flushing: Mutex<()>,
    maintaining: Mutex<()>,
    /// What the background threads, and those waiting for them, share.
    background: Background,
    #[cfg(test)]
    hook: Mutex<Option<Hook>>,
}

/// Called at named points of a flush or a build, to have something happen
/// right there.
#[cfg(test)]
type Hook = Box<dyn FnMut(&Engine, &'static str) + Send>;

impl Engine {
    fn create(dir: &Path, config: CollectionConfig, opts: OpenOptions) -> Result<Self> {
        config.validate()?;
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
        let wal = Wal::create(dir, 1, config.dim, true)?;
        sync_parent_dir(wal.path())?;
        manifest.write(dir)?;
        let growing = Arc::new(RwLock::new(Growing::new(config.dim, &config.fields)));
        let view = View { sealed: Vec::new(), frozen: Vec::new(), growing };
        Ok(Self::assemble(dir, lock, opts, config, manifest, view, Writer::new(wal, Vec::new(), 1)))
    }

    fn open(dir: &Path, opts: OpenOptions) -> Result<(Self, OpenReport)> {
        let lock = lock_dir(dir)?;
        let manifest = Manifest::read(dir)?;
        let config = manifest.config.clone();
        let mut sealed = Vec::with_capacity(manifest.segments.len());
        for e in &manifest.segments {
            sealed.push(Arc::new(load_segment(dir, &config, e, &opts)?));
        }
        let mut growing = Growing::new(config.dim, &config.fields);
        let mut frozen: Vec<Arc<Frozen>> = Vec::new();
        let mut next_seq = manifest.flushed_seq + 1;
        let (mut vector, mut row) = (Vec::new(), Vec::new());
        // Each log but the last held the writes of a frozen segment, and
        // becomes one again.
        let chain = replay_chain(dir, manifest.wal_gen, config.dim, next_seq, |event| {
            let (seq, op) = match event {
                Event::Op(seq, op) => (seq, op),
                Event::EndOfLog { generation } => {
                    let full = std::mem::replace(&mut growing, Growing::new(config.dim, &config.fields));
                    let rows = Arc::new(RwLock::new(full));
                    frozen.push(Arc::new(Frozen { rows, last_seq: next_seq - 1, wal: generation }));
                    return Ok(());
                }
            };
            match op {
                LoggedOp::Upsert { id, vector: v, attrs } => {
                    let a = Attrs::decode(attrs)?;
                    a.validate(&config.fields).map_err(|e| Error::Corrupt(format!("WAL write {seq}: {e}")))?;
                    prepare_vector(config.metric, v, &mut vector);
                    row.clear();
                    if !config.fields.is_empty() {
                        encode_row(&a, &config.fields, &mut row);
                    }
                    delete_live(&sealed, &frozen, &mut growing, id, seq);
                    growing.push(id, &vector, &row, seq);
                }
                LoggedOp::Delete { id } => {
                    delete_live(&sealed, &frozen, &mut growing, id, seq);
                }
            }
            next_seq = seq + 1;
            Ok(())
        })?;
        // Only once everything the manifest names has loaded.
        if Manifest::is_old(dir)? {
            manifest.write(dir)?;
        }
        let mut removed = remove_unreferenced(dir, &manifest)?;
        removed.extend(chain.dropped);
        removed.sort();
        let frozen_at: Vec<u64> = frozen.iter().map(|f| f.last_seq).collect();
        let view = View { sealed, frozen, growing: Arc::new(RwLock::new(growing)) };
        let writer = Writer::new(chain.active, chain.closed, next_seq);
        let c = Self::assemble(dir, lock, opts, config, manifest, view, writer);
        frozen_at.into_iter().for_each(|seq| c.register(seq));
        Ok((c, OpenReport { replay: chain.report, removed }))
    }

    fn assemble(
        dir: &Path,
        lock: File,
        opts: OpenOptions,
        config: CollectionConfig,
        manifest: Manifest,
        view: View,
        writer: Writer,
    ) -> Self {
        Self {
            dir: dir.to_path_buf(),
            _lock: lock,
            opts,
            config: Arc::new(config),
            manifest: Mutex::new(manifest),
            visible: AtomicU64::new(writer.next_seq - 1),
            writer: Mutex::new(writer),
            queue: Mutex::new(Queue::default()),
            queue_cv: Condvar::new(),
            view: RwLock::new(Arc::new(view)),
            snapshots: Mutex::new(BTreeMap::new()),
            flushing: Mutex::new(()),
            maintaining: Mutex::new(()),
            background: Background::default(),
            #[cfg(test)]
            hook: Mutex::new(None),
        }
    }

    /// A named point (see `Hook`).
    #[inline(always)]
    fn at(&self, _point: &'static str) {
        #[cfg(test)]
        {
            let hook = self.hook.lock().unwrap().take();
            if let Some(mut h) = hook {
                h(self, _point);
                self.hook.lock().unwrap().get_or_insert(h);
            }
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
    pub(crate) fn snapshot(&self) -> Snapshot<'_> {
        let mut open = self.snapshots.lock().unwrap();
        let (view, seq) = {
            let v = self.view.read().unwrap();
            (v.clone(), self.visible.load(Ordering::Acquire))
        };
        *open.entry(seq).or_insert(0) += 1;
        drop(open);
        Snapshot { c: self, view, seq }
    }

    /// Keeps the sequence numbers of deletes after write `seq` (the last
    /// one applied, or an earlier one already registered).
    fn register(&self, seq: u64) {
        *self.snapshots.lock().unwrap().entry(seq).or_insert(0) += 1;
    }

    fn unregister(&self, seq: u64) {
        let mut open = self.snapshots.lock().unwrap();
        if let Some(n) = open.get_mut(&seq) {
            *n -= 1;
            if *n == 0 {
                open.remove(&seq);
            }
        }
    }

    /// Live rows now.
    pub fn len(&self) -> usize {
        let v = self.view();
        let memory: usize = v.memory().map(|g| g.read().unwrap().live()).sum();
        memory + v.sealed.iter().map(|s| s.live()).sum::<usize>()
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
        let (wal_bytes, next_seq, writes) = {
            let mut w = self.writer.lock().unwrap();
            (w.logs_mut().map(|l| l.bytes()).sum(), w.next_seq, (w.batches, w.groups, w.syncs))
        };
        let bg = self.background.progress();
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
        let (frozen_rows, frozen_live, frozen_attrs) = v.frozen.iter().fold((0, 0, 0), |(r, l, a), f| {
            let g = f.rows.read().unwrap();
            (r + g.rows(), l + g.live(), a + g.attr_ram())
        });
        let g = v.growing.read().unwrap();
        CollectionStats {
            live: g.live() + frozen_live + v.sealed.iter().map(|s| s.live()).sum::<usize>(),
            growing_rows: g.rows(),
            growing_live: g.live(),
            frozen: v.frozen.len(),
            frozen_rows,
            growing_attr_bytes: g.attr_ram() + frozen_attrs,
            segments,
            wal_bytes,
            next_seq,
            delete_records: v.sealed.iter().map(|s| s.tomb.recent_len()).sum(),
            flush_error: bg.flush_error,
            build_error: bg.build_error,
            flushes: bg.flushes,
            builds: bg.builds,
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
    /// none.
    ///
    /// Waits for room first when flushes or index builds are far behind
    /// ([`Error::Unavailable`] if that is because they keep failing). A
    /// failed flush alone does not fail a write, which is logged and
    /// applied; it is retried (see [`CollectionStats::flush_error`]).
    pub fn write(&self, ops: &[WriteOp]) -> Result<()> {
        self.write_ops(ops).map(|_| ())
    }

    /// As [`write`](Self::write); per op, whether its id had a live row.
    pub fn write_ops(&self, ops: &[WriteOp]) -> Result<Vec<bool>> {
        if ops.is_empty() {
            return Ok(Vec::new());
        }
        let p = self.prepare(ops)?;
        self.make_room()?;
        let r = self.enqueue(p);
        if !self.opts.background {
            self.flush_frozen();
        }
        r
    }

    /// Waits, or fails, while there is no room for another write.
    fn make_room(&self) -> Result<()> {
        loop {
            match self.room(&self.background.progress()) {
                Room::Yes => return Ok(()),
                Room::Wait => {
                    self.background.wait(|p| !matches!(self.room(p), Room::Wait));
                }
                Room::No(why) => {
                    // A last try, as only writes freeze (and, without
                    // background threads, flush).
                    let mut w = self.writer.lock().unwrap();
                    if self.view().frozen.len() < flush::MAX_FROZEN {
                        let r = self.freeze(&mut w);
                        self.background.flushed(&r.map(|()| false));
                    }
                    drop(w);
                    if !self.opts.background {
                        self.flush_frozen();
                    }
                    return match self.room(&self.background.progress()) {
                        Room::No(_) => Err(Error::Unavailable(why)),
                        _ => Ok(()),
                    };
                }
            }
        }
    }

    /// Has `p` logged and applied, with the batches waiting at the time.
    fn enqueue(&self, p: Prepared) -> Result<Vec<bool>> {
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
                cfg.check_row(vector, attrs)?;
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
                    had.push(delete_live(&view.sealed, &view.frozen, &mut g, id, seq));
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
        if full && view.frozen.len() < flush::MAX_FROZEN {
            if let Err(e) = self.freeze(w) {
                self.background.flushed(&Err(e));
            }
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
            Some(Place::Memory(m, r)) => {
                let g = m.read().unwrap();
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
}

/// Where a live row is.
enum Place<'a> {
    /// In the growing segment or a frozen one: its row there.
    Memory(&'a RwLock<Growing>, usize),
    /// Index into the view's sealed segments, and the position.
    Sealed(usize, usize),
}

/// The row of `id` live as of write `seq` in `view`. An id's rows are
/// newer in newer segments, so the newest that is not deleted is the one.
fn locate_at(view: &View, id: u64, seq: u64) -> Option<Place<'_>> {
    for m in view.memory() {
        if let Some(r) = m.read().unwrap().row_at(id, seq) {
            return Some(Place::Memory(m, r));
        }
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
/// other segments are `sealed` and `frozen`).
fn delete_live(sealed: &[Arc<Sealed>], frozen: &[Arc<Frozen>], g: &mut Growing, id: u64, seq: u64) -> bool {
    if g.delete(id, seq) {
        return true;
    }
    for f in frozen.iter().rev() {
        // Only the writer changes which rows are live.
        let here = f.rows.read().unwrap().row_of(id).is_some();
        if here {
            return f.rows.write().unwrap().delete(id, seq);
        }
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

pub(crate) fn lock_dir(dir: &Path) -> Result<File> {
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
    let mut keep: Vec<String> = Vec::new();
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
        // The logs from `wal_gen` on were replayed.
        if !is_ours(&name) || keep.contains(&name) || wal_generation(&name).is_some_and(|g| g >= m.wal_gen) {
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
    if wal_generation(base).is_some() {
        return true;
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
impl Engine {
    pub(crate) fn wal_path(&self) -> PathBuf {
        self.writer.lock().unwrap().wal.path().to_path_buf()
    }

    pub(crate) fn wal_bytes(&self) -> u64 {
        self.writer.lock().unwrap().wal.bytes()
    }

    /// Runs `f` at a moment when nothing is being committed or logged, with
    /// every log still needed: its path, length, and length known to be
    /// durable.
    pub(crate) fn quiesced<T>(&self, f: impl FnOnce(&[(PathBuf, u64, u64)]) -> T) -> T {
        let _m = self.manifest.lock().unwrap();
        let mut w = self.writer.lock().unwrap();
        let logs: Vec<_> = w.logs_mut().map(|l| (l.path().to_path_buf(), l.bytes(), l.synced())).collect();
        f(&logs)
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
        for m in v.memory() {
            let g = m.read().unwrap();
            out.extend((0..g.rows()).filter(|&r| !g.is_dead(r)).map(|r| g.ids[r]));
        }
        out
    }
}

#[cfg(test)]
mod tests;
