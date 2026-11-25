//! Index builds and compaction: replacing sealed segments with one new
//! segment built from their live rows.
//!
//! Jobs are run by the background builders (see `background`) or by
//! [`Engine::maintain`], until none is due. There are two lanes, by what a
//! job replaces, with one job at a time in each. On flat segments ([`Lane`]),
//! in this order:
//!
//! 1. **index:** they hold `index_bytes` of live vectors: build an SSD index
//!    from them. In the background not from all of them if they are far
//!    behind, but from the oldest, an eighth of `max_unindexed_bytes` (or
//!    `index_bytes`, if that is more): writes that wait have less to wait for.
//! 2. **drop:** one has no live rows left: remove it.
//! 3. **shrink:** they take `index_bytes`, and at least half of that is
//!    deleted rows: merge them into one flat segment. (So they never take
//!    much more than twice `index_bytes`, whatever is upserted.)
//!
//! On SSD segments:
//!
//! 1. **drop:** as above.
//! 2. **compact:** one has `compact_deleted` of its rows deleted: rebuild it
//!    without them.
//! 3. **merge:** there are more than `max_indexed_segments`: merge the
//!    smallest, so many that there are not.
//!
//! [`Engine::optimize`] takes the segments of both lanes, as they are when it
//! starts; later ones are left to the lanes.
//!
//! An output with fewer than [`MIN_SSD_ROWS`] rows is always a flat segment.
//! Each job commits with one manifest write; the files of the segments it
//! replaced are removed after that (searches that still hold those segments
//! read them through open files).
//!
//! Jobs run while searches and writes go on. A job takes the rows live in
//! its inputs and builds without locks, or a snapshot: that would keep the
//! sequence number of every delete made during the build. It commits in two
//! steps. First, with the rows deleted meanwhile (deletes and upserts of
//! their ids) written to the new segment's tombstone file, it installs the
//! manifest. Then, under the writer lock, it takes over the deletes made
//! since, with their numbers, and replaces the segments in the view.
//!
//! The new segment lacks rows that writes still only in the logs deleted
//! (the first step fsyncs the logs, so a power loss cannot take those writes
//! back). That is fine for a reopen, which replays them: deleting a row that
//! is already gone does nothing.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::sync::{Arc, MutexGuard};
use std::time::Instant;

use super::manifest::{MANIFEST, SegmentEntry, SegmentKind};
use super::columns::AttrWriter;
use super::segment::{Body, Sealed, attrs_name, del_name, disk_name, flat_name};
use super::{CollectionConfig, Engine, OpenOptions, Snapshot};
use crate::cancel::Cancel;
use crate::dataset::VectorSource;
use crate::disk::{DiskIndex, DiskOpenOptions, DiskWriteOptions, MergePrune, PartitionedBuildParams};
use crate::error::Result;
use crate::format::{sync_parent_dir, write_flat_segment};
use crate::quant::PqParams;
use crate::store::{InMemoryStore, VectorStore};

/// Smallest output written as an SSD index; smaller ones stay flat.
pub const MIN_SSD_ROWS: usize = 2048;

/// Jobs are of two kinds, by the segments they replace. One of each can run
/// at a time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Lane {
    Flat = 0,
    Ssd = 1,
}

impl Lane {
    fn has(self, s: &Sealed) -> bool {
        matches!(s.body, Body::Flat(_)) == (self == Lane::Flat)
    }
}

/// One job that [`Engine::maintain`] ran.
#[derive(Clone, Debug, PartialEq)]
pub struct JobReport {
    /// "index", "drop", "shrink", "compact", "merge" or "optimize".
    pub reason: &'static str,
    /// Segments replaced.
    pub inputs: Vec<u64>,
    /// The new segment, if any rows were left: its id and kind ("flat" or "ssd").
    pub output: Option<(u64, &'static str)>,
    pub rows: usize,
    /// Rows deleted while the job ran, carried into the new segment.
    pub carried_deletes: usize,
    pub seconds: f64,
}

impl Engine {
    /// Runs the index builds and compactions that are due (see the module
    /// docs), after a flush. Returns when they are done; cheap when nothing
    /// is due. Searches and writes go on meanwhile. Not needed with
    /// background threads, which do the same as it becomes due.
    pub fn maintain(&self) -> Result<Vec<JobReport>> {
        if self.no_job() {
            return Ok(Vec::new());
        }
        self.flush()?;
        let mut done = Vec::new();
        loop {
            let before = done.len();
            for lane in [Lane::Flat, Lane::Ssd] {
                let _one = self.lane(lane);
                while let Some((reason, inputs)) = self.next_job(lane) {
                    done.push(self.rebuild(&inputs, reason, self.opts.threads)?);
                }
            }
            if done.len() == before {
                return Ok(done);
            }
        }
    }

    /// Flushes, then merges every sealed segment into one, built from the
    /// live rows (an SSD index unless there are very few). Segments flushed
    /// meanwhile are not part of it.
    pub fn optimize(&self) -> Result<Option<JobReport>> {
        // In this order, as a job on SSD segments can take hours, and index
        // builds have to go on until it is done.
        let _ssd = self.lane(Lane::Ssd);
        let flat = self.lane(Lane::Flat);
        self.flush()?;
        let all: Vec<u64> = self.view().sealed.iter().map(|s| s.id).collect();
        if all.is_empty() {
            return Ok(None);
        }
        let _taken = Taken::new(self, &all);
        drop(flat);
        Ok(Some(self.rebuild(&all, "optimize", self.opts.threads)?))
    }

    /// Waits for its turn to run jobs of `lane`.
    fn lane(&self, lane: Lane) -> MutexGuard<'_, ()> {
        self.lanes[lane as usize].lock().unwrap()
    }

    /// The sealed segments of `lane` that no `optimize` has taken, oldest
    /// first.
    pub(super) fn lane_segments(&self, lane: Lane) -> Vec<Arc<Sealed>> {
        // In this order: what is taken is let go of after it has left the
        // view.
        let taken = self.optimizing.lock().unwrap();
        let view = self.view();
        view.sealed.iter().filter(|s| lane.has(s) && !taken.contains(&s.id)).cloned().collect()
    }

    pub(super) fn no_job(&self) -> bool {
        [Lane::Flat, Lane::Ssd].iter().all(|&l| self.next_job(l).is_none())
    }

    /// The next job due in `lane`: its reason and its input segments' ids.
    /// It stays so until its segments are replaced, so whoever runs it holds
    /// the lane.
    pub(super) fn next_job(&self, lane: Lane) -> Option<(&'static str, Vec<u64>)> {
        let cfg = &self.config;
        let segs = self.lane_segments(lane);
        let ids = |segs: &[Arc<Sealed>]| segs.iter().map(|s| s.id).collect::<Vec<u64>>();
        let bytes = |rows: usize| rows as u64 * cfg.dim as u64 * 4;
        let live: usize = segs.iter().map(|s| s.live()).sum();
        // Below MIN_SSD_ROWS the output would be flat again: not a job.
        if lane == Lane::Flat && bytes(live) >= cfg.index_bytes && live >= MIN_SSD_ROWS {
            let (most, mut rows, mut n) = (self.index_batch(), 0, 0);
            while n < segs.len() && (bytes(rows) < most || rows < MIN_SSD_ROWS) {
                rows += segs[n].live();
                n += 1;
            }
            return Some(("index", ids(&segs[..n])));
        }
        if let Some(s) = segs.iter().find(|s| s.live() == 0) {
            return Some(("drop", vec![s.id]));
        }
        if lane == Lane::Flat {
            let taken = bytes(segs.iter().map(|s| s.positions()).sum());
            let shrink = taken >= cfg.index_bytes && taken >= bytes(live).saturating_mul(2);
            return shrink.then(|| ("shrink", ids(&segs)));
        }
        let deleted = |s: &&Arc<Sealed>| s.tomb.count() as f64 >= cfg.compact_deleted as f64 * s.positions() as f64;
        if let Some(s) = segs.iter().find(deleted) {
            return Some(("compact", vec![s.id]));
        }
        if segs.len() > cfg.max_indexed_segments {
            let mut smallest: Vec<usize> = (0..segs.len()).collect();
            smallest.sort_by_key(|&i| (segs[i].live(), i));
            smallest.truncate(segs.len() - cfg.max_indexed_segments + 1);
            smallest.sort_unstable();
            return Some(("merge", smallest.iter().map(|&i| segs[i].id).collect()));
        }
        None
    }

    /// The bytes of vectors one index build takes from the flat segments, if
    /// there are more.
    fn index_batch(&self) -> u64 {
        match self.opts.max_unindexed_bytes {
            limit if self.opts.background && limit != 0 => (limit / 8).max(self.config.index_bytes),
            _ => u64::MAX,
        }
    }

    /// Replaces the segments `inputs` (ids, oldest first) with one segment
    /// of their live rows (see the module docs), built by `threads` threads
    /// (fewer while other builds run).
    pub(super) fn rebuild(&self, inputs: &[u64], reason: &'static str, threads: usize) -> Result<JobReport> {
        let t0 = Instant::now();
        let running = self.limit.running();
        let cfg = self.config.clone();
        // 1. The inputs and the rows live in them, and an id for the output:
        // taken now, so a flush during the build numbers its segment after
        // this one (the next manifest records both).
        let id = {
            let mut m = self.manifest.lock().unwrap();
            m.next_segment += 1;
            m.next_segment - 1
        };
        let snap = self.snapshot();
        let segs = inputs
            .iter()
            .map(|&id| {
                snap.view.sealed.iter().find(|s| s.id == id).cloned().ok_or_else(|| {
                    crate::error::Error::InvalidArgument(format!("segment {id} is not in the collection"))
                })
            })
            .collect::<Result<Vec<Arc<Sealed>>>>()?;
        let mut rows: Vec<(u32, u32)> = Vec::new();
        let dead: Vec<Vec<u64>> = segs.iter().map(|s| s.tomb.words_as_of(snap.seq)).collect();
        drop(snap);
        for (j, (s, deleted)) in segs.iter().zip(&dead).enumerate() {
            let live = (0..s.positions()).filter(|&p| deleted[p / 64] >> (p % 64) & 1 == 0);
            rows.extend(live.map(|p| (j as u32, p as u32)));
        }
        let ids: Vec<u64> = rows.iter().map(|&(j, p)| segs[j as usize].id_at(p as usize)).collect();

        // 2. The new segment and its files. `order`: for each of its
        // positions, the entry in `rows`.
        let mut created = Vec::new();
        let built = (|| -> Result<Option<(Sealed, Vec<Vec<u32>>)>> {
            if rows.is_empty() {
                return Ok(None);
            }
            let src = LiveRows::new(&segs, &rows, cfg.dim);
            let (mut sealed, order) = if rows.len() < MIN_SSD_ROWS || reason == "shrink" {
                let mut store = InMemoryStore::with_capacity(cfg.dim, rows.len())?;
                let mut v = vec![0.0f32; cfg.dim];
                for (i, &id) in ids.iter().enumerate() {
                    src.read_rows(i, &mut v)?;
                    store.push(id, &v)?;
                }
                let path = self.dir.join(flat_name(id));
                created.push(path.clone());
                write_flat_segment(&path, cfg.metric, &store)?;
                (Sealed::flat(id, store, None, vec![path]), (0..rows.len()).collect::<Vec<usize>>())
            } else {
                let path = self.dir.join(disk_name(id));
                created.push(path.clone());
                let params = build_params(&cfg, &self.opts, threads, self.background.cancel_build(&running));
                DiskIndex::build_partitioned(&src, Some(&ids), cfg.metric, &path, &params)?;
                let index = DiskIndex::open(&path, &DiskOpenOptions { direct: self.opts.direct_io, io: self.opts.io })?;
                // The index orders ids by node.
                let row_of: HashMap<u64, usize> = ids.iter().enumerate().map(|(r, &id)| (id, r)).collect();
                let order = (0..index.len()).map(|p| row_of[&index.id_at(p)]).collect();
                (Sealed::disk(id, index, None, vec![path])?, order)
            };
            // Where each input row went.
            let mut new_pos: Vec<Vec<u32>> = segs.iter().map(|s| vec![u32::MAX; s.positions()]).collect();
            for (np, &r) in order.iter().enumerate() {
                let (j, p) = rows[r];
                new_pos[j as usize][p as usize] = np as u32;
            }
            if !cfg.fields.is_empty() {
                // Rows are copied in each input's file order; only the
                // offsets table follows the new positions.
                let ap = self.dir.join(attrs_name(id));
                created.push(ap.clone());
                let mut w = AttrWriter::create(&ap, &cfg.fields, order.len())?;
                for (j, s) in segs.iter().enumerate() {
                    let a = s.attrs.as_ref().expect("segments of a collection with fields have attributes");
                    a.for_each_row(|p, row| match new_pos[j][p] {
                        u32::MAX => Ok(()),
                        np => w.push(np as usize, row),
                    })?;
                }
                sealed.attrs = Some(w.finish()?);
                sealed.files.push(ap);
            }
            Ok(Some((sealed, new_pos)))
        })();
        let cleanup = |created: &[std::path::PathBuf]| {
            created.iter().for_each(|p| {
                let _ = fs::remove_file(p);
            });
        };
        let built = match built {
            Ok(b) => b,
            Err(e) => {
                cleanup(&created);
                return Err(e);
            }
        };

        self.at("build: built");

        // 3. Commit, from the manifest as it is now (flushes may have come
        // in between, but none can from here on).
        let mut man = self.manifest.lock().unwrap();
        let mut m = man.clone();
        m.generation += 1;
        let result = (|| -> Result<(usize, Option<Snapshot>)> {
            let first = m.segments.iter().position(|e| inputs.contains(&e.id)).expect("inputs are in the manifest");
            let (mut carried, mut snap) = (0, None);
            if let Some((sealed, new_pos)) = &built {
                // The deletes so far, all logged (and then made durable):
                // under the writer lock, with a snapshot, so that the later
                // ones keep their numbers.
                let now: Vec<Vec<u64>> = {
                    let _w = self.writer.lock().unwrap();
                    snap = Some(self.snapshot());
                    segs.iter().map(|s| s.tomb.words()).collect()
                };
                self.at("build: deletes collected");
                for ((now, dead), new_pos) in now.iter().zip(&dead).zip(new_pos) {
                    for (i, w) in now.iter().zip(dead).map(|(n, d)| n & !d).enumerate() {
                        for b in (0..64).filter(|b| w >> b & 1 == 1) {
                            sealed.tomb.settle(new_pos[i * 64 + b] as usize);
                            carried += 1;
                        }
                    }
                }
                let (mut deleted, mut del_gen) = (0, 0);
                if carried > 0 {
                    let path = self.dir.join(del_name(id, m.generation));
                    created.push(path.clone());
                    deleted = sealed.tomb.write(&path, id, &sealed.tomb.words())? as u64;
                    del_gen = m.generation;
                }
                // The new segment takes the place of the oldest one it
                // replaces.
                m.segments.insert(
                    first,
                    SegmentEntry {
                        id,
                        kind: match sealed.body {
                            Body::Flat(_) => SegmentKind::Flat,
                            Body::Disk { .. } => SegmentKind::Disk,
                        },
                        positions: sealed.positions() as u64,
                        deleted,
                        del_gen,
                    },
                );
            }
            m.segments.retain(|e| !inputs.contains(&e.id));
            // The inputs' rows may be gone because of writes still only in
            // the logs. Those must not be lost after this: an upsert's new
            // row is only there.
            self.sync_logs()?;
            sync_parent_dir(&self.dir.join(MANIFEST))?;
            m.install(&self.dir)?;
            Ok((carried, snap))
        })();
        let (mut carried, snap) = match result {
            Ok(r) => r,
            Err(e) => {
                cleanup(&created);
                return Err(e);
            }
        };

        // Committed. As in a flush, the old files stay until the rename is
        // known to be durable.
        let durable = sync_parent_dir(&self.dir.join(MANIFEST));
        self.at("build: installed");
        let output = built.as_ref().map(|(s, _)| {
            (
                s.id,
                match s.body {
                    Body::Flat(_) => "flat",
                    Body::Disk { .. } => "ssd",
                },
            )
        });
        let mut old_files = Vec::new();
        {
            let _w = self.writer.lock().unwrap();
            let view = self.view();
            for (e, s) in man.segments.iter().zip(&view.sealed) {
                debug_assert_eq!(e.id, s.id);
                if inputs.contains(&e.id) {
                    old_files.extend(s.files.iter().cloned());
                    if e.del_gen != 0 {
                        old_files.push(self.dir.join(del_name(e.id, e.del_gen)));
                    }
                }
            }
            let mut fresh = built.map(|(s, new_pos)| {
                // Deletes since the tombstone file: only in the logs, with
                // their numbers (see `flush`).
                let seq = snap.as_ref().expect("taken").seq;
                let mut late = Vec::new();
                for (s, new_pos) in segs.iter().zip(&new_pos) {
                    late.extend(s.tomb.deleted_after(seq).into_iter().map(|(p, d)| (d, new_pos[p as usize])));
                }
                late.sort_unstable();
                for (d, p) in late {
                    s.tomb.delete(p as usize, d);
                    carried += 1;
                }
                Arc::new(s)
            });
            let mut sealed = Vec::with_capacity(view.sealed.len());
            for s in &view.sealed {
                if inputs.contains(&s.id) {
                    sealed.extend(fresh.take());
                } else {
                    sealed.push(s.clone());
                }
            }
            let next = super::View { sealed, frozen: view.frozen.clone(), growing: view.growing.clone() };
            *self.view.write().unwrap() = Arc::new(next);
        }
        *man = m;
        drop(man);
        drop(snap);
        let lanes = [Lane::Flat, Lane::Ssd].map(|l| segs.iter().any(|s| l.has(s)));
        drop(segs);
        self.at("build: replaced");
        durable?;
        for p in old_files {
            let _ = fs::remove_file(p);
        }
        self.background.built(lanes);
        Ok(JobReport {
            reason,
            inputs: inputs.to_vec(),
            output,
            rows: rows.len(),
            carried_deletes: carried,
            seconds: t0.elapsed().as_secs_f64(),
        })
    }
}

/// The segments a running `optimize` replaces: not for the lanes.
struct Taken<'a>(&'a Engine);

impl<'a> Taken<'a> {
    fn new(e: &'a Engine, ids: &[u64]) -> Self {
        *e.optimizing.lock().unwrap() = ids.iter().copied().collect::<BTreeSet<u64>>();
        Self(e)
    }
}

impl Drop for Taken<'_> {
    fn drop(&mut self) {
        self.0.optimizing.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.0.background.notify();
    }
}

fn build_params(cfg: &CollectionConfig, opts: &OpenOptions, threads: usize, cancel: Cancel) -> PartitionedBuildParams {
    let i = &cfg.index;
    PartitionedBuildParams {
        graph: cfg.graph_params(threads),
        pq: PqParams { threads, ..PqParams::new(i.pq_m) },
        write: DiskWriteOptions { layout: i.layout, codec: i.codec },
        memory_budget: opts.build_memory,
        partitions: 0,
        overlap: 2,
        partition_degree: 0,
        merge: MergePrune::Pq,
        scratch_dir: None,
        progress: opts.progress,
        cancel,
    }
}

/// The live rows of some sealed segments, in order, as a vector source for
/// a build. Rows come back as stored (already prepared for the metric).
struct LiveRows<'a> {
    segs: &'a [Arc<Sealed>],
    rows: &'a [(u32, u32)],
    dim: usize,
}

impl<'a> LiveRows<'a> {
    fn new(segs: &'a [Arc<Sealed>], rows: &'a [(u32, u32)], dim: usize) -> Self {
        Self { segs, rows, dim }
    }
}

impl VectorSource for LiveRows<'_> {
    fn dim(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        self.rows.len()
    }

    fn row_bytes(&self) -> usize {
        self.dim * 4
    }

    fn prepared(&self) -> bool {
        true
    }

    fn read_rows(&self, first: usize, out: &mut [f32]) -> Result<()> {
        // A run of rows from one SSD segment is read as one range of
        // nodes, through gaps of up to this many nodes.
        const GAP: u32 = 8;
        const MAX_RUN: u32 = 4096;
        let dim = self.dim;
        let count = out.len() / dim;
        let mut buf = Vec::new();
        let mut i = 0;
        while i < count {
            let (j, p) = self.rows[first + i];
            match &self.segs[j as usize].body {
                Body::Flat(store) => {
                    out[i * dim..(i + 1) * dim].copy_from_slice(store.vector(p as usize));
                    i += 1;
                }
                Body::Disk { index, .. } => {
                    let n0 = index.node_of(p as usize);
                    let (mut last, mut end) = (n0, i + 1);
                    while end < count {
                        let (j2, p2) = self.rows[first + end];
                        let n = if j2 == j { index.node_of(p2 as usize) } else { break };
                        if n < last || n > last + GAP || n - n0 >= MAX_RUN {
                            break;
                        }
                        last = n;
                        end += 1;
                    }
                    buf.resize((last - n0 + 1) as usize * dim, 0.0);
                    let raw = self.segs[j as usize].raw.as_ref().expect("SSD segment");
                    index.read_node_vectors(raw, n0, &mut buf)?;
                    for r in i..end {
                        let n = (index.node_of(self.rows[first + r].1 as usize) - n0) as usize;
                        out[r * dim..(r + 1) * dim].copy_from_slice(&buf[n * dim..(n + 1) * dim]);
                    }
                    i = end;
                }
            }
        }
        Ok(())
    }
}
