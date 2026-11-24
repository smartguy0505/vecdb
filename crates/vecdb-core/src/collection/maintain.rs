//! Index builds and compaction: replacing sealed segments with one new
//! segment built from their live rows.
//!
//! When a job is due, [`Collection::maintain`] first flushes (so the WAL is
//! empty and every tombstone is on disk), then runs jobs until none is due,
//! in this order:
//!
//! 1. **index:** the flat segments hold `index_bytes` of live vectors:
//!    build one SSD index from all of them.
//! 2. **drop:** a segment has no live rows left: remove it.
//! 3. **compact:** an SSD segment has `compact_deleted` of its rows
//!    deleted: rebuild it without them.
//! 4. **merge:** there are more than `max_indexed_segments` SSD segments:
//!    merge the two smallest.
//!
//! A job's output with fewer than [`MIN_SSD_ROWS`] rows is written as a flat
//! segment instead. Each job commits with one manifest write; the files of
//! the segments it replaced are removed after that (searches that still
//! hold those segments read them through open files).
//!
//! Jobs run while searches and writes go on. A job takes the rows live in
//! its inputs, builds without locks, then commits under the writer lock:
//! rows deleted while it ran (deletes and upserts of their ids) are deleted
//! in the new segment too, and written to its tombstone file in the same
//! commit, as a flush meanwhile may have moved them out of the WAL.

use std::collections::HashMap;
use std::fs;
use std::sync::Arc;
use std::time::Instant;

use super::manifest::{SegmentEntry, SegmentKind};
use super::columns::AttrWriter;
use super::segment::{Body, Sealed, attrs_name, del_name, disk_name, flat_name};
use super::{Collection, CollectionConfig};
use crate::dataset::VectorSource;
use crate::disk::{DiskIndex, DiskOpenOptions, DiskWriteOptions, MergePrune, PartitionedBuildParams};
use crate::error::Result;
use crate::format::write_flat_segment;
use crate::quant::PqParams;
use crate::store::{InMemoryStore, VectorStore};

/// Smallest output written as an SSD index; smaller ones stay flat.
pub const MIN_SSD_ROWS: usize = 2048;

/// One job [`Collection::maintain`] ran.
#[derive(Clone, Debug, PartialEq)]
pub struct JobReport {
    /// "index", "drop", "compact", "merge" or "optimize".
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

impl Collection {
    /// Runs the index builds and compactions that are due (see the module
    /// docs), after a flush. Returns when they are done; cheap when nothing
    /// is due. Searches and writes go on meanwhile; one `maintain` runs at
    /// a time.
    pub fn maintain(&self) -> Result<Vec<JobReport>> {
        let _one = self.maintaining.lock().unwrap();
        if self.next_job().is_none() {
            return Ok(Vec::new());
        }
        self.flush()?;
        let mut done = Vec::new();
        while let Some((reason, job)) = self.next_job() {
            done.push(self.rebuild(&job, reason)?);
        }
        Ok(done)
    }

    /// Flushes, then merges every sealed segment into one, built from the
    /// live rows (an SSD index unless there are very few).
    pub fn optimize(&self) -> Result<Option<JobReport>> {
        let _one = self.maintaining.lock().unwrap();
        self.flush()?;
        let all: Vec<u64> = self.view().sealed.iter().map(|s| s.id).collect();
        if all.is_empty() {
            return Ok(None);
        }
        Ok(Some(self.rebuild(&all, "optimize")?))
    }

    /// The next job that is due: its reason and its input segments' ids.
    fn next_job(&self) -> Option<(&'static str, Vec<u64>)> {
        let cfg = &self.config;
        let view = self.view();
        let sealed = &view.sealed;
        let ids = |idx: Vec<usize>| idx.iter().map(|&i| sealed[i].id).collect::<Vec<u64>>();
        let is_flat = |i: &usize| matches!(sealed[*i].body, Body::Flat(_));
        let flats: Vec<usize> = (0..sealed.len()).filter(is_flat).collect();
        let flat_bytes: u64 = flats.iter().map(|&i| sealed[i].live() as u64 * cfg.dim as u64 * 4).sum();
        // Below MIN_SSD_ROWS the output would be flat again: not a job.
        let flat_rows: usize = flats.iter().map(|&i| sealed[i].live()).sum();
        if flat_bytes >= cfg.index_bytes && flat_rows >= MIN_SSD_ROWS {
            return Some(("index", ids(flats)));
        }
        if let Some(i) = (0..sealed.len()).find(|&i| sealed[i].live() == 0) {
            return Some(("drop", ids(vec![i])));
        }
        let ssd: Vec<usize> = (0..sealed.len()).filter(|i| !is_flat(i)).collect();
        for &i in &ssd {
            let s = &sealed[i];
            if s.tomb.count() as f64 >= cfg.compact_deleted as f64 * s.positions() as f64 {
                return Some(("compact", ids(vec![i])));
            }
        }
        if ssd.len() > cfg.max_indexed_segments {
            let mut by_size = ssd;
            by_size.sort_by_key(|&i| (sealed[i].live(), i));
            let mut two = vec![by_size[0], by_size[1]];
            two.sort_unstable();
            return Some(("merge", ids(two)));
        }
        None
    }

    /// Replaces the segments `inputs` (ids, oldest first) with one segment
    /// of their live rows. The rows are taken under the writer lock, the
    /// segment is built without it, and the commit takes it again: rows
    /// deleted meanwhile are deleted in the new segment too.
    fn rebuild(&self, inputs: &[u64], reason: &'static str) -> Result<JobReport> {
        let t0 = Instant::now();
        let cfg = self.config.clone();
        // 1. The inputs, the rows live in them now, an id for the output.
        let (segs, rows, id) = {
            let mut w = self.writer.lock().unwrap();
            let view = self.view();
            let segs = inputs
                .iter()
                .map(|&id| {
                    view.sealed.iter().find(|s| s.id == id).cloned().ok_or_else(|| {
                        crate::error::Error::InvalidArgument(format!("segment {id} is not in the collection"))
                    })
                })
                .collect::<Result<Vec<Arc<Sealed>>>>()?;
            let mut rows: Vec<(u32, u32)> = Vec::new();
            for (j, s) in segs.iter().enumerate() {
                rows.extend((0..s.positions()).filter(|&p| !s.tomb.is_set(p)).map(|p| (j as u32, p as u32)));
            }
            // Taken now (the next manifest records it), so a flush during
            // the build numbers its segment after this one.
            let id = w.manifest.next_segment;
            w.manifest.next_segment += 1;
            (segs, rows, id)
        };
        let ids: Vec<u64> = rows.iter().map(|&(j, p)| segs[j as usize].id_at(p as usize)).collect();

        // 2. The new segment and its files. `order`: for each of its
        // positions, the entry in `rows`.
        let mut created = Vec::new();
        let built = (|| -> Result<Option<(Sealed, Vec<usize>)>> {
            if rows.is_empty() {
                return Ok(None);
            }
            let src = LiveRows::new(&segs, &rows, cfg.dim);
            let (mut sealed, order) = if rows.len() < MIN_SSD_ROWS {
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
                let params = build_params(&cfg, &self.opts);
                DiskIndex::build_partitioned(&src, Some(&ids), cfg.metric, &path, &params)?;
                let index = DiskIndex::open(&path, &DiskOpenOptions { direct: self.opts.direct_io, io: self.opts.io })?;
                // The index orders ids by node.
                let row_of: HashMap<u64, usize> = ids.iter().enumerate().map(|(r, &id)| (id, r)).collect();
                let order = (0..index.len()).map(|p| row_of[&index.id_at(p)]).collect();
                (Sealed::disk(id, index, None, vec![path])?, order)
            };
            if !cfg.fields.is_empty() {
                // Rows are copied in each input's file order; only the
                // offsets table follows the new positions.
                let mut new_pos: Vec<Vec<u32>> = segs.iter().map(|s| vec![u32::MAX; s.positions()]).collect();
                for (np, &r) in order.iter().enumerate() {
                    let (j, p) = rows[r];
                    new_pos[j as usize][p as usize] = np as u32;
                }
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
            Ok(Some((sealed, order)))
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

        // 3. Commit, from the manifest and view as they are now (flushes
        // may have come in between).
        let mut w = self.writer.lock().unwrap();
        let view = self.view();
        let mut m = w.manifest.clone();
        m.generation += 1;
        let mut carried = 0;
        let result = (|| -> Result<Option<Sealed>> {
            let Some((sealed, order)) = built else { return Ok(None) };
            // Rows deleted since step 1 (all before any search that will
            // see the new segment, so for every snapshot).
            for (np, &r) in order.iter().enumerate() {
                let (j, p) = rows[r];
                if segs[j as usize].tomb.is_set(p as usize) {
                    sealed.tomb.settle(np);
                    carried += 1;
                }
            }
            // A flush may have moved those deletes from the WAL to the
            // inputs' tombstone files, which this commit retires.
            let (mut deleted, mut del_gen) = (0, 0);
            if carried > 0 {
                let path = self.dir.join(del_name(id, m.generation));
                created.push(path.clone());
                deleted = sealed.tomb.write(&path, id)? as u64;
                del_gen = m.generation;
            }
            let entry = SegmentEntry {
                id,
                kind: match sealed.body {
                    Body::Flat(_) => SegmentKind::Flat,
                    Body::Disk { .. } => SegmentKind::Disk,
                },
                positions: sealed.positions() as u64,
                deleted,
                del_gen,
            };
            // The new segment takes the place of the oldest one it replaces.
            let first = m.segments.iter().position(|e| inputs.contains(&e.id)).expect("inputs are in the manifest");
            m.segments.insert(first, entry);
            Ok(Some(sealed))
        })()
        .and_then(|fresh| {
            m.segments.retain(|e| !inputs.contains(&e.id));
            if let Some(f) = created.first() {
                crate::format::sync_parent_dir(f)?;
            }
            m.install(&self.dir)?;
            Ok(fresh)
        });
        let fresh = match result {
            Ok(f) => f,
            Err(e) => {
                cleanup(&created);
                return Err(e);
            }
        };

        // Committed: swap the view, then drop the old segments' files.
        let mut old_files = Vec::new();
        for (e, s) in w.manifest.segments.iter().zip(&view.sealed) {
            if inputs.contains(&e.id) {
                old_files.extend(s.files.iter().cloned());
                if e.del_gen != 0 {
                    old_files.push(self.dir.join(del_name(e.id, e.del_gen)));
                }
            }
        }
        let output = fresh.as_ref().map(|s| {
            (
                s.id,
                match s.body {
                    Body::Flat(_) => "flat",
                    Body::Disk { .. } => "ssd",
                },
            )
        });
        let mut sealed = Vec::with_capacity(view.sealed.len());
        let mut fresh = fresh.map(Arc::new);
        for s in &view.sealed {
            if inputs.contains(&s.id) {
                if let Some(f) = fresh.take() {
                    sealed.push(f);
                }
            } else {
                sealed.push(s.clone());
            }
        }
        *self.view.write().unwrap() = Arc::new(super::View { sealed, growing: view.growing.clone() });
        w.manifest = m;
        drop(segs);
        // As in flush: old files go only once the rename is durable, and
        // no writes are taken until it is.
        if let Err(e) = crate::format::sync_parent_dir(&self.dir.join(super::manifest::MANIFEST)) {
            w.wal.poison();
            return Err(e);
        }
        drop(w);
        for p in old_files {
            let _ = fs::remove_file(p);
        }
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

fn build_params(cfg: &CollectionConfig, opts: &super::OpenOptions) -> PartitionedBuildParams {
    let i = &cfg.index;
    PartitionedBuildParams {
        graph: cfg.graph_params(opts.threads),
        pq: PqParams { threads: opts.threads, ..PqParams::new(i.pq_m) },
        write: DiskWriteOptions { layout: i.layout, codec: i.codec },
        memory_budget: opts.build_memory,
        partitions: 0,
        overlap: 2,
        partition_degree: 0,
        merge: MergePrune::Pq,
        scratch_dir: None,
        progress: opts.progress,
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
