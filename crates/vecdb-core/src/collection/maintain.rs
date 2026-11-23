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
//! the segments it replaced are removed after that.

use std::collections::HashMap;
use std::fs::{self, File};
use std::sync::Arc;
use std::time::Instant;

use super::manifest::{SegmentEntry, SegmentKind};
use super::segment::{AttrTable, Body, Sealed, attrs_name, del_name, disk_name, flat_name};
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
    pub seconds: f64,
}

impl Collection {
    /// Runs the index builds and compactions that are due (see the module
    /// docs), after a flush. Synchronous: returns when they are done; cheap
    /// when nothing is due.
    pub fn maintain(&mut self) -> Result<Vec<JobReport>> {
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
    pub fn optimize(&mut self) -> Result<Option<JobReport>> {
        self.flush()?;
        let all: Vec<usize> = (0..self.state.sealed.len()).collect();
        if all.is_empty() {
            return Ok(None);
        }
        Ok(Some(self.rebuild(&all, "optimize")?))
    }

    fn next_job(&self) -> Option<(&'static str, Vec<usize>)> {
        let cfg = &self.state.config;
        let sealed = &self.state.sealed;
        let is_flat = |i: &usize| matches!(sealed[*i].body, Body::Flat(_));
        let flats: Vec<usize> = (0..sealed.len()).filter(is_flat).collect();
        let flat_bytes: u64 = flats.iter().map(|&i| sealed[i].live() as u64 * cfg.dim as u64 * 4).sum();
        // Below MIN_SSD_ROWS the output would be flat again: not a job.
        let flat_rows: usize = flats.iter().map(|&i| sealed[i].live()).sum();
        if flat_bytes >= cfg.index_bytes && flat_rows >= MIN_SSD_ROWS {
            return Some(("index", flats));
        }
        if let Some(i) = (0..sealed.len()).find(|&i| sealed[i].live() == 0) {
            return Some(("drop", vec![i]));
        }
        let ssd: Vec<usize> = (0..sealed.len()).filter(|i| !is_flat(i)).collect();
        for &i in &ssd {
            let s = &sealed[i];
            if s.tomb.count() as f64 >= cfg.compact_deleted as f64 * s.positions() as f64 {
                return Some(("compact", vec![i]));
            }
        }
        if ssd.len() > cfg.max_indexed_segments {
            let mut by_size = ssd;
            by_size.sort_by_key(|&i| (sealed[i].live(), i));
            let mut two = vec![by_size[0], by_size[1]];
            two.sort_unstable();
            return Some(("merge", two));
        }
        None
    }

    /// Replaces segments `idx` (indexes into the segment list, ascending)
    /// with one segment of their live rows.
    fn rebuild(&mut self, idx: &[usize], reason: &'static str) -> Result<JobReport> {
        let t0 = Instant::now();
        let cfg = self.state.config.clone();
        let segs: Vec<Arc<Sealed>> = idx.iter().map(|&i| self.state.sealed[i].clone()).collect();
        let inputs: Vec<u64> = segs.iter().map(|s| s.id).collect();
        let mut rows: Vec<(u32, u32)> = Vec::new();
        for (j, s) in segs.iter().enumerate() {
            rows.extend((0..s.positions()).filter(|&p| !s.tomb.is_set(p)).map(|p| (j as u32, p as u32)));
        }
        let ids: Vec<u64> = rows.iter().map(|&(j, p)| segs[j as usize].id_at(p as usize)).collect();
        let attrs_of = |&(j, p): &(u32, u32)| segs[j as usize].attrs.rows.get(p as usize).cloned().unwrap_or_default();

        let mut m = self.manifest.clone();
        m.generation += 1;
        let mut created = Vec::new();
        let result = (|| -> Result<Option<Sealed>> {
            let fresh = if rows.is_empty() {
                None
            } else {
                let id = m.next_segment;
                m.next_segment += 1;
                let src = LiveRows::new(&segs, &rows, cfg.dim)?;
                let mut attrs = AttrTable::default();
                let sealed = if rows.len() < MIN_SSD_ROWS {
                    let mut store = InMemoryStore::with_capacity(cfg.dim, rows.len())?;
                    let mut v = vec![0.0f32; cfg.dim];
                    for (i, &id) in ids.iter().enumerate() {
                        src.read_rows(i, &mut v)?;
                        store.push(id, &v)?;
                    }
                    let path = self.dir.join(flat_name(id));
                    created.push(path.clone());
                    write_flat_segment(&path, cfg.metric, &store)?;
                    if !cfg.fields.is_empty() {
                        attrs.rows = rows.iter().map(attrs_of).collect();
                    }
                    Sealed::flat(id, store, AttrTable::default(), vec![path])
                } else {
                    let path = self.dir.join(disk_name(id));
                    created.push(path.clone());
                    let params = build_params(&cfg, &self.opts);
                    DiskIndex::build_partitioned(&src, Some(&ids), cfg.metric, &path, &params)?;
                    let index =
                        DiskIndex::open(&path, &DiskOpenOptions { direct: self.opts.direct_io, io: self.opts.io })?;
                    if !cfg.fields.is_empty() {
                        // The index orders ids by node: write attributes in its order.
                        let row_of: HashMap<u64, usize> = ids.iter().enumerate().map(|(r, &id)| (id, r)).collect();
                        attrs.rows = (0..index.len()).map(|p| attrs_of(&rows[row_of[&index.id_at(p)]])).collect();
                    }
                    Sealed::disk(id, index, AttrTable::default(), vec![path])
                };
                let mut sealed = sealed;
                if !cfg.fields.is_empty() {
                    let ap = self.dir.join(attrs_name(id));
                    created.push(ap.clone());
                    attrs.write(&ap)?;
                    sealed.files.push(ap);
                    sealed.attrs = attrs;
                }
                Some(sealed)
            };
            // The new segment takes the place of the oldest one it replaces.
            let entry = fresh.as_ref().map(|s| SegmentEntry {
                id: s.id,
                kind: match s.body {
                    Body::Flat(_) => SegmentKind::Flat,
                    Body::Disk { .. } => SegmentKind::Disk,
                },
                positions: s.positions() as u64,
                deleted: 0,
                del_gen: 0,
            });
            let mut segments = Vec::new();
            for (i, e) in m.segments.iter().enumerate() {
                if i == idx[0]
                    && let Some(e) = &entry
                {
                    segments.push(e.clone());
                }
                if !idx.contains(&i) {
                    segments.push(e.clone());
                }
            }
            m.segments = segments;
            if let Some(f) = created.first() {
                crate::format::sync_parent_dir(f)?;
            }
            m.install(&self.dir)?;
            Ok(fresh)
        })();
        let fresh = match result {
            Ok(f) => f,
            Err(e) => {
                created.iter().for_each(|p| {
                    let _ = fs::remove_file(p);
                });
                return Err(e);
            }
        };

        // Committed: drop the old segments and their files.
        let mut old_files = Vec::new();
        for (&i, s) in idx.iter().zip(&segs) {
            old_files.extend(s.files.iter().cloned());
            let e = &self.manifest.segments[i];
            if e.del_gen != 0 {
                old_files.push(self.dir.join(del_name(e.id, e.del_gen)));
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
        let mut sealed = Vec::with_capacity(self.state.sealed.len());
        let mut fresh = fresh.map(Arc::new);
        for (i, s) in self.state.sealed.drain(..).enumerate() {
            if i == idx[0]
                && let Some(f) = fresh.take()
            {
                sealed.push(f);
            }
            if !idx.contains(&i) {
                sealed.push(s);
            }
        }
        self.state.sealed = sealed;
        self.manifest = m;
        drop(segs);
        // As in flush: old files go only once the rename is durable, and
        // no writes are taken until it is.
        if let Err(e) = crate::format::sync_parent_dir(&self.dir.join(super::manifest::MANIFEST)) {
            self.wal.poison();
            return Err(e);
        }
        for p in old_files {
            let _ = fs::remove_file(p);
        }
        Ok(JobReport {
            reason,
            inputs,
            output,
            rows: rows.len(),
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
    /// Per segment: its index file, opened for buffered reads.
    files: Vec<Option<File>>,
}

impl<'a> LiveRows<'a> {
    fn new(segs: &'a [Arc<Sealed>], rows: &'a [(u32, u32)], dim: usize) -> Result<Self> {
        let files = segs
            .iter()
            .map(|s| match s.body {
                Body::Disk { .. } => File::open(&s.files[0]).map(Some),
                Body::Flat(_) => Ok(None),
            })
            .collect::<std::io::Result<_>>()?;
        Ok(Self { segs, rows, dim, files })
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
                    index.read_node_vectors(self.files[j as usize].as_ref().expect("disk segment"), n0, &mut buf)?;
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
