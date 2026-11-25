//! Sealing: from the growing segment in RAM to a segment file.
//!
//! **Freezing** ([`Engine::freeze`]) is quick: under the writer lock the
//! growing segment moves to the view's list of frozen ones, and an empty one
//! and a new log take its place. Frozen rows are searched, and can be
//! deleted, as before.
//!
//! **Flushing** ([`Engine::flush_oldest`]) writes the oldest frozen
//! segment's rows to a flat segment file and the tombstones of every sealed
//! segment that gained deletes, and commits both with a manifest, after
//! which the frozen segment's log is not needed any more. Writes and
//! searches go on meanwhile.
//!
//! What a flush writes is exactly the state as of the last write before the
//! freeze, and the manifest says so (`flushed_seq`): later deletes are left
//! to the later logs, which a reopen replays on top. A flush that wrote them
//! too would keep them when a power loss took those (buffered) logs back, so
//! an upsert could lose its old row without its new one. So deletes are
//! carried into the new segment with the sequence numbers of their writes,
//! and the freeze registers a snapshot, which keeps the numbers of deletes
//! in sealed segments until the flush is done.

use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, MutexGuard, RwLock};

use super::columns::AttrWriter;
use super::manifest::{MANIFEST, SegmentEntry, SegmentKind};
use super::segment::{Frozen, Growing, Sealed, attrs_name, del_name, flat_name};
use super::wal::{Wal, wal_name};
use super::{Durability, Engine, View, Writer, growing_chunk_rows};
use crate::error::{Error, Result};
use crate::format::{sync_parent_dir, write_flat_segment};
use crate::store::InMemoryStore;

/// Frozen segments that may wait to be flushed before a full growing
/// segment is left to grow on.
pub(super) const MAX_FROZEN: usize = 2;

impl Engine {
    /// Writes everything in RAM to segment files: freezes the growing
    /// segment, then flushes every frozen one. Does nothing when there is
    /// nothing to write. Searches and writes go on meanwhile.
    pub fn flush(&self) -> Result<()> {
        let upto = {
            let mut w = self.writer.lock().unwrap();
            if w.wal.batches() > 0 {
                self.freeze(&mut w)?;
            }
            match self.view().frozen.last() {
                Some(f) => f.last_seq,
                None => return Ok(()),
            }
        };
        let one = self.flushing.lock().unwrap();
        while self.view().frozen.first().is_some_and(|f| f.last_seq <= upto) {
            let r = self.flush_oldest(&one);
            self.background.flushed(&r);
            r?;
        }
        Ok(())
    }

    /// Moves the growing segment to the frozen ones and starts a new log.
    /// On error nothing has changed.
    pub(super) fn freeze(&self, w: &mut Writer) -> Result<()> {
        if w.wal.is_poisoned() {
            return Err(Error::Unavailable("an earlier WAL write failed: reopen the collection".into()));
        }
        let cfg = &self.config;
        let generation = w.wal.generation() + 1;
        let path = self.dir.join(wal_name(generation));
        // Left by a freeze that failed after creating it.
        let _ = fs::remove_file(&path);
        // Under `Sync` a write counts once it is fsynced, so its log must
        // already be there for good. A buffered one may be lost.
        let sync = self.opts.durability == Durability::Sync;
        let wal = Wal::create(&self.dir, generation, cfg.dim, sync)
            .and_then(|wal| {
                if sync {
                    sync_parent_dir(&path)?;
                }
                Ok(wal)
            })
            .inspect_err(|_| {
                let _ = fs::remove_file(&path);
            })?;
        let last_seq = w.next_seq - 1;
        // The flush reads tombstones as of `last_seq`.
        self.register(last_seq);
        let view = self.view();
        let mut frozen = view.frozen.clone();
        frozen.push(Arc::new(Frozen { rows: view.growing.clone(), last_seq, wal: w.wal.generation() }));
        let growing = Arc::new(RwLock::new(Growing::new(cfg.dim, &cfg.fields)));
        *self.view.write().unwrap() = Arc::new(View { sealed: view.sealed.clone(), frozen, growing });
        let old = std::mem::replace(&mut w.wal, wal);
        w.closed.push(old);
        self.background.notify();
        Ok(())
    }

    /// Flushes the oldest frozen segment, if there is one; whether there
    /// was. The caller holds `flushing`: one flush runs at a time. On error
    /// it stays frozen.
    pub(super) fn flush_oldest(&self, _one: &MutexGuard<()>) -> Result<bool> {
        let Some(f) = self.view().frozen.first().cloned() else { return Ok(false) };
        let at = f.last_seq;
        let id = {
            let mut m = self.manifest.lock().unwrap();
            m.next_segment += 1;
            m.next_segment - 1
        };
        let mut created: Vec<PathBuf> = Vec::new();
        let cleanup = |files: &[PathBuf]| {
            files.iter().for_each(|p| {
                let _ = fs::remove_file(p);
            })
        };
        // The rows, and the log the manifest will start from.
        let fresh = self
            .write_flat(&f, id, &mut created)
            .and_then(|fresh| self.sync_log(f.wal + 1).map(|()| fresh))
            .inspect_err(|_| cleanup(&created))?;
        self.at("flush: rows written");

        // Commit, from the manifest as it is now: a build may have committed
        // in between. From here on none can (`manifest` is held), so the
        // view's sealed segments are the manifest's, one for one.
        let mut man = self.manifest.lock().unwrap();
        let mut m = man.clone();
        m.generation += 1;
        m.flushed_seq = at;
        m.wal_gen = f.wal + 1;
        // Files not needed once the manifest is in place.
        let mut old: Vec<PathBuf> = Vec::new();
        let view = self.view();
        debug_assert!(view.sealed.iter().map(|s| s.id).eq(m.segments.iter().map(|e| e.id)));
        // Tombstones that gained deletes up to `at`. A segment built
        // meanwhile may have more in its file: those are durable in the
        // logs, and stay.
        (|| -> Result<()> {
            for (s, e) in view.sealed.iter().zip(&mut m.segments) {
                let words = s.tomb.words_as_of(at);
                let count: usize = words.iter().map(|w| w.count_ones() as usize).sum();
                if count as u64 > e.deleted {
                    let path = self.dir.join(del_name(s.id, m.generation));
                    created.push(path.clone());
                    s.tomb.write(&path, s.id, &words)?;
                    if e.del_gen != 0 {
                        old.push(self.dir.join(del_name(s.id, e.del_gen)));
                    }
                    e.deleted = count as u64;
                    e.del_gen = m.generation;
                }
            }
            Ok(())
        })()
        .inspect_err(|_| cleanup(&created))?;
        if let Some((s, _)) = &fresh {
            m.segments.push(SegmentEntry {
                id,
                kind: SegmentKind::Flat,
                positions: s.positions() as u64,
                deleted: 0,
                del_gen: 0,
            });
        }
        sync_parent_dir(&self.dir.join(MANIFEST))
            .and_then(|()| m.install(&self.dir))
            .inspect_err(|_| cleanup(&created))?;
        // The new manifest is in place, so the state follows it. Until the
        // directory is fsynced a crash may bring the old one back, so the
        // files it needs stay (the next open removes them).
        let durable = sync_parent_dir(&self.dir.join(MANIFEST));
        self.at("flush: installed");
        {
            let mut w = self.writer.lock().unwrap();
            let view = self.view();
            let mut sealed = view.sealed.clone();
            if let Some((s, rows)) = fresh {
                // Rows deleted since the freeze, by the writes that did it.
                let g = f.rows.read().unwrap();
                let mut late: Vec<(u64, usize)> =
                    rows.iter().enumerate().map(|(p, &r)| (g.deleted_by(r), p)).filter(|&(d, _)| d != 0).collect();
                late.sort_unstable();
                for (d, p) in late {
                    s.tomb.delete(p, d);
                }
                sealed.push(Arc::new(s));
            }
            let next = View { sealed, frozen: view.frozen[1..].to_vec(), growing: view.growing.clone() };
            *self.view.write().unwrap() = Arc::new(next);
            w.closed.retain(|wal| {
                let needed = wal.generation() > f.wal;
                if !needed {
                    old.push(wal.path().to_path_buf());
                }
                needed
            });
        }
        *man = m;
        drop(man);
        self.unregister(at);
        self.at("flush: replaced");
        durable?;
        cleanup(&old);
        Ok(true)
    }

    /// Writes the rows of `f` live at its freeze as flat segment `id` (and
    /// their attributes); `None` if there are none. Also the row of each of
    /// its positions.
    fn write_flat(&self, f: &Frozen, id: u64, created: &mut Vec<PathBuf>) -> Result<Option<(Sealed, Vec<usize>)>> {
        let (at, cfg) = (f.last_seq, &self.config);
        // In chunks, each under its own read lock: writes that delete rows
        // here wait for one at most.
        let n = f.rows.read().unwrap().rows();
        let chunk = growing_chunk_rows(cfg.dim);
        let mut live: Vec<usize> = Vec::new();
        let mut store = InMemoryStore::with_capacity(cfg.dim, n)?;
        for start in (0..n).step_by(chunk) {
            let g = f.rows.read().unwrap();
            for r in (start..(start + chunk).min(n)).filter(|&r| g.live_at(r, at)) {
                store.push(g.ids[r], g.vector(r))?;
                live.push(r);
            }
        }
        if live.is_empty() {
            return Ok(None);
        }
        let path = self.dir.join(flat_name(id));
        created.push(path.clone());
        write_flat_segment(&path, cfg.metric, &store)?;
        let mut files = vec![path];
        let mut attrs = None;
        if !cfg.fields.is_empty() {
            let ap = self.dir.join(attrs_name(id));
            created.push(ap.clone());
            let mut aw = AttrWriter::create(&ap, &cfg.fields, live.len())?;
            for (i, rows) in live.chunks(chunk).enumerate() {
                let g = f.rows.read().unwrap();
                for (j, &r) in rows.iter().enumerate() {
                    aw.push(i * chunk + j, g.attrs(r))?;
                }
            }
            attrs = Some(aw.finish()?);
            files.push(ap);
        }
        Ok(Some((Sealed::flat(id, store, attrs, files), live)))
    }

    /// Makes log `generation` and its name durable.
    fn sync_log(&self, generation: u64) -> Result<()> {
        let file = {
            let w = self.writer.lock().unwrap();
            let wal = w.closed.iter().chain([&w.wal]).find(|wal| wal.generation() == generation);
            wal.expect("a log follows every frozen segment").unsynced()?
        };
        self.sync_files(file.map(|(f, len)| (generation, f, len)).into_iter().collect())
    }

    /// Makes every write logged so far durable, whatever the durability
    /// writes were made with. Writes go on meanwhile.
    pub(super) fn sync_logs(&self) -> Result<()> {
        let mut files = Vec::new();
        for l in self.writer.lock().unwrap().logs_mut() {
            if let Some((f, len)) = l.unsynced()? {
                files.push((l.generation(), f, len));
            }
        }
        self.sync_files(files)
    }

    /// Fsyncs logs and the directory: generation, file, and the length that
    /// makes durable.
    fn sync_files(&self, files: Vec<(u64, fs::File, u64)>) -> Result<()> {
        if files.is_empty() {
            return Ok(());
        }
        for (_, f, _) in &files {
            f.sync_all().inspect_err(|_| {
                // The kernel may have dropped the pages: nothing about the
                // logs can be trusted any more.
                self.writer.lock().unwrap().wal.poison();
            })?;
        }
        sync_parent_dir(&self.dir.join(MANIFEST))?;
        let mut w = self.writer.lock().unwrap();
        for (generation, _, len) in files {
            w.logs_mut().filter(|l| l.generation() == generation).for_each(|l| l.mark_synced(len));
        }
        Ok(())
    }

    /// After a write: flushes what is frozen. If another thread is flushing,
    /// leaves it to that one, unless as many are frozen as may be.
    pub(super) fn flush_frozen(&self) {
        let waiting = self.view().frozen.len();
        if waiting == 0 {
            return;
        }
        let one = if waiting >= MAX_FROZEN {
            self.flushing.lock().unwrap()
        } else {
            match self.flushing.try_lock() {
                Ok(g) => g,
                Err(_) => return,
            }
        };
        loop {
            let r = self.flush_oldest(&one);
            self.background.flushed(&r);
            if !matches!(r, Ok(true)) {
                return;
            }
        }
    }
}

impl Writer {
    /// Every log still needed, oldest first.
    pub(super) fn logs_mut(&mut self) -> impl Iterator<Item = &mut Wal> {
        self.closed.iter_mut().chain([&mut self.wal])
    }
}
