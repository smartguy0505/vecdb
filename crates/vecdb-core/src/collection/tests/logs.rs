//! Several logs, frozen segments, and what a power loss leaves of them.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::*;
use crate::format::durable;

/// What the disk holds after a power loss now: `from`, with each log cut
/// somewhere between what is known to be durable and its end (`keep` picks
/// where: 0 to 1), or gone, if its name may be. Every other file is fsynced
/// before a manifest names it.
pub(super) fn power_loss_copy(c: &Collection, from: &Path, to: &Path, mut keep: impl FnMut() -> f64) {
    c.quiesced(|logs| {
        let _ = fs::remove_dir_all(to);
        fs::create_dir_all(to).unwrap();
        for name in names(from) {
            // Files no longer needed may be on their way out, and ones being
            // written on their way in.
            match fs::copy(from.join(&name), to.join(&name)) {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                r => {
                    r.unwrap();
                }
            }
        }
        for (path, len, synced) in logs {
            let log = to.join(path.file_name().unwrap());
            let cut = synced + ((len - synced) as f64 * keep()) as u64;
            if cut == 0 || (!durable::is(path) && keep() < 0.5) {
                fs::remove_file(log).unwrap();
            } else {
                fs::OpenOptions::new().write(true).open(log).unwrap().set_len(cut).unwrap();
            }
        }
    });
}

/// Sets a flag when dropped.
pub(super) struct Stop<'a>(pub &'a AtomicBool);

impl Drop for Stop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// The batches written so far, in order.
#[derive(Default)]
struct History {
    batches: Vec<Vec<OwnedOp>>,
}

#[derive(Clone)]
struct OwnedOp {
    id: u64,
    /// `None`: a delete.
    row: Option<(Vec<f32>, Attrs)>,
}

impl OwnedOp {
    fn op(&self) -> WriteOp<'_> {
        match &self.row {
            Some((v, a)) => WriteOp::Upsert { id: self.id, vector: v, attrs: a },
            None => WriteOp::Delete { id: self.id },
        }
    }
}

impl History {
    /// A batch of 1 to 4 random ops on ids below `ids`.
    fn random_batch(rng: &mut SplitMix64, dim: usize, ids: u64) -> Vec<OwnedOp> {
        (0..1 + rng.next_u64() % 4)
            .map(|_| {
                let id = rng.next_u64() % ids;
                let row = (rng.next_u64() % 4 != 0).then(|| (int_vector(rng, dim), Attrs::new().int("n", id as i64)));
                OwnedOp { id, row }
            })
            .collect()
    }

    /// The rows after the first `writes` writes, which must be whole batches.
    fn model_after(&self, writes: u64) -> Model {
        let mut m = Model::default();
        let mut n = 0;
        for b in &self.batches {
            if n == writes {
                break;
            }
            b.iter().for_each(|op| m.apply(&op.op()));
            n += b.len() as u64;
        }
        assert_eq!(n, writes, "not a whole number of batches");
        m
    }

    /// Checks that `dir` opens to the state after some number of whole
    /// batches, and how many writes those were.
    fn check_prefix(&self, dir: &Path, opts: &OpenOptions) -> u64 {
        let (c, _) = Collection::open(dir, opts.clone()).unwrap();
        let writes = c.stats().next_seq - 1;
        let model = self.model_after(writes);
        same_rows(&c, &model);
        drop(c);
        // And again: what the first open did to the files is fine too.
        let (c, report) = Collection::open(dir, opts.clone()).unwrap();
        assert_eq!((c.stats().next_seq - 1, report.replay.cut_bytes), (writes, 0));
        same_rows(&c, &model);
        writes
    }
}

/// The collection holds exactly the model's rows, each once.
fn same_rows(c: &Collection, model: &Model) {
    let mut live = c.live_ids();
    live.sort_unstable();
    let mut want: Vec<u64> = model.rows.keys().copied().collect();
    want.sort_unstable();
    assert_eq!(live, want);
    assert_eq!(c.len(), want.len());
    for (id, (v, a)) in &model.rows {
        assert_eq!(c.get(*id).unwrap(), Some((v.clone(), a.clone())), "id {id}");
    }
}

fn write(c: &Collection, batch: &[OwnedOp]) {
    let ops: Vec<WriteOp> = batch.iter().map(OwnedOp::op).collect();
    c.write(&ops).unwrap();
}

/// Writes without flushing what is frozen, as with a flusher that lags.
fn write_only(c: &Engine, batch: &[OwnedOp]) {
    let ops: Vec<WriteOp> = batch.iter().map(OwnedOp::op).collect();
    c.enqueue(c.prepare(&ops).unwrap()).unwrap();
}

fn freeze(c: &Collection) {
    c.freeze(&mut c.writer.lock().unwrap()).unwrap();
}

fn fields_config(dim: usize, seal_rows: usize) -> CollectionConfig {
    let mut cfg = small_config(dim, Metric::L2, seal_rows);
    cfg.fields = vec![Field::int("n")];
    cfg
}

/// Frozen segments are searched, deleted from and read like the growing
/// one, come back from their logs after a reopen, and end up in flat
/// segments with their later deletes.
#[test]
fn frozen_segments_behave_like_the_growing_one() {
    let dir = test_dir("coll-frozen");
    let dim = 4;
    let cfg = fields_config(dim, 100_000);
    let mut c = Collection::create(&dir, cfg, test_opts()).unwrap();
    let mut rng = SplitMix64::new(21);
    let mut h = History::default();
    let step = |c: &Collection, h: &mut History, rng: &mut SplitMix64, n: usize| {
        for _ in 0..n {
            let b = History::random_batch(rng, dim, 60);
            write_only(c, &b);
            h.batches.push(b);
        }
        let writes = h.batches.iter().map(|b| b.len() as u64).sum();
        let model = h.model_after(writes);
        same_rows(c, &model);
        check(c, &model, rng);
        let opts = SearchOptions { filter: Some(Filter::range("n", 10..30)), ..Default::default() };
        let got = c.search(&[0.0; 4], 100, &opts).unwrap();
        assert_eq!(got.len(), model.rows.keys().filter(|id| (10..30).contains(*id)).count());
        model
    };
    // A sealed segment, then two frozen ones with a log each.
    step(&c, &mut h, &mut rng, 40);
    c.flush().unwrap();
    for _ in 0..2 {
        step(&c, &mut h, &mut rng, 30);
        freeze(&c);
    }
    let model = step(&c, &mut h, &mut rng, 30);
    let st = c.stats();
    assert_eq!((st.frozen, st.segments.len()), (2, 1));
    assert!(st.growing_rows > 0 && st.frozen_rows > 0);

    // Reopened, from three logs.
    drop(c);
    let (c2, report) = Collection::open(&dir, test_opts()).unwrap();
    c = c2;
    assert_eq!((report.replay.logs, report.replay.cut_bytes), (3, 0));
    assert!(report.removed.is_empty(), "{:?}", report.removed);
    assert_eq!(c.stats().frozen, 2);
    same_rows(&c, &model);
    check(&c, &model, &mut rng);

    // One write flushes what is frozen (no one else does here).
    let b = History::random_batch(&mut rng, dim, 60);
    write(&c, &b);
    h.batches.push(b);
    let st = c.stats();
    assert_eq!((st.frozen, st.segments.len()), (0, 3));
    assert_eq!(names(&dir).iter().filter(|n| wal_generation(n).is_some()).count(), 1);
    let model = step(&c, &mut h, &mut rng, 30);
    drop(c);
    let (c, report) = Collection::open(&dir, test_opts()).unwrap();
    assert_eq!(report.replay.logs, 1);
    same_rows(&c, &model);
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// Random batches, freezes, flushes and rebuilds, with power losses at
/// random points: what is left is always the state after some number of
/// whole batches, every acknowledged one under `Sync`, and all that a flush
/// wrote under either.
#[test]
fn a_power_loss_leaves_whole_batches() {
    for durability in [Durability::Buffered, Durability::Sync] {
        let dir = test_dir("coll-power");
        let lost = test_dir("coll-power-lost");
        let dim = 3;
        let opts = OpenOptions { durability, ..test_opts() };
        let mut c = Collection::create(&dir, fields_config(dim, 100_000), opts.clone()).unwrap();
        let mut rng = SplitMix64::new(33);
        let mut h = History::default();
        let (mut written, mut flushed, mut losses, mut cuts) = (0u64, 0u64, 0, 0);
        for _ in 0..600 {
            let b = History::random_batch(&mut rng, dim, 40);
            write_only(&c, &b);
            written += b.len() as u64;
            h.batches.push(b);
            match rng.next_u64() % 30 {
                0 => {
                    c.flush().unwrap();
                    flushed = written;
                }
                1 | 2 => freeze(&c),
                9 => {
                    // The oldest frozen segment only.
                    c.flush_oldest(&c.flushing.lock().unwrap()).unwrap();
                }
                3 => {
                    let ids: Vec<u64> = c.sealed().iter().map(|s| s.id).collect();
                    if !ids.is_empty() {
                        c.rebuild(&ids, "test", 2).unwrap();
                    }
                }
                4 => {
                    drop(c);
                    let (c2, report) = Collection::open(&dir, opts.clone()).unwrap();
                    assert_eq!((report.replay.cut_bytes, c2.stats().next_seq - 1), (0, written));
                    c = c2;
                }
                5..=8 => {
                    let pick = rng.next_u64() % 3;
                    power_loss_copy(&c, &dir, &lost, || match pick {
                        0 => 0.0,
                        1 => rng.uniform(0.0, 1.0) as f64,
                        _ => 1.0,
                    });
                    let left = h.check_prefix(&lost, &opts);
                    assert!(left >= flushed, "{left} writes left, {flushed} flushed");
                    if durability == Durability::Sync || pick == 2 {
                        assert_eq!(left, written);
                    }
                    losses += 1;
                    cuts += (left < written) as usize;
                }
                _ => {}
            }
        }
        assert!(losses > 50 && (cuts > 10) == (durability == Durability::Buffered), "{losses} losses, {cuts} with cuts");
        drop(c);
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&lost);
    }
}

/// The same with flushes and rebuilds running beside the writer, so they
/// commit between its batches: deletes land in segments being written.
#[test]
fn a_power_loss_during_flushes_and_builds_leaves_whole_batches() {
    for durability in [Durability::Buffered, Durability::Sync] {
        let dir = test_dir("coll-powerbg");
        let dim = 3;
        let opts = OpenOptions { durability, ..test_opts() };
        let c = Collection::create(&dir, fields_config(dim, 100_000), opts.clone()).unwrap();
        let stop = AtomicBool::new(false);
        let (flushes, builds) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let mut h = History::default();
        let mut copies: Vec<(PathBuf, u64)> = Vec::new();
        std::thread::scope(|s| {
            // A panic below must not leave the others running.
            let _stop = Stop(&stop);
            s.spawn(|| {
                let _stop = Stop(&stop);
                while !stop.load(Ordering::Relaxed) {
                    let some = c.flush_oldest(&c.flushing.lock().unwrap()).unwrap();
                    flushes.fetch_add(some as usize, Ordering::Relaxed);
                    std::thread::yield_now();
                }
            });
            s.spawn(|| {
                let _stop = Stop(&stop);
                while !stop.load(Ordering::Relaxed) {
                    let ids: Vec<u64> = c.sealed().iter().map(|s| s.id).collect();
                    if ids.len() > 1 {
                        c.rebuild(&ids, "test", 2).unwrap();
                        builds.fetch_add(1, Ordering::Relaxed);
                    }
                    std::thread::yield_now();
                }
            });
            let mut rng = SplitMix64::new(44);
            let mut written = 0;
            let t0 = std::time::Instant::now();
            for i in 0.. {
                // Until enough of both ran beside the writes.
                let ran = (flushes.load(Ordering::Relaxed), builds.load(Ordering::Relaxed));
                if (ran.0 >= 30 && ran.1 >= 10) || stop.load(Ordering::Relaxed) {
                    break;
                }
                assert!(t0.elapsed().as_secs() < 60, "{ran:?} flushes and builds");
                let b = History::random_batch(&mut rng, dim, 40);
                write_only(&c, &b);
                written += b.len() as u64;
                h.batches.push(b);
                if durability == Durability::Buffered {
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
                if i % 10 == 0 && c.stats().frozen < 4 {
                    freeze(&c);
                }
                if i % 25 == 20 {
                    let lost = dir.with_extension(format!("lost{i}"));
                    let pick = rng.next_u64() % 3;
                    power_loss_copy(&c, &dir, &lost, || match pick {
                        0 => 0.0,
                        1 => rng.uniform(0.0, 1.0) as f64,
                        _ => 1.0,
                    });
                    copies.push((lost, if durability == Durability::Sync || pick == 2 { written } else { 0 }));
                }
            }
        });
        assert!(copies.len() > 10, "{} copies", copies.len());
        for (lost, at_least) in copies {
            let left = h.check_prefix(&lost, &opts);
            assert!(left >= at_least, "{left} writes left of {at_least}");
            let _ = fs::remove_dir_all(&lost);
        }
        let written = h.batches.iter().map(|b| b.len() as u64).sum();
        same_rows(&c, &h.model_after(written));
        drop(c);
        assert_eq!(h.check_prefix(&dir, &opts), written);
        let _ = fs::remove_dir_all(&dir);
    }
}

/// A refused open changes no file, also when an earlier log has a torn end
/// that would be cut.
#[test]
fn a_refused_open_cuts_nothing() {
    let dir = test_dir("coll-refused");
    let opts = OpenOptions { durability: Durability::Sync, ..test_opts() };
    let c = Collection::create(&dir, fields_config(3, 100_000), opts.clone()).unwrap();
    for i in 0..6 {
        write_only(&c, &[up(i, 1.0)]);
        if i == 2 {
            freeze(&c);
        }
    }
    drop(c);
    // The last record of the first log and the first one of the second.
    let (first, second) = (dir.join(wal_name(1)), dir.join(wal_name(2)));
    let mut bytes = fs::read(&first).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    fs::write(&first, &bytes).unwrap();
    let mut bytes2 = fs::read(&second).unwrap();
    bytes2[wal::HEADER_LEN + 20] ^= 1;
    fs::write(&second, &bytes2).unwrap();
    let err = Collection::open(&dir, opts.clone()).err().expect("opened without acknowledged writes");
    assert!(matches!(&err, Error::Corrupt(m) if m.contains(&wal_name(2)) && m.contains("byte 32")), "{err}");
    assert!(fs::read(&first).unwrap() == bytes && fs::read(&second).unwrap() == bytes2);
    let _ = fs::remove_dir_all(&dir);
}

/// With `Sync`, writes missing at the end of one log before another with
/// writes were acknowledged: the open refuses, changes nothing, and says
/// how to recover the writes before. Buffered ones are dropped.
#[test]
fn writes_missing_between_logs() {
    for durability in [Durability::Buffered, Durability::Sync] {
        for missing in [false, true] {
            let dir = test_dir("coll-between");
            let dim = 3;
            let opts = OpenOptions { durability, ..test_opts() };
            let c = Collection::create(&dir, fields_config(dim, 100_000), opts.clone()).unwrap();
            let mut rng = SplitMix64::new(8);
            let mut h = History::default();
            let mut ends = Vec::new();
            for i in 0..9 {
                let b = History::random_batch(&mut rng, dim, 20);
                write_only(&c, &b);
                ends.push((c.wal_bytes(), c.stats().next_seq - 1));
                h.batches.push(b);
                if i % 3 == 2 {
                    freeze(&c);
                }
            }
            drop(c);
            // Batches 3, 4, 5 are in the second log of four (the last is
            // empty). Damage: its last record, or the whole log is gone.
            let second = dir.join(wal_name(2));
            if missing {
                fs::remove_file(&second).unwrap();
            } else {
                let mut bytes = fs::read(&second).unwrap();
                bytes[ends[4].0 as usize + 20] ^= 1;
                fs::write(&second, bytes).unwrap();
            }
            let left = if missing { ends[2].1 } else { ends[4].1 };
            let before: Vec<_> = names(&dir).into_iter().map(|n| (fs::read(dir.join(&n)).unwrap(), n)).collect();
            if durability == Durability::Sync {
                let err = Collection::open(&dir, opts.clone()).err().expect("opened without acknowledged writes");
                let (at, log) = if missing { (ends[2].0, 1) } else { (ends[4].0, 2) };
                let want = [format!("byte {at} "), wal_name(log), wal_name(3)];
                assert!(matches!(&err, Error::Corrupt(m) if want.iter().all(|w| m.contains(w))), "{err}");
                let after: Vec<_> = names(&dir).into_iter().map(|n| (fs::read(dir.join(&n)).unwrap(), n)).collect();
                assert!(before == after);
                // As it says.
                FsOpenOptions::new().write(true).open(dir.join(wal_name(log))).unwrap().set_len(at).unwrap();
                for g in log + 1..=4 {
                    let _ = fs::remove_file(dir.join(wal_name(g)));
                }
                assert_eq!(h.check_prefix(&dir, &opts), left);
            } else {
                let (c, report) = Collection::open(&dir, opts.clone()).unwrap();
                let dropped: Vec<String> = (3..=4).map(wal_name).collect();
                assert_eq!(report.removed, dropped);
                assert!(report.replay.cut_bytes > 0);
                assert_eq!(c.stats().next_seq - 1, left);
                same_rows(&c, &h.model_after(left));
                // Writes go on from there, and do not meet the dropped ones.
                h.batches.truncate(if missing { 3 } else { 5 });
                for _ in 0..5 {
                    let b = History::random_batch(&mut rng, dim, 20);
                    write(&c, &b);
                    h.batches.push(b);
                }
                drop(c);
                h.check_prefix(&dir, &opts);
            }
            let _ = fs::remove_dir_all(&dir);
        }
    }
}

fn up(id: u64, x: f32) -> OwnedOp {
    OwnedOp { id, row: Some((vec![x; 3], Attrs::new().int("n", id as i64))) }
}

fn del(id: u64) -> OwnedOp {
    OwnedOp { id, row: None }
}

/// Writes a batch and remembers it, for hooks to call too.
#[derive(Clone, Default)]
struct Log(Arc<Mutex<History>>);

impl Log {
    fn write(&self, c: &Engine, ops: &[OwnedOp]) {
        write_only(c, ops);
        self.0.lock().unwrap().batches.push(ops.to_vec());
    }

    fn written(&self) -> u64 {
        self.0.lock().unwrap().batches.iter().map(|b| b.len() as u64).sum()
    }

    fn model(&self) -> Model {
        let writes = self.written();
        self.0.lock().unwrap().model_after(writes)
    }

    /// A copy of `dir` as it is now opens to all writes.
    fn check_copy(&self, c: &Collection, dir: &Path) {
        let copy = dir.with_extension("copy");
        power_loss_copy(c, dir, &copy, || 1.0);
        let opts = OpenOptions { durability: Durability::Buffered, ..test_opts() };
        let left = self.0.lock().unwrap().check_prefix(&copy, &opts);
        assert_eq!(left, self.written());
        let _ = fs::remove_dir_all(&copy);
    }

    /// After a power loss that takes all that is not known to be durable,
    /// `dir` opens to whole batches: at least the first `at_least` writes.
    fn check_power_loss(&self, c: &Collection, dir: &Path, at_least: u64) {
        let lost = dir.with_extension("lost");
        power_loss_copy(c, dir, &lost, || 0.0);
        let opts = OpenOptions { durability: Durability::Buffered, ..test_opts() };
        let left = self.0.lock().unwrap().check_prefix(&lost, &opts);
        assert!(left >= at_least && left < self.written(), "{left} of {} writes left", self.written());
        let _ = fs::remove_dir_all(&lost);
    }
}

/// A flush writes the state as of its freeze, and leaves later deletes to
/// the logs, so losing those loses the deletes too. Checked for deletes
/// that come before the flush and at two points in the middle of it, of
/// rows in sealed segments, in the one being flushed, and in one flushed
/// before.
#[test]
fn a_flush_leaves_later_deletes_to_the_logs() {
    let dir = test_dir("coll-later");
    let opts = OpenOptions { durability: Durability::Buffered, ..test_opts() };
    let c = Collection::create(&dir, fields_config(3, 100_000), opts).unwrap();
    let log = Log::default();
    let range = |r: std::ops::Range<u64>, x: f32| r.map(|id| up(id, x)).collect::<Vec<_>>();
    log.write(&c, &range(0..20, 1.0));
    c.flush().unwrap();
    // Three frozen segments, so that the later deletes are in a log that
    // the first two flushes do not fsync.
    let mut frozen_at = Vec::new();
    for base in [100, 200, 300] {
        log.write(&c, &range(base..base + 20, 2.0));
        freeze(&c);
        frozen_at.push(log.written());
    }
    // An upsert and a delete each, in a batch, of rows at `n`.
    let later = |n: u64| vec![up(n, 5.0), del(n + 1), up(100 + n, 5.0), del(101 + n), up(200 + n, 5.0), del(201 + n)];
    log.write(&c, &later(0));
    let points = Arc::new(Mutex::new(Vec::new()));
    let (l2, p2, mut n) = (log.clone(), points.clone(), 2);
    *c.hook.lock().unwrap() = Some(Box::new(move |c, point| {
        if point.ends_with("replaced") {
            return;
        }
        l2.write(c, &later(n));
        n += 2;
        p2.lock().unwrap().push(point);
    }));
    for &at in &frozen_at[..2] {
        c.flush_oldest(&c.flushing.lock().unwrap()).unwrap();
        log.check_power_loss(&c, &dir, at);
        same_rows(&c, &log.model());
    }
    *c.hook.lock().unwrap() = None;
    assert_eq!(*points.lock().unwrap(), ["flush: rows written", "flush: installed"].repeat(2));
    log.write(&c, &later(10));
    c.flush().unwrap();
    log.check_copy(&c, &dir);
    let _ = fs::remove_dir_all(&dir);
}

/// Deletes made while a build runs are in the new segment, whenever they
/// come, and the ones after its tombstone file was written stay out of the
/// files until the logs they are in are durable.
#[test]
fn a_build_takes_over_the_deletes_made_meanwhile() {
    let dir = test_dir("coll-takeover");
    let opts = OpenOptions { durability: Durability::Buffered, ..test_opts() };
    let c = Collection::create(&dir, fields_config(3, 100_000), opts).unwrap();
    let log = Log::default();
    let range = |r: std::ops::Range<u64>, x: f32| r.map(|id| up(id, x)).collect::<Vec<_>>();
    for base in [0, 10] {
        log.write(&c, &range(base..base + 10, 1.0));
        c.flush().unwrap();
    }
    // Two frozen segments, so that flushing the first does not fsync the
    // log with the deletes.
    for base in [100, 200] {
        log.write(&c, &range(base..base + 5, 2.0));
        freeze(&c);
    }
    let frozen = log.written();
    let points = Arc::new(Mutex::new(Vec::new()));
    let (l2, p2, mut n) = (log.clone(), points.clone(), 0);
    *c.hook.lock().unwrap() = Some(Box::new(move |c, point| {
        if point.ends_with("replaced") {
            return;
        }
        // Of both inputs.
        l2.write(c, &[up(n, 5.0), del(n + 1), up(10 + n, 5.0), del(11 + n)]);
        n += 2;
        p2.lock().unwrap().push(point);
    }));
    let ids: Vec<u64> = c.sealed().iter().map(|s| s.id).collect();
    assert_eq!(ids.len(), 2);
    let job = c.rebuild(&ids, "test", 2).unwrap();
    *c.hook.lock().unwrap() = None;
    assert_eq!(*points.lock().unwrap(), ["build: built", "build: deletes collected", "build: installed"]);
    assert_eq!((job.rows, job.carried_deletes), (20, 12));
    let s = c.sealed()[0].clone();
    assert_eq!((s.positions(), s.live()), (20, 8));
    same_rows(&c, &log.model());
    log.check_power_loss(&c, &dir, frozen);
    // A flush of what was frozen before the deletes.
    c.flush_oldest(&c.flushing.lock().unwrap()).unwrap();
    log.check_power_loss(&c, &dir, frozen);
    same_rows(&c, &log.model());
    c.flush().unwrap();
    log.check_copy(&c, &dir);
    let _ = fs::remove_dir_all(&dir);
}

/// A build keeps no snapshot while it runs, only while it commits: deletes
/// made during a long build leave no records behind.
#[test]
fn deletes_during_a_build_keep_no_records() {
    let dir = test_dir("coll-records");
    let c = Collection::create(&dir, fields_config(3, 100_000), test_opts()).unwrap();
    let log = Log::default();
    for base in [0, 100] {
        log.write(&c, &(base..base + 100).map(|id| up(id, 1.0)).collect::<Vec<_>>());
        c.flush().unwrap();
    }
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (l2, s2, mut n) = (log.clone(), seen.clone(), 0);
    *c.hook.lock().unwrap() = Some(Box::new(move |c, point| {
        if point.ends_with("replaced") {
            return;
        }
        for _ in 0..10 {
            l2.write(c, &[del(n), del(100 + n)]);
            n += 1;
        }
        s2.lock().unwrap().push((point, c.stats().delete_records));
    }));
    let ids: Vec<u64> = c.sealed().iter().map(|s| s.id).collect();
    let job = c.rebuild(&ids, "test", 2).unwrap();
    *c.hook.lock().unwrap() = None;
    assert_eq!(*seen.lock().unwrap(), [("build: built", 0), ("build: deletes collected", 20), ("build: installed", 40)]);
    assert_eq!(job.carried_deletes, 60);
    log.write(&c, &[del(99)]);
    assert_eq!(c.stats().delete_records, 0);
    same_rows(&c, &log.model());
    log.check_copy(&c, &dir);
    let _ = fs::remove_dir_all(&dir);
}

/// A flush that fails leaves its segment frozen and searchable, the writes
/// succeed, and a later one flushes it.
#[test]
fn a_failed_flush_is_retried() {
    let dir = test_dir("coll-flushfail");
    let dim = 4;
    let c = Collection::create(&dir, fields_config(dim, 20), test_opts()).unwrap();
    let mut rng = SplitMix64::new(3);
    let mut h = History::default();
    // A directory where the flush wants to put its segment.
    fs::create_dir_all(dir.join(flat_name(1)).join("x")).unwrap();
    let mut failed = false;
    for _ in 0..60 {
        let b = History::random_batch(&mut rng, dim, 1000);
        write(&c, &b);
        h.batches.push(b);
        let st = c.stats();
        if let Some(e) = &st.flush_error {
            assert!(!failed, "failed twice: {e}");
            failed = true;
            assert_eq!((st.frozen, st.segments.len()), (1, 0), "{e}");
            let model = h.model_after(st.next_seq - 1);
            same_rows(&c, &model);
            check(&c, &model, &mut rng);
        }
    }
    assert!(failed);
    let st = c.stats();
    assert_eq!((st.frozen, st.flush_error), (0, None));
    assert!(st.segments.len() >= 3);
    let written = st.next_seq - 1;
    same_rows(&c, &h.model_after(written));
    drop(c);
    assert_eq!(h.check_prefix(&dir, &test_opts()), written);
    let _ = fs::remove_dir_all(&dir);
}
