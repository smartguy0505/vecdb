//! Flushes and builds on background threads.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use super::*;
use super::maintain::Lane;
use crate::cancel::Cancel;

fn opts() -> OpenOptions {
    OpenOptions { background: true, ..test_opts() }
}

/// Waits until `done` (at most 30 s).
fn until(what: &str, mut done: impl FnMut() -> bool) {
    let t0 = Instant::now();
    while !done() {
        assert!(t0.elapsed() < Duration::from_secs(30), "waited too long: {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn upsert_batch(c: &Collection, model: &mut Model, rng: &mut SplitMix64, ids: std::ops::Range<u64>) {
    let rows: Vec<(u64, Vec<f32>)> = ids.map(|id| (id, float_vector(rng, c.config().dim))).collect();
    let ops: Vec<WriteOp> =
        rows.iter().map(|(id, v)| WriteOp::Upsert { id: *id, vector: v, attrs: &NO_ATTRS }).collect();
    c.write(&ops).unwrap();
    ops.iter().for_each(|op| model.apply(op));
}

/// Without any call but writes and searches, full segments are flushed,
/// indexes built, and segments with many deleted rows compacted.
#[test]
fn flushes_builds_and_compactions_run_by_themselves() {
    let dir = test_dir("coll-bg");
    let dim = 8;
    let mut cfg = ssd_config(dim, Metric::L2, 500, 2500);
    cfg.compact_deleted = 0.3;
    let c = Collection::create(&dir, cfg, opts()).unwrap();
    let mut rng = SplitMix64::new(1);
    let mut model = Model::default();
    for i in 0..60 {
        upsert_batch(&c, &mut model, &mut rng, i * 100..(i + 1) * 100);
    }
    c.wait_idle().unwrap();
    let st = c.stats();
    assert_eq!((st.frozen, &st.flush_error, &st.build_error), (0, &None, &None));
    assert!(st.growing_rows < 500 && st.flushes >= 10 && st.builds >= 1, "{st:?}");
    let flat: usize = st.segments.iter().filter(|s| s.kind == "flat").map(|s| s.rows).sum();
    assert!(flat < 2500 && st.segments.iter().any(|s| s.kind == "ssd"), "{st:?}");
    check_approx(&c, &model, &mut rng, 0.9);

    // Deletes alone, too few to fill a log: noticed all the same.
    let builds = st.builds;
    let ids: Vec<u64> = model.rows.keys().copied().filter(|id| id % 2 == 0).collect();
    let ops: Vec<WriteOp> = ids.iter().map(|&id| WriteOp::Delete { id }).collect();
    c.write(&ops).unwrap();
    ops.iter().for_each(|op| model.apply(op));
    until("a compaction", || c.stats().builds > builds);
    c.wait_idle().unwrap();
    let st = c.stats();
    assert!(st.segments.iter().filter(|s| s.kind == "ssd").all(|s| s.deleted * 10 < s.rows * 3), "{st:?}");
    check_approx(&c, &model, &mut rng, 0.9);
    drop(c);
    let (c, _) = Collection::open(&dir, opts()).unwrap();
    check_approx(&c, &model, &mut rng, 0.9);
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// What was frozen when the collection was closed (or its process died)
/// is flushed after the next open.
#[test]
fn frozen_segments_are_flushed_after_an_open() {
    let dir = test_dir("coll-bg-reopen");
    let c = Collection::create(&dir, small_config(4, Metric::L2, 100_000), test_opts()).unwrap();
    let mut rng = SplitMix64::new(1);
    let mut model = Model::default();
    for i in 0..3u64 {
        for id in i * 20..(i + 1) * 20 {
            // (Not `write`, which would flush what is frozen.)
            let v = float_vector(&mut rng, 4);
            let op = WriteOp::Upsert { id, vector: &v, attrs: &NO_ATTRS };
            c.enqueue(c.prepare(&[op]).unwrap()).unwrap();
            model.apply(&op);
        }
        if i < 2 {
            c.freeze(&mut c.writer.lock().unwrap()).unwrap();
        }
    }
    assert_eq!(c.stats().frozen, 2);
    drop(c);
    let (c, report) = Collection::open(&dir, opts()).unwrap();
    assert_eq!(report.replay.logs, 3);
    c.wait_idle().unwrap();
    let st = c.stats();
    assert_eq!((st.frozen, st.segments.len(), st.growing_rows, st.live), (0, 2, 20, 60));
    check_approx(&c, &model, &mut rng, 1.0);
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// A flush that fails is retried; writes go on until four times a full
/// segment waits, are refused then, and accepted again once it succeeds.
#[test]
fn a_failing_flush_is_retried_in_the_background() {
    for background in [true, false] {
        let dir = test_dir("coll-bg-fail");
        let opts = OpenOptions { background, ..opts() };
        let c = Collection::create(&dir, small_config(4, Metric::L2, 50), opts.clone()).unwrap();
        // Directories where the flush wants to put its segment.
        let blocks: Vec<PathBuf> = (1..500).map(|id| dir.join(flat_name(id)).join("x")).collect();
        blocks.iter().for_each(|b| fs::create_dir_all(b).unwrap());
        let mut rng = SplitMix64::new(2);
        let mut model = Model::default();
        let mut next = 0;
        let refused = loop {
            let v = float_vector(&mut rng, 4);
            match c.upsert(next, &v, &NO_ATTRS) {
                Ok(()) => model.apply(&WriteOp::Upsert { id: next, vector: &v, attrs: &NO_ATTRS }),
                Err(e) => break e,
            }
            next += 1;
            // (In the background the first failure takes a moment.)
            if next == 100 {
                until("a failed flush", || c.stats().flush_error.is_some());
            }
            assert!(next < 1000);
        };
        assert!(matches!(&refused, Error::Unavailable(m) if m.contains("flush")), "{refused}");
        let st = c.stats();
        assert_eq!((st.frozen, st.growing_rows, st.segments.len()), (2, 200, 0));
        assert!(st.flush_error.is_some());
        check_approx(&c, &model, &mut rng, 1.0);
        assert!(c.wait_idle().is_err() || !background);
        blocks.iter().for_each(|b| fs::remove_dir_all(b.parent().unwrap()).unwrap());
        until("a successful flush", || {
            let v = float_vector(&mut rng, 4);
            let r = c.upsert(next, &v, &NO_ATTRS);
            if r.is_ok() {
                model.apply(&WriteOp::Upsert { id: next, vector: &v, attrs: &NO_ATTRS });
                next += 1;
            }
            c.stats().flush_error.is_none()
        });
        // (Without more writes, which may be faster than the flushes.)
        c.wait_idle().unwrap();
        assert_eq!(c.stats().frozen, 0);
        check_approx(&c, &model, &mut rng, 1.0);
        drop(c);
        let (c, _) = Collection::open(&dir, opts).unwrap();
        check_approx(&c, &model, &mut rng, 1.0);
        drop(c);
        let _ = fs::remove_dir_all(&dir);
    }
}

/// Has the next `point` of a background job wait until something is sent.
fn block_at(c: &Collection, point: &'static str) -> (mpsc::Sender<()>, Arc<AtomicBool>) {
    block_thread_at(c, None, point)
}

/// The same for the thread of that name only. (The others pass, also while
/// it waits.)
fn block_thread_at(
    c: &Collection,
    thread: Option<&'static str>,
    point: &'static str,
) -> (mpsc::Sender<()>, Arc<AtomicBool>) {
    let (go, wait) = mpsc::channel();
    let blocked = Arc::new(AtomicBool::new(false));
    let (b, mut done) = (blocked.clone(), false);
    *c.hook.lock().unwrap() = Some(Box::new(move |_, at| {
        let here = thread.is_none() || std::thread::current().name() == thread;
        if at == point && here && !done {
            done = true;
            b.store(true, Ordering::Relaxed);
            let _ = wait.recv_timeout(Duration::from_secs(60));
        }
    }));
    (go, blocked)
}

/// Writes one row after the other until told to stop, which it counts.
fn write_on(c: &Collection, written: &AtomicU64, stop: &AtomicBool) {
    let mut rng = SplitMix64::new(9);
    let dim = c.config().dim;
    while !stop.load(Ordering::Relaxed) {
        let id = written.load(Ordering::Relaxed);
        c.upsert(id, &float_vector(&mut rng, dim), &NO_ATTRS).unwrap();
        written.store(id + 1, Ordering::Relaxed);
    }
}

/// The count once it has stopped changing (for a second).
fn settled(n: &AtomicU64) -> u64 {
    let (mut last, mut same) = (u64::MAX, 0);
    until("writes to stop", || {
        std::thread::sleep(Duration::from_millis(100));
        let now = n.load(Ordering::Relaxed);
        same = if now == last { same + 1 } else { 0 };
        last = now;
        same >= 10
    });
    last
}

/// Writes wait when the growing segment is full and two are frozen.
#[test]
fn writes_wait_for_a_slow_flush() {
    let dir = test_dir("coll-bg-slow");
    // (Wide enough that a segment is full before its log is.)
    let c = Collection::create(&dir, small_config(32, Metric::L2, 50), opts()).unwrap();
    let (go, blocked) = block_at(&c, "flush: rows written");
    let (written, stop) = (AtomicU64::new(0), AtomicBool::new(false));
    std::thread::scope(|s| {
        let _stop = logs::Stop(&stop);
        s.spawn(|| write_on(&c, &written, &stop));
        until("a flush", || blocked.load(Ordering::Relaxed));
        assert_eq!(settled(&written), 150);
        let st = c.stats();
        assert_eq!((st.frozen, st.frozen_rows, st.growing_rows), (2, 100, 50));
        assert_eq!(c.len(), 150);
        go.send(()).unwrap();
        until("more writes", || written.load(Ordering::Relaxed) > 300);
    });
    c.flush().unwrap();
    assert_eq!((c.len() as u64, c.stats().flush_error), (written.load(Ordering::Relaxed), None));
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// Writes wait when `max_unindexed_bytes` of flat segments wait for an
/// index build; and flushes go on during one.
#[test]
fn writes_wait_for_a_slow_build() {
    let dir = test_dir("coll-bg-slowbuild");
    let dim = 4;
    let cfg = ssd_config(dim, Metric::L2, 500, 2500);
    let opts = OpenOptions { max_unindexed_bytes: 4000 * dim as u64 * 4, ..opts() };
    let c = Collection::create(&dir, cfg, opts).unwrap();
    let (go, blocked) = block_at(&c, "build: built");
    let (written, stop) = (AtomicU64::new(0), AtomicBool::new(false));
    std::thread::scope(|s| {
        let _stop = logs::Stop(&stop);
        s.spawn(|| write_on(&c, &written, &stop));
        until("a build", || blocked.load(Ordering::Relaxed));
        let n = settled(&written);
        // 4000 in flat segments, with what may be frozen or growing by the
        // time the last of them is.
        assert!((4000..=5500).contains(&n), "{n} rows");
        let st = c.stats();
        assert!(st.segments.iter().all(|s| s.kind == "flat") && st.builds == 0, "{st:?}");
        go.send(()).unwrap();
        until("more writes", || written.load(Ordering::Relaxed) > n + 2000);
    });
    c.flush().unwrap();
    c.wait_idle().unwrap();
    let st = c.stats();
    assert_eq!((st.live as u64, st.build_error), (written.load(Ordering::Relaxed), None));
    assert!(st.builds >= 1 && st.segments.iter().any(|s| s.kind == "ssd"));
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// `wait_idle` also waits for what comes after the view: the old files are
/// gone, and the counts right, when it returns.
#[test]
fn idle_means_done() {
    for point in ["flush: replaced", "build: replaced"] {
        let dir = test_dir("coll-bg-idle");
        let cfg = ssd_config(4, Metric::L2, 2500, 2500);
        let c = Collection::create(&dir, cfg, opts()).unwrap();
        let (go, blocked) = block_at(&c, point);
        let mut rng = SplitMix64::new(4);
        upsert_batch(&c, &mut Model::default(), &mut rng, 0..2500);
        until(point, || blocked.load(Ordering::Relaxed));
        until("the view", || c.stats().segments.iter().any(|s| s.kind == "ssd"));
        assert!(c.stats().frozen == 0 && c.no_job());
        std::thread::scope(|s| {
            let idle = s.spawn(|| c.wait_idle().unwrap());
            std::thread::sleep(Duration::from_millis(200));
            assert!(!idle.is_finished(), "{point}");
            let t0 = Instant::now();
            go.send(()).unwrap();
            idle.join().unwrap();
            assert!(t0.elapsed() < Duration::from_millis(700), "told late");
        });
        let st = c.stats();
        assert_eq!((st.flushes, st.builds), (1, 1));
        let left = names(&dir);
        assert!(!left.iter().any(|n| n.contains("flat")) && left.iter().filter(|n| n.contains("wal")).count() == 1, "{left:?}");
        drop(c);
        let _ = fs::remove_dir_all(&dir);
    }
}

/// The running build's cancel flag, and whether it has been asked for.
static STOPPED_BUILD: Mutex<Option<(Cancel, bool)>> = Mutex::new(None);

/// Dropping a collection stops its index build, which leaves no files.
#[test]
fn dropping_a_collection_stops_its_build() {
    // In the middle of the build, waits for the drop.
    fn wait_for_drop(line: &str) {
        if line.starts_with("encode") {
            let mut build = STOPPED_BUILD.lock().unwrap();
            let (cancel, started) = build.as_mut().unwrap();
            *started = true;
            let cancel = cancel.clone();
            drop(build);
            until("the drop", || cancel.is_cancelled());
        }
    }
    let dir = test_dir("coll-bg-drop");
    let dim = 16;
    let opts = OpenOptions { progress: Some(wait_for_drop), ..opts() };
    let c = Collection::create(&dir, ssd_config(dim, Metric::L2, 1000, 3000), opts).unwrap();
    *STOPPED_BUILD.lock().unwrap() = Some((c.background.cancel(), false));
    let mut rng = SplitMix64::new(4);
    let mut model = Model::default();
    for i in 0..32 {
        upsert_batch(&c, &mut model, &mut rng, i * 100..(i + 1) * 100);
    }
    until("the build", || STOPPED_BUILD.lock().unwrap().as_ref().unwrap().1);
    assert!(names(&dir).iter().any(|n| n.contains(".vdx.build-")));
    let t0 = Instant::now();
    drop(c);
    assert!(t0.elapsed() < Duration::from_secs(20), "the build was not told to stop");
    assert!(names(&dir).iter().all(|n| !n.contains(".vdx")), "{:?}", names(&dir));
    let (c, report) = Collection::open(&dir, test_opts()).unwrap();
    assert!(report.removed.is_empty(), "{:?}", report.removed);
    let st = c.stats();
    assert!(st.segments.iter().all(|s| s.kind == "flat"), "{st:?}");
    check_approx(&c, &model, &mut rng, 1.0);
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// With `opts`, a collection with an SSD segment of 3,600 rows (ids from
/// 1,000,000) and no job due, that seals at 500 rows, builds indexes from
/// 2,500 and compacts at 30% deleted.
fn with_ssd_segment(dir: &Path, opts: OpenOptions) -> Collection {
    let dim = 4;
    let mut cfg = ssd_config(dim, Metric::L2, 500, 2500);
    cfg.compact_deleted = 0.3;
    cfg.max_indexed_segments = 1000;
    let c = Collection::create(dir, cfg, opts).unwrap();
    let mut rng = SplitMix64::new(4);
    for i in 0..36 {
        upsert_batch(&c, &mut Model::default(), &mut rng, 1_000_000 + i * 100..1_000_100 + i * 100);
    }
    c.flush().unwrap();
    c.optimize().unwrap();
    c.wait_idle().unwrap();
    let st = c.stats();
    assert!(st.segments.len() == 1 && st.segments[0].kind == "ssd" && st.live == 3600, "{st:?}");
    c
}

/// Deletes a third of the SSD segment's rows, so that it is to be compacted.
fn spoil(c: &Collection) {
    let ops: Vec<WriteOp> = (0..1200).map(|i| WriteOp::Delete { id: 1_000_000 + i * 3 }).collect();
    c.write(&ops).unwrap();
}

fn ssd_segments(c: &Collection) -> usize {
    c.stats().segments.iter().filter(|s| s.kind == "ssd").count()
}

/// What writes at the limit wait for is not held up by a compaction, however
/// long it takes, of this collection or of another under the same limit; a
/// second compaction waits for the first.
#[test]
fn index_builds_do_not_wait_for_compactions() {
    let dirs = [test_dir("coll-bg-lanes-a"), test_dir("coll-bg-lanes-b")];
    let limit = 4000 * 4 * 4;
    let opts = OpenOptions { max_unindexed_bytes: limit, builds: Some(BuildLimit::new(1)), ..opts() };
    let [a, b] = dirs.each_ref().map(|d| with_ssd_segment(d, opts.clone()));
    let (go, blocked) = block_at(&a, "build: built");
    spoil(&a);
    until("a compaction", || blocked.load(Ordering::Relaxed));
    spoil(&b);
    let before = [&a, &b].map(|c| c.stats().builds);
    let (written, stop) = ([AtomicU64::new(0), AtomicU64::new(0)], AtomicBool::new(false));
    std::thread::scope(|s| {
        let _stop = logs::Stop(&stop);
        for (c, n) in [&a, &b].into_iter().zip(&written) {
            s.spawn(|| write_on(c, n, &stop));
        }
        // Four times the limit, and one that comes next.
        until("writes", || written.iter().all(|n| n.load(Ordering::Relaxed) > 18_000));
    });
    // The compactions have not come: neither has lost its deleted rows.
    for (c, before) in [&a, &b].into_iter().zip(before) {
        let st = c.stats();
        assert_eq!((st.segments[0].rows, st.segments[0].deleted), (3600, 1200), "{st:?}");
        assert!(st.builds >= before + 5 && ssd_segments(c) >= 6, "{st:?}");
    }
    go.send(()).unwrap();
    [&a, &b].iter().for_each(|c| c.wait_idle().unwrap());
    for (c, n) in [a, b].into_iter().zip(&written) {
        let st = c.stats();
        assert_eq!((st.segments[0].rows, st.build_error), (2400, None));
        assert_eq!(st.live as u64, 2400 + n.load(Ordering::Relaxed));
        let ids: std::collections::HashSet<u64> = c.live_ids().into_iter().collect();
        assert_eq!(ids.len(), st.live);
        assert!((0..n.load(Ordering::Relaxed)).all(|id| ids.contains(&id)));
    }
    dirs.iter().for_each(|d| drop(fs::remove_dir_all(d)));
}

/// A segment has its jobs in one lane only, so that two threads never work
/// on it: also when it is to be dropped.
#[test]
fn lanes_do_not_share_segments() {
    let dir = test_dir("coll-bg-lanes-own");
    let c = with_ssd_segment(&dir, OpenOptions { background: false, ..opts() });
    let rows: Vec<Vec<f32>> = (0..300).map(|i| vec![i as f32; 4]).collect();
    rows.iter().enumerate().for_each(|(id, v)| c.upsert(id as u64, v, &NO_ATTRS).unwrap());
    c.flush().unwrap();
    let [ssd, flat] = &c.sealed()[..] else { panic!("{:?}", c.stats()) };
    assert_eq!((c.next_job(Lane::Flat), c.next_job(Lane::Ssd)), (None, None));
    let delete = |ids: std::ops::Range<u64>| {
        c.write(&ids.map(|id| WriteOp::Delete { id }).collect::<Vec<_>>()).unwrap();
    };
    delete(1_000_000..1_003_600);
    assert_eq!((c.next_job(Lane::Flat), c.next_job(Lane::Ssd)), (None, Some(("drop", vec![ssd.id]))));
    c.maintain().unwrap();
    delete(0..300);
    assert_eq!((c.next_job(Lane::Flat), c.next_job(Lane::Ssd)), (Some(("drop", vec![flat.id])), None));
    c.maintain().unwrap();
    assert!(c.sealed().is_empty());
    let _ = fs::remove_dir_all(&dir);
}

/// `optimize` waits for a compaction without keeping index builds out, and
/// takes the segments there are when it starts: later ones are indexed
/// meanwhile, and the ones taken do not count towards the limit.
#[test]
fn index_builds_go_on_during_an_optimize() {
    let dir = test_dir("coll-bg-lanes-opt");
    let opts = OpenOptions { max_unindexed_bytes: 4000 * 4 * 4, ..opts() };
    let c = with_ssd_segment(&dir, opts.clone());
    let (go, blocked) = block_thread_at(&c, Some("vecdb-compact"), "build: built");
    spoil(&c);
    until("a compaction", || blocked.load(Ordering::Relaxed));
    let written = AtomicU64::new(0);
    let write_more = |rows: u64| {
        let from = written.load(Ordering::Relaxed);
        let stop = AtomicBool::new(false);
        std::thread::scope(|s| {
            let _stop = logs::Stop(&stop);
            s.spawn(|| write_on(&c, &written, &stop));
            until("writes", || written.load(Ordering::Relaxed) > from + rows);
        });
    };
    std::thread::scope(|s| {
        let named = std::thread::Builder::new().name("optimize".into());
        let opt = named.spawn_scoped(s, || c.optimize().unwrap().unwrap()).unwrap();
        write_more(12_000);
        assert!(c.optimizing.lock().unwrap().is_empty());
        let (go2, blocked2) = block_thread_at(&c, Some("optimize"), "build: built");
        go.send(()).unwrap();
        until("the optimize", || blocked2.load(Ordering::Relaxed));
        let taken = c.optimizing.lock().unwrap().clone();
        write_more(12_000);
        let sealed: BTreeSet<u64> = c.sealed().iter().map(|s| s.id).collect();
        assert!(taken.len() > 1 && taken.is_subset(&sealed), "{taken:?} {sealed:?}");
        assert!(c.sealed().iter().any(|s| matches!(s.body, Body::Disk { .. }) && !taken.contains(&s.id)));
        go2.send(()).unwrap();
        let job = opt.join().unwrap();
        assert_eq!(job.inputs.iter().copied().collect::<BTreeSet<u64>>(), taken);
    });
    assert!(c.optimizing.lock().unwrap().is_empty());
    c.wait_idle().unwrap();
    let n = written.load(Ordering::Relaxed);
    let check = |c: &Collection| {
        let ids: BTreeSet<u64> = c.live_ids().into_iter().collect();
        assert_eq!((c.len(), ids.len()), (2400 + n as usize, 2400 + n as usize));
        assert!((0..n).all(|id| ids.contains(&id)));
        assert!((0..3600).all(|i| ids.contains(&(1_000_000 + i)) == (i % 3 != 0)));
    };
    check(&c);
    drop(c);
    let (c, _) = Collection::open(&dir, opts).unwrap();
    check(&c);
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// A compaction that keeps failing is reported, and does not stand in the
/// way of writes and index builds.
#[test]
fn a_failing_compaction_does_not_refuse_writes() {
    let dir = test_dir("coll-bg-lanes-fail");
    let opts = OpenOptions { max_unindexed_bytes: 4000 * 4 * 4, ..opts() };
    let c = with_ssd_segment(&dir, opts);
    // Damage the rows on disk.
    let id = c.sealed()[0].id;
    let path = dir.join(disk_name(id));
    let mut bytes = fs::read(&path).unwrap();
    let middle = bytes.len() / 2;
    bytes[middle - 2000..middle + 2000].iter_mut().for_each(|b| *b ^= 0x55);
    let f = FsOpenOptions::new().write(true).open(&path).unwrap();
    std::os::unix::fs::FileExt::write_all_at(&f, &bytes, 0).unwrap();
    spoil(&c);
    until("a failed compaction", || c.stats().build_error.is_some());
    assert!(c.wait_idle().is_err());
    let (written, stop) = (AtomicU64::new(0), AtomicBool::new(false));
    std::thread::scope(|s| {
        let _stop = logs::Stop(&stop);
        s.spawn(|| write_on(&c, &written, &stop));
        until("writes", || written.load(Ordering::Relaxed) > 18_000);
    });
    // Nor do the index builds that succeed clear the error.
    let st = c.stats();
    assert!(st.builds >= 5 && st.build_error.is_some(), "{st:?}");
    // And once it is gone, the segment and the error with it.
    let ops: Vec<WriteOp> = (0..3600).map(|i| WriteOp::Delete { id: 1_000_000 + i }).collect();
    c.write(&ops).unwrap();
    until("a drop", || c.stats().build_error.is_none());
    c.wait_idle().unwrap();
    assert_eq!(c.len() as u64, written.load(Ordering::Relaxed));
    assert!(c.sealed().iter().all(|s| s.id != id));
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// Index builds that fail are retried; writes go on to twice the limit, are
/// refused then, and accepted again once one succeeds.
#[test]
fn failing_index_builds_refuse_writes() {
    let dir = test_dir("coll-bg-index-fail");
    let cfg = CollectionConfig { max_indexed_segments: 1000, ..ssd_config(4, Metric::L2, 500, 2500) };
    // (Those that wait for a build are refused for another reason.)
    let max_write_wait = Some(Duration::from_secs(20));
    let opts = OpenOptions { max_unindexed_bytes: 4000 * 4 * 4, max_write_wait, ..opts() };
    let c = Collection::create(&dir, cfg, opts).unwrap();
    // Directories where the builds want to put their indexes.
    let blocks: Vec<PathBuf> = (1..500).map(|id| dir.join(disk_name(id)).join("x")).collect();
    blocks.iter().for_each(|b| fs::create_dir_all(b).unwrap());
    let mut rng = SplitMix64::new(5);
    let mut next = 0;
    let refused = loop {
        match c.upsert(next, &float_vector(&mut rng, 4), &NO_ATTRS) {
            Ok(()) => next += 1,
            Err(e) => break e,
        }
        assert!(next < 20_000);
    };
    assert!(matches!(&refused, Error::Unavailable(m) if m.contains("index builds fail")), "{refused}");
    let st = c.stats();
    let flat: usize = st.segments.iter().map(|s| s.rows).sum();
    assert!((8000..10_000).contains(&flat) && st.build_error.is_some(), "{st:?}");
    assert!(c.wait_idle().is_err());
    blocks.iter().for_each(|b| fs::remove_dir_all(b.parent().unwrap()).unwrap());
    until("a successful build", || {
        let written = c.upsert(next, &float_vector(&mut rng, 4), &NO_ATTRS).is_ok();
        next += written as u64;
        c.stats().build_error.is_none()
    });
    c.wait_idle().unwrap();
    assert!(ssd_segments(&c) >= 2 && c.len() as u64 == next, "{:?}", c.stats());
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// When index builds are far behind, each takes a part of the flat segments.
#[test]
fn index_builds_take_a_part_at_a_time() {
    let dir = test_dir("coll-bg-part");
    let cfg = CollectionConfig { max_indexed_segments: 1000, ..ssd_config(4, Metric::L2, 500, 1000) };
    // An eighth is 1,250 rows, but an SSD index has to have 2,048.
    let opts = OpenOptions { max_unindexed_bytes: 10_000 * 4 * 4, ..opts() };
    let c = Collection::create(&dir, cfg, opts).unwrap();
    let (go, blocked) = block_at(&c, "build: built");
    let mut rng = SplitMix64::new(3);
    let mut model = Model::default();
    for i in 0..90 {
        upsert_batch(&c, &mut model, &mut rng, i * 100..(i + 1) * 100);
    }
    until("a build", || blocked.load(Ordering::Relaxed));
    go.send(()).unwrap();
    c.flush().unwrap();
    c.wait_idle().unwrap();
    let st = c.stats();
    let ssd: Vec<usize> = st.segments.iter().filter(|s| s.kind == "ssd").map(|s| s.rows).collect();
    assert!(ssd.len() >= 3 && ssd.iter().all(|r| (MIN_SSD_ROWS..=2600).contains(r)), "{st:?}");
    // The oldest first, in the place of those taken.
    let first: Vec<u64> = c.sealed().iter().map(|s| (0..s.positions()).map(|p| s.id_at(p)).min().unwrap()).collect();
    assert!(first.is_sorted(), "{first:?}");
    check_approx(&c, &model, &mut rng, 0.9);
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// A write gives up after `max_write_wait`, and changes nothing then.
#[test]
fn writes_give_up_waiting() {
    let wait = Duration::from_millis(300);
    for (point, what) in [("flush: rows written", "flushes"), ("build: built", "index builds")] {
        let dir = test_dir("coll-bg-giveup");
        let cfg = ssd_config(32, Metric::L2, 500, 2500);
        let limit = if what == "flushes" { 0 } else { 4000 * 32 * 4 };
        let opts = OpenOptions { max_write_wait: Some(wait), max_unindexed_bytes: limit, ..opts() };
        let c = Collection::create(&dir, cfg, opts).unwrap();
        let (go, blocked) = block_at(&c, point);
        let mut rng = SplitMix64::new(5);
        let mut next = 0;
        let (e, took) = loop {
            let v = float_vector(&mut rng, 32);
            let op = WriteOp::Upsert { id: next, vector: &v, attrs: &NO_ATTRS };
            let t0 = Instant::now();
            match c.write(&[op]) {
                Ok(()) => next += 1,
                // (The other flushes can be slow.)
                Err(Error::Unavailable(m)) if !m.contains(what) && m.contains("flushes") => {}
                Err(e) => break (e, t0.elapsed()),
            }
            assert!(next < 10_000);
        };
        assert!(matches!(&e, Error::Unavailable(m) if m.contains(what)), "{e}");
        assert!((wait..wait * 5).contains(&took), "{took:?}");
        until(point, || blocked.load(Ordering::Relaxed));
        assert_eq!((c.len(), c.get(next).unwrap()), (next as usize, None));
        go.send(()).unwrap();
        until("room", || c.upsert(next, &[0.0; 32], &NO_ATTRS).is_ok());
        c.wait_idle().unwrap();
        assert_eq!(c.len(), next as usize + 1);
        drop(c);
        let _ = fs::remove_dir_all(&dir);
    }
}
