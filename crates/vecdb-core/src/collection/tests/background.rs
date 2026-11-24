//! Flushes and builds on background threads.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use super::*;
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
    let (go, wait) = mpsc::channel();
    let blocked = Arc::new(AtomicBool::new(false));
    let (b, mut done) = (blocked.clone(), false);
    *c.hook.lock().unwrap() = Some(Box::new(move |_, at| {
        if at == point && !done {
            done = true;
            b.store(true, Ordering::Relaxed);
            let _ = wait.recv_timeout(Duration::from_secs(30));
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
        assert!(c.stats().frozen == 0 && c.next_job().is_none());
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
