//! Flushes and builds on background threads.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use super::*;
use super::background::BuildLimit;
use super::maintain::Lane;
use crate::cancel::Cancel;
use crate::disk::partition_node_bytes;

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

/// Bytes of all vectors in no SSD index, whatever becomes of them.
fn flat_and_growing(c: &Collection) -> f64 {
    let st = c.stats();
    let flat: usize = st.segments.iter().filter(|s| s.kind == "flat").map(|s| s.rows).sum();
    ((flat + st.frozen_rows + st.growing_rows) * c.config().dim * 4) as f64
}

/// Writes stop when `max_unindexed_bytes` are in no index, less what a
/// running build has come through; and flushes go on during one.
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
        // The build has taken what there was when it began: 2,500 rows, or
        // more if it began late (5,000 at most). It is through with 96% of them.
        let inputs = c.indexing.lock().unwrap().runs.as_ref().expect("an index build").inputs.clone();
        let taken: usize = c.sealed().iter().filter(|s| inputs.contains(&s.id)).map(|s| s.positions()).sum();
        assert!((2500..=5000).contains(&taken), "{taken} rows taken");
        let left = c.unindexed() / (dim * 4) as f64;
        assert!((4000.0..4002.0).contains(&left), "{left}");
        assert!((n as f64 - (left + 0.96 * taken as f64)).abs() < 0.5, "{n} rows");
        let st = c.stats();
        assert!(st.segments.iter().all(|s| s.kind == "flat") && st.builds == 0, "{st:?}");
        assert!(st.frozen == 0 && st.growing_rows < 500 && st.write_rate == Some(0), "{st:?}");
        go.send(()).unwrap();
        until("more writes", || written.load(Ordering::Relaxed) > n + 2000);
    });
    c.flush().unwrap();
    c.wait_idle().unwrap();
    let st = c.stats();
    assert_eq!((st.live as u64, &st.build_error), (written.load(Ordering::Relaxed), &None));
    assert!(st.builds >= 1 && st.segments.iter().any(|s| s.kind == "ssd"));
    // Those who waited have been counted, and nobody waits now.
    assert!(st.write_wait >= Duration::from_secs(1) && st.write_rate.is_none(), "{st:?}");
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// What an index build has come through does not count any more while it
/// runs, and nothing of what it took once it is done; all of it does again if
/// it fails.
#[test]
fn what_a_build_has_done_does_not_count() {
    let dir = test_dir("coll-bg-done");
    let dim = 8;
    let cfg = CollectionConfig { max_indexed_segments: 1000, ..ssd_config(dim, Metric::L2, 500, 2500) };
    let opts = OpenOptions { max_unindexed_bytes: 100_000 * dim as u64 * 4, ..opts() };
    let c = Collection::create(&dir, cfg, opts).unwrap();
    let mut rng = SplitMix64::new(6);
    let mut model = Model::default();
    assert_eq!(c.unindexed(), 0.0);
    // Growing, frozen and flat rows do, also deleted ones.
    let (go, blocked) = block_at(&c, "flush: rows written");
    upsert_batch(&c, &mut model, &mut rng, 0..500);
    upsert_batch(&c, &mut model, &mut rng, 400..1300);
    until("a flush", || blocked.load(Ordering::Relaxed));
    assert_eq!((c.stats().frozen, c.unindexed()), (2, 1400.0 * 32.0));
    go.send(()).unwrap();
    c.flush().unwrap();
    assert_eq!((c.stats().frozen, c.unindexed()), (0, 1400.0 * 32.0));

    let (go, blocked) = block_at(&c, "build: built");
    for i in 0..30 {
        upsert_batch(&c, &mut model, &mut rng, 2000 + i * 100..2100 + i * 100);
    }
    until("a build", || blocked.load(Ordering::Relaxed));
    upsert_batch(&c, &mut model, &mut rng, 9000..9010);
    let inputs = c.indexing.lock().unwrap().runs.as_ref().expect("an index build").inputs.clone();
    let taken: usize = c.sealed().iter().filter(|s| inputs.contains(&s.id)).map(|s| s.positions()).sum();
    assert!(taken >= 2500 && inputs.len() < 30, "{taken}");
    let (all, left) = (flat_and_growing(&c), c.unindexed());
    let want = all - 0.96 * taken as f64 * 32.0;
    assert!((left - want).abs() < 2.0 && left < all, "{left} of {all}, not {want}");
    go.send(()).unwrap();
    c.wait_idle().unwrap();
    assert!(c.indexing.lock().unwrap().runs.is_none() && ssd_segments(&c) >= 1);
    assert_eq!(c.unindexed(), flat_and_growing(&c));
    assert!(c.unindexed() <= all - taken as f64 * 32.0);
    drop(c);

    // A build that fails after all.
    let (c, _) = Collection::open(&dir, OpenOptions { background: false, ..test_opts() }).unwrap();
    let next = c.manifest.lock().unwrap().next_segment;
    let blocks: Vec<PathBuf> = (next..next + 50).map(|id| dir.join(disk_name(id)).join("x")).collect();
    for i in 0..30 {
        upsert_batch(&c, &mut model, &mut rng, 20_000 + i * 100..20_100 + i * 100);
    }
    blocks.iter().for_each(|b| fs::create_dir_all(b).unwrap());
    assert!(c.maintain().is_err());
    assert!(c.indexing.lock().unwrap().runs.is_none());
    assert_eq!(c.unindexed(), flat_and_growing(&c));
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// A build that waits for its turn counts for what it has come through of
/// the line, and goes on from there.
#[test]
fn the_way_through_the_line_counts() {
    let dir = test_dir("coll-bg-line");
    let cfg = CollectionConfig { max_indexed_segments: 1000, ..ssd_config(8, Metric::L2, 500, 2500) };
    let builds = BuildLimit::new(1);
    let opts = OpenOptions { max_unindexed_bytes: 100_000 * 32, builds: Some(builds.clone()), ..opts() };
    let c = Collection::create(&dir, cfg, opts).unwrap();
    let mut rng = SplitMix64::new(6);
    let mut model = Model::default();
    let waits = |inputs: usize| {
        until("the line", || c.indexing.lock().unwrap().waits.as_ref().is_some_and(|b| b.inputs.len() == inputs));
    };
    let check = |far: f64, rows: f64| {
        let (all, left) = (flat_and_growing(&c), c.unindexed());
        assert!((all - left - far * rows * 32.0).abs() < 2.0, "{left} of {all}, {far}");
    };
    // Another's build, that has 80,000 bytes to go.
    let mut other = builds.line_up(Lane::Flat, 100_000);
    assert!(other.come());
    other.way.done.set(0.2);
    for i in 0..25 {
        upsert_batch(&c, &mut model, &mut rng, i * 100..i * 100 + 100);
    }
    c.flush().unwrap();
    waits(5);
    check(0.0, 0.0);
    // (2500 rows are as much.)
    other.way.done.set(0.6);
    check(0.25, 2500.0);
    // More that come count from then on.
    for i in 25..31 {
        upsert_batch(&c, &mut model, &mut rng, i * 100..i * 100 + 100);
    }
    c.flush().unwrap();
    waits(7);
    check(40_000.0 / 179_200.0, 3100.0);
    other.way.done.set(0.96);
    check(76_000.0 / 179_200.0, 3100.0);
    // One that has to line up again starts over, for as long as it takes.
    let one = c.lanes[Lane::Flat as usize].lock().unwrap();
    drop(other);
    until("the lane to be busy", || c.unindexed() == flat_and_growing(&c));
    for _ in 0..50 {
        std::thread::sleep(Duration::from_millis(2));
        check(0.0, 0.0);
    }
    // Its turn, after another's.
    let mut other = builds.line_up(Lane::Flat, 99_200);
    until("a turn", || other.come());
    drop(one);
    waits(7);
    other.way.done.set(0.5);
    check(0.25, 3100.0);
    // If it has to be a merge instead, that is not on its way to an index.
    let deleted: Vec<WriteOp> = (0..1600).map(|id| WriteOp::Delete { id }).collect();
    c.write(&deleted).unwrap();
    until("no index build", || c.indexing.lock().unwrap().waits.is_none());
    check(0.0, 0.0);
    // (What a build takes in: the live rows of its input.)
    let ids: Vec<u64> = c.lane_segments(Lane::Flat).iter().map(|s| s.id).collect();
    let bytes = |segments: &[u64]| c.index_bytes_of("index", segments);
    assert_eq!((bytes(&ids), bytes(&ids[3..5]), bytes(&ids[..3])), (1500 * 32, 900 * 32, 0));
    assert_eq!(c.index_bytes_of("shrink", &ids), 0);
    for i in 0..16 {
        upsert_batch(&c, &mut model, &mut rng, i * 100..i * 100 + 100);
    }
    c.flush().unwrap();
    waits(11);
    check(0.25, 4700.0);
    // It goes on from where the line ends.
    let (go, blocked) = block_at(&c, "build: built");
    drop(other);
    until("a build", || blocked.load(Ordering::Relaxed));
    check(0.5 + 0.5 * 0.96, 4700.0);
    go.send(()).unwrap();
    c.wait_idle().unwrap();
    let i = c.indexing.lock().unwrap();
    assert!(i.runs.is_none() && i.waits.is_none());
    drop(i);
    assert!(c.stats().builds >= 1 && c.unindexed() == flat_and_growing(&c));
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// Jobs of the other lane leave it at that, and so does the end of a wait
/// for what is not an index build any more.
#[test]
fn only_index_builds_change_what_index_builds_have_done() {
    let dir = test_dir("coll-bg-lanes-done");
    let cfg = CollectionConfig { max_indexed_segments: 1000, ..ssd_config(8, Metric::L2, 500, 3000) };
    let builds = BuildLimit::new(1);
    let opts = OpenOptions { max_unindexed_bytes: 100_000 * 32, builds: Some(builds.clone()), ..opts() };
    let c = Collection::create(&dir, cfg, opts).unwrap();
    let mut rng = SplitMix64::new(6);
    let mut model = Model::default();
    let mut upsert = |from: u64| {
        (from..from + 3000).step_by(100).for_each(|i| upsert_batch(&c, &mut model, &mut rng, i..i + 100));
        c.flush().unwrap();
    };
    let delete = |ids: &mut dyn Iterator<Item = u64>| {
        c.write(&ids.map(|id| WriteOp::Delete { id }).collect::<Vec<_>>()).unwrap();
    };
    let ssd = || c.lane_segments(Lane::Ssd).iter().map(|s| s.id).collect::<Vec<_>>();
    let stays = |far: f64| {
        for _ in 0..50 {
            let (is, all) = (c.unindexed(), flat_and_growing(&c));
            assert!((is - (all - far * 3000.0 * 32.0)).abs() < 2.0, "{is} of {all}");
            std::thread::sleep(Duration::from_millis(1));
        }
    };
    upsert(0);
    c.wait_idle().unwrap();
    assert_eq!(ssd().len(), 1);

    // Both lanes wait for others' builds. That for an index does not look
    // again.
    let [mut flat, mut other] = [Lane::Flat, Lane::Ssd].map(|lane| builds.line_up(lane, 96_000));
    assert!(flat.come() && other.come());
    let (go, waits) = block_thread_at(&c, Some("vecdb-index"), "build: waits");
    upsert(4000);
    until("the line", || waits.load(Ordering::Relaxed));
    flat.way.done.set(0.5);
    stays(0.25);
    // A compaction lines up, runs and is done.
    let before = ssd();
    delete(&mut (0..700));
    until("a compaction", || c.next_job(Lane::Ssd).is_some());
    stays(0.25);
    drop(other);
    until("the compaction", || c.next_job(Lane::Ssd).is_none() && ssd() != before);
    stays(0.25);
    // So does one beside an index build, which is called for.
    let (go_on, built) = block_thread_at(&c, None, "build: built");
    std::thread::scope(|s| {
        let call = s.spawn(|| c.maintain().unwrap());
        until("the build", || built.load(Ordering::Relaxed));
        stays(0.96);
        c.rebuild(&ssd(), "test", 1).unwrap();
        stays(0.96);
        go_on.send(()).unwrap();
        assert_eq!(call.join().unwrap().len(), 1);
    });
    go.send(()).unwrap();
    until("the line", || c.indexing.lock().unwrap().waits.is_none());
    assert_eq!((ssd().len(), c.unindexed()), (2, flat_and_growing(&c)));

    // What waits for an index is merged into a flat segment instead.
    let (go, waits) = block_thread_at(&c, Some("vecdb-index"), "build: waits");
    upsert(8000);
    until("the line", || waits.load(Ordering::Relaxed));
    flat.way.done.set(0.75);
    stays(24.0 / 144.0);
    delete(&mut (8000..11_000).filter(|id| id % 3 != 0));
    assert_eq!(c.next_job(Lane::Flat).unwrap().0, "shrink");
    let (go_on, built) = block_thread_at(&c, Some("vecdb-index"), "build: built");
    drop(flat);
    go.send(()).unwrap();
    until("the merge", || built.load(Ordering::Relaxed));
    stays(0.0);
    go_on.send(()).unwrap();
    c.wait_idle().unwrap();
    assert_eq!((ssd().len(), c.unindexed()), (2, flat_and_growing(&c)));
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// What waits for its turn can be another job by then, which has more to do.
#[test]
fn a_build_counts_what_it_takes_in_when_its_turn_comes() {
    let dir = test_dir("coll-bg-takes");
    let builds = BuildLimit::new(1);
    let opts = OpenOptions { max_unindexed_bytes: 100_000 * 32, builds: Some(builds.clone()), ..opts() };
    let c = Collection::create(&dir, ssd_config(8, Metric::L2, 500, 3000), opts).unwrap();
    let mut rng = SplitMix64::new(6);
    let mut model = Model::default();
    for (n, late) in [(0, false), (1, true)] {
        let mut other = builds.line_up(Lane::Flat, 96_000);
        assert!(other.come());
        // A segment to drop waits.
        let (go, waits) = block_thread_at(&c, Some("vecdb-index"), "build: waits");
        let ids = n * 10_000..n * 10_000 + 500;
        upsert_batch(&c, &mut model, &mut rng, ids.clone());
        c.flush().unwrap();
        c.write(&ids.map(|id| WriteOp::Delete { id }).collect::<Vec<_>>()).unwrap();
        assert_eq!(c.next_job(Lane::Flat).unwrap().0, "drop");
        until("the line", || waits.load(Ordering::Relaxed));
        if !late {
            go.send(()).unwrap();
        }
        // It builds an index by now.
        let from = n * 10_000 + 1000;
        (from..from + 3000).step_by(100).for_each(|i| upsert_batch(&c, &mut model, &mut rng, i..i + 100));
        c.flush().unwrap();
        let (reason, inputs) = c.next_job(Lane::Flat).unwrap();
        assert_eq!((reason, inputs.len()), ("index", 7));
        let stays = |far: f64| {
            for _ in 0..50 {
                let (is, all) = (c.unindexed(), flat_and_growing(&c));
                assert!((is - (all - far * 3500.0 * 32.0)).abs() < 2.0, "{is} of {all}");
                std::thread::sleep(Duration::from_millis(1));
            }
        };
        other.way.done.set(0.5);
        if !late {
            until("the line", || c.indexing.lock().unwrap().waits.is_some());
            stays(0.25);
        }
        let (go_on, built) = block_thread_at(&c, Some("vecdb-index"), "build: built");
        drop(other);
        if late {
            go.send(()).unwrap();
        }
        until("the build", || built.load(Ordering::Relaxed));
        stays(0.5 + 0.5 * 0.96);
        go_on.send(()).unwrap();
        c.wait_idle().unwrap();
        assert_eq!((ssd_segments(&c), c.unindexed()), (n as usize + 1, flat_and_growing(&c)));
    }
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// Only index builds free flat segments bit by bit. What `optimize` has taken
/// does not count at all, as before.
#[test]
fn other_jobs_do_not_make_room_early() {
    let dir = test_dir("coll-bg-done-others");
    let dim = 8;
    let cfg = ssd_config(dim, Metric::L2, 500, 2500);
    let c = Collection::create(&dir, cfg, OpenOptions { background: false, ..opts() }).unwrap();
    let mut rng = SplitMix64::new(6);
    let mut model = Model::default();
    let hook = |point: &'static str, seen: Arc<Mutex<Vec<(bool, f64, f64)>>>| -> Hook {
        Box::new(move |e, at| {
            if at == point {
                let st = e.stats();
                let flat: usize = st.segments.iter().filter(|s| s.kind == "flat").map(|s| s.rows).sum();
                let all = ((flat + st.frozen_rows + st.growing_rows) * 8 * 4) as f64;
                let indexing = e.indexing.lock().unwrap().runs.is_some();
                seen.lock().unwrap().push((indexing, e.unindexed(), all));
            }
        })
    };
    // An index build, for comparison.
    let seen = Arc::new(Mutex::new(Vec::new()));
    upsert_batch(&c, &mut model, &mut rng, 0..4500);
    *c.hook.lock().unwrap() = Some(hook("build: built", seen.clone()));
    assert_eq!(c.maintain().unwrap()[0].reason, "index");
    let (indexing, left, all) = seen.lock().unwrap()[0];
    assert!(indexing && left < all / 10.0, "{left} of {all}");
    // A compaction, a shrink, and an optimize.
    let delete = |ids: &mut dyn Iterator<Item = u64>| {
        c.write(&ids.map(|id| WriteOp::Delete { id }).collect::<Vec<_>>()).unwrap();
    };
    delete(&mut (0..1500));
    upsert_batch(&c, &mut model, &mut rng, 5000..5400);
    c.flush().unwrap();
    assert_eq!(c.maintain().unwrap()[0].reason, "compact");
    for i in 0..6 {
        upsert_batch(&c, &mut model, &mut rng, 8000 + i * 500..8500 + i * 500);
    }
    delete(&mut (8000..11_000).filter(|id| id % 5 < 3));
    c.flush().unwrap();
    let jobs: Vec<&str> = c.maintain().unwrap().iter().map(|j| j.reason).collect();
    assert_eq!(jobs, ["shrink"]);
    upsert_batch(&c, &mut model, &mut rng, 7000..7300);
    c.optimize().unwrap().unwrap();
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 4, "{seen:?}");
    for (i, (indexing, left, all)) in seen.iter().enumerate().skip(1) {
        assert!(!indexing, "{i}");
        let last = i + 1 == seen.len();
        assert!(if last { *left == 0.0 && *all >= 300.0 * 32.0 } else { left == all }, "{i}: {left} of {all}");
    }
    let _ = fs::remove_dir_all(&dir);
}

/// Writes are slowed down only as long as something is to come that makes
/// room; and those that take none never.
#[test]
fn writes_are_not_slowed_down_for_nothing() {
    let dim = 4;
    let limit = 4000 * dim as u64 * 4;
    let mut rng = SplitMix64::new(6);
    let mut fill = |c: &Collection, rows: u64| {
        for i in 0..rows / 100 {
            upsert_batch(c, &mut Model::default(), &mut rng, i * 100..(i + 1) * 100);
        }
    };
    // Without a limit, without background threads, and without a job.
    let never = [
        (2500, OpenOptions { max_unindexed_bytes: 0, ..opts() }),
        (2500, OpenOptions { max_unindexed_bytes: limit, background: false, ..opts() }),
        (1_000_000, OpenOptions { max_unindexed_bytes: limit, ..opts() }),
    ];
    for (index_rows, opts) in never {
        let dir = test_dir("coll-bg-never");
        let c = Collection::create(&dir, ssd_config(dim, Metric::L2, 500, index_rows), opts).unwrap();
        let (go, _) = block_at(&c, "build: built");
        fill(&c, 12_000);
        let st = c.stats();
        assert_eq!((st.write_rate, c.write_rate(), st.builds), (None, None, 0));
        let _ = go.send(());
        drop(c);
        let _ = fs::remove_dir_all(&dir);
    }

    // A flush that is to make an index build due; not if it fails.
    let dir = test_dir("coll-bg-flushfirst");
    let opts = OpenOptions { max_unindexed_bytes: limit, max_write_wait: Some(Duration::from_millis(100)), ..opts() };
    let mut cfg = ssd_config(dim, Metric::L2, 1300, 2500);
    cfg.max_indexed_segments = 1000;
    let c = Collection::create(&dir, cfg, opts).unwrap();
    let (go, blocked) = block_at(&c, "flush: rows written");
    fill(&c, 3800);
    until("a flush", || blocked.load(Ordering::Relaxed));
    // (From 3825 rows, to 5100.)
    assert_eq!((c.stats().frozen, c.write_rate()), (2, None));
    upsert_batch(&c, &mut Model::default(), &mut SplitMix64::new(1), 5000..5050);
    let rate = c.write_rate().expect("slowed down");
    assert!(rate < 256e6 && rate > 4096.0 && c.next_job(Lane::Flat).is_none(), "{rate}");
    assert_eq!(c.stats().write_rate, Some(rate as u64));
    c.background.flushed(&Err(Error::Unavailable("made up".into())));
    assert_eq!(c.write_rate(), None);
    c.background.flushed(&Ok(true));
    assert!(c.write_rate().is_some());

    // At the limit writes give up after a while, but deletes do not wait.
    let (go2, blocked2) = block_at(&c, "build: built");
    go.send(()).unwrap();
    until("a build", || blocked2.load(Ordering::Relaxed));
    let v = [0.5; 4];
    let upsert = |id: u64, rows: u64| {
        let ops: Vec<WriteOp> = (id..id + rows).map(|id| WriteOp::Upsert { id, vector: &v, attrs: &NO_ATTRS }).collect();
        c.write(&ops)
    };
    let mut id = 10_000;
    for rows in [100, 1] {
        loop {
            match upsert(id, rows) {
                Ok(()) => id += rows,
                // (Flushes can be slow.)
                Err(Error::Unavailable(m)) if m == "flushes are behind" => {}
                Err(_) => break,
            }
        }
    }
    until("flushes", || c.stats().frozen == 0);
    let (rows, waited) = (c.len(), c.stats().write_wait);
    assert!(rows > 7500 && c.unindexed() >= 5100.0 * 16.0 && c.write_rate() == Some(0.0), "{rows}");
    let mixed = [WriteOp::Delete { id: 300 }, WriteOp::Upsert { id, vector: &v, attrs: &NO_ATTRS }];
    let refused = c.write(&mixed).unwrap_err();
    assert!(matches!(&refused, Error::Unavailable(m) if m == "index builds are behind"), "{refused}");
    assert!(c.contains(300) && !c.contains(id) && c.len() == rows);
    let waited = c.stats().write_wait - waited;
    assert!(waited >= Duration::from_millis(99) && waited < Duration::from_secs(2), "{waited:?}");
    let t0 = Instant::now();
    (0..200).for_each(|id| assert!(c.delete(id).unwrap()));
    assert!(t0.elapsed() < Duration::from_secs(5), "{:?}", t0.elapsed());
    // A failed build has the same effect as a failed flush.
    c.background.build_failed(Lane::Flat, &Error::Unavailable("made up".into()));
    assert_eq!(c.write_rate(), None);
    upsert(id, 100).unwrap();
    assert_eq!(c.len(), rows - 200 + 100);
    go2.send(()).unwrap();
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// A writer that is much faster than builds is slowed down to their rate
/// instead of stopped for each; and they always have something to do. With
/// several collections (or shards), whose builds go one at a time, each batch
/// is for all of them.
fn keeps_pace(name: &str, collections: u64) {
    let dim = 64;
    let row = dim as u64 * 4;
    let builds = BuildLimit::new(1);
    let all: Vec<_> = (0..collections)
        .map(|i| {
            let mut cfg = CollectionConfig { max_indexed_segments: 1000, ..ssd_config(dim, Metric::L2, 1000, 4000) };
            (cfg.index.build_list, cfg.index.passes, cfg.index.max_degree) = (500, 2, 48);
            let opts =
                OpenOptions { max_unindexed_bytes: 8000 * row, threads: 1, builds: Some(builds.clone()), ..opts() };
            let dir = test_dir(&format!("{name}-{i}"));
            (Collection::create(&dir, cfg, opts).unwrap(), dir)
        })
        .collect();
    let mut rng = SplitMix64::new(6);
    let rows: Vec<Vec<f32>> = (0..500).map(|_| float_vector(&mut rng, dim)).collect();
    let (mut slowest, mut next, mut started) = (Duration::ZERO, 0, None);
    let mut times = Vec::new();
    let least = || all.iter().map(|(c, _)| c.stats().builds).min().unwrap();
    while least() < 4 {
        let ops: Vec<WriteOp> = (next..next + 50)
            .map(|id| WriteOp::Upsert { id, vector: &rows[id as usize % 500], attrs: &NO_ATTRS })
            .collect();
        let t0 = Instant::now();
        all.iter().for_each(|(c, _)| c.write(&ops).unwrap());
        slowest = slowest.max(t0.elapsed());
        if started.is_some() {
            times.push(t0.elapsed());
        }
        next += 50;
        assert!(next < 100_000);
        if started.is_none() && least() == 1 {
            started = Some((Instant::now(), next));
        }
        // At most the limit, and in RAM what one build takes in besides:
        // twice `index_bytes`.
        for (c, _) in &all {
            let (u, f) = (c.unindexed() / row as f64, flat_and_growing(c) / row as f64);
            assert!(u < 8100.0 && f < 16_200.0, "{u} {f}");
        }
    }
    let (at, rows_then) = started.unwrap();
    // One build of each. Writes that were stopped for each would wait for
    // most of it three times; one that waits for half of it may be a stall
    // of the machine (a loaded one has them: all tests at once).
    let round = at.elapsed() / 3;
    let long = times.iter().filter(|&&t| t > round / 2).count();
    assert!(slowest < round && long <= 1, "{slowest:?}, {long} long; a round of builds takes {round:?}");
    // What was written meanwhile is what was indexed: builds take 4,000 to
    // 8,000 rows (and a segment), what there is when they begin.
    let per = (next - rows_then) / 3;
    assert!((2500..9000).contains(&per), "{per} rows per build");
    let waited: Duration = all.iter().map(|(c, _)| c.stats().write_wait).sum();
    assert!(waited > round, "{waited:?} of {round:?}");
    for (c, dir) in all {
        assert_eq!(c.stats().build_error, None);
        drop(c);
        let _ = fs::remove_dir_all(&dir);
    }
}

#[test]
fn writes_keep_pace_with_builds() {
    keeps_pace("coll-bg-pace", 1);
}

#[test]
fn writes_keep_pace_with_the_builds_of_all_shards() {
    keeps_pace("coll-bg-paces", 4);
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
    // The compactions have not come: neither has lost its deleted rows. But
    // index builds have: in none are more than the limit, what one build
    // takes in (5,000 rows, twice `index_bytes`) and a segment.
    for (c, before) in [&a, &b].into_iter().zip(before) {
        let st = c.stats();
        assert_eq!((st.segments[0].rows, st.segments[0].deleted), (3600, 1200), "{st:?}");
        let left = flat_and_growing(c) / 16.0;
        assert!(st.builds >= before + 2 && ssd_segments(c) >= 3 && left < 10_000.0, "{left} {st:?}");
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

/// They take up to twice `index_bytes`, if an eighth of the limit is less, as
/// long as one partition of a build holds that and the largest segment more.
/// A build that waits for its turn counts for what it takes in by then.
#[test]
fn index_builds_take_up_to_twice_index_bytes() {
    // Rows of 16 bytes, `index_bytes` 3,000 of them: the limit, the rows the
    // build memory holds, and the rows a build is to take, by the largest
    // segment. (Segments are 500 rows or more: some fill while flushes are
    // behind.)
    type Rows = fn(usize) -> usize;
    let cases: [(u64, Option<usize>, Rows); 4] = [
        (20_000, None, |_| 6000),
        (20_000, Some(4750), |largest| (4750 - largest).max(3000)),
        // One does not even hold `index_bytes`: that, in partitions, as before.
        (20_000, Some(2000), |_| 3000),
        // An eighth of the limit is more.
        (64_000, None, |_| 8000),
    ];
    for (limit, memory, want) in cases {
        let dir = test_dir("coll-bg-twice");
        let cfg = CollectionConfig { max_indexed_segments: 1000, ..ssd_config(4, Metric::L2, 500, 3000) };
        let builds = BuildLimit::new(1);
        let mut opts = OpenOptions { max_unindexed_bytes: limit * 16, builds: Some(builds.clone()), ..opts() };
        if let Some(nodes) = memory {
            opts.build_memory = nodes as u64 * partition_node_bytes(4, cfg.index.max_degree);
        }
        let c = Collection::create(&dir, cfg, opts).unwrap();
        let mut other = builds.line_up(Lane::Flat, 96_000);
        assert!(other.come());
        let mut rng = SplitMix64::new(4);
        let mut model = Model::default();
        // One batch of 1,000 rows makes a segment larger than the others,
        // after 4,000 rows.
        let mut at = 0;
        while at < 10_000 {
            let n = if at == 4000 { 1000 } else { 100 };
            upsert_batch(&c, &mut model, &mut rng, at..at + n);
            at += n;
        }
        c.flush().unwrap();
        // The oldest segments, until they have what is wanted.
        let flat: Vec<(u64, usize)> = c.sealed().iter().map(|s| (s.id, s.live())).collect();
        let largest = flat.iter().map(|f| f.1).max().unwrap();
        assert!((1000..1750).contains(&largest), "{flat:?}");
        let want = want(largest);
        let (mut rows, mut taken) = (0, Vec::new());
        for &(id, live) in &flat {
            if rows >= want {
                break;
            }
            rows += live;
            taken.push(id);
        }
        // There are more, and one partition holds them where it holds `index_bytes`.
        assert!(rows >= want && rows < 10_000, "{flat:?}");
        assert!(memory.is_none_or(|nodes| nodes < 3000 || rows <= nodes), "{flat:?}");
        assert_eq!(c.next_job(Lane::Flat), Some(("index", taken)), "{limit} {memory:?}");
        // Half of the way through the line, which is the other build and this one.
        other.way.done.set(0.5);
        let gone = 0.5 * 96_000.0 / (rows * 16 + 96_000) as f64 * (rows * 16) as f64;
        until("the build that waits", || (c.unindexed() - (flat_and_growing(&c) - gone)).abs() < 2.0);
        drop(other);
        c.wait_idle().unwrap();
        let st = c.stats();
        assert_eq!(st.segments[0].kind, "ssd");
        assert_eq!((st.segments[0].rows, st.live), (rows, 10_000), "{st:?}");
        check_approx(&c, &model, &mut rng, 0.9);
        drop(c);
        let _ = fs::remove_dir_all(&dir);
    }
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

/// A write that has been refused is known when it comes again, by what is in
/// it; also one that was let through, but then had to wait for a flush.
#[test]
fn a_refused_write_is_known_again() {
    let dim = 4;
    let max_write_wait = Some(Duration::from_millis(100));
    let limit = 4000 * dim as u64 * 4;
    let opts = OpenOptions { max_unindexed_bytes: limit, max_write_wait, ..opts() };
    let dir = test_dir("coll-bg-known");
    let mut cfg = ssd_config(dim, Metric::L2, 1300, 2500);
    cfg.max_indexed_segments = 1000;
    let c = Collection::create(&dir, cfg, opts).unwrap();
    let (go, blocked) = block_at(&c, "flush: rows written");
    let attrs = NO_ATTRS.clone();
    let write = |ids: std::ops::Range<u64>, x: f32| {
        let rows: Vec<(u64, [f32; 4])> = ids.map(|id| (id, [id as f32, (id * 7 % 1000) as f32, 1.0, x])).collect();
        let ops: Vec<WriteOp> =
            rows.iter().map(|(id, v)| WriteOp::Upsert { id: *id, vector: v, attrs: &attrs }).collect();
        match c.write(&ops) {
            Ok(()) => None,
            Err(Error::Unavailable(why)) => Some(why),
            Err(e) => panic!("{e}"),
        }
    };
    // Two segments to flush and one full, and a rate (see
    // `writes_are_not_slowed_down_for_nothing`).
    (0..77).for_each(|i| assert_eq!(write(i * 50..(i + 1) * 50, 0.0), None));
    until("a flush", || blocked.load(Ordering::Relaxed));
    // What is written is owed: the one after it has to wait for that.
    assert_eq!(c.line.owed(), 0.0);
    assert_eq!(write(3850..3900, 0.0), None);
    assert_eq!(c.line.owed(), 50.0 * 16.0);
    assert!(c.stats().frozen == 2 && c.write_rate().is_some_and(|r| r > 1e6));
    let (rows, flushes) = (c.len(), Some("flushes are behind".to_string()));
    let last = |n: usize| {
        let (line, kept) = c.line.places();
        assert!(line.is_empty() && kept.len() >= n, "{line:?} {kept:?}");
        kept[kept.len() - n..].to_vec()
    };
    assert!(c.line.places().1.is_empty());
    assert_eq!(write(5000..5100, 1.0), flushes);
    assert_eq!(c.line.owed(), 0.0);
    assert_eq!(write(5000..5100, 2.0), flushes);
    let [a, b] = last(2).try_into().unwrap();
    assert!(a.0 != b.0 && a.1 < b.1, "{a:?} {b:?}");
    // It has the place that it had.
    assert_eq!(write(5000..5100, 1.0), flushes);
    assert_eq!(last(2), [b, a]);
    // Another one has not: fewer rows, other ids, other numbers.
    for other in [write(5000..5099, 1.0), write(5001..5101, 1.0), write(5000..5100, 1.5)] {
        assert_eq!(other, flushes);
    }
    let all = last(5);
    assert!(all[..2] == [b, a] && all[2].1 > b.1 && all[2..].is_sorted_by(|x, y| x.1 < y.1), "{all:?}");
    assert_eq!(c.len(), rows);

    // The same for those that index builds are to blame for.
    let (go2, blocked2) = block_at(&c, "build: built");
    go.send(()).unwrap();
    until("a build", || blocked2.load(Ordering::Relaxed));
    let mut id = 10_000;
    for n in [100, 1] {
        while write(id..id + n, 3.0).is_none_or(|why| why == flushes.clone().unwrap()) {
            id += n;
        }
    }
    until("flushes", || c.stats().frozen == 0);
    assert_eq!(c.write_rate(), Some(0.0));
    let (rows, builds) = (c.len(), Some("index builds are behind".to_string()));
    assert_eq!((write(20_000..20_001, 1.0), write(20_000..20_001, 2.0)), (builds.clone(), builds.clone()));
    let [a, b] = last(2).try_into().unwrap();
    assert_eq!(write(20_000..20_001, 1.0), builds);
    assert_eq!(last(2), [b, a]);
    // Not deletes, which do not wait.
    let before = c.line.places();
    assert!(!c.delete(20_000).unwrap());
    assert_eq!(c.line.places(), before);

    // Nothing is written of one that has begun and is not finished.
    c.background.build_failed(Lane::Flat, &Error::Unavailable("made up".into()));
    let v = [0.5; 4];
    let ops = [WriteOp::Upsert { id: 20_000, vector: &v, attrs: &attrs }, WriteOp::Delete { id: 5 }];
    let seq = c.stats().next_seq;
    drop(c.begin(&ops).unwrap());
    assert!(!c.contains(20_000) && c.contains(5) && (c.len(), c.stats().next_seq) == (rows, seq));
    assert_eq!(c.begin(&ops).unwrap().finish().unwrap(), [false, true]);
    assert!(c.contains(20_000) && !c.contains(5) && (c.len(), c.stats().next_seq) == (rows, seq + 2));
    let batches = c.stats().write_batches;
    assert_eq!(c.begin(&[]).unwrap().finish().unwrap(), [false; 0]);
    assert_eq!(c.stats().write_batches, batches);
    go2.send(()).unwrap();
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}
