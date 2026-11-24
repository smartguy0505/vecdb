//! Searches, writes and index builds at the same time: what a search sees,
//! and what a build keeps.

use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::*;

#[test]
fn a_collection_is_shared_between_threads() {
    fn shared<T: Send + Sync>() {}
    shared::<Collection>();
}

/// Every live id once: a search with room for all rows and a list long
/// enough to walk every SSD node.
fn search_all(c: &Collection, snap: Option<&Snapshot>, n: usize) -> Vec<SearchResult> {
    let q = vec![0.0; c.config().dim];
    let opts = SearchOptions { l: 4 * n + 100, ..Default::default() };
    let res = match snap {
        Some(s) => c.search_in(s, &q, n + 50, &opts).unwrap().0,
        None => c.search(&q, n + 50, &opts).unwrap(),
    };
    let mut seen = HashSet::new();
    for r in &res {
        assert!(seen.insert(r.id), "id {} returned twice", r.id);
    }
    res
}

fn ids_of(res: &[SearchResult]) -> Vec<u64> {
    let mut ids: Vec<u64> = res.iter().map(|r| r.id).collect();
    ids.sort_unstable();
    ids
}

#[test]
fn old_snapshots_keep_their_view() {
    let dir = test_dir("coll-snapshot");
    let dim = 8;
    let mut cfg = ssd_config(dim, Metric::L2, 4000, 2048);
    cfg.fields = vec![Field::int("g")];
    let c = Collection::create(&dir, cfg, test_opts()).unwrap();
    let mut rng = SplitMix64::new(5);
    let mut vectors: HashMap<u64, Vec<f32>> = HashMap::new();
    let mut put = |c: &Collection, ids: std::ops::Range<u64>, rng: &mut SplitMix64| {
        for id in ids {
            let v = float_vector(rng, dim);
            c.upsert(id, &v, &Attrs::new().int("g", (id % 50) as i64)).unwrap();
            vectors.insert(id, v);
        }
    };
    // An SSD segment, a flat one and growing rows.
    put(&c, 0..3000, &mut rng);
    c.flush().unwrap();
    assert_eq!(c.maintain().unwrap().len(), 1);
    put(&c, 3000..3100, &mut rng);
    c.flush().unwrap();
    put(&c, 3100..3110, &mut rng);
    assert_eq!(kinds(&c), ["ssd", "flat"]);
    let before: Vec<u64> = (0..3110).collect();

    let old = c.snapshot();
    // In each kind of segment: one id upserted, one deleted. And a new id.
    let moved = [5u64, 3050, 3105];
    let gone = [6u64, 3051, 3106];
    for &id in &moved {
        c.upsert(id, &[9.0; 8], &Attrs::new().int("g", 99)).unwrap();
    }
    for &id in &gone {
        assert!(c.delete(id).unwrap());
    }
    c.upsert(9999, &[9.0; 8], &Attrs::new().int("g", 99)).unwrap();
    assert!(c.stats().delete_records >= 4, "{:?}", c.stats());

    let check_old = |c: &Collection| {
        assert_eq!(ids_of(&search_all(c, Some(&old), 3110)), before);
        for id in moved.iter().chain(&gone) {
            let (v, a) = c.get_in(&old, *id).unwrap().unwrap();
            assert_eq!(v, vectors[id], "id {id}");
            assert_eq!(a, Attrs::new().int("g", (id % 50) as i64));
        }
        assert!(c.get_in(&old, 9999).unwrap().is_none());
        // Its filters see the old rows too: the scan and the walk.
        for mode in [FilterMode::Scan, FilterMode::Graph] {
            let opts = SearchOptions {
                l: 12000,
                filter: Some(Filter::int_eq("g", 6)),
                filter_mode: mode,
                ..Default::default()
            };
            let got = c.search_in(&old, &vectors[&6], 1, &opts).unwrap().0;
            assert_eq!((got[0].id, got[0].distance), (6, 0.0), "{mode:?}");
            let none = SearchOptions { filter: Some(Filter::int_eq("g", 99)), ..opts };
            assert!(c.search_in(&old, &[9.0; 8], 5, &none).unwrap().0.is_empty(), "{mode:?}");
        }
    };
    check_old(&c);
    let mut after: Vec<u64> = before.iter().copied().filter(|id| !gone.contains(id)).chain([9999]).collect();
    after.sort_unstable();
    let check_new = |c: &Collection| {
        assert_eq!(ids_of(&search_all(c, None, 3110)), after);
        for &id in &moved {
            assert_eq!(c.get(id).unwrap().unwrap().0, vec![9.0; 8]);
        }
        assert!(gone.iter().all(|&id| c.get(id).unwrap().is_none() && !c.contains(id)));
        let opts = SearchOptions { filter: Some(Filter::int_eq("g", 99)), ..Default::default() };
        assert_eq!(c.search(&[9.0; 8], 10, &opts).unwrap().len(), 4);
    };
    check_new(&c);

    // A flush and a rebuild of everything: the old snapshot still reads the
    // segments they replaced, from files that are gone.
    c.flush().unwrap();
    c.optimize().unwrap();
    assert_eq!(kinds(&c), ["ssd"]);
    check_old(&c);
    check_new(&c);

    // Once no search predates them, deletes drop their sequence numbers.
    drop(old);
    c.upsert(10_000, &[1.0; 8], &Attrs::new().int("g", 1)).unwrap();
    assert!(c.delete(10_000).unwrap());
    c.upsert(10_001, &[1.0; 8], &Attrs::new().int("g", 1)).unwrap();
    assert_eq!(c.stats().delete_records, 0);
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

fn kinds(c: &Collection) -> Vec<&'static str> {
    c.stats().segments.iter().map(|s| s.kind).collect()
}

/// A group's rows are written together, all with the same vector: its
/// round, and the group. From a query at the origin, the group's rows are
/// at one distance, which gives the round back.
fn group_vector(dim: usize, group: u64, round: u64) -> Vec<f32> {
    let mut v = vec![0.0; dim];
    v[0] = round as f32;
    v[1] = (group + 1) as f32;
    v
}

fn round_of(group: u64, distance: f32) -> u64 {
    let g = (group + 1) as f32;
    (distance - g * g).sqrt().round() as u64
}

const GROUP: u64 = 8;

fn write_group(c: &Collection, group: u64, round: u64) {
    let dim = c.config().dim;
    let v = group_vector(dim, group, round);
    let attrs = Attrs::new().int("g", group as i64);
    let ops: Vec<WriteOp> =
        (0..GROUP).map(|i| WriteOp::Upsert { id: group * GROUP + i, vector: &v, attrs: &attrs }).collect();
    c.write(&ops).unwrap();
}

/// Rounds per group in a search's results: every group whole, each at one
/// round.
fn rounds(res: &[SearchResult], groups: u64) -> Vec<u64> {
    let mut by_group: Vec<Vec<f32>> = vec![Vec::new(); groups as usize];
    for r in res {
        by_group[(r.id / GROUP) as usize].push(r.distance);
    }
    by_group
        .iter()
        .enumerate()
        .map(|(g, ds)| {
            assert_eq!(ds.len() as u64, GROUP, "group {g}: {} rows", ds.len());
            assert!(ds.iter().all(|&d| d == ds[0]), "group {g} mixes rounds: {ds:?}");
            round_of(g as u64, ds[0])
        })
        .collect()
}

/// Writers upsert whole groups while readers search and flushes, index
/// builds, compactions and merges run: every search sees each id once, each
/// group from one batch, and never an older round than it saw before.
#[test]
fn searches_see_whole_batches_and_each_id_once() {
    whole_batches("coll-concurrent", &test_opts());
}

#[test]
fn searches_see_whole_batches_with_background_threads() {
    whole_batches("coll-concurrent-bg", &OpenOptions { background: true, ..test_opts() });
}

#[test]
fn searches_see_whole_batches_with_a_pool() {
    let pool = Some(SearchPool::new(3, 4).unwrap());
    whole_batches("coll-concurrent-pool", &OpenOptions { background: true, pool, ..test_opts() });
}

fn whole_batches(name: &str, opts: &OpenOptions) {
    let dir = test_dir(name);
    let (dim, groups) = (8, 300u64);
    let n = (groups * GROUP) as usize;
    let mut cfg = ssd_config(dim, Metric::L2, 400, 2048);
    cfg.fields = vec![Field::int("g")];
    let c = Collection::create(&dir, cfg, opts.clone()).unwrap();
    for g in 0..groups {
        write_group(&c, g, 0);
    }
    c.maintain().unwrap();
    c.wait_idle().unwrap();
    let round = AtomicU64::new(1);
    let last = Mutex::new(vec![0u64; groups as usize]);
    let stop = AtomicBool::new(false);
    let (searches, jobs) = (AtomicU64::new(0), Mutex::new(Vec::new()));
    let t0 = std::time::Instant::now();
    std::thread::scope(|s| {
        let writers: Vec<_> = (0..2u64)
            .map(|w| {
                let (c, round, last, t0) = (&c, &round, &last, &t0);
                s.spawn(move || {
                    let mut rng = SplitMix64::new(w + 10);
                    for i in 0.. {
                        // Until enough has happened.
                        if i >= 300 && (c.stats().builds > 4 || t0.elapsed().as_secs() > 30) {
                            break;
                        }
                        let g = (rng.next_u64() % (groups / 2)) * 2 + w;
                        let r = round.fetch_add(1, Ordering::Relaxed);
                        write_group(c, g, r);
                        last.lock().unwrap()[g as usize] = r;
                    }
                })
            })
            .collect();
        s.spawn(|| {
            while !opts.background && !stop.load(Ordering::Relaxed) {
                jobs.lock().unwrap().extend(c.maintain().unwrap());
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        });
        for t in 0..3u64 {
            let (c, stop, searches) = (&c, &stop, &searches);
            s.spawn(move || {
                let mut rng = SplitMix64::new(t);
                let mut seen = vec![0u64; groups as usize];
                while !stop.load(Ordering::Relaxed) {
                    let now = rounds(&search_all(c, None, n), groups);
                    for g in 0..groups as usize {
                        assert!(now[g] >= seen[g], "group {g} went back from round {} to {}", seen[g], now[g]);
                    }
                    seen = now;
                    // One group by filter (a scan in SSD segments).
                    let g = rng.next_u64() % groups;
                    let opts =
                        SearchOptions { l: 64, filter: Some(Filter::int_eq("g", g as i64)), ..Default::default() };
                    let got = c.search(&vec![0.0; dim], 20, &opts).unwrap();
                    assert_eq!(got.len() as u64, GROUP, "group {g}");
                    assert!(got.iter().all(|r| r.distance == got[0].distance), "group {g}: {got:?}");
                    assert!(round_of(g, got[0].distance) >= seen[g as usize]);
                    let id = rng.next_u64() % n as u64;
                    assert!(c.get(id).unwrap().is_some(), "id {id}");
                    searches.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
        for w in writers {
            w.join().unwrap();
        }
        stop.store(true, Ordering::Relaxed);
    });
    let jobs = jobs.into_inner().unwrap();
    let reasons: HashSet<&str> = jobs.iter().map(|j| j.reason).collect();
    assert!(searches.load(Ordering::Relaxed) >= 20, "{} searches", searches.load(Ordering::Relaxed));
    let st = c.stats();
    assert!(st.flushes > 2 && st.builds > 2, "{st:?}");
    assert!(opts.background || reasons.contains("compact") || reasons.contains("merge"), "jobs: {reasons:?}");
    c.wait_idle().unwrap();
    let want = last.into_inner().unwrap();
    assert_eq!(rounds(&search_all(&c, None, n), groups), want);
    // No search is open: the next write drops every delete record.
    write_group(&c, 0, want[0]);
    assert_eq!(c.stats().delete_records, 0);
    drop(c);
    let (c, _) = Collection::open(&dir, opts.clone()).unwrap();
    assert_eq!(rounds(&search_all(&c, None, n), groups), want);
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// Deletes and upserts while an index build runs, with flushes in
/// between: the new segment leaves out what was deleted, in memory, after
/// a crash right after the commit, and after a clean reopen.
#[test]
fn builds_carry_deletes_made_while_they_run() {
    let dir = test_dir("coll-carry");
    let dim = 16;
    let mut cfg = ssd_config(dim, Metric::L2, 1000, 2048);
    cfg.fields = vec![Field::int("g")];
    // One build: no compaction of its output, which would hide what it
    // carried.
    cfg.compact_deleted = 1.0;
    let c = Collection::create(&dir, cfg, test_opts()).unwrap();
    let mut rng = SplitMix64::new(3);
    let mut model = Model::default();
    let attrs = |id: u64| Attrs::new().int("g", (id % 7) as i64);
    for batch in 0..60u64 {
        let rows: Vec<(u64, Vec<f32>, Attrs)> =
            (batch * 100..(batch + 1) * 100).map(|id| (id, float_vector(&mut rng, dim), attrs(id))).collect();
        let ops: Vec<WriteOp> =
            rows.iter().map(|(id, v, a)| WriteOp::Upsert { id: *id, vector: v, attrs: a }).collect();
        c.write(&ops).unwrap();
        ops.iter().for_each(|op| model.apply(op));
    }
    assert!(kinds(&c).iter().all(|&k| k == "flat"));
    let mut ids: Vec<u64> = model.rows.keys().copied().collect();
    ids.sort_unstable();
    let reports = std::thread::scope(|s| {
        let job = s.spawn(|| c.maintain().unwrap());
        let mut ops = 0;
        // Until the build commits: a flush after it would write what it
        // carried (it would be the commit's job).
        while !job.is_finished() && !kinds(&c).contains(&"ssd") && !ids.is_empty() {
            let id = ids.swap_remove((rng.next_u64() % ids.len() as u64) as usize);
            if rng.next_u64() % 10 < 7 {
                let op = WriteOp::Delete { id };
                c.write(&[op]).unwrap();
                model.apply(&op);
            } else {
                let v = float_vector(&mut rng, dim);
                let a = attrs(id + 1);
                let op = WriteOp::Upsert { id, vector: &v, attrs: &a };
                c.write(&[op]).unwrap();
                model.apply(&op);
            }
            ops += 1;
            // Flushes during the build move its deletes from the WAL to the
            // inputs' tombstone files, which its commit retires.
            if ops % 250 == 0 {
                c.flush().unwrap();
            }
        }
        job.join().unwrap()
    });
    let carried: usize = reports.iter().map(|r| r.carried_deletes).sum();
    assert!(carried > 0, "the build ended before any write: {reports:?}");
    let check = |c: &Collection| {
        let mut live = c.live_ids();
        live.sort_unstable();
        let mut want: Vec<u64> = model.rows.keys().copied().collect();
        want.sort_unstable();
        assert_eq!(live, want);
        for (id, (v, a)) in &model.rows {
            assert_eq!(c.get(*id).unwrap(), Some((v.clone(), a.clone())), "id {id}");
        }
    };
    check(&c);
    // A crash right after the commit: the copy holds what the disk holds.
    let crashed = test_dir("coll-carry-crash");
    copy_dir(&dir, &crashed);
    let (c2, _) = Collection::open(&crashed, test_opts()).unwrap();
    check(&c2);
    drop(c2);
    drop(c);
    let (c, _) = Collection::open(&dir, test_opts()).unwrap();
    check(&c);
    drop(c);
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&crashed);
}

/// Writers with `Sync` durability at once: their batches share WAL writes
/// and fsyncs, each writer sees its write as soon as it returns, and every
/// acknowledged write survives a reopen.
#[test]
fn concurrent_writers_share_fsyncs() {
    let dir = test_dir("coll-group");
    let dim = 4;
    let opts = OpenOptions { durability: Durability::Sync, ..test_opts() };
    let c = Collection::create(&dir, small_config(dim, Metric::L2, 100_000), opts.clone()).unwrap();
    let (threads, each) = (16u64, 60u64);
    std::thread::scope(|s| {
        for t in 0..threads {
            let c = &c;
            s.spawn(move || {
                for i in 0..each {
                    let id = t * each + i;
                    c.upsert(id, &[id as f32, 0.0, 0.0, 0.0], &Attrs::default()).unwrap();
                    assert_eq!(c.get(id).unwrap().unwrap().0[0], id as f32);
                    if i % 3 == 2 {
                        assert!(c.delete(id - 1).unwrap());
                        assert!(!c.delete(id - 1).unwrap());
                    }
                }
            });
        }
    });
    let st = c.stats();
    let deletes = threads * (each / 3) * 2;
    assert_eq!(st.write_batches, threads * each + deletes);
    assert_eq!(st.wal_syncs, st.write_groups);
    assert!(st.write_groups < st.write_batches, "no batches shared a group: {st:?}");
    let live = |c: &Collection| {
        let mut ids = c.live_ids();
        ids.sort_unstable();
        ids
    };
    let want: Vec<u64> = (0..threads * each).filter(|id| id % each % 3 != 1).collect();
    assert_eq!(live(&c), want);
    drop(c);
    let (c, report) = Collection::open(&dir, opts).unwrap();
    assert_eq!(report.replay.batches, threads * each + deletes);
    assert_eq!(live(&c), want);
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// A group that fails to log (here: the WAL poisoned, as after a failed
/// fsync) answers each of its writers with an error, including those
/// waiting for another writer to log it, and applies none of it.
#[test]
fn a_failed_group_answers_every_writer() {
    let dir = test_dir("coll-group-fail");
    let c = std::sync::Arc::new(Collection::create(&dir, small_config(4, Metric::L2, 100_000), test_opts()).unwrap());
    let (tx, rx) = std::sync::mpsc::channel();
    let threads = 8u64;
    for t in 0..threads {
        let (c, tx) = (c.clone(), tx.clone());
        std::thread::spawn(move || {
            let mut ok = 0u64;
            while c.upsert(t << 32 | ok, &[1.0; 4], &Attrs::default()).is_ok() {
                ok += 1;
            }
            drop(c);
            tx.send(ok).unwrap();
        });
    }
    std::thread::sleep(std::time::Duration::from_millis(100));
    c.poison_wal();
    // Not a scoped join: a writer left waiting would hang the test.
    let mut acked = 0;
    for _ in 0..threads {
        acked += rx.recv_timeout(std::time::Duration::from_secs(10)).expect("a writer got no answer");
    }
    assert!(acked > 0);
    assert_eq!(c.live_ids().len() as u64, acked);
    assert!(c.write(&[WriteOp::Delete { id: 0 }]).is_err());
    let c = std::sync::Arc::into_inner(c).expect("the writers are done");
    drop(c);
    let (c, _) = Collection::open(&dir, test_opts()).unwrap();
    assert_eq!(c.live_ids().len() as u64, acked);
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// An index build leaves out rows that buffered writes deleted. After a
/// power loss that takes those writes back, the rows must be there again:
/// an upsert's old row may go only once its new row is durable.
#[test]
fn builds_do_not_outlive_the_buffered_deletes_they_apply() {
    let dir = test_dir("coll-powerloss");
    let dim = 8;
    let c = Collection::create(&dir, small_config(dim, Metric::L2, 100_000), test_opts()).unwrap();
    for id in 0..50u64 {
        c.upsert(id, &[id as f32; 8], &Attrs::default()).unwrap();
    }
    c.flush().unwrap();
    // Only in the WAL, not fsynced (durability is Buffered).
    c.upsert(5, &[100.0; 8], &Attrs::default()).unwrap();
    assert!(c.delete(6).unwrap());
    let ids: Vec<u64> = c.sealed().iter().map(|s| s.id).collect();
    c.rebuild(&ids, "test", 2).unwrap();
    let crashed = test_dir("coll-powerloss-crash");
    super::logs::power_loss_copy(&c, &dir, &crashed, || 0.0);
    drop(c);
    let (c, _) = Collection::open(&crashed, test_opts()).unwrap();
    // A prefix of the batches: before both, after the upsert, or after both.
    let five = c.get(5).unwrap().map(|(v, _)| v[0]);
    match c.stats().next_seq - 51 {
        0 => assert!(five == Some(5.0) && c.contains(6), "id 5: {five:?}, id 6: {}", c.contains(6)),
        1 => assert!(five == Some(100.0) && c.contains(6), "id 5: {five:?}, id 6: {}", c.contains(6)),
        2 => assert!(five == Some(100.0) && !c.contains(6), "id 5: {five:?}, id 6: {}", c.contains(6)),
        n => panic!("{n} writes after the flush"),
    }
    drop(c);
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&crashed);
}
