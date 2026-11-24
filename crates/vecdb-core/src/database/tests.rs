use std::collections::HashMap;
use std::time::Instant;

use super::*;
use crate::collection::{Attrs, Durability, Field, Filter, SearchOptions};
use crate::disk::test_dir;
use crate::distance::Metric;
use crate::rng::SplitMix64;

const DIM: usize = 4;

fn opts() -> DatabaseOptions {
    let shard =
        OpenOptions { durability: Durability::Buffered, direct_io: false, threads: 2, ..OpenOptions::default() };
    DatabaseOptions { shard, search_threads: 3, search_cpus: 0, builds: 1 }
}

fn config(seal_rows: usize) -> CollectionConfig {
    let mut c = CollectionConfig::new(DIM, Metric::L2);
    c.seal_bytes = (seal_rows * DIM * 4) as u64;
    c.index_bytes = u64::MAX;
    c.fields = vec![Field::int("n")];
    c
}

fn open(dir: &Path) -> (Database, DatabaseReport) {
    Database::open(dir, &opts()).unwrap()
}

/// Small whole numbers, so that distances are exact and tie often.
fn vector(rng: &mut SplitMix64) -> Vec<f32> {
    (0..DIM).map(|_| (rng.next_u64() % 9) as f32 - 4.0).collect()
}

fn upsert(id: u64, vector: Vec<f32>) -> Op {
    Op::Upsert(Row { id, vector, attrs: Attrs::new().int("n", (id % 10) as i64) })
}

/// The ids of all rows, by searching.
fn ids(c: &ShardedCollection) -> Vec<u64> {
    let req = SearchRequest { vector: vec![0.0; DIM], k: 100_000, options: SearchOptions::default() };
    let mut ids: Vec<u64> = c.search(&req).unwrap().results.iter().map(|r| r.id).collect();
    ids.sort_unstable();
    ids
}

fn entries(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> =
        fs::read_dir(dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    v.sort();
    v
}

fn copy_tree(from: &Path, to: &Path) {
    let _ = fs::remove_dir_all(to);
    fs::create_dir_all(to).unwrap();
    for e in fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        if e.file_type().unwrap().is_dir() {
            copy_tree(&e.path(), &to.join(e.file_name()));
        } else {
            fs::copy(e.path(), to.join(e.file_name())).unwrap();
        }
    }
}

#[test]
fn collections_are_created_found_and_dropped() {
    let dir = test_dir("db-basic");
    let (db, report) = open(&dir);
    assert!(report.opened.is_empty() && db.names().is_empty());
    for (i, name) in ["b", "a", "c.1-x_Y"].into_iter().enumerate() {
        let c = db.create_collection(name, config(50), i + 1).unwrap();
        assert_eq!((c.name(), c.shards().len(), c.config()), (name, i + 1, &config(50)));
        c.write(vec![upsert(i as u64, vec![1.0; DIM])]).unwrap();
    }
    assert!(matches!(db.create_collection("a", config(50), 1), Err(Error::AlreadyExists(_))));
    for bad in ["", ".", "..", ".a", "-a", "a/b", "a b", "a\0", "\u{e9}", &"x".repeat(MAX_NAME + 1)] {
        assert!(matches!(db.create_collection(bad, config(50), 1), Err(Error::InvalidArgument(_))), "{bad:?}");
    }
    for shards in [0, MAX_SHARDS + 1] {
        assert!(matches!(db.create_collection("d", config(50), shards), Err(Error::InvalidArgument(_))));
    }
    let mut bad = config(50);
    bad.dim = 0;
    assert!(db.create_collection("d", bad, 1).is_err());
    assert_eq!(db.names(), ["a", "b", "c.1-x_Y"]);
    assert!(matches!(db.collection("d"), Err(Error::NotFound(_))));
    assert!(matches!(db.drop_collection("d"), Err(Error::NotFound(_))));
    // Only one process at a time.
    assert!(Database::open(&dir, &opts()).is_err());
    drop(db);

    let (db, report) = open(&dir);
    assert_eq!(report.opened.iter().map(|(n, r)| (n.as_str(), r.len())).collect::<Vec<_>>(), [
        ("a", 2),
        ("b", 1),
        ("c.1-x_Y", 3)
    ]);
    assert!(report.failed.is_empty() && report.removed.is_empty() && report.unknown.is_empty());
    for (i, name) in ["b", "a", "c.1-x_Y"].into_iter().enumerate() {
        assert_eq!(ids(&db.collection(name).unwrap()), [i as u64]);
    }
    // A name can be used again; the new collection is empty.
    db.drop_collection("a").unwrap();
    assert_eq!(db.names(), ["b", "c.1-x_Y"]);
    let c = db.create_collection("a", config(50), 1).unwrap();
    assert!(ids(&c).is_empty());
    drop(c);
    drop(db);
    assert_eq!(entries(&dir), ["CATALOG", "LOCK", "col-00000001", "col-00000003", "col-00000004"]);
    let (db, report) = open(&dir);
    assert!(report.removed.is_empty());
    assert!(ids(&db.collection("a").unwrap()).is_empty());
    drop(db);
    let _ = fs::remove_dir_all(&dir);
}

/// Random batches over three shards match a model: results per op, gets,
/// searches with and without filters (exact here), and a reopen.
#[test]
fn shards_together_match_the_model() {
    let dir = test_dir("db-model");
    let (db, _) = open(&dir);
    let c = db.create_collection("m", config(40), 3).unwrap();
    let mut rng = SplitMix64::new(7);
    let mut model: HashMap<u64, Vec<f32>> = HashMap::new();
    let check = |c: &ShardedCollection, model: &HashMap<u64, Vec<f32>>, rng: &mut SplitMix64| {
        assert_eq!(c.len().unwrap(), model.len());
        for id in 0..300 {
            let want = model.get(&id).map(|v| match upsert(id, v.clone()) {
                Op::Upsert(r) => r,
                _ => unreachable!(),
            });
            assert_eq!(c.get(id).unwrap(), want, "id {id}");
        }
        for filtered in [false, true] {
            let q = vector(rng);
            let mut want: Vec<(f32, u64)> = model
                .iter()
                .filter(|(id, _)| !filtered || (2..5).contains(&(*id % 10)))
                .map(|(&id, v)| (v.iter().zip(&q).map(|(a, b)| (a - b) * (a - b)).sum(), id))
                .collect();
            want.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            want.truncate(12);
            let options =
                SearchOptions { filter: filtered.then(|| Filter::range("n", 2..5)), ..SearchOptions::default() };
            let got = c.search(&SearchRequest { vector: q, k: 12, options }).unwrap();
            assert_eq!(got.results.iter().map(|r| (r.distance, r.id)).collect::<Vec<_>>(), want);
        }
    };
    for round in 0..300 {
        let mut want = Vec::new();
        let batch: Vec<Op> = (0..1 + rng.next_u64() % 12)
            .map(|_| {
                let id = rng.next_u64() % 300;
                if rng.next_u64() % 4 == 0 {
                    want.push(model.remove(&id).is_some());
                    Op::Delete { id }
                } else {
                    let v = vector(&mut rng);
                    want.push(model.insert(id, v.clone()).is_some());
                    upsert(id, v)
                }
            })
            .collect();
        assert_eq!(c.write(batch).unwrap(), want);
        if round % 20 == 0 {
            check(&c, &model, &mut rng);
        }
    }
    assert!(c.write(Vec::new()).unwrap().is_empty());
    let per_shard: Vec<usize> = c.stats().unwrap().iter().map(|s| s.live).collect();
    assert!(per_shard.iter().all(|&n| n * 5 > model.len()), "{per_shard:?}");
    c.wait_idle().unwrap();
    check(&c, &model, &mut rng);
    drop((c, db));
    let (db, _) = open(&dir);
    let c = db.collection("m").unwrap();
    check(&c, &model, &mut rng);
    c.flush().unwrap();
    c.optimize().unwrap();
    assert!(c.stats().unwrap().iter().all(|s| s.segments.len() == 1 && s.growing_rows == 0));
    check(&c, &model, &mut rng);
    drop((c, db));
    let _ = fs::remove_dir_all(&dir);
}

/// A shard as if it were in another process; one that does not answer.
struct Elsewhere(Arc<dyn Shard>, bool);

impl Shard for Elsewhere {
    fn write(&self, ops: &[Op]) -> Result<Vec<bool>> {
        self.0.write(ops)
    }
    fn search(&self, req: &SearchRequest) -> Result<SearchReply> {
        if self.1 { Err(Error::Unavailable("gone".into())) } else { self.0.search(req) }
    }
    fn get(&self, id: u64) -> Result<Option<Row>> {
        self.0.get(id)
    }
    fn stats(&self) -> Result<CollectionStats> {
        self.0.stats()
    }
    fn flush(&self) -> Result<()> {
        self.0.flush()
    }
    fn optimize(&self) -> Result<()> {
        self.0.optimize()
    }
    fn wait_idle(&self) -> Result<()> {
        self.0.wait_idle()
    }
}

/// Where shards are changes no results and no counts.
#[test]
fn shards_elsewhere_are_asked() {
    let dir = test_dir("db-elsewhere");
    for search_threads in [1, 4] {
        let (db, _) = Database::open(&dir, &DatabaseOptions { search_threads, ..opts() }).unwrap();
        let c = db.create_collection("c", config(60), 3).unwrap();
        let mut rng = SplitMix64::new(5);
        for ids in (0..12).map(|i| i * 50..(i + 1) * 50) {
            c.write(ids.map(|id| upsert(id, vector(&mut rng))).collect()).unwrap();
        }
        c.write((0..600).step_by(7).map(|id| Op::Delete { id }).collect()).unwrap();
        c.wait_idle().unwrap();
        let filter = Some(Filter::range("n", 2..5));
        let reqs: Vec<SearchRequest> = (0..20)
            .map(|i| {
                let filter = filter.clone().filter(|_| i % 2 == 0);
                let options = SearchOptions { filter, ..SearchOptions::default() };
                SearchRequest { vector: vector(&mut rng), k: 15, options }
            })
            .collect();
        let want: Vec<SearchReply> = reqs.iter().map(|r| c.search(r).unwrap()).collect();
        assert!(want.iter().all(|r| r.results.len() == 15 && r.stats.segments >= 6));
        let with = |away: &[usize], gone: bool| ShardedCollection {
            name: "c".into(),
            config: c.config.clone(),
            shards: (0..3)
                .map(|i| match away.contains(&i) {
                    true => Arc::new(Elsewhere(c.shards[i].clone(), gone)) as Arc<dyn Shard>,
                    false => c.shards[i].clone(),
                })
                .collect(),
            pool: c.pool.clone(),
            dir: PathBuf::new(),
            dropped: false.into(),
        };
        for away in [&[1][..], &[0, 2], &[0, 1, 2]] {
            let c = with(away, false);
            for (r, want) in reqs.iter().zip(&want) {
                assert_eq!(c.search(r).unwrap(), *want, "{away:?}");
            }
            let c = with(away, true);
            assert!(matches!(c.search(&reqs[0]), Err(Error::Unavailable(_))));
        }
        drop(c);
        db.drop_collection("c").unwrap();
    }
    let _ = fs::remove_dir_all(&dir);
}

/// One bad op and nothing is applied, on any shard.
#[test]
fn a_bad_op_changes_nothing_on_any_shard() {
    let dir = test_dir("db-bad");
    let (db, _) = open(&dir);
    for shards in [1, 4] {
        let c = db.create_collection(&format!("c{shards}"), config(50), shards).unwrap();
        let good = || (0..40).map(|id| upsert(id, vec![1.0; DIM])).collect::<Vec<_>>();
        let row = |vector: Vec<f32>, attrs: Attrs| Op::Upsert(Row { id: 40, vector, attrs });
        for bad in [
            row(vec![1.0; DIM + 1], Attrs::new()),
            row(vec![f32::NAN; DIM], Attrs::new()),
            row(vec![1.0; DIM], Attrs::new().int("other", 1)),
        ] {
            let mut ops = good();
            ops.push(bad);
            assert!(c.write(ops).is_err());
            assert_eq!(c.len().unwrap(), 0);
        }
        c.write(good()).unwrap();
        assert_eq!(c.len().unwrap(), 40);
        let req = SearchRequest { vector: vec![0.0; DIM + 1], k: 3, options: SearchOptions::default() };
        assert!(matches!(c.search(&req), Err(Error::DimensionMismatch { .. })));
    }
    drop(db);
    let _ = fs::remove_dir_all(&dir);
}

/// Rows are stored by it, so it must never change.
#[test]
fn rows_stay_in_their_shards() {
    let got: Vec<usize> = [0, 1, 2, 3, 1000, 123_456_789, u64::MAX].iter().map(|&id| shard_of(id, 8)).collect();
    assert_eq!(got, [7, 4, 4, 0, 1, 1, 7]);
    assert!((0..100).all(|id| shard_of(id, 1) == 0));
    // Even, for ids one after the other or far apart.
    for step in [1u64, 8, 1 << 32] {
        let mut n = [0usize; 7];
        (0..70_000).for_each(|i| n[shard_of(i * step, 7)] += 1);
        assert!(n.iter().all(|&n| (9500..10_500).contains(&n)), "{n:?}");
    }
}

/// A crash in the middle of creating or dropping a collection.
#[test]
fn interrupted_creates_and_drops() {
    let dir = test_dir("db-crash");
    let work = test_dir("db-crash-work");
    let (db, _) = open(&dir);
    db.create_collection("keep", config(50), 2).unwrap().write(vec![upsert(5, vec![1.0; DIM])]).unwrap();
    let before = fs::read(dir.join(CATALOG)).unwrap();
    db.create_collection("next", config(50), 2).unwrap().write(vec![upsert(6, vec![1.0; DIM])]).unwrap();
    let created = fs::read(dir.join(CATALOG)).unwrap();
    drop(db);

    // Creating: not renamed yet.
    copy_tree(&dir, &work);
    fs::write(work.join(CATALOG), &before).unwrap();
    fs::rename(work.join("col-00000002"), work.join("new-00000002")).unwrap();
    let (db, report) = open(&work);
    assert_eq!(report.removed, ["new-00000002"]);
    assert_eq!((db.names(), report.unknown), (vec!["keep".to_string()], vec![]));
    assert_eq!(ids(&db.collection("keep").unwrap()), [5]);
    drop(db);

    // Renamed, but not in the catalog: left alone, and its id not reused.
    copy_tree(&dir, &work);
    fs::write(work.join(CATALOG), &before).unwrap();
    let (db, report) = open(&work);
    assert_eq!(report.unknown, ["col-00000002"]);
    assert!(report.removed.is_empty());
    let c = db.create_collection("next", config(50), 1).unwrap();
    assert!(ids(&c).is_empty());
    drop((c, db));
    assert!(entries(&work).contains(&"col-00000002".to_string()) && entries(&work).contains(&"col-00000003".to_string()));
    let (db, _) = open(&work);
    assert_eq!(db.names(), ["keep", "next"]);
    drop(db);

    // Dropping: in the catalog, files still there.
    copy_tree(&dir, &work);
    let mut catalog = Catalog::decode(&created).unwrap();
    let i = catalog.find("next").unwrap();
    let e = catalog.entries.remove(i);
    catalog.dropped.push(e.id);
    catalog.install(&work).unwrap();
    let (db, report) = open(&work);
    assert_eq!((report.removed, report.unknown), (vec!["col-00000002".to_string()], vec![]));
    assert_eq!(db.names(), ["keep"]);
    assert_eq!(ids(&db.collection("keep").unwrap()), [5]);
    drop(db);
    assert!(Catalog::read(&work).unwrap().dropped.is_empty());

    // No catalog.
    copy_tree(&dir, &work);
    fs::remove_file(work.join(CATALOG)).unwrap();
    assert!(matches!(Database::open(&work, &opts()), Err(Error::Corrupt(_))));
    assert!(!work.join(CATALOG).exists());
    // A damaged one.
    let mut bytes = created.clone();
    bytes[20] ^= 1;
    fs::write(work.join(CATALOG), bytes).unwrap();
    assert!(matches!(Database::open(&work, &opts()), Err(Error::Corrupt(_))));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&work);
}

/// One collection that cannot be opened does not keep the others from use.
#[test]
fn a_damaged_collection_fails_alone() {
    let dir = test_dir("db-damaged");
    let (db, _) = open(&dir);
    for name in ["a", "b"] {
        db.create_collection(name, config(50), 2).unwrap().write(vec![upsert(1, vec![1.0; DIM])]).unwrap();
    }
    drop(db);
    let manifest = shard_dir(&collection_dir(&dir, 1), 1).join("MANIFEST");
    let good = fs::read(&manifest).unwrap();
    fs::write(&manifest, &good[..good.len() - 1]).unwrap();
    let (db, report) = open(&dir);
    assert!(report.failed.len() == 1 && report.failed[0].0 == "a", "{report:?}");
    assert!(matches!(db.collection("a"), Err(Error::Unavailable(m)) if m.contains("checksum")));
    assert_eq!(ids(&db.collection("b").unwrap()), [1]);
    assert!(matches!(db.create_collection("a", config(50), 1), Err(Error::AlreadyExists(_))));
    assert_eq!(db.names(), ["a", "b"]);
    drop(db);
    // Repaired, it is back.
    fs::write(&manifest, &good).unwrap();
    let (db, _) = open(&dir);
    assert_eq!(ids(&db.collection("a").unwrap()), [1]);
    drop(db);
    // Damaged, it can still be dropped.
    fs::write(&manifest, &good[..good.len() - 1]).unwrap();
    let (db, _) = open(&dir);
    db.drop_collection("a").unwrap();
    assert_eq!(db.names(), ["b"]);
    drop(db);
    assert_eq!(entries(&dir), ["CATALOG", "LOCK", "col-00000002"]);
    let _ = fs::remove_dir_all(&dir);
}

/// A dropped collection answers those who still hold it with an error, and
/// its files go with the last of them.
#[test]
fn dropping_a_collection_in_use() {
    let dir = test_dir("db-drop");
    let (db, _) = open(&dir);
    let c = db.create_collection("c", config(50), 2).unwrap();
    c.write((0..200).map(|id| upsert(id, vec![1.0; DIM])).collect()).unwrap();
    db.drop_collection("c").unwrap();
    assert!(matches!(db.collection("c"), Err(Error::NotFound(_))));
    let req = SearchRequest { vector: vec![0.0; DIM], k: 3, options: SearchOptions::default() };
    assert!(matches!(c.search(&req), Err(Error::NotFound(_))));
    assert!(matches!(c.write(vec![upsert(1, vec![1.0; DIM])]), Err(Error::NotFound(_))));
    assert!(matches!(c.get(1), Err(Error::NotFound(_))));
    assert!(matches!(c.flush(), Err(Error::NotFound(_))));
    assert!(entries(&dir).contains(&"col-00000001".to_string()));
    drop(c);
    assert_eq!(entries(&dir), ["CATALOG", "LOCK"]);
    // The catalog remembers drops only while their files may be there.
    for i in 0..5 {
        db.create_collection("c", config(50), 1).unwrap();
        db.drop_collection("c").unwrap();
        assert_eq!(Catalog::read(&dir).unwrap().dropped, [2 + i]);
    }
    let held = db.create_collection("c", config(50), 1).unwrap();
    db.drop_collection("c").unwrap();
    db.create_collection("c", config(50), 1).unwrap();
    db.drop_collection("c").unwrap();
    assert_eq!(Catalog::read(&dir).unwrap().dropped, [7, 8]);
    drop(held);
    // While others search it.
    let c = db.create_collection("c", config(50), 3).unwrap();
    c.write((0..200).map(|id| upsert(id, vec![1.0; DIM])).collect()).unwrap();
    std::thread::scope(|s| {
        for _ in 0..4 {
            let (c, req) = (c.clone(), req.clone());
            s.spawn(move || {
                loop {
                    match c.search(&req) {
                        Ok(r) => assert_eq!(r.results.len(), 3),
                        Err(e) => break assert!(matches!(e, Error::NotFound(_)), "{e}"),
                    }
                }
            });
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
        db.drop_collection("c").unwrap();
        drop(c);
    });
    assert_eq!(entries(&dir), ["CATALOG", "LOCK"]);
    drop(db);
    let (_, report) = open(&dir);
    assert!(report.removed.is_empty() && report.opened.is_empty());
    let _ = fs::remove_dir_all(&dir);
}

/// Closing a database right after searches on its pool leaves nothing
/// locked: it can be opened again at once.
#[test]
fn a_closed_database_can_be_opened_at_once() {
    let dir = test_dir("db-close");
    let (db, _) = open(&dir);
    let c = db.create_collection("c", config(50), 4).unwrap();
    c.write((0..400).map(|id| upsert(id, vec![(id % 7) as f32; DIM])).collect()).unwrap();
    drop((c, db));
    for _ in 0..30 {
        let (db, _) = open(&dir);
        let c = db.collection("c").unwrap();
        std::thread::scope(|s| {
            for _ in 0..3 {
                s.spawn(|| assert_eq!(ids(&c).len(), 400));
            }
        });
        drop((c, db));
    }
    let _ = fs::remove_dir_all(&dir);
}

/// Index builds of all collections share one limit.
#[test]
fn builds_share_a_limit() {
    use std::sync::atomic::AtomicUsize;
    static NOW: AtomicUsize = AtomicUsize::new(0);
    static MOST: AtomicUsize = AtomicUsize::new(0);
    // From the first line of a build to its last.
    fn watch(line: &str) {
        if line.starts_with("scan") {
            MOST.fetch_max(NOW.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(100));
        } else if line.starts_with("merge") {
            NOW.fetch_sub(1, Ordering::SeqCst);
        }
    }
    let dir = test_dir("db-builds");
    for (builds, most) in [(1, 1), (3, 3)] {
        MOST.store(0, Ordering::SeqCst);
        let mut o = opts();
        o.builds = builds;
        o.shard.progress = Some(watch);
        let (db, _) = Database::open(&dir, &o).unwrap();
        let mut cfg = config(1000);
        cfg.index_bytes = (2500 * DIM * 4) as u64;
        let c = db.create_collection("c", cfg, 3).unwrap();
        let mut rng = SplitMix64::new(1);
        for i in 0..100u64 {
            c.write((i * 100..(i + 1) * 100).map(|id| upsert(id, vector(&mut rng))).collect()).unwrap();
        }
        c.flush().unwrap();
        let t0 = Instant::now();
        c.wait_idle().unwrap();
        assert!(t0.elapsed().as_secs() < 60);
        assert!(c.stats().unwrap().iter().all(|s| s.builds >= 1));
        let seen = MOST.load(Ordering::SeqCst);
        assert!(seen <= most && (builds == 1 || seen > 1), "{seen} builds at once, {builds} allowed");
        drop(c);
        db.drop_collection("c").unwrap();
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn catalog_round_trips_and_detects_damage() {
    let e = |id, name: &str, shards| Entry { id, name: name.into(), shards };
    let mut c = Catalog { next_id: 9, dropped: vec![4, 2], ..Default::default() };
    for x in [e(7, "zz", 3), e(1, "a", 1), e(8, "m", MAX_SHARDS)] {
        c.insert(x);
    }
    assert_eq!(c.entries.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(), ["a", "m", "zz"]);
    let bytes = c.encode();
    assert_eq!(Catalog::decode(&bytes).unwrap(), c);
    for i in 0..bytes.len() {
        let mut b = bytes.clone();
        b[i] ^= 4;
        assert!(Catalog::decode(&b).is_err(), "byte {i}");
        assert!(Catalog::decode(&bytes[..i]).is_err(), "cut at {i}");
    }
    // Good checksums, bad contents.
    for bad in [
        Catalog { next_id: 8, ..c.clone() },
        Catalog { dropped: vec![1], ..c.clone() },
        Catalog { entries: vec![e(1, "b", 1), e(2, "a", 1)], dropped: vec![], next_id: 9 },
        Catalog { entries: vec![e(1, "a", 1), e(1, "b", 1)], dropped: vec![], next_id: 9 },
        Catalog { entries: vec![e(1, "../a", 1)], dropped: vec![], next_id: 9 },
        Catalog { entries: vec![e(1, "a", 0)], dropped: vec![], next_id: 9 },
    ] {
        assert!(matches!(Catalog::decode(&bad.encode()), Err(Error::Corrupt(_))), "{bad:?}");
    }
}
