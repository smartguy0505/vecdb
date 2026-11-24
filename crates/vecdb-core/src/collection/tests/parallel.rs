//! Searches on several threads.

use super::*;

/// Every segment kind, with deletes and filters, gives the same results and
/// counts with a pool as without.
#[test]
fn a_pool_changes_no_results() {
    let dir = test_dir("coll-parallel");
    let dim = 16;
    let mut cfg = ssd_config(dim, Metric::L2, 100_000, 2500);
    cfg.fields = vec![Field::int("g")];
    cfg.max_indexed_segments = 10;
    let c = Collection::create(&dir, cfg, test_opts()).unwrap();
    let mut rng = SplitMix64::new(31);
    let put = |ids: std::ops::Range<u64>, rng: &mut SplitMix64| {
        for id in ids {
            let v = float_vector(rng, dim);
            let attrs = Attrs::new().int("g", (id % 40) as i64);
            let op = WriteOp::Upsert { id, vector: &v, attrs: &attrs };
            c.enqueue(c.prepare(&[op]).unwrap()).unwrap();
        }
    };
    // Two SSD segments, two flat ones, two frozen and the growing one.
    for base in [0, 3000] {
        put(base..base + 3000, &mut rng);
        assert_eq!(c.maintain().unwrap().len(), 0);
        c.flush().unwrap();
        assert_eq!(c.maintain().unwrap().len(), 1);
    }
    for base in [6000, 6800] {
        put(base..base + 800, &mut rng);
        c.flush().unwrap();
    }
    for base in [7600, 8000] {
        put(base..base + 400, &mut rng);
        c.freeze(&mut c.writer.lock().unwrap()).unwrap();
    }
    put(8400..8900, &mut rng);
    // Deletes everywhere.
    for id in (0..8900).step_by(7) {
        let op = WriteOp::Delete { id };
        c.enqueue(c.prepare(&[op]).unwrap()).unwrap();
    }
    let kinds: Vec<&str> = c.stats().segments.iter().map(|s| s.kind).collect();
    assert_eq!((kinds, c.stats().frozen), (vec!["ssd", "ssd", "flat", "flat"], 2));
    let mut searches = vec![SearchOptions { l: 60, ..Default::default() }];
    for filter in [Filter::int_eq("g", 3), Filter::range("g", 5..30), Filter::int_eq("g", 99)] {
        for filter_mode in [FilterMode::Auto, FilterMode::Graph, FilterMode::Scan] {
            searches.push(SearchOptions { l: 60, filter: Some(filter.clone()), filter_mode, ..Default::default() });
        }
    }
    let queries: Vec<Vec<f32>> = (0..20).map(|_| float_vector(&mut rng, dim)).collect();
    let all = |c: &Collection| {
        let one = |o: &SearchOptions| queries.iter().map(|q| c.search_with_stats(q, 25, o).unwrap()).collect::<Vec<_>>();
        searches.iter().map(one).collect::<Vec<_>>()
    };
    let alone = all(&c);
    assert!(alone[0].iter().all(|(r, st)| r.len() == 25 && st.segments == 7 && st.reads > 0 && st.scanned > 2000));
    assert!(alone[4].iter().all(|(r, st)| r.len() == 25 && st.matched > 4000));
    assert!(alone[9].iter().all(|(r, _)| r.is_empty()));
    drop(c);
    // (A reopen keeps what is frozen so.)
    for threads in [1, 3, 8] {
        let pool = SearchPool::new(threads, 4).unwrap();
        let (c, _) = Collection::open(&dir, OpenOptions { pool: Some(pool), ..test_opts() }).unwrap();
        assert_eq!(c.stats().frozen, 2);
        for round in 0..5 {
            assert_eq!(all(&c), alone, "{threads} threads, round {round}");
        }
        // From several threads at once.
        std::thread::scope(|s| {
            for _ in 0..3 {
                s.spawn(|| assert_eq!(all(&c), alone));
            }
        });
        // An error in one task is the search's.
        assert!(c.search(&queries[0][..8], 5, &searches[0]).is_err());
    }
    // Direct I/O: the same results. The walks count as waiting, not as
    // computing, and come first.
    let pool = SearchPool::new(8, 1).unwrap();
    let (c, _) = Collection::open(&dir, OpenOptions { pool: Some(pool), direct_io: true, ..test_opts() }).unwrap();
    for (q, want) in queries.iter().zip(&alone[0]) {
        assert_eq!(c.search_with_stats(q, 25, &searches[0]).unwrap(), *want);
    }
    let direct = c.sealed().iter().filter(|s| matches!(&s.body, Body::Disk { index, .. } if index.is_direct())).count();
    let snap = c.snapshot();
    let plan = search::Plan::new(&c.config, &[&snap], vec![], queries[0].clone(), queries[0].clone(), 25, &searches[0], None);
    assert_eq!(plan.waits, direct);
    drop(snap);
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// A task that fails fails the search, and the pool goes on.
#[test]
fn a_failed_task_fails_the_search() {
    let dir = test_dir("coll-parallel-fail");
    let dim = 16;
    let pool = SearchPool::new(3, 4).unwrap();
    let opts = OpenOptions { pool: Some(pool), ..test_opts() };
    let mut c = Collection::create(&dir, ssd_config(dim, Metric::L2, 5000, 2500), opts).unwrap();
    let mut rng = SplitMix64::new(3);
    let mut model = Model::default();
    upsert_many(&mut c, &mut model, &mut rng, 0..3000);
    c.flush().unwrap();
    c.maintain().unwrap();
    upsert_many(&mut c, &mut model, &mut rng, 3000..3500);
    let q = float_vector(&mut rng, dim);
    let opts = SearchOptions { l: 60, ..Default::default() };
    c.search(&q, 10, &opts).unwrap();
    // The index file cut short under the open collection.
    let index = dir.join(disk_name(c.sealed()[0].id));
    let bytes = fs::read(&index).unwrap();
    FsOpenOptions::new().write(true).open(&index).unwrap().set_len(8192).unwrap();
    for _ in 0..20 {
        assert!(c.search(&q, 10, &opts).is_err());
    }
    fs::write(&index, bytes).unwrap();
    for _ in 0..20 {
        assert_eq!(c.search(&q, 10, &opts).unwrap().len(), 10);
    }
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}
