use std::collections::HashMap;

use super::*;
use crate::disk::test_dir;
use crate::rng::SplitMix64;

/// Small integer coordinates: L2 and inner-product distances are exact in
/// f32, so results can be compared with a reference exactly, ties included.
fn int_vector(rng: &mut SplitMix64, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| (rng.next_u64() % 9) as f32 - 4.0).collect()
}

fn small_config(dim: usize, metric: Metric, seal_rows: usize) -> CollectionConfig {
    let mut c = CollectionConfig::new(dim, metric);
    c.seal_bytes = (seal_rows * dim * 4) as u64;
    c.index_bytes = u64::MAX; // no SSD indexes in these tests
    c
}

fn test_opts() -> OpenOptions {
    OpenOptions { durability: Durability::Buffered, direct_io: false, threads: 2, ..Default::default() }
}

/// The reference: live rows by id.
#[derive(Clone, Default)]
struct Model {
    rows: HashMap<u64, (Vec<f32>, Attrs)>,
}

impl Model {
    fn apply(&mut self, op: &WriteOp) {
        match *op {
            WriteOp::Upsert { id, vector, attrs } => {
                self.rows.insert(id, (vector.to_vec(), attrs.clone()));
            }
            WriteOp::Delete { id } => {
                self.rows.remove(&id);
            }
        }
    }

    fn search(&self, metric: Metric, q: &[f32], k: usize) -> Vec<(u64, f32)> {
        let mut all: Vec<(u64, f32)> = self
            .rows
            .iter()
            .map(|(&id, (v, _))| {
                let d = match metric {
                    Metric::L2 => v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum::<f32>(),
                    Metric::InnerProduct => -v.iter().zip(q).map(|(a, b)| a * b).sum::<f32>(),
                    Metric::Cosine => unreachable!(),
                };
                (id, d)
            })
            .collect();
        all.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        all.truncate(k);
        all
    }
}

fn check(c: &Collection, model: &Model, rng: &mut SplitMix64) {
    let cfg = c.config();
    assert_eq!(c.len(), model.rows.len());
    for _ in 0..4 {
        let q = int_vector(rng, cfg.dim);
        let k = 1 + (rng.next_u64() % 12) as usize;
        let got: Vec<(u64, f32)> =
            c.search(&q, k, &SearchOptions::default()).unwrap().iter().map(|r| (r.id, r.distance)).collect();
        assert_eq!(got, model.search(cfg.metric, &q, k));
    }
    // A few ids, present or not.
    let ids: Vec<u64> = model.rows.keys().copied().take(5).chain([u64::MAX, 7_777_777]).collect();
    for id in ids {
        match (c.get(id).unwrap(), model.rows.get(&id)) {
            (Some((v, a)), Some((want, want_a))) => {
                assert_eq!(&v, want);
                assert_eq!(&a, want_a);
            }
            (None, None) => {}
            (got, want) => panic!("id {id}: got {got:?}, want {want:?}"),
        }
        assert_eq!(c.contains(id), model.rows.contains_key(&id));
    }
}

/// Random upserts (new and existing ids), deletes (present and missing),
/// flushes and reopens, checked against the model after every step.
#[test]
fn random_writes_match_the_model() {
    for (metric, fields) in [(Metric::L2, false), (Metric::InnerProduct, true)] {
        let dir = test_dir(&format!("coll-random-{}", metric.name()));
        let dim = 6;
        let mut cfg = small_config(dim, metric, 25);
        if fields {
            cfg.fields = vec![Field::int("n"), Field::keyword("tag")];
        }
        let mut c = Collection::create(&dir, cfg, test_opts()).unwrap();
        let mut model = Model::default();
        let mut rng = SplitMix64::new(11);
        for step in 0..400 {
            let n_ops = 1 + (rng.next_u64() % 5) as usize;
            let vectors: Vec<Vec<f32>> = (0..n_ops).map(|_| int_vector(&mut rng, dim)).collect();
            let attrs: Vec<Attrs> = (0..n_ops)
                .map(|i| if fields { Attrs::new().int("n", (step * 10 + i) as i64).keyword("tag", "x") } else { Attrs::new() })
                .collect();
            let ops: Vec<WriteOp> = (0..n_ops)
                .map(|i| {
                    let id = rng.next_u64() % 120;
                    if rng.next_u64() % 4 == 0 {
                        WriteOp::Delete { id }
                    } else {
                        WriteOp::Upsert { id, vector: &vectors[i], attrs: &attrs[i] }
                    }
                })
                .collect();
            c.write(&ops).unwrap();
            ops.iter().for_each(|op| model.apply(op));
            match rng.next_u64() % 20 {
                0 => c.flush().unwrap(),
                1 => {
                    drop(c);
                    let (c2, report) = Collection::open(&dir, test_opts()).unwrap();
                    assert!(report.removed.is_empty(), "{:?}", report.removed);
                    assert_eq!(report.replay.cut_bytes, 0);
                    c = c2;
                }
                _ => {}
            }
            check(&c, &model, &mut rng);
        }
        let st = c.stats();
        assert!(st.segments.len() > 5, "{st:?}");
        assert!(st.segments.iter().any(|s| s.deleted > 0));
        // Each id has one live row at most.
        let mut seen = std::collections::HashSet::new();
        for s in &c.state.sealed {
            for p in 0..s.positions() {
                if !s.tomb.is_set(p) {
                    assert!(seen.insert(s.id_at(p)), "id {} live twice", s.id_at(p));
                }
            }
        }
        let g = &c.state.growing;
        for r in (0..g.rows()).filter(|&r| !g.is_dead(r)) {
            assert!(seen.insert(g.ids[r]));
        }
        drop(c);
        let _ = fs::remove_dir_all(&dir);
    }
}

fn copy_dir(from: &Path, to: &Path) {
    let _ = fs::remove_dir_all(to);
    fs::create_dir_all(to).unwrap();
    for e in fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        fs::copy(e.path(), to.join(e.file_name())).unwrap();
    }
}

/// Cutting the WAL at any byte loses exactly the batches from the cut on:
/// the reopened collection matches the model after the batches before it.
#[test]
fn wal_cut_at_every_byte_keeps_whole_batches() {
    let dir = test_dir("coll-walcut");
    let dim = 3;
    let mut c = Collection::create(&dir, small_config(dim, Metric::L2, 1000), test_opts()).unwrap();
    let mut rng = SplitMix64::new(5);
    let mut models = vec![Model::default()];
    let mut ends = vec![wal::HEADER_LEN as u64];
    for b in 0..6 {
        let vs: Vec<Vec<f32>> = (0..3).map(|_| int_vector(&mut rng, dim)).collect();
        let ops = vec![
            WriteOp::Upsert { id: b, vector: &vs[0], attrs: &NO_ATTRS },
            WriteOp::Upsert { id: 100 + b, vector: &vs[1], attrs: &NO_ATTRS },
            WriteOp::Delete { id: b.saturating_sub(1) },
            WriteOp::Upsert { id: b, vector: &vs[2], attrs: &NO_ATTRS },
        ];
        c.write(&ops).unwrap();
        let mut m = models.last().unwrap().clone();
        ops.iter().for_each(|op| m.apply(op));
        models.push(m);
        ends.push(c.wal.bytes());
    }
    let wal_path = c.wal.path().to_path_buf();
    drop(c);
    let full = fs::read(&wal_path).unwrap();
    assert_eq!(full.len() as u64, *ends.last().unwrap());
    let cut_dir = dir.with_extension("cut");
    for cut in wal::HEADER_LEN as u64..=full.len() as u64 {
        copy_dir(&dir, &cut_dir);
        let f = FsOpenOptions::new().write(true).open(cut_dir.join(wal_path.file_name().unwrap())).unwrap();
        f.set_len(cut).unwrap();
        drop(f);
        let (c, report) = Collection::open(&cut_dir, test_opts()).unwrap();
        let batches = ends.iter().filter(|&&e| e <= cut).count() - 1;
        assert_eq!(report.replay.batches as usize, batches, "cut at {cut}");
        assert_eq!(report.replay.cut_bytes, cut - ends[batches]);
        check(&c, &models[batches], &mut rng);
        // The cut log takes new writes after its valid end.
        drop(c);
    }
    // A flipped byte inside batch 3 ends the replay there.
    copy_dir(&dir, &cut_dir);
    let wp = cut_dir.join(wal_path.file_name().unwrap());
    let mut bytes = fs::read(&wp).unwrap();
    bytes[(ends[2] + 20) as usize] ^= 0x40;
    fs::write(&wp, &bytes).unwrap();
    let (mut c, report) = Collection::open(&cut_dir, test_opts()).unwrap();
    assert_eq!(report.replay.batches, 2);
    assert_eq!(report.replay.cut_bytes, full.len() as u64 - ends[2]);
    check(&c, &models[2], &mut rng);
    // Writing on after the cut, then reopening, keeps both.
    let v = int_vector(&mut rng, dim);
    c.upsert(999, &v, &NO_ATTRS).unwrap();
    drop(c);
    let (c, report) = Collection::open(&cut_dir, test_opts()).unwrap();
    assert_eq!((report.replay.batches, report.replay.cut_bytes), (3, 0));
    let mut m = models[2].clone();
    m.apply(&WriteOp::Upsert { id: 999, vector: &v, attrs: &NO_ATTRS });
    check(&c, &m, &mut rng);
    drop(c);
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&cut_dir);
}

/// Files an interrupted flush or build can leave behind are removed on
/// open; files that are not ours are left alone.
#[test]
fn open_removes_only_unreferenced_files_of_ours() {
    let dir = test_dir("coll-orphans");
    let dim = 4;
    let mut c = Collection::create(&dir, small_config(dim, Metric::L2, 10), test_opts()).unwrap();
    let mut model = Model::default();
    let mut rng = SplitMix64::new(9);
    for id in 0..25 {
        let v = int_vector(&mut rng, dim);
        c.upsert(id, &v, &NO_ATTRS).unwrap();
        model.apply(&WriteOp::Upsert { id, vector: &v, attrs: &NO_ATTRS });
    }
    c.delete(3).unwrap();
    model.rows.remove(&3);
    drop(c);
    let orphans = [
        "seg-00000099.flat",
        "seg-00000099.attrs",
        "seg-00000001.del-00000042",
        "wal-00000077.log",
        "seg-00000001.flat.tmp",
        "MANIFEST.tmp",
    ];
    for o in orphans {
        fs::write(dir.join(o), b"x").unwrap();
    }
    fs::create_dir(dir.join("seg-00000099.vdx.build-123")).unwrap();
    fs::write(dir.join("seg-00000099.vdx.build-123/part-0"), b"x").unwrap();
    let foreign = ["notes.txt", "seg-1.flat", "wal-00000001.log.bak", "seg-00000001.flatx", "MANIFEST.old"];
    for f in foreign {
        fs::write(dir.join(f), b"keep").unwrap();
    }
    let (c, report) = Collection::open(&dir, test_opts()).unwrap();
    let mut want: Vec<String> = orphans.iter().map(|s| s.to_string()).collect();
    want.push("seg-00000099.vdx.build-123".into());
    want.sort();
    assert_eq!(report.removed, want);
    for f in foreign {
        assert!(dir.join(f).exists(), "{f} removed");
    }
    check(&c, &model, &mut rng);
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_collection_opens_once() {
    let dir = test_dir("coll-lock");
    let c = Collection::create(&dir, small_config(4, Metric::L2, 10), test_opts()).unwrap();
    assert!(Collection::open(&dir, test_opts()).is_err());
    assert!(Collection::create(&dir, small_config(4, Metric::L2, 10), test_opts()).is_err());
    drop(c);
    let (c, _) = Collection::open(&dir, test_opts()).unwrap();
    drop(c);
    // A directory that holds a collection cannot be created over.
    assert!(Collection::create(&dir, small_config(4, Metric::L2, 10), test_opts()).is_err());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn bad_writes_change_nothing() {
    let dir = test_dir("coll-bad");
    let mut cfg = small_config(3, Metric::L2, 10);
    cfg.fields = vec![Field::int("year")];
    let mut c = Collection::create(&dir, cfg, test_opts()).unwrap();
    c.upsert(1, &[1.0, 2.0, 3.0], &Attrs::new().int("year", 2020)).unwrap();
    let wal_bytes = c.wal.bytes();
    let good = [1.0, 1.0, 1.0];
    let bad: Vec<Vec<WriteOp>> = vec![
        vec![WriteOp::Upsert { id: 2, vector: &[1.0, 2.0], attrs: &NO_ATTRS }],
        vec![WriteOp::Upsert { id: 2, vector: &[1.0, f32::NAN, 3.0], attrs: &NO_ATTRS }],
        vec![
            WriteOp::Upsert { id: 2, vector: &good, attrs: &NO_ATTRS },
            WriteOp::Upsert { id: 3, vector: &[f32::INFINITY, 0.0, 0.0], attrs: &NO_ATTRS },
        ],
    ];
    for ops in &bad {
        assert!(c.write(ops).is_err());
    }
    let unknown = Attrs::new().int("month", 3);
    let wrong_type = Attrs::new().keyword("year", "2020");
    assert!(c.upsert(2, &good, &unknown).is_err());
    assert!(c.upsert(2, &good, &wrong_type).is_err());
    assert_eq!(c.wal.bytes(), wal_bytes);
    assert_eq!(c.len(), 1);
    assert!(c.search(&[1.0, 2.0], 1, &SearchOptions::default()).is_err());
    assert!(c.search(&[1.0, 2.0, f32::NAN], 1, &SearchOptions::default()).is_err());
    assert!(c.search(&good, 0, &SearchOptions::default()).unwrap().is_empty());
    drop(c);

    let mut bad_cfg = small_config(3, Metric::L2, 10);
    bad_cfg.index.pq_m = 4;
    assert!(Collection::create(dir.join("x"), bad_cfg, test_opts()).is_err());
    let mut bad_cfg = small_config(3, Metric::L2, 10);
    bad_cfg.fields = vec![Field::int("a"), Field::keyword("a")];
    assert!(Collection::create(dir.join("y"), bad_cfg, test_opts()).is_err());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn manifest_round_trips_and_detects_damage() {
    let mut cfg = CollectionConfig::new(96, Metric::Cosine);
    cfg.fields = vec![Field::int("year"), Field::keyword("lang")];
    cfg.index.layout = Some(Layout::Separate);
    cfg.index.codec = Codec::F16;
    let m = Manifest {
        config: cfg,
        generation: 7,
        flushed_seq: 1234,
        wal_gen: 5,
        next_segment: 4,
        segments: vec![
            SegmentEntry { id: 1, kind: SegmentKind::Disk, positions: 1000, deleted: 10, del_gen: 6 },
            SegmentEntry { id: 3, kind: SegmentKind::Flat, positions: 50, deleted: 0, del_gen: 0 },
        ],
    };
    let bytes = m.encode();
    assert_eq!(Manifest::decode(&bytes).unwrap(), m);
    for i in 0..bytes.len() {
        let mut b = bytes.clone();
        b[i] ^= 0x10;
        assert!(Manifest::decode(&b).is_err(), "byte {i}");
    }
    assert!(Manifest::decode(&bytes[..bytes.len() - 1]).is_err());
}

#[test]
fn recognizes_its_own_file_names() {
    for n in [
        "wal-00000001.log",
        "seg-00000001.flat",
        "seg-12345678.vdx",
        "seg-00000001.attrs",
        "seg-00000001.del-00000003",
        "seg-00000001.vdx.build-77",
        "seg-00000001.vdx.tmp",
        "MANIFEST.tmp",
    ] {
        assert!(is_ours(n), "{n}");
    }
    for n in ["MANIFEST", "LOCK", "wal-1.log", "seg-0000001.flat", "seg-00000001.flat.bak", "seg-00000001.del-1", "x.tmp"] {
        assert!(!is_ours(n), "{n}");
    }
}

// ---- SSD segments: index builds, compaction, merges ----

fn float_vector(rng: &mut SplitMix64, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.uniform(-1.0, 1.0)).collect()
}

fn ssd_config(dim: usize, metric: Metric, seal_rows: usize, index_rows: usize) -> CollectionConfig {
    let mut c = small_config(dim, metric, seal_rows);
    c.index_bytes = (index_rows * dim * 4) as u64;
    c.index.pq_m = dim / 2;
    c.index.max_degree = 24;
    c.index.build_list = 48;
    c.max_indexed_segments = 2;
    c
}

/// Results are live rows with their true distances, and most of the true
/// top 10 (the SSD index is approximate).
fn check_approx(c: &Collection, model: &Model, rng: &mut SplitMix64, min_recall: f64) {
    let cfg = c.config();
    assert_eq!(c.len(), model.rows.len());
    let (mut hit, mut total) = (0, 0);
    for _ in 0..30 {
        let q = float_vector(rng, cfg.dim);
        let got = c.search(&q, 10, &SearchOptions { l: 60, ..Default::default() }).unwrap();
        let want = model.search(cfg.metric, &q, 10);
        for r in &got {
            let (v, _) = model.rows.get(&r.id).unwrap_or_else(|| panic!("id {} is not live", r.id));
            let d = model.search(cfg.metric, &q, model.rows.len()).iter().find(|x| x.0 == r.id).unwrap().1;
            assert!((r.distance - d).abs() <= 1e-4 * (1.0 + d.abs()), "id {}: {} vs {d} ({v:?})", r.id, r.distance);
        }
        hit += want.iter().filter(|w| got.iter().any(|g| g.id == w.0)).count();
        total += want.len();
    }
    let recall = hit as f64 / total as f64;
    assert!(recall >= min_recall, "recall {recall}");
    // Each id is live in one place at most.
    let mut seen = std::collections::HashSet::new();
    for s in &c.state.sealed {
        for p in (0..s.positions()).filter(|&p| !s.tomb.is_set(p)) {
            assert!(seen.insert(s.id_at(p)), "id {} live twice", s.id_at(p));
        }
    }
}

fn upsert_many(c: &mut Collection, model: &mut Model, rng: &mut SplitMix64, ids: impl Iterator<Item = u64>) {
    let ids: Vec<u64> = ids.collect();
    for chunk in ids.chunks(100) {
        let vs: Vec<Vec<f32>> = chunk.iter().map(|_| float_vector(rng, c.config().dim)).collect();
        let ops: Vec<WriteOp> =
            chunk.iter().zip(&vs).map(|(&id, v)| WriteOp::Upsert { id, vector: v, attrs: &NO_ATTRS }).collect();
        c.write(&ops).unwrap();
        ops.iter().for_each(|op| model.apply(op));
    }
}

#[test]
fn maintain_builds_compacts_and_merges() {
    let dir = test_dir("coll-maintain");
    let dim = 16;
    let mut c = Collection::create(&dir, ssd_config(dim, Metric::L2, 1000, 4000), test_opts()).unwrap();
    let mut model = Model::default();
    let mut rng = SplitMix64::new(21);
    let kinds = |c: &Collection| c.stats().segments.iter().map(|s| s.kind).collect::<Vec<_>>();

    // Five flat segments, then one SSD index from all of them.
    upsert_many(&mut c, &mut model, &mut rng, 0..5000);
    assert_eq!(kinds(&c), ["flat"; 5]);
    let jobs = c.maintain().unwrap();
    assert_eq!(jobs.len(), 1);
    assert_eq!((jobs[0].reason, jobs[0].rows, jobs[0].output.unwrap().1), ("index", 5000, "ssd"));
    assert_eq!(kinds(&c), ["ssd"]);
    check_approx(&c, &model, &mut rng, 0.95);

    // A few deletes on the SSD segment survive a reopen through the WAL,
    // then through its tombstone file.
    for id in [1u64, 2, 3, 4999] {
        assert!(c.delete(id).unwrap());
        model.rows.remove(&id);
    }
    drop(c);
    let (mut c, report) = Collection::open(&dir, test_opts()).unwrap();
    assert_eq!(report.replay.ops, 4);
    check_approx(&c, &model, &mut rng, 0.95);
    c.flush().unwrap();
    drop(c);
    let (mut c, report) = Collection::open(&dir, test_opts()).unwrap();
    assert_eq!(report.replay.ops, 0);
    assert_eq!(c.stats().segments[0].deleted, 4);
    check_approx(&c, &model, &mut rng, 0.95);

    // Deletes and replacements: 30% of the SSD segment's rows die, so it is
    // rebuilt; the replacements stay flat.
    for id in (0..5000).step_by(4) {
        assert!(c.delete(id).unwrap());
        model.rows.remove(&id);
    }
    upsert_many(&mut c, &mut model, &mut rng, (1..5000).step_by(20));
    check_approx(&c, &model, &mut rng, 0.9);
    let jobs = c.maintain().unwrap();
    assert_eq!(jobs.iter().map(|j| j.reason).collect::<Vec<_>>(), ["compact"]);
    assert_eq!(c.stats().segments.iter().map(|s| s.deleted).sum::<usize>(), 0);
    check_approx(&c, &model, &mut rng, 0.95);
    drop(c);
    let (mut c, report) = Collection::open(&dir, test_opts()).unwrap();
    assert!(report.removed.is_empty(), "{:?}", report.removed);
    check_approx(&c, &model, &mut rng, 0.95);

    // Two more SSD segments: one too many, so the two smallest merge.
    upsert_many(&mut c, &mut model, &mut rng, 10_000..14_000);
    c.maintain().unwrap();
    upsert_many(&mut c, &mut model, &mut rng, 20_000..24_000);
    let jobs = c.maintain().unwrap();
    assert_eq!(jobs.iter().map(|j| j.reason).collect::<Vec<_>>(), ["index", "merge"]);
    assert_eq!(kinds(&c).iter().filter(|k| **k == "ssd").count(), 2);
    check_approx(&c, &model, &mut rng, 0.95);

    // Everything deleted from a segment: it is dropped.
    let first: Vec<u64> = (0..c.state.sealed[0].positions())
        .filter(|&p| !c.state.sealed[0].tomb.is_set(p))
        .map(|p| c.state.sealed[0].id_at(p))
        .collect();
    let ops: Vec<WriteOp> = first.iter().map(|&id| WriteOp::Delete { id }).collect();
    c.write(&ops).unwrap();
    first.iter().for_each(|id| {
        model.rows.remove(id);
    });
    let n = c.stats().segments.len();
    let jobs = c.maintain().unwrap();
    assert_eq!(jobs[0].reason, "drop");
    assert_eq!(c.stats().segments.len(), n - 1);
    check_approx(&c, &model, &mut rng, 0.95);
    drop(c);
    let (c, report) = Collection::open(&dir, test_opts()).unwrap();
    assert!(report.removed.is_empty(), "{:?}", report.removed);
    check_approx(&c, &model, &mut rng, 0.95);
    // Only the files of the current segments are left.
    let mut names: Vec<String> = fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    names.retain(|n| n.starts_with("seg-"));
    assert_eq!(names.len(), c.stats().segments.len());
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// Rebuilding a segment from its own rows gives the same file, so rows do
/// not drift over repeated compactions: with f32 from the first rebuild
/// on, with f16 from the second (the first builds from f16-rounded rows).
#[test]
fn rebuilding_unchanged_rows_writes_the_same_file() {
    for (metric, codec, layout) in [
        (Metric::Cosine, Codec::F32, Layout::Inline),
        (Metric::Cosine, Codec::F16, Layout::Inline),
        (Metric::L2, Codec::BF16, Layout::Inline),
        (Metric::InnerProduct, Codec::F32, Layout::Separate),
    ] {
        let dir = test_dir(&format!("coll-same-{}-{}-{}", metric.name(), codec.name(), layout.name()));
        let mut cfg = ssd_config(24, metric, 800, 1 << 20);
        cfg.index.codec = codec;
        cfg.index.layout = Some(layout);
        let mut c = Collection::create(&dir, cfg, test_opts()).unwrap();
        let mut model = Model::default();
        let mut rng = SplitMix64::new(3);
        upsert_many(&mut c, &mut model, &mut rng, 0..3000);
        let mut files = Vec::new();
        for _ in 0..3 {
            let job = c.optimize().unwrap().unwrap();
            let id = job.output.unwrap().0;
            files.push(fs::read(dir.join(disk_name(id))).unwrap());
        }
        if codec == Codec::F32 {
            assert!(files[0] == files[1], "{codec:?}");
        }
        assert!(files[1] == files[2], "{codec:?}");
        // Rows read back as written (f32), and cosine rows at unit length.
        let (v, _) = c.get(17).unwrap().unwrap();
        if codec == Codec::F32 {
            let mut want = model.rows[&17].0.clone();
            if metric == Metric::Cosine {
                crate::distance::normalize(&mut want);
            }
            assert_eq!(v, want);
        }
        if metric == Metric::Cosine {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-2, "{norm}");
        }
        drop(c);
        let _ = fs::remove_dir_all(&dir);
    }
}

/// Identical vectors share one SSD graph node but keep their own ids,
/// attributes and deletes.
#[test]
fn identical_vectors_keep_their_own_rows() {
    let dir = test_dir("coll-twins");
    let dim = 8;
    let mut cfg = ssd_config(dim, Metric::L2, 5000, 1 << 20);
    cfg.fields = vec![Field::keyword("name"), Field::int("n")];
    let mut c = Collection::create(&dir, cfg, test_opts()).unwrap();
    let mut rng = SplitMix64::new(8);
    let base: Vec<Vec<f32>> = (0..2500).map(|_| float_vector(&mut rng, dim)).collect();
    // Ids 2i and 2i+1 hold the same vector with different attributes.
    let attrs: Vec<Attrs> =
        (0..5000).map(|i| Attrs::new().keyword("name", &format!("row{i}")).int("n", i as i64)).collect();
    let ops: Vec<WriteOp> = (0..5000)
        .map(|i| WriteOp::Upsert { id: i as u64, vector: &base[i / 2], attrs: &attrs[i] })
        .collect();
    for chunk in ops.chunks(500) {
        c.write(chunk).unwrap();
    }
    c.optimize().unwrap();
    let Body::Disk { index, .. } = &c.state.sealed[0].body else { panic!("not an SSD segment") };
    assert_eq!((index.len(), index.nodes()), (5000, 2500));
    for i in (0..5000u64).step_by(3) {
        assert!(c.delete(i).unwrap());
    }
    for round in 0..2 {
        for i in 0..5000u64 {
            let got = c.get(i).unwrap();
            if i % 3 == 0 {
                assert!(got.is_none());
                continue;
            }
            let (v, a) = got.unwrap();
            assert_eq!(v, base[i as usize / 2]);
            assert_eq!(a, attrs[i as usize], "round {round}");
        }
        for i in (0..2500).step_by(97) {
            let res = c.search(&base[i], 2, &SearchOptions::default()).unwrap();
            let want: Vec<u64> = [2 * i as u64, 2 * i as u64 + 1].into_iter().filter(|id| id % 3 != 0).collect();
            let got: Vec<u64> = res.iter().filter(|r| r.distance == 0.0).map(|r| r.id).collect();
            assert_eq!(got, want);
        }
        // The second round reads the rebuilt segment.
        c.optimize().unwrap();
    }
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// One thread's search context serves SSD indexes of different dimensions
/// and layouts in turn.
#[test]
fn searches_alternate_between_indexes() {
    let mut cols = Vec::new();
    for (i, (dim, layout)) in [(16, Layout::Inline), (40, Layout::Separate)].into_iter().enumerate() {
        let dir = test_dir(&format!("coll-alt-{i}"));
        let mut cfg = ssd_config(dim, Metric::L2, 4000, 1 << 20);
        cfg.index.layout = Some(layout);
        let mut c = Collection::create(&dir, cfg, test_opts()).unwrap();
        let mut rng = SplitMix64::new(i as u64);
        upsert_many(&mut c, &mut Model::default(), &mut rng, 0..3000);
        c.optimize().unwrap();
        cols.push((dir, c));
    }
    let mut rng = SplitMix64::new(99);
    let queries: Vec<Vec<Vec<f32>>> =
        cols.iter().map(|(_, c)| (0..20).map(|_| float_vector(&mut rng, c.config().dim)).collect()).collect();
    let alone: Vec<Vec<_>> = cols
        .iter()
        .zip(&queries)
        .map(|((_, c), qs)| qs.iter().map(|q| c.search(q, 10, &SearchOptions::default()).unwrap()).collect())
        .collect();
    for j in 0..20 {
        for (i, (_, c)) in cols.iter().enumerate() {
            assert_eq!(c.search(&queries[i][j], 10, &SearchOptions::default()).unwrap(), alone[i][j]);
        }
    }
    for (dir, c) in cols {
        drop(c);
        let _ = fs::remove_dir_all(&dir);
    }
}

fn names(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    v.retain(|n| n != LOCK);
    v.sort();
    v
}

/// The two directory states a crash in a flush or a rebuild can leave:
/// the new files written but the manifest not yet replaced, and the
/// manifest replaced but the old files not yet removed. Both open to the
/// same rows, and lose nothing.
#[test]
fn crashes_around_the_manifest_commit_lose_nothing() {
    let dir = test_dir("coll-crash");
    let dim = 16;
    let mut c = Collection::create(&dir, ssd_config(dim, Metric::L2, 100_000, 3000), test_opts()).unwrap();
    let mut model = Model::default();
    let mut rng = SplitMix64::new(4);
    upsert_many(&mut c, &mut model, &mut rng, 0..2500);
    c.flush().unwrap();
    upsert_many(&mut c, &mut model, &mut rng, 2000..3500);
    for id in (0..3500).step_by(7) {
        c.delete(id).unwrap();
        model.rows.remove(&id);
    }
    // Two steps: a flush (new flat segment, tombstones, WAL) and an index
    // build (two flat segments into one SSD segment).
    let steps: [fn(&mut Collection); 2] = [|c| c.flush().unwrap(), |c| assert_eq!(c.maintain().unwrap().len(), 1)];
    let (before, after, crash) = (dir.with_extension("before"), dir.with_extension("after"), dir.with_extension("x"));
    for step in steps {
        drop(c);
        copy_dir(&dir, &before);
        let (mut c2, _) = Collection::open(&dir, test_opts()).unwrap();
        step(&mut c2);
        drop(c2);
        copy_dir(&dir, &after);
        let (old, new) = (names(&before), names(&after));
        let added: Vec<&String> = new.iter().filter(|n| !old.contains(n) && *n != MANIFEST).collect();
        let gone: Vec<&String> = old.iter().filter(|n| !new.contains(n)).collect();
        assert!(!added.is_empty() && !gone.is_empty());

        // Crash before the manifest rename: the old state plus the new files.
        copy_dir(&before, &crash);
        added.iter().for_each(|n| {
            fs::copy(after.join(n), crash.join(n)).unwrap();
        });
        let (cc, report) = Collection::open(&crash, test_opts()).unwrap();
        assert_eq!(report.removed, added.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        check_approx(&cc, &model, &mut rng, 0.95);
        drop(cc);

        // Crash after it: the new state plus the old files.
        copy_dir(&after, &crash);
        gone.iter().for_each(|n| {
            fs::copy(before.join(n), crash.join(n)).unwrap();
        });
        let (cc, report) = Collection::open(&crash, test_opts()).unwrap();
        assert_eq!(report.removed, gone.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        check_approx(&cc, &model, &mut rng, 0.95);
        drop(cc);
        c = Collection::open(&dir, test_opts()).unwrap().0;
    }
    assert_eq!(c.stats().segments.iter().map(|s| s.kind).collect::<Vec<_>>(), ["ssd"]);
    drop(c);
    for d in [&dir, &before, &after, &crash] {
        let _ = fs::remove_dir_all(d);
    }
}

/// A log written with Sync whose middle record is damaged held acknowledged
/// writes after the damage: opening fails, and says where to cut the log.
#[test]
fn damage_before_synced_writes_refuses_to_open() {
    let dir = test_dir("coll-syncdamage");
    let dim = 3;
    let opts = OpenOptions { durability: Durability::Sync, ..test_opts() };
    let mut c = Collection::create(&dir, small_config(dim, Metric::L2, 1000), opts.clone()).unwrap();
    let mut rng = SplitMix64::new(6);
    let mut models = vec![Model::default()];
    let mut ends = vec![wal::HEADER_LEN as u64];
    for id in 0..5 {
        let v = int_vector(&mut rng, dim);
        c.upsert(id, &v, &NO_ATTRS).unwrap();
        let mut m = models.last().unwrap().clone();
        m.apply(&WriteOp::Upsert { id, vector: &v, attrs: &NO_ATTRS });
        models.push(m);
        ends.push(c.wal.bytes());
    }
    let wal_path = c.wal.path().to_path_buf();
    drop(c);
    let mut bytes = fs::read(&wal_path).unwrap();
    bytes[ends[2] as usize + 30] ^= 1;
    fs::write(&wal_path, &bytes).unwrap();
    let err = Collection::open(&dir, opts.clone()).err().expect("opened over damaged acknowledged writes");
    assert!(matches!(&err, Error::Corrupt(m) if m.contains(&format!("byte {}", ends[2]))), "{err}");
    // The log is untouched; cutting it where the error says recovers the
    // writes before the damage.
    assert_eq!(fs::read(&wal_path).unwrap(), bytes);
    FsOpenOptions::new().write(true).open(&wal_path).unwrap().set_len(ends[2]).unwrap();
    let (c, report) = Collection::open(&dir, opts).unwrap();
    assert_eq!(report.replay.batches, 2);
    check(&c, &models[2], &mut rng);
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// Flat segments too small for an SSD index do not make maintain loop, and
/// a workload of deletes alone still rotates its WAL.
#[test]
fn small_flat_segments_and_delete_only_writes() {
    let dir = test_dir("coll-small");
    let dim = 4;
    let mut cfg = small_config(dim, Metric::L2, 50);
    cfg.index_bytes = (10 * dim * 4) as u64;
    let mut c = Collection::create(&dir, cfg, test_opts()).unwrap();
    let mut model = Model::default();
    let mut rng = SplitMix64::new(2);
    for id in 0..400 {
        let v = int_vector(&mut rng, dim);
        c.upsert(id, &v, &NO_ATTRS).unwrap();
        model.apply(&WriteOp::Upsert { id, vector: &v, attrs: &NO_ATTRS });
    }
    let segs = c.stats().segments.len();
    assert!(segs >= 7);
    assert!(c.maintain().unwrap().is_empty());
    assert_eq!(c.stats().segments.len(), segs);
    // Deletes of absent ids grow only the WAL: it is rotated at twice seal_bytes.
    let seal = c.config().seal_bytes;
    for id in 10_000..20_000u64 {
        c.delete(id).unwrap();
        assert!(c.wal.bytes() < 2 * seal + 64, "WAL {} bytes", c.wal.bytes());
    }
    check(&c, &model, &mut rng);
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn create_refuses_collection_files_without_a_manifest() {
    let dir = test_dir("coll-nomanifest");
    fs::write(dir.join("seg-00000001.flat"), b"x").unwrap();
    assert!(Collection::create(&dir, small_config(4, Metric::L2, 10), test_opts()).is_err());
    assert!(dir.join("seg-00000001.flat").exists());
    fs::remove_file(dir.join("seg-00000001.flat")).unwrap();
    // What an interrupted create leaves is cleaned up.
    fs::write(dir.join("wal-00000001.log"), b"x").unwrap();
    fs::write(dir.join("MANIFEST.tmp"), b"x").unwrap();
    let c = Collection::create(&dir, small_config(4, Metric::L2, 10), test_opts()).unwrap();
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// Buffered records between the damage and a Sync record do not hide it.
#[test]
fn damage_is_found_past_buffered_records() {
    let dir = test_dir("coll-mixeddamage");
    let dim = 3;
    let mut c = Collection::create(&dir, small_config(dim, Metric::L2, 1000), test_opts()).unwrap();
    let mut rng = SplitMix64::new(7);
    let mut ends = vec![wal::HEADER_LEN as u64];
    for id in 0..3 {
        c.upsert(id, &int_vector(&mut rng, dim), &NO_ATTRS).unwrap();
        ends.push(c.wal.bytes());
    }
    drop(c);
    let sync = OpenOptions { durability: Durability::Sync, ..test_opts() };
    let (mut c, _) = Collection::open(&dir, sync.clone()).unwrap();
    c.upsert(3, &int_vector(&mut rng, dim), &NO_ATTRS).unwrap();
    let wal_path = c.wal.path().to_path_buf();
    drop(c);
    let mut bytes = fs::read(&wal_path).unwrap();
    bytes[ends[0] as usize + 30] ^= 1;
    fs::write(&wal_path, &bytes).unwrap();
    for opts in [test_opts(), sync] {
        assert!(matches!(Collection::open(&dir, opts), Err(Error::Corrupt(_))));
    }
    let _ = fs::remove_dir_all(&dir);
}

mod filters;
