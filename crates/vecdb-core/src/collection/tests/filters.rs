//! Attribute filters: the indexed evaluation against `Filter::matches`, and
//! filtered search against the model on every kind of segment.

use std::ops::Bound;

use super::*;
use crate::collection::columns::{AttrWriter, ColumnBuilder, SegmentAttrs, encode_row};
use crate::collection::filter::{ColumnSource, Compiled, MAX_FILTER_DEPTH};

fn fields() -> Vec<Field> {
    vec![Field::int("a"), Field::int("b"), Field::keyword("t"), Field::keyword("u")]
}

const EDGE_INTS: [i64; 6] = [i64::MIN, -1, 0, 1, 7, i64::MAX];
const KEYS: [&str; 5] = ["", "x", "y", "zz", "é"];

fn random_attrs(rng: &mut SplitMix64) -> Attrs {
    let mut a = Attrs::new();
    for _ in 0..rng.next_u64() % 3 {
        a = a.int("a", (rng.next_u64() % 10) as i64 - 5);
    }
    for _ in 0..rng.next_u64() % 3 {
        a = a.int("b", EDGE_INTS[(rng.next_u64() % 6) as usize]);
    }
    for _ in 0..rng.next_u64() % 4 {
        a = a.keyword("t", KEYS[(rng.next_u64() % 5) as usize]);
    }
    if rng.next_u64() % 2 == 0 {
        a = a.keyword("u", &format!("u{}", rng.next_u64() % 20));
    }
    a
}

fn random_int(rng: &mut SplitMix64, field: &str) -> i64 {
    if field == "a" { (rng.next_u64() % 12) as i64 - 6 } else { EDGE_INTS[(rng.next_u64() % 6) as usize] }
}

fn random_key(rng: &mut SplitMix64, field: &str) -> String {
    if field == "t" { KEYS[(rng.next_u64() % 5) as usize].into() } else { format!("u{}", rng.next_u64() % 22) }
}

fn random_filter(rng: &mut SplitMix64, depth: usize) -> Filter {
    if depth == 0 || rng.next_u64() % 3 == 0 {
        let int = rng.next_u64() % 2 == 0;
        let field = if int { ["a", "b"][(rng.next_u64() % 2) as usize] } else { ["t", "u"][(rng.next_u64() % 2) as usize] };
        let value = |rng: &mut SplitMix64| {
            if int { Value::Int(random_int(rng, field)) } else { Value::Keyword(random_key(rng, field)) }
        };
        return match rng.next_u64() % if int { 3 } else { 2 } {
            0 => Filter::Eq(field.into(), value(rng)),
            1 => Filter::In(field.into(), (0..rng.next_u64() % 4).map(|_| value(rng)).collect()),
            _ => {
                let (x, y) = (random_int(rng, field), random_int(rng, field));
                Filter::Range { field: field.into(), min: x.min(y), max: if rng.next_u64() % 5 == 0 { x.min(y) - 1 } else { x.max(y) } }
            }
        };
    }
    let kids = |rng: &mut SplitMix64| (0..rng.next_u64() % 4).map(|_| random_filter(rng, depth - 1)).collect::<Vec<_>>();
    match rng.next_u64() % 3 {
        0 => Filter::And(kids(rng)),
        1 => Filter::Or(kids(rng)),
        _ => Filter::not(random_filter(rng, depth - 1)),
    }
}

#[test]
fn filter_semantics_and_checks() {
    let a = Attrs::new().keyword("t", "x").keyword("t", "y").int("a", 5).int("a", 50);
    let none = Attrs::new();
    assert!(Filter::keyword_eq("t", "y").matches(&a));
    assert!(!Filter::keyword_eq("t", "z").matches(&a));
    assert!(Filter::keyword_in("t", &["q", "x"]).matches(&a));
    assert!(Filter::range("a", 10..=60).matches(&a));
    assert!(!Filter::range("a", 6..50).matches(&a));
    // Not: rows without any value match, and rows with one matching value don't.
    assert!(Filter::not(Filter::keyword_eq("t", "x")).matches(&none));
    assert!(!Filter::not(Filter::keyword_eq("t", "x")).matches(&a));
    assert!(Filter::not(Filter::int_eq("a", 5)).matches(&Attrs::new().int("a", 6)));
    assert!(!Filter::In("t".into(), vec![]).matches(&a));
    assert!(Filter::and([]).matches(&none));
    assert!(!Filter::or([]).matches(&a));
    assert!(!Filter::Range { field: "a".into(), min: 60, max: 10 }.matches(&a));

    let r = |min, max| Filter::Range { field: "a".into(), min, max };
    assert_eq!(Filter::range("a", ..), r(i64::MIN, i64::MAX));
    assert_eq!(Filter::range("a", 3..7), r(3, 6));
    assert_eq!(Filter::range("a", (Bound::Excluded(3), Bound::Included(7))), r(4, 7));
    for f in [Filter::range("a", ..i64::MIN), Filter::range("a", (Bound::Excluded(i64::MAX), Bound::Unbounded))] {
        assert!(matches!(f, Filter::Range { min, max, .. } if min > max), "{f:?}");
    }

    let fields = fields();
    for bad in [
        Filter::keyword_eq("nope", "x"),
        Filter::int_eq("t", 1),
        Filter::keyword_eq("a", "1"),
        Filter::range("t", 0..1),
        Filter::In("a".into(), vec![Value::Int(1), Value::Keyword("1".into())]),
        Filter::and([Filter::int_eq("a", 1), Filter::not(Filter::keyword_eq("zz", "x"))]),
    ] {
        assert!(matches!(bad.compile(&fields), Err(Error::InvalidArgument(_))), "{bad:?}");
    }
    let nest = |n: usize| (0..n).fold(Filter::int_eq("a", 1), |f, _| Filter::not(f));
    assert!(nest(MAX_FILTER_DEPTH - 1).compile(&fields).is_ok());
    assert!(nest(MAX_FILTER_DEPTH).compile(&fields).is_err());
    assert!(Filter::int_in("a", &vec![1; 1 << 16]).compile(&fields).is_err());

    // Constant parts fold away.
    assert_eq!(Filter::and([]).compile(&fields).unwrap(), Compiled::All);
    assert_eq!(Filter::or([Filter::In("a".into(), vec![])]).compile(&fields).unwrap(), Compiled::Nothing);
    assert_eq!(Filter::not(Filter::or([])).compile(&fields).unwrap(), Compiled::All);
    assert_eq!(
        Filter::and([Filter::and([]), Filter::int_eq("a", 2)]).compile(&fields).unwrap(),
        Compiled::Ints { field: 0, ranges: vec![(2, 2)] }
    );
    // Even a collection without fields takes filters without leaves.
    assert_eq!(Filter::and([]).compile(&[]).unwrap(), Compiled::All);
}

/// Random filter trees over random multi-valued rows: the growing
/// segment's columns, the columns written to a file and the columns read
/// back all match `Filter::matches` row for row.
#[test]
fn indexed_filters_match_the_reference() {
    // Multi-valued rows, then rows with exactly one value per field (where
    // wide filters clear the bitmap instead of filling it).
    for single in [false, true] {
        let mut rng = SplitMix64::new(5);
        let rows: Vec<Attrs> = (0..700)
            .map(|_| {
                if single {
                    Attrs::new()
                        .int("a", random_int(&mut rng, "a"))
                        .int("b", random_int(&mut rng, "b"))
                        .keyword("t", &random_key(&mut rng, "t"))
                        .keyword("u", &random_key(&mut rng, "u"))
                } else {
                    random_attrs(&mut rng)
                }
            })
            .collect();
        check_columns(&rows, &mut rng, single);
    }
}

#[test]
fn ranges_merge() {
    use crate::collection::filter::merge_ranges;
    assert_eq!(merge_ranges(&[(5, 9), (1, 2), (3, 4), (8, 12), (20, 19)]), vec![(1, 12)]);
    assert_eq!(merge_ranges(&[(0, 1), (3, 4)]), vec![(0, 1), (3, 4)]);
    assert_eq!(merge_ranges(&[(i64::MIN, -1), (0, i64::MAX)]), vec![(i64::MIN, i64::MAX)]);
    assert_eq!(merge_ranges(&[(i64::MAX, i64::MAX), (5, i64::MAX), (1, 1)]), vec![(1, 1), (5, i64::MAX)]);
    assert!(merge_ranges(&[(3, 2)]).is_empty());
}

fn check_columns(rows: &[Attrs], rng: &mut SplitMix64, single: bool) {
    let fields = fields();
    let n = rows.len();
    let encoded: Vec<Vec<u8>> = rows
        .iter()
        .map(|a| {
            a.validate(&fields).unwrap();
            let mut e = Vec::new();
            encode_row(a, &fields, &mut e);
            e
        })
        .collect();
    let mut growing = ColumnBuilder::new(&fields);
    for (i, e) in encoded.iter().enumerate() {
        growing.add(i, e, &fields).unwrap();
    }
    // The file takes rows in any order.
    let dir = test_dir(&format!("coll-columns-{single}"));
    let path = dir.join("seg-00000001.attrs");
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        order.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
    }
    let mut w = AttrWriter::create(&path, &fields, n).unwrap();
    for &i in &order {
        w.push(i, &encoded[i]).unwrap();
    }
    assert!(w.push(order[0], &encoded[order[0]]).is_err());
    let written = w.finish().unwrap();
    let opened = SegmentAttrs::open(&path, &fields, n).unwrap();
    for (i, a) in rows.iter().enumerate() {
        assert_eq!(&opened.row(i, &fields).unwrap(), a);
    }
    let mut visited = Vec::new();
    opened
        .for_each_row(|p, row| {
            assert_eq!(row, &encoded[p][..]);
            visited.push(p);
            Ok(())
        })
        .unwrap();
    assert_eq!(visited, order);

    let mut matched = 0;
    for _ in 0..400 {
        let f = random_filter(rng, 4);
        let c = f.compile(&fields).unwrap();
        let want: Vec<usize> = (0..n).filter(|&i| f.matches(&rows[i])).collect();
        matched += want.len();
        for src in [&growing as &dyn ColumnSource, &written.columns, &opened.columns] {
            assert_eq!(c.eval(src).ones().collect::<Vec<_>>(), want, "{f:?}");
        }
    }
    // The filters were neither all empty nor all full.
    assert!(matched > 400 * n / 10 && matched < 400 * n * 9 / 10, "{matched}");

    // Damage is detected: in the columns at open, in a row when it is read.
    let good = fs::read(&path).unwrap();
    let mut bad = good.clone();
    let last = bad.len() - 10;
    bad[last] ^= 1;
    fs::write(&path, &bad).unwrap();
    assert!(matches!(SegmentAttrs::open(&path, &fields, n), Err(Error::Corrupt(_))));
    let mut bad = good.clone();
    bad[48 + 20] ^= 1;
    fs::write(&path, &bad).unwrap();
    let a = SegmentAttrs::open(&path, &fields, n).unwrap();
    assert!(matches!(a.row(order[0], &fields), Err(Error::Corrupt(_))));
    assert!(a.for_each_row(|_, _| Ok(())).is_err());
    assert!(SegmentAttrs::open(&path, &fields, n + 1).is_err());
    let _ = fs::remove_dir_all(&dir);
}

/// The model's exact top-k over the rows `f` matches.
fn model_search(model: &Model, metric: Metric, q: &[f32], k: usize, f: &Filter) -> Vec<(u64, f32)> {
    let mut m = model.clone();
    m.rows.retain(|_, (_, a)| f.matches(a));
    m.search(metric, q, k)
}

/// The same ids in the same order, distances equal up to rounding (SIMD
/// sums differ from the model's in the last bits).
#[track_caller]
fn assert_close(got: &[(u64, f32)], want: &[(u64, f32)], tag: &str) {
    let ids = |v: &[(u64, f32)]| v.iter().map(|x| x.0).collect::<Vec<_>>();
    assert_eq!(ids(got), ids(want), "{tag}");
    for (g, w) in got.iter().zip(want) {
        assert!((g.1 - w.1).abs() <= 1e-5 * (1.0 + w.1.abs()), "{tag}: {g:?} vs {w:?}");
    }
}

fn filtered(f: &Filter, l: usize, mode: FilterMode) -> SearchOptions {
    SearchOptions { l, filter: Some(f.clone()), filter_mode: mode, ..Default::default() }
}

/// Growing and flat segments are searched exactly, so filtered search
/// equals the model's over random writes, deletes, flushes and reopens.
#[test]
fn filtered_search_matches_the_model() {
    let dir = test_dir("coll-filter-exact");
    let dim = 5;
    let mut cfg = small_config(dim, Metric::L2, 40);
    cfg.fields = fields();
    let mut c = Collection::create(&dir, cfg, test_opts()).unwrap();
    let mut model = Model::default();
    let mut rng = SplitMix64::new(21);
    let mut nonempty = 0;
    for step in 0..250 {
        let n_ops = 1 + (rng.next_u64() % 6) as usize;
        let vs: Vec<Vec<f32>> = (0..n_ops).map(|_| int_vector(&mut rng, dim)).collect();
        let attrs: Vec<Attrs> = (0..n_ops).map(|_| random_attrs(&mut rng)).collect();
        let ops: Vec<WriteOp> = (0..n_ops)
            .map(|i| {
                let id = rng.next_u64() % 150;
                if rng.next_u64() % 5 == 0 {
                    WriteOp::Delete { id }
                } else {
                    WriteOp::Upsert { id, vector: &vs[i], attrs: &attrs[i] }
                }
            })
            .collect();
        c.write(&ops).unwrap();
        ops.iter().for_each(|op| model.apply(op));
        match rng.next_u64() % 25 {
            0 => c.flush().unwrap(),
            1 => {
                drop(c);
                c = Collection::open(&dir, test_opts()).unwrap().0;
            }
            _ => {}
        }
        for _ in 0..3 {
            let f = random_filter(&mut rng, 3);
            let q = int_vector(&mut rng, dim);
            let k = 1 + (rng.next_u64() % 10) as usize;
            let (got, st) = c.search_with_stats(&q, k, &filtered(&f, 10, FilterMode::Auto)).unwrap();
            let got: Vec<(u64, f32)> = got.iter().map(|r| (r.id, r.distance)).collect();
            assert_eq!(got, model_search(&model, Metric::L2, &q, k, &f), "step {step}: {f:?}");
            let matching = model.rows.values().filter(|(_, a)| f.matches(a)).count();
            if !matches!(f.compile(&fields()).unwrap(), Compiled::All | Compiled::Nothing) {
                assert_eq!(st.matched, matching, "{f:?}");
            }
            nonempty += usize::from(!got.is_empty() && matching < model.rows.len());
        }
    }
    assert!(nonempty > 200, "{nonempty}");
    assert!(c.stats().segments.len() > 3);
    drop(c);
    let _ = fs::remove_dir_all(&dir);
}

/// SSD segments under filters, in every mode and both layouts: no
/// matches, fewer than k, a duplicate vector of which one row matches,
/// deletes, all rows matching, and recall on a 5% filter.
#[test]
fn ssd_filters_in_every_mode() {
    for layout in [Layout::Inline, Layout::Separate] {
        let dir = test_dir(&format!("coll-filter-ssd-{}", layout.name()));
        let dim = 16;
        let mut cfg = ssd_config(dim, Metric::L2, 10_000, 2500);
        cfg.index.layout = Some(layout);
        cfg.fields = vec![Field::int("g"), Field::keyword("rare")];
        let mut c = Collection::create(&dir, cfg, test_opts()).unwrap();
        let mut model = Model::default();
        let mut rng = SplitMix64::new(9);
        let vs: Vec<Vec<f32>> = (0..3000).map(|_| float_vector(&mut rng, dim)).collect();
        let mut attrs: Vec<Attrs> = (0..3000).map(|i| Attrs::new().int("g", i as i64 % 10)).collect();
        for i in [5, 77, 1234] {
            attrs[i] = attrs[i].clone().keyword("rare", "yes");
        }
        // Three more copies of row 10's vector, each in its own group.
        let dups: Vec<Attrs> = (0..3).map(|j| Attrs::new().int("g", 100 + j)).collect();
        let mut ops: Vec<WriteOp> =
            (0..3000).map(|i| WriteOp::Upsert { id: i as u64, vector: &vs[i], attrs: &attrs[i] }).collect();
        ops.extend((0..3).map(|j| WriteOp::Upsert { id: 3000 + j as u64, vector: &vs[10], attrs: &dups[j] }));
        for chunk in ops.chunks(500) {
            c.write(chunk).unwrap();
            chunk.iter().for_each(|op| model.apply(op));
        }
        c.flush().unwrap();
        c.maintain().unwrap();
        assert_eq!(c.stats().segments.iter().map(|s| s.kind).collect::<Vec<_>>(), ["ssd"]);
        // Delete half of group 3: it keeps 5% of the rows.
        for id in (3..3000).step_by(20) {
            assert!(c.delete(id).unwrap());
            model.rows.remove(&id);
        }

        for mode in [FilterMode::Scan, FilterMode::Graph, FilterMode::Auto] {
            let search = |f: &Filter, k: usize, l: usize| {
                let (r, st) = c.search_with_stats(&vs[10], k, &filtered(f, l, mode)).unwrap();
                (r.iter().map(|r| (r.id, r.distance)).collect::<Vec<_>>(), st)
            };
            let exact = |f: &Filter, k: usize| model_search(&model, Metric::L2, &vs[10], k, f);
            let tag = format!("{} {mode:?}", layout.name());

            assert!(search(&Filter::int_eq("g", 55), 10, 40).0.is_empty(), "{tag}");
            // Fewer matches than k: a scan finds all of them.
            let rare = Filter::keyword_eq("rare", "yes");
            let (got, st) = search(&rare, 10, 40);
            if mode == FilterMode::Graph {
                assert!(got.iter().all(|g| exact(&rare, 10).contains(g)), "{tag}");
            } else {
                assert_close(&got, &exact(&rare, 10), &tag);
                assert_eq!(st.scan_segments, 1, "{tag}");
            }
            // One of four identical vectors matches.
            let one = Filter::int_eq("g", 101);
            let got = search(&one, 5, 40).0;
            assert!(got.iter().all(|g| g.0 == 3001), "{tag}: {got:?}");
            if mode != FilterMode::Graph {
                assert_eq!(got.len(), 1, "{tag}");
            }
            // A scan reading every match is exact.
            let g3 = Filter::int_eq("g", 3);
            if mode == FilterMode::Scan {
                assert_close(&search(&g3, 10, 200).0, &exact(&g3, 10), &tag);
            }
            // Recall over the 5% group, never a deleted or unmatched row.
            let mut hits = 0;
            for qi in 0..40 {
                let q = &vs[qi * 7];
                let (got, _) = c.search_with_stats(q, 10, &filtered(&g3, 40, mode)).unwrap();
                let want = model_search(&model, Metric::L2, q, 10, &g3);
                for r in &got {
                    let (_, a) = model.rows.get(&r.id).expect("live");
                    assert!(g3.matches(a), "{tag}");
                }
                hits += want.iter().filter(|w| got.iter().any(|g| g.id == w.0)).count();
            }
            let recall = hits as f64 / 400.0;
            assert!(recall >= if mode == FilterMode::Graph { 0.5 } else { 0.9 }, "{tag}: recall {recall}");
            // Every row matching: the same search as without a filter.
            let all = Filter::range("g", ..);
            let plain = c.search(&vs[10], 10, &SearchOptions { l: 40, ..Default::default() }).unwrap();
            let plain: Vec<(u64, f32)> = plain.iter().map(|r| (r.id, r.distance)).collect();
            if mode == FilterMode::Graph {
                assert_eq!(search(&all, 10, 40).0, plain, "{tag}");
            }
        }
        // Filters survive a reopen (columns read from the file).
        drop(c);
        let (c, _) = Collection::open(&dir, test_opts()).unwrap();
        let rare = Filter::keyword_eq("rare", "yes");
        let got: Vec<(u64, f32)> =
            c.search(&vs[0], 10, &filtered(&rare, 40, FilterMode::Auto)).unwrap().iter().map(|r| (r.id, r.distance)).collect();
        assert_close(&got, &model_search(&model, Metric::L2, &vs[0], 10, &rare), "reopened");
        assert_eq!(c.get(77).unwrap().unwrap().1, attrs[77]);
        drop(c);
        let _ = fs::remove_dir_all(&dir);
    }
}

#[test]
fn bad_filters_and_wal_attributes_are_errors() {
    let dir = test_dir("coll-filter-bad");
    let mut c = Collection::create(&dir, small_config(3, Metric::L2, 100), test_opts()).unwrap();
    c.upsert(1, &[1.0, 2.0, 3.0], &NO_ATTRS).unwrap();
    let q = [0.0; 3];
    assert!(c.search(&q, 5, &filtered(&Filter::int_eq("x", 1), 10, FilterMode::Auto)).is_err());
    assert_eq!(c.search(&q, 5, &filtered(&Filter::and([]), 10, FilterMode::Auto)).unwrap().len(), 1);
    assert!(c.search(&q, 5, &filtered(&Filter::or([]), 10, FilterMode::Auto)).unwrap().is_empty());
    drop(c);

    // A WAL whose attributes do not fit the fields: written by a collection
    // where "a" is an integer, replayed where it is a keyword.
    let other = test_dir("coll-filter-bad-2");
    let mut cfg = small_config(3, Metric::L2, 100);
    cfg.fields = vec![Field::int("a")];
    let mut c = Collection::create(&other, cfg.clone(), test_opts()).unwrap();
    c.upsert(1, &[1.0, 2.0, 3.0], &Attrs::new().int("a", 4)).unwrap();
    let wal = c.wal.path().to_path_buf();
    drop(c);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    cfg.fields = vec![Field::keyword("a")];
    drop(Collection::create(&dir, cfg, test_opts()).unwrap());
    fs::copy(&wal, dir.join(wal.file_name().unwrap())).unwrap();
    assert!(matches!(Collection::open(&dir, test_opts()), Err(Error::Corrupt(_))));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&other);
}

/// The scan-or-walk rule at points measured on SIFT1M (1M rows, W 4),
/// each the faster of the two there: with direct I/O a scan wins below
/// about 20-30% matching (more at larger L); from the page cache, below
/// about 6-10%.
#[test]
fn scan_or_walk_follows_the_measurements() {
    let n = 1_000_000;
    for (fraction, l, direct, scan) in [
        (0.001, 40, false, true),
        (0.05, 40, false, true),
        (0.1, 40, false, false),
        (0.1, 160, false, true),
        (0.2, 160, false, false),
        (0.5, 40, false, false),
        (0.05, 40, true, true),
        (0.1, 40, true, true),
        (0.2, 80, true, true),
        (0.2, 160, true, true),
        (0.3, 40, true, false),
        (0.3, 160, true, true),
        (0.5, 160, true, false),
    ] {
        let m = (fraction * n as f64) as usize;
        assert_eq!(prefer_scan(m, n, l, 4, direct), scan, "{fraction} L {l} direct {direct}");
    }
}
