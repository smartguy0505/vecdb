//! Tests of the partitioned build (`build.rs`).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use super::*;
use crate::cancel::Cancel;
use crate::dataset::Vecs;
use crate::flat::FlatIndex;
use crate::graph::GraphParams;
use crate::quant::PqParams;
use crate::quant::pq::tests::structured;

/// A test's directory, removed when the test ends.
struct Dir(PathBuf);

impl Dir {
    fn new(name: &str) -> Self {
        Self(test_dir(name))
    }

    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    /// Entries left in the directory other than `keep`.
    fn others(&self, keep: &[&PathBuf]) -> Vec<PathBuf> {
        std::fs::read_dir(&self.0).unwrap().map(|e| e.unwrap().path()).filter(|p| !keep.contains(&p)).collect()
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn graph_params(threads: usize) -> GraphParams {
    GraphParams { max_degree: 20, build_list: 40, alpha: 1.2, passes: 1, seed: 9, threads }
}

fn pq_params(max_train: usize) -> PqParams {
    PqParams { m: 8, iters: 6, max_train, seed: 3, threads: 2 }
}

fn params(partitions: usize, merge: MergePrune, threads: usize) -> PartitionedBuildParams {
    PartitionedBuildParams {
        graph: graph_params(threads),
        pq: pq_params(4096),
        write: DiskWriteOptions::default(),
        memory_budget: 0,
        partitions,
        overlap: 2,
        partition_degree: 0,
        merge,
        scratch_dir: None,
        progress: None,
        cancel: Cancel::default(),
    }
}

/// `n` structured rows plus repeats: row 0 three more times, row 1 as -0.0
/// where it has 0.0 (equal under `==`), so identical vectors are exercised.
fn data_with_repeats(n: usize, dim: usize, seed: u64) -> Vecs<f32> {
    let mut data = structured(n, dim, seed);
    data[dim + 1] = 0.0;
    let mut copy1 = data[dim..2 * dim].to_vec();
    copy1[1] = -0.0;
    let copy0 = data[..dim].to_vec();
    for (at, row) in [(n / 3, &copy0), (n / 2, &copy1), (n / 2 + 5, &copy0), (n - 1, &copy0)] {
        data[at * dim..(at + 1) * dim].copy_from_slice(row);
    }
    Vecs { dim, data }
}

fn ids(n: usize) -> Vec<u64> {
    (0..n as u64).map(|i| i * 5 + 11).collect()
}

fn flat(metric: Metric, v: &Vecs<f32>, ids: &[u64]) -> FlatIndex {
    let mut f = FlatIndex::new(v.dim, metric).unwrap();
    for (i, &id) in ids.iter().enumerate() {
        f.add(id, v.row(i)).unwrap();
    }
    f
}

fn recall(idx: &DiskIndex, flat: &FlatIndex, dim: usize, l: usize) -> f64 {
    let (mut hit, mut total) = (0, 0);
    for q in structured(30, dim, 77).chunks(dim) {
        let p = DiskSearchParams { l, beam_width: 4, rerank: 0 };
        let got = idx.search(q, 10, &p).unwrap();
        let want: HashSet<u64> = flat.search(q, 10).unwrap().iter().map(|r| r.id).collect();
        hit += got.iter().filter(|r| want.contains(&r.id)).count();
        total += want.len();
    }
    hit as f64 / total as f64
}

fn open(path: &Path) -> DiskIndex {
    DiskIndex::open(path, &DiskOpenOptions { direct: false, io: IoMode::Pread }).unwrap()
}

/// One partition runs the whole pipeline (scan, sample, encode, partition
/// build, merge, streaming write) and must write the very file the
/// in-memory build writes: same nodes for identical rows, same PQ sample,
/// same entry, same bytes. Every metric, both layouts, a codec, and a PQ
/// sample smaller than the collection.
#[test]
fn one_partition_writes_the_in_memory_file() {
    let (n, dim) = (1500, 24);
    let cases = [
        (Metric::L2, Layout::Inline, Codec::F32, 4096),
        (Metric::InnerProduct, Layout::Separate, Codec::F32, 600),
        (Metric::Cosine, Layout::Inline, Codec::F16, 600),
    ];
    for (metric, layout, codec, max_train) in cases {
        let dir = Dir::new(&format!("part1-{metric:?}"));
        let v = data_with_repeats(n, dim, 1);
        let ids = ids(n);
        let write = DiskWriteOptions { layout: Some(layout), codec };

        let mut g = GraphIndex::build(metric, dim, &ids, v.data.clone(), &graph_params(2)).unwrap();
        g.train_pq(&pq_params(max_train)).unwrap();
        assert_eq!(g.nodes(), n - 4, "test data should hold 4 repeats");
        let want = dir.file("memory.vdx");
        DiskIndex::write(&g, &want, &write).unwrap();

        let got = dir.file("partitioned.vdx");
        let p = PartitionedBuildParams { pq: pq_params(max_train), write, ..params(1, MergePrune::Pq, 3) };
        let rep = DiskIndex::build_partitioned(&v, Some(&ids), metric, &got, &p).unwrap();
        assert_eq!((rep.rows, rep.nodes, rep.partitions, rep.overlap), (n, n - 4, 1, 1));
        assert_eq!((rep.unreachable, rep.unrepaired), (0, 0));
        assert!(std::fs::read(&got).unwrap() == std::fs::read(&want).unwrap(), "{metric:?}: files differ");
        assert!(dir.others(&[&want, &got]).is_empty(), "scratch left behind");
    }
}

/// Several partitions: every merge rule gives a sound, fully reachable index
/// whose recall is close to the in-memory build's.
#[test]
fn partitions_merge_into_a_searchable_index() {
    let (n, dim) = (4000, 16);
    let dir = Dir::new("parts");
    for metric in [Metric::L2, Metric::InnerProduct] {
        let v = data_with_repeats(n, dim, 2);
        let ids = ids(n);
        let flat = flat(metric, &v, &ids);
        let mut g = GraphIndex::build(metric, dim, &ids, v.data.clone(), &graph_params(2)).unwrap();
        g.train_pq(&pq_params(4096)).unwrap();
        let mem_path = dir.file("memory.vdx");
        DiskIndex::write(&g, &mem_path, &DiskWriteOptions::default()).unwrap();
        let base = recall(&open(&mem_path), &flat, dim, 60);
        assert!(base >= 0.9, "{metric:?}: in-memory recall {base}");

        for merge in [MergePrune::Pq, MergePrune::Random] {
            let path = dir.file(&format!("{}.vdx", merge.name()));
            let rep = DiskIndex::build_partitioned(&v, Some(&ids), metric, &path, &params(6, merge, 2)).unwrap();
            assert_eq!((rep.partitions, rep.overlap, rep.nodes), (6, 2, n - 4));
            assert_eq!(rep.unrepaired, 0, "{metric:?} {merge:?}");
            assert!(rep.mean_degree <= 20.0);
            assert!(rep.largest_partition < n, "largest partition {}", rep.largest_partition);
            let idx = open(&path);
            idx.verify().unwrap();
            assert_eq!((idx.len(), idx.nodes(), idx.max_degree()), (n, n - 4, 20));
            let r = recall(&idx, &flat, dim, 60);
            assert!(r >= base - 0.06, "{metric:?} {merge:?}: recall {r}, in-memory {base}");
        }
    }
}

/// The partition count, the partitions and the merge do not depend on the
/// thread count.
#[test]
fn same_file_for_any_thread_count() {
    let (n, dim) = (3000, 12);
    let dir = Dir::new("threads");
    let v = data_with_repeats(n, dim, 3);
    let files: Vec<Vec<u8>> = [1, 3, 8]
        .iter()
        .map(|&threads| {
            let path = dir.file(&format!("t{threads}.vdx"));
            let p = PartitionedBuildParams {
                memory_budget: partition_node_bytes(dim, 20) * 1200,
                pq: PqParams { threads, ..pq_params(4096) },
                ..params(0, MergePrune::Pq, threads)
            };
            let rep = DiskIndex::build_partitioned(&v, None, Metric::Cosine, &path, &p).unwrap();
            assert!(rep.partitions >= 5, "{} partitions", rep.partitions);
            std::fs::read(&path).unwrap()
        })
        .collect();
    assert!(files[0] == files[1] && files[0] == files[2]);
}

/// A memory budget sets the partition count, and no partition outgrows it.
#[test]
fn memory_budget_bounds_the_partitions() {
    let (n, dim) = (6000, 16);
    let dir = Dir::new("budget");
    let v = Vecs { dim, data: structured(n, dim, 4) };
    let path = dir.file("i.vdx");
    for budget_nodes in [7000, 2500, 900] {
        let p = PartitionedBuildParams {
            memory_budget: partition_node_bytes(dim, 20) * budget_nodes,
            ..params(0, MergePrune::Pq, 2)
        };
        let rep = DiskIndex::build_partitioned(&v, None, Metric::L2, &path, &p);
        if budget_nodes < 1000 {
            assert!(matches!(rep, Err(Error::InvalidArgument(_))), "{rep:?}");
            continue;
        }
        let rep = rep.unwrap();
        assert_eq!(rep.budget_nodes, Some(budget_nodes as usize));
        assert!(rep.largest_partition <= budget_nodes as usize, "{rep:?}");
        if budget_nodes as usize >= n {
            assert_eq!(rep.partitions, 1);
        } else {
            assert!(rep.partitions >= 2 * n / budget_nodes as usize, "{rep:?}");
        }
        assert_eq!(rep.unrepaired, 0);
        open(&path).verify().unwrap();
    }
}

/// Copies of a vector are one node however the rows are partitioned, and a
/// search returns every copy's id.
#[test]
fn identical_vectors_keep_all_their_ids_across_partitions() {
    let dim = 8;
    let mut data = structured(3000, dim, 6);
    let copy = data[..dim].to_vec();
    for at in [500, 1700, 2999] {
        data[at * dim..(at + 1) * dim].copy_from_slice(&copy);
    }
    let v = Vecs { dim, data };
    let dir = Dir::new("repeats");
    let path = dir.file("i.vdx");
    let rep = DiskIndex::build_partitioned(&v, None, Metric::L2, &path, &params(5, MergePrune::Pq, 2)).unwrap();
    assert_eq!(rep.nodes, 2997);
    let idx = open(&path);
    assert_eq!((idx.len(), idx.nodes()), (3000, 2997));
    let got = idx.search(&copy, 4, &DiskSearchParams { l: 50, beam_width: 4, rerank: 0 }).unwrap();
    let mut found: Vec<u64> = got.iter().filter(|r| r.distance == 0.0).map(|r| r.id).collect();
    found.sort();
    assert_eq!(found, vec![0, 500, 1700, 2999]);
}

#[test]
fn rejects_bad_input_and_cleans_up() {
    let dim = 8;
    let dir = Dir::new("bad");
    let path = dir.file("i.vdx");
    let v = Vecs { dim, data: structured(600, dim, 8) };
    let ok = params(3, MergePrune::Pq, 2);
    let build = |v: &Vecs<f32>, ids: Option<&[u64]>, p: &PartitionedBuildParams| {
        DiskIndex::build_partitioned(v, ids, Metric::L2, &path, p)
    };
    let mut dup_ids = ids(600);
    dup_ids[7] = dup_ids[3];
    assert!(matches!(build(&v, Some(&dup_ids), &ok), Err(Error::DuplicateId(_))));
    assert!(matches!(build(&v, Some(&dup_ids[1..]), &ok), Err(Error::InvalidArgument(_))));
    let bad_degree = PartitionedBuildParams { partition_degree: 21, ..ok.clone() };
    assert!(matches!(build(&v, None, &bad_degree), Err(Error::InvalidArgument(_))));
    let no_budget = params(0, MergePrune::Pq, 2);
    assert!(matches!(build(&v, None, &no_budget), Err(Error::InvalidArgument(_))));
    let empty = Vecs { dim, data: Vec::new() };
    assert!(build(&empty, None, &ok).is_err());
    // Fewer vectors than PQ needs to train.
    let tiny = Vecs { dim, data: structured(100, dim, 8) };
    assert!(build(&tiny, None, &ok).is_err());
    let mut nan = v.clone();
    nan.data[dim * 420 + 3] = f32::NAN;
    assert!(matches!(build(&nan, None, &ok), Err(Error::NonFiniteValue)));
    assert!(dir.others(&[]).is_empty(), "left behind: {:?}", dir.others(&[]));
    build(&v, None, &ok).unwrap();
    assert!(dir.others(&[&path]).is_empty());
}

/// Reading the rows from a file (by position, in runs with gaps) builds the
/// same index as reading them from memory. 300-dim rows make the gather
/// skip gaps of more than 13 rows, so it reads both runs and single rows.
#[test]
fn file_source_builds_the_same_index() {
    let (n, dim) = (2500, 300);
    let dir = Dir::new("file-source");
    let v = data_with_repeats(n, dim, 9);
    let fvecs = dir.file("base.fvecs");
    crate::dataset::write_fvecs(&fvecs, dim, &v.data).unwrap();
    let file = crate::dataset::VectorFile::open(&fvecs, None).unwrap();
    let p = PartitionedBuildParams { pq: pq_params(1024), ..params(7, MergePrune::Pq, 4) };
    let (a, b) = (dir.file("memory.vdx"), dir.file("file.vdx"));
    DiskIndex::build_partitioned(&v, None, Metric::L2, &a, &p).unwrap();
    let rep = DiskIndex::build_partitioned(&file, None, Metric::L2, &b, &p).unwrap();
    assert_eq!((rep.nodes, rep.partitions), (n - 4, 7));
    assert!(std::fs::read(&a).unwrap() == std::fs::read(&b).unwrap());
}

thread_local! {
    /// Progress lines left until the build is cancelled, and its flag.
    static CANCEL_AT: std::cell::RefCell<(usize, Cancel)> = Default::default();
}

/// Cancelled after any of its progress lines, a build stops and leaves no
/// files.
#[test]
fn a_cancelled_build_leaves_nothing() {
    fn count_down(_: &str) {
        CANCEL_AT.with_borrow_mut(|(left, cancel)| {
            *left = left.saturating_sub(1);
            if *left == 0 {
                cancel.cancel();
            }
        });
    }
    let dir = Dir::new("build-cancel");
    let path = dir.file("i.vdx");
    let data = Vecs { dim: 8, data: structured(3000, 8, 1) };
    let mut lines = 0;
    loop {
        lines += 1;
        let cancel = Cancel::default();
        CANCEL_AT.set((lines, cancel.clone()));
        let p = PartitionedBuildParams {
            progress: Some(count_down),
            cancel: cancel.clone(),
            ..params(2, MergePrune::Pq, 2)
        };
        let r = DiskIndex::build_partitioned(&data, None, Metric::L2, &path, &p);
        if !cancel.is_cancelled() {
            r.unwrap();
            break;
        }
        assert!(matches!(r, Err(Error::Cancelled)), "after line {lines}: {:?}", r.map(|r| r.rows));
        assert!(dir.others(&[]).is_empty(), "{:?}", dir.others(&[]));
    }
    assert!(lines > 5, "{lines} lines");
    // And before it starts.
    let p = PartitionedBuildParams { cancel: Cancel::default(), ..params(2, MergePrune::Pq, 2) };
    p.cancel.cancel();
    let again = dir.file("again.vdx");
    assert!(matches!(DiskIndex::build_partitioned(&data, None, Metric::L2, &again, &p), Err(Error::Cancelled)));
    assert_eq!(dir.others(&[&path]).len(), 0);
}

/// In the middle of a graph, too.
#[test]
fn a_graph_build_is_cancelled_midway() {
    let data = structured(20_000, 8, 1);
    let points = crate::graph::build::Points::new(&data, 8, Metric::L2, 1);
    let p = GraphParams { max_degree: 32, build_list: 200, ..graph_params(1) };
    let t0 = std::time::Instant::now();
    crate::graph::build::build(&points, &p, &Cancel::default()).unwrap();
    let full = t0.elapsed();
    let cancel = Cancel::default();
    std::thread::scope(|s| {
        s.spawn(|| {
            std::thread::sleep(full / 5);
            cancel.cancel();
        });
        let t0 = std::time::Instant::now();
        assert!(matches!(crate::graph::build::build(&points, &p, &cancel), Err(Error::Cancelled)));
        assert!(t0.elapsed() < full / 2, "{:?} of {full:?}", t0.elapsed());
    });
}
