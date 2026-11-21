//! File round-trip and corruption tests for segments and dataset formats.

use std::fs;
use std::path::PathBuf;

use vecdb_core::dataset::{read_bvecs, read_fvecs, read_ivecs, read_vectors_f32, write_fvecs, write_ivecs};
use vecdb_core::format::{read_flat_segment, write_flat_segment};
use vecdb_core::rng::SplitMix64;
use vecdb_core::{Error, FlatIndex, Metric, VectorStore};

fn test_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn random_index(metric: Metric, dim: usize, n: usize, seed: u64) -> FlatIndex {
    let mut rng = SplitMix64::new(seed);
    let mut index = FlatIndex::with_capacity(dim, metric, n).unwrap();
    for i in 0..n {
        let v: Vec<f32> = (0..dim).map(|_| rng.uniform(-1.0, 1.0)).collect();
        // Sparse, non-sequential ids to make sure ids are stored, not implied.
        index.add(i as u64 * 1_000_003 + 17, &v).unwrap();
    }
    index
}

#[test]
fn segment_round_trip_all_metrics() {
    let dir = test_dir("segment_round_trip");
    for (i, metric) in [Metric::L2, Metric::InnerProduct, Metric::Cosine].into_iter().enumerate() {
        let index = random_index(metric, 37, 500, i as u64);
        let path = dir.join(format!("{}.vdb", metric.name()));
        write_flat_segment(&path, metric, index.store()).unwrap();

        let (m, store) = read_flat_segment(&path).unwrap();
        assert_eq!(m, metric);
        assert_eq!(store.dim(), 37);
        assert_eq!(store.ids(), index.store().ids());
        assert_eq!(store.raw_vectors(), index.store().raw_vectors());

        let loaded = FlatIndex::from_store(store, m);
        let q: Vec<f32> = (0..37).map(|j| (j as f32).sin()).collect();
        assert_eq!(loaded.search(&q, 10).unwrap(), index.search(&q, 10).unwrap());
        assert!(!dir.join(format!("{}.vdb.tmp", metric.name())).exists());
    }
}

#[test]
fn segment_round_trip_dimension_extremes() {
    let dir = test_dir("segment_dims");
    for dim in [32usize, 8192, vecdb_core::MAX_DIM] {
        let index = random_index(Metric::Cosine, dim, 20, dim as u64);
        let path = dir.join(format!("d{dim}.vdb"));
        write_flat_segment(&path, Metric::Cosine, index.store()).unwrap();
        let (_, store) = read_flat_segment(&path).unwrap();
        assert_eq!(store.dim(), dim);
        assert_eq!(store.raw_vectors(), index.store().raw_vectors());
    }
}

#[test]
fn empty_segment_round_trip() {
    let dir = test_dir("empty_segment");
    let index = FlatIndex::new(8, Metric::L2).unwrap();
    let path = dir.join("empty.vdb");
    write_flat_segment(&path, Metric::L2, index.store()).unwrap();
    let (_, store) = read_flat_segment(&path).unwrap();
    assert_eq!(store.len(), 0);
    assert_eq!(store.dim(), 8);
}

#[test]
fn segment_overwrite_replaces_file() {
    let dir = test_dir("segment_overwrite");
    let path = dir.join("seg.vdb");
    write_flat_segment(&path, Metric::L2, random_index(Metric::L2, 4, 10, 1).store()).unwrap();
    write_flat_segment(&path, Metric::L2, random_index(Metric::L2, 4, 3, 2).store()).unwrap();
    assert_eq!(read_flat_segment(&path).unwrap().1.len(), 3);
}

#[test]
fn segment_detects_corruption() {
    let dir = test_dir("segment_corruption");
    let path = dir.join("seg.vdb");
    write_flat_segment(&path, Metric::L2, random_index(Metric::L2, 16, 100, 9).store()).unwrap();
    let good = fs::read(&path).unwrap();

    let expect_corrupt = |bytes: &[u8], what: &str| {
        fs::write(&path, bytes).unwrap();
        match read_flat_segment(&path) {
            Err(Error::Corrupt(_)) => {}
            other => panic!("{what}: expected Corrupt, got {other:?}"),
        }
    };

    let mut b = good.clone();
    let last = b.len() - 3;
    b[last] ^= 0x40; // a vector byte
    expect_corrupt(&b, "flipped vector byte");

    let mut b = good.clone();
    b[64] ^= 1; // an id byte
    expect_corrupt(&b, "flipped id byte");

    expect_corrupt(&good[..good.len() - 4], "truncated");
    expect_corrupt(&good[..10], "shorter than header");

    let mut b = good.clone();
    b.extend_from_slice(&[0; 4]);
    expect_corrupt(&b, "trailing bytes");

    let mut b = good.clone();
    b[0] = b'X';
    expect_corrupt(&b, "bad magic");
}

#[test]
fn fvecs_ivecs_round_trip_and_limit() {
    let dir = test_dir("vecs_round_trip");
    let data: Vec<f32> = (0..30).map(|i| i as f32 * 0.5).collect();
    let fpath = dir.join("x.fvecs");
    write_fvecs(&fpath, 3, &data).unwrap();
    let all = read_fvecs(&fpath, None).unwrap();
    assert_eq!((all.dim, all.len()), (3, 10));
    assert_eq!(all.data, data);
    assert_eq!(all.row(2), &[3.0, 3.5, 4.0]);
    assert_eq!(read_fvecs(&fpath, Some(4)).unwrap().len(), 4);
    assert_eq!(read_vectors_f32(&fpath, None).unwrap(), all);

    let ids: Vec<i32> = (0..20).collect();
    let ipath = dir.join("gt.ivecs");
    write_ivecs(&ipath, 5, &ids).unwrap();
    let gt = read_ivecs(&ipath, None).unwrap();
    assert_eq!((gt.dim, gt.data), (5, ids));
}

#[test]
fn bvecs_reads_as_bytes_and_f32() {
    let dir = test_dir("bvecs");
    let path = dir.join("x.bvecs");
    let mut raw = Vec::new();
    for row in [[1u8, 2], [250, 0]] {
        raw.extend_from_slice(&2i32.to_le_bytes());
        raw.extend_from_slice(&row);
    }
    fs::write(&path, &raw).unwrap();
    assert_eq!(read_bvecs(&path, None).unwrap().data, vec![1, 2, 250, 0]);
    assert_eq!(read_vectors_f32(&path, None).unwrap().data, vec![1.0, 2.0, 250.0, 0.0]);
}

#[test]
fn vecs_reject_malformed_files() {
    let dir = test_dir("vecs_malformed");
    let path = dir.join("bad.fvecs");
    write_fvecs(&path, 4, &[1.0; 8]).unwrap();
    let good = fs::read(&path).unwrap();

    fs::write(&path, &good[..good.len() - 1]).unwrap();
    assert!(matches!(read_fvecs(&path, None), Err(Error::Corrupt(_))), "truncated row");

    fs::write(&path, &good[..good.len() - 18]).unwrap();
    assert!(matches!(read_fvecs(&path, None), Err(Error::Corrupt(_))), "truncated header");

    let mut b = good.clone();
    b[20..24].copy_from_slice(&3i32.to_le_bytes()); // second row claims dim 3
    fs::write(&path, &b).unwrap();
    assert!(matches!(read_fvecs(&path, None), Err(Error::Corrupt(_))), "mixed dims");

    let mut b = good.clone();
    b[0..4].copy_from_slice(&(-1i32).to_le_bytes());
    fs::write(&path, &b).unwrap();
    assert!(matches!(read_fvecs(&path, None), Err(Error::Corrupt(_))), "negative dim");

    fs::write(&path, b"").unwrap();
    assert!(read_fvecs(&path, None).unwrap().is_empty());

    assert!(read_vectors_f32(&dir.join("x.txt"), None).is_err());
}
