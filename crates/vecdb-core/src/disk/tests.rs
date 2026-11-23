use std::path::PathBuf;

use super::io::tests::temp_path;
use super::*;
use crate::flat::FlatIndex;
use crate::graph::{GraphParams, Navigation};
use crate::quant::PqParams;
use crate::quant::pq::tests::structured;

const METRICS: [Metric; 3] = [Metric::L2, Metric::InnerProduct, Metric::Cosine];

/// Removes the file when the test ends, pass or fail.
struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn graph(metric: Metric, n: usize, dim: usize, seed: u64) -> (GraphIndex, Vec<f32>) {
    let data = structured(n, dim, seed);
    let ids: Vec<u64> = (0..n as u64).map(|i| i * 3 + 7).collect();
    let p = GraphParams { max_degree: 24, build_list: 48, alpha: 1.2, passes: 1, seed: 9, threads: 2 };
    let mut g = GraphIndex::build(metric, dim, &ids, data.clone(), &p).unwrap();
    g.train_pq(&PqParams { m: 8, iters: 8, max_train: 4096, seed: 3, threads: 2 }).unwrap();
    (g, data)
}

fn write(g: &GraphIndex, name: &str, layout: Layout, codec: Codec) -> (TempFile, DiskWriteReport) {
    let path = temp_path(name);
    let report = DiskIndex::write(g, &path, &DiskWriteOptions { layout: Some(layout), codec }).unwrap();
    (TempFile(path), report)
}

fn open(path: &TempFile, io: IoMode) -> DiskIndex {
    DiskIndex::open(&path.0, &DiskOpenOptions { direct: true, io }).unwrap()
}

fn params(l: usize, beam_width: usize, rerank: usize) -> DiskSearchParams {
    DiskSearchParams { l, beam_width, rerank }
}

fn queries(dim: usize) -> Vec<Vec<f32>> {
    structured(20, dim, 99).chunks(dim).map(|q| q.to_vec()).collect()
}

/// Beam width 1 must be the in-memory PQ-guided search, read from disk:
/// same results, same distances, same nodes expanded. Both layouts (the
/// separate one re-ranking every expanded node) and both I/O backends.
#[test]
fn beam_width_1_matches_in_memory_pq_search() {
    let dim = 24;
    for metric in METRICS {
        let (g, _) = graph(metric, 1200, dim, 1);
        for layout in [Layout::Inline, Layout::Separate] {
            let (path, _) = write(&g, &format!("w1-{metric:?}-{layout:?}"), layout, Codec::F32);
            for io in [IoMode::Auto, IoMode::Pread] {
                let idx = open(&path, io);
                let mut ctx = idx.context().unwrap();
                for q in queries(dim) {
                    let (want, ws) = g.search_with(&q, 10, 30, Navigation::Pq).unwrap();
                    let (got, st) = idx.search_with(&mut ctx, &q, 10, &params(30, 1, 0)).unwrap();
                    assert_eq!(got, want, "{metric:?} {layout:?} {}", ctx.io_backend());
                    assert_eq!(st.expanded, ws.expanded);
                    assert_eq!(st.pq_distances, ws.distances);
                    assert_eq!(st.rounds, st.expanded + usize::from(layout == Layout::Separate));
                    assert_eq!(st.exact_distances, st.expanded);
                }
            }
        }
    }
}

/// The same check with records and vectors longer than a sector: 1,100
/// f32 dims make inline records of two sectors and separate-layout vector
/// blocks of two sectors (the GIST and high-dimension read paths).
#[test]
fn multi_sector_records_match_in_memory_search() {
    let dim = 1100;
    let (g, _) = graph(Metric::L2, 500, dim, 12);
    for layout in [Layout::Inline, Layout::Separate] {
        let (path, _) = write(&g, &format!("multi-sector-{layout:?}"), layout, Codec::F32);
        let idx = open(&path, IoMode::Auto);
        let expect = match layout {
            Layout::Inline => (2 * SECTOR, 0),
            Layout::Separate => (SECTOR, 2 * SECTOR),
        };
        assert_eq!((idx.block_bytes(), idx.vector_block_bytes()), expect);
        let mut ctx = idx.context().unwrap();
        for q in queries(dim).iter().take(5) {
            let (want, ws) = g.search_with(q, 10, 30, Navigation::Pq).unwrap();
            let (got, st) = idx.search_with(&mut ctx, q, 10, &params(30, 1, 0)).unwrap();
            assert_eq!(got, want, "{layout:?}");
            assert_eq!(st.expanded, ws.expanded);
            let wide = idx.search_with(&mut ctx, q, 10, &params(30, 8, 0)).unwrap().0;
            assert_eq!(wide.len(), 10);
        }
        idx.verify().unwrap();
    }
}

#[test]
fn full_list_equals_flat_search() {
    let (n, dim) = (600, 16);
    for metric in METRICS {
        let (g, data) = graph(metric, n, dim, 2);
        let mut flat = FlatIndex::new(dim, metric).unwrap();
        for i in 0..n {
            flat.add(i as u64 * 3 + 7, &data[i * dim..(i + 1) * dim]).unwrap();
        }
        for layout in [Layout::Inline, Layout::Separate] {
            let (path, _) = write(&g, &format!("full-{metric:?}-{layout:?}"), layout, Codec::F32);
            let idx = open(&path, IoMode::Auto);
            for q in queries(dim) {
                let got = idx.search(&q, 10, &params(n, 8, 0)).unwrap();
                let want = flat.search(&q, 10).unwrap();
                let ids = |r: &[SearchResult]| r.iter().map(|x| x.id).collect::<Vec<_>>();
                assert_eq!(ids(&got), ids(&want), "{metric:?} {layout:?}");
                for (a, b) in got.iter().zip(&want) {
                    assert!((a.distance - b.distance).abs() <= 1e-4 * b.distance.abs().max(1.0));
                }
            }
        }
    }
}

fn recall(idx: &DiskIndex, flat: &FlatIndex, dim: usize, p: &DiskSearchParams) -> (f64, DiskSearchStats) {
    let mut ctx = idx.context().unwrap();
    let (mut hit, mut total) = (0, 0);
    let mut sum = DiskSearchStats::default();
    for q in queries(dim) {
        let (got, st) = idx.search_with(&mut ctx, &q, 10, p).unwrap();
        let want: HashSet<u64> = flat.search(&q, 10).unwrap().iter().map(|r| r.id).collect();
        hit += got.iter().filter(|r| want.contains(&r.id)).count();
        total += want.len();
        sum.rounds += st.rounds;
        sum.expanded += st.expanded;
        sum.reads += st.reads;
        sum.cache_hits += st.cache_hits;
    }
    (hit as f64 / total as f64, sum)
}

#[test]
fn wider_beams_keep_recall_in_fewer_rounds() {
    let (n, dim) = (3000, 32);
    let (g, data) = graph(Metric::L2, n, dim, 4);
    let mut flat = FlatIndex::new(dim, Metric::L2).unwrap();
    for i in 0..n {
        flat.add(i as u64 * 3 + 7, &data[i * dim..(i + 1) * dim]).unwrap();
    }
    let (path, _) = write(&g, "beams", Layout::Inline, Codec::F32);
    let idx = open(&path, IoMode::Auto);
    let (r1, s1) = recall(&idx, &flat, dim, &params(40, 1, 0));
    assert!(r1 >= 0.9, "beam 1 recall {r1}");
    for w in [2, 4, 8] {
        let (r, s) = recall(&idx, &flat, dim, &params(40, w, 0));
        assert!(r >= r1 - 0.02, "beam {w}: recall {r} vs {r1}");
        assert!(s.rounds * 3 < s1.rounds * 2, "beam {w}: {} rounds vs {}", s.rounds, s1.rounds);
    }
}

#[test]
fn codecs_and_rerank_counts() {
    let (n, dim) = (2000, 48);
    let (g, data) = graph(Metric::Cosine, n, dim, 5);
    let mut flat = FlatIndex::new(dim, Metric::Cosine).unwrap();
    for i in 0..n {
        flat.add(i as u64 * 3 + 7, &data[i * dim..(i + 1) * dim]).unwrap();
    }
    for codec in [Codec::F32, Codec::F16, Codec::BF16] {
        for layout in [Layout::Inline, Layout::Separate] {
            let (path, rep) = write(&g, &format!("codec-{}-{layout:?}", codec.name()), layout, codec);
            let idx = open(&path, IoMode::Auto);
            assert_eq!((idx.codec(), idx.layout()), (codec, layout));
            assert_eq!(rep.file_bytes, idx.file_bytes());
            let (r, _) = recall(&idx, &flat, dim, &params(60, 4, 0));
            assert!(r >= 0.9, "{} {layout:?}: recall {r}", codec.name());
            idx.verify().unwrap();
        }
    }
    // Re-ranking fewer candidates reads fewer vectors and still returns k.
    let (path, _) = write(&g, "rerank", Layout::Separate, Codec::F32);
    let idx = open(&path, IoMode::Auto);
    // Recall can only grow with the re-rank count (a superset of candidates
    // is scored exactly); how many are needed depends on how well PQ orders
    // the candidates, which is a per-dataset measurement.
    let (r_all, all) = recall(&idx, &flat, dim, &params(60, 4, 0));
    let (r40, _) = recall(&idx, &flat, dim, &params(60, 4, 40));
    let (r20, few) = recall(&idx, &flat, dim, &params(60, 4, 20));
    assert!(few.reads < all.reads);
    assert!(r20 <= r40 && r40 <= r_all && r_all >= 0.9, "recall {r20} {r40} {r_all}");
    let got = idx.search(&queries(dim)[0], 10, &params(60, 4, 3)).unwrap();
    assert_eq!(got.len(), 10, "rerank is raised to k");
}

#[test]
fn identical_vectors_keep_all_their_ids() {
    let dim = 8;
    let mut data = structured(400, dim, 6);
    let copy = data[..dim].to_vec();
    data.extend_from_slice(&copy);
    data.extend_from_slice(&copy);
    let ids: Vec<u64> = (0..402).collect();
    let p = GraphParams { max_degree: 16, build_list: 32, ..GraphParams::default() };
    let mut g = GraphIndex::build(Metric::L2, dim, &ids, data, &p).unwrap();
    g.train_pq(&PqParams { m: 4, iters: 5, ..PqParams::new(4) }).unwrap();
    assert_eq!(g.nodes(), 400);
    let (path, _) = write(&g, "dups", Layout::Inline, Codec::F32);
    let idx = open(&path, IoMode::Auto);
    assert_eq!((idx.len(), idx.nodes()), (402, 400));
    let got = idx.search(&copy, 3, &params(50, 4, 0)).unwrap();
    let mut found: Vec<u64> = got.iter().map(|r| r.id).collect();
    found.sort();
    assert_eq!(found, vec![0, 400, 401]);
    assert!(got.iter().all(|r| r.distance == 0.0));
}

#[test]
fn cache_changes_reads_not_results() {
    let dim = 24;
    let (g, _) = graph(Metric::L2, 1500, dim, 7);
    let (path, _) = write(&g, "cache", Layout::Inline, Codec::F32);
    let mut idx = open(&path, IoMode::Auto);
    let mut ctx = idx.context().unwrap();
    let before: Vec<_> =
        queries(dim).iter().map(|q| idx.search_with(&mut ctx, q, 10, &params(40, 4, 0)).unwrap()).collect();
    assert_eq!(idx.cache_hot_nodes(200).unwrap(), 200);
    assert!(idx.ram_bytes().1 >= 200 * 4 * dim);
    for (q, (want, ws)) in queries(dim).iter().zip(&before) {
        let (got, st) = idx.search_with(&mut ctx, q, 10, &params(40, 4, 0)).unwrap();
        assert_eq!(&got, want);
        assert!(st.cache_hits > 0);
        assert_eq!(st.reads + st.cache_hits, ws.reads);
    }
}

fn flip(path: &TempFile, at: u64) {
    use std::os::unix::fs::FileExt;
    let f = std::fs::OpenOptions::new().read(true).write(true).open(&path.0).unwrap();
    let mut b = [0u8; 1];
    f.read_exact_at(&mut b, at).unwrap();
    f.write_all_at(&[b[0] ^ 0x40], at).unwrap();
}

#[test]
fn damage_is_detected_not_panicked_on() {
    let dim = 16;
    let (g, _) = graph(Metric::L2, 800, dim, 8);
    let q = &queries(dim)[0];
    for layout in [Layout::Inline, Layout::Separate] {
        let (path, _) = write(&g, &format!("damage-{layout:?}"), layout, Codec::F32);
        let h = open(&path, IoMode::Auto).header.clone();

        // Truncation is caught at open.
        let len = std::fs::metadata(&path.0).unwrap().len();
        let short = TempFile(temp_path(&format!("damage-short-{layout:?}")));
        std::fs::copy(&path.0, &short.0).unwrap();
        std::fs::OpenOptions::new().write(true).open(&short.0).unwrap().set_len(len - SECTOR as u64).unwrap();
        assert!(matches!(DiskIndex::open(&short.0, &DiskOpenOptions::default()), Err(Error::Corrupt(_))));

        // A flipped bit in a RAM section is caught at open.
        flip(&path, h.sections[CODES].offset + 5);
        assert!(matches!(DiskIndex::open(&path.0, &DiskOpenOptions::default()), Err(Error::Corrupt(_))));
        flip(&path, h.sections[CODES].offset + 5);

        // A flipped bit in the entry node's record is caught by the first read.
        let geo = h.geometry();
        let (block, within) = geo.locate(h.entry);
        flip(&path, h.sections[GRAPH].offset + block + within as u64 + 4);
        let idx = open(&path, IoMode::Auto);
        assert!(matches!(idx.search(q, 10, &params(30, 4, 0)), Err(Error::Corrupt(_))));
        assert!(matches!(idx.verify(), Err(Error::Corrupt(_))));
        flip(&path, h.sections[GRAPH].offset + block + within as u64 + 4);
        let idx = open(&path, IoMode::Auto);
        idx.verify().unwrap();
        let (good, _) = idx.search_with(&mut idx.context().unwrap(), q, 10, &params(30, 4, 0)).unwrap();

        if layout == Layout::Separate {
            // A flipped bit in a returned vector is caught when it is re-ranked.
            let node = (0..idx.nodes() as u32).find(|&v| idx.ids_of(v)[0] == good[0].id).unwrap();
            flip(&path, h.sections[VECTORS].offset + node as u64 * geo.vector_block_bytes as u64 + 2);
            let idx = open(&path, IoMode::Auto);
            assert!(matches!(idx.search(q, 10, &params(30, 4, 0)), Err(Error::Corrupt(_))));
            assert!(matches!(idx.verify(), Err(Error::Corrupt(_))));
        }
    }
}

#[test]
fn rejects_bad_input() {
    let dim = 8;
    let data = structured(300, dim, 10);
    let ids: Vec<u64> = (0..300).collect();
    let p = GraphParams { max_degree: 8, build_list: 16, ..GraphParams::default() };
    let mut g = GraphIndex::build(Metric::L2, dim, &ids, data, &p).unwrap();
    let path = TempFile(temp_path("bad-input"));
    let err = DiskIndex::write(&g, &path.0, &DiskWriteOptions::default());
    assert!(matches!(err, Err(Error::InvalidArgument(_))), "PQ not trained");
    g.train_pq(&PqParams { m: 2, iters: 3, ..PqParams::new(2) }).unwrap();
    let opts = DiskWriteOptions { layout: None, codec: Codec::SQ8 };
    assert!(matches!(DiskIndex::write(&g, &path.0, &opts), Err(Error::InvalidArgument(_))));
    assert!(!path.0.exists());
    let rep = DiskIndex::write(&g, &path.0, &DiskWriteOptions::default()).unwrap();
    assert_eq!(rep.layout, Layout::Inline);
    let idx = open(&path, IoMode::Pread);
    assert!(matches!(idx.search(&[0.0; 7], 5, &params(10, 1, 0)), Err(Error::DimensionMismatch { .. })));
    assert!(matches!(idx.search(&[f32::NAN; 8], 5, &params(10, 1, 0)), Err(Error::NonFiniteValue)));
    assert!(idx.search(&[0.0; 8], 5, &params(10, 0, 0)).is_err());
    assert!(idx.search(&[0.0; 8], 5, &params(10, MAX_BEAM_WIDTH + 1, 0)).is_err());
    assert!(idx.search(&[0.0; 8], 0, &params(10, 1, 0)).unwrap().is_empty());
    assert_eq!(idx.search(&[0.0; 8], 500, &params(10, 4, 0)).unwrap().len(), 300, "k > n returns everything reachable");
    let not_index = TempFile(temp_path("not-an-index"));
    std::fs::write(&not_index.0, vec![7u8; 3 * SECTOR]).unwrap();
    assert!(matches!(DiskIndex::open(&not_index.0, &DiskOpenOptions::default()), Err(Error::Corrupt(_))));
}

#[test]
fn opens_without_direct_io() {
    let (g, _) = graph(Metric::L2, 400, 8, 11);
    let (path, _) = write(&g, "buffered", Layout::Inline, Codec::F32);
    let idx = DiskIndex::open(&path.0, &DiskOpenOptions { direct: false, io: IoMode::Auto }).unwrap();
    assert!(!idx.is_direct());
    let want = open(&path, IoMode::Auto).search(&queries(8)[0], 5, &DiskSearchParams::default()).unwrap();
    assert_eq!(idx.search(&queries(8)[0], 5, &DiskSearchParams::default()).unwrap(), want);
}
