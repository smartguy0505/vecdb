//! vecdb-bench: dataset tools and the recall/latency harness.

mod coll;

use std::collections::HashMap;
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::{Duration, Instant};

use vecdb_core::dataset::{self, VectorFile, VectorSource, Vecs};
use vecdb_core::format::{read_flat_segment, write_flat_segment};
use vecdb_core::kernels::{self, Kernels};
use vecdb_core::rng::SplitMix64;
use vecdb_core::quant::{Codec, PqParams};
use vecdb_core::disk::{
    DiskIndex, DiskOpenOptions, DiskSearchParams, DiskWriteOptions, IoMode, Layout, MergePrune, PartitionedBuildParams,
    SearchContext, partition_node_bytes,
};
use vecdb_core::{FlatIndex, GraphIndex, GraphParams, Metric, Navigation, PqIndex, SearchResult};

type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;

const USAGE: &str = "\
vecdb-bench <command> [--flag value]...

commands:
  synth        --out DIR [--dim 128] [--n 100000] [--nq 1000] [--seed 42]
               [--intrinsic R [--noise 0.05]]
               write DIR/base.fvecs and DIR/query.fvecs: uniform random, or with
               --intrinsic, embedding-like data from an R-dim subspace plus noise
  groundtruth  --base F --query F --out gt.ivecs [--k 100] [--metric l2] [--threads 4] [--limit N]
               exact k-NN with the flat index (base ids = row numbers)
  build        --base F --out seg.vdb [--metric l2] [--limit N]
               write a flat segment file
  search       (--base F | --segment S) --query F [--gt gt.ivecs] [--k 10]
               [--metric l2] [--threads 4] [--limit N] [--nq N]
               run queries; print QPS, latency percentiles and recall@k
  pq           --base F --query F --gt G [--metric l2] [--m 8,16,32] [--rerank 0,10,100]
               [--codecs f32,f16,bf16,sq8] [--k 10] [--threads 4] [--limit N] [--nq N]
               [--max-train 65536] [--iters 25]
               product quantization: recall and speed per code size, re-rank depth, codec
  graph        --base F --query F --gt G [--metric l2] [--k 10] [--l 10,20,40,80,160]
               [--degree 64] [--build-l 100] [--alpha 1.2] [--passes 1] [--threads 4]
               [--pq-m M] [--iters 15] [--max-train 65536] [--limit N] [--nq N]
               Vamana graph: build once, then recall, distances, hops, QPS (--threads)
               and 1-thread latency per search list size L; with --pq-m, also
               PQ-guided search (codes steer, expanded nodes re-scored exactly)
  disk-build   --base F --out PREFIX --pq-m M [--metric l2] [--layout auto] [--codec f32]
               [--degree 64] [--build-l 100] [--alpha 1.2] [--passes 1] [--threads 4]
               [--iters 15] [--max-train 65536] [--limit N]
               [--build-mem GB | --partitions K] [--overlap 2] [--part-degree R]
               [--merge pq] [--scratch DIR]
               build the graph and PQ once, then write PREFIX.<layout>.<codec>.vdx for
               every --layout (auto, inline, separate) and --codec (f32, f16, bf16) listed.
               With --build-mem or --partitions: a partitioned build that reads F by
               position (never loads it), builds overlapping partitions one at a time
               within the budget and merges them (--merge pq or random);
               one layout and codec only
  disk-search  --index FILE --query F --gt G [--k 10] [--l 20,40,80,160] [--w 1,2,4,8]
               [--rerank 0] [--threads 4] [--cache 0] [--io auto] [--direct true]
               [--nq N] [--warmup 200] [--verify false]
               open an SSD index in a fresh process and search it: recall, reads,
               rounds, 1-thread latency and QPS (--threads) per beam width W and L
  coll-ingest  --base F --dir D [--metric l2] [--batch 1000] [--durability buffered|sync]
               [--seal-mb 32] [--index-mb 64] [--max-ssd 8] [--pq-m M] [--threads 6]
               [--build-mem 2] [--maintain false] [--optimize false] [--secs S] [--limit N]
               [--labels false]
               create a collection in D and upsert F (ids = row numbers) through the WAL:
               rows/s and batch latency; --maintain runs index builds as they fall due,
               --optimize merges everything into one SSD index at the end; then reopen.
               --labels adds int fields u (uniform 0..999) and p (bucket of a random
               projection) for coll-filter
  coll-search  --from D0 --dir D --base F --query Q [--k 10] [--l 40,80,160] [--w 4]
               [--nq 1000] [--delete 0,0.1,0.3] [--seed 1] [--adapt true|false|both]
               [--compact false]
               copy collection D0 to D, delete random rows in stages (cumulative
               fractions), and measure recall against exact search over the live rows,
               reads and 1-thread latency; --compact runs maintain() after the last stage
  coll-filter  --dir D --base F --query Q [--k 10] [--l 40,80,160] [--w 4] [--nq 300]
               [--filter u,p,near] [--select 0.001,0.01,0.05,0.2,0.5] [--mode graph,scan,auto]
               [--warm true] [--direct true]
               search a collection made with --labels under range filters matching the
               given fraction of rows: u = uniform labels, p = one slab of the projection
               (matches far from most queries), near = a slab around each query's own
               projection, none = no filter (the baseline); recall against exact search
               over the matching rows
  kernels      [--dims 32,64,...,8192] [--ms 200] [--repeat 5]
               ns per distance for every kernel set this CPU can run (data in cache)

F = .fvecs or .bvecs (disk-build --build-mem/--partitions also reads .fbin and .u8bin).
Metrics: l2, ip, cosine.
Set VECDB_KERNELS=scalar (or avx2-fma) to force a kernel set for A/B runs.
";

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> AnyResult<()> {
    let mut argv = std::env::args().skip(1);
    let cmd = argv.next().unwrap_or_default();
    if matches!(cmd.as_str(), "" | "help" | "-h" | "--help") {
        print!("{USAGE}");
        return Ok(());
    }
    let args = Args::parse(argv)?;
    match cmd.as_str() {
        "synth" => synth(&args),
        "groundtruth" => groundtruth(&args),
        "build" => build(&args),
        "search" => search(&args),
        "kernels" => kernel_bench(&args),
        "pq" => pq_bench(&args),
        "graph" => graph_bench(&args),
        "disk-build" => disk_build(&args),
        "disk-search" => disk_search(&args),
        "coll-ingest" => coll::ingest(&args),
        "coll-search" => coll::search(&args),
        "coll-filter" => coll::filter(&args),
        other => Err(format!("unknown command {other:?}\n\n{USAGE}").into()),
    }
}

struct Args {
    map: HashMap<String, String>,
}

impl Args {
    fn parse(it: impl Iterator<Item = String>) -> AnyResult<Self> {
        let mut map = HashMap::new();
        let mut it = it;
        while let Some(a) = it.next() {
            let key = a.strip_prefix("--").ok_or_else(|| format!("expected --flag, got {a:?}"))?;
            let val = it.next().ok_or_else(|| format!("--{key} needs a value"))?;
            if map.insert(key.to_string(), val).is_some() {
                return Err(format!("--{key} given twice").into());
            }
        }
        Ok(Self { map })
    }

    fn allow(&self, keys: &[&str]) -> AnyResult<()> {
        for k in self.map.keys() {
            if !keys.contains(&k.as_str()) {
                return Err(format!("unknown flag --{k}").into());
            }
        }
        Ok(())
    }

    fn path(&self, key: &str) -> Option<PathBuf> {
        self.map.get(key).map(PathBuf::from)
    }

    fn req_path(&self, key: &str) -> AnyResult<PathBuf> {
        self.path(key).ok_or_else(|| format!("--{key} is required").into())
    }

    fn num<T: FromStr>(&self, key: &str, default: T) -> AnyResult<T>
    where
        T::Err: Display,
    {
        match self.map.get(key) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|e| format!("--{key} {v:?}: {e}").into()),
        }
    }

    fn opt_num<T: FromStr>(&self, key: &str) -> AnyResult<Option<T>>
    where
        T::Err: Display,
    {
        self.map
            .get(key)
            .map(|v| v.parse().map_err(|e| format!("--{key} {v:?}: {e}").into()))
            .transpose()
    }

    fn metric(&self) -> AnyResult<Metric> {
        let s = self.map.get("metric").map(String::as_str).unwrap_or("l2");
        Metric::parse(s).ok_or_else(|| format!("unknown metric {s:?} (l2, ip, cosine)").into())
    }
}

fn synth(a: &Args) -> AnyResult<()> {
    a.allow(&["out", "dim", "n", "nq", "seed", "intrinsic", "noise"])?;
    let out = a.req_path("out")?;
    let dim: usize = a.num("dim", 128)?;
    let n: usize = a.num("n", 100_000)?;
    let nq: usize = a.num("nq", 1_000)?;
    let mut rng = SplitMix64::new(a.num("seed", 42)?);
    std::fs::create_dir_all(&out)?;
    let intrinsic: usize = a.num("intrinsic", 0)?;
    let (base, query) = if intrinsic == 0 {
        // Uniform in [-1, 1): no structure; the worst case for quantization.
        let base: Vec<f32> = (0..n * dim).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let query: Vec<f32> = (0..nq * dim).map(|_| rng.uniform(-1.0, 1.0)).collect();
        (base, query)
    } else {
        // Embedding-like: x = A z / sqrt(r) + noise, z ~ N(0, I_r), r = intrinsic.
        let noise: f32 = a.num("noise", 0.05)?;
        let mat: Vec<f32> = (0..dim * intrinsic).map(|_| rng.gaussian()).collect();
        let scale = 1.0 / (intrinsic as f32).sqrt();
        let mut gen_rows = |rows: usize| -> Vec<f32> {
            let mut out = Vec::with_capacity(rows * dim);
            let mut z = vec![0.0f32; intrinsic];
            for _ in 0..rows {
                z.iter_mut().for_each(|v| *v = rng.gaussian());
                for d in 0..dim {
                    let x = vecdb_core::distance::dot(&mat[d * intrinsic..(d + 1) * intrinsic], &z);
                    out.push(x * scale + noise * rng.gaussian());
                }
            }
            out
        };
        let base = gen_rows(n);
        let query = gen_rows(nq);
        (base, query)
    };
    dataset::write_fvecs(&out.join("base.fvecs"), dim, &base)?;
    dataset::write_fvecs(&out.join("query.fvecs"), dim, &query)?;
    println!("wrote {n} base + {nq} query vectors, dim {dim}, to {}", out.display());
    Ok(())
}

fn load_vectors(path: &Path, limit: Option<usize>) -> AnyResult<Vecs<f32>> {
    let v = dataset::read_vectors_f32(path, limit)?;
    if v.is_empty() {
        return Err(format!("{}: no vectors", path.display()).into());
    }
    Ok(v)
}

/// Builds a flat index whose ids are the row numbers of `base`.
fn index_from_base(base: &Vecs<f32>, metric: Metric) -> AnyResult<FlatIndex> {
    let mut index = FlatIndex::with_capacity(base.dim, metric, base.len())?;
    for i in 0..base.len() {
        index.add(i as u64, base.row(i))?;
    }
    Ok(index)
}

struct RunStats {
    results: Vec<Vec<SearchResult>>,
    latencies: Vec<Duration>,
    wall: Duration,
}

/// Runs every query through `search` on `threads` threads; records each
/// query's latency and the wall time of the whole run.
fn run_queries<F>(search: &F, queries: &Vecs<f32>, threads: usize) -> AnyResult<RunStats>
where
    F: Fn(&[f32]) -> vecdb_core::Result<Vec<SearchResult>> + Sync,
{
    let nq = queries.len();
    let threads = threads.clamp(1, nq.max(1));
    let per_thread = nq.div_ceil(threads);
    let start = Instant::now();
    let parts: Vec<vecdb_core::Result<Vec<(Vec<SearchResult>, Duration)>>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let (lo, hi) = (t * per_thread, ((t + 1) * per_thread).min(nq));
                s.spawn(move || {
                    (lo..hi)
                        .map(|i| {
                            let t0 = Instant::now();
                            let r = search(queries.row(i))?;
                            Ok((r, t0.elapsed()))
                        })
                        .collect()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("search thread panicked")).collect()
    });
    let wall = start.elapsed();
    let mut results = Vec::with_capacity(nq);
    let mut latencies = Vec::with_capacity(nq);
    for part in parts {
        for (r, d) in part? {
            results.push(r);
            latencies.push(d);
        }
    }
    Ok(RunStats { results, latencies, wall })
}

impl RunStats {
    fn qps(&self) -> f64 {
        self.results.len() as f64 / self.wall.as_secs_f64()
    }

    /// Latency percentile in milliseconds.
    fn pct_ms(&self, p: f64) -> f64 {
        let mut lat = self.latencies.clone();
        lat.sort_unstable();
        lat[((p / 100.0) * (lat.len() - 1) as f64).round() as usize].as_secs_f64() * 1e3
    }
}

fn check_query_dim(queries: &Vecs<f32>, dim: usize) -> AnyResult<()> {
    if queries.dim != dim {
        return Err(format!("query dim {} != index dim {dim}", queries.dim).into());
    }
    Ok(())
}

fn groundtruth(a: &Args) -> AnyResult<()> {
    a.allow(&["base", "query", "out", "k", "metric", "threads", "limit"])?;
    let out = a.req_path("out")?;
    let k: usize = a.num("k", 100)?;
    let metric = a.metric()?;
    let base = load_vectors(&a.req_path("base")?, a.opt_num("limit")?)?;
    let queries = load_vectors(&a.req_path("query")?, None)?;
    if base.len() < k {
        return Err(format!("base has {} vectors, fewer than k = {k}", base.len()).into());
    }
    if base.len() > i32::MAX as usize {
        return Err("ivecs ids are i32; base too large".into());
    }
    let index = index_from_base(&base, metric)?;
    check_query_dim(&queries, index.dim())?;
    let stats = run_queries(&|q: &[f32]| index.search(q, k), &queries, a.num("threads", 4)?)?;
    let ids: Vec<i32> = stats.results.iter().flatten().map(|r| r.id as i32).collect();
    dataset::write_ivecs(&out, k, &ids)?;
    println!(
        "wrote top-{k} ({}) for {} queries to {} in {:.2} s",
        metric.name(),
        queries.len(),
        out.display(),
        stats.wall.as_secs_f64()
    );
    Ok(())
}

fn build(a: &Args) -> AnyResult<()> {
    a.allow(&["base", "out", "metric", "limit"])?;
    let out = a.req_path("out")?;
    let metric = a.metric()?;
    let base = load_vectors(&a.req_path("base")?, a.opt_num("limit")?)?;
    let t0 = Instant::now();
    let index = index_from_base(&base, metric)?;
    write_flat_segment(&out, metric, index.store())?;
    let size = std::fs::metadata(&out)?.len();
    println!(
        "wrote {} ({} vectors x {} dims, {}, {:.1} MiB) in {:.2} s",
        out.display(),
        index.len(),
        index.dim(),
        metric.name(),
        size as f64 / (1 << 20) as f64,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn search(a: &Args) -> AnyResult<()> {
    a.allow(&["base", "segment", "query", "gt", "k", "metric", "threads", "limit", "nq"])?;
    let k: usize = a.num("k", 10)?;
    let threads: usize = a.num("threads", 4)?;
    let t0 = Instant::now();
    let index = match (a.path("segment"), a.path("base")) {
        (Some(seg), None) => {
            if a.map.contains_key("metric") || a.map.contains_key("limit") {
                return Err("--metric/--limit come from the segment; drop them".into());
            }
            let (metric, store) = read_flat_segment(&seg)?;
            FlatIndex::from_store(store, metric)
        }
        (None, Some(base)) => index_from_base(&load_vectors(&base, a.opt_num("limit")?)?, a.metric()?)?,
        _ => return Err("give exactly one of --base or --segment".into()),
    };
    let load_time = t0.elapsed();
    let queries = load_vectors(&a.req_path("query")?, a.opt_num("nq")?)?;
    check_query_dim(&queries, index.dim())?;
    let stats = run_queries(&|q: &[f32]| index.search(q, k), &queries, threads)?;
    let nq = queries.len();

    println!(
        "index    flat, {}, {} vectors x {} dims, kernels {} (loaded in {:.2} s)",
        index.metric().name(),
        index.len(),
        index.dim(),
        kernels::best().name,
        load_time.as_secs_f64()
    );
    println!("queries  {nq}, k {k}, threads {}", threads.clamp(1, nq));
    println!(
        "speed    {:.1} QPS; latency ms p50 {:.3}  p95 {:.3}  p99 {:.3}  max {:.3}",
        stats.qps(),
        stats.pct_ms(50.0),
        stats.pct_ms(95.0),
        stats.pct_ms(99.0),
        stats.pct_ms(100.0)
    );

    if let Some(gt_path) = a.path("gt") {
        let gt = load_gt(&gt_path, nq, k)?;
        println!("recall   recall@{k} = {:.4}", recall(&stats.results, &gt, k));
        let metric = index.metric();
        let m = explain_misses(&stats.results, &gt, k, &queries, metric, |id| index.store().get(id));
        m.print();
    }
    Ok(())
}

fn load_gt(path: &Path, nq: usize, k: usize) -> AnyResult<Vecs<i32>> {
    let gt = dataset::read_ivecs(path, Some(nq))?;
    if gt.len() < nq || gt.dim < k {
        return Err(format!("ground truth has {} rows x {} ids; need {nq} x {k}", gt.len(), gt.dim).into());
    }
    Ok(gt)
}

fn recall(results: &[Vec<SearchResult>], gt: &Vecs<i32>, k: usize) -> f64 {
    let mut hits = 0usize;
    for (i, res) in results.iter().enumerate() {
        let truth = &gt.row(i)[..k];
        hits += res.iter().filter(|r| truth.contains(&(r.id as i32))).count();
    }
    hits as f64 / (results.len() * k).max(1) as f64
}

/// Exact distance in f64. Cosine normalizes both sides, so it is correct
/// whether or not `v` is stored normalized.
fn exact_distance(metric: Metric, q: &[f32], v: &[f32]) -> f64 {
    let dot: f64 = q.iter().zip(v).map(|(&a, &b)| a as f64 * b as f64).sum();
    match metric {
        Metric::L2 => q.iter().zip(v).map(|(&a, &b)| (a as f64 - b as f64).powi(2)).sum(),
        Metric::InnerProduct => -dot,
        Metric::Cosine => {
            let nq = q.iter().map(|&a| a as f64 * a as f64).sum::<f64>().sqrt();
            let nv = v.iter().map(|&b| b as f64 * b as f64).sum::<f64>().sqrt();
            if nq == 0.0 || nv == 0.0 { 1.0 } else { 1.0 - dot / (nq * nv) }
        }
    }
}

#[derive(Default)]
struct MissReport {
    queries: usize,
    tie_only: usize,
    real: usize,
    unknown: usize,
}

impl MissReport {
    fn print(&self) {
        if self.queries == 0 {
            println!("misses   none");
            return;
        }
        println!(
            "misses   {} queries miss a true neighbor: {} only at exact ties with the k-th result, \
             {} real (a closer vector was skipped){}",
            self.queries,
            self.tie_only,
            self.real,
            if self.unknown > 0 { format!(", {} unknown ids", self.unknown) } else { String::new() }
        );
    }
}

/// For each query that misses a ground-truth neighbor, decides whether the
/// miss is a tie (the missed vector is no closer than our farthest result,
/// by exact f64 distance) or a real miss.
fn explain_misses<'a>(
    results: &[Vec<SearchResult>],
    gt: &Vecs<i32>,
    k: usize,
    queries: &Vecs<f32>,
    metric: Metric,
    vector_of: impl Fn(u64) -> Option<&'a [f32]>,
) -> MissReport {
    let mut rep = MissReport::default();
    for (i, res) in results.iter().enumerate() {
        let truth = &gt.row(i)[..k];
        let ours: Vec<u64> = res.iter().map(|r| r.id).collect();
        let missing: Vec<u64> =
            truth.iter().map(|&t| t as u64).filter(|t| !ours.contains(t)).collect();
        if missing.is_empty() {
            continue;
        }
        rep.queries += 1;
        let q = queries.row(i);
        let dist = |id: u64| vector_of(id).map(|v| exact_distance(metric, q, v));
        let farthest_ours = ours.iter().filter_map(|&id| dist(id)).fold(f64::NEG_INFINITY, f64::max);
        let mut real = false;
        for id in missing {
            match dist(id) {
                None => rep.unknown += 1,
                Some(d) if d < farthest_ours - 1e-5 * farthest_ours.abs().max(1e-12) => real = true,
                Some(_) => {}
            }
        }
        if real { rep.real += 1 } else { rep.tie_only += 1 }
    }
    rep
}

const DEFAULT_DIMS: &str = "32,64,128,256,512,768,1024,1536,2048,4096,8192";

/// Calls `f` repeatedly for about `budget`; returns nanoseconds per call.
fn ns_per_call(budget: Duration, mut f: impl FnMut()) -> f64 {
    f(); // warm-up: page in, train branch predictors
    let start = Instant::now();
    let mut calls = 0u64;
    loop {
        f();
        calls += 1;
        let elapsed = start.elapsed();
        if elapsed >= budget {
            return elapsed.as_nanos() as f64 / calls as f64;
        }
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn kernel_bench(a: &Args) -> AnyResult<()> {
    a.allow(&["dims", "ms", "repeat"])?;
    let dims: Vec<usize> = a
        .map
        .get("dims")
        .map(String::as_str)
        .unwrap_or(DEFAULT_DIMS)
        .split(',')
        .map(|d| d.trim().parse::<usize>().map_err(|e| format!("--dims {d:?}: {e}")))
        .collect::<Result<_, _>>()?;
    if dims.iter().any(|&d| d == 0 || d > vecdb_core::MAX_DIM) {
        return Err("--dims values must be in 1..=65536".into());
    }
    let budget = Duration::from_millis(a.num("ms", 200)?);
    let repeat: usize = a.num::<usize>("repeat", 5)?.max(1);
    let sets = kernels::available();
    let names: Vec<&str> = sets.iter().map(|k| k.name).collect();
    println!("kernel sets: {} (default: {})", names.join(", "), kernels::best().name);
    println!(
        "ns per distance, data in cache (64 KiB block, at least 4 rows), median of {repeat}; \
         batch = one call per block, single = one call per vector"
    );
    print!("{:>5} {:>4}", "dim", "op");
    for k in &sets {
        print!(" {:>16} {:>16}", format!("{}/batch", k.name), format!("{}/single", k.name));
    }
    println!(" {:>8}", "speedup");

    type Batch = fn(&[f32], &[f32], &mut [f32]);
    type Single = fn(&[f32], &[f32]) -> f32;
    let ops: [(&str, fn(&Kernels) -> (Batch, Single)); 2] = [
        ("l2", |k| (k.l2_squared_batch, k.l2_squared)),
        ("dot", |k| (k.dot_batch, k.dot)),
    ];
    for &dim in &dims {
        let rows = (64 * 1024 / (dim * 4)).clamp(4, 1024);
        let mut rng = SplitMix64::new(dim as u64);
        let q: Vec<f32> = (0..dim).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let data: Vec<f32> = (0..rows * dim).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let mut out = vec![0.0f32; rows];
        for (op, pick) in &ops {
            print!("{dim:>5} {op:>4}");
            let mut batch_ns = Vec::new();
            for k in &sets {
                let (batch, single) = pick(k);
                let b = median(
                    (0..repeat)
                        .map(|_| {
                            ns_per_call(budget, || {
                                batch(std::hint::black_box(&q), &data, &mut out);
                                std::hint::black_box(&mut out);
                            }) / rows as f64
                        })
                        .collect(),
                );
                let s = median(
                    (0..repeat)
                        .map(|_| {
                            ns_per_call(budget, || {
                                for row in data.chunks_exact(dim) {
                                    std::hint::black_box(single(std::hint::black_box(&q), row));
                                }
                            }) / rows as f64
                        })
                        .collect(),
                );
                print!(" {b:>16.2} {s:>16.2}");
                batch_ns.push(b);
            }
            println!(" {:>7.2}x", batch_ns[0] / batch_ns[batch_ns.len() - 1]);
        }
    }
    Ok(())
}

fn parse_list<T: FromStr>(a: &Args, key: &str, default: &str) -> AnyResult<Vec<T>>
where
    T::Err: Display,
{
    a.map
        .get(key)
        .map(String::as_str)
        .unwrap_or(default)
        .split(',')
        .map(|v| v.trim().parse::<T>().map_err(|e| format!("--{key} {v:?}: {e}").into()))
        .collect()
}

fn pq_bench(a: &Args) -> AnyResult<()> {
    a.allow(&[
        "base", "query", "gt", "metric", "m", "rerank", "codecs", "k", "threads", "limit", "nq", "max-train",
        "iters",
    ])?;
    let metric = a.metric()?;
    let k: usize = a.num("k", 10)?;
    let threads: usize = a.num("threads", 4)?;
    let ms: Vec<usize> = parse_list(a, "m", "8,16,32")?;
    let reranks: Vec<usize> = parse_list(a, "rerank", "0,10,100")?;
    let codecs: Vec<Codec> = parse_list::<String>(a, "codecs", "f32,f16,bf16,sq8")?
        .iter()
        .map(|c| Codec::parse(c).ok_or_else(|| format!("unknown codec {c:?} (f32, f16, bf16, sq8)")))
        .collect::<Result<_, _>>()?;
    let base = load_vectors(&a.req_path("base")?, a.opt_num("limit")?)?;
    let queries = load_vectors(&a.req_path("query")?, a.opt_num("nq")?)?;
    check_query_dim(&queries, base.dim)?;
    let gt = load_gt(&a.req_path("gt")?, queries.len(), k)?;
    let ids: Vec<u64> = (0..base.len() as u64).collect();
    println!(
        "data     {} x {} ({}), {} queries, k {k}, threads {threads}, kernels {}",
        base.len(),
        base.dim,
        metric.name(),
        queries.len(),
        kernels::best().name
    );
    println!(
        "{:>4} {:>8} {:>8} {:>7} {:>6} {:>9} {:>10} {:>9} {:>8}",
        "m", "PQ B/vec", "build s", "rerank", "codec", "full B/v", "recall@k", "QPS", "p50 ms"
    );
    for &m in &ms {
        let params = PqParams {
            m,
            iters: a.num("iters", 25)?,
            max_train: a.num("max-train", 65_536)?,
            seed: 0x5eed,
            threads,
        };
        let t0 = Instant::now();
        let mut index = PqIndex::build(metric, base.dim, &ids, &base.data, &params, None)?;
        let build_s = t0.elapsed().as_secs_f64();
        let mut first = true;
        let mut row = |rerank: usize, codec: Option<Codec>, index: &PqIndex| -> AnyResult<()> {
            let stats = run_queries(&|q: &[f32]| index.search(q, k, rerank), &queries, threads)?;
            let (codec_name, full_b) = match index.full_vectors() {
                Some((c, b)) if codec.is_some() => (c.name().to_string(), b.to_string()),
                _ => ("-".to_string(), "-".to_string()),
            };
            println!(
                "{:>4} {:>8} {:>8} {:>7} {:>6} {:>9} {:>10.4} {:>9.1} {:>8.3}",
                if first { m.to_string() } else { String::new() },
                if first { index.code_size().to_string() } else { String::new() },
                if first { format!("{build_s:.1}") } else { String::new() },
                rerank,
                codec_name,
                full_b,
                recall(&stats.results, &gt, k),
                stats.qps(),
                stats.pct_ms(50.0)
            );
            first = false;
            Ok(())
        };
        if reranks.contains(&0) {
            row(0, None, &index)?;
        }
        for &codec in &codecs {
            index.set_full_vectors(&base.data, Some(codec))?;
            for &r in reranks.iter().filter(|&&r| r > 0) {
                row(r, Some(codec), &index)?;
            }
        }
    }
    Ok(())
}

fn graph_bench(a: &Args) -> AnyResult<()> {
    a.allow(&[
        "base", "query", "gt", "metric", "k", "l", "degree", "build-l", "alpha", "passes", "threads", "pq-m", "iters",
        "max-train", "limit", "nq",
    ])?;
    let metric = a.metric()?;
    let k: usize = a.num("k", 10)?;
    let threads: usize = a.num("threads", 4)?;
    let ls: Vec<usize> = parse_list(a, "l", "10,20,40,80,160")?;
    let params = GraphParams {
        max_degree: a.num("degree", 64)?,
        build_list: a.num("build-l", 100)?,
        alpha: a.num("alpha", 1.2)?,
        passes: a.num("passes", 1)?,
        seed: 0x5eed,
        threads,
    };
    let base = load_vectors(&a.req_path("base")?, a.opt_num("limit")?)?;
    let queries = load_vectors(&a.req_path("query")?, a.opt_num("nq")?)?;
    check_query_dim(&queries, base.dim)?;
    let gt = load_gt(&a.req_path("gt")?, queries.len(), k)?;
    let ids: Vec<u64> = (0..base.len() as u64).collect();
    println!(
        "data     {} x {} ({}), {} queries, k {k}, kernels {}",
        base.len(),
        base.dim,
        metric.name(),
        queries.len(),
        kernels::best().name
    );
    let t0 = Instant::now();
    // The index takes the vectors over: one copy in memory.
    let mut index = GraphIndex::build(metric, base.dim, &ids, base.data, &params)?;
    println!(
        "graph    R {}, build L {}, alpha {}, passes {}, {threads} threads: built in {:.1} s",
        params.max_degree,
        params.build_list,
        params.alpha,
        params.passes,
        t0.elapsed().as_secs_f64()
    );
    println!(
        "         {} nodes ({} identical vectors merged), mean degree {:.1}, {} reachable from the entry",
        index.nodes(),
        index.len() - index.nodes(),
        index.mean_degree(),
        index.reachable()
    );
    let mut navs = vec![Navigation::Exact];
    if let Some(m) = a.opt_num::<usize>("pq-m")? {
        let t0 = Instant::now();
        index.train_pq(&PqParams {
            m,
            iters: a.num("iters", 15)?,
            max_train: a.num("max-train", 65_536)?,
            seed: 0x5eed,
            threads,
        })?;
        println!("pq       {m} B/vector, trained and encoded in {:.1} s", t0.elapsed().as_secs_f64());
        navs.push(Navigation::Pq);
    }
    let (graph_b, vec_b, pq_b) = index.memory();
    let per = |b: usize| b as f64 / index.len() as f64;
    println!(
        "memory   B/vector: graph {:.0}, full vectors {:.0}, PQ codes {:.0}",
        per(graph_b),
        per(vec_b),
        per(pq_b)
    );
    // Warm-up: page everything in once before timing.
    let _ = run_queries(&|q: &[f32]| index.search(q, k, ls[0]), &queries, threads)?;
    println!(
        "{:>5} {:>5} {:>9} {:>8} {:>7} {:>10} {:>8} {:>8}",
        "nav",
        "L",
        format!("recall@{k}"),
        "dist/q",
        "hops/q",
        format!("QPS {threads}t"),
        "p50 ms",
        "p99 ms"
    );
    for &nav in &navs {
        for &l in &ls {
            let (dists, hops) = (AtomicUsize::new(0), AtomicUsize::new(0));
            let search = |q: &[f32]| {
                let (res, st) = index.search_with(q, k, l, nav)?;
                dists.fetch_add(st.distances, Relaxed);
                hops.fetch_add(st.expanded, Relaxed);
                Ok(res)
            };
            let single = run_queries(&search, &queries, 1)?;
            let nq = queries.len() as f64;
            let (d, h) = (dists.load(Relaxed) as f64 / nq, hops.load(Relaxed) as f64 / nq);
            let multi = if threads > 1 { run_queries(&search, &queries, threads)?.qps() } else { single.qps() };
            println!(
                "{:>5} {:>5} {:>9.4} {:>8.0} {:>7.1} {:>10.0} {:>8.3} {:>8.3}",
                match nav {
                    Navigation::Exact => "exact",
                    Navigation::Pq => "pq",
                },
                l,
                recall(&single.results, &gt, k),
                d,
                h,
                multi,
                single.pct_ms(50.0),
                single.pct_ms(99.0)
            );
        }
    }
    println!("latency: 1 thread; dist/q: distance computations that steer the search (PQ for nav pq);");
    println!("hops/q: nodes expanded (a disk index reads one block per hop)");
    Ok(())
}

fn disk_build(a: &Args) -> AnyResult<()> {
    a.allow(&[
        "base", "out", "metric", "layout", "codec", "degree", "build-l", "alpha", "passes", "threads", "pq-m", "iters",
        "max-train", "limit", "build-mem", "partitions", "overlap", "part-degree", "merge", "scratch",
    ])?;
    let metric = a.metric()?;
    let threads: usize = a.num("threads", 4)?;
    let out = a.req_path("out")?;
    let layouts: Vec<String> = parse_list(a, "layout", "auto")?;
    let layouts: Vec<Option<Layout>> = layouts
        .iter()
        .map(|s| match s.as_str() {
            "auto" => Ok(None),
            s => Layout::parse(s).map(Some).ok_or_else(|| format!("unknown layout {s:?} (auto, inline, separate)")),
        })
        .collect::<Result<_, _>>()?;
    let codecs: Vec<String> = parse_list(a, "codec", "f32")?;
    let codecs: Vec<Codec> = codecs
        .iter()
        .map(|s| Codec::parse(s).ok_or_else(|| format!("unknown codec {s:?}")))
        .collect::<Result<_, _>>()?;
    let m: usize = a.opt_num("pq-m")?.ok_or("--pq-m is required")?;
    let params = GraphParams {
        max_degree: a.num("degree", 64)?,
        build_list: a.num("build-l", 100)?,
        alpha: a.num("alpha", 1.2)?,
        passes: a.num("passes", 1)?,
        seed: 0x5eed,
        threads,
    };
    let pq = PqParams { m, iters: a.num("iters", 15)?, max_train: a.num("max-train", 65_536)?, seed: 0x5eed, threads };
    if a.map.contains_key("build-mem") || a.map.contains_key("partitions") {
        return disk_build_partitioned(a, metric, &out, &layouts, &codecs, params, pq);
    }
    let base = load_vectors(&a.req_path("base")?, a.opt_num("limit")?)?;
    let (n, dim) = (base.len(), base.dim);
    println!("data     {n} x {dim} ({}), kernels {}", metric.name(), kernels::best().name);
    let ids: Vec<u64> = (0..n as u64).collect();
    let t0 = Instant::now();
    let mut index = GraphIndex::build(metric, dim, &ids, base.data, &params)?;
    println!(
        "graph    R {}, build L {}, alpha {}, passes {}, {threads} threads: built in {:.1} s; {} nodes, mean degree {:.1}, {} reachable",
        params.max_degree,
        params.build_list,
        params.alpha,
        params.passes,
        t0.elapsed().as_secs_f64(),
        index.nodes(),
        index.mean_degree(),
        index.reachable()
    );
    let t0 = Instant::now();
    index.train_pq(&pq)?;
    println!("pq       {m} B/vector, trained and encoded in {:.1} s", t0.elapsed().as_secs_f64());
    for layout in &layouts {
        for &codec in &codecs {
            let chosen = layout.unwrap_or_else(|| Layout::auto(dim, codec, params.max_degree));
            let name = format!("{}.{}.{}.vdx", out.display(), chosen.name(), codec.name());
            let t0 = Instant::now();
            let rep = DiskIndex::write(&index, &name, &DiskWriteOptions { layout: Some(chosen), codec })?;
            println!(
                "wrote    {name}: {:.2} GB in {:.1} s; SSD {:.0} B/node, RAM {:.1} B/vector",
                rep.file_bytes as f64 / 1e9,
                t0.elapsed().as_secs_f64(),
                rep.ssd_bytes_per_node,
                rep.ram_bytes as f64 / n as f64
            );
        }
    }
    Ok(())
}

static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

fn progress(msg: &str) {
    println!("  {:8.1} s  {msg}", START.get_or_init(Instant::now).elapsed().as_secs_f64());
}

fn disk_build_partitioned(
    a: &Args,
    metric: Metric,
    out: &Path,
    layouts: &[Option<Layout>],
    codecs: &[Codec],
    graph: GraphParams,
    pq: PqParams,
) -> AnyResult<()> {
    let (&[layout], &[codec]) = (layouts, codecs) else {
        return Err("a partitioned build writes one --layout and one --codec".into());
    };
    START.get_or_init(Instant::now);
    let base = a.req_path("base")?;
    let src = VectorFile::open(&base, a.opt_num("limit")?)?;
    let (n, dim) = (src.len(), src.dim());
    let gb: Option<f64> = a.opt_num("build-mem")?;
    let merge_name = a.map.get("merge").map(String::as_str).unwrap_or("pq");
    let p = PartitionedBuildParams {
        memory_budget: gb.map_or(0, |gb| (gb * 1e9) as u64),
        partitions: a.num("partitions", 0)?,
        overlap: a.num("overlap", 2)?,
        partition_degree: a.num("part-degree", 0)?,
        merge: MergePrune::parse(merge_name).ok_or_else(|| format!("unknown --merge {merge_name:?} (pq, random)"))?,
        scratch_dir: a.path("scratch"),
        write: DiskWriteOptions { layout, codec },
        progress: Some(progress),
        graph,
        pq,
    };
    let part_r = if p.partition_degree == 0 { p.graph.max_degree } else { p.partition_degree };
    println!(
        "data     {n} x {dim} ({}) read by position from {}, kernels {}",
        metric.name(),
        base.display(),
        kernels::best().name
    );
    match gb {
        Some(gb) => println!(
            "budget   {gb} GB per partition = {} nodes at {} B/node",
            (gb * 1e9) as u64 / partition_node_bytes(dim, part_r),
            partition_node_bytes(dim, part_r)
        ),
        None => println!("budget   {} partitions given", p.partitions),
    }
    println!(
        "graph    R {}, partition R {part_r}, build L {}, alpha {}, passes {}, overlap {}, merge {}, {} threads",
        p.graph.max_degree,
        p.graph.build_list,
        p.graph.alpha,
        p.graph.passes,
        p.overlap,
        p.merge.name(),
        p.graph.threads
    );
    let chosen = layout.unwrap_or_else(|| Layout::auto(dim, codec, p.graph.max_degree));
    let name = format!("{}.{}.{}.vdx", out.display(), chosen.name(), codec.name());
    let t0 = Instant::now();
    let read0 = storage_read_bytes();
    let rep = DiskIndex::build_partitioned(&src, None, metric, &name, &p)?;
    let total = t0.elapsed().as_secs_f64();
    let read = storage_read_bytes() - read0;
    println!(
        "built    {} partitions (largest {} nodes{}), {} of {} rows distinct, mean degree {:.1}, {} unreachable after merge ({} unrepaired)",
        rep.partitions,
        rep.largest_partition,
        rep.budget_nodes.map_or(String::new(), |b| format!(", budget {b}")),
        rep.nodes,
        rep.rows,
        rep.mean_degree,
        rep.unreachable,
        rep.unrepaired
    );
    println!(
        "wrote    {name}: {:.2} GB; SSD {:.0} B/node, RAM {:.1} B/vector",
        rep.write.file_bytes as f64 / 1e9,
        rep.write.ssd_bytes_per_node,
        rep.write.ram_bytes as f64 / n as f64
    );
    let steps: Vec<String> = rep.steps.iter().map(|(s, t)| format!("{s} {t:.1}")).collect();
    println!("time     {total:.1} s: {}", steps.join(", "));
    let (_, peak) = memory_usage();
    println!(
        "memory   peak RSS {:.2} GB; scratch disk peak {:.2} GB; read {:.1} GB from the device, page-cache hits not counted (input file {:.2} GB)",
        peak as f64 / 1e9,
        rep.scratch_bytes as f64 / 1e9,
        read as f64 / 1e9,
        std::fs::metadata(&base).map_or(0, |m| m.len()) as f64 / 1e9
    );
    Ok(())
}

/// (VmRSS, VmHWM) of this process in bytes.
fn memory_usage() -> (u64, u64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |name: &str| {
        status
            .lines()
            .find_map(|l| l.strip_prefix(name))
            .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
            .map_or(0, |kb| kb * 1024)
    };
    (field("VmRSS:"), field("VmHWM:"))
}

/// (user, system) CPU seconds used by this process so far, all threads
/// included, from /proc/self/stat (USER_HZ is 100 on Linux).
fn cpu_seconds() -> (f64, f64) {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let rest = stat.rsplit_once(')').map_or("", |(_, r)| r);
    let f: Vec<&str> = rest.split_whitespace().collect();
    let tick = |i: usize| f.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0) / 100.0;
    (tick(11), tick(12))
}

/// Bytes this process has caused to be read from storage (/proc/self/io).
fn storage_read_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/io")
        .unwrap_or_default()
        .lines()
        .find_map(|l| l.strip_prefix("read_bytes:"))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0)
}

thread_local! {
    static DISK_CTX: std::cell::RefCell<Option<SearchContext>> = const { std::cell::RefCell::new(None) };
}

fn disk_search(a: &Args) -> AnyResult<()> {
    a.allow(&[
        "index", "query", "gt", "k", "l", "w", "rerank", "threads", "cache", "io", "direct", "nq", "warmup", "verify",
    ])?;
    let k: usize = a.num("k", 10)?;
    let threads: usize = a.num("threads", 4)?;
    let ls: Vec<usize> = parse_list(a, "l", "20,40,80,160")?;
    let ws: Vec<usize> = parse_list(a, "w", "1,2,4,8")?;
    let reranks: Vec<usize> = parse_list(a, "rerank", "0")?;
    let io_name = a.map.get("io").map(String::as_str).unwrap_or("auto");
    let io = IoMode::parse(io_name).ok_or_else(|| format!("unknown --io {io_name:?} (auto, uring, pread)"))?;
    let direct: bool = a.num("direct", true)?;
    let path = a.req_path("index")?;

    let (rss0, _) = memory_usage();
    let t0 = Instant::now();
    let mut index = DiskIndex::open(&path, &DiskOpenOptions { direct, io })?;
    let open_s = t0.elapsed().as_secs_f64();
    let (rss1, _) = memory_usage();
    let (ram, _) = index.ram_bytes();
    let n = index.len();
    println!(
        "index    {}: {} vectors ({} nodes) x {} ({}), {} layout, {} vectors, PQ {} B, R {}; {:.2} GB on SSD",
        path.display(),
        n,
        index.nodes(),
        index.dim(),
        index.metric().name(),
        index.layout().name(),
        index.codec().name(),
        index.pq_code_size(),
        index.max_degree(),
        index.file_bytes() as f64 / 1e9
    );
    let probe = index.context()?;
    println!(
        "open     {open_s:.2} s; RAM: index {:.1} MB ({:.1} B/vector), process RSS +{:.1} MB; reads: {} B graph blocks{}, {}, {}",
        ram as f64 / 1e6,
        ram as f64 / n as f64,
        rss1.saturating_sub(rss0) as f64 / 1e6,
        index.block_bytes(),
        if index.vector_block_bytes() > 0 { format!(" + {} B vector blocks", index.vector_block_bytes()) } else { String::new() },
        probe.io_backend(),
        if index.is_direct() { "O_DIRECT" } else { "BUFFERED (page cache: latency below is NOT the SSD's)" }
    );
    drop(probe);
    if a.num("verify", false)? {
        let t0 = Instant::now();
        index.verify()?;
        println!("verify   every record and vector checked in {:.1} s", t0.elapsed().as_secs_f64());
    }
    let cache: usize = a.num("cache", 0)?;
    if cache > 0 {
        let t0 = Instant::now();
        let got = index.cache_hot_nodes(cache)?;
        println!(
            "cache    {got} nodes nearest the entry, {:.1} MB, loaded in {:.1} s",
            index.ram_bytes().1 as f64 / 1e6,
            t0.elapsed().as_secs_f64()
        );
    }
    let queries = load_vectors(&a.req_path("query")?, a.opt_num("nq")?)?;
    check_query_dim(&queries, index.dim())?;
    let gt = load_gt(&a.req_path("gt")?, queries.len(), k)?;
    let index = &index;

    let counters: [AtomicUsize; 6] = Default::default(); // reads, bytes, rounds, hops, cache hits, pq dists
    let run = |p: &DiskSearchParams, qs: &Vecs<f32>, t: usize| {
        let search = |q: &[f32]| {
            DISK_CTX.with_borrow_mut(|slot| {
                if slot.is_none() {
                    *slot = Some(index.context()?);
                }
                let (res, st) = index.search_with(slot.as_mut().expect("set above"), q, k, p)?;
                for (c, v) in counters.iter().zip([st.reads, st.read_bytes, st.rounds, st.expanded, st.cache_hits, st.pq_distances]) {
                    c.fetch_add(v, Relaxed);
                }
                Ok(res)
            })
        };
        run_queries(&search, qs, t)
    };

    // Warm-up: touch the PQ codes and ids once (they are in RAM, but cold).
    let warm = a.num("warmup", 200usize)?.min(queries.len());
    if warm > 0 {
        let wq = Vecs { dim: queries.dim, data: queries.data[..warm * queries.dim].to_vec() };
        run(&DiskSearchParams { l: ls[0], beam_width: ws[0], rerank: 0 }, &wq, threads)?;
    }
    println!(
        "{:>3} {:>4} {:>6} {:>9} {:>7} {:>7} {:>8} {:>7} {:>7} {:>8} {:>8} {:>8} {:>7} {:>7} {:>9} {:>6}",
        "W",
        "L",
        "rerank",
        format!("recall@{k}"),
        "reads/q",
        "KB/q",
        "rounds/q",
        "hops/q",
        "cache/q",
        "mean ms",
        "p50 ms",
        "p99 ms",
        "usr us",
        "sys us",
        format!("QPS {threads}t"),
        "dev/app"
    );
    for &w in &ws {
        for &l in &ls {
            for &rerank in &reranks {
                if rerank > 0 && index.layout() == Layout::Inline {
                    continue;
                }
                let p = DiskSearchParams { l, beam_width: w, rerank };
                counters.iter().for_each(|c| c.store(0, Relaxed));
                let dev0 = storage_read_bytes();
                let (u0, s0) = cpu_seconds();
                let single = run(&p, &queries, 1)?;
                let (u1, s1) = cpu_seconds();
                let dev = storage_read_bytes() - dev0;
                let per_q_us = |x: f64| x * 1e6 / queries.len() as f64;
                let c: Vec<f64> = counters.iter().map(|c| c.load(Relaxed) as f64 / queries.len() as f64).collect();
                let app_bytes = c[1] * queries.len() as f64;
                let qps = if threads > 1 { run(&p, &queries, threads)?.qps() } else { single.qps() };
                let mean = single.latencies.iter().map(Duration::as_secs_f64).sum::<f64>() / queries.len() as f64 * 1e3;
                println!(
                    "{:>3} {:>4} {:>6} {:>9.4} {:>7.1} {:>7.0} {:>8.1} {:>7.1} {:>7.1} {:>8.3} {:>8.3} {:>8.3} {:>7.0} {:>7.0} {:>9.0} {:>6.2}",
                    w,
                    l,
                    if index.layout() == Layout::Inline { "-".to_string() } else if rerank == 0 { "all".to_string() } else { rerank.to_string() },
                    recall(&single.results, &gt, k),
                    c[0],
                    c[1] / 1024.0,
                    c[2],
                    c[3],
                    c[4],
                    mean,
                    single.pct_ms(50.0),
                    single.pct_ms(99.0),
                    per_q_us(u1 - u0),
                    per_q_us(s1 - s0),
                    qps,
                    if app_bytes > 0.0 { dev as f64 / app_bytes } else { 0.0 }
                );
            }
        }
    }
    let (_, hwm) = memory_usage();
    println!("latency: 1 thread; reads/q and KB/q: blocks read from the file per query; rounds/q: batches waited on");
    println!("usr/sys us: CPU time per query in user space / in the kernel during the 1-thread run (10 ms ticks)");
    println!("dev/app: bytes the kernel read from storage / bytes the index asked for (about 1.0 = reads hit the SSD)");
    println!("peak RSS {:.1} MB", hwm as f64 / 1e6);
    Ok(())
}
