//! vecdb-bench: dataset tools and the recall/latency harness.

use std::collections::HashMap;
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{Duration, Instant};

use vecdb_core::dataset::{self, Vecs};
use vecdb_core::format::{read_flat_segment, write_flat_segment};
use vecdb_core::kernels::{self, Kernels};
use vecdb_core::rng::SplitMix64;
use vecdb_core::{FlatIndex, Metric, SearchResult, VectorStore};

type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;

const USAGE: &str = "\
vecdb-bench <command> [--flag value]...

commands:
  synth        --out DIR [--dim 128] [--n 100000] [--nq 1000] [--seed 42]
               write DIR/base.fvecs and DIR/query.fvecs (uniform random)
  groundtruth  --base F --query F --out gt.ivecs [--k 100] [--metric l2] [--threads 4]
               exact k-NN with the flat index (base ids = row numbers)
  build        --base F --out seg.vdb [--metric l2] [--limit N]
               write a flat segment file
  search       (--base F | --segment S) --query F [--gt gt.ivecs] [--k 10]
               [--metric l2] [--threads 4] [--limit N] [--nq N]
               run queries; print QPS, latency percentiles and recall@k
  kernels      [--dims 32,64,...,8192] [--ms 200] [--repeat 5]
               ns per distance for every kernel set this CPU can run (data in cache)

F = .fvecs or .bvecs. Metrics: l2, ip, cosine.
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
    a.allow(&["out", "dim", "n", "nq", "seed"])?;
    let out = a.req_path("out")?;
    let dim: usize = a.num("dim", 128)?;
    let n: usize = a.num("n", 100_000)?;
    let nq: usize = a.num("nq", 1_000)?;
    let mut rng = SplitMix64::new(a.num("seed", 42)?);
    std::fs::create_dir_all(&out)?;
    let base: Vec<f32> = (0..n * dim).map(|_| rng.uniform(-1.0, 1.0)).collect();
    let query: Vec<f32> = (0..nq * dim).map(|_| rng.uniform(-1.0, 1.0)).collect();
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

fn run_queries<S: VectorStore + Sync>(
    index: &FlatIndex<S>,
    queries: &Vecs<f32>,
    k: usize,
    threads: usize,
) -> AnyResult<RunStats> {
    if queries.dim != index.dim() {
        return Err(format!("query dim {} != index dim {}", queries.dim, index.dim()).into());
    }
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
                            let r = index.search(queries.row(i), k)?;
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

fn groundtruth(a: &Args) -> AnyResult<()> {
    a.allow(&["base", "query", "out", "k", "metric", "threads"])?;
    let out = a.req_path("out")?;
    let k: usize = a.num("k", 100)?;
    let metric = a.metric()?;
    let base = load_vectors(&a.req_path("base")?, None)?;
    let queries = load_vectors(&a.req_path("query")?, None)?;
    if base.len() < k {
        return Err(format!("base has {} vectors, fewer than k = {k}", base.len()).into());
    }
    if base.len() > i32::MAX as usize {
        return Err("ivecs ids are i32; base too large".into());
    }
    let index = index_from_base(&base, metric)?;
    let stats = run_queries(&index, &queries, k, a.num("threads", 4)?)?;
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
    let stats = run_queries(&index, &queries, k, threads)?;
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
    let mut lat = stats.latencies.clone();
    lat.sort_unstable();
    let pct = |p: f64| lat[((p / 100.0) * (lat.len() - 1) as f64).round() as usize].as_secs_f64() * 1e3;
    println!(
        "speed    {:.1} QPS; latency ms p50 {:.3}  p95 {:.3}  p99 {:.3}  max {:.3}",
        nq as f64 / stats.wall.as_secs_f64(),
        pct(50.0),
        pct(95.0),
        pct(99.0),
        pct(100.0)
    );

    if let Some(gt_path) = a.path("gt") {
        let gt = dataset::read_ivecs(&gt_path, Some(nq))?;
        if gt.len() < nq || gt.dim < k {
            return Err(format!(
                "ground truth has {} rows x {} ids; need {nq} x {k}",
                gt.len(),
                gt.dim
            )
            .into());
        }
        let mut hits = 0usize;
        for (i, res) in stats.results.iter().enumerate() {
            let truth = &gt.row(i)[..k];
            hits += res.iter().filter(|r| truth.contains(&(r.id as i32))).count();
        }
        println!("recall   recall@{k} = {:.4}", hits as f64 / (nq * k) as f64);
    }
    Ok(())
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
