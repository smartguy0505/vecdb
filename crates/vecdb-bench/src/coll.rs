//! Collection benchmarks: ingest through the WAL, and search with deletes.

use std::path::Path;
use std::time::Instant;

use vecdb_core::collection::NO_ATTRS;
use vecdb_core::disk::IoMode;
use vecdb_core::{Collection, CollectionConfig, Durability, FlatIndex, OpenOptions, SearchOptions, WriteOp};

use super::{AnyResult, Args, load_vectors, memory_usage, parse_list, progress, run_queries};
use vecdb_core::dataset::Vecs;

fn open_options(a: &Args) -> AnyResult<OpenOptions> {
    let durability = match a.map.get("durability").map(String::as_str).unwrap_or("buffered") {
        "sync" => Durability::Sync,
        "buffered" => Durability::Buffered,
        d => return Err(format!("unknown durability {d:?} (sync, buffered)").into()),
    };
    Ok(OpenOptions {
        durability,
        threads: a.num("threads", 6)?,
        build_memory: (a.num::<f64>("build-mem", 2.0)? * (1u64 << 30) as f64) as u64,
        direct_io: a.num("direct", true)?,
        io: IoMode::Auto,
        progress: Some(progress),
    })
}

/// coll-ingest: creates a collection and upserts a vector file into it.
pub(crate) fn ingest(a: &Args) -> AnyResult<()> {
    a.allow(&[
        "base", "dir", "metric", "limit", "batch", "durability", "seal-mb", "index-mb", "max-ssd", "pq-m", "threads",
        "build-mem", "maintain", "optimize", "secs", "direct",
    ])?;
    let base = load_vectors(&a.req_path("base")?, a.opt_num("limit")?)?;
    let dir = a.req_path("dir")?;
    let opts = open_options(a)?;
    let mut cfg = CollectionConfig::new(base.dim, a.metric()?);
    cfg.seal_bytes = (a.num::<f64>("seal-mb", (cfg.seal_bytes >> 20) as f64)? * (1 << 20) as f64) as u64;
    cfg.index_bytes = (a.num::<f64>("index-mb", (cfg.index_bytes >> 20) as f64)? * (1 << 20) as f64) as u64;
    cfg.max_indexed_segments = a.num("max-ssd", cfg.max_indexed_segments)?;
    if let Some(m) = a.opt_num("pq-m")? {
        cfg.index.pq_m = m;
    }
    let batch: usize = a.num("batch", 1000)?;
    let maintain: bool = a.num("maintain", false)?;
    let secs: f64 = a.num("secs", f64::INFINITY)?;
    let mut c = Collection::create(&dir, cfg.clone(), opts.clone())?;
    println!(
        "ingest   {} x {} ({}) into {}: batches of {batch}, {:?}, seal {} MiB, index {} MiB, maintain {maintain}",
        base.len(),
        base.dim,
        cfg.metric.name(),
        dir.display(),
        opts.durability,
        cfg.seal_bytes >> 20,
        cfg.index_bytes >> 20,
    );
    let t0 = Instant::now();
    let mut lat = Vec::new();
    let (mut maintain_s, mut jobs) = (0.0, 0);
    let mut rows = 0;
    while rows < base.len() && t0.elapsed().as_secs_f64() < secs {
        let end = (rows + batch).min(base.len());
        let ops: Vec<WriteOp> =
            (rows..end).map(|i| WriteOp::Upsert { id: i as u64, vector: base.row(i), attrs: &NO_ATTRS }).collect();
        let tb = Instant::now();
        c.write(&ops)?;
        lat.push(tb.elapsed().as_secs_f64() * 1e3);
        if maintain {
            let tm = Instant::now();
            jobs += c.maintain()?.len();
            maintain_s += tm.elapsed().as_secs_f64();
        }
        rows = end;
    }
    let ingest_s = t0.elapsed().as_secs_f64();
    lat.sort_by(f64::total_cmp);
    let pct = |p: f64| lat[((p / 100.0) * (lat.len() - 1) as f64).round() as usize];
    println!(
        "wrote    {rows} rows in {ingest_s:.2} s = {:.0} rows/s ({:.1} MB/s of vectors); batch ms p50 {:.3} p99 {:.3} max {:.1}",
        rows as f64 / ingest_s,
        rows as f64 * base.dim as f64 * 4.0 / ingest_s / 1e6,
        pct(50.0),
        pct(99.0),
        pct(100.0),
    );
    if maintain {
        println!("maintain {jobs} jobs, {maintain_s:.1} s of the ingest time");
    }
    if a.num("optimize", false)? {
        let t = Instant::now();
        let job = c.optimize()?;
        println!("optimize {:.1} s: {job:?}", t.elapsed().as_secs_f64());
    }
    print_stats(&c);
    drop(c);
    let t = Instant::now();
    let (c, report) = Collection::open(&dir, opts)?;
    println!(
        "reopen   {:.2} s: replayed {} batches ({} writes, {:.1} MB of WAL), removed {:?}",
        t.elapsed().as_secs_f64(),
        report.replay.batches,
        report.replay.ops,
        report.replay.bytes as f64 / 1e6,
        report.removed
    );
    drop(c);
    println!("memory   peak RSS {:.2} GB", memory_usage().1 as f64 / 1e9);
    Ok(())
}

fn print_stats(c: &Collection) {
    let st = c.stats();
    let segs: Vec<String> = st.segments.iter().map(|s| format!("{} {} ({} deleted)", s.kind, s.rows, s.deleted)).collect();
    println!(
        "state    {} live; growing {} rows; WAL {:.1} MB; segments: {}",
        st.live,
        st.growing_rows,
        st.wal_bytes as f64 / 1e6,
        if segs.is_empty() { "none".into() } else { segs.join(", ") }
    );
}

fn copy_dir(from: &Path, to: &Path) -> AnyResult<()> {
    if to.exists() {
        std::fs::remove_dir_all(to)?;
    }
    std::fs::create_dir_all(to)?;
    for e in std::fs::read_dir(from)? {
        let e = e?;
        if e.file_name() != "LOCK" {
            std::fs::copy(e.path(), to.join(e.file_name()))?;
        }
    }
    Ok(())
}

/// Exact top-k over the rows of `base` that `live` keeps, ids = row numbers.
fn ground_truth(base: &Vecs<f32>, live: &[bool], metric: vecdb_core::Metric, queries: &Vecs<f32>, k: usize) -> AnyResult<Vec<Vec<u64>>> {
    let mut flat = FlatIndex::with_capacity(base.dim, metric, live.iter().filter(|&&l| l).count())?;
    for (i, _) in live.iter().enumerate().filter(|(_, l)| **l) {
        flat.add(i as u64, base.row(i))?;
    }
    let run = run_queries(&|q| flat.search(q, k), queries, 6)?;
    Ok(run.results.into_iter().map(|r| r.into_iter().map(|x| x.id).collect()).collect())
}

/// coll-search: copies a collection, deletes growing random fractions of
/// its rows, and measures recall (against exact search over the live rows)
/// and latency at each stage; optionally compacts at the end.
pub(crate) fn search(a: &Args) -> AnyResult<()> {
    a.allow(&[
        "from", "dir", "base", "query", "k", "l", "w", "nq", "delete", "seed", "compact", "adapt", "threads",
        "build-mem", "direct", "limit",
    ])?;
    let (from, dir) = (a.req_path("from")?, a.req_path("dir")?);
    let base = load_vectors(&a.req_path("base")?, a.opt_num("limit")?)?;
    let nq: usize = a.num("nq", 1000)?;
    let queries = load_vectors(&a.req_path("query")?, Some(nq))?;
    let k: usize = a.num("k", 10)?;
    let ls: Vec<usize> = parse_list(a, "l", "40,80,160")?;
    let w: usize = a.num("w", 4)?;
    let fractions: Vec<f64> = parse_list(a, "delete", "0")?;
    let adapt: Vec<bool> = match a.map.get("adapt").map(String::as_str).unwrap_or("true") {
        "both" => vec![false, true],
        v => vec![v.parse()?],
    };
    copy_dir(&from, &dir)?;
    let opts = open_options(a)?;
    let (mut c, _) = Collection::open(&dir, opts)?;
    let metric = c.config().metric;
    print_stats(&c);

    let n = base.len();
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = vecdb_core::rng::SplitMix64::new(a.num("seed", 1)?);
    for i in (1..n).rev() {
        order.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
    }
    let mut live = vec![true; n];
    let mut deleted = 0;
    let mut stages: Vec<(String, f64)> = fractions.iter().map(|&f| (format!("{:.0}% deleted", f * 100.0), f)).collect();
    if a.num("compact", false)? {
        stages.push(("after compaction".into(), *fractions.last().unwrap_or(&0.0)));
    }
    println!("  stage               adapt    L  recall@{k}  scanned/q  reads/q  rounds/q  expanded/q  mean ms  p50 ms  p99 ms");
    for (si, (name, f)) in stages.iter().enumerate() {
        let target = (f * n as f64).round() as usize;
        if deleted < target {
            let ids: Vec<u64> = order[deleted..target].iter().map(|&i| i as u64).collect();
            for chunk in ids.chunks(10_000) {
                let ops: Vec<WriteOp> = chunk.iter().map(|&id| WriteOp::Delete { id }).collect();
                c.write(&ops)?;
            }
            order[deleted..target].iter().for_each(|&i| live[i] = false);
            deleted = target;
        }
        if si == fractions.len() {
            let t = Instant::now();
            let jobs = c.maintain()?;
            println!("  maintain {:.1} s: {:?}", t.elapsed().as_secs_f64(), jobs.iter().map(|j| (j.reason, j.rows)).collect::<Vec<_>>());
            print_stats(&c);
        }
        let gt = ground_truth(&base, &live, metric, &queries, k)?;
        // Warm-up: one pass at the largest L, so every row reads warm data.
        let warm = SearchOptions { l: *ls.iter().max().unwrap_or(&100), beam_width: w, ..Default::default() };
        for i in 0..queries.len() {
            c.search_with_stats(queries.row(i), k, &warm)?;
        }
        for &ad in &adapt {
            for &l in &ls {
                let so = SearchOptions { l, beam_width: w, rerank: 0, adapt_l: ad };
                let t0 = Instant::now();
                let mut lat = Vec::with_capacity(queries.len());
                let (mut hits, mut reads, mut rounds, mut expanded, mut scanned) = (0usize, 0usize, 0usize, 0usize, 0usize);
                for (i, want) in gt.iter().enumerate() {
                    let tq = Instant::now();
                    let (res, st) = c.search_with_stats(queries.row(i), k, &so)?;
                    lat.push(tq.elapsed().as_secs_f64() * 1e3);
                    hits += res.iter().filter(|r| want.contains(&r.id)).count();
                    reads += st.reads;
                    rounds += st.rounds;
                    expanded += st.expanded;
                    scanned += st.scanned;
                }
                let total = t0.elapsed().as_secs_f64();
                lat.sort_by(f64::total_cmp);
                let q = queries.len() as f64;
                let pct = |p: f64| lat[((p / 100.0) * (lat.len() - 1) as f64).round() as usize];
                println!(
                    "  {name:18} {ad:6} {l:4}  {:9.4}  {:9.0}  {:7.1}  {:8.1}  {:10.1}  {:7.3}  {:6.3}  {:6.3}",
                    hits as f64 / (q * k as f64),
                    scanned as f64 / q,
                    reads as f64 / q,
                    rounds as f64 / q,
                    expanded as f64 / q,
                    total * 1e3 / q,
                    pct(50.0),
                    pct(99.0),
                );
            }
        }
    }
    drop(c);
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
