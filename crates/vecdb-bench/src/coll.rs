//! Collection benchmarks: ingest through the WAL, and search with deletes.

use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use vecdb_core::collection::NO_ATTRS;
use vecdb_core::disk::IoMode;
use vecdb_core::rng::SplitMix64;
use vecdb_core::{
    Attrs, Collection, CollectionConfig, CollectionStats, Database, DatabaseOptions, Durability, Error, Field, Filter,
    FilterMode, FlatIndex, Metric, Op, OpenOptions, Row, SearchOptions, SearchPool, ShardedCollection, WriteOp,
};

use super::{AnyResult, Args, load_vectors, memory_usage, parse_list, progress, run_queries};
use vecdb_core::dataset::Vecs;

fn open_options(a: &Args) -> AnyResult<OpenOptions> {
    let durability = match a.map.get("durability").map(String::as_str).unwrap_or("buffered") {
        "sync" => Durability::Sync,
        "buffered" => Durability::Buffered,
        d => return Err(format!("unknown durability {d:?} (sync, buffered)").into()),
    };
    let pool: usize = a.num("pool", 0)?;
    Ok(OpenOptions {
        durability,
        background: a.num("background", false)?,
        max_unindexed_bytes: (a.num::<f64>("unindexed-mb", 1024.0)? * (1 << 20) as f64) as u64,
        max_write_wait: Some(a.num("write-wait", 0.0)?).filter(|&s| s > 0.0).map(Duration::from_secs_f64),
        builds: None,
        pool: if pool > 0 { Some(SearchPool::new(pool, a.num("pool-cpus", 0)?)?) } else { None },
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
        "build-mem", "maintain", "optimize", "secs", "direct", "labels", "writers", "background", "unindexed-mb",
        "rate", "trace", "shards", "builds", "write-wait",
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
    let with_labels: bool = a.num("labels", false)?;
    let labels = if with_labels {
        cfg.fields = vec![Field::int("u"), Field::int("p")];
        Some(Labels::new(&base))
    } else {
        None
    };
    let batch: usize = a.num("batch", 1000)?;
    let maintain: bool = a.num("maintain", false)?;
    let secs: f64 = a.num("secs", f64::INFINITY)?;
    let writers: usize = a.num("writers", 1)?;
    // Rows per second over all writers; 0: as fast as they go.
    let rate: f64 = a.num("rate", 0.0)?;
    // A file for what there is every 0.1 s, and what batches take.
    let mut trace = match a.map.get("trace") {
        Some(p) => Some(std::io::BufWriter::new(std::fs::File::create(p)?)),
        None => None,
    };
    if writers > 1 && maintain {
        return Err("--maintain runs with one writer only".into());
    }
    // 0: not a database.
    let shards: usize = a.num("shards", 0)?;
    let optimize: bool = a.num("optimize", false)?;
    if shards > 0 && (maintain || optimize || with_labels) {
        return Err("--shards goes without --maintain, --optimize and --labels".into());
    }
    let db_opts = DatabaseOptions { shard: opts.clone(), builds: a.num("builds", 1)?, ..DatabaseOptions::default() };
    let db = if shards > 0 { Some(Database::open(&dir, &db_opts)?.0) } else { None };
    let target = match &db {
        Some(db) => Target::Sharded(db.create_collection("ingest", cfg.clone(), shards)?),
        None => Target::One(Collection::create(&dir, cfg.clone(), opts.clone())?),
    };
    println!(
        "ingest   {} x {} ({}) into {}: batches of {batch}, {writers} writers, {:?}, seal {} MiB, index {} MiB, maintain {maintain}, background {}, shards {shards}, builds at once {}",
        base.len(),
        base.dim,
        cfg.metric.name(),
        dir.display(),
        opts.durability,
        cfg.seal_bytes >> 20,
        cfg.index_bytes >> 20,
        opts.background,
        db_opts.builds,
    );
    let write_rows = |start: usize, end: usize| -> vecdb_core::Result<()> {
        let c = match &target {
            Target::One(c) => c,
            Target::Sharded(c) => {
                let row = |i: usize| Row { id: i as u64, vector: base.row(i).to_vec(), attrs: Attrs::new() };
                return c.write((start..end).map(|i| Op::Upsert(row(i))).collect()).map(|_| ());
            }
        };
        let attrs: Vec<Attrs> = match &labels {
            Some(lb) => (start..end).map(|i| Attrs::new().int("u", lb.u[i]).int("p", lb.p[i])).collect(),
            None => Vec::new(),
        };
        let ops: Vec<WriteOp> = (start..end)
            .map(|i| WriteOp::Upsert {
                id: i as u64,
                vector: base.row(i),
                attrs: attrs.get(i - start).unwrap_or(&NO_ATTRS),
            })
            .collect();
        c.write(&ops)
    };
    let seq_before: u64 = target.stats().iter().map(|s| s.next_seq).sum();
    let t0 = Instant::now();
    let mut lat = Vec::new();
    let (mut maintain_s, mut jobs) = (0.0, 0);
    // Writers take the next batch of rows until all are written (or time
    // is up); one writer goes through them in order.
    let next = AtomicUsize::new(0);
    let written = AtomicUsize::new(0);
    let ended = AtomicBool::new(false);
    let peak = AtomicUsize::new(0);
    let refused = AtomicUsize::new(0);
    let lats = std::thread::scope(|s| {
        // What is in no SSD index, meanwhile.
        s.spawn(|| {
            while !ended.load(Ordering::Relaxed) {
                let all = target.stats();
                let sum = |f: &dyn Fn(&CollectionStats) -> usize| all.iter().map(f).sum::<usize>();
                let flat = sum(&|s| s.segments.iter().filter(|s| s.kind == "flat").map(|s| s.rows).sum());
                let (frozen, growing) = (sum(&|s| s.frozen_rows), sum(&|s| s.growing_rows));
                peak.fetch_max(flat + frozen + growing, Ordering::Relaxed);
                if let Some(t) = &mut trace {
                    let _ = writeln!(
                        t,
                        "at {:.3} rows {} flat {flat} frozen {frozen} growing {growing} rate {:?} wait {:.3} builds {} flushes {}",
                        t0.elapsed().as_secs_f64(),
                        written.load(Ordering::Relaxed),
                        all.iter().map(|s| s.write_rate).collect::<Vec<_>>(),
                        all.iter().map(|s| s.write_wait.as_secs_f64()).sum::<f64>(),
                        sum(&|s| s.builds as usize),
                        sum(&|s| s.flushes as usize),
                    );
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        });
        let handles: Vec<_> = (0..writers)
            .map(|_| {
                s.spawn(|| -> Result<(Vec<(f64, f64)>, usize, f64), String> {
                    let (mut lat, mut jobs, mut maintain_s) = (Vec::new(), 0, 0.0);
                    while t0.elapsed().as_secs_f64() < secs {
                        let start = next.fetch_add(batch, Ordering::Relaxed);
                        if start >= base.len() {
                            break;
                        }
                        let end = (start + batch).min(base.len());
                        if rate > 0.0 {
                            let due = start as f64 / rate - t0.elapsed().as_secs_f64();
                            std::thread::sleep(Duration::from_secs_f64(due.max(0.0)));
                        }
                        let tb = Instant::now();
                        loop {
                            match write_rows(start, end) {
                                Ok(()) => break,
                                // As a client does that is told to retry.
                                Err(Error::Unavailable(_)) if opts.max_write_wait.is_some() => {
                                    refused.fetch_add(1, Ordering::Relaxed);
                                    std::thread::sleep(Duration::from_secs(1));
                                }
                                Err(e) => return Err(e.to_string()),
                            }
                        }
                        lat.push((tb.duration_since(t0).as_secs_f64(), tb.elapsed().as_secs_f64() * 1e3));
                        written.fetch_add(end - start, Ordering::Relaxed);
                        if let (true, Target::One(c)) = (maintain, &target) {
                            let tm = Instant::now();
                            jobs += c.maintain().map_err(|e| e.to_string())?.len();
                            maintain_s += tm.elapsed().as_secs_f64();
                        }
                    }
                    Ok((lat, jobs, maintain_s))
                })
            })
            .collect();
        let lats = handles.into_iter().map(|h| h.join()).collect::<Vec<_>>();
        ended.store(true, Ordering::Relaxed);
        lats.into_iter().map(|l| l.expect("writer thread")).collect::<Vec<_>>()
    });
    let rows = written.load(Ordering::Relaxed);
    let ingest_s = t0.elapsed().as_secs_f64();
    for l in lats {
        let (l, j, m) = l?;
        if let Some(t) = &mut trace {
            l.iter().try_for_each(|(at, ms)| writeln!(t, "batch {at:.3} {ms:.3}"))?;
        }
        lat.extend(l.into_iter().map(|(_, ms)| ms));
        jobs += j;
        maintain_s += m;
    }
    lat.sort_by(f64::total_cmp);
    let pct = |p: f64| lat[((p / 100.0) * (lat.len() - 1) as f64).round() as usize];
    println!(
        "wrote    {rows} rows in {ingest_s:.2} s = {:.0} rows/s ({:.1} MB/s of vectors); batch ms p50 {:.3} p99 {:.3} p99.9 {:.1} max {:.1}",
        rows as f64 / ingest_s,
        rows as f64 * base.dim as f64 * 4.0 / ingest_s / 1e6,
        pct(50.0),
        pct(99.0),
        pct(99.9),
        pct(100.0),
    );
    let over = |ms: f64| lat.len() - lat.partition_point(|&l| l <= ms);
    let sum = |f: &dyn Fn(&CollectionStats) -> u64| target.stats().iter().map(f).sum::<u64>();
    println!(
        "waits    batches over 0.1 / 1 / 10 s: {} / {} / {}; refused {} times; most in no index: {} rows = {:.0} MiB; each row applied {:.2} times",
        over(100.0),
        over(1000.0),
        over(10_000.0),
        refused.load(Ordering::Relaxed),
        peak.load(Ordering::Relaxed),
        peak.load(Ordering::Relaxed) as f64 * base.dim as f64 * 4.0 / (1 << 20) as f64,
        (sum(&|s| s.next_seq) - seq_before) as f64 / rows.max(1) as f64,
    );
    if opts.background {
        let (flushes, builds) = (sum(&|s| s.flushes), sum(&|s| s.builds));
        let t = Instant::now();
        match &target {
            Target::One(c) => c.wait_idle()?,
            Target::Sharded(c) => c.wait_idle()?,
        }
        println!(
            "idle     after {:.1} s more; {flushes} flushes and {builds} builds by the end of the ingest, then {} and {}",
            t.elapsed().as_secs_f64(),
            sum(&|s| s.flushes) - flushes,
            sum(&|s| s.builds) - builds,
        );
    }
    let (batches, groups) = (sum(&|s| s.write_batches), sum(&|s| s.write_groups));
    println!(
        "wal      {batches} batches in {groups} writes, {} fsyncs: {:.1} batches per write",
        sum(&|s| s.wal_syncs),
        batches as f64 / groups.max(1) as f64
    );
    if maintain {
        println!("maintain {jobs} jobs, {maintain_s:.1} s of the ingest time");
    }
    if let (true, Target::One(c)) = (optimize, &target) {
        let t = Instant::now();
        let job = c.optimize()?;
        println!("optimize {:.1} s: {job:?}", t.elapsed().as_secs_f64());
    }
    target.stats().iter().for_each(print_state);
    drop((target, db));
    let t = Instant::now();
    let reports = if shards > 0 {
        Database::open(&dir, &db_opts)?.1.opened.remove(0).1
    } else {
        vec![Collection::open(&dir, opts)?.1]
    };
    for report in reports {
        println!(
            "reopen   {:.2} s: replayed {} batches ({} writes, {:.1} MB of WAL), removed {:?}",
            t.elapsed().as_secs_f64(),
            report.replay.batches,
            report.replay.ops,
            report.replay.bytes as f64 / 1e6,
            report.removed
        );
    }
    println!("memory   peak RSS {:.2} GB", memory_usage().1 as f64 / 1e9);
    Ok(())
}

/// What an ingest goes into.
enum Target {
    One(Collection),
    Sharded(Arc<ShardedCollection>),
}

impl Target {
    fn stats(&self) -> Vec<CollectionStats> {
        match self {
            Target::One(c) => vec![c.stats()],
            Target::Sharded(c) => c.stats().expect("not dropped"),
        }
    }
}

fn print_stats(c: &Collection) {
    print_state(&c.stats());
}

fn print_state(st: &CollectionStats) {
    let segs: Vec<String> = st.segments.iter().map(|s| format!("{} {} ({} deleted)", s.kind, s.rows, s.deleted)).collect();
    println!(
        "state    {} live; growing {} rows; frozen {}; WAL {:.1} MB; segments: {}",
        st.live,
        st.growing_rows,
        st.frozen,
        st.wal_bytes as f64 / 1e6,
        if segs.is_empty() { "none".into() } else { segs.join(", ") }
    );
    let (bytes, values): (usize, usize) = st.segments.iter().map(|s| (s.attr_bytes, s.attr_values)).fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    if values > 0 {
        println!(
            "attrs    {values} values in sealed segments: {:.1} MB of columns in RAM ({:.2} bytes per value); growing {:.1} MB",
            bytes as f64 / 1e6,
            bytes as f64 / values as f64,
            st.growing_attr_bytes as f64 / 1e6
        );
    }
}

/// Bench labels per row. "u": uniform in 0..1000, unrelated to the vector.
/// "p": the row's bucket (0..1000) by the rank of its projection onto a
/// fixed random direction, so rows with nearby "p" lie near each other
/// along that direction.
struct Labels {
    u: Vec<i64>,
    p: Vec<i64>,
    dir: Vec<f32>,
    /// The rows' projections, ascending.
    sorted: Vec<f32>,
}

const BUCKETS: i64 = 1000;

impl Labels {
    fn new(base: &Vecs<f32>) -> Self {
        let n = base.len();
        let u = (0..n).map(|i| (SplitMix64::new(i as u64 ^ 0x5eed_0001).next_u64() % BUCKETS as u64) as i64).collect();
        let mut rng = SplitMix64::new(0x5eed_0002);
        let dir: Vec<f32> = (0..base.dim).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let proj: Vec<f32> = (0..n).map(|i| dot(base.row(i), &dir)).collect();
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by(|&a, &b| proj[a as usize].total_cmp(&proj[b as usize]).then(a.cmp(&b)));
        let mut p = vec![0i64; n];
        for (r, &i) in order.iter().enumerate() {
            p[i as usize] = r as i64 * BUCKETS / n as i64;
        }
        let sorted = order.iter().map(|&i| proj[i as usize]).collect();
        Self { u, p, dir, sorted }
    }

    /// The bucket a query falls in.
    fn bucket(&self, q: &[f32]) -> i64 {
        let r = self.sorted.partition_point(|&x| x < dot(q, &self.dir));
        (r as i64 * BUCKETS / self.sorted.len() as i64).min(BUCKETS - 1)
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    (vecdb_core::kernels::best().dot)(a, b)
}

pub(crate) fn copy_dir(from: &Path, to: &Path) -> AnyResult<()> {
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
pub(crate) fn ground_truth(base: &Vecs<f32>, live: &[bool], metric: vecdb_core::Metric, queries: &Vecs<f32>, k: usize) -> AnyResult<Vec<Vec<u64>>> {
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
        "build-mem", "direct", "limit", "pool", "pool-cpus",
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
    let (c, _) = Collection::open(&dir, opts)?;
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
                let so = SearchOptions { l, beam_width: w, rerank: 0, adapt_l: ad, ..Default::default() };
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

/// One filter of the coll-filter sweep, for one query: buckets `lo..=hi`
/// of label "u" or "p".
fn bench_filter(kind: &str, width: i64, q: &[f32], lb: &Labels) -> AnyResult<(&'static str, i64, i64)> {
    Ok(match kind {
        // Uniform labels: matches spread over the whole space.
        "u" => ("u", 0, width - 1),
        // A fixed slab at one end of the projection: matches far from most
        // queries (the hard case for a graph walk).
        "p" => ("p", 0, width - 1),
        // A slab around the query's own projection: matches near it.
        "near" => {
            let lo = (lb.bucket(q) - width / 2).clamp(0, BUCKETS - width);
            ("p", lo, lo + width - 1)
        }
        k => return Err(format!("unknown --filter {k:?} (none, u, p, near)").into()),
    })
}

/// coll-filter: searches a collection ingested with --labels under filters
/// of growing selectivity, in each filter mode, against exact search over
/// the matching rows.
pub(crate) fn filter(a: &Args) -> AnyResult<()> {
    a.allow(&["dir", "base", "query", "k", "l", "w", "nq", "filter", "select", "mode", "threads", "direct", "limit", "warm", "pool", "pool-cpus"])?;
    let dir = a.req_path("dir")?;
    let base = load_vectors(&a.req_path("base")?, a.opt_num("limit")?)?;
    let queries = load_vectors(&a.req_path("query")?, Some(a.num("nq", 300)?))?;
    let k: usize = a.num("k", 10)?;
    let ls: Vec<usize> = parse_list(a, "l", "40,80,160")?;
    let w: usize = a.num("w", 4)?;
    let kinds: Vec<String> = parse_list(a, "filter", "u,p,near")?;
    let selects: Vec<f64> = parse_list(a, "select", "0.001,0.01,0.05,0.2,0.5")?;
    let modes: Vec<FilterMode> = parse_list::<String>(a, "mode", "graph,scan,auto")?
        .iter()
        .map(|m| match m.as_str() {
            "graph" => Ok(FilterMode::Graph),
            "scan" => Ok(FilterMode::Scan),
            "auto" => Ok(FilterMode::Auto),
            m => Err(format!("unknown --mode {m:?}")),
        })
        .collect::<Result<_, _>>()?;
    let warm: bool = a.num("warm", true)?;
    let (c, _) = Collection::open(&dir, open_options(a)?)?;
    print_stats(&c);
    if c.config().metric != Metric::L2 {
        return Err("coll-filter computes ground truth for l2 only".into());
    }
    let lb = Labels::new(&base);
    let kern = vecdb_core::kernels::best();
    let threads: usize = a.num("threads", 6)?;
    println!("  filter  select  mode     L  recall@{k}  matched/q  reads/q  rounds/q   pq/q  fallback  mean ms  p50 ms  p99 ms");
    for kind in &kinds {
        // "none": no filter, the baseline (once, whatever --select says).
        let none = kind == "none";
        let selects = if none { vec![1.0] } else { selects.clone() };
        for &sel in &selects {
            let width = ((sel * BUCKETS as f64).round() as i64).clamp(1, BUCKETS);
            let filters: Vec<Option<Filter>> = (0..queries.len())
                .map(|i| match none {
                    true => Ok(None),
                    false => bench_filter(kind, width, queries.row(i), &lb).map(|(f, lo, hi)| Some(Filter::range(f, lo..=hi))),
                })
                .collect::<AnyResult<_>>()?;
            // Exact top-k over the matching rows, per query (threads).
            let label = if kind == "u" { &lb.u } else { &lb.p };
            let nq = queries.len();
            let per = nq.div_ceil(threads);
            let gt: Vec<Vec<u64>> = std::thread::scope(|s| {
                let hs: Vec<_> = (0..threads)
                    .map(|t| {
                        let (filters, label) = (&filters, label);
                        let base = &base;
                        let queries = &queries;
                        s.spawn(move || {
                            (t * per..((t + 1) * per).min(nq))
                                .map(|qi| {
                                    let (min, max) = match &filters[qi] {
                                        Some(Filter::Range { min, max, .. }) => (*min, *max),
                                        None => (i64::MIN, i64::MAX),
                                        Some(_) => unreachable!(),
                                    };
                                    let q = queries.row(qi);
                                    let mut top = vecdb_core::topk::TopK::new(k);
                                    for (i, &v) in label.iter().enumerate() {
                                        if (min..=max).contains(&v) {
                                            top.push(i as u64, (kern.l2_squared)(q, base.row(i)));
                                        }
                                    }
                                    top.into_sorted().into_iter().map(|r| r.id).collect::<Vec<u64>>()
                                })
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect();
                hs.into_iter().flat_map(|h| h.join().expect("ground truth thread")).collect()
            });
            if warm {
                let l = *ls.iter().max().unwrap_or(&100);
                for mode in [FilterMode::Graph, FilterMode::Scan] {
                    for (i, f) in filters.iter().enumerate() {
                        let so = SearchOptions { l, beam_width: w, filter: f.clone(), filter_mode: mode, ..Default::default() };
                        c.search_with_stats(queries.row(i), k, &so)?;
                    }
                }
            }
            for &mode in &modes {
                for &l in &ls {
                    let mut lat = Vec::with_capacity(nq);
                    let (mut hits, mut want_n) = (0usize, 0usize);
                    let mut tot = vecdb_core::collection::SearchStats::default();
                    let t0 = Instant::now();
                    for (i, want) in gt.iter().enumerate() {
                        let so = SearchOptions { l, beam_width: w, filter: filters[i].clone(), filter_mode: mode, ..Default::default() };
                        let tq = Instant::now();
                        let (res, st) = c.search_with_stats(queries.row(i), k, &so)?;
                        lat.push(tq.elapsed().as_secs_f64() * 1e3);
                        hits += res.iter().filter(|r| want.contains(&r.id)).count();
                        want_n += want.len();
                        tot.matched += st.matched;
                        tot.reads += st.reads;
                        tot.rounds += st.rounds;
                        tot.pq_scanned += st.pq_scanned;
                        tot.fallbacks += st.fallbacks;
                    }
                    let total = t0.elapsed().as_secs_f64();
                    lat.sort_by(f64::total_cmp);
                    let pct = |p: f64| lat[((p / 100.0) * (lat.len() - 1) as f64).round() as usize];
                    let q = nq as f64;
                    println!(
                        "  {kind:6}  {sel:6}  {:6} {l:4}  {:9.4}  {:9.0}  {:7.1}  {:8.1}  {:5.0}  {:8.2}  {:7.3}  {:6.3}  {:6.3}",
                        format!("{mode:?}").to_lowercase(),
                        hits as f64 / want_n.max(1) as f64,
                        tot.matched as f64 / q,
                        tot.reads as f64 / q,
                        tot.rounds as f64 / q,
                        tot.pq_scanned as f64 / q,
                        tot.fallbacks as f64 / q,
                        total * 1e3 / q,
                        pct(50.0),
                        pct(99.0),
                    );
                }
            }
        }
    }
    Ok(())
}

/// What one coll-concurrent measurement needs.
struct Bench<'a> {
    c: &'a Collection,
    base: &'a Vecs<f32>,
    /// Rows not deleted by --churn: writers upsert only these.
    live: &'a [bool],
    queries: &'a Vecs<f32>,
    gt: &'a [Vec<u64>],
    so: &'a SearchOptions,
    k: usize,
    secs: f64,
    write_batch: usize,
    /// Rows per second over all writers; 0: as fast as they go.
    write_rate: f64,
    /// Searches per second over all readers; 0: as fast as they go.
    read_rate: f64,
}

/// coll-concurrent: copies a collection, then searches it from several
/// threads at once for a few seconds each: alone, with writers upserting
/// rows (the same vectors again, so the ground truth holds), and while an
/// index build runs.
pub(crate) fn concurrent(a: &Args) -> AnyResult<()> {
    a.allow(&[
        "from", "dir", "base", "query", "k", "l", "w", "nq", "readers", "secs", "writers", "write-batch",
        "write-rate", "durability", "churn", "growing", "build", "windows", "threads", "build-mem", "direct",
        "limit", "pool", "pool-cpus", "background", "unindexed-mb", "read-rate",
    ])?;
    let (from, dir) = (a.req_path("from")?, a.req_path("dir")?);
    let base = load_vectors(&a.req_path("base")?, a.opt_num("limit")?)?;
    let queries = load_vectors(&a.req_path("query")?, Some(a.num("nq", 1000)?))?;
    let k: usize = a.num("k", 10)?;
    let so = SearchOptions { l: a.num("l", 40)?, beam_width: a.num("w", 4)?, ..Default::default() };
    let readers: Vec<usize> = parse_list(a, "readers", "1,2,4,8")?;
    let writers: usize = a.num("writers", 0)?;
    let churn: f64 = a.num("churn", 0.0)?;
    copy_dir(&from, &dir)?;
    let (c, _) = Collection::open(&dir, open_options(a)?)?;
    print_stats(&c);
    let n = base.len();
    let mut live = vec![true; n];
    if churn > 0.0 {
        // Deletes of random rows: tombstones (a compaction falls due at
        // 20%), and no rows left to scan in full.
        let mut rng = SplitMix64::new(7);
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            order.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
        }
        let gone = &order[..(churn * n as f64) as usize];
        for chunk in gone.chunks(10_000) {
            let ops: Vec<WriteOp> = chunk.iter().map(|&i| WriteOp::Delete { id: i as u64 }).collect();
            c.write(&ops)?;
        }
        c.flush()?;
        gone.iter().for_each(|&i| live[i] = false);
        println!("churn    deleted {} random rows", gone.len());
        print_stats(&c);
    }
    let gt = ground_truth(&base, &live, c.config().metric, &queries, k)?;
    for i in 0..queries.len() {
        c.search(queries.row(i), k, &so)?;
    }
    let growing: usize = a.num("growing", 0)?;
    if growing > 0 {
        // Rows left in the growing segment: what searches scan in full.
        let mut rng = SplitMix64::new(8);
        let ids: Vec<usize> = (0..growing).map(|_| (rng.next_u64() % n as u64) as usize).filter(|&i| live[i]).collect();
        for chunk in ids.chunks(1000) {
            let ops: Vec<WriteOp> =
                chunk.iter().map(|&i| WriteOp::Upsert { id: i as u64, vector: base.row(i), attrs: &NO_ATTRS }).collect();
            c.write(&ops)?;
        }
        print_stats(&c);
    }
    let b = Bench {
        c: &c,
        base: &base,
        live: &live,
        queries: &queries,
        gt: &gt,
        so: &so,
        k,
        secs: a.num("secs", 5.0)?,
        write_batch: a.num("write-batch", 100)?,
        write_rate: a.num("write-rate", 0.0)?,
        read_rate: a.num("read-rate", 0.0)?,
    };
    println!(
        "  phase   readers writers      QPS  recall@{k}  mean ms  p50 ms  p99 ms  cpu ms/q  write rows/s  batch p50 ms  p99 ms  max ms  batches/write  flushes  builds"
    );
    for &r in &readers {
        measure(&b, if writers > 0 { "writes" } else { "alone" }, r, writers)?;
    }
    if a.num("build", false)? {
        // Searches while maintain() builds what is due, in windows of
        // --secs, then once it is done.
        let r = *readers.last().unwrap_or(&1);
        let windows: usize = a.num("windows", 3)?;
        let t = Instant::now();
        let jobs = std::thread::scope(|s| -> AnyResult<_> {
            let job = s.spawn(|| c.maintain().map_err(|e| e.to_string()));
            let mut done = 0;
            while !job.is_finished() && done < windows {
                measure(&b, "build", r, 0)?;
                done += 1;
            }
            Ok(job.join().expect("maintain thread")?)
        })?;
        let summary: Vec<String> = jobs.iter().map(|j| format!("{} {} rows {:.1} s", j.reason, j.rows, j.seconds)).collect();
        println!("  maintain {:.1} s: {}", t.elapsed().as_secs_f64(), summary.join(", "));
        measure(&b, "after", r, 0)?;
    }
    print_stats(&c);
    drop(c);
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

/// CPU time this process has used, all threads (0 where there is no /proc).
fn cpu_seconds() -> f64 {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    // Fields after the name, which ends with the last ')': utime and stime
    // are the 12th and 13th, in ticks of 10 ms.
    let rest = stat.rsplit(')').next().unwrap_or_default();
    rest.split_whitespace().skip(11).take(2).filter_map(|x| x.parse::<f64>().ok()).sum::<f64>() / 100.0
}

/// One measurement: `readers` threads search (each through the queries
/// from its own offset) and `writers` threads upsert, for `secs` seconds.
fn measure(b: &Bench, phase: &str, readers: usize, writers: usize) -> AnyResult<()> {
    let before = b.c.stats();
    let cpu0 = cpu_seconds();
    let t0 = Instant::now();
    let done = AtomicBool::new(false);
    let nq = b.queries.len();
    let n = b.base.len();
    let (reads, writes) = std::thread::scope(|s| {
        let rh: Vec<_> = (0..readers)
            .map(|t| {
                let done = &done;
                s.spawn(move || -> Result<(Vec<f64>, usize, usize), String> {
                    let (mut lat, mut hits, mut want) = (Vec::new(), 0, 0);
                    let mut i = t * nq / readers.max(1);
                    // Each reader its own times, one after the other.
                    let every = if b.read_rate > 0.0 { readers as f64 / b.read_rate } else { 0.0 };
                    let mut due = every * t as f64 / readers as f64;
                    while !done.load(Ordering::Relaxed) {
                        let qi = i % nq;
                        if every > 0.0 {
                            std::thread::sleep(Duration::from_secs_f64((due - t0.elapsed().as_secs_f64()).max(0.0)));
                            due = (due + every).max(t0.elapsed().as_secs_f64());
                        }
                        let tq = Instant::now();
                        let res = b.c.search(b.queries.row(qi), b.k, b.so).map_err(|e| e.to_string())?;
                        lat.push(tq.elapsed().as_secs_f64() * 1e3);
                        hits += res.iter().filter(|r| b.gt[qi].contains(&r.id)).count();
                        want += b.gt[qi].len();
                        i += 1;
                    }
                    Ok((lat, hits, want))
                })
            })
            .collect();
        let wh: Vec<_> = (0..writers)
            .map(|t| {
                let done = &done;
                s.spawn(move || -> Result<(Vec<f64>, usize), String> {
                    let mut rng = SplitMix64::new(100 + t as u64);
                    let per_writer = b.write_rate / writers as f64;
                    let (mut lat, mut rows) = (Vec::new(), 0usize);
                    while !done.load(Ordering::Relaxed) {
                        if per_writer > 0.0 {
                            let due = rows as f64 / per_writer - t0.elapsed().as_secs_f64();
                            if due > 0.0 {
                                std::thread::sleep(Duration::from_secs_f64(due.min(0.02)));
                                continue;
                            }
                        }
                        let ids: Vec<usize> = (0..b.write_batch)
                            .map(|_| (rng.next_u64() % n as u64) as usize)
                            .filter(|&i| b.live[i])
                            .collect();
                        let ops: Vec<WriteOp> = ids
                            .iter()
                            .map(|&i| WriteOp::Upsert { id: i as u64, vector: b.base.row(i), attrs: &NO_ATTRS })
                            .collect();
                        let tb = Instant::now();
                        b.c.write(&ops).map_err(|e| e.to_string())?;
                        lat.push(tb.elapsed().as_secs_f64() * 1e3);
                        rows += ops.len();
                    }
                    Ok((lat, rows))
                })
            })
            .collect();
        std::thread::sleep(Duration::from_secs_f64(b.secs));
        done.store(true, Ordering::Relaxed);
        let reads: Vec<_> = rh.into_iter().map(|h| h.join().expect("reader thread")).collect();
        let writes: Vec<_> = wh.into_iter().map(|h| h.join().expect("writer thread")).collect();
        (reads, writes)
    });
    let secs = t0.elapsed().as_secs_f64();
    let cpu = cpu_seconds() - cpu0;
    let (mut lat, mut hits, mut want) = (Vec::new(), 0, 0);
    for r in reads {
        let (l, h, w) = r?;
        lat.extend(l);
        hits += h;
        want += w;
    }
    let (mut wlat, mut rows) = (Vec::new(), 0);
    for w in writes {
        let (l, r) = w?;
        wlat.extend(l);
        rows += r;
    }
    let pct = |v: &mut Vec<f64>, p: f64| {
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(f64::total_cmp);
        v[((p / 100.0) * (v.len() - 1) as f64).round() as usize]
    };
    let after = b.c.stats();
    let batches = after.write_batches - before.write_batches;
    let groups = after.write_groups - before.write_groups;
    let mean = lat.iter().sum::<f64>() / lat.len().max(1) as f64;
    println!(
        "  {phase:7} {readers:7} {writers:7} {:8.0}  {:9.4}  {:7.3}  {:6.3}  {:6.3}  {:8.2}  {:12.0}  {:12.3}  {:6.3}  {:6.1}  {:13.1}  {:7}  {:6}",
        lat.len() as f64 / secs,
        hits as f64 / want.max(1) as f64,
        mean,
        pct(&mut lat, 50.0),
        pct(&mut lat, 99.0),
        cpu * 1e3 / lat.len().max(1) as f64,
        rows as f64 / secs,
        pct(&mut wlat, 50.0),
        pct(&mut wlat, 99.0),
        pct(&mut wlat, 100.0),
        batches as f64 / groups.max(1) as f64,
        after.flushes - before.flushes,
        after.builds - before.builds,
    );
    Ok(())
}
