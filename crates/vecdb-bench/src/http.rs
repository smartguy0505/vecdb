//! The server against the same collection in this process.

use std::fmt::Write as _;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use vecdb_core::dataset::Vecs;
use vecdb_core::{
    Attrs, Collection, CollectionConfig, Database, DatabaseOptions, Durability, OpenOptions, Op, Row, SearchOptions,
    SearchRequest, ShardedCollection,
};

use super::coll::{copy_dir, ground_truth};
use super::{AnyResult, Args, load_vectors, parse_list};

#[derive(Clone, Copy, PartialEq)]
enum How {
    /// Calls in this process.
    Here,
    /// Requests with vectors as arrays of numbers.
    Json,
    /// The same in base64.
    Base64,
}

const HOWS: [(How, &str); 3] = [(How::Here, "in process"), (How::Json, "HTTP"), (How::Base64, "HTTP, base64")];

/// A server process.
struct Server {
    child: Child,
    addr: SocketAddr,
}

impl Server {
    fn start(program: &Path, data: &Path, more: &[String]) -> AnyResult<Self> {
        let mut child = Command::new(program)
            .arg("--data")
            .arg(data)
            .args(["--listen", "127.0.0.1:0"])
            .args(more)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()?;
        let mut log = BufReader::new(child.stderr.take().expect("piped"));
        let addr = loop {
            let mut line = String::new();
            if log.read_line(&mut line)? == 0 {
                return Err("the server did not start".into());
            }
            if let Some(a) = line.trim().strip_prefix("listening on http://") {
                break a.parse()?;
            }
        };
        // (What else it says.)
        std::thread::spawn(move || std::io::copy(&mut log, &mut std::io::stderr()));
        Ok(Self { child, addr })
    }

    /// CPU seconds it has used.
    fn cpu(&self) -> f64 {
        cpu_seconds(&self.child.id().to_string())
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn cpu_seconds(pid: &str) -> f64 {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
    // utime and stime, in ticks of 10 ms.
    let rest = stat.rsplit(')').next().unwrap_or_default();
    rest.split_whitespace().skip(11).take(2).filter_map(|x| x.parse::<f64>().ok()).sum::<f64>() / 100.0
}

/// A connection.
struct Client(BufReader<TcpStream>);

impl Client {
    fn new(addr: SocketAddr) -> std::io::Result<Self> {
        let s = TcpStream::connect(addr)?;
        s.set_nodelay(true)?;
        Ok(Self(BufReader::with_capacity(64 << 10, s)))
    }

    /// Sends a whole request; the reply's body, if it is a success.
    fn ask(&mut self, request: &[u8], body: &mut Vec<u8>) -> AnyResult<()> {
        self.0.get_mut().write_all(request)?;
        let mut line = String::new();
        self.0.read_line(&mut line)?;
        let ok = line.starts_with("HTTP/1.1 2");
        let status = line.trim().to_string();
        let mut length = 0;
        loop {
            line.clear();
            self.0.read_line(&mut line)?;
            let Some((name, value)) = line.trim_end().split_once(':') else { break };
            if name.eq_ignore_ascii_case("content-length") {
                length = value.trim().parse()?;
            }
        }
        body.resize(length, 0);
        self.0.read_exact(body)?;
        match ok {
            true => Ok(()),
            false => Err(format!("{status}: {}", String::from_utf8_lossy(body)).into()),
        }
    }
}

fn request(method: &str, path: &str, body: &str) -> Vec<u8> {
    let n = body.len();
    format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {n}\r\n\r\n{body}"
    )
    .into_bytes()
}

/// As JSON, or as a string of base64.
fn vector_text(out: &mut String, v: &[f32], how: How) {
    if how == How::Json {
        let mut sep = '[';
        for x in v {
            let _ = write!(out, "{sep}{x}");
            sep = ',';
        }
        out.push(']');
        return;
    }
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    out.push('"');
    for chunk in bytes.chunks(3) {
        let mut x = [0; 4];
        x[1..1 + chunk.len()].copy_from_slice(chunk);
        let x = u32::from_be_bytes(x);
        // (Without padding.)
        (0..=chunk.len()).for_each(|i| out.push(ALPHABET[(x >> (18 - 6 * i)) as usize & 63] as char));
    }
    out.push('"');
}

/// The numbers after each `"id":`.
fn ids(body: &[u8]) -> Vec<u64> {
    let text = String::from_utf8_lossy(body);
    text.split("\"id\":")
        .skip(1)
        .map(|s| s[..s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len())].parse().unwrap_or(u64::MAX))
        .collect()
}

/// A database with one collection, "c", of one shard: a copy of `from`.
fn import(from: &Path, dir: &Path) -> AnyResult<PathBuf> {
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    let config = {
        let (c, _) = Collection::open(from, OpenOptions::default())?;
        c.config().clone()
    };
    let (db, _) = Database::open(dir, &DatabaseOptions::default())?;
    db.create_collection("c", config, 1)?;
    drop(db);
    copy_dir(from, &dir.join("col-00000001").join("shard-0000"))?;
    Ok(dir.to_path_buf())
}

/// One measurement.
struct Run {
    per_sec: f64,
    latencies: Vec<Duration>,
    cpu: f64,
    hits: usize,
}

/// Runs `work` on `clients` threads (with their number) until it says there
/// is no more; it returns what it counts.
fn run<T: Send>(
    clients: usize,
    start: impl Fn() -> AnyResult<T> + Sync,
    work: impl Fn(&mut T, usize) -> AnyResult<Option<usize>> + Sync,
) -> AnyResult<(Duration, Vec<Duration>, usize)> {
    let next = AtomicUsize::new(0);
    let t0 = Instant::now();
    let parts: Vec<Result<(Vec<Duration>, usize), String>> = std::thread::scope(|s| {
        let one = || {
            let mut state = start().map_err(|e| e.to_string())?;
            let (mut lat, mut sum) = (Vec::new(), 0);
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let t = Instant::now();
                match work(&mut state, i).map_err(|e| e.to_string())? {
                    Some(n) => sum += n,
                    None => return Ok((lat, sum)),
                }
                lat.push(t.elapsed());
            }
        };
        let threads: Vec<_> = (0..clients).map(|_| s.spawn(one)).collect();
        threads.into_iter().map(|t| t.join().expect("no panic")).collect()
    });
    let took = t0.elapsed();
    let (mut lat, mut sum) = (Vec::new(), 0);
    for p in parts {
        let (l, n) = p?;
        lat.extend(l);
        sum += n;
    }
    lat.sort_unstable();
    Ok((took, lat, sum))
}

fn print(name: &str, clients: usize, runs: &[Run], recall: Option<usize>) {
    let range = |f: &dyn Fn(&Run) -> f64, digits: usize| {
        let v: Vec<f64> = runs.iter().map(f).collect();
        let (lo, hi) = v.iter().fold((f64::MAX, 0f64), |(lo, hi), &x| (lo.min(x), hi.max(x)));
        format!("{lo:.digits$}-{hi:.digits$}")
    };
    let at = |p: f64| move |r: &Run| r.latencies[(r.latencies.len() as f64 * p) as usize].as_secs_f64() * 1e3;
    println!(
        "  {name:<13} {clients:>3} clients: {:>13}/s  p50 {:>11} ms  p99 {:>11} ms  CPU {:>11} ms{}",
        range(&|r| r.per_sec, 0),
        range(&at(0.5), 2),
        range(&at(0.99), 2),
        range(&|r| r.cpu * 1e3 / r.latencies.len() as f64, 2),
        recall.map_or(String::new(), |n| format!("  recall {}", range(&|r| r.hits as f64 / n as f64, 4))),
    );
}

pub(crate) fn http(a: &Args) -> AnyResult<()> {
    a.allow(&[
        "from", "dir", "base", "query", "server", "k", "l", "w", "nq", "passes", "clients", "rounds", "direct", "limit",
        "rows", "batch", "durability", "pool", "pool-cpus",
    ])?;
    let (from, dir) = (a.req_path("from")?, a.req_path("dir")?);
    let direct: bool = a.num("direct", false)?;
    let sync = a.map.get("durability").map(String::as_str).unwrap_or("buffered") == "sync";
    let (here, there) = (import(&from, &dir.join("here"))?, import(&from, &dir.join("there"))?);
    let cpus = std::thread::available_parallelism()?.get();
    let shard = OpenOptions {
        durability: if sync { Durability::Sync } else { Durability::Buffered },
        direct_io: direct,
        threads: (cpus / 2).max(1),
        ..OpenOptions::default()
    };
    let (search_threads, search_cpus) = (a.num("pool", 0)?, a.num("pool-cpus", 0)?);
    let opts = DatabaseOptions { shard, search_threads, search_cpus, ..DatabaseOptions::default() };
    let (db, _) = Database::open(&here, &opts)?;
    let more: Vec<String> = [
        ("durability", if sync { "sync" } else { "buffered" }),
        ("direct-io", if direct { "true" } else { "false" }),
        ("max-k", "100000"),
        ("search-threads", &search_threads.to_string()),
        ("search-cpus", &search_cpus.to_string()),
    ]
    .iter()
    .flat_map(|(k, v)| [format!("--{k}"), v.to_string()])
    .collect();
    let server = Server::start(&a.req_path("server")?, &there, &more)?;
    let base = load_vectors(&a.req_path("base")?, a.opt_num("limit")?)?;
    let clients: Vec<usize> = parse_list(a, "clients", "1,4,8")?;
    let rounds: usize = a.num("rounds", 3)?;
    searches(a, &db, &server, &base, &clients, rounds)?;
    if a.num("rows", 0usize)? > 0 {
        writes(a, &db, &server, &base, &clients, rounds)?;
    }
    Ok(())
}

fn searches(
    a: &Args,
    db: &Database,
    server: &Server,
    base: &Vecs<f32>,
    clients: &[usize],
    rounds: usize,
) -> AnyResult<()> {
    let c = db.collection("c")?;
    let queries = load_vectors(&a.req_path("query")?, Some(a.num("nq", 1000)?))?;
    let (nq, k): (usize, usize) = (queries.len(), a.num("k", 10)?);
    let passes: usize = a.num("passes", 2)?;
    let options = SearchOptions { l: a.num("l", 40)?, beam_width: a.num("w", 4)?, ..Default::default() };
    let live = vec![true; base.len()];
    let gt = ground_truth(base, &live, c.config().metric, &queries, k)?;
    let hits = |q: usize, ids: &[u64]| ids.iter().filter(|id| gt[q].contains(id)).count();
    let requests = |how: How| -> Vec<Vec<u8>> {
        let one = |q: usize| {
            let mut body = String::from("{\"vector\":");
            vector_text(&mut body, queries.row(q), how);
            let _ = write!(body, ",\"k\":{k},\"l\":{},\"beam_width\":{}}}", options.l, options.beam_width);
            request("POST", "/v1/collections/c/search", &body)
        };
        (0..nq).map(one).collect()
    };
    let requests = [Vec::new(), requests(How::Json), requests(How::Base64)];
    println!("searches: k {k}, L {}, request {} or {} bytes", options.l, requests[1][0].len(), requests[2][0].len());
    let search = |n: usize, i: usize, total: usize| -> AnyResult<Run> {
        let how = HOWS[i].0;
        let cpu = || if how == How::Here { cpu_seconds("self") } else { server.cpu() };
        let cpu0 = cpu();
        let start = || Ok((Client::new(server.addr)?, Vec::new()));
        let (took, latencies, hits) = run(n, start, |(client, body), at| {
            let q = at % nq;
            if at >= total {
                return Ok(None);
            }
            if how != How::Here {
                client.ask(&requests[i][q], body)?;
                return Ok(Some(hits(q, &ids(body))));
            }
            let r = c.search(&SearchRequest::new(queries.row(q).to_vec(), k, options.clone()))?;
            Ok(Some(hits(q, &r.results.iter().map(|r| r.id).collect::<Vec<_>>())))
        })?;
        let per_sec = latencies.len() as f64 / took.as_secs_f64();
        Ok(Run { per_sec, latencies, cpu: cpu() - cpu0, hits })
    };
    // Warm caches.
    for i in [0, 1, 0, 1] {
        search(2, i, nq)?;
    }
    for &n in clients {
        let mut runs: [Vec<Run>; 3] = Default::default();
        for _ in 0..rounds {
            for (i, runs) in runs.iter_mut().enumerate() {
                runs.push(search(n, i, nq * passes)?);
            }
        }
        for ((_, name), runs) in HOWS.iter().zip(&runs) {
            print(name, n, runs, Some(nq * passes * k));
        }
    }
    Ok(())
}

fn writes(
    a: &Args,
    db: &Database,
    server: &Server,
    base: &Vecs<f32>,
    clients: &[usize],
    rounds: usize,
) -> AnyResult<()> {
    let (rows, batch): (usize, usize) = (a.num("rows", 0)?, a.num("batch", 100)?);
    let rows = rows.min(base.len());
    let metric = db.collection("c")?.config().metric;
    // Into RAM and the log only.
    let config = || CollectionConfig { seal_bytes: 4 << 30, ..CollectionConfig::new(base.dim, metric) };
    let requests = |how: How| -> Vec<Vec<u8>> {
        let one = |b: usize| {
            let mut body = String::from("{\"rows\":");
            let mut sep = '[';
            for i in b * batch..((b + 1) * batch).min(rows) {
                let _ = write!(body, "{sep}{{\"id\":{i},\"vector\":");
                vector_text(&mut body, base.row(i), how);
                body.push('}');
                sep = ',';
            }
            body.push_str("]}");
            request("POST", "/v1/collections/w/upsert", &body)
        };
        (0..rows.div_ceil(batch)).map(one).collect()
    };
    let requests = [Vec::new(), requests(How::Json), requests(How::Base64)];
    println!("writes: {rows} rows of dim {}, {batch} per batch (latency and CPU are per batch)", base.dim);
    println!("  request {} or {} bytes", requests[1][0].len(), requests[2][0].len());
    for &n in clients {
        let mut all: [Vec<Run>; 3] = Default::default();
        for _ in 0..rounds {
            for (i, runs) in all.iter_mut().enumerate() {
                let how = HOWS[i].0;
                let mut control = Client::new(server.addr)?;
                let mut body = Vec::new();
                let c: Option<Arc<ShardedCollection>> = match how {
                    How::Here => Some(db.create_collection("w", config(), 1)?),
                    _ => {
                        let settings = format!(
                            "{{\"dim\":{},\"metric\":\"{}\",\"seal_bytes\":{}}}",
                            base.dim,
                            metric.name(),
                            4u64 << 30
                        );
                        control.ask(&request("PUT", "/v1/collections/w", &settings), &mut body)?;
                        None
                    }
                };
                let cpu = || if how == How::Here { cpu_seconds("self") } else { server.cpu() };
                let cpu0 = cpu();
                let start = || Ok((Client::new(server.addr)?, Vec::new()));
                let (took, latencies, _) = run(n, start, |(client, body), b| {
                    if b * batch >= rows {
                        return Ok(None);
                    }
                    match &c {
                        Some(c) => {
                            let row = |i: usize| {
                                Op::Upsert(Row { id: i as u64, vector: base.row(i).to_vec(), attrs: Attrs::new() })
                            };
                            c.write((b * batch..((b + 1) * batch).min(rows)).map(row).collect())?;
                        }
                        None => client.ask(&requests[i][b], body)?,
                    }
                    Ok(Some(0))
                })?;
                let cpu = cpu() - cpu0;
                runs.push(Run { per_sec: rows as f64 / took.as_secs_f64(), latencies, cpu, hits: 0 });
                match c {
                    Some(c) => {
                        assert_eq!(c.len()?, rows);
                        drop(c);
                        db.drop_collection("w")?;
                    }
                    None => control.ask(&request("DELETE", "/v1/collections/w", ""), &mut body)?,
                }
            }
        }
        for ((_, name), runs) in HOWS.iter().zip(&all) {
            print(name, n, runs, None);
        }
    }
    Ok(())
}
