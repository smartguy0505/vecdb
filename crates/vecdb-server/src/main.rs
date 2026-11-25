//! The server's command line.

#![deny(unsafe_code)]

use std::collections::HashMap;
use std::fmt::Display;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::str::FromStr;
use std::time::{Duration, Instant};

use tokio::net::TcpListener;
use tokio::signal::unix::{SignalKind, signal};
use vecdb_core::{Database, DatabaseOptions, Durability, OpenOptions};
use vecdb_server::{Access, App, Config, log};

const USAGE: &str = "\
vecdb-server --data DIR [--flag value ...]

  --data DIR                 the database's directory (created if missing)
  --listen ADDR              127.0.0.1:7410; there is no TLS: put a proxy in front of it
  --api-key-file FILE        requests need `Authorization: Bearer <what is in it>`
                             (or the key in the environment: VECDB_API_KEY)
  --allow-unauthenticated B  false: without a key, whether to answer anyone; otherwise
                             only on this machine, and not web pages

storage
  --durability sync|buffered sync: a write is answered once it survives a power loss
  --direct-io B              true: read SSD indexes past the page cache
  --search-threads N         threads that help searches (0: two per CPU)
  --search-cpus N            CPUs searches share with those (0: a quarter of them)
  --build-threads N          threads of index builds, together (0: half of the CPUs)
  --builds N                 1: index builds at once, and compactions at once besides
  --build-memory-mb N        2048: RAM of a build, for one partition
  --max-unindexed-mb N       1024: vectors per shard that may wait for an index (0: no limit)
  --write-wait-secs N        10: how long a write may wait for that, then it gets 503

requests
  --searches N               at once (0: two per CPU)
  --writes N                 64: at once
  --queue N                  1024: that may wait, of each kind
  --queue-secs N             10: how long
  --max-body-mb N            64: largest body of a write (4 for other requests)
  --max-bodies-mb N          512: large bodies (over 256 KB) in memory, together
  --max-reply-mb N           64: about the largest reply
  --max-replies-mb N         512: large replies in memory, together
  --max-ops N                100000: writes in a request
  --max-k N, --max-l N       10000: largest k and l of a search
  --max-shards N             64: shards of a new collection
  --max-total-shards N       1024: shards of all (each has three threads)
  --max-connections N        4096
  --network-threads N        that read requests and write replies (0: one per CPU, 4 at most)
  --idle-secs N              75: until a connection without a request is closed
  --drain-secs N             30: for requests to finish when the server stops
  --slow-ms N                log requests that take longer (0: none)";

type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;

struct Args(HashMap<String, String>);

impl Args {
    fn parse(mut it: impl Iterator<Item = String>) -> AnyResult<Self> {
        let mut map = HashMap::new();
        while let Some(a) = it.next() {
            let key = a.strip_prefix("--").ok_or_else(|| format!("expected --flag, got {a:?}"))?;
            let val = it.next().ok_or_else(|| format!("--{key} needs a value"))?;
            if map.insert(key.to_string(), val).is_some() {
                return Err(format!("--{key} given twice").into());
            }
        }
        Ok(Self(map))
    }

    /// Takes a flag out, so that what is left at the end is not known.
    fn take<T: FromStr>(&mut self, key: &str, default: T) -> AnyResult<T>
    where
        T::Err: Display,
    {
        match self.0.remove(key) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|e| format!("--{key} {v:?}: {e}").into()),
        }
    }

    fn mb(&mut self, key: &str, default: usize) -> AnyResult<usize> {
        self.take(key, default)?.checked_mul(1 << 20).ok_or_else(|| format!("--{key} is too large").into())
    }

    fn secs(&mut self, key: &str, default: Duration) -> AnyResult<Duration> {
        Ok(Duration::from_secs(self.take(key, default.as_secs())?))
    }
}

/// Raises the limit on open files as far as it goes. What it is then.
#[allow(unsafe_code)]
fn raise_file_limit() -> u64 {
    let mut limit = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
    // SAFETY: reads and writes `limit` only.
    unsafe {
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) != 0 {
            return 1024;
        }
        let raised = libc::rlimit { rlim_cur: limit.rlim_max, ..limit };
        match libc::setrlimit(libc::RLIMIT_NOFILE, &raised) {
            0 => raised.rlim_cur,
            _ => limit.rlim_cur,
        }
    }
}

fn api_key(a: &mut Args) -> AnyResult<Option<String>> {
    let key = match a.0.remove("api-key-file") {
        Some(file) => std::fs::read_to_string(&file).map_err(|e| format!("--api-key-file {file}: {e}"))?,
        None => match std::env::var("VECDB_API_KEY") {
            Ok(key) => key,
            Err(_) => return Ok(None),
        },
    };
    let key = key.trim().to_string();
    if key.len() < 16 || !key.bytes().all(|b| b.is_ascii_graphic()) {
        return Err("the API key must be 16 or more printable ASCII characters".into());
    }
    Ok(Some(key))
}

fn or(n: usize, default: usize) -> usize {
    if n == 0 { default } else { n }
}

fn storage(a: &mut Args, cpus: usize) -> AnyResult<DatabaseOptions> {
    let durability = match a.take("durability", "sync".to_string())?.as_str() {
        "sync" => Durability::Sync,
        "buffered" => Durability::Buffered,
        d => return Err(format!("--durability {d:?}: use sync or buffered").into()),
    };
    Ok(DatabaseOptions {
        shard: OpenOptions {
            durability,
            direct_io: a.take("direct-io", true)?,
            threads: or(a.take("build-threads", 0)?, (cpus / 2).max(1)),
            build_memory: a.mb("build-memory-mb", 2048)? as u64,
            max_unindexed_bytes: a.mb("max-unindexed-mb", 1024)? as u64,
            max_write_wait: Some(a.secs("write-wait-secs", Duration::from_secs(10))?),
            ..OpenOptions::default()
        },
        search_threads: a.take("search-threads", 0)?,
        search_cpus: a.take("search-cpus", 0)?,
        builds: a.take("builds", 1)?,
    })
}

fn run() -> AnyResult<()> {
    let mut a = Args::parse(std::env::args().skip(1))?;
    let data: PathBuf = a.0.remove("data").ok_or("--data is required")?.into();
    let listen: SocketAddr = a.take("listen", "127.0.0.1:7410".parse()?)?;
    let anyone = a.take("allow-unauthenticated", false)?;
    let access = match api_key(&mut a)? {
        Some(key) => Access::Key(key),
        None if anyone => Access::Anyone,
        None if listen.ip().is_loopback() => Access::Local,
        None => {
            return Err(format!(
                "{listen} can be reached by others: give an API key (--api-key-file), \
                 or --allow-unauthenticated true if something else keeps them out"
            )
            .into());
        }
    };
    let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
    let opts = storage(&mut a, cpus)?;
    // Connections may take half of them, so that they do not keep the engine
    // from opening its files.
    let files = raise_file_limit();
    let d = Config::default();
    let slow: u64 = a.take("slow-ms", 0)?;
    let cfg = Config {
        access,
        searches: or(a.take("searches", 0)?, d.searches),
        writes: a.take("writes", d.writes)?.max(1),
        queue: a.take("queue", d.queue)?,
        queue_wait: a.secs("queue-secs", d.queue_wait)?,
        max_body: a.mb("max-body-mb", d.max_body >> 20)?,
        max_bodies: a.mb("max-bodies-mb", d.max_bodies >> 20)?,
        max_reply: a.mb("max-reply-mb", d.max_reply >> 20)?,
        max_replies: a.mb("max-replies-mb", d.max_replies >> 20)?,
        max_ops: a.take("max-ops", d.max_ops)?,
        max_total_shards: a.take("max-total-shards", d.max_total_shards)?,
        max_k: a.take("max-k", d.max_k)?,
        max_l: a.take("max-l", d.max_l)?,
        max_shards: a.take("max-shards", d.max_shards)?,
        max_connections: a.take("max-connections", d.max_connections)?.min(files as usize / 2),
        idle: a.secs("idle-secs", d.idle)?,
        drain: a.secs("drain-secs", d.drain)?,
        slow: (slow > 0).then(|| Duration::from_millis(slow)),
        ..d
    };
    let network = or(a.take("network-threads", 0)?, cpus.min(4));
    if let Some(flag) = a.0.keys().next() {
        return Err(format!("unknown flag --{flag}").into());
    }
    let start = Instant::now();
    let (db, report) = Database::open(&data, &opts)?;
    let rows: usize = db.names().iter().filter_map(|n| db.collection(n).and_then(|c| c.len()).ok()).sum();
    let took = start.elapsed();
    log!("{}: {} collections, {rows} rows, opened in {took:.1?}", data.display(), report.opened.len());
    for (name, e) in &report.failed {
        log!("collection {name} could not be opened: {e}");
    }
    for (name, shards) in &report.opened {
        let cut: u64 = shards.iter().map(|s| s.replay.cut_bytes).sum();
        if cut > 0 {
            log!("collection {name}: {cut} bytes of unfinished writes at the end of its logs were cut off");
        }
    }
    for name in &report.unknown {
        log!("{name} is not in the catalog and was left alone");
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(network)
        .max_blocking_threads(cfg.threads())
        // Those that search keep what they need for it.
        .thread_keep_alive(Duration::from_secs(300))
        .enable_all()
        .build()?;
    let app = App::new(db, cfg);
    let done = runtime.block_on(async {
        let listener = TcpListener::bind(listen).await.map_err(|e| format!("--listen {listen}: {e}"))?;
        let (mut term, mut int) = (signal(SignalKind::terminate())?, signal(SignalKind::interrupt())?);
        log!("listening on http://{}", listener.local_addr()?);
        let stop = async move {
            tokio::select! { _ = term.recv() => {}, _ = int.recv() => {} }
            log!("stopping");
            // A second signal is not waited for.
            tokio::spawn(async move {
                tokio::select! { _ = term.recv() => {}, _ = int.recv() => {} }
                std::process::exit(1);
            });
        };
        AnyResult::Ok(app.clone().serve(listener, stop).await)
    })?;
    if !done {
        log!("some requests did not finish in time");
    }
    // If nothing uses the database any more, dropping it waits for flushes
    // and stops builds. What still runs (a request, an optimize) is cut off,
    // as by a crash: what was answered is in the logs.
    runtime.shutdown_timeout(Duration::from_secs(if done { 1 } else { 0 }));
    drop(app);
    log!("stopped");
    Ok(())
}

fn main() -> ExitCode {
    if std::env::args().skip(1).any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            log!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(flags: &str) -> Args {
        Args::parse(flags.split_whitespace().map(String::from)).unwrap()
    }

    #[test]
    fn storage_flags() {
        let o = storage(&mut args(""), 12).unwrap();
        assert!(matches!(o.shard.durability, Durability::Sync));
        assert!(o.shard.direct_io);
        assert_eq!(o.shard.threads, 6);
        assert_eq!(o.shard.build_memory, 2 << 30);
        assert_eq!(o.shard.max_unindexed_bytes, 1 << 30);
        assert_eq!(o.shard.max_write_wait, Some(Duration::from_secs(10)));
        assert_eq!((o.search_threads, o.search_cpus, o.builds), (0, 0, 1));
        assert_eq!(storage(&mut args(""), 1).unwrap().shard.threads, 1);
        let mut a = args(
            "--durability buffered --direct-io false --build-threads 3 --build-memory-mb 5 --max-unindexed-mb 7 \
             --write-wait-secs 2 --search-threads 4 --search-cpus 8 --builds 9",
        );
        let o = storage(&mut a, 12).unwrap();
        assert!(a.0.is_empty(), "{:?}", a.0);
        assert!(matches!(o.shard.durability, Durability::Buffered));
        assert!(!o.shard.direct_io);
        assert_eq!(o.shard.threads, 3);
        assert_eq!(o.shard.build_memory, 5 << 20);
        assert_eq!(o.shard.max_unindexed_bytes, 7 << 20);
        assert_eq!(o.shard.max_write_wait, Some(Duration::from_secs(2)));
        assert_eq!((o.search_threads, o.search_cpus, o.builds), (4, 8, 9));
        assert!(storage(&mut args("--durability often"), 1).is_err());
        assert!(storage(&mut args("--builds x"), 1).is_err());
        assert!(storage(&mut args(&format!("--build-memory-mb {}", usize::MAX)), 1).is_err());
    }
}
