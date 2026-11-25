//! A server to test, and a small HTTP client.

#![allow(dead_code)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use vecdb_core::{Database, DatabaseOptions, Durability, OpenOptions};
use vecdb_server::{App, Config};

/// A directory that is removed in the end, unless the test fails.
pub struct Dir(pub PathBuf);

impl Dir {
    pub fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("vecdb-server-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        Self(path)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

pub fn open(dir: &Path) -> Database {
    let shard =
        OpenOptions { durability: Durability::Buffered, direct_io: false, threads: 2, ..OpenOptions::default() };
    Database::open(dir, &DatabaseOptions { shard, search_threads: 2, search_cpus: 0, builds: 1 }).unwrap().0
}

/// A server in this process.
pub struct Server {
    pub addr: SocketAddr,
    stop: Option<oneshot::Sender<()>>,
    thread: Option<JoinHandle<bool>>,
}

impl Server {
    pub fn start(dir: &Dir, cfg: Config) -> Self {
        Self::of(open(&dir.0), cfg)
    }

    pub fn of(db: Database, cfg: Config) -> Self {
        let app = App::new(db, cfg);
        let (stop, stopped) = oneshot::channel();
        let (tell, addr) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
            rt.block_on(async {
                let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
                tell.send(listener.local_addr().unwrap()).unwrap();
                app.serve(listener, async move { drop(stopped.await) }).await
            })
        });
        Self { addr: addr.recv().unwrap(), stop: Some(stop), thread: Some(thread) }
    }

    /// Asks it to stop.
    pub fn stop(&mut self) {
        let _ = self.stop.take().map(|s| s.send(()));
    }

    /// Whether all requests were answered.
    pub fn stopped(&mut self) -> bool {
        self.stop();
        self.thread.take().unwrap().join().unwrap()
    }

    pub fn client(&self) -> Client {
        Client::new(self.addr)
    }

    /// The value of a metric.
    pub fn metric(&self, name: &str) -> f64 {
        let text = self.client().get("/metrics").text();
        let line = text.lines().find(|l| l.split(' ').next() == Some(name));
        line.map_or(0.0, |l| l.split(' ').nth(1).unwrap().parse().unwrap())
    }

    pub fn wait_metric(&self, name: &str, ok: impl Fn(f64) -> bool) {
        wait(name, || ok(self.metric(name)));
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

pub fn wait(what: &str, mut done: impl FnMut() -> bool) {
    let start = Instant::now();
    while !done() {
        assert!(start.elapsed() < Duration::from_secs(60), "{what}: too late");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[derive(Debug)]
pub struct Reply {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Reply {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(n, _)| n.eq_ignore_ascii_case(name)).map(|(_, v)| &v[..])
    }

    pub fn text(&self) -> String {
        String::from_utf8(self.body.clone()).unwrap()
    }

    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or_else(|e| panic!("{e}: {self:?}"))
    }

    /// The JSON of a reply that has to be a success.
    pub fn ok(self) -> Value {
        assert!((200..300).contains(&self.status), "{} {}", self.status, self.text());
        self.json()
    }

    /// Checks that it is this error; its message.
    pub fn err(self, status: u16, code: &str) -> String {
        assert_eq!(self.status, status, "{}", self.text());
        let v = self.json();
        assert_eq!(v["error"]["code"], code, "{v}");
        v["error"]["message"].as_str().unwrap().to_string()
    }
}

/// A connection, opened again when the server has closed it.
pub struct Client {
    addr: SocketAddr,
    pub key: Option<String>,
    conn: Option<BufReader<TcpStream>>,
}

impl Client {
    pub fn new(addr: SocketAddr) -> Self {
        Self { addr, key: None, conn: None }
    }

    pub fn stream(&mut self) -> &mut TcpStream {
        let addr = self.addr;
        let conn = self.conn.get_or_insert_with(|| {
            let s = TcpStream::connect(addr).unwrap();
            s.set_nodelay(true).unwrap();
            s.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
            BufReader::new(s)
        });
        conn.get_mut()
    }

    pub fn send_raw(&mut self, bytes: &[u8]) -> &mut Self {
        self.stream().write_all(bytes).unwrap();
        self
    }

    pub fn head(&self, method: &str, path: &str, more: &str) -> String {
        let key = self.key.as_ref().map_or(String::new(), |k| format!("Authorization: Bearer {k}\r\n"));
        format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\n{key}{more}\r\n")
    }

    /// The next reply, if the connection is not closed instead.
    pub fn try_reply(&mut self) -> std::io::Result<Option<Reply>> {
        self.stream();
        let conn = self.conn.as_mut().unwrap();
        let mut line = String::new();
        if conn.read_line(&mut line)? == 0 {
            self.conn = None;
            return Ok(None);
        }
        let status = line.split(' ').nth(1).unwrap_or_else(|| panic!("{line:?}")).parse().unwrap();
        let mut headers = Vec::new();
        loop {
            let mut line = String::new();
            conn.read_line(&mut line)?;
            match line.trim_end().split_once(": ") {
                Some((n, v)) => headers.push((n.to_string(), v.to_string())),
                None => break,
            }
        }
        let mut reply = Reply { status, headers, body: Vec::new() };
        if status == 100 {
            return Ok(Some(reply));
        }
        let n = reply.header("content-length").expect("a length").parse().unwrap();
        reply.body.resize(n, 0);
        conn.read_exact(&mut reply.body)?;
        if reply.header("connection") == Some("close") {
            self.conn = None;
        }
        Ok(Some(reply))
    }

    pub fn reply(&mut self) -> Reply {
        self.try_reply().unwrap().expect("a reply")
    }

    pub fn ask(&mut self, method: &str, path: &str, body: &str) -> Reply {
        let more = match body {
            "" => String::new(),
            b => format!("Content-Type: application/json\r\nContent-Length: {}\r\n", b.len()),
        };
        let head = self.head(method, path, &more);
        self.send_raw(&[head.as_bytes(), body.as_bytes()].concat());
        self.reply()
    }

    pub fn get(&mut self, path: &str) -> Reply {
        self.ask("GET", path, "")
    }

    pub fn post(&mut self, path: &str, body: &Value) -> Reply {
        self.ask("POST", path, &body.to_string())
    }

    pub fn put(&mut self, path: &str, body: &Value) -> Reply {
        self.ask("PUT", path, &body.to_string())
    }

    pub fn delete(&mut self, path: &str) -> Reply {
        self.ask("DELETE", path, "")
    }
}
