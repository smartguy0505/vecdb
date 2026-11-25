//! The program itself: its flags, signals and restarts.

mod common;

use std::io::{BufRead, BufReader, Read};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use common::{Client, Dir, wait};
use serde_json::{Value, json};

fn command(args: &[&str]) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_vecdb-server"));
    c.args(args).env_remove("VECDB_API_KEY").stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    c
}

struct Server {
    child: Child,
    log: BufReader<std::process::ChildStderr>,
    addr: std::net::SocketAddr,
}

impl Server {
    fn start(dir: &Dir, args: &[&str]) -> Self {
        Self::start_with(dir, args, None)
    }

    fn start_with(dir: &Dir, args: &[&str], key: Option<&str>) -> Self {
        let mut all = vec!["--data", dir.0.to_str().unwrap(), "--listen", "127.0.0.1:0", "--build-threads", "2"];
        all.extend(args);
        let mut command = command(&all);
        command.envs(key.map(|k| ("VECDB_API_KEY", k)));
        let mut child = command.spawn().unwrap();
        let mut log = BufReader::new(child.stderr.take().unwrap());
        let addr = loop {
            let mut line = String::new();
            assert!(log.read_line(&mut line).unwrap() > 0, "it stopped");
            if let Some(a) = line.trim().strip_prefix("listening on http://") {
                break a.parse().unwrap();
            }
        };
        Self { child, log, addr }
    }

    fn signal(&self, name: &str) {
        assert!(Command::new("kill").args(["-s", name, &self.child.id().to_string()]).status().unwrap().success());
    }

    /// Its exit code (none if it was killed) and what it still said.
    fn end(mut self) -> (Option<i32>, String) {
        let mut rest = String::new();
        self.log.read_to_string(&mut rest).unwrap();
        (self.child.wait().unwrap().code(), rest)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn restarts() {
    let dir = Dir::new("restarts");
    let settings = json!({"dim": 3, "metric": "cosine", "shards": 2, "fields": {"n": "int"}});
    let mut rows = 0;
    for how in ["TERM", "INT", "KILL", "TERM"] {
        let server = Server::start(&dir, &[]);
        let mut c = Client::new(server.addr);
        if rows == 0 {
            c.put("/v1/collections/c", &settings).ok();
        }
        assert_eq!(c.get("/v1/collections/c").ok()["rows"], rows);
        for _ in 0..5 {
            let new: Vec<Value> =
                (rows..rows + 100).map(|i| json!({"id": i, "vector": [i, 1, 2], "attrs": {"n": i}})).collect();
            c.post("/v1/collections/c/upsert", &json!({"rows": new})).ok();
            rows += 100;
        }
        if rows == 1000 {
            c.ask("POST", "/v1/collections/c/flush", "").ok();
        }
        server.signal(how);
        let (code, log) = server.end();
        match how {
            "KILL" => assert_eq!(code, None),
            _ => {
                assert_eq!(code, Some(0), "{log}");
                assert_eq!(log, "stopping\nstopped\n");
            }
        }
    }
    let server = Server::start(&dir, &[]);
    let mut c = Client::new(server.addr);
    let body = json!({"vector": [1999, 1, 2], "k": 3, "filter": {"field": "n", "gte": 1999}, "with_attrs": true});
    assert_eq!(c.post("/v1/collections/c/search", &body).ok()["results"].as_array().unwrap().len(), 1);
    assert_eq!(c.get("/v1/collections/c").ok()["rows"], 2000);
    // Only one of them at a time.
    let out = command(&["--data", dir.0.to_str().unwrap(), "--listen", "127.0.0.1:0"]).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("already open"), "{out:?}");
}

/// A request that does not end holds it up for so long, or until a second
/// signal.
#[test]
fn signals() {
    let dir = Dir::new("signals");
    for second in [false, true] {
        let server = Server::start(&dir, &["--drain-secs", if second { "60" } else { "1" }]);
        let mut c = Client::new(server.addr);
        let head = c.head("PUT", "/v1/collections/c", "Content-Type: application/json\r\nContent-Length: 50\r\n");
        c.send_raw(head.as_bytes()).send_raw(b"{");
        std::thread::sleep(Duration::from_millis(200));
        server.signal("TERM");
        if second {
            wait("the listener to close", || std::net::TcpStream::connect(server.addr).is_err());
            server.signal("TERM");
        }
        let (code, log) = server.end();
        match second {
            true => assert_eq!((code, &log[..]), (Some(1), "stopping\n")),
            false => {
                assert_eq!((code, &log[..]), (Some(0), "stopping\nsome requests did not finish in time\nstopped\n"))
            }
        }
    }
}

#[test]
fn flags() {
    let dir = Dir::new("flags");
    std::fs::create_dir_all(&dir.0).unwrap();
    let data = dir.0.join("db");
    let data = data.to_str().unwrap();
    let file = |name: &str, text: &str| {
        let path = dir.0.join(name);
        std::fs::write(&path, text).unwrap();
        path.to_str().unwrap().to_string()
    };
    let (key, short) = (file("key", "  0123456789abcdef\n"), file("short", "0123456789abcde\n"));
    let spaced = file("spaced", "0123456789 abcdef\n");
    for (args, why) in [
        (&["--listen", "127.0.0.1:0"][..], "--data is required"),
        (&["--data"], "--data needs a value"),
        (&["data", "x"], "expected --flag"),
        (&["--data", data, "--data", data], "given twice"),
        (&["--data", data, "--nothing", "1"], "unknown flag --nothing"),
        (&["--data", data, "--searches", "x"], "--searches"),
        (&["--data", data, "--listen", "localhost"], "--listen"),
        (&["--data", data, "--durability", "none"], "use sync or buffered"),
        (&["--data", data, "--max-body-mb", "18446744073709551615"], "too large"),
        (&["--data", data, "--api-key-file", "/nonexistent"], "/nonexistent"),
        (&["--data", data, "--api-key-file", &short], "16 or more"),
        (&["--data", data, "--api-key-file", &spaced], "16 or more"),
        // Beyond this machine. (Addresses for documentation, which no
        // machine has: should they not be refused, it cannot listen there.)
        (&["--data", data, "--listen", "192.0.2.1:0"], "can be reached by others"),
        (&["--data", data, "--listen", "[2001:db8::1]:0"], "can be reached by others"),
    ] {
        let out = command(args).output().unwrap();
        let said = String::from_utf8_lossy(&out.stderr).into_owned();
        assert_eq!(out.status.code(), Some(1), "{args:?}: {said}");
        assert!(said.starts_with("error: ") && said.contains(why), "{args:?}: {said}");
    }
    assert!(!std::path::Path::new(data).exists());
    let out = command(&["--help"]).output().unwrap();
    assert!(out.status.success() && String::from_utf8_lossy(&out.stdout).contains("--data DIR"));

    // The key of a file, or of the environment.
    for (args, env) in [(&["--api-key-file", &key][..], None), (&[], Some("0123456789abcdef"))] {
        let server = Server::start_with(&dir, args, env);
        let mut c = Client::new(server.addr);
        c.get("/v1/collections").err(401, "unauthorized");
        c.key = Some("0123456789abcdeg".into());
        c.get("/v1/collections").err(401, "unauthorized");
        c.key = Some("0123456789abcdef".into());
        c.get("/v1/collections").ok();
    }
}
