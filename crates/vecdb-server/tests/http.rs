//! The server over real sockets.

mod common;

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use common::{Dir, Server, wait};
use serde_json::{Value, json};
use vecdb_core::rng::SplitMix64;
use vecdb_core::{Database, DatabaseOptions, Durability, OpenOptions};
use vecdb_server::{Access, Config};

fn random(rng: &mut SplitMix64, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| (rng.next_u64() % 10_000) as f32 / 100.0 - 50.0).collect()
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

fn ids(v: &Value) -> Vec<u64> {
    v["results"].as_array().unwrap().iter().map(|r| r["id"].as_u64().unwrap()).collect()
}

#[test]
fn a_collection_from_creation_to_drop() {
    let dir = Dir::new("life");
    let mut server = Server::start(&dir, Config::default());
    let mut c = server.client();
    assert_eq!(c.get("/healthz").ok(), json!({"status": "ok"}));
    assert_eq!(c.get("/v1/collections").ok(), json!({"collections": []}));

    let settings = json!({"dim": 8, "metric": "l2", "shards": 3, "seal_bytes": 4096, "index_bytes": 16384,
        "fields": {"n": "int", "tag": "keyword"}, "index": {"max_degree": 16, "build_list": 32, "pq_m": 4}});
    let created = c.put("/v1/collections/docs", &settings);
    assert_eq!(created.status, 201);
    let created = created.ok();
    assert_eq!(created["config"]["shards"], 3);
    c.put("/v1/collections/docs", &settings).err(409, "already_exists");
    let info = c.get("/v1/collections/docs").ok();
    assert_eq!((&info["config"], &info["rows"]), (&created["config"], &json!(0)));
    // What it says of itself makes another like it.
    c.put("/v1/collections/same", &info["config"]).ok();
    assert_eq!(c.get("/v1/collections/same").ok()["config"], created["config"]);
    assert_eq!(c.delete("/v1/collections/same").ok(), json!({"dropped": "same"}));

    // Against a model, over flushes and index builds.
    let mut rng = SplitMix64::new(1);
    let mut model: BTreeMap<u64, Vec<f32>> = BTreeMap::new();
    let attrs = |id: u64| json!({"n": id % 10, "tag": [format!("t{}", id % 3), "all"]});
    for round in 0..40 {
        let rows: BTreeMap<u64, Vec<f32>> =
            (0..100).map(|_| (rng.next_u64() % 3000, random(&mut rng, 8))).collect();
        let body: Vec<Value> =
            rows.iter().map(|(id, v)| json!({"id": id, "vector": v, "attrs": attrs(*id)})).collect();
        let replaced = rows.keys().filter(|id| model.contains_key(id)).count();
        let got = c.post("/v1/collections/docs/upsert", &json!({"rows": body})).ok();
        assert_eq!(got, json!({"upserted": rows.len(), "replaced": replaced}));
        model.extend(rows);

        let gone: Vec<u64> = (0..20).map(|_| rng.next_u64() % 3000).collect();
        let had = gone.iter().filter(|id| model.remove(id).is_some()).count();
        let got = c.post("/v1/collections/docs/delete", &json!({"ids": gone})).ok();
        assert_eq!(got, json!({"deleted": had}));

        // In order: the last one counts.
        let (id, v) = (rng.next_u64() % 3000, random(&mut rng, 8));
        let had = model.contains_key(&id);
        let ops = json!({"ops": [{"delete": {"id": id}}, {"upsert": {"id": id, "vector": v, "attrs": attrs(id)}},
                                 {"delete": {"id": id}}, {"upsert": {"id": id, "vector": v, "attrs": attrs(id)}}]});
        let got = c.post("/v1/collections/docs/write", &ops).ok();
        assert_eq!(got["existed"], json!([had, false, true, false]));
        model.insert(id, v);
        if round % 10 == 5 {
            c.ask("POST", "/v1/collections/docs/flush", "").ok();
        }

        let q = random(&mut rng, 8);
        for filtered in [false, true] {
            let mut want: Vec<(f32, u64)> = model
                .iter()
                .filter(|(id, _)| !filtered || ((3..6).contains(&(*id % 10)) && *id % 3 == 1))
                .map(|(id, v)| (v.iter().zip(&q).map(|(a, b)| (a - b) * (a - b)).sum(), *id))
                .collect();
            want.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            want.truncate(10);
            let mut body = json!({"vector": q, "k": 10, "l": 400, "with_attrs": true, "with_vectors": true});
            if filtered {
                body["filter"] = json!({"and": [{"field": "n", "gte": 3, "lt": 6}, {"field": "tag", "eq": "t1"}]});
            }
            let got = c.post("/v1/collections/docs/search", &body).ok();
            assert!(got.get("stats").is_none());
            let got = got["results"].as_array().unwrap().clone();
            let found: Vec<u64> = got.iter().map(|r| r["id"].as_u64().unwrap()).collect();
            assert_eq!(found, want.iter().map(|w| w.1).collect::<Vec<_>>(), "round {round}");
            for (r, (d, id)) in got.iter().zip(&want) {
                assert!((r["distance"].as_f64().unwrap() as f32 - d).abs() <= d * 1e-5);
                assert_eq!((floats(&r["vector"]), &r["attrs"]), (model[id].clone(), &attrs(*id)));
            }
        }
    }
    let info = c.get("/v1/collections/docs").ok();
    assert_eq!(info["rows"], model.len());
    assert_eq!(c.get("/v1/collections").ok()["collections"], json!([{"name": "docs", "rows": model.len()}]));
    assert!(info["state"]["flat_rows"].as_u64().unwrap() + info["state"]["ssd_rows"].as_u64().unwrap() > 0);

    // By id: in the order asked for, without those that are not there.
    let some: Vec<u64> = model.keys().rev().step_by(100).copied().collect();
    let asked: Vec<u64> = some.iter().copied().chain([1 << 40]).collect();
    let got = c.post("/v1/collections/docs/get", &json!({"ids": asked})).ok();
    let got = got["rows"].as_array().unwrap();
    assert_eq!(got.len(), some.len());
    for (row, id) in got.iter().zip(&some) {
        assert_eq!((row["id"].as_u64(), &row["attrs"]), (Some(*id), &attrs(*id)));
        assert_eq!(floats(&row["vector"]), model[id]);
    }

    // Counters, if they are asked for.
    let q = json!({"vector": random(&mut rng, 8), "with_stats": true});
    let got = c.post("/v1/collections/docs/search", &q).ok();
    assert_eq!(ids(&got).len(), 10);
    assert!(got["stats"]["segments"].as_u64().unwrap() > 0 && got["results"][0].get("vector").is_none());

    // Merges all of a shard; the state says when it is done.
    let started = c.ask("POST", "/v1/collections/docs/optimize", "");
    assert_eq!((started.status, started.ok()), (202, json!({"optimizing": true})));
    let mut info = Value::Null;
    wait("optimize", || {
        info = c.get("/v1/collections/docs").ok();
        info["state"]["optimizing"] == false
    });
    assert_eq!(info["state"]["optimize_error"], Value::Null);
    let segments = info["state"]["flat_segments"].as_u64().unwrap() + info["state"]["ssd_segments"].as_u64().unwrap();
    assert!(segments <= 3, "{info}");
    assert_eq!(info["rows"], model.len());

    assert_eq!(c.delete("/v1/collections/docs").ok(), json!({"dropped": "docs"}));
    c.get("/v1/collections/docs").err(404, "not_found");
    c.delete("/v1/collections/docs").err(404, "not_found");
    c.post("/v1/collections/docs/search", &q).err(404, "not_found");
    assert_eq!(c.get("/v1/collections").ok(), json!({"collections": []}));
    assert!(server.stopped());
}

#[test]
fn what_is_wrong_with_a_request() {
    let dir = Dir::new("wrong");
    let server = Server::start(&dir, Config { max_shards: 4, max_total_shards: 6, ..Config::default() });
    let mut c = server.client();
    c.put("/v1/collections/c", &json!({"dim": 2, "metric": "l2", "fields": {"n": "int"}})).ok();
    c.post("/v1/collections/c/upsert", &json!({"rows": [{"id": 1, "vector": [1, 2]}]})).ok();

    for (method, path, status, code) in [
        ("GET", "/", 404, "not_found"),
        ("GET", "/v1", 404, "not_found"),
        ("GET", "/v1/collections/", 404, "not_found"),
        ("GET", "/v1/collections/c/", 404, "not_found"),
        ("GET", "/v1/collections/c/search/x", 404, "not_found"),
        ("GET", "/v2/collections", 404, "not_found"),
        ("GET", "/v1/collections/d", 404, "not_found"),
        ("GET", "/v1/collections/%63", 404, "not_found"),
        ("GET", "/v1/collections/..", 404, "not_found"),
        ("GET", "/v1/collections?limit=3", 400, "invalid_argument"),
        ("POST", "/v1/collections", 405, "method_not_allowed"),
        ("POST", "/v1/collections/c", 405, "method_not_allowed"),
        ("GET", "/v1/collections/c/search", 405, "method_not_allowed"),
        ("DELETE", "/v1/collections/c/delete", 405, "method_not_allowed"),
        ("HEADS", "/v1/collections/c", 405, "method_not_allowed"),
        ("POST", "/v1/collections/d/flush", 404, "not_found"),
        ("POST", "/v1/collections/d/optimize", 404, "not_found"),
        ("DELETE", "/v1/collections/d", 404, "not_found"),
    ] {
        c.ask(method, path, "").err(status, code);
    }
    assert_eq!(c.ask("PATCH", "/v1/collections/c", "").header("allow"), Some("GET, PUT, DELETE"));

    for (name, body) in [
        ("x", r#"{"dim": 0, "metric": "l2"}"#),
        ("x", r#"{"dim": 65537, "metric": "l2"}"#),
        ("x", r#"{"dim": 4, "metric": "manhattan"}"#),
        ("x", r#"{"dim": 4, "metric": "l2", "shards": 0}"#),
        ("x", r#"{"dim": 4, "metric": "l2", "shards": 5}"#),
        ("x", r#"{"dim": 4, "metric": "l2", "shards": 18446744073709551615}"#),
        ("x", r#"{"dim": 4, "metric": "l2", "seal_bytes": 0}"#),
        ("x", r#"{"dim": 4, "metric": "l2", "index": {"max_degree": 1000000}}"#),
        ("x", r#"{"dim": 4, "metric": "l2", "index": {"alpha": 0.1}}"#),
        ("x", r#"{"dim": 4, "metric": "l2", "index": {"alpha": 1e40}}"#),
        ("x", r#"{"dim": 4, "metric": "l2", "index": {"pq_m": 5}}"#),
        ("x", r#"{"dim": 4, "metric": "l2", "compact_deleted": -1}"#),
        ("x", r#"{"dim": 4, "metric": "l2", "max_indexed_segments": 0}"#),
        ("x", r#"{"dim": 4, "metric": "l2", "fields": {"": "int"}}"#),
        (".x", r#"{"dim": 4, "metric": "l2"}"#),
        ("x%2fy", r#"{"dim": 4, "metric": "l2"}"#),
        ("x", r#"{"dim": 4, "metric": "l2", "#),
        ("x", r#"[]"#),
        ("x", "\u{feff}{}"),
    ] {
        c.ask("PUT", &format!("/v1/collections/{name}"), body).err(400, "invalid_argument");
    }
    assert_eq!(c.get("/v1/collections").ok()["collections"].as_array().unwrap().len(), 1);
    // Shards of all collections.
    c.put("/v1/collections/d", &json!({"dim": 4, "metric": "ip", "shards": 4})).ok();
    c.put("/v1/collections/e", &json!({"dim": 4, "metric": "ip", "shards": 2})).err(409, "too_many_shards");
    c.put("/v1/collections/e", &json!({"dim": 4, "metric": "ip", "shards": 1})).ok();

    for (op, body) in [
        ("upsert", r#"{"rows": [{"id": 2, "vector": [1]}]}"#),
        ("upsert", r#"{"rows": [{"id": 2, "vector": [1, 2, 3]}]}"#),
        ("upsert", r#"{"rows": [{"id": 2, "vector": [1, 1e39]}]}"#),
        ("upsert", r#"{"rows": [{"id": 2, "vector": "AACAPw=="}]}"#),
        ("upsert", r#"{"rows": [{"id": 2, "vector": "AADAfwAAAAA="}]}"#),
        ("upsert", r#"{"rows": [{"id": 2, "vector": [1, 2], "attrs": {"m": 1}}]}"#),
        ("upsert", r#"{"rows": [{"id": 2, "vector": [1, 2], "attrs": {"n": "1"}}]}"#),
        ("upsert", r#"{"rows": [{"id": 2, "vector": [1, 2]}, {"id": 3}]}"#),
        ("upsert", r#"{"rows": {}}"#),
        ("upsert", r#"{}"#),
        ("upsert", r#""#),
        ("upsert", r#"null"#),
        ("delete", r#"{"ids": [-1]}"#),
        ("delete", r#"{"ids": 1}"#),
        ("write", r#"{"ops": [{"upsert": {"id": 2, "vector": [1, 2]}}, {"delete": {}}]}"#),
        ("get", r#"{"ids": ["1"]}"#),
        ("search", r#"{"vector": [1]}"#),
        ("search", r#"{"vector": []}"#),
        ("search", r#"{"vector": [1, 2], "k": 0}"#),
        ("search", r#"{"vector": [1, 2], "k": 10001}"#),
        ("search", r#"{"vector": [1, 2], "l": 10001}"#),
        ("search", r#"{"vector": [1, 2], "beam_width": 1000000}"#),
        ("search", r#"{"vector": [1, 2], "filter": {"field": "m", "eq": 1}}"#),
        ("search", r#"{"vector": [1, 2], "filter": {"field": "n", "eq": "1"}}"#),
        ("search", r#"{"vector": [1, 2], "filter": {"field": "n"}}"#),
        ("search", r#"{"vector": [1, 2]} x"#),
        ("search", r#"{"vector": [1, 2], "K": 3}"#),
    ] {
        // (An empty body, as a JSON one.)
        let more = format!("Content-Type: application/json\r\nContent-Length: {}\r\n", body.len());
        let head = c.head("POST", &format!("/v1/collections/c/{op}"), &more);
        c.send_raw(format!("{head}{body}").as_bytes());
        let reply = c.reply();
        assert_eq!(reply.status, 400, "{op} {body}: {}", reply.text());
        reply.err(400, "invalid_argument");
    }
    let deep = format!(r#"{{"vector": [1, 2], "filter": {}"#, r#"{"not": "#.repeat(50_000));
    c.ask("POST", "/v1/collections/c/search", &deep).err(400, "invalid_argument");
    let deep = format!("{}{}{}", r#"{"not": "#.repeat(40), r#"{"field": "n", "eq": 1}"#, "}".repeat(40));
    let deep = format!(r#"{{"vector": [1, 2], "filter": {deep}}}"#);
    let why = c.ask("POST", "/v1/collections/c/search", &deep).err(400, "invalid_argument");
    assert!(why.contains("deep") || why.contains("nested"), "{why}");
    // A bad row, and none of them was written.
    assert_eq!(c.get("/v1/collections/c").ok()["rows"], 1);
    // It is not HTTP.
    c.send_raw(b"\x16\x03\x01 hello\r\n\r\n");
    assert_eq!(c.try_reply().unwrap().map(|r| r.status).unwrap_or(400), 400);
}

/// Ids and integers of all 64 bits, and the bits of floats, survive.
#[test]
fn numbers_are_exact() {
    let dir = Dir::new("exact");
    let server = Server::start(&dir, Config::default());
    let mut c = server.client();
    c.put("/v1/collections/c", &json!({"dim": 4, "metric": "l2", "fields": {"n": "int", "s": "keyword"}})).ok();
    let vectors = [
        [f32::MIN_POSITIVE, -0.0, 1e-45, 16777217.0],
        [0.1, -1.0 / 3.0, f32::EPSILON, 1.0000001],
        [8.5e18, -8.5e18, 123456.79, 5e-324f64 as f32],
    ];
    let all = [(0, i64::MIN), (u64::MAX, i64::MAX), ((1 << 53) + 1, (1 << 53) + 1)];
    let rows: Vec<Value> = all
        .iter()
        .zip(&vectors)
        .map(|((id, n), v)| json!({"id": id, "vector": v, "attrs": {"n": n, "s": "\u{0}\"\\é\u{1F600}\n"}}))
        .collect();
    c.post("/v1/collections/c/upsert", &json!({"rows": rows})).ok();
    for flushed in [false, true] {
        if flushed {
            c.ask("POST", "/v1/collections/c/flush", "").ok();
        }
        for ((id, n), v) in all.iter().zip(&vectors) {
            let filter = json!({"field": "n", "eq": n});
            let body = json!({"vector": v, "filter": filter, "with_attrs": true, "with_vectors": true});
            let got = c.post("/v1/collections/c/search", &body).ok();
            let [row] = &got["results"].as_array().unwrap()[..] else { panic!("{got}") };
            assert_eq!((row["id"].as_u64(), row["attrs"]["n"].as_i64()), (Some(*id), Some(*n)));
            assert_eq!(row["attrs"]["s"], "\u{0}\"\\é\u{1F600}\n");
            assert_eq!(row["distance"], 0.0);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&floats(&row["vector"])), bits(v));
        }
    }
    // Base64 gives the same vector.
    let b64 = "AACAPwAAIMAAAAAAAACAvw=="; // 1, -2.5, 0, -1
    c.post("/v1/collections/c/upsert", &json!({"rows": [{"id": 5, "vector": b64}]})).ok();
    let got = c.post("/v1/collections/c/get", &json!({"ids": [5]})).ok();
    assert_eq!(got["rows"], json!([{"id": 5, "vector": [1.0, -2.5, 0.0, -1.0], "attrs": {}}]));
    let got = c.post("/v1/collections/c/search", &json!({"vector": b64, "k": 1})).ok();
    assert_eq!(got["results"], json!([{"id": 5, "distance": 0.0}]));
}

#[test]
fn keys() {
    let dir = Dir::new("keys");
    let key = "0123456789abcdef";
    let server = Server::start(&dir, Config { access: Access::Key(key.into()), ..Config::default() });
    let mut c = server.client();
    c.get("/healthz").ok();
    for auth in [
        "",
        "Authorization: Bearer\r\n",
        "Authorization: Bearer \r\n",
        "Authorization: 0123456789abcdef\r\n",
        "Authorization: Basic 0123456789abcdef\r\n",
        "Authorization: Bearer 0123456789abcdeg\r\n",
        "Authorization: Bearer 0123456789abcde\r\n",
        "Authorization: Bearer 0123456789abcdef0\r\n",
        "Authorization: Bearer 0123456789ABCDEF\r\n",
        "X-Api-Key: 0123456789abcdef\r\n",
    ] {
        for (method, path) in [("GET", "/metrics"), ("GET", "/v1/collections"), ("DELETE", "/v1/collections/c"), ("GET", "/x")] {
            let head = c.head(method, path, auth);
            c.send_raw(head.as_bytes());
            let reply = c.reply();
            assert_eq!(reply.header("www-authenticate"), Some("Bearer"), "{auth}");
            reply.err(401, "unauthorized");
        }
    }
    let head = c.head("GET", "/v1/collections", "authorization: bearer 0123456789abcdef\r\nOrigin: http://x\r\n");
    c.send_raw(head.as_bytes());
    c.reply().ok();
    c.key = Some(key.into());
    c.get("/metrics").text();
    c.put("/v1/collections/c", &json!({"dim": 2, "metric": "l2"})).ok();

    // Without keys, only programs on this machine, or anyone.
    for anyone in [false, true] {
        let dir = Dir::new("no-keys");
        let access = if anyone { Access::Anyone } else { Access::Local };
        let server = Server::start(&dir, Config { access, ..Config::default() });
        for (headers, local) in [
            ("", true),
            ("Host: localhost\r\n", true),
            ("Host: LocalHost:7410\r\n", true),
            ("Host: 127.0.0.1:7410\r\n", true),
            ("Host: 10.1.2.3\r\n", true),
            ("Host: [::1]:7410\r\n", true),
            ("Host: [::1]\r\n", true),
            ("Host: localhost\r\nOrigin: http://localhost:7410\r\n", false),
            ("Host: localhost\r\nOrigin: null\r\n", false),
            ("Host: example.com:7410\r\n", false),
            ("Host: localhost.example.com\r\n", false),
            ("Host: 127.0.0.1.example.com\r\n", false),
            ("Host: [example.com]\r\n", false),
            ("Host: \r\n", false),
            ("Host: \u{e9}\r\n", false),
        ] {
            let mut c = server.client();
            c.send_raw(format!("GET /v1/collections HTTP/1.1\r\n{headers}\r\n").as_bytes());
            match local || anyone {
                true => drop(c.reply().ok()),
                false => drop(c.reply().err(403, "forbidden")),
            }
            c.get("/healthz").ok();
        }
    }
}

#[test]
fn bodies() {
    let dir = Dir::new("bodies");
    let cfg = Config {
        max_body: 1 << 20,
        max_query: 10_000,
        max_ops: 1000,
        body_idle: Duration::from_millis(500),
        body_total: Duration::from_millis(1500),
        ..Config::default()
    };
    let server = Server::start(&dir, cfg);
    let mut c = server.client();
    c.put("/v1/collections/c", &json!({"dim": 2, "metric": "l2"})).ok();
    let json_body = "Content-Type: application/json\r\n";
    let post = |more: &str| server.client().head("POST", "/v1/collections/c/upsert", more);

    // In chunks, of any size.
    let body = r#"{"rows": [{"id": 1, "vector": [1, 2]}, {"id": 2, "vector": [3, 4]}]}"#;
    for size in [1, 7, 1000] {
        let mut out = post(&format!("{json_body}Transfer-Encoding: chunked\r\n"));
        for chunk in body.as_bytes().chunks(size) {
            out += &format!("{:x}\r\n{}\r\n", chunk.len(), std::str::from_utf8(chunk).unwrap());
        }
        c.send_raw(format!("{out}0\r\n\r\n").as_bytes());
        assert_eq!(c.reply().ok()["upserted"], 2);
    }
    // Several requests at once are answered in order.
    let one = |n: usize| {
        let body = json!({"ids": vec![1; n]}).to_string();
        format!("{}{body}", c.head("POST", "/v1/collections/c/get", &format!("{json_body}Content-Length: {}\r\n", body.len())))
    };
    let all: String = (1..20).map(one).collect();
    c.send_raw(all.as_bytes());
    for n in 1..20 {
        assert_eq!(c.reply().ok()["rows"].as_array().unwrap().len(), n);
    }
    // The client waits for the go-ahead.
    c.send_raw(post(&format!("{json_body}Expect: 100-continue\r\nContent-Length: {}\r\n", body.len())).as_bytes());
    assert_eq!(c.reply().status, 100);
    c.send_raw(body.as_bytes());
    c.reply().ok();

    // Not JSON, or not plain.
    for more in ["", "Content-Type: text/plain\r\n", "Content-Type: application/jsonl\r\n"] {
        c.send_raw(format!("{}{body}", post(&format!("{more}Content-Length: {}\r\n", body.len()))).as_bytes());
        c.reply().err(415, "unsupported");
    }
    c.send_raw(format!("{}{body}", post(&format!("{json_body}Content-Encoding: gzip\r\nContent-Length: {}\r\n", body.len()))).as_bytes());
    c.reply().err(415, "unsupported");
    c.send_raw(format!("{}{body}", post(&format!("content-type: Application/JSON ; charset=utf-8\r\nContent-Length: {}\r\n", body.len()))).as_bytes());
    c.reply().ok();

    // Too large: said at once, or noticed.
    let mut c = server.client();
    c.send_raw(post(&format!("{json_body}Expect: 100-continue\r\nContent-Length: {}\r\n", (1 << 20) + 1)).as_bytes());
    c.reply().err(413, "too_large");
    let mut c = server.client();
    c.send_raw(post(&format!("{json_body}Content-Length: 99999999999999999999\r\n")).as_bytes());
    assert_eq!(c.reply().status, 400);
    let mut c = server.client();
    c.send_raw(post(&format!("{json_body}Transfer-Encoding: chunked\r\n")).as_bytes());
    let chunk = format!("10000\r\n{}\r\n", " ".repeat(0x10000));
    for _ in 0..16 {
        c.send_raw(chunk.as_bytes());
    }
    c.send_raw(b"1\r\n \r\n");
    c.reply().err(413, "too_large");
    let mut c = server.client();
    let padded = |n: usize| format!("{}{}", r#"{"vector": [1, 2]}"#, " ".repeat(n));
    c.ask("POST", "/v1/collections/c/search", &padded(9000)).ok();
    c.ask("POST", "/v1/collections/c/search", &padded(10_000)).err(413, "too_large");
    let mut c = server.client();
    let rows = |n: u64| json!({"rows": (0..n).map(|id| json!({"id": id, "vector": [id, 0]})).collect::<Vec<_>>()});
    c.post("/v1/collections/c/upsert", &rows(1000)).ok();
    c.post("/v1/collections/c/upsert", &rows(1001)).err(400, "invalid_argument");
    c.post("/v1/collections/c/delete", &json!({"ids": vec![7; 1001]})).err(400, "invalid_argument");
    assert_eq!(c.get("/v1/collections/c").ok()["rows"], 1000);

    // Too slow: nothing for a while, or too long in all.
    for (more, from, to) in [(false, 400, 1300), (true, 1400, 10_000)] {
        let mut slow = server.client();
        slow.send_raw(post(&format!("{json_body}Content-Length: 1000\r\n")).as_bytes()).send_raw(b"{");
        let start = Instant::now();
        let mut writer = slow.stream().try_clone().unwrap();
        let done = AtomicBool::new(false);
        std::thread::scope(|s| {
            let done = &done;
            s.spawn(move || {
                while more && !done.load(Ordering::Relaxed) && writer.write_all(b" ").is_ok() {
                    std::thread::sleep(Duration::from_millis(20));
                }
            });
            slow.reply().err(408, "timeout");
            done.store(true, Ordering::Relaxed);
        });
        let took = start.elapsed().as_millis();
        assert!((from..to).contains(&took), "{more}: {took}");
    }

    // A client that goes away leaves nothing behind.
    for n in [100, 900_000] {
        let mut gone = server.client();
        gone.send_raw(post(&format!("{json_body}Content-Length: {n}\r\n")).as_bytes()).send_raw(b"{");
        if n > 256 << 10 {
            server.wait_metric("vecdb_large_body_bytes", |n| n > 0.0);
        }
    }
    server.wait_metric("vecdb_large_body_bytes", |n| n == 0.0);
    server.wait_metric("vecdb_active{kind=\"write\"}", |n| n == 0.0);
    server.client().get("/v1/collections/c").ok();
}

/// Connections with nothing to say are closed.
#[test]
fn idle_connections() {
    let dir = Dir::new("idle");
    let server = Server::start(&dir, Config { idle: Duration::from_millis(300), ..Config::default() });
    for (first, partial) in [(false, ""), (false, "GET /healthz HTTP/1.1\r\nHo"), (true, ""), (true, "GET /hea")] {
        let mut c = server.client();
        if first {
            c.get("/healthz").ok();
        }
        c.send_raw(partial.as_bytes());
        let start = Instant::now();
        let mut rest = Vec::new();
        let _ = c.stream().read_to_end(&mut rest);
        let took = start.elapsed();
        assert!((200..20_000).contains(&took.as_millis()), "{first} {partial:?}: {took:?}");
    }
    server.client().get("/healthz").ok();
}

/// Beyond what is worked on and what may wait, requests are refused at once.
#[test]
fn overload() {
    let dir = Dir::new("overload");
    let cfg = Config { searches: 1, queue: 2, ..Config::default() };
    let server = Server::start(&dir, cfg);
    let mut c = server.client();
    c.put("/v1/collections/c", &json!({"dim": 64, "metric": "l2"})).ok();
    let mut rng = SplitMix64::new(2);
    for part in 0..10 {
        let rows: Vec<Value> = (0..1000).map(|i| json!({"id": part * 1000 + i, "vector": random(&mut rng, 64)})).collect();
        c.post("/v1/collections/c/upsert", &json!({"rows": rows})).ok();
    }
    let body = json!({"vector": random(&mut rng, 64), "k": 1000}).to_string();
    let (done, refused) = (AtomicUsize::new(0), AtomicUsize::new(0));
    std::thread::scope(|s| {
        for _ in 0..12 {
            s.spawn(|| {
                let mut c = server.client();
                for _ in 0..40 {
                    let reply = c.ask("POST", "/v1/collections/c/search", &body);
                    if reply.status == 200 {
                        assert_eq!(ids(&reply.json()).len(), 1000);
                        done.fetch_add(1, Ordering::Relaxed);
                    } else {
                        assert_eq!(reply.header("retry-after"), Some("1"));
                        reply.err(503, "busy");
                        refused.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
        }
        // The others' places are their own.
        let mut c = server.client();
        for id in 0..20 {
            c.post("/v1/collections/c/delete", &json!({"ids": [id]})).ok();
            c.get("/v1/collections").ok();
        }
    });
    let (done, refused) = (done.into_inner(), refused.into_inner());
    assert!(done > 0 && refused > 0, "{done} {refused}");
    assert_eq!(server.metric("vecdb_requests_total{op=\"search\",code=\"200\"}"), done as f64);
    assert_eq!(server.metric("vecdb_requests_total{op=\"search\",code=\"503\"}"), refused as f64);
    for kind in ["search", "write", "other", "look"] {
        // (One of those asks this.)
        let own = if kind == "look" { 1.0 } else { 0.0 };
        server.wait_metric(&format!("vecdb_active{{kind=\"{kind}\"}}"), |n| n == own);
        server.wait_metric(&format!("vecdb_waiting{{kind=\"{kind}\"}}"), |n| n == 0.0);
    }
}

/// A write that has to wait for an index build is refused after a while. It
/// keeps its place, and its body's part of the budget, until then.
#[test]
fn writes_that_wait() {
    const DIM: usize = 256;
    let dir = Dir::new("wait");
    let shard = OpenOptions {
        durability: Durability::Buffered,
        direct_io: false,
        threads: 1,
        max_unindexed_bytes: 8 << 20,
        max_write_wait: Some(Duration::from_millis(1500)),
        ..OpenOptions::default()
    };
    let options = DatabaseOptions { shard, search_threads: 1, search_cpus: 0, builds: 1 };
    let server = Server::of(Database::open(&dir.0, &options).unwrap().0, Config::default());
    let mut c = server.client();
    let mut rng = SplitMix64::new(4);
    // (A row is 1 KiB.)
    let mut upsert = |name: &str, from: u64| {
        let rows: Vec<Value> = (from..from + 500).map(|id| json!({"id": id, "vector": random(&mut rng, DIM)})).collect();
        (format!("/v1/collections/{name}/upsert"), json!({"rows": rows}))
    };
    // The only build there may be at a time, and a long one.
    c.put("/v1/collections/slow", &json!({"dim": DIM, "metric": "l2", "seal_bytes": 1 << 20, "index_bytes": 4 << 20,
        "index": {"build_list": 1000, "passes": 4}})).ok();
    for from in (0..5500).step_by(500) {
        let (path, body) = upsert("slow", from);
        c.post(&path, &body).ok();
    }
    c.ask("POST", "/v1/collections/slow/flush", "").ok();
    // Another collection that needs one.
    let settings = json!({"dim": DIM, "metric": "l2", "seal_bytes": 1 << 20, "index_bytes": 2200 << 10});
    c.put("/v1/collections/c", &settings).ok();
    let (mut rows, mut last) = (0, 200);
    let refused = loop {
        let (path, body) = upsert("c", rows);
        let start = Instant::now();
        let reply = c.post(&path, &body);
        match (reply.status, last) {
            (200, _) => rows += 500,
            (_, 200) => assert!(start.elapsed() > Duration::from_millis(1400), "{:?}", start.elapsed()),
            // (After one refusal the next may go. Not after two.)
            _ => break reply,
        }
        last = reply.status;
        assert!(rows < 20_000);
    };
    assert_eq!(refused.header("retry-after"), Some("1"));
    let why = refused.err(503, "unavailable");
    assert!(why.contains("index builds are behind"), "{why}");
    // (Slowed down from 6144 rows on; 8192 are the limit. The last 500 took
    // 3 s, the next would take 44.)
    assert_eq!(rows, 8000);
    let info = c.get("/v1/collections/c").ok();
    assert_eq!((&info["rows"], &info["state"]["slowing_writes"]), (&json!(rows), &json!(true)));
    assert_eq!(server.metric("vecdb_shards_slowing_writes{collection=\"c\"}"), 1.0);
    assert_eq!(server.metric("vecdb_shards_slowing_writes{collection=\"slow\"}"), 0.0);
    assert_eq!(server.metric("vecdb_write_wait_seconds_total{collection=\"slow\"}"), 0.0);
    let waited = server.metric("vecdb_write_wait_seconds_total{collection=\"c\"}");
    assert!((1.4..10.0).contains(&waited), "{waited}");
    let (path, body) = upsert("c", rows);
    std::thread::scope(|s| {
        s.spawn(|| server.client().post(&path, &body).err(503, "unavailable"));
        server.wait_metric("vecdb_active{kind=\"write\"}", |n| n == 1.0);
        let text = c.get("/metrics").text();
        let active = "vecdb_active{kind=\"write\"} 1\n";
        assert!(text.contains(active) && !text.contains("vecdb_large_body_bytes 0\n"), "{text}");
        // Others go on.
        c.post("/v1/collections/slow/delete", &json!({"ids": [1]})).ok();
        c.post("/v1/collections/c/search", &json!({"vector": vec![0.0; DIM]})).ok();
    });
    // Also when its client goes away: the engine does not.
    let mut gone = server.client();
    let text = body.to_string();
    let more = format!("Content-Type: application/json\r\nContent-Length: {}\r\n", text.len());
    let head = gone.head("POST", &path, &more);
    gone.send_raw(head.as_bytes()).send_raw(text.as_bytes());
    server.wait_metric("vecdb_active{kind=\"write\"}", |n| n == 1.0);
    drop(gone);
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(server.metric("vecdb_active{kind=\"write\"}"), 1.0);
    server.wait_metric("vecdb_active{kind=\"write\"}", |n| n == 0.0);
    // Without the long one, builds and writes catch up.
    c.delete("/v1/collections/slow").ok();
    wait("a write", || c.post(&path, &body).status == 200);
    assert_eq!(c.get("/v1/collections/c").ok()["rows"], rows + 500);
    server.wait_metric("vecdb_shards_slowing_writes{collection=\"c\"}", |n| n == 0.0);
    assert_eq!(c.get("/v1/collections/c").ok()["state"]["slowing_writes"], false);
    server.wait_metric("vecdb_large_body_bytes", |n| n == 0.0);
}

/// Large replies share a budget until their clients have them.
#[test]
fn large_replies() {
    const DIM: usize = 1024;
    let dir = Dir::new("replies");
    let cfg = Config {
        max_reply: 36 << 20,
        max_replies: 36 << 20,
        body_total: Duration::from_secs(5),
        ..Config::default()
    };
    let server = Server::start(&dir, cfg);
    let mut c = server.client();
    c.put("/v1/collections/c", &json!({"dim": DIM, "metric": "ip", "fields": {"s": "keyword"}})).ok();
    let mut rng = SplitMix64::new(3);
    for part in 0..10 {
        let rows: Vec<Value> = (0..200)
            .map(|i| {
                let v: Vec<f32> = (0..DIM).map(|_| (rng.next_u64() % 1_000_000) as f32 / 1e7).collect();
                json!({"id": part * 200 + i, "vector": v, "attrs": {"s": vec!["x".repeat(1000); 25]}})
            })
            .collect();
        c.post("/v1/collections/c/upsert", &json!({"rows": rows})).ok();
    }
    let search = |k: usize, attrs: bool, vectors: bool| {
        json!({"vector": vec![1.0; DIM], "k": k, "with_attrs": attrs, "with_vectors": vectors}).to_string()
    };
    let path = "/v1/collections/c/search";
    // More than there may be: known beforehand, or found on the way.
    c.ask("POST", path, &search(2305, false, true)).err(400, "invalid_argument");
    let why = c.ask("POST", path, &search(1800, true, false)).err(400, "invalid_argument");
    assert!(why.contains("37748736 bytes"), "{why}");
    let all: Vec<u64> = (0..2000).collect();
    c.post("/v1/collections/c/get", &json!({"ids": all})).err(400, "invalid_argument");
    c.ask("POST", path, &search(10_000, false, false)).ok();

    // One that is not taken by its client.
    let large = search(2000, false, true);
    let mut slow = server.client();
    let head = slow.head("POST", path, &format!("Content-Type: application/json\r\nContent-Length: {}\r\n", large.len()));
    slow.send_raw(format!("{head}{large}").as_bytes());
    server.wait_metric("vecdb_large_reply_bytes", |n| n > (15 << 20) as f64);
    // There is no place for another like it, only for smaller ones.
    let mut other = server.client();
    other.ask("POST", path, &large).err(503, "busy");
    other.ask("POST", path, &search(20, true, false)).ok();
    other.ask("POST", path, &search(200, true, false)).ok();
    // Taken after all.
    assert_eq!(ids(&slow.reply().ok()).len(), 2000);
    server.wait_metric("vecdb_large_reply_bytes", |n| n == 0.0);
    assert_eq!(ids(&other.ask("POST", path, &large).ok()).len(), 2000);
    // Its client goes away instead.
    slow.send_raw(format!("{head}{large}").as_bytes());
    server.wait_metric("vecdb_large_reply_bytes", |n| n > 0.0);
    drop(slow);
    server.wait_metric("vecdb_large_reply_bytes", |n| n == 0.0);
    assert_eq!(ids(&other.ask("POST", path, &large).ok()).len(), 2000);
    // Or it stays, and takes nothing: not for ever.
    let mut stuck = server.client();
    stuck.send_raw(format!("{head}{large}").as_bytes());
    server.wait_metric("vecdb_large_reply_bytes", |n| n > 0.0);
    let start = Instant::now();
    server.wait_metric("vecdb_large_reply_bytes", |n| n == 0.0);
    assert!(start.elapsed() > Duration::from_secs(3), "{:?}", start.elapsed());
    let mut rest = Vec::new();
    let _ = stuck.stream().read_to_end(&mut rest);
    assert!(rest.len() < 20 << 20);

    // One optimize at a time.
    assert_eq!(c.ask("POST", "/v1/collections/c/optimize", "").status, 202);
    c.ask("POST", "/v1/collections/c/optimize", "").err(409, "already_running");
    wait("optimize", || c.get("/v1/collections/c").ok()["state"]["optimizing"] == false);
    assert_eq!(c.ask("POST", "/v1/collections/c/optimize", "").status, 202);
    assert_eq!(ids(&c.ask("POST", path, &search(2000, false, false)).ok()).len(), 2000);
    // (Not cut off by the end of the test, which removes its files.)
    wait("optimize", || c.get("/v1/collections/c").ok()["state"]["optimizing"] == false);
}

/// The directories of collections, oldest first.
fn collections(dir: &Dir) -> Vec<std::path::PathBuf> {
    let all = std::fs::read_dir(&dir.0).unwrap().map(|e| e.unwrap().path());
    let mut all: Vec<_> = all.filter(|p| p.is_dir()).collect();
    all.sort();
    all
}

/// A collection is dropped, and another created in its place, under those
/// that use it.
#[test]
fn drops_under_load() {
    let dir = Dir::new("drops");
    let server = Server::start(&dir, Config::default());
    let stop = AtomicBool::new(false);
    let seen = [(); 4].map(|_| AtomicUsize::new(0));
    std::thread::scope(|s| {
        for t in 0..4u64 {
            let (server, stop, seen) = (&server, &stop, &seen);
            s.spawn(move || {
                let mut c = server.client();
                let mut rng = SplitMix64::new(t);
                while !stop.load(Ordering::Relaxed) {
                    let v = random(&mut rng, 4);
                    let reply = match t % 2 {
                        0 => c.post("/v1/collections/c/search", &json!({"vector": v, "with_vectors": true})),
                        _ => {
                            let row = json!({"id": rng.next_u64() % 1000, "vector": v});
                            c.post("/v1/collections/c/upsert", &json!({"rows": [row]}))
                        }
                    };
                    match reply.status {
                        200 => seen[(t % 2) as usize].fetch_add(1, Ordering::Relaxed),
                        404 => seen[2 + (t % 2) as usize].fetch_add(1, Ordering::Relaxed),
                        _ => panic!("{}", reply.text()),
                    };
                }
            });
        }
        let mut c = server.client();
        for i in 0..20 {
            c.put("/v1/collections/c", &json!({"dim": 4, "metric": "l2", "shards": 1 + i % 3, "seal_bytes": 1024})).ok();
            std::thread::sleep(Duration::from_millis(40));
            if i % 2 == 0 {
                c.ask("POST", "/v1/collections/c/optimize", "").ok();
            }
            c.delete("/v1/collections/c").ok();
            std::thread::sleep(Duration::from_millis(10));
        }
        stop.store(true, Ordering::Relaxed);
    });
    assert!(seen.iter().all(|n| n.load(Ordering::Relaxed) > 0), "{seen:?}");
    // Their files go with the last who uses them.
    wait("files", || collections(&dir).is_empty());
}

/// One that is damaged can be listed and dropped; the others are served.
#[test]
fn a_damaged_collection() {
    let dir = Dir::new("damaged");
    let mut server = Server::start(&dir, Config::default());
    let mut c = server.client();
    for name in ["bad", "good"] {
        c.put(&format!("/v1/collections/{name}"), &json!({"dim": 2, "metric": "l2"})).ok();
        let rows = json!({"rows": [{"id": 1, "vector": [1, 2]}]});
        c.post(&format!("/v1/collections/{name}/upsert"), &rows).ok();
        c.ask("POST", &format!("/v1/collections/{name}/flush"), "").ok();
    }
    drop(c);
    assert!(server.stopped());
    let all = collections(&dir);
    assert_eq!(all.len(), 2);
    let mut files = Vec::new();
    let mut dirs = vec![all[0].clone()];
    while let Some(d) = dirs.pop() {
        for e in std::fs::read_dir(d).unwrap().map(|e| e.unwrap().path()) {
            if e.is_dir() { dirs.push(e) } else { files.push(e) }
        }
    }
    assert!(files.len() > 2, "{files:?}");
    files.iter().for_each(|f| std::fs::write(f, b"damaged").unwrap());

    let server = Server::start(&dir, Config::default());
    let mut c = server.client();
    for (method, path, body) in [
        ("GET", "", ""),
        ("POST", "/search", r#"{"vector": [1, 2]}"#),
        ("POST", "/get", r#"{"ids": [1]}"#),
        ("POST", "/upsert", r#"{"rows": [{"id": 1, "vector": [1, 2]}]}"#),
        ("POST", "/flush", ""),
        ("POST", "/optimize", ""),
    ] {
        c.ask(method, &format!("/v1/collections/bad{path}"), body).err(503, "unavailable");
        assert_eq!(c.ask(method, &format!("/v1/collections/good{path}"), body).status / 100, 2);
    }
    let all = c.get("/v1/collections").ok();
    assert!(all["collections"][0]["error"].as_str().unwrap().contains("could not be opened"), "{all}");
    assert_eq!(all["collections"][1], json!({"name": "good", "rows": 1}));
    assert_eq!(server.metric("vecdb_up{collection=\"bad\"}"), 0.0);
    assert_eq!(server.metric("vecdb_up{collection=\"good\"}"), 1.0);
    c.put("/v1/collections/bad", &json!({"dim": 2, "metric": "l2"})).err(409, "already_exists");
    c.delete("/v1/collections/bad").ok();
    c.put("/v1/collections/bad", &json!({"dim": 2, "metric": "l2"})).ok();
    assert_eq!(c.get("/v1/collections/bad").ok()["rows"], 0);
    wait("optimize", || c.get("/v1/collections/good").ok()["state"]["optimizing"] == false);
}

#[test]
fn metrics() {
    let dir = Dir::new("metrics");
    let server = Server::start(&dir, Config::default());
    let mut c = server.client();
    c.put("/v1/collections/c.1", &json!({"dim": 2, "metric": "l2", "shards": 2})).ok();
    let rows: Vec<Value> = (0..50).map(|id| json!({"id": id, "vector": [id, 1]})).collect();
    c.post("/v1/collections/c.1/upsert", &json!({"rows": rows})).ok();
    c.post("/v1/collections/c.1/delete", &json!({"ids": [1, 2, 99]})).ok();
    c.post("/v1/collections/c.1/search", &json!({"vector": [0, 0, 0]})).err(400, "invalid_argument");
    c.ask("GET", "/nothing", "");
    let reply = c.get("/metrics");
    assert_eq!(reply.header("content-type"), Some("text/plain; version=0.0.4"));
    let text = reply.text();
    for line in [
        "vecdb_requests_total{op=\"create\",code=\"201\"} 1",
        "vecdb_requests_total{op=\"upsert\",code=\"200\"} 1",
        "vecdb_requests_total{op=\"search\",code=\"400\"} 1",
        "vecdb_requests_total{op=\"other\",code=\"404\"} 1",
        "vecdb_request_seconds_count{op=\"upsert\"} 1",
        "vecdb_active{kind=\"look\"} 1",
        "vecdb_active{kind=\"search\"} 0",
        "vecdb_up{collection=\"c.1\"} 1",
        "vecdb_rows{collection=\"c.1\"} 48",
        "vecdb_shards_failing_builds{collection=\"c.1\"} 0",
    ] {
        assert!(text.lines().any(|l| l == line), "{line}\n{text}");
    }
    // Each is declared once, and before its values.
    let mut declared = std::collections::HashSet::new();
    for l in text.lines() {
        if let Some(rest) = l.strip_prefix("# TYPE ") {
            assert!(declared.insert(rest.split(' ').next().unwrap().to_string()), "{l}");
        } else {
            let name = l.split(['{', ' ']).next().unwrap();
            let base = ["_bucket", "_sum", "_count"].iter().find_map(|s| name.strip_suffix(s));
            assert!(declared.contains(name) || base.is_some_and(|b| declared.contains(b)), "{l}");
        }
    }
}

/// What is being answered is answered; nothing new is.
#[test]
fn stops_gracefully() {
    let dir = Dir::new("stop");
    let mut server = Server::start(&dir, Config { drain: Duration::from_secs(20), ..Config::default() });
    let mut c = server.client();
    c.put("/v1/collections/c", &json!({"dim": 2, "metric": "l2"})).ok();
    let body = r#"{"rows": [{"id": 1, "vector": [1, 2]}]}"#;
    let mut half = server.client();
    let more = format!("Content-Type: application/json\r\nContent-Length: {}\r\n", body.len());
    let head = half.head("POST", "/v1/collections/c/upsert", &more);
    half.send_raw(head.as_bytes()).send_raw(&body.as_bytes()[..10]);
    std::thread::sleep(Duration::from_millis(200));
    server.stop();
    // An idle connection is closed.
    assert!(c.try_reply().unwrap().is_none());
    wait("the listener to close", || TcpStream::connect(server.addr).is_err());
    half.send_raw(&body.as_bytes()[10..]);
    assert_eq!(half.reply().ok()["upserted"], 1);
    assert!(server.stopped());

    // Not for ever.
    let mut server = Server::start(&dir, Config { drain: Duration::from_millis(300), ..Config::default() });
    assert_eq!(server.client().get("/v1/collections/c").ok()["rows"], 1);
    let mut half = server.client();
    half.send_raw(head.as_bytes()).send_raw(&body.as_bytes()[..10]);
    std::thread::sleep(Duration::from_millis(200));
    assert!(!server.stopped());
}

/// Only so many connections; more wait to be accepted.
#[test]
fn connections_are_limited() {
    let dir = Dir::new("connections");
    let server = Server::start(&dir, Config { max_connections: 2, ..Config::default() });
    let mut held: Vec<_> = (0..2).map(|_| server.client()).collect();
    held.iter_mut().for_each(|c| drop(c.get("/healthz").ok()));
    let mut third = server.client();
    let head = third.head("GET", "/healthz", "");
    third.send_raw(head.as_bytes());
    third.stream().set_read_timeout(Some(Duration::from_millis(300))).unwrap();
    assert!(third.stream().peek(&mut [0]).is_err());
    drop(held.pop());
    third.stream().set_read_timeout(Some(Duration::from_secs(60))).unwrap();
    third.reply().ok();
}
