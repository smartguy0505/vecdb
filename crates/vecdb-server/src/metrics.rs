//! Counters, in the Prometheus text format.

use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// What requests are counted by.
pub const OPS: [&str; 14] = [
    "create", "delete", "drop", "flush", "get", "health", "info", "list", "metrics", "optimize", "other", "search",
    "upsert", "write",
];
const CODES: [u16; 15] = [200, 201, 202, 400, 401, 403, 404, 405, 408, 409, 413, 415, 500, 503, 0];
/// Upper bounds of the latency buckets, in microseconds.
const BOUNDS: [u64; 16] = [
    250, 500, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000, 256_000, 512_000, 1_024_000, 4_096_000,
    16_384_000, 65_536_000,
];

#[derive(Default)]
struct Op {
    codes: [AtomicU64; CODES.len()],
    buckets: [AtomicU64; BOUNDS.len() + 1],
    micros: AtomicU64,
}

#[derive(Default)]
pub struct Metrics {
    ops: [Op; OPS.len()],
}

impl Metrics {
    /// Counts a request that was answered with `code` after `took`.
    pub fn record(&self, op: &str, code: u16, took: Duration) {
        let op = &self.ops[OPS.iter().position(|&o| o == op).expect("a known op")];
        let code = CODES.iter().position(|&c| c == code).unwrap_or(CODES.len() - 1);
        let micros = took.as_micros() as u64;
        let bucket = BOUNDS.partition_point(|&b| b < micros);
        for c in [&op.codes[code], &op.buckets[bucket]] {
            c.fetch_add(1, Ordering::Relaxed);
        }
        op.micros.fetch_add(micros, Ordering::Relaxed);
    }

    pub fn write(&self, out: &mut String) {
        let _ = writeln!(out, "# TYPE vecdb_requests_total counter");
        for (name, op) in OPS.iter().zip(&self.ops) {
            for (code, n) in CODES.iter().zip(&op.codes) {
                let n = n.load(Ordering::Relaxed);
                if n > 0 {
                    let _ = writeln!(out, "vecdb_requests_total{{op=\"{name}\",code=\"{code}\"}} {n}");
                }
            }
        }
        let _ = writeln!(out, "# TYPE vecdb_request_seconds histogram");
        for (name, op) in OPS.iter().zip(&self.ops) {
            let counts: Vec<u64> = op.buckets.iter().map(|b| b.load(Ordering::Relaxed)).collect();
            let total: u64 = counts.iter().sum();
            if total == 0 {
                continue;
            }
            let mut below = 0;
            for (bound, n) in BOUNDS.iter().zip(&counts) {
                below += n;
                let le = *bound as f64 / 1e6;
                let _ = writeln!(out, "vecdb_request_seconds_bucket{{op=\"{name}\",le=\"{le}\"}} {below}");
            }
            let sum = op.micros.load(Ordering::Relaxed) as f64 / 1e6;
            let _ = writeln!(out, "vecdb_request_seconds_bucket{{op=\"{name}\",le=\"+Inf\"}} {total}");
            let _ = writeln!(out, "vecdb_request_seconds_sum{{op=\"{name}\"}} {sum}");
            let _ = writeln!(out, "vecdb_request_seconds_count{{op=\"{name}\"}} {total}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts() {
        let m = Metrics::default();
        for (op, code, micros) in
            [("search", 200, 250), ("search", 200, 251), ("search", 503, 70_000_000), ("write", 418, 0)]
        {
            m.record(op, code, Duration::from_micros(micros));
        }
        let mut out = String::new();
        m.write(&mut out);
        for line in [
            "vecdb_requests_total{op=\"search\",code=\"200\"} 2",
            "vecdb_requests_total{op=\"search\",code=\"503\"} 1",
            "vecdb_requests_total{op=\"write\",code=\"0\"} 1",
            "vecdb_request_seconds_bucket{op=\"search\",le=\"0.00025\"} 1",
            "vecdb_request_seconds_bucket{op=\"search\",le=\"0.0005\"} 2",
            "vecdb_request_seconds_bucket{op=\"search\",le=\"65.536\"} 2",
            "vecdb_request_seconds_bucket{op=\"search\",le=\"+Inf\"} 3",
            "vecdb_request_seconds_sum{op=\"search\"} 70.000501",
            "vecdb_request_seconds_count{op=\"search\"} 3",
        ] {
            assert!(out.lines().any(|l| l == line), "{line}\n{out}");
        }
        assert!(!out.contains("op=\"flush\""));
    }
}
