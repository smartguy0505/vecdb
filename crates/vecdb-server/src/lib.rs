//! An HTTP/1.1 server for a vecdb database, with JSON requests and replies.
//!
//! ```text
//! GET    /healthz                            200 once the database is open
//! GET    /metrics                            counters, in the Prometheus text format
//! GET    /v1/collections                     names and states
//! PUT    /v1/collections/{name}              create
//! GET    /v1/collections/{name}              settings and state
//! DELETE /v1/collections/{name}              drop, with its data
//! POST   /v1/collections/{name}/upsert       {"rows": [{"id", "vector", "attrs"}, ...]}
//! POST   /v1/collections/{name}/delete       {"ids": [...]}
//! POST   /v1/collections/{name}/write        {"ops": [{"upsert": row} or {"delete": {"id"}}, ...]}
//! POST   /v1/collections/{name}/get          {"ids": [...]}
//! POST   /v1/collections/{name}/search       {"vector", "k", "l", "filter", ...}
//! POST   /v1/collections/{name}/flush        writes what is in RAM to segment files
//! POST   /v1/collections/{name}/optimize     merges each shard's segments (starts it: 202)
//! ```
//!
//! **Threads.** The engine's calls block (on the disk, on locks, on writes
//! that wait for an index build), so they run on threads of their own, and a
//! few others do the networking. Searches, writes, requests that can take
//! long and those that only look each have a [`Gate`](gate::Gate): so many at
//! a time, so many waiting, for so long. Beyond that they are refused with
//! 503 and `Retry-After`, instead of all of them becoming slow. Writes cannot
//! take the places of searches.
//!
//! **Memory.** Bodies and replies have a size limit, and large ones a budget
//! they share, until they have been worked on or sent. What a request may ask
//! for (`k`, `l`, the number of rows) is bounded.
//!
//! **Writes** are not atomic across shards (see
//! [`ShardedCollection::write`](vecdb_core::ShardedCollection::write)). A
//! write that fails with 5xx, or whose reply is lost, may have been applied
//! in part or in whole. Sending it again is safe: upserts replace, and
//! deleting what is gone does nothing.

#![forbid(unsafe_code)]

#[doc(hidden)]
/// Writes a line to the standard error. Unlike `eprintln!`, does without if
/// nobody reads it.
#[macro_export]
macro_rules! log {
    ($($args:tt)*) => {{
        use std::io::Write as _;
        let _ = writeln!(std::io::stderr(), $($args)*);
    }};
}

pub mod app;
mod base64;
pub mod gate;
pub mod json;
pub mod metrics;

pub use app::{Access, App, Config};
