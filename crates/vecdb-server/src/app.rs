//! Requests and replies.

use std::collections::HashMap;
use std::convert::Infallible;
use std::fmt::Write as _;
use std::future::Future;
use std::net::IpAddr;
use std::pin::pin;
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::{Body, Incoming};
use hyper::header::{
    ALLOW, AUTHORIZATION, CONTENT_ENCODING, CONTENT_TYPE, HOST, HeaderMap, ORIGIN, RETRY_AFTER, WWW_AUTHENTICATE,
};
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::server::graceful::GracefulShutdown;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::json;
use socket2::SockRef;
use tokio::net::TcpListener;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use vecdb_core::{CollectionStats, Database, Error, Fetch, Op, ShardedCollection};

use crate::gate::{self, Gate};
use crate::json as dto;
use crate::metrics::Metrics;

/// Who may ask for things.
#[derive(Clone, Debug)]
pub enum Access {
    /// Who has the key: requests have to carry it as a bearer token.
    Key(String),
    /// Programs on this machine, if the server listens there only. Not web
    /// pages in its browsers, which could reach the server as well.
    Local,
    Anyone,
}

/// How the server behaves.
#[derive(Clone, Debug)]
pub struct Config {
    pub access: Access,
    /// Requests worked on at a time.
    pub searches: usize,
    pub writes: usize,
    /// Of those that can take long (create, drop, flush, optimize), and of
    /// those that only look (list, info, metrics).
    pub others: usize,
    pub looks: usize,
    /// Requests of each kind that may wait, and for how long.
    pub queue: usize,
    pub queue_wait: Duration,
    /// Largest body of a write, and of another request.
    pub max_body: usize,
    pub max_query: usize,
    /// Bytes of large bodies there may be at a time.
    pub max_bodies: usize,
    /// About the largest reply, and bytes of large replies there may be at a
    /// time (until their clients have them).
    pub max_reply: usize,
    pub max_replies: usize,
    pub max_k: usize,
    pub max_l: usize,
    /// Writes in a request, or rows it may get.
    pub max_ops: usize,
    /// Shards of a collection, and of all.
    pub max_shards: usize,
    pub max_total_shards: usize,
    pub max_connections: usize,
    /// A connection with no request, or with a part of one's head, is closed
    /// after this.
    pub idle: Duration,
    /// Longest wait for more of a body, and for all of it.
    pub body_idle: Duration,
    pub body_total: Duration,
    /// Time for requests to finish when the server stops.
    pub drain: Duration,
    /// Requests that take longer are logged.
    pub slow: Option<Duration>,
}

impl Default for Config {
    fn default() -> Self {
        let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
        Self {
            access: Access::Local,
            searches: 2 * cpus,
            writes: 64,
            others: 8,
            looks: 4,
            queue: 1024,
            queue_wait: Duration::from_secs(10),
            max_body: 64 << 20,
            max_query: 4 << 20,
            max_bodies: 512 << 20,
            max_reply: 64 << 20,
            max_replies: 512 << 20,
            max_k: 10_000,
            max_l: 10_000,
            max_ops: 100_000,
            max_shards: 64,
            max_total_shards: 1024,
            max_connections: 4096,
            idle: Duration::from_secs(75),
            body_idle: Duration::from_secs(10),
            body_total: Duration::from_secs(120),
            drain: Duration::from_secs(30),
            slow: None,
        }
    }
}

impl Config {
    /// The most threads the engine's calls take at a time.
    pub fn threads(&self) -> usize {
        self.searches + self.writes + self.others + self.looks
    }
}

/// Bodies and replies up to this need nothing of the budgets for large ones.
const SMALL: usize = 256 << 10;
/// Those are counted in units of this.
const UNIT: usize = 1 << 10;

/// What may be wrong with a request, or with the server.
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
    allow: Option<&'static str>,
}

impl ApiError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self { status, code, message: message.into(), allow: None }
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_argument", message)
    }

    fn busy(what: &str) -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, "busy", format!("too many {what} at once: try again later"))
    }
}

impl From<Error> for ApiError {
    fn from(e: Error) -> Self {
        let (status, code) = match &e {
            Error::DimensionMismatch { .. }
            | Error::NonFiniteValue
            | Error::DuplicateId(_)
            | Error::InvalidArgument(_)
            | Error::Unsupported(_) => (StatusCode::BAD_REQUEST, "invalid_argument"),
            Error::NotFound(_) => (StatusCode::NOT_FOUND, "not_found"),
            Error::AlreadyExists(_) => (StatusCode::CONFLICT, "already_exists"),
            Error::Unavailable(_) | Error::Cancelled => (StatusCode::SERVICE_UNAVAILABLE, "unavailable"),
            Error::Corrupt(_) | Error::Io(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal"),
        };
        Self::new(status, code, e.to_string())
    }
}

/// A reply, ready to send.
pub struct Reply {
    status: StatusCode,
    content_type: &'static str,
    body: Vec<u8>,
    allow: Option<&'static str>,
    /// The body's part of the budget, if it is large.
    budget: Option<OwnedSemaphorePermit>,
}

impl Reply {
    fn json(status: StatusCode, value: &impl Serialize) -> Self {
        let body = serde_json::to_vec(value).expect("plain data");
        Self { status, content_type: "application/json", body, allow: None, budget: None }
    }

    fn ok(value: &impl Serialize) -> Self {
        Self::json(StatusCode::OK, value)
    }
}

impl From<ApiError> for Reply {
    fn from(e: ApiError) -> Self {
        let body = json!({ "error": { "code": e.code, "message": e.message } });
        Self { allow: e.allow, ..Self::json(e.status, &body) }
    }
}

/// What is asked for.
#[derive(Debug, PartialEq)]
enum Route {
    Health,
    Metrics,
    List,
    Create(String),
    Info(String),
    Drop(String),
    Upsert(String),
    Delete(String),
    Write(String),
    Get(String),
    Search(String),
    Flush(String),
    Optimize(String),
}

impl Route {
    fn parse(method: &Method, path: &str) -> Result<Self, ApiError> {
        let parts: Vec<&str> = path.strip_prefix('/').unwrap_or("").split('/').collect();
        let allow = |m: &'static str| ApiError {
            allow: Some(m),
            ..ApiError::new(StatusCode::METHOD_NOT_ALLOWED, "method_not_allowed", format!("use {m}"))
        };
        let get = |r: Self| if method == Method::GET { Ok(r) } else { Err(allow("GET")) };
        let post = |r: fn(String) -> Self, name: &str| {
            if method == Method::POST { Ok(r(name.into())) } else { Err(allow("POST")) }
        };
        match parts[..] {
            ["healthz"] => get(Self::Health),
            ["metrics"] => get(Self::Metrics),
            ["v1", "collections"] => get(Self::List),
            ["v1", "collections", name] if !name.is_empty() => match *method {
                Method::GET => Ok(Self::Info(name.into())),
                Method::PUT => Ok(Self::Create(name.into())),
                Method::DELETE => Ok(Self::Drop(name.into())),
                _ => Err(allow("GET, PUT, DELETE")),
            },
            ["v1", "collections", name, "upsert"] => post(Self::Upsert, name),
            ["v1", "collections", name, "delete"] => post(Self::Delete, name),
            ["v1", "collections", name, "write"] => post(Self::Write, name),
            ["v1", "collections", name, "get"] => post(Self::Get, name),
            ["v1", "collections", name, "search"] => post(Self::Search, name),
            ["v1", "collections", name, "flush"] => post(Self::Flush, name),
            ["v1", "collections", name, "optimize"] => post(Self::Optimize, name),
            _ => Err(ApiError::new(StatusCode::NOT_FOUND, "not_found", "no such path")),
        }
    }

    /// For the metrics.
    fn op(&self) -> &'static str {
        match self {
            Self::Health => "health",
            Self::Metrics => "metrics",
            Self::List => "list",
            Self::Create(_) => "create",
            Self::Info(_) => "info",
            Self::Drop(_) => "drop",
            Self::Upsert(_) => "upsert",
            Self::Delete(_) => "delete",
            Self::Write(_) => "write",
            Self::Get(_) => "get",
            Self::Search(_) => "search",
            Self::Flush(_) => "flush",
            Self::Optimize(_) => "optimize",
        }
    }

    /// The largest body it may have, if it has one.
    fn max_body(&self, cfg: &Config) -> Option<usize> {
        match self {
            Self::Upsert(_) | Self::Delete(_) | Self::Write(_) => Some(cfg.max_body),
            Self::Create(_) | Self::Get(_) | Self::Search(_) => Some(cfg.max_query.min(cfg.max_body)),
            _ => None,
        }
    }
}

/// A reply's bytes, and their part of the budget until they have been sent.
struct Sending {
    body: Vec<u8>,
    _budget: OwnedSemaphorePermit,
}

impl AsRef<[u8]> for Sending {
    fn as_ref(&self) -> &[u8] {
        &self.body
    }
}

/// An `optimize` that runs, or did.
struct Optimize {
    of: Weak<ShardedCollection>,
    running: bool,
    error: Option<String>,
}

/// The server's state, shared by its connections.
pub struct App {
    db: Database,
    cfg: Config,
    searches: Gate,
    writes: Gate,
    others: Gate,
    looks: Gate,
    /// The budgets for large bodies and replies.
    bodies: Gate,
    replies: Gate,
    metrics: Metrics,
    optimizes: Arc<Optimizes>,
    /// Collections are created one at a time, so that their shards can be
    /// counted.
    creating: Mutex<()>,
}

/// By the address of the collection they are of.
type Optimizes = Mutex<HashMap<usize, Optimize>>;

impl App {
    pub fn new(db: Database, cfg: Config) -> Arc<Self> {
        let gate = |size| Gate::new(size, cfg.queue, cfg.queue_wait);
        Arc::new(Self {
            searches: gate(cfg.searches),
            writes: gate(cfg.writes),
            others: gate(cfg.others),
            looks: gate(cfg.looks),
            bodies: gate(cfg.max_bodies.max(cfg.max_body).div_ceil(UNIT)),
            replies: gate(cfg.max_replies.max(cfg.max_reply).div_ceil(UNIT)),
            metrics: Metrics::default(),
            optimizes: Arc::default(),
            creating: Mutex::default(),
            db,
            cfg,
        })
    }

    /// Serves connections until `stop`; then refuses new ones and waits for
    /// the requests that are being answered, for `drain` at most. Whether
    /// all of them were.
    pub async fn serve(self: Arc<Self>, listener: TcpListener, stop: impl Future<Output = ()>) -> bool {
        let graceful = GracefulShutdown::new();
        let places = Arc::new(Semaphore::new(self.cfg.max_connections.clamp(1, Semaphore::MAX_PERMITS)));
        let mut stop = pin!(stop);
        loop {
            let place = tokio::select! {
                p = places.clone().acquire_owned() => p.expect("never closed"),
                () = &mut stop => break,
            };
            let stream = tokio::select! {
                r = listener.accept() => match r {
                    Ok((stream, _)) => stream,
                    Err(e) => {
                        // Out of file descriptors, say. (One aborted by its
                        // client is not worth more than a retry either.)
                        log!("accept: {e}");
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        continue;
                    }
                },
                () = &mut stop => break,
            };
            let _ = stream.set_nodelay(true);
            // A client that does not take its reply is gone after a while.
            let _ = SockRef::from(&stream).set_tcp_user_timeout(Some(self.cfg.body_total));
            let app = self.clone();
            let service = service_fn(move |req| app.clone().handle(req));
            let conn = hyper::server::conn::http1::Builder::new()
                .timer(TokioTimer::new())
                .header_read_timeout(self.cfg.idle)
                .max_buf_size(128 << 10)
                .serve_connection(TokioIo::new(stream), service);
            let conn = graceful.watch(conn);
            tokio::spawn(async move {
                // An error is the client's doing (it went away, or did not
                // speak HTTP), and has been answered if it could be.
                let _ = conn.await;
                drop(place);
            });
        }
        drop(listener);
        tokio::time::timeout(self.cfg.drain, graceful.shutdown()).await.is_ok()
    }

    async fn handle(self: Arc<Self>, req: Request<Incoming>) -> Result<Response<Full<Bytes>>, Infallible> {
        let start = Instant::now();
        let (head, body) = req.into_parts();
        let (route, allowed) = (Route::parse(&head.method, head.uri.path()), self.allowed(&head.headers));
        let op = route.as_ref().map_or("other", Route::op);
        let reply = match route {
            Ok(Route::Health) => Ok(Reply::ok(&json!({ "status": "ok" }))),
            _ if allowed.is_err() => Err(allowed.unwrap_err()),
            Ok(route) if head.uri.query().is_some_and(|q| !q.is_empty()) && route != Route::Metrics => {
                Err(ApiError::invalid("there are no parameters in the path: use the body"))
            }
            Ok(route) => self.clone().answer(route, &head.headers, body).await,
            Err(e) => Err(e),
        };
        let reply = match reply.unwrap_or_else(Reply::from) {
            reply if reply.body.len() <= SMALL => reply,
            // (Waiting for a part of the budget would keep the reply in
            // memory meanwhile.)
            reply => match self.replies.try_enter_many(reply.body.len().div_ceil(UNIT)) {
                Ok(p) => Reply { budget: Some(p), ..reply },
                Err(gate::Full) => ApiError::busy("large replies").into(),
            },
        };
        let took = start.elapsed();
        self.metrics.record(op, reply.status.as_u16(), took);
        if reply.status.is_server_error() && reply.status != StatusCode::SERVICE_UNAVAILABLE {
            let why = String::from_utf8_lossy(&reply.body);
            log!("{} {}: {} {why}", head.method, head.uri.path(), reply.status.as_u16());
        } else if self.cfg.slow.is_some_and(|slow| took >= slow) {
            log!("slow: {} {}: {} after {took:?}", head.method, head.uri.path(), reply.status.as_u16());
        }
        let mut r = Response::builder().status(reply.status).header(CONTENT_TYPE, reply.content_type);
        match reply.status {
            StatusCode::SERVICE_UNAVAILABLE => r = r.header(RETRY_AFTER, "1"),
            StatusCode::UNAUTHORIZED => r = r.header(WWW_AUTHENTICATE, "Bearer"),
            _ => {}
        }
        if let Some(methods) = reply.allow {
            r = r.header(ALLOW, methods);
        }
        let body = match reply.budget {
            Some(_budget) => Bytes::from_owner(Sending { body: reply.body, _budget }),
            None => Bytes::from(reply.body),
        };
        Ok(r.body(Full::new(body)).expect("a valid head"))
    }

    fn allowed(&self, headers: &HeaderMap) -> Result<(), ApiError> {
        let key = match &self.cfg.access {
            Access::Key(key) => key,
            Access::Local => return local(headers),
            Access::Anyone => return Ok(()),
        };
        let given = headers.get(AUTHORIZATION).map_or(&b""[..], |v| v.as_bytes());
        if let Some((scheme, given)) = given.split_at_checked(7)
            && scheme.eq_ignore_ascii_case(b"bearer ")
            && key.len() == given.len()
            // In a time that does not depend on where they differ.
            && key.bytes().zip(given).fold(0, |acc, (a, b)| std::hint::black_box(acc | (a ^ b))) == 0
        {
            return Ok(());
        }
        Err(ApiError::new(StatusCode::UNAUTHORIZED, "unauthorized", "a valid API key is needed"))
    }

    async fn answer(self: Arc<Self>, route: Route, headers: &HeaderMap, body: Incoming) -> Result<Reply, ApiError> {
        let (body, budget) = match route.max_body(&self.cfg) {
            Some(max) => self.read(headers, body, max).await?,
            None => Default::default(),
        };
        let (gate, what) = match route {
            Route::Search(_) | Route::Get(_) => (&self.searches, "searches"),
            Route::Upsert(_) | Route::Delete(_) | Route::Write(_) => (&self.writes, "writes"),
            Route::List | Route::Info(_) | Route::Metrics => (&self.looks, "requests"),
            _ => (&self.others, "requests"),
        };
        let place = gate.enter().await.map_err(|gate::Full| ApiError::busy(what))?;
        let app = self.clone();
        let work = tokio::task::spawn_blocking(move || {
            let reply = app.run(route, body);
            // (Also when nobody waits for the reply any more.)
            drop((place, budget));
            reply
        });
        // A panic has been printed.
        work.await.unwrap_or_else(|_| {
            Err(ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", "the request failed (a bug)"))
        })
    }

    /// The body, and its part of the budget for large ones.
    async fn read(
        &self,
        headers: &HeaderMap,
        mut body: Incoming,
        max: usize,
    ) -> Result<(Vec<u8>, Option<OwnedSemaphorePermit>), ApiError> {
        let too_large = || {
            let m = format!("the body may have {max} bytes");
            Err(ApiError::new(StatusCode::PAYLOAD_TOO_LARGE, "too_large", m))
        };
        let unsupported = |m| Err(ApiError::new(StatusCode::UNSUPPORTED_MEDIA_TYPE, "unsupported", m));
        let content_type = headers.get(CONTENT_TYPE).map_or(&b""[..], |v| v.as_bytes());
        let content_type = content_type.split(|&b| b == b';').next().expect("one at least");
        // (Which pages in a browser cannot send elsewhere without asking.)
        if !content_type.trim_ascii().eq_ignore_ascii_case(b"application/json") {
            return unsupported("the body has to be JSON: send `Content-Type: application/json`");
        }
        if headers.contains_key(CONTENT_ENCODING) {
            return unsupported("compressed bodies are not supported");
        }
        // The length, or nothing if the body comes in chunks.
        let size = body.size_hint().exact().and_then(|n| usize::try_from(n).ok());
        if size.is_some_and(|n| n > max) || body.size_hint().lower() > max as u64 {
            return too_large();
        }
        let most = size.unwrap_or(max);
        let budget = match most > SMALL {
            true => {
                let units = most.div_ceil(UNIT);
                Some(self.bodies.enter_many(units).await.map_err(|gate::Full| ApiError::busy("large bodies"))?)
            }
            false => None,
        };
        let late = || ApiError::new(StatusCode::REQUEST_TIMEOUT, "timeout", "the body did not come in time");
        let start = Instant::now();
        let mut out = Vec::with_capacity(size.unwrap_or(0));
        loop {
            let left = self.cfg.body_total.saturating_sub(start.elapsed()).min(self.cfg.body_idle);
            let frame = tokio::time::timeout(left, body.frame()).await.map_err(|_| late())?;
            match frame {
                None => return Ok((out, budget)),
                Some(Err(e)) => return Err(ApiError::invalid(format!("reading the body: {e}"))),
                Some(Ok(f)) => {
                    let Ok(data) = f.into_data() else { continue };
                    if out.len() + data.len() > max {
                        return too_large();
                    }
                    out.extend_from_slice(&data);
                }
            }
        }
    }

    /// Does what is asked for. Blocks.
    fn run(self: &Arc<Self>, route: Route, body: Vec<u8>) -> Result<Reply, ApiError> {
        match route {
            Route::Health => unreachable!("answered"),
            Route::Metrics => Ok(Reply {
                status: StatusCode::OK,
                content_type: "text/plain; version=0.0.4",
                body: self.metrics_text().into_bytes(),
                allow: None,
                budget: None,
            }),
            Route::List => {
                let one = |name: &String| match self.db.collection(name).and_then(|c| c.len()) {
                    Ok(rows) => Some(json!({ "name": name, "rows": rows })),
                    Err(Error::NotFound(_)) => None,
                    Err(e) => Some(json!({ "name": name, "error": e.to_string() })),
                };
                let all: Vec<_> = self.db.names().iter().filter_map(one).collect();
                Ok(Reply::ok(&json!({ "collections": all })))
            }
            Route::Create(name) => {
                let c: dto::Create = parse(&body)?;
                let shards = c.shards.unwrap_or(1);
                if !(1..=self.cfg.max_shards).contains(&shards) {
                    return Err(ApiError::invalid(format!("shards must be in 1..={}", self.cfg.max_shards)));
                }
                let config = c.config()?;
                let _one = self.creating.lock().unwrap_or_else(|e| e.into_inner());
                let total: usize = self.collections().iter().map(|c| c.shards().len()).sum();
                if total + shards > self.cfg.max_total_shards {
                    let m = format!("there are {total} shards, and {} at most", self.cfg.max_total_shards);
                    return Err(ApiError::new(StatusCode::CONFLICT, "too_many_shards", m));
                }
                let c = self.db.create_collection(&name, config, shards)?;
                let config = dto::ConfigOut::new(c.config(), shards);
                Ok(Reply::json(StatusCode::CREATED, &dto::CreatedOut { name: &name, config }))
            }
            Route::Info(name) => {
                let c = self.db.collection(&name)?;
                self.info(&c)
            }
            Route::Drop(name) => {
                self.db.drop_collection(&name)?;
                Ok(Reply::ok(&json!({ "dropped": name })))
            }
            Route::Upsert(name) => {
                let ops = parse::<dto::Upsert>(&body)?.ops();
                drop(body);
                let had = self.write(&name, ops)?;
                let replaced = had.iter().filter(|&&h| h).count();
                Ok(Reply::ok(&json!({ "upserted": had.len(), "replaced": replaced })))
            }
            Route::Delete(name) => {
                let had = self.write(&name, parse::<dto::Ids>(&body)?.deletes())?;
                Ok(Reply::ok(&json!({ "deleted": had.iter().filter(|&&h| h).count() })))
            }
            Route::Write(name) => {
                let ops = parse::<dto::Write>(&body)?.ops();
                drop(body);
                let had = self.write(&name, ops)?;
                Ok(Reply::ok(&json!({ "existed": had })))
            }
            Route::Get(name) => {
                let c = self.db.collection(&name)?;
                let ids = parse::<dto::Ids>(&body)?.ids;
                self.check_reply(&c, ids.len(), true)?;
                let (mut rows, mut left) = (Vec::new(), self.cfg.max_reply);
                for id in ids {
                    let Some(row) = c.get(id)? else { continue };
                    left = left.checked_sub(row.bytes()).ok_or_else(|| {
                        ApiError::invalid(format!("the rows take more than {} bytes", self.cfg.max_reply))
                    })?;
                    rows.push(row);
                }
                let rows = rows.iter().map(|r| dto::RowOut::new(r.id, None, Some(r), Fetch::ALL)).collect();
                Ok(Reply::ok(&dto::RowsOut { rows }))
            }
            Route::Search(name) => {
                let c = self.db.collection(&name)?;
                let cfg = &self.cfg;
                let (req, with_stats) =
                    parse::<dto::Search>(&body)?.request(cfg.max_k, cfg.max_l, cfg.max_reply)?;
                self.check_reply(&c, req.k, req.fetch.vector)?;
                let found = c.search(&req)?;
                let hits = found.results.iter().enumerate();
                let results =
                    hits.map(|(i, r)| dto::RowOut::new(r.id, Some(r.distance), found.rows.get(i), req.fetch)).collect();
                Ok(Reply::ok(&dto::FoundOut { results, stats: with_stats.then(|| (&found.stats).into()) }))
            }
            Route::Flush(name) => {
                self.db.collection(&name)?.flush()?;
                Ok(Reply::ok(&json!({ "flushed": true })))
            }
            Route::Optimize(name) => self.optimize(self.db.collection(&name)?),
        }
    }

    fn write(&self, name: &str, ops: Vec<Op>) -> Result<Vec<bool>, ApiError> {
        if ops.len() > self.cfg.max_ops {
            return Err(ApiError::invalid(format!("at most {} writes in a request", self.cfg.max_ops)));
        }
        Ok(self.db.collection(name)?.write(ops)?)
    }

    /// Refuses a reply of `rows` rows that could get too large.
    fn check_reply(&self, c: &ShardedCollection, rows: usize, vectors: bool) -> Result<(), ApiError> {
        // Up to 15 characters per number, as JSON, and a comma.
        let max = if vectors { self.cfg.max_reply / (c.config().dim * 16) } else { usize::MAX };
        if rows > max.min(self.cfg.max_ops) {
            let with = if vectors { " with their vectors" } else { "" };
            return Err(ApiError::invalid(format!("at most {} rows{with}", max.min(self.cfg.max_ops))));
        }
        Ok(())
    }

    /// Those that could be opened.
    fn collections(&self) -> Vec<Arc<ShardedCollection>> {
        self.db.names().iter().filter_map(|n| self.db.collection(n).ok()).collect()
    }

    fn info(&self, c: &Arc<ShardedCollection>) -> Result<Reply, ApiError> {
        let stats = c.stats()?;
        let s = Sums::new(&stats);
        let (optimizing, optimize_error) =
            lock(&self.optimizes).get(&key(c)).map_or((false, None), |o| (o.running, o.error.clone()));
        Ok(Reply::ok(&dto::InfoOut {
            name: c.name(),
            config: dto::ConfigOut::new(c.config(), stats.len()),
            rows: s.rows,
            state: dto::StateOut {
                unflushed_rows: s.unflushed,
                flat_segments: s.flat.0,
                flat_rows: s.flat.1,
                ssd_segments: s.ssd.0,
                ssd_rows: s.ssd.1,
                deleted_rows: s.deleted,
                wal_bytes: s.wal_bytes,
                flush_error: stats.iter().find_map(|s| s.flush_error.clone()),
                build_error: stats.iter().find_map(|s| s.build_error.clone()),
                optimizing,
                optimize_error,
            },
        }))
    }

    /// Starts an `optimize`, unless one runs.
    fn optimize(&self, c: Arc<ShardedCollection>) -> Result<Reply, ApiError> {
        let mut all = lock(&self.optimizes);
        all.retain(|_, o| o.running || o.of.strong_count() > 0);
        if all.get(&key(&c)).is_some_and(|o| o.running) {
            return Err(ApiError::new(StatusCode::CONFLICT, "already_running", "an optimize is already running"));
        }
        let optimizes = self.optimizes.clone();
        let (key, of) = (key(&c), Arc::downgrade(&c));
        let run = move || {
            let error = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| c.optimize())) {
                Ok(r) => r.err().map(|e| e.to_string()),
                Err(_) => Some("it panicked (a bug)".into()),
            };
            if let Some(e) = &error {
                log!("optimize of {}: {e}", c.name());
            }
            if let Some(o) = lock(&optimizes).get_mut(&key) {
                (o.running, o.error) = (false, error);
            }
        };
        std::thread::Builder::new().name("vecdb-optimize".into()).spawn(run).map_err(Error::from)?;
        all.insert(key, Optimize { of, running: true, error: None });
        Ok(Reply::json(StatusCode::ACCEPTED, &json!({ "optimizing": true })))
    }

    fn metrics_text(&self) -> String {
        let mut out = String::new();
        self.metrics.write(&mut out);
        let gates =
            [("search", &self.searches), ("write", &self.writes), ("other", &self.others), ("look", &self.looks)];
        let _ = writeln!(out, "# TYPE vecdb_active gauge");
        for (name, gate) in gates {
            let _ = writeln!(out, "vecdb_active{{kind=\"{name}\"}} {}", gate.load().0);
        }
        let _ = writeln!(out, "# TYPE vecdb_waiting gauge");
        for (name, gate) in gates {
            let _ = writeln!(out, "vecdb_waiting{{kind=\"{name}\"}} {}", gate.load().1);
        }
        for (what, budget) in [("body", &self.bodies), ("reply", &self.replies)] {
            let name = format!("vecdb_large_{what}_bytes");
            let _ = writeln!(out, "# TYPE {name} gauge\n{name} {}", budget.load().0 * UNIT);
        }
        let mut all: Vec<(String, Option<Vec<CollectionStats>>)> = Vec::new();
        for name in self.db.names() {
            match self.db.collection(&name).and_then(|c| c.stats()) {
                Ok(stats) => all.push((name, Some(stats))),
                Err(Error::NotFound(_)) => {}
                Err(_) => all.push((name, None)),
            }
        }
        let _ = writeln!(out, "# TYPE vecdb_up gauge");
        for (name, stats) in &all {
            let _ = writeln!(out, "vecdb_up{{collection=\"{name}\"}} {}", stats.is_some() as u8);
        }
        type Of = fn(&[CollectionStats], &Sums) -> u64;
        let values: [(&str, Of); 14] = [
            ("rows", |_, s| s.rows as u64),
            ("unflushed_rows", |_, s| s.unflushed as u64),
            ("flat_segments", |_, s| s.flat.0 as u64),
            ("flat_rows", |_, s| s.flat.1 as u64),
            ("ssd_segments", |_, s| s.ssd.0 as u64),
            ("ssd_rows", |_, s| s.ssd.1 as u64),
            ("deleted_rows", |_, s| s.deleted as u64),
            ("wal_bytes", |_, s| s.wal_bytes),
            ("shards_failing_flushes", |c, _| c.iter().filter(|s| s.flush_error.is_some()).count() as u64),
            ("shards_failing_builds", |c, _| c.iter().filter(|s| s.build_error.is_some()).count() as u64),
            ("flushes_total", |c, _| c.iter().map(|s| s.flushes).sum()),
            ("builds_total", |c, _| c.iter().map(|s| s.builds).sum()),
            ("write_batches_total", |c, _| c.iter().map(|s| s.write_batches).sum()),
            ("wal_syncs_total", |c, _| c.iter().map(|s| s.wal_syncs).sum()),
        ];
        for (what, of) in values {
            let kind = if what.ends_with("_total") { "counter" } else { "gauge" };
            let _ = writeln!(out, "# TYPE vecdb_{what} {kind}");
            for (name, stats) in &all {
                if let Some(stats) = stats {
                    let n = of(stats, &Sums::new(stats));
                    let _ = writeln!(out, "vecdb_{what}{{collection=\"{name}\"}} {n}");
                }
            }
        }
        out
    }
}

/// Refuses what a web page can send: to this machine, from another origin,
/// or under a name of its own that leads here.
fn local(headers: &HeaderMap) -> Result<(), ApiError> {
    let host = headers.get(HOST).map_or(&b"localhost"[..], |v| v.as_bytes());
    let host = std::str::from_utf8(host).unwrap_or_default();
    let name = match host.strip_prefix('[') {
        Some(v6) => v6.split(']').next(),
        None => host.split(':').next(),
    };
    let name = name.unwrap_or_default();
    if headers.contains_key(ORIGIN) || !(name.eq_ignore_ascii_case("localhost") || name.parse::<IpAddr>().is_ok()) {
        let m = "without an API key, only programs that ask for localhost or an address are answered";
        return Err(ApiError::new(StatusCode::FORBIDDEN, "forbidden", m));
    }
    Ok(())
}

fn lock(o: &Optimizes) -> MutexGuard<'_, HashMap<usize, Optimize>> {
    o.lock().unwrap_or_else(|e| e.into_inner())
}

/// (Not used again as long as there is a `Weak` to it.)
fn key(c: &Arc<ShardedCollection>) -> usize {
    Arc::as_ptr(c) as usize
}

/// Sums over the shards of a collection.
struct Sums {
    rows: usize,
    unflushed: usize,
    /// Segments and their rows.
    flat: (usize, usize),
    ssd: (usize, usize),
    deleted: usize,
    wal_bytes: u64,
}

impl Sums {
    fn new(stats: &[CollectionStats]) -> Self {
        let kind = |kind: &str| {
            let of = stats.iter().flat_map(|s| &s.segments).filter(|s| s.kind == kind);
            of.fold((0, 0), |(n, rows), s| (n + 1, rows + s.rows))
        };
        Self {
            rows: stats.iter().map(|s| s.live).sum(),
            unflushed: stats.iter().map(|s| s.growing_rows + s.frozen_rows).sum(),
            flat: kind("flat"),
            ssd: kind("ssd"),
            deleted: stats.iter().flat_map(|s| &s.segments).map(|s| s.deleted).sum(),
            wal_bytes: stats.iter().map(|s| s.wal_bytes).sum(),
        }
    }
}

fn parse<T: DeserializeOwned>(body: &[u8]) -> Result<T, ApiError> {
    serde_json::from_slice(body).map_err(|e| ApiError::invalid(format!("the body: {e}")))
}
