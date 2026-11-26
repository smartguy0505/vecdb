//! A database: named collections, each split into shards.
//!
//! ```text
//! <dir>/CATALOG                  the collections: names, ids, shard counts
//!       LOCK                     held while a process has the database open
//!       col-<id>/shard-<n>/      one shard (see `collection`)
//!       new-<id>/                a collection being created
//! ```
//!
//! A row is in the shard its id hashes to ([`shard_of`]). A write goes to
//! the shards of its ids; a search covers the segments of every shard, as
//! one set of tasks (a shard in another process would be one more task).
//! The caller sees one [`ShardedCollection`]; the shards behind it are
//! [`Shard`]s, all of them here for now.
//!
//! **Creating and dropping** a collection commit by replacing the catalog.
//! A collection is prepared as `new-<id>` and renamed before that; a crash
//! in between leaves an empty collection the catalog does not know, which
//! an open reports and leaves alone, as it does every directory that only
//! the catalog could vouch for. Only the files of collections the catalog
//! lists as dropped are removed.
//!
//! ```
//! # fn main() -> vecdb_core::Result<()> {
//! use vecdb_core::{Attrs, CollectionConfig, Database, DatabaseOptions, Metric, Op, Row};
//! use vecdb_core::{SearchOptions, SearchRequest};
//!
//! # let data = std::env::temp_dir().join(format!("vecdb-doc-{}", std::process::id()));
//! # let query = vec![0.1, 1.0, 0.0, 0.0];
//! # let vector = query.clone();
//! let (db, _report) = Database::open(&data, &DatabaseOptions::default())?;
//! let c = db.create_collection("docs", CollectionConfig::new(4, Metric::L2), 2)?; // 2 shards
//! c.write(vec![Op::Upsert(Row { id: 42, vector, attrs: Attrs::default() })])?;
//! let options = SearchOptions::default();
//! let reply = c.search(&SearchRequest::new(query, 10, options))?;
//! assert_eq!(reply.results[0].id, 42);
//! # drop((c, db));
//! # std::fs::remove_dir_all(&data)?;
//! # Ok(())
//! # }
//! ```

mod catalog;
mod shard;

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, RwLock};

use crate::collection::search::{Ask, search_all};
use crate::collection::{
    BuildLimit, Collection, CollectionConfig, CollectionStats, OpenOptions, OpenReport, Snapshot, lock_dir,
};
use crate::error::{Error, Result};
use crate::format::sync_parent_dir;
use crate::par;
use crate::pool::SearchPool;
use catalog::{CATALOG, Catalog, Entry, check_name};

pub use catalog::{MAX_NAME, MAX_SHARDS};
pub use shard::{Op, PendingWrite, Row, SearchReply, SearchRequest, Shard, shard_of};
use shard::fetch_rows;

/// How this process uses a database (not stored).
#[derive(Clone, Debug)]
pub struct DatabaseOptions {
    /// For every shard. Its `pool` and `builds` are set by the database.
    pub shard: OpenOptions,
    /// Threads that help searches, over all collections, and the CPUs that
    /// searches may use before they are left to their own threads; 0 = see
    /// [`SearchPool::new`].
    pub search_threads: usize,
    pub search_cpus: usize,
    /// Index builds that run at once, over all collections.
    pub builds: usize,
}

impl Default for DatabaseOptions {
    fn default() -> Self {
        Self { shard: OpenOptions::default(), search_threads: 0, search_cpus: 0, builds: 1 }
    }
}

/// What [`Database::open`] found.
#[derive(Debug, Default)]
pub struct DatabaseReport {
    /// Shards opened, by collection.
    pub opened: Vec<(String, Vec<OpenReport>)>,
    /// Collections that could not be opened, and why. The others work.
    pub failed: Vec<(String, Error)>,
    /// Directories of dropped collections and of interrupted creates,
    /// removed.
    pub removed: Vec<String>,
    /// Collection directories the catalog does not know, left alone.
    pub unknown: Vec<String>,
}

/// A directory of named collections, shared by any number of threads.
pub struct Database {
    dir: PathBuf,
    opts: OpenOptions,
    /// Held while a collection is created or dropped.
    catalog: Mutex<Catalog>,
    /// By name; the error of those that could not be opened.
    open: RwLock<BTreeMap<String, Result<Arc<ShardedCollection>>>>,
    // After the collections, which have locks of their own.
    _lock: File,
}

fn collection_dir(dir: &Path, id: u64) -> PathBuf {
    dir.join(format!("col-{id:08}"))
}

fn shard_dir(collection: &Path, shard: usize) -> PathBuf {
    collection.join(format!("shard-{shard:04}"))
}

/// The id in a name of the form `<prefix><8 digits or more>`.
fn id_in(name: &str, prefix: &str) -> Option<u64> {
    let digits = name.strip_prefix(prefix)?;
    (digits.len() >= 8 && digits.bytes().all(|b| b.is_ascii_digit())).then(|| digits.parse().ok())?
}

impl Database {
    /// Opens the database in `dir` and its collections; creates it if `dir`
    /// is missing or empty.
    pub fn open(dir: impl AsRef<Path>, opts: &DatabaseOptions) -> Result<(Self, DatabaseReport)> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;
        let lock = lock_dir(dir)?;
        let mut report = DatabaseReport::default();
        let mut dirs: BTreeMap<u64, String> = BTreeMap::new();
        for entry in fs::read_dir(dir)? {
            let name = entry?.file_name().to_string_lossy().into_owned();
            if let Some(id) = id_in(&name, "col-") {
                dirs.insert(id, name);
            } else if id_in(&name, "new-").is_some() {
                fs::remove_dir_all(dir.join(&name))?;
                report.removed.push(name);
            }
        }
        let mut catalog = match Catalog::read(dir) {
            Ok(c) => c,
            Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                if let Some(name) = dirs.values().next() {
                    return Err(Error::Corrupt(format!("{} has collections ({name}) but no catalog", dir.display())));
                }
                let c = Catalog { next_id: 1, ..Default::default() };
                c.install(dir)?;
                sync_parent_dir(&dir.join(CATALOG))?;
                c
            }
            Err(e) => return Err(e),
        };
        if !catalog.dropped.is_empty() {
            for id in std::mem::take(&mut catalog.dropped) {
                if let Some(name) = dirs.remove(&id) {
                    fs::remove_dir_all(dir.join(&name))?;
                    report.removed.push(name);
                }
            }
            sync_parent_dir(&dir.join(CATALOG))?;
            catalog.install(dir)?;
            sync_parent_dir(&dir.join(CATALOG))?;
        }
        for (&id, name) in &dirs {
            if !catalog.entries.iter().any(|e| e.id == id) {
                report.unknown.push(name.clone());
                catalog.next_id = catalog.next_id.max(id + 1);
            }
        }
        report.removed.sort();
        let pool = SearchPool::new(opts.search_threads, opts.search_cpus)?;
        let builds = BuildLimit::new(opts.builds);
        let db = Self {
            dir: dir.to_path_buf(),
            opts: OpenOptions { pool: Some(pool), builds: Some(builds), ..opts.shard.clone() },
            catalog: Mutex::new(catalog.clone()),
            open: RwLock::default(),
            _lock: lock,
        };
        // Several at once: most of an open is reading files.
        let entries = &catalog.entries;
        let opened = par::map_ranges(entries.len(), 4, |r| {
            r.map(|i| ShardedCollection::open(&db.dir, &entries[i], &db.opts)).collect::<Vec<_>>()
        });
        for (e, r) in entries.iter().zip(opened.into_iter().flatten()) {
            let c = match r {
                Ok((c, reports)) => {
                    report.opened.push((e.name.clone(), reports));
                    Ok(Arc::new(c))
                }
                Err(err) => {
                    report.failed.push((e.name.clone(), err.duplicate()));
                    Err(err)
                }
            };
            db.open.write().unwrap().insert(e.name.clone(), c);
        }
        Ok((db, report))
    }

    /// Creates a collection of `shards` shards, each with `config` (the
    /// thresholds in it are per shard).
    pub fn create_collection(
        &self,
        name: &str,
        config: CollectionConfig,
        shards: usize,
    ) -> Result<Arc<ShardedCollection>> {
        check_name(name)?;
        config.validate()?;
        if !(1..=MAX_SHARDS).contains(&shards) {
            return Err(Error::InvalidArgument(format!("shards must be in 1..={MAX_SHARDS}")));
        }
        let mut catalog = self.catalog.lock().unwrap();
        if catalog.find(name).is_some() {
            return Err(Error::AlreadyExists(format!("collection {name:?}")));
        }
        let entry = Entry { id: catalog.next_id, name: name.to_string(), shards };
        let new = self.dir.join(format!("new-{:08}", entry.id));
        let dir = collection_dir(&self.dir, entry.id);
        let mut next = catalog.clone();
        next.next_id += 1;
        next.insert(entry.clone());
        let prepared = (|| -> Result<()> {
            if new.exists() {
                fs::remove_dir_all(&new)?;
            }
            let opts = OpenOptions { background: false, ..self.opts.clone() };
            for s in 0..shards {
                Collection::create(shard_dir(&new, s), config.clone(), opts.clone())?;
            }
            sync_parent_dir(&shard_dir(&new, 0))?;
            fs::rename(&new, &dir)?;
            sync_parent_dir(&dir)?;
            next.install(&self.dir)
        })();
        if let Err(e) = prepared {
            // Not in the catalog, so nothing was ever written to it.
            let _ = fs::remove_dir_all(&new);
            let _ = fs::remove_dir_all(&dir);
            return Err(e);
        }
        // It exists now, whatever else fails.
        *catalog = next;
        let durable = sync_parent_dir(&dir);
        let opened = ShardedCollection::open(&self.dir, &entry, &self.opts).map(|(c, _)| Arc::new(c));
        let result = match &opened {
            Ok(c) => Ok(c.clone()),
            Err(e) => Err(e.duplicate()),
        };
        self.open.write().unwrap().insert(name.to_string(), opened);
        durable?;
        result
    }

    /// Drops a collection and its data. Those who still hold it get
    /// [`Error::NotFound`]; its files go when the last of them lets go.
    pub fn drop_collection(&self, name: &str) -> Result<()> {
        let mut catalog = self.catalog.lock().unwrap();
        let Some(i) = catalog.find(name) else { return Err(unknown(name)) };
        let mut next = catalog.clone();
        let entry = next.entries.remove(i);
        next.dropped.push(entry.id);
        // Those whose files are gone for good need no entry.
        let left: Vec<u64> =
            next.dropped.iter().copied().filter(|&id| collection_dir(&self.dir, id).exists()).collect();
        if left.len() < next.dropped.len() {
            sync_parent_dir(&self.dir.join(CATALOG))?;
            next.dropped = left;
        }
        next.install(&self.dir)?;
        *catalog = next;
        let durable = sync_parent_dir(&self.dir.join(CATALOG));
        let open = self.open.write().unwrap().remove(name);
        durable?;
        match open {
            Some(Ok(c)) => c.dropped.store(true, Ordering::Release),
            // Could not be opened: nobody holds it.
            _ => drop(fs::remove_dir_all(collection_dir(&self.dir, entry.id))),
        }
        Ok(())
    }

    pub fn collection(&self, name: &str) -> Result<Arc<ShardedCollection>> {
        match self.open.read().unwrap().get(name) {
            Some(Ok(c)) => Ok(c.clone()),
            Some(Err(e)) => Err(Error::Unavailable(format!("collection {name:?} could not be opened: {e}"))),
            None => Err(unknown(name)),
        }
    }

    /// Names of all collections, in order.
    pub fn names(&self) -> Vec<String> {
        self.open.read().unwrap().keys().cloned().collect()
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

/// What a write has for one shard.
struct Part<'a> {
    shard: &'a dyn Shard,
    /// Where its ops are in those of the write.
    at: Vec<usize>,
    ops: Vec<Op>,
}

impl Part<'_> {
    /// Writes it once there is room for all parts. `None` if there is none
    /// for another one.
    fn write(self, place: Place) -> (Vec<usize>, Result<Option<Vec<bool>>>) {
        let write = || {
            let write = self.shard.begin(&self.ops)?;
            if place.all_ready() { write().map(Some) } else { Ok(None) }
        };
        let r = write();
        (self.at, r)
    }
}

/// Where the parts of a write wait for each other.
struct Meet {
    state: Mutex<Parts>,
    change: Condvar,
}

struct Parts {
    /// That are not ready yet.
    left: usize,
    failed: bool,
}

/// A part's place there. The part has failed if this is dropped (which does
/// not matter any more once all are ready).
struct Place<'a>(&'a Meet);

impl Meet {
    fn new(parts: usize) -> Self {
        Self { state: Mutex::new(Parts { left: parts, failed: false }), change: Condvar::new() }
    }

    fn place(&self) -> Place<'_> {
        Place(self)
    }

    fn lock(&self) -> MutexGuard<'_, Parts> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Place<'_> {
    /// Says that this part is ready, and waits for the others. Whether all
    /// are.
    fn all_ready(self) -> bool {
        let mut s = self.0.lock();
        s.left -= 1;
        self.0.change.notify_all();
        while s.left > 0 && !s.failed {
            s = self.0.change.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        s.left == 0
    }
}

impl Drop for Place<'_> {
    fn drop(&mut self) {
        self.0.lock().failed = true;
        self.0.change.notify_all();
    }
}

fn unknown(name: &str) -> Error {
    Error::NotFound(format!("collection {name:?}"))
}

/// A named collection: rows spread over shards by id.
pub struct ShardedCollection {
    name: String,
    config: Arc<CollectionConfig>,
    shards: Vec<Arc<dyn Shard>>,
    pool: Option<Arc<SearchPool>>,
    dir: PathBuf,
    /// No longer in the catalog: the files go with the last handle.
    dropped: AtomicBool,
}

impl Drop for ShardedCollection {
    fn drop(&mut self) {
        self.shards.clear();
        if self.dropped.load(Ordering::Acquire) {
            // What cannot be removed now is at the next open.
            let _ = fs::remove_dir_all(&self.dir);
        }
    }
}

impl ShardedCollection {
    fn open(db: &Path, e: &Entry, opts: &OpenOptions) -> Result<(Self, Vec<OpenReport>)> {
        let dir = collection_dir(db, e.id);
        let mut shards: Vec<Arc<dyn Shard>> = Vec::new();
        let (mut reports, mut config) = (Vec::new(), None);
        let opened = par::map_ranges(e.shards, 4, |r| {
            r.map(|s| Collection::open(shard_dir(&dir, s), opts.clone())).collect::<Vec<_>>()
        });
        for (s, r) in opened.into_iter().flatten().enumerate() {
            let (c, report) = r?;
            if config.get_or_insert_with(|| c.config().clone()) != c.config() {
                return Err(Error::Corrupt(format!("{}: shards 0 and {s} differ in their settings", dir.display())));
            }
            shards.push(Arc::new(c));
            reports.push(report);
        }
        let config = Arc::new(config.expect("at least one shard"));
        let c = Self { name: e.name.clone(), config, shards, pool: opts.pool.clone(), dir, dropped: false.into() };
        Ok((c, reports))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// What each shard is configured as.
    pub fn config(&self) -> &CollectionConfig {
        &self.config
    }

    pub fn shards(&self) -> &[Arc<dyn Shard>] {
        &self.shards
    }

    fn check(&self) -> Result<()> {
        if self.dropped.load(Ordering::Acquire) { Err(unknown(&self.name)) } else { Ok(()) }
    }

    /// Applies `ops`, in order for each id; per op, whether its id had a row
    /// before. All or none within a shard; if a shard fails to write, others
    /// may have applied their part. Nothing is applied if any op is invalid,
    /// or if a shard has no room (see [`OpenOptions::max_write_wait`]).
    pub fn write(&self, ops: Vec<Op>) -> Result<Vec<bool>> {
        self.check()?;
        let n = self.shards.len();
        if n == 1 {
            return self.shards[0].write(&ops);
        }
        for op in &ops {
            if let Op::Upsert(r) = op {
                self.config.check_row(&r.vector, &r.attrs)?;
            }
        }
        let part = |shard| Part { shard, at: Vec::new(), ops: Vec::new() };
        let mut parts: Vec<Part> = self.shards.iter().map(|s| part(&**s)).collect();
        let total = ops.len();
        for (i, op) in ops.into_iter().enumerate() {
            let part = &mut parts[shard_of(op.id(), n)];
            part.at.push(i);
            part.ops.push(op);
        }
        parts.retain(|p| !p.at.is_empty());
        let meet = Meet::new(parts.len());
        let mut busy: Vec<_> = parts.into_iter().map(|p| (p, meet.place())).collect();
        let Some((last, place)) = busy.pop() else { return Ok(Vec::new()) };
        // At once, as each may wait for room and for the disk; and nothing
        // before there is room for all. Or a shard that has applied its part
        // would have to do so again when the write is tried again, as a new
        // one in its line, and be late itself then.
        let results = std::thread::scope(|s| {
            let others: Vec<_> = busy.into_iter().map(|(p, place)| s.spawn(move || p.write(place))).collect();
            let mut results = vec![last.write(place)];
            results.extend(others.into_iter().map(|h| h.join().expect("a write panicked")));
            results
        });
        let mut out = vec![false; total];
        let mut written = true;
        for (at, r) in results {
            match r? {
                Some(had) => at.into_iter().zip(had).for_each(|(i, had)| out[i] = had),
                None => written = false,
            }
        }
        // (One that was not written has another one that failed.)
        if written { Ok(out) } else { Err(Error::Unavailable("a part of the write has failed".into())) }
    }

    /// The `k` nearest live rows over all shards, closest first. Each shard
    /// answers as of the last write it has applied.
    pub fn search(&self, req: &SearchRequest) -> Result<SearchReply> {
        self.check()?;
        // Per shard; of those that are elsewhere, the rows they found.
        let mut snaps: Vec<Option<(&Collection, Snapshot)>> = Vec::new();
        let mut asks: Vec<Ask> = Vec::new();
        let elsewhere: Arc<Mutex<HashMap<u64, Row>>> = Arc::default();
        for s in &self.shards {
            snaps.push(s.local().map(|c| (c, c.snapshot())));
            if s.local().is_none() {
                let (s, req, rows) = (s.clone(), req.clone(), elsewhere.clone());
                asks.push(Box::new(move || {
                    let r = s.search(&req)?;
                    rows.lock().unwrap().extend(r.rows.into_iter().map(|r| (r.id, r)));
                    Ok((r.results, r.stats))
                }));
            }
        }
        let here: Vec<&Snapshot> = snaps.iter().flatten().map(|(_, s)| s).collect();
        let pool = self.pool.as_deref();
        let (results, stats) = search_all(&self.config, &here, asks, pool, &req.vector, req.k, &req.options)?;
        let rows = fetch_rows(&results, req, |id| match &snaps[shard_of(id, snaps.len())] {
            Some((c, snap)) => c.get_in(snap, id, req.fetch),
            None => Ok(elsewhere.lock().unwrap().remove(&id).map(|r| (r.vector, r.attrs))),
        })?;
        Ok(SearchReply { results, rows, stats })
    }

    pub fn get(&self, id: u64) -> Result<Option<Row>> {
        self.check()?;
        self.shards[shard_of(id, self.shards.len())].get(id)
    }

    /// Per shard.
    pub fn stats(&self) -> Result<Vec<CollectionStats>> {
        self.each(|s| s.stats())
    }

    /// Live rows.
    pub fn len(&self) -> Result<usize> {
        Ok(self.stats()?.iter().map(|s| s.live).sum())
    }

    /// Writes what is in RAM to segment files.
    pub fn flush(&self) -> Result<()> {
        self.each(|s| s.flush()).map(drop)
    }

    /// Merges each shard's segments into one index.
    pub fn optimize(&self) -> Result<()> {
        self.each(|s| s.optimize()).map(drop)
    }

    /// Waits until background work has caught up.
    pub fn wait_idle(&self) -> Result<()> {
        self.each(|s| s.wait_idle()).map(drop)
    }

    fn each<T>(&self, f: impl Fn(&dyn Shard) -> Result<T>) -> Result<Vec<T>> {
        self.check()?;
        self.shards.iter().map(|s| f(&**s)).collect()
    }
}

#[cfg(test)]
mod tests;
