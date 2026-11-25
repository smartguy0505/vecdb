//! Searching collections: one task per SSD segment, one per few MiB of
//! vectors in RAM, and one per shard asked elsewhere, worked through by the
//! calling thread and threads of the
//! [`SearchPool`](crate::pool::SearchPool), if there are any to spare. The
//! results do not depend on who does what: every task returns exact
//! distances, and ties go to the smaller id.

use std::ops::Range;
use std::sync::{Arc, OnceLock, RwLock};

use super::bitmap::Bitmap;
use super::filter::Compiled;
use super::segment::{self, AsOf, Body, Growing, Sealed};
use super::{
    CollectionConfig, Engine, FilterMode, SearchOptions, SearchStats, Snapshot, View, growing_chunk_rows,
};
use crate::disk::{DiskIndex, DiskSearchParams};
use crate::distance::{check_finite, normalize};
use crate::error::{Error, Result};
use crate::kernels;
use crate::pool::{Claims, SearchPool, run_alone};
use crate::store::VectorStore;
use crate::topk::{SearchResult, TopK};

/// Vector bytes a scan task covers, about a millisecond's work.
const TASK_BYTES: usize = 4 << 20;
/// Vectors a search scans alone rather than wake threads for: about 0.2 ms.
const ALONE_BYTES: usize = 1 << 20;

/// Rows of a scan task: whole blocks of the scan. Tests use few, so their
/// segments still make several tasks.
fn task_rows(dim: usize) -> usize {
    if cfg!(test) {
        return 100;
    }
    (TASK_BYTES / (dim * 4)).max(256) / 256 * 256
}

impl Engine {
    pub fn search(&self, query: &[f32], k: usize, opts: &SearchOptions) -> Result<Vec<SearchResult>> {
        Ok(self.search_with_stats(query, k, opts)?.0)
    }

    /// The `k` nearest live rows over all segments, closest first, as of
    /// the last write applied when the search starts.
    pub fn search_with_stats(
        &self,
        query: &[f32],
        k: usize,
        opts: &SearchOptions,
    ) -> Result<(Vec<SearchResult>, SearchStats)> {
        self.search_in(&self.snapshot(), query, k, opts)
    }

    pub(crate) fn search_in(
        &self,
        snap: &Snapshot,
        query: &[f32],
        k: usize,
        opts: &SearchOptions,
    ) -> Result<(Vec<SearchResult>, SearchStats)> {
        search_all(&self.config, &[snap], Vec::new(), self.opts.pool.as_deref(), query, k, opts)
    }
}

/// Asks a shard that is not in this process for its nearest rows.
pub(crate) type Ask = Box<dyn Fn() -> Result<(Vec<SearchResult>, SearchStats)> + Send + Sync>;

/// The `k` nearest live rows over collections that have the settings `cfg`,
/// and what is asked for elsewhere.
pub(crate) fn search_all(
    cfg: &Arc<CollectionConfig>,
    snaps: &[&Snapshot],
    asks: Vec<Ask>,
    pool: Option<&SearchPool>,
    query: &[f32],
    k: usize,
    opts: &SearchOptions,
) -> Result<(Vec<SearchResult>, SearchStats)> {
    if query.len() != cfg.dim {
        return Err(Error::DimensionMismatch { expected: cfg.dim, got: query.len() });
    }
    check_finite(query)?;
    if k == 0 {
        return Ok(Default::default());
    }
    let filter = match &opts.filter {
        None => None,
        Some(f) => match f.compile(&cfg.fields)? {
            Compiled::All => None,
            Compiled::Nothing => return Ok(Default::default()),
            c => Some(c),
        },
    };
    let mut prepared = query.to_vec();
    if cfg.metric.needs_normalization() {
        normalize(&mut prepared);
    }
    let plan = Plan::new(cfg, snaps, asks, query.to_vec(), prepared, k, opts, filter);
    let (n, waits) = (plan.tasks.len(), plan.waits);
    let mut st = SearchStats { segments: plan.segments, ..Default::default() };
    let parts = match pool {
        Some(pool) => {
            let most = if plan.worth_sharing() { n } else { 0 };
            pool.run(n, waits, most, move |c| plan.run(c))
        }
        None => run_alone(n, |c| plan.run(c)),
    };
    let mut top = TopK::new(k);
    for part in parts {
        let (found, s) = part?;
        st.add(&s);
        found.iter().for_each(|r| top.push(r.id, r.distance));
    }
    Ok((top.into_sorted(), st))
}

impl SearchStats {
    fn add(&mut self, o: &Self) {
        self.segments += o.segments;
        self.scanned += o.scanned;
        self.reads += o.reads;
        self.rounds += o.rounds;
        self.expanded += o.expanded;
        self.matched += o.matched;
        self.graph_segments += o.graph_segments;
        self.scan_segments += o.scan_segments;
        self.fallbacks += o.fallbacks;
        self.pq_scanned += o.pq_scanned;
    }
}

/// One search, split into tasks.
pub(super) struct Plan {
    cfg: Arc<CollectionConfig>,
    parts: Vec<Part>,
    asks: Vec<Ask>,
    query: Vec<f32>,
    /// The query prepared for the metric, as rows are stored.
    prepared: Vec<f32>,
    k: usize,
    opts: SearchOptions,
    filter: Option<Compiled>,
    tasks: Vec<Task>,
    /// Of the tasks, the first ones: those that mostly wait.
    pub waits: usize,
    segments: usize,
    /// Bytes of vectors to scan, and SSD segments to walk.
    scan_bytes: usize,
    disks: usize,
}

/// One collection searched, as a snapshot of it was.
struct Part {
    view: Arc<View>,
    seq: u64,
    /// Under a filter: the rows that match (and, in sealed segments, are
    /// live), per segment in RAM and per sealed one. Worked out by the
    /// first task that needs it.
    memory_matching: Vec<OnceLock<Bitmap>>,
    sealed_matching: Vec<OnceLock<(Bitmap, usize)>>,
}

enum Task {
    Ask(usize),
    /// Of a part, rows of a segment in RAM (its place in `View::memory`).
    Memory(usize, usize, Range<usize>),
    /// Of a part, rows of a flat segment.
    Flat(usize, usize, Range<usize>),
    /// Of a part, an SSD segment.
    Disk(usize, usize),
}

impl Plan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: &Arc<CollectionConfig>,
        snaps: &[&Snapshot],
        asks: Vec<Ask>,
        query: Vec<f32>,
        prepared: Vec<f32>,
        k: usize,
        opts: &SearchOptions,
        filter: Option<Compiled>,
    ) -> Self {
        let step = task_rows(cfg.dim);
        let chunks = |n: usize| (0..n).step_by(step).map(move |a| a..(a + step).min(n));
        let mut tasks: Vec<Task> = (0..asks.len()).map(Task::Ask).collect();
        // The long ones first, so that no one ends up waiting for them.
        let disks = |direct: bool| {
            let all = snaps.iter().enumerate();
            let sealed = all.flat_map(|(p, snap)| snap.view.sealed.iter().enumerate().map(move |s| (p, s)));
            sealed.filter_map(move |(p, (i, s))| match &s.body {
                Body::Disk { index, .. } if index.is_direct() == direct => Some(Task::Disk(p, i)),
                _ => None,
            })
        };
        tasks.extend(disks(true));
        let waits = tasks.len();
        tasks.extend(disks(false));
        let disks = tasks.len() - asks.len();
        let (mut segments, mut rows) = (0, 0);
        for (p, snap) in snaps.iter().enumerate() {
            segments += snap.view.sealed.len();
            for (i, m) in snap.view.memory().enumerate() {
                let n = m.read().unwrap().rows_at(snap.seq);
                segments += (n > 0) as usize;
                rows += n;
                tasks.extend(chunks(n).map(|r| Task::Memory(p, i, r)));
            }
            for (i, s) in snap.view.sealed.iter().enumerate() {
                if let Body::Flat(_) = s.body {
                    rows += s.positions();
                    tasks.extend(chunks(s.positions()).map(|r| Task::Flat(p, i, r)));
                }
            }
        }
        let part = |snap: &&Snapshot| {
            let (memory, sealed) = match filter {
                Some(_) => (snap.view.memory().count(), snap.view.sealed.len()),
                None => (0, 0),
            };
            Part {
                view: snap.view.clone(),
                seq: snap.seq,
                memory_matching: std::iter::repeat_with(OnceLock::new).take(memory).collect(),
                sealed_matching: std::iter::repeat_with(OnceLock::new).take(sealed).collect(),
            }
        };
        Self {
            cfg: cfg.clone(),
            parts: snaps.iter().map(part).collect(),
            asks,
            query,
            prepared,
            k,
            opts: SearchOptions { filter: None, ..opts.clone() },
            filter,
            tasks,
            waits,
            segments,
            scan_bytes: rows * cfg.dim * 4,
            disks,
        }
    }

    /// Whether other threads could save more than waking them costs.
    fn worth_sharing(&self) -> bool {
        self.disks + self.asks.len() > 1 || self.scan_bytes > ALONE_BYTES || cfg!(test)
    }

    /// Does the tasks `claims` hands out: the best rows in them.
    fn run(&self, claims: &Claims) -> Result<(Vec<SearchResult>, SearchStats)> {
        let mut top = TopK::new(self.k);
        let mut st = SearchStats::default();
        while let Some(i) = claims.next() {
            let done = match &self.tasks[i] {
                Task::Ask(a) => self.asks[*a]().map(|(found, s)| {
                    found.iter().for_each(|r| top.push(r.id, r.distance));
                    st.add(&s);
                }),
                Task::Memory(p, m, rows) => {
                    self.memory(&self.parts[*p], *m, rows.clone(), &mut top, &mut st);
                    Ok(())
                }
                Task::Flat(p, s, rows) => {
                    self.flat(&self.parts[*p], *s, rows.clone(), &mut top, &mut st);
                    Ok(())
                }
                Task::Disk(p, s) => self.disk(&self.parts[*p], *s, &mut top, &mut st),
            };
            if let Err(e) = done {
                claims.close();
                return Err(e);
            }
        }
        Ok((top.into_sorted(), st))
    }

    fn memory(&self, part: &Part, m: usize, rows: Range<usize>, top: &mut TopK, st: &mut SearchStats) {
        let (cfg, seq) = (&self.cfg, part.seq);
        let mem: &RwLock<Growing> = part.view.memory().nth(m).expect("planned");
        let matching = self.filter.as_ref().map(|c| {
            part.memory_matching[m]
                .get_or_init(|| c.eval(mem.read().unwrap().cols.as_ref().expect("a filter with fields")))
        });
        // In chunks, each under its own read lock: a write waits for one
        // chunk, not a whole scan (and searches arriving meanwhile wait for
        // the write). Rows the writer adds or deletes between chunks are
        // newer than the snapshot.
        let chunk = growing_chunk_rows(cfg.dim);
        for start in rows.clone().step_by(chunk) {
            let end = (start + chunk).min(rows.end);
            let g = mem.read().unwrap();
            let data = &g.data[start * cfg.dim..end * cfg.dim];
            let push = |r: usize, d: f32| top.push(g.ids[start + r], d);
            let scanned = match &matching {
                None => self.scan(data, |r| g.live_at(start + r, seq), push),
                Some(m) => self.scan(data, |r| m.get(start + r) && g.live_at(start + r, seq), push),
            };
            st.scanned += scanned;
            if matching.is_some() {
                st.matched += scanned;
            }
        }
    }

    fn scan(&self, data: &[f32], admit: impl Fn(usize) -> bool, push: impl FnMut(usize, f32)) -> usize {
        segment::scan(kernels::best(), self.cfg.metric, &self.prepared, data, admit, push)
    }

    /// The live rows of sealed segment `s` that match the filter, if there
    /// is one, and how many.
    fn matching<'a>(&self, part: &'a Part, s: usize) -> Option<&'a (Bitmap, usize)> {
        let c = self.filter.as_ref()?;
        Some(part.sealed_matching[s].get_or_init(|| {
            let seg: &Sealed = &part.view.sealed[s];
            let mut m = c.eval(&seg.attrs.as_ref().expect("a filter with fields").columns);
            m.and_not_words(seg.tomb.words_as_of(part.seq).into_iter());
            let n = m.count();
            (m, n)
        }))
    }

    fn flat(&self, part: &Part, s: usize, rows: Range<usize>, top: &mut TopK, st: &mut SearchStats) {
        let seg = &part.view.sealed[s];
        let Body::Flat(store) = &seg.body else { unreachable!("planned as flat") };
        let start = rows.start;
        let data = &store.raw_vectors()[start * self.cfg.dim..rows.end * self.cfg.dim];
        let push = |r: usize, d: f32| top.push(store.id(start + r), d);
        match self.matching(part, s) {
            None => {
                let deleted = AsOf::new(&seg.tomb, part.seq);
                st.scanned += self.scan(data, |r| !deleted.deleted(start + r), push);
            }
            Some((_, 0)) => {}
            Some((m, _)) => {
                let scanned = self.scan(data, |r| m.get(start + r), push);
                st.scanned += scanned;
                st.matched += scanned;
            }
        }
    }

    fn disk(&self, part: &Part, s: usize, top: &mut TopK, st: &mut SearchStats) -> Result<()> {
        let seg = &part.view.sealed[s];
        let Body::Disk { index, .. } = &seg.body else { unreachable!("planned as SSD") };
        let res = match self.matching(part, s) {
            None => {
                let deleted = AsOf::new(&seg.tomb, part.seq);
                let live = seg.live() as f64 / seg.positions() as f64;
                self.walk(index, live, &|p| !deleted.deleted(p), st)?
            }
            Some((_, 0)) => return Ok(()),
            Some((m, n)) => {
                st.matched += n;
                self.filtered(index, m, *n, st)?
            }
        };
        res.iter().for_each(|r| top.push(r.id, r.distance));
        Ok(())
    }

    /// The graph walk over one SSD segment, `admitted` of whose rows can be
    /// returned.
    fn walk(
        &self,
        index: &DiskIndex,
        admitted: f64,
        admit: &dyn Fn(usize) -> bool,
        st: &mut SearchStats,
    ) -> Result<Vec<SearchResult>> {
        let (opts, l) = (&self.opts, self.opts.l.max(self.k));
        let params = DiskSearchParams {
            l: if opts.adapt_l { boosted_l(l, admitted, MAX_BOOST) } else { l },
            beam_width: opts.beam_width,
            rerank: opts.rerank,
        };
        let (res, ds) = index.search_filtered(&self.query, self.k, &params, admit)?;
        st.reads += ds.reads;
        st.rounds += ds.rounds;
        st.expanded += ds.expanded;
        Ok(res)
    }

    /// One SSD segment under a filter that `matching` (`n` live positions)
    /// holds.
    fn filtered(&self, index: &DiskIndex, matching: &Bitmap, n: usize, st: &mut SearchStats) -> Result<Vec<SearchResult>> {
        let (opts, k) = (&self.opts, self.k);
        let admit = |p: usize| matching.get(p);
        let l = opts.l.max(k);
        let scan = match opts.filter_mode {
            FilterMode::Scan => true,
            FilterMode::Graph => false,
            FilterMode::Auto => prefer_scan(n, index.len(), l, opts.beam_width, index.is_direct()),
        };
        if !scan {
            st.graph_segments += 1;
            let params = DiskSearchParams {
                l: if opts.adapt_l { boosted_l(l, n as f64 / index.len() as f64, MAX_FILTER_BOOST) } else { l },
                beam_width: opts.beam_width,
                rerank: opts.rerank,
            };
            let (res, ds) = index.search_filtered(&self.query, k, &params, &admit)?;
            st.reads += ds.reads;
            st.rounds += ds.rounds;
            st.expanded += ds.expanded;
            // The walk should meet about `l` matching nodes; far fewer means
            // the matches lie away from its path (or the boost was capped).
            if opts.filter_mode == FilterMode::Graph || ds.admitted * 2 >= l.min(n) {
                return Ok(res);
            }
            st.fallbacks += 1;
        }
        st.scan_segments += 1;
        let (res, ds) = index.search_subset(&self.query, k, l, &mut matching.ones(), &admit)?;
        st.reads += ds.reads;
        st.rounds += ds.rounds;
        st.pq_scanned += ds.pq_distances;
        Ok(res)
    }
}

/// Largest factor the candidate list of a walk is scaled by for deleted rows.
const MAX_BOOST: f64 = 8.0;
/// The same for a filter. `Auto` scans a segment where fewer than 1 / this
/// of the rows match, as the walk could not be scaled enough.
const MAX_FILTER_BOOST: f64 = 16.0;
/// Rows a filtered scan scores in the time of one round of a walk (one
/// read latency plus its CPU). Measured on SIFT1M (128 dimensions, 32-byte
/// PQ codes), 1 thread: a round took about 140 µs with direct I/O (reads
/// capped at 100,000 per second) and 16 µs from the page cache; a scan
/// about 40 ns per matching row (bitmap, PQ distance, code fetch).
const SCAN_ROWS_PER_ROUND_DIRECT: f64 = 3500.0;
const SCAN_ROWS_PER_ROUND_CACHED: f64 = 400.0;

/// Whether a filtered SSD segment with `matching` of its `positions` rows
/// is cheaper to scan than to walk with beam width `width`: the walk takes
/// about L / W + 4 rounds at its scaled L, the scan time grows with the
/// rows it scores.
pub(super) fn prefer_scan(matching: usize, positions: usize, l: usize, width: usize, direct: bool) -> bool {
    let f = matching as f64 / positions as f64;
    if f < 1.0 / MAX_FILTER_BOOST {
        return true;
    }
    let rounds = boosted_l(l, f, MAX_FILTER_BOOST) as f64 / width.max(1) as f64 + 4.0;
    let per_round = if direct { SCAN_ROWS_PER_ROUND_DIRECT } else { SCAN_ROWS_PER_ROUND_CACHED };
    (matching as f64) < rounds * per_round
}

/// Candidate list size for an SSD segment where only `admitted` of the
/// rows can be returned: the walk passes through rejected nodes, so it
/// needs a longer list to expand as many returnable ones.
pub(super) fn boosted_l(l: usize, admitted: f64, max_boost: f64) -> usize {
    let boost = (1.0 / admitted.max(1e-9)).min(max_boost);
    (l as f64 * boost).ceil() as usize
}
