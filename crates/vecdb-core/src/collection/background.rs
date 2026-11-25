//! Background work: a thread that flushes frozen segments, one that builds
//! indexes from flat segments, and one that rebuilds SSD segments. They are
//! separate so that none holds up another: the flushes bound the RAM and
//! logs a collection uses, index builds free the flat segments that writes
//! may wait for, and a compaction of a large segment can take hours.
//!
//! Failures are retried, with growing pauses, and reported by
//! [`Engine::stats`]. Writes that outrun the threads are slowed down to what
//! index builds keep up with (see `pace`), and wait for flushes
//! ([`Engine::room`]), but never for a compaction. Dropping the
//! [`Collection`](super::Collection) stops builds early, and waits for the
//! flush that is running.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, TryLockError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::flush::MAX_FROZEN;
use super::maintain::{IndexBuild, IndexingNow, Lane};
use super::{Collection, Engine};
use crate::cancel::{Cancel, Done};
use crate::error::{Error, Result};
use crate::par;

/// How long a waiting thread goes without looking for itself. Nothing
/// announces that a segment has enough deleted rows to be compacted.
const POLL: Duration = Duration::from_secs(1);
/// Pause before a failed flush or build is retried; doubles up to the
/// second value while it keeps failing.
const RETRY: (Duration, Duration) = (Duration::from_millis(100), Duration::from_secs(30));

/// How many background builds may run at once, over all the collections
/// that share the limit: so many of flat segments, and so many of SSD
/// segments besides. Those that wait go in the order they came. The builds
/// that run share their threads.
#[derive(Debug)]
pub struct BuildLimit {
    lanes: [Mutex<Places>; 2],
    /// Builds that run, also those not under the limit (a call of
    /// `maintain` or `optimize`).
    running: Arc<AtomicUsize>,
}

#[derive(Debug)]
struct Places {
    free: usize,
    /// Those that have a place or wait for one, in the order they came.
    line: Vec<InLine>,
    next: u64,
}

#[derive(Debug)]
struct InLine {
    ticket: u64,
    waits: bool,
    /// What it takes in, and how far it is.
    bytes: u64,
    done: Arc<Done>,
}

impl BuildLimit {
    /// `builds` of each kind at once (at least one).
    pub fn new(builds: usize) -> Arc<Self> {
        let places = || Mutex::new(Places { free: builds.max(1), line: Vec::new(), next: 0 });
        Arc::new(Self { lanes: [places(), places()], running: Arc::default() })
    }

    fn places(&self, lane: Lane) -> MutexGuard<'_, Places> {
        self.lanes[lane as usize].lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Lines up for a place, for a build that takes `bytes` in.
    pub(super) fn line_up(&self, lane: Lane, bytes: u64) -> Turn<'_> {
        let mut p = self.places(lane);
        let ticket = p.next;
        p.next += 1;
        let done = Arc::new(Done::default());
        // It does not wait for any of them if there is a place.
        let free = p.free > p.line.iter().filter(|b| b.waits).count();
        let before = if free { Vec::new() } else { p.line.iter().map(Before::new).collect() };
        p.line.push(InLine { ticket, waits: true, bytes, done: done.clone() });
        let way = Arc::new(Way { from: 0.0, before, bytes: bytes as f64, done });
        Turn { limit: self, lane, ticket, has: false, way }
    }

    /// Counts a build that runs until the result is dropped.
    pub(super) fn running(&self) -> Running {
        self.running.fetch_add(1, Ordering::Relaxed);
        Running(self.running.clone())
    }
}

/// A build's place under a limit, or in the line for one.
pub(super) struct Turn<'a> {
    limit: &'a BuildLimit,
    lane: Lane,
    ticket: u64,
    has: bool,
    pub way: Arc<Way>,
}

impl Turn<'_> {
    /// What it takes in, which can change while it waits: another job is due,
    /// or it has more to do.
    pub fn takes(&mut self, bytes: u64) {
        if self.way.bytes != bytes as f64 {
            let mut p = self.limit.places(self.lane);
            p.line.iter_mut().filter(|b| b.ticket == self.ticket).for_each(|b| b.bytes = bytes);
            let Way { from, before, done, .. } = &*self.way;
            let (before, done) = (before.clone(), done.clone());
            self.way = Arc::new(Way { from: *from, before, bytes: bytes as f64, done });
        }
    }

    /// Takes a place if there is one for each of those before in the line,
    /// and one more.
    pub fn come(&mut self) -> bool {
        let mut p = self.limit.places(self.lane);
        let free = p.free;
        if let Some(b) = p.line.iter_mut().filter(|b| b.waits).take(free).find(|b| b.ticket == self.ticket) {
            b.waits = false;
            p.free -= 1;
            self.has = true;
        }
        self.has
    }
}

impl Drop for Turn<'_> {
    fn drop(&mut self) {
        let mut p = self.limit.places(self.lane);
        p.line.retain(|b| b.ticket != self.ticket);
        if self.has {
            p.free += 1;
        }
        // Those behind need not wait for it any more, whatever came of it.
        self.way.done.set(1.0);
    }
}

/// How far a build has come since it lined up: first through the builds
/// before it, by their bytes, and then its own work.
#[derive(Debug, Default)]
pub(super) struct Way {
    /// How far it was when its turn came.
    from: f64,
    /// Those it waits for, until then.
    before: Vec<Before>,
    bytes: f64,
    /// Its own work.
    pub done: Arc<Done>,
}

/// A build that another one waits for.
#[derive(Clone, Debug)]
struct Before {
    /// What was left of it when that one lined up.
    bytes: f64,
    from: f64,
    done: Arc<Done>,
}

impl Before {
    fn new(b: &InLine) -> Self {
        let from = b.done.get() as f64;
        Self { bytes: b.bytes as f64 * (1.0 - from), from, done: b.done.clone() }
    }

    fn done(&self) -> f64 {
        if self.from < 1.0 { (self.done.get() as f64 - self.from) / (1.0 - self.from) } else { 1.0 }
    }
}

impl Way {
    /// How far, from 0 to 1.
    pub fn far(&self) -> f64 {
        let sum = |f: &dyn Fn(&Before) -> f64| self.before.iter().map(f).sum::<f64>();
        let far = if self.before.is_empty() {
            self.done.get() as f64
        } else {
            sum(&|b| b.bytes * b.done()) / (self.bytes + sum(&|b| b.bytes)).max(1.0)
        };
        self.from + (1.0 - self.from) * far
    }

    /// The rest of the way, when its turn has come.
    fn starts(&self) -> Arc<Self> {
        Arc::new(Self { from: self.far(), done: self.done.clone(), ..Self::default() })
    }
}

/// A build that runs (see [`BuildLimit`]).
pub(super) struct Running(Arc<AtomicUsize>);

impl Running {
    pub fn count(&self) -> &Arc<AtomicUsize> {
        &self.0
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// How long a builder waits before it looks whether its turn has come.
const FULL: Duration = Duration::from_millis(20);

/// What has been done, and what failed last.
#[derive(Clone, Debug, Default)]
pub(super) struct Progress {
    pub flush_error: Option<String>,
    /// Per lane.
    pub build_error: [Option<String>; 2],
    pub flushes: u64,
    pub builds: u64,
}

#[derive(Default)]
struct State {
    stop: bool,
    progress: Progress,
}

/// What the background threads, and those waiting for them, share.
#[derive(Default)]
pub(super) struct Background {
    state: Mutex<State>,
    changed: Condvar,
    /// Stops builds.
    cancel: Cancel,
}

impl Background {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Has waiting threads look again.
    pub fn notify(&self) {
        let _state = self.lock();
        self.changed.notify_all();
    }

    pub fn progress(&self) -> Progress {
        self.lock().progress.clone()
    }

    #[cfg(test)]
    pub fn cancel(&self) -> Cancel {
        self.cancel.clone()
    }

    /// Stops a build that is counted in `running`.
    pub fn cancel_build(&self, running: &Running) -> Cancel {
        self.cancel.sharing(running.count())
    }

    /// Records a flush (`Ok(false)`: there was nothing to flush).
    pub fn flushed(&self, r: &Result<bool>) {
        let mut s = self.lock();
        match r {
            Ok(true) => {
                s.progress.flushes += 1;
                s.progress.flush_error = None;
            }
            Ok(false) => {}
            Err(e) => s.progress.flush_error = Some(e.to_string()),
        }
        self.changed.notify_all();
    }

    /// Records a build that took segments of `lanes`.
    pub fn built(&self, lanes: [bool; 2]) {
        let mut s = self.lock();
        s.progress.builds += 1;
        s.progress.build_error.iter_mut().zip(lanes).filter(|(_, l)| *l).for_each(|(e, _)| *e = None);
        self.changed.notify_all();
    }

    pub fn build_failed(&self, lane: Lane, e: &Error) {
        self.lock().progress.build_error[lane as usize] = Some(e.to_string());
        self.changed.notify_all();
    }

    /// Waits until `done`; `false` if the threads are to stop instead.
    pub fn wait(&self, done: impl Fn(&Progress) -> bool) -> bool {
        self.wait_until(None, done) == Waited::Done
    }

    /// As [`wait`](Self::wait), but not longer than until `deadline`.
    pub fn wait_until(&self, deadline: Option<Instant>, done: impl Fn(&Progress) -> bool) -> Waited {
        let mut s = self.lock();
        loop {
            if s.stop {
                return Waited::Stopped;
            }
            if done(&s.progress) {
                return Waited::Done;
            }
            let left = deadline.map_or(POLL, |d| d.saturating_duration_since(Instant::now()));
            if left.is_zero() {
                return Waited::Late;
            }
            s = self.changed.wait_timeout(s, left.min(POLL)).unwrap_or_else(|e| e.into_inner()).0;
        }
    }

    /// Waits for `time`; `false` if the threads are to stop.
    fn pause(&self, time: Duration) -> bool {
        let s = self.lock();
        let (s, _) = self.changed.wait_timeout_while(s, time, |s| !s.stop).unwrap_or_else(|e| e.into_inner());
        !s.stop
    }

    fn stop(&self) {
        self.cancel.cancel();
        self.lock().stop = true;
        self.changed.notify_all();
    }
}

/// How a wait for background work ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Waited {
    Done,
    /// The threads are to stop.
    Stopped,
    Late,
}

/// The background threads of a collection; dropping it stops them.
pub(super) struct Workers {
    engine: Arc<Engine>,
    threads: Vec<JoinHandle<()>>,
}

impl Workers {
    pub fn start(engine: &Arc<Engine>) -> Result<Self> {
        let mut w = Self { engine: engine.clone(), threads: Vec::new() };
        if engine.opts.background {
            let runs: [(&str, fn(&Engine)); 3] = [
                ("vecdb-flush", flusher),
                ("vecdb-index", |e| builder(e, Lane::Flat)),
                ("vecdb-compact", |e| builder(e, Lane::Ssd)),
            ];
            for (name, run) in runs {
                let e = engine.clone();
                w.threads.push(thread::Builder::new().name(name.into()).spawn(move || run(&e))?);
            }
        }
        Ok(w)
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        self.engine.background.stop();
        for t in self.threads.drain(..) {
            // (One that panicked has said so in the stats.)
            let _ = t.join();
        }
    }
}

/// If a background thread panics, writes that wait for it get an error.
struct Died<F: Fn(Error)>(F);

impl<F: Fn(Error)> Drop for Died<F> {
    fn drop(&mut self) {
        if thread::panicking() {
            (self.0)(Error::Unavailable("the background thread panicked".into()));
        }
    }
}

fn flusher(e: &Engine) {
    let _died = Died(|err| e.background.flushed(&Err(err)));
    let mut retry = RETRY.0;
    while e.background.wait(|_| !e.view().frozen.is_empty()) {
        let one = e.flushing.lock().unwrap();
        let r = e.flush_oldest(&one);
        e.background.flushed(&r);
        drop(one);
        e.background.notify();
        if r.is_ok() {
            retry = RETRY.0;
        } else if e.background.pause(retry) {
            retry = (retry * 2).min(RETRY.1);
        }
    }
}

fn builder(e: &Engine, lane: Lane) {
    let _died = Died(|err| e.background.build_failed(lane, &err));
    let threads = match e.opts.threads {
        0 => (par::effective_threads(0) / 2).max(1),
        n => n,
    };
    let mut retry = RETRY.0;
    while e.background.wait(|_| e.next_job(lane).is_some()) {
        let bytes = e.next_job(lane).map_or(0, |(reason, inputs)| e.index_bytes_of(reason, &inputs));
        let mut turn = e.limit.line_up(lane, bytes);
        // What waits for an index, which jobs of the other lane do not change.
        // (Dropped before its turn is.)
        let waits = (lane == Lane::Flat).then(|| IndexingNow::new(e, |i| &mut i.waits, None));
        let wait = |reason: &str, inputs: Vec<u64>, way: &Arc<Way>| {
            if let Some(w) = &waits {
                w.set((reason == "index").then(|| IndexBuild::new(inputs, way)));
            }
        };
        while !turn.come() {
            let Some((reason, inputs)) = e.next_job(lane) else { break };
            turn.takes(e.index_bytes_of(reason, &inputs));
            wait(reason, inputs, &turn.way);
            e.at("build: waits");
            if !e.background.pause(FULL) {
                break;
            }
        }
        if !turn.has {
            continue;
        }
        // Waits for neither of the two with the other in its hands: a call
        // of `optimize` holds a lane for as long as it takes.
        let one = match e.lanes[lane as usize].try_lock() {
            Err(TryLockError::WouldBlock) => {
                drop((waits, turn));
                e.background.pause(FULL);
                continue;
            }
            one => one.unwrap(),
        };
        let r = e.next_job(lane).map(|(reason, inputs)| {
            turn.takes(e.index_bytes_of(reason, &inputs));
            let way = turn.way.starts();
            wait(reason, inputs.clone(), &way);
            e.rebuild_on(&inputs, reason, threads, &way)
        });
        drop((waits, one, turn));
        e.background.notify();
        match r {
            None => {}
            Some(Ok(_)) => retry = RETRY.0,
            Some(Err(Error::Cancelled)) => {}
            Some(Err(err)) => {
                e.background.build_failed(lane, &err);
                if e.background.pause(retry) {
                    retry = (retry * 2).min(RETRY.1);
                }
            }
        }
    }
}

/// Whether a write may go ahead.
pub(super) enum Room {
    Yes,
    /// Once the background threads have caught up; with what.
    Wait(&'static str),
    /// They keep failing; why.
    No(String),
}

impl Engine {
    /// Whether there is room for another write: the growing segment is not
    /// full or can be frozen. While flushes or index builds fail, it is let
    /// grow to 4 times its size, and the flat segments to twice the limit.
    pub(super) fn room(&self, p: &Progress) -> Room {
        let view = self.view();
        let seal = self.config.seal_bytes;
        let growing = view.growing.read().unwrap().bytes();
        match &p.flush_error {
            Some(e) if growing >= seal.saturating_mul(4) => return Room::No(format!("flushes fail: {e}")),
            None if self.opts.background && growing >= seal && view.frozen.len() >= MAX_FROZEN => {
                return Room::Wait("flushes");
            }
            _ => {}
        }
        if let (Some(zone), Some(e)) = (&self.zone, &p.build_error[Lane::Flat as usize]) {
            let rows: usize = self.lane_segments(Lane::Flat).iter().map(|s| s.positions()).sum();
            if rows as f64 * self.config.dim as f64 * 4.0 >= 2.0 * zone.to && self.room_due() {
                return Room::No(format!("index builds fail: {e}"));
            }
        }
        Room::Yes
    }

    /// Whether a job is due that gives back the room of flat segments.
    fn room_due(&self) -> bool {
        matches!(self.next_job(Lane::Flat), Some(("index" | "shrink", _)))
    }

    /// The bytes of vectors per second that may be written now; `None` if any
    /// number. They are only slowed down as long as a job is due that makes
    /// room, or may be after a flush, and neither fails.
    pub(super) fn write_rate(&self) -> Option<f64> {
        let rate = self.zone.as_ref()?.rate(self.unindexed())?;
        let p = self.background.progress();
        let flushing = !self.view().frozen.is_empty() && p.flush_error.is_none();
        let fails = p.build_error[Lane::Flat as usize].is_some();
        (!fails && (flushing || self.room_due())).then_some(rate)
    }

    /// With background threads: waits until there is nothing for them to
    /// do, and nothing half done. Fails if a flush or a build keeps failing.
    /// Other threads' writes can keep it from returning. (What is in the
    /// growing segment stays there; see [`flush`](Self::flush).)
    pub fn wait_idle(&self) -> Result<()> {
        if !self.opts.background {
            return Ok(());
        }
        let failed = |p: &Progress| p.flush_error.iter().chain(p.build_error.iter().flatten()).next().cloned();
        // A commit replaces segments in the view before it is done with its
        // old files and the counts, so the view alone does not tell. (Those
        // who hold these locks may take the state's, which is held here:
        // only try.)
        let idle = || match (self.flushing.try_lock(), self.lanes[0].try_lock(), self.lanes[1].try_lock()) {
            (Ok(_f), Ok(_a), Ok(_b)) => self.view().frozen.is_empty() && self.no_job(),
            _ => false,
        };
        self.background.wait(|p| failed(p).is_some() || idle());
        match failed(&self.background.progress()) {
            Some(e) => Err(Error::Unavailable(e)),
            None => Ok(()),
        }
    }
}

/// Shareable between threads, with or without background threads.
const _: () = {
    const fn shared<T: Send + Sync>() {}
    shared::<Collection>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_go_in_the_order_they_line_up() {
        let limit = BuildLimit::new(2);
        let mut turns: Vec<Turn> = (0..5).map(|_| limit.line_up(Lane::Ssd, 1)).collect();
        // The third has to wait, whoever asks first; the second need not
        // wait for the first to take its place.
        assert!(!turns[2].come() && turns[1].come() && turns[1].come() && !turns[2].come() && turns[0].come());
        assert!(!turns[2].come() && !turns[3].come());
        // The other lane has places of its own.
        let mut other = limit.line_up(Lane::Flat, 1);
        assert!(other.come());
        // One leaves the line, two their places: the next two have them.
        drop(turns.remove(2));
        drop(turns.remove(0));
        assert!(!turns[2].come());
        assert!(turns[1].come() && !turns[2].come());
        drop(turns.remove(0));
        assert!(turns[1].come());
        drop(turns);
        assert!(limit.places(Lane::Ssd).line.is_empty());
        assert_eq!((limit.places(Lane::Ssd).free, limit.places(Lane::Flat).free), (2, 1));
        drop(other);
        assert_eq!(limit.places(Lane::Flat).free, 2);
    }

    #[test]
    fn builds_come_closer_in_the_line() {
        let near = |x: f64, y: f64| assert!((x - y).abs() < 1e-6, "{x}, not {y}");
        let limit = BuildLimit::new(1);
        let mut a = limit.line_up(Lane::Flat, 100);
        assert!(a.come());
        let way = a.way.starts();
        way.done.set(0.25);
        near(way.far(), 0.25);
        // 75 are left of the first.
        let mut b = limit.line_up(Lane::Flat, 300);
        near(b.way.far(), 0.0);
        a.way.done.set(0.5);
        near(b.way.far(), 25.0 / 375.0);
        let mut c = limit.line_up(Lane::Flat, 125);
        near(c.way.far(), 0.0);
        // (Not what is in the other lane.)
        let other = limit.line_up(Lane::Ssd, 1000);
        assert!(other.way.before.is_empty());
        drop((a, other));
        near(b.way.far(), 75.0 / 375.0);
        near(c.way.far(), 50.0 / 475.0);
        assert!(!c.come() && b.come());
        let way = b.way.starts();
        near(way.far(), 0.2);
        way.done.set(0.5);
        near(way.far(), 0.6);
        near(c.way.far(), 200.0 / 475.0);
        // One that leaves without having done all is out of the way as well.
        drop(b);
        near(c.way.far(), 350.0 / 475.0);
        assert!(c.come());
        let way = c.way.starts();
        way.done.set(1.0);
        near(way.far(), 1.0);
        // Also from the middle of the line. Jobs that build nothing do not
        // count, nor what is as good as done.
        let x = limit.line_up(Lane::Flat, 100);
        let y = limit.line_up(Lane::Flat, 0);
        let z = limit.line_up(Lane::Flat, 100);
        near(z.way.far(), 0.0);
        drop(y);
        near(z.way.far(), 0.0);
        drop(x);
        near(z.way.far(), 0.5);
        drop(c);
        near(z.way.far(), 0.5);
        // Another job, which takes more in.
        let mut a = limit.line_up(Lane::Flat, 100);
        assert!(!a.come());
        z.way.done.set(0.5);
        near(a.way.far(), 50.0 / 200.0);
        a.takes(350);
        near(a.way.far(), 50.0 / 450.0);
        let b = limit.line_up(Lane::Flat, 100);
        near(b.way.before[1].bytes, 350.0);
        drop(b);
        a.takes(0);
        near(a.way.far(), 0.5);
        // It is the same one to those behind it.
        a.takes(350);
        let b = limit.line_up(Lane::Flat, 100);
        a.takes(700);
        a.way.done.set(0.5);
        near(b.way.far(), 175.0 / 500.0);
        drop(a);
        near(b.way.far(), 350.0 / 500.0);
        drop(b);
        // Nothing to go by.
        let none = limit.line_up(Lane::Flat, 0);
        drop(z);
        near(none.way.far(), 1.0);
        drop(none);
        let mut idle = limit.line_up(Lane::Flat, 0);
        assert!(idle.way.before.is_empty() && idle.come());
        let only = limit.line_up(Lane::Flat, 0);
        near(only.way.far(), 0.0);
        assert_eq!(only.way.before.len(), 1);
    }

    /// Those who need not wait for anybody are not on their way yet.
    #[test]
    fn builds_do_not_wait_for_those_beside_them() {
        let limit = BuildLimit::new(2);
        let mut a = limit.line_up(Lane::Flat, 100);
        assert!(a.come());
        let mut b = limit.line_up(Lane::Flat, 100);
        a.way.done.set(0.5);
        assert!(b.way.before.is_empty() && b.way.far() == 0.0);
        // (As both places will be taken.)
        let c = limit.line_up(Lane::Flat, 100);
        assert!(c.way.before.len() == 2 && b.come());
        drop(a);
        assert!((c.way.far() - 50.0 / 250.0).abs() < 1e-6);
        let d = limit.line_up(Lane::Flat, 100);
        assert_eq!(d.way.before.len(), 2);
    }

    #[test]
    fn builds_that_run_share_their_threads() {
        let limit = BuildLimit::new(1);
        let background = Background::default();
        let a = limit.running();
        let cancel = background.cancel_build(&a);
        assert_eq!((cancel.threads(6), cancel.threads(0)), (6, 0));
        let b = limit.running();
        assert_eq!((cancel.threads(6), cancel.threads(1)), (3, 1));
        assert_eq!(cancel.threads(0), (par::effective_threads(0) / 2).max(1));
        let c = limit.running();
        assert_eq!(cancel.threads(6), 2);
        drop((b, c));
        assert_eq!(cancel.threads(6), 6);
        assert!(!cancel.is_cancelled());
        background.stop();
        assert!(cancel.is_cancelled());
        drop(a);
        assert_eq!(limit.running.load(Ordering::Relaxed), 0);
    }
}
