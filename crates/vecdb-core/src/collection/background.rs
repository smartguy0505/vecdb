//! Background work: a thread that flushes frozen segments, one that builds
//! indexes from flat segments, and one that rebuilds SSD segments. They are
//! separate so that none holds up another: the flushes bound the RAM and
//! logs a collection uses, index builds free the flat segments that writes
//! may wait for, and a compaction of a large segment can take hours.
//!
//! Failures are retried, with growing pauses, and reported by
//! [`Engine::stats`]. Writes that outrun the threads wait
//! ([`Engine::room`]), for a flush or an index build but never for a
//! compaction. Dropping the [`Collection`](super::Collection) stops builds
//! early, and waits for the flush that is running.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, TryLockError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::flush::MAX_FROZEN;
use super::maintain::Lane;
use super::{Collection, Engine};
use crate::cancel::Cancel;
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
    /// Tickets of those that wait, in order.
    waiting: VecDeque<u64>,
    next: u64,
}

impl BuildLimit {
    /// `builds` of each kind at once (at least one).
    pub fn new(builds: usize) -> Arc<Self> {
        let places = || Mutex::new(Places { free: builds.max(1), waiting: VecDeque::new(), next: 0 });
        Arc::new(Self { lanes: [places(), places()], running: Arc::default() })
    }

    fn places(&self, lane: Lane) -> MutexGuard<'_, Places> {
        self.lanes[lane as usize].lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Lines up for a place.
    fn line_up(&self, lane: Lane) -> Turn<'_> {
        let mut p = self.places(lane);
        let ticket = p.next;
        p.next += 1;
        p.waiting.push_back(ticket);
        Turn { limit: self, lane, ticket, has: false }
    }

    /// Counts a build that runs until the result is dropped.
    pub(super) fn running(&self) -> Running {
        self.running.fetch_add(1, Ordering::Relaxed);
        Running(self.running.clone())
    }
}

/// A build's place under a limit, or in the line for one.
struct Turn<'a> {
    limit: &'a BuildLimit,
    lane: Lane,
    ticket: u64,
    has: bool,
}

impl Turn<'_> {
    /// Takes a place if there is one for each of those before in the line,
    /// and one more.
    fn come(&mut self) -> bool {
        let mut p = self.limit.places(self.lane);
        if let Some(before) = p.waiting.iter().position(|&t| t == self.ticket)
            && before < p.free
        {
            p.waiting.remove(before);
            p.free -= 1;
            self.has = true;
        }
        self.has
    }
}

impl Drop for Turn<'_> {
    fn drop(&mut self) {
        let mut p = self.limit.places(self.lane);
        if self.has {
            p.free += 1;
        } else {
            p.waiting.retain(|&t| t != self.ticket);
        }
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
        let mut turn = e.limit.line_up(lane);
        while !turn.come() && e.next_job(lane).is_some() && e.background.pause(FULL) {}
        if !turn.has {
            continue;
        }
        // Waits for neither of the two with the other in its hands: a call
        // of `optimize` holds a lane for as long as it takes.
        let one = match e.lanes[lane as usize].try_lock() {
            Err(TryLockError::WouldBlock) => {
                drop(turn);
                e.background.pause(FULL);
                continue;
            }
            one => one.unwrap(),
        };
        let r = e.next_job(lane).map(|(reason, inputs)| e.rebuild(&inputs, reason, threads));
        drop((one, turn));
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
    /// full or can be frozen, and `max_unindexed_bytes` is not reached.
    /// While flushes or index builds fail, up to 4 and 2 times as much is
    /// let in.
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
        let limit = self.opts.max_unindexed_bytes;
        if self.opts.background && limit != 0 {
            let bytes = self.lane_segments(Lane::Flat).iter().map(|s| s.positions() as u64).sum::<u64>()
                * self.config.dim as u64
                * 4;
            if bytes >= limit && matches!(self.next_job(Lane::Flat), Some(("index" | "shrink", _))) {
                match &p.build_error[Lane::Flat as usize] {
                    None => return Room::Wait("index builds"),
                    Some(e) if bytes >= limit.saturating_mul(2) => {
                        return Room::No(format!("index builds fail: {e}"));
                    }
                    Some(_) => {}
                }
            }
        }
        Room::Yes
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
        let mut turns: Vec<Turn> = (0..5).map(|_| limit.line_up(Lane::Ssd)).collect();
        // The third has to wait, whoever asks first; the second need not
        // wait for the first to take its place.
        assert!(!turns[2].come() && turns[1].come() && turns[1].come() && !turns[2].come() && turns[0].come());
        assert!(!turns[2].come() && !turns[3].come());
        // The other lane has places of its own.
        let mut other = limit.line_up(Lane::Flat);
        assert!(other.come());
        // One leaves the line, two their places: the next two have them.
        drop(turns.remove(2));
        drop(turns.remove(0));
        assert!(!turns[2].come());
        assert!(turns[1].come() && !turns[2].come());
        drop(turns.remove(0));
        assert!(turns[1].come());
        drop(turns);
        assert!(limit.lanes.iter().all(|l| l.lock().unwrap().waiting.is_empty()));
        assert_eq!((limit.places(Lane::Ssd).free, limit.places(Lane::Flat).free), (2, 1));
        drop(other);
        assert_eq!(limit.places(Lane::Flat).free, 2);
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
