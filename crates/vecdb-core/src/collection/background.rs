//! Background work: a thread that flushes frozen segments and one that
//! builds indexes and compacts. They are separate, so a build, which can
//! take minutes, does not hold up the flushes that bound the RAM and logs a
//! collection uses.
//!
//! Failures are retried, with growing pauses, and reported by
//! [`Engine::stats`]. Writes that outrun the threads wait
//! ([`Engine::room`]). Dropping the [`Collection`](super::Collection) stops
//! a build early, and waits for the flush that is running.

use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use super::flush::MAX_FROZEN;
use super::segment::Body;
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

/// How many background builds may run at once.
#[derive(Debug)]
pub struct BuildLimit {
    free: Mutex<usize>,
}

impl BuildLimit {
    /// `builds` at once (at least one).
    pub fn new(builds: usize) -> Arc<Self> {
        Arc::new(Self { free: Mutex::new(builds.max(1)) })
    }

    fn take(&self) -> Option<Slot<'_>> {
        let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
        *free = free.checked_sub(1)?;
        Some(Slot(self))
    }
}

/// A build's place under a limit.
struct Slot<'a>(&'a BuildLimit);

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        *self.0.free.lock().unwrap_or_else(|e| e.into_inner()) += 1;
    }
}

/// How long a builder waits before it looks for a free place again.
const FULL: Duration = Duration::from_millis(100);

/// What has been done, and what failed last.
#[derive(Clone, Debug, Default)]
pub(super) struct Progress {
    pub flush_error: Option<String>,
    pub build_error: Option<String>,
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

    pub fn cancel(&self) -> Cancel {
        self.cancel.clone()
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

    /// Records a build or compaction.
    pub fn built(&self, r: &Result<()>) {
        let mut s = self.lock();
        match r {
            Ok(()) => {
                s.progress.builds += 1;
                s.progress.build_error = None;
            }
            Err(e) => s.progress.build_error = Some(e.to_string()),
        }
        self.changed.notify_all();
    }

    /// Waits until `done`; `false` if the threads are to stop instead.
    pub fn wait(&self, done: impl Fn(&Progress) -> bool) -> bool {
        let mut s = self.lock();
        loop {
            if s.stop {
                return false;
            }
            if done(&s.progress) {
                return true;
            }
            s = self.changed.wait_timeout(s, POLL).unwrap_or_else(|e| e.into_inner()).0;
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

/// The background threads of a collection; dropping it stops them.
pub(super) struct Workers {
    engine: Arc<Engine>,
    threads: Vec<JoinHandle<()>>,
}

impl Workers {
    pub fn start(engine: &Arc<Engine>) -> Result<Self> {
        let mut w = Self { engine: engine.clone(), threads: Vec::new() };
        if engine.opts.background {
            for (name, run) in [("vecdb-flush", flusher as fn(&Engine)), ("vecdb-build", builder)] {
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
struct Died<'a> {
    engine: &'a Engine,
    record: fn(&Background, Error),
}

impl Drop for Died<'_> {
    fn drop(&mut self) {
        if thread::panicking() {
            (self.record)(&self.engine.background, Error::Unavailable("the background thread panicked".into()));
        }
    }
}

fn flusher(e: &Engine) {
    let _died = Died { engine: e, record: |b, err| b.flushed(&Err(err)) };
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

fn builder(e: &Engine) {
    let _died = Died { engine: e, record: |b, err| b.built(&Err(err)) };
    let threads = match e.opts.threads {
        0 => (par::effective_threads(0) / 2).max(1),
        n => n,
    };
    let mut retry = RETRY.0;
    while e.background.wait(|_| e.next_job().is_some()) {
        let one = e.maintaining.lock().unwrap();
        let Some((reason, job)) = e.next_job() else {
            drop(one);
            e.background.notify();
            continue;
        };
        let slot = e.opts.builds.as_ref().map(|limit| limit.take());
        if let Some(None) = slot {
            drop(one);
            e.background.pause(FULL);
            continue;
        }
        let r = e.rebuild(&job, reason, threads);
        drop(slot);
        drop(one);
        e.background.notify();
        match r {
            Ok(_) => retry = RETRY.0,
            Err(Error::Cancelled) => {}
            Err(err) => {
                e.background.built(&Err(err));
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
    /// Once the background threads have caught up.
    Wait,
    /// They keep failing; why.
    No(String),
}

impl Engine {
    /// Whether there is room for another write: the growing segment is not
    /// full or can be frozen, and `max_unindexed_bytes` is not reached.
    /// While flushes or builds fail, up to 4 and 2 times as much is let in.
    pub(super) fn room(&self, p: &Progress) -> Room {
        let view = self.view();
        let seal = self.config.seal_bytes;
        let growing = view.growing.read().unwrap().bytes();
        match &p.flush_error {
            Some(e) if growing >= seal.saturating_mul(4) => return Room::No(format!("flushes fail: {e}")),
            None if self.opts.background && growing >= seal && view.frozen.len() >= MAX_FROZEN => return Room::Wait,
            _ => {}
        }
        let limit = self.opts.max_unindexed_bytes;
        if self.opts.background && limit != 0 {
            let flat = view.sealed.iter().filter(|s| matches!(s.body, Body::Flat(_))).map(|s| s.positions() as u64);
            let bytes = flat.sum::<u64>() * self.config.dim as u64 * 4;
            if bytes >= limit && matches!(self.next_job(), Some(("index" | "shrink", _))) {
                match &p.build_error {
                    None => return Room::Wait,
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
        let failed = |p: &Progress| p.flush_error.clone().or(p.build_error.clone());
        // A commit replaces segments in the view before it is done with its
        // old files and the counts, so the view alone does not tell. (Those
        // who hold these locks may take the state's, which is held here:
        // only try.)
        let idle = || match (self.flushing.try_lock(), self.maintaining.try_lock()) {
            (Ok(_f), Ok(_m)) => self.view().frozen.is_empty() && self.next_job().is_none(),
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
