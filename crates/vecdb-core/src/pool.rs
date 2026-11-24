//! A pool of threads that help searches. Each search is split into tasks,
//! which the calling thread works through with whatever help there is:
//!
//! - **Tasks that mostly wait** (walks with direct I/O, shards elsewhere)
//!   take any idle thread.
//! - **Tasks that compute** share `cpus` evenly between the searches that
//!   have some. One search may use all of them; with more searches than
//!   that, each is left to its own thread, as without a pool. More threads
//!   would not add work done: scans of vectors in RAM are limited by the
//!   memory bandwidth, which a few cores use up, and a search with many
//!   threads takes it from those with one. So helpers look at every task
//!   whether they are still within the share of the search they help, and
//!   otherwise go to another.
//!
//! A search never waits for a busy pool: what nobody helps with, its own
//! thread does.

use std::any::Any;
use std::cell::Cell;
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::{self, JoinHandle};

use crate::error::Result;
use crate::par;

/// Threads that help searches, shared by any number of collections. Stops
/// when the last `Arc` to it is dropped.
pub struct SearchPool {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
}

struct Shared {
    /// CPUs for tasks that compute.
    cpus: usize,
    threads: usize,
    /// Searches with threads in tasks that compute, the callers' included.
    searches: AtomicUsize,
    state: Mutex<State>,
    wake: Condvar,
}

#[derive(Default)]
struct State {
    /// Work that threads are invited to, once per thread.
    queue: VecDeque<Arc<dyn Work>>,
    idle: usize,
    stop: bool,
}

trait Work: Send + Sync {
    /// Takes part until there is nothing left to claim.
    fn help(&self, pool: &Shared);
}

/// The tasks of one search, `0..n`; the first `waits` of them mostly wait.
struct Tasks {
    next: AtomicUsize,
    n: usize,
    waits: usize,
    /// Threads in tasks that compute, and the helpers among them.
    computing: AtomicUsize,
    helpers: AtomicUsize,
}

impl Tasks {
    fn new(n: usize, waits: usize) -> Self {
        Self { next: AtomicUsize::new(0), n, waits, computing: AtomicUsize::new(0), helpers: AtomicUsize::new(0) }
    }
}

/// One thread's part in a search: claims task numbers, each handed out once.
pub(crate) struct Claims<'a> {
    tasks: &'a Tasks,
    /// Whose CPUs it counts against.
    pool: Option<&'a Shared>,
    helper: bool,
    computing: Cell<bool>,
}

impl<'a> Claims<'a> {
    fn new(tasks: &'a Tasks, pool: Option<&'a Shared>, helper: bool) -> Self {
        Self { tasks, pool, helper, computing: Cell::new(false) }
    }

    /// The next task, which the last one claimed is done before.
    pub fn next(&self) -> Option<usize> {
        let t = self.tasks;
        if self.helper {
            let waiting = update(&t.next, |i| (i < t.waits).then_some(i + 1));
            if waiting.is_some() {
                return waiting;
            }
            if !self.within_share() {
                return None;
            }
        }
        let i = t.next.fetch_add(1, Ordering::Relaxed);
        if i >= t.n {
            return None;
        }
        if i >= t.waits && !self.computing.get() {
            self.start_computing();
        }
        Some(i)
    }

    /// Hands out no more, to anyone.
    pub fn close(&self) {
        self.tasks.next.store(self.tasks.n, Ordering::Relaxed);
    }

    /// Whether a helper may compute: there is a place for it next to the
    /// search's own thread. Takes the place, or gives it back.
    fn within_share(&self) -> bool {
        let Some(p) = self.pool else { return true };
        let (count, places) = (&self.tasks.helpers, p.share(0) - 1);
        if !self.computing.get() {
            let in_now = update(count, |n| (n < places).then_some(n + 1)).is_some();
            if in_now {
                self.start_computing();
            }
            in_now
        } else if update(count, |n| (n > places).then(|| n - 1)).is_some() {
            self.stop_computing();
            false
        } else {
            true
        }
    }

    fn start_computing(&self) {
        self.computing.set(true);
        if let (Some(p), 0) = (self.pool, self.tasks.computing.fetch_add(1, Ordering::Relaxed)) {
            p.searches.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn stop_computing(&self) {
        self.computing.set(false);
        if let (Some(p), 1) = (self.pool, self.tasks.computing.fetch_sub(1, Ordering::Relaxed)) {
            p.searches.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl Drop for Claims<'_> {
    fn drop(&mut self) {
        if self.computing.get() {
            if self.helper {
                self.tasks.helpers.fetch_sub(1, Ordering::Relaxed);
            }
            self.stop_computing();
        }
    }
}

/// Replaces the number by what `to` makes of it, if anything: the old one.
fn update(a: &AtomicUsize, to: impl Fn(usize) -> Option<usize>) -> Option<usize> {
    let mut n = a.load(Ordering::Relaxed);
    while let Some(x) = to(n) {
        match a.compare_exchange_weak(n, x, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return Some(n),
            Err(now) => n = now,
        }
    }
    None
}

struct Job<T, F> {
    tasks: Tasks,
    /// Gone once the caller is done, and with it what it holds on to: the
    /// job itself may stay queued for a while.
    run: RwLock<Option<F>>,
    state: Mutex<JobState<T>>,
    /// The last helper left.
    left: Condvar,
}

struct JobState<T> {
    out: Vec<T>,
    helping: usize,
    /// The caller is done: whoever comes now is too late.
    closed: bool,
    panic: Option<Box<dyn Any + Send>>,
}

impl<T: Send, F: Fn(&Claims) -> T + Send + Sync> Work for Job<T, F> {
    fn help(&self, pool: &Shared) {
        let claims = Claims::new(&self.tasks, Some(pool), true);
        {
            let mut s = self.state.lock().unwrap();
            if s.closed {
                return;
            }
            s.helping += 1;
        }
        let r = {
            let run = self.run.read().unwrap();
            let run = run.as_ref().expect("there while anyone helps");
            catch_unwind(AssertUnwindSafe(|| run(&claims)))
        };
        drop(claims);
        let mut s = self.state.lock().unwrap();
        match r {
            Ok(v) => s.out.push(v),
            Err(p) => s.panic = Some(p),
        }
        s.helping -= 1;
        if s.helping == 0 {
            self.left.notify_all();
        }
    }
}

impl SearchPool {
    /// Starts `threads` threads (0 = two per CPU) that help searches, with
    /// tasks that compute as long as fewer than `cpus` searches have some
    /// (see the module docs). 0 = a quarter of the CPUs, which on the machine
    /// measured (12 hardware threads, 2 memory channels) gave a search
    /// nearly all it can gain and cost the others nothing.
    pub fn new(threads: usize, cpus: usize) -> Result<Arc<Self>> {
        let have = par::effective_threads(0);
        let threads = if threads == 0 { have * 2 } else { threads };
        let shared = Shared {
            cpus: if cpus == 0 { (have / 4).max(have.min(2)) } else { cpus },
            threads,
            searches: AtomicUsize::new(0),
            state: Mutex::default(),
            wake: Condvar::new(),
        };
        let mut pool = Self { shared: Arc::new(shared), threads: Vec::new() };
        for i in 0..threads {
            let shared = pool.shared.clone();
            let t = thread::Builder::new().name(format!("vecdb-search-{i}")).spawn(move || shared.work())?;
            pool.threads.push(t);
        }
        Ok(Arc::new(pool))
    }

    pub fn threads(&self) -> usize {
        self.threads.len()
    }

    /// Runs `run` on this thread and on helping ones (at most `most`), all
    /// of which claim tasks below `n` until there are none left; the first
    /// `waits` tasks mostly wait. Returns what each of them made of its
    /// tasks, once all are done.
    pub(crate) fn run<T, F>(&self, n: usize, waits: usize, most: usize, run: F) -> Vec<T>
    where
        T: Send + 'static,
        F: Fn(&Claims) -> T + Send + Sync + 'static,
    {
        let tasks = Tasks::new(n, waits.min(n));
        let most = most.min(n.saturating_sub(1));
        if most == 0 {
            return vec![run(&Claims::new(&tasks, Some(&self.shared), false))];
        }
        let state = JobState { out: Vec::new(), helping: 0, closed: false, panic: None };
        let job = Arc::new(Job { tasks, run: RwLock::new(Some(run)), state: Mutex::new(state), left: Condvar::new() });
        self.shared.invite(most, &job.tasks, &job);
        let mine = {
            let run = job.run.read().unwrap();
            let claims = Claims::new(&job.tasks, Some(&self.shared), false);
            catch_unwind(AssertUnwindSafe(|| run.as_ref().expect("not taken yet")(&claims)))
        };
        let mut s = job.state.lock().unwrap();
        s.closed = true;
        while s.helping > 0 {
            s = job.left.wait(s).unwrap();
        }
        let (mut out, panic) = (std::mem::take(&mut s.out), s.panic.take());
        drop(s);
        drop(job.run.write().unwrap().take());
        match (mine, panic) {
            (Err(p), _) | (_, Some(p)) => resume_unwind(p),
            (Ok(mine), None) => out.push(mine),
        }
        out
    }
}

/// What `run` does without a pool.
pub(crate) fn run_alone<T>(n: usize, run: impl FnOnce(&Claims) -> T) -> Vec<T> {
    vec![run(&Claims::new(&Tasks::new(n, 0), None, false))]
}

impl Drop for SearchPool {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().stop = true;
        self.shared.wake.notify_all();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

impl std::fmt::Debug for SearchPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SearchPool({} threads, {} CPUs)", self.threads.len(), self.shared.cpus)
    }
}

impl Shared {
    /// Threads a search may compute with, next to the others that do (and
    /// `more`).
    fn share(&self, more: usize) -> usize {
        (self.cpus / (self.searches.load(Ordering::Relaxed) + more).max(1)).max(1)
    }

    fn invite<W: Work + 'static>(&self, most: usize, tasks: &Tasks, work: &Arc<W>) {
        let most = most.min(tasks.waits.max(self.share(1)) - 1);
        if most == 0 {
            return;
        }
        let mut s = self.state.lock().unwrap();
        for _ in 0..most {
            // The oldest invitations are likely to be out of date.
            if s.queue.len() >= self.threads {
                s.queue.pop_front();
            }
            s.queue.push_back(work.clone());
        }
        let wake = s.idle.min(most);
        drop(s);
        for _ in 0..wake {
            self.wake.notify_one();
        }
    }

    fn work(&self) {
        let mut s = self.state.lock().unwrap();
        loop {
            if let Some(w) = s.queue.pop_front() {
                drop(s);
                w.help(self);
                drop(w);
                s = self.state.lock().unwrap();
            } else if s.stop {
                return;
            } else {
                s.idle += 1;
                s = self.wake.wait(s).unwrap();
                s.idle -= 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, Instant};

    fn sum(c: &Claims) -> Vec<usize> {
        let mut mine = Vec::new();
        while let Some(i) = c.next() {
            mine.push(i);
        }
        mine
    }

    fn wait_idle(p: &SearchPool) {
        let t0 = Instant::now();
        while p.shared.state.lock().unwrap().idle < p.threads() {
            assert!(t0.elapsed() < Duration::from_secs(10));
            thread::yield_now();
        }
    }

    #[test]
    fn every_task_is_done_once() {
        let pool = SearchPool::new(3, 4).unwrap();
        for n in [0, 1, 2, 5, 1000] {
            for _ in 0..50 {
                let mut all: Vec<usize> = pool.run(n, n / 2, n, sum).concat();
                all.sort_unstable();
                assert_eq!(all, (0..n).collect::<Vec<_>>());
            }
        }
        assert_eq!(run_alone(4, sum), [vec![0, 1, 2, 3]]);
        // From many threads at once.
        thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..200 {
                        let mut all: Vec<usize> = pool.run(17, 3, 17, sum).concat();
                        all.sort_unstable();
                        assert_eq!(all, (0..17).collect::<Vec<_>>());
                    }
                });
            }
        });
    }

    #[test]
    fn idle_threads_help_and_busy_ones_are_not_waited_for() {
        let pool = SearchPool::new(2, 4).unwrap();
        wait_idle(&pool);
        // Three tasks that can only end together: all three threads run.
        let meet = Arc::new((Mutex::new(0), Condvar::new()));
        let m = meet.clone();
        let parts = pool.run(3, 0, 3, move |c| {
            let mut done = 0;
            while c.next().is_some() {
                let mut here = m.0.lock().unwrap();
                *here += 1;
                m.1.notify_all();
                let t0 = Instant::now();
                while *here < 3 {
                    assert!(t0.elapsed() < Duration::from_secs(10));
                    here = m.1.wait_timeout(here, Duration::from_secs(10)).unwrap().0;
                }
                done += 1;
            }
            done
        });
        assert_eq!(parts.iter().sum::<usize>(), 3);
        assert_eq!(parts.len(), 3);

        // Both pool threads held up: a search is done without them.
        wait_idle(&pool);
        let held = Arc::new(AtomicBool::new(true));
        thread::scope(|s| {
            let h = held.clone();
            s.spawn(|| {
                pool.run(3, 0, 3, move |c| {
                    while c.next().is_some() {
                        while h.load(Ordering::Relaxed) {
                            thread::sleep(Duration::from_millis(1));
                        }
                    }
                })
            });
            let t0 = Instant::now();
            while pool.shared.state.lock().unwrap().idle > 0 {
                assert!(t0.elapsed() < Duration::from_secs(10));
                thread::yield_now();
            }
            assert_eq!(pool.run(5, 0, 5, sum), [vec![0, 1, 2, 3, 4]]);
            held.store(false, Ordering::Relaxed);
        });
    }

    /// When a search returns, nothing holds on to what it captured.
    #[test]
    fn captures_are_gone_when_run_returns() {
        let pool = SearchPool::new(4, 5).unwrap();
        let held = Arc::new(());
        for _ in 0..2000 {
            let h = held.clone();
            pool.run(5, 0, 5, move |c| {
                let _h = &h;
                sum(c)
            });
            assert_eq!(Arc::strong_count(&held), 1);
        }
        assert_eq!(searches(&pool), 0);
    }

    #[test]
    fn a_panic_in_a_helper_reaches_the_caller_and_the_pool_lives_on() {
        let pool = SearchPool::new(2, 4).unwrap();
        let caller = thread::current().id();
        for _ in 0..20 {
            wait_idle(&pool);
            let r = catch_unwind(AssertUnwindSafe(|| {
                pool.run(3, 0, 3, move |c| {
                    while c.next().is_some() {
                        if thread::current().id() != caller {
                            panic!("helper");
                        }
                        thread::sleep(Duration::from_millis(20));
                    }
                })
            }));
            assert!(r.is_err());
        }
        wait_idle(&pool);
        assert_eq!(pool.run(9, 0, 9, sum).concat().len(), 9);
        assert_eq!(searches(&pool), 0);
    }

    fn searches(p: &SearchPool) -> usize {
        // No thread has died of a panic outside of a task, either.
        assert!(p.threads.iter().all(|t| !t.is_finished()));
        p.shared.searches.load(Ordering::Relaxed)
    }

    /// Runs `n` tasks of a few ms: the threads that took part, the most of
    /// them at once, and the most at once in tasks that compute.
    fn crowd(pool: &SearchPool, n: usize, waits: usize) -> (usize, usize, usize) {
        wait_idle(pool);
        let now: Arc<[(AtomicUsize, AtomicUsize); 2]> = Arc::default();
        let at = now.clone();
        let parts = pool.run(n, waits, n, move |c| {
            let mut did = 0;
            while let Some(i) = c.next() {
                let in_it = if i < waits { &at[..1] } else { &at[..] };
                for a in in_it {
                    a.1.fetch_max(a.0.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
                }
                thread::sleep(Duration::from_millis(5));
                for a in in_it {
                    a.0.fetch_sub(1, Ordering::SeqCst);
                }
                did += 1;
            }
            did
        });
        assert_eq!(parts.iter().sum::<usize>(), n);
        let most = |i: usize| now[i].1.load(Ordering::SeqCst);
        (parts.iter().filter(|&&did| did > 0).count(), most(0), most(1))
    }

    /// Runs `f` while `n` searches are held up in a task (one that waits,
    /// or one that computes).
    fn with_searches(pool: &SearchPool, n: usize, waits: usize, f: impl FnOnce()) {
        let hold = Arc::new(std::sync::Barrier::new(n + 1));
        thread::scope(|s| {
            for _ in 0..n {
                let hold = hold.clone();
                s.spawn(move || {
                    pool.run(1, waits, 0, move |c| {
                        c.next();
                        hold.wait();
                        hold.wait();
                    })
                });
            }
            hold.wait();
            let r = catch_unwind(AssertUnwindSafe(f));
            hold.wait();
            r.unwrap_or_else(|p| resume_unwind(p));
        });
        assert_eq!(searches(pool), 0);
    }

    #[test]
    fn tasks_that_compute_share_the_cpus() {
        let pool = SearchPool::new(8, 4).unwrap();
        // One search: all CPUs, and no more.
        assert_eq!(crowd(&pool, 100, 0), (4, 4, 4));
        // Tasks that wait: any idle thread, at most one per task.
        let (threads, at_once, _) = crowd(&pool, 18, 18);
        assert!(threads > 4 && at_once > 4, "{threads} threads, {at_once} at once");
        assert!(crowd(&pool, 3, 3).0 <= 3);
        // Both: as many as wait, and then four.
        let (threads, at_once, computing) = crowd(&pool, 60, 9);
        assert!(threads > 4 && at_once > 4 && computing == 4, "{threads} threads, {at_once}, {computing} at once");
        assert_eq!(searches(&pool), 0);
        // As many searches computing as there are CPUs: nobody gets help
        // with that, only with waiting.
        with_searches(&pool, 4, 0, || {
            assert_eq!(searches(&pool), 4);
            assert_eq!(crowd(&pool, 20, 0), (1, 1, 1));
            let (threads, at_once, computing) = crowd(&pool, 48, 18);
            assert!(threads > 4 && at_once > 4 && computing == 1, "{threads} threads, {at_once}, {computing} at once");
        });
        // Those that wait take nothing from those that compute.
        with_searches(&pool, 4, 1, || {
            assert_eq!(searches(&pool), 0);
            assert_eq!(crowd(&pool, 100, 0), (4, 4, 4));
        });
    }

    /// A search that goes on until told to stop, and its threads: when each
    /// last took a task.
    #[derive(Clone, Default)]
    struct Endless {
        seen: Arc<Mutex<std::collections::HashMap<thread::ThreadId, Instant>>>,
        stop: Arc<AtomicBool>,
    }

    impl Endless {
        fn run(&self, pool: &SearchPool) {
            let me = self.clone();
            pool.run(1_000_000, 0, 1_000_000, move |c| {
                while c.next().is_some() && !me.stop.load(Ordering::Relaxed) {
                    me.seen.lock().unwrap().insert(thread::current().id(), Instant::now());
                    thread::sleep(Duration::from_millis(2));
                }
            });
        }

        fn threads(&self) -> usize {
            self.seen.lock().unwrap().values().filter(|t| t.elapsed() < Duration::from_millis(100)).count()
        }
    }

    impl Drop for Endless {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
        }
    }

    /// Helpers leave one search for another, until both have their share,
    /// and stay so.
    #[test]
    fn helpers_leave_when_others_compute() {
        let pool = SearchPool::new(8, 4).unwrap();
        let until = |what: &str, done: &dyn Fn() -> bool| {
            let t0 = Instant::now();
            while !done() {
                assert!(t0.elapsed() < Duration::from_secs(20), "{what}");
                thread::sleep(Duration::from_millis(5));
            }
        };
        wait_idle(&pool);
        thread::scope(|s| {
            let pool = &pool;
            // (Dropped, they stop, also when an assertion fails.)
            let (a, b) = (Endless::default(), Endless::default());
            let run = |e: &Endless| {
                let e = Endless { seen: e.seen.clone(), stop: e.stop.clone() };
                s.spawn(move || e.run(pool));
            };
            run(&a);
            until("four", &|| a.threads() == 4);
            run(&b);
            until("two each", &|| (a.threads(), b.threads()) == (2, 2));
            let t0 = Instant::now();
            while t0.elapsed() < Duration::from_millis(300) {
                // (Fewer, if one is not run for a while.)
                let now = (a.threads(), b.threads());
                assert!(now.0 <= 2 && now.1 <= 2, "{now:?}");
                thread::sleep(Duration::from_millis(5));
            }
            until("two each, still", &|| (a.threads(), b.threads()) == (2, 2));
        });
        assert_eq!(searches(&pool), 0);
    }
}
