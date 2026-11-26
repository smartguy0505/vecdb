//! Slowing writes down to what index builds keep up with, before they have
//! to stop.
//!
//! What counts is the vectors that are in no SSD index yet: in RAM, and
//! scanned by every search. A build gives their room back when it is done,
//! which can take minutes, and more if it has to wait for others first. To
//! have writes wait for that would stop them for as long, and let them run at
//! full speed in between. So the share of the way to the end of the next
//! build that it has come (through the line of builds, and its own work)
//! counts as gone from its input already
//! ([`Engine::unindexed`](super::Engine::unindexed)), and what is left of
//! the room ([`Zone`]) sets the rate of writes: each 16th of the way to the
//! limit halves it. Wherever the rate of the builds is, that of the writes
//! settles there. No estimate of either is needed.
//!
//! Writes go in the order they come ([`Line`]), each once the ones before it
//! are paid for ([`Bucket`]): a write that comes after a pause does not wait,
//! however large it is, and those after it do. One that has waited for as long
//! as it may gives up, and its place is kept for a while: it has that again
//! if it comes again. Or those at the end of a long line would be late time
//! and again, as everyone who is let through lines up before they are back.
//! It is known by what is in it: whoever sends it has nothing to remember.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

use super::CollectionConfig;

/// Bytes per second where writes begin to be slowed down: about as fast as a
/// collection can take them.
const FASTEST: f64 = (256u64 << 20) as f64;
/// How often that is halved on the way to the limit: down to 4 KiB a second.
/// A steeper curve has the rate swing more with the steps that builds take; a
/// flatter one does not get down to what is left for each of many shards.
const HALVINGS: f64 = 16.0;
/// The pause after which the first in the line looks again: builds do not
/// tell how far they are.
const LOOK: Duration = Duration::from_millis(10);
/// What was not used of the rate lately still may be, for this long: writes
/// wake up later than they are due.
const CREDIT: Duration = Duration::from_millis(2);

/// How long the place of a write that has given up is kept, and of how many:
/// an order to go in is all they take.
const KEEP: Duration = Duration::from_secs(60);
const MOST_KEPT: usize = 1024;

/// The bytes of vectors in no SSD index at which writes are slowed down, and
/// stopped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Zone {
    pub from: f64,
    pub to: f64,
}

impl Zone {
    /// The last quarter before `limit`. While a build runs, the input of the
    /// next one has to come together, or the builds would stand idle, and
    /// something has to be left beyond that: there is always room for
    /// `index_bytes` and a segment, and a quarter of it (or another segment).
    pub fn new(limit: u64, cfg: &CollectionConfig) -> Self {
        let (index, seal) = (cfg.index_bytes, cfg.seal_bytes);
        let spare = (index / 4).max(seal.saturating_mul(2));
        let to = limit.max(index.saturating_add(spare));
        let from = (to - to / 4).max(index.saturating_add(seal));
        Self { from: from as f64, to: to as f64 }
    }

    /// Bytes per second that may be added to `bytes`; `None` if any number.
    pub fn rate(&self, bytes: f64) -> Option<f64> {
        if bytes <= self.from {
            None
        } else if bytes >= self.to {
            Some(0.0)
        } else {
            Some(FASTEST * (-HALVINGS * (bytes - self.from) / (self.to - self.from)).exp2())
        }
    }
}

/// What was let through and is not paid for yet, in bytes.
#[derive(Default)]
struct Bucket {
    owed: f64,
    paid_at: Option<Instant>,
}

impl Bucket {
    /// Pays at `rate` up to `now`; how long until all is paid, at this rate.
    fn pay(&mut self, now: Instant, rate: f64) -> Duration {
        let paid_at = self.paid_at.unwrap_or(now);
        let time = now.saturating_duration_since(paid_at);
        self.paid_at = Some(now.max(paid_at));
        self.owed = (self.owed - rate * time.as_secs_f64()).max(-rate * CREDIT.as_secs_f64());
        if rate <= 0.0 {
            return Duration::MAX;
        }
        Duration::try_from_secs_f64(self.owed.max(0.0) / rate).unwrap_or(Duration::MAX)
    }
}

#[derive(Default)]
struct State {
    /// Tickets of those that wait, in order.
    line: VecDeque<(u64, Thread)>,
    /// The places of those that have given up, the last one last.
    kept: VecDeque<Kept>,
    next: u64,
    bucket: Bucket,
}

struct Kept {
    key: u64,
    ticket: u64,
    until: Instant,
}

impl State {
    fn keep(&mut self, key: u64, ticket: u64) {
        let now = Instant::now();
        while self.kept.len() >= MOST_KEPT || self.kept.front().is_some_and(|k| k.until <= now) {
            self.kept.pop_front();
        }
        self.kept.push_back(Kept { key, ticket, until: now + KEEP });
    }

    /// Lines up: where it was, or at the end.
    fn line_up(&mut self, key: u64) -> u64 {
        let now = Instant::now();
        let kept = self.kept.iter().position(|k| k.key == key && k.until > now);
        let ticket = match kept.and_then(|i| self.kept.remove(i)) {
            Some(k) => k.ticket,
            None => {
                self.next += 1;
                self.next - 1
            }
        };
        let at = self.line.partition_point(|(t, _)| *t < ticket);
        self.line.insert(at, (ticket, thread::current()));
        ticket
    }
}

/// The line writes wait in.
#[derive(Default)]
pub(super) struct Line {
    state: Mutex<State>,
    /// There has been no rate to keep to: what is owed need not be paid.
    free: AtomicBool,
}

impl Line {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The tickets of those that wait, and the places that are kept, of which
    /// writes.
    #[cfg(test)]
    pub fn places(&self) -> (Vec<u64>, Vec<(u64, u64)>) {
        let s = self.lock();
        (s.line.iter().map(|(t, _)| *t).collect(), s.kept.iter().map(|k| (k.key, k.ticket)).collect())
    }

    #[cfg(test)]
    pub fn owed(&self) -> f64 {
        self.lock().bucket.owed
    }

    /// Waits for its turn to add `bytes` (which are then owed), at the
    /// `rate` there is along the way. How long it waited; an error if its
    /// turn had not come by `deadline`. It is known by `key` if it comes
    /// again then.
    pub fn wait(
        &self,
        bytes: u64,
        key: impl FnOnce() -> u64,
        deadline: Option<Instant>,
        rate: impl Fn() -> Option<f64>,
    ) -> Result<(Duration, Pass<'_>), Duration> {
        if rate().is_none() {
            if !self.free.load(Ordering::Relaxed) {
                self.free.store(true, Ordering::Relaxed);
            }
            return Ok((Duration::ZERO, Pass(None)));
        }
        let start = Instant::now();
        let mut place = Place::new(self, key());
        let mut waited = Duration::ZERO;
        loop {
            let rate = rate();
            let mut s = self.lock();
            let now = Instant::now();
            let first = s.line.front().is_some_and(|(t, _)| *t == place.ticket);
            if rate.is_none() || self.free.swap(false, Ordering::Relaxed) {
                s.bucket = Bucket::default();
            }
            let left = match rate {
                None => return Ok((waited, Pass(None))),
                Some(r) if first => s.bucket.pay(now, r),
                Some(_) => Duration::MAX,
            };
            if left.is_zero() {
                s.bucket.owed += bytes as f64;
                let (line, key, ticket) = (self, place.key, place.ticket);
                return Ok((waited, Pass(Some(Owes { line, key, ticket, bytes }))));
            }
            drop(s);
            let until = deadline.map_or(Duration::MAX, |d| d.saturating_duration_since(now));
            if until.is_zero() {
                place.keep = true;
                return Err(waited);
            }
            // (The others are woken when they are the first.)
            let look = if first { LOOK } else { Duration::from_secs(1) };
            thread::park_timeout(left.min(until).min(look));
            waited = start.elapsed();
        }
    }
}

/// A write's place in the line.
struct Place<'a> {
    line: &'a Line,
    key: u64,
    ticket: u64,
    /// Whether it is kept when this is dropped.
    keep: bool,
}

impl<'a> Place<'a> {
    fn new(line: &'a Line, key: u64) -> Self {
        let ticket = line.lock().line_up(key);
        Self { line, key, ticket, keep: false }
    }
}

impl Drop for Place<'_> {
    fn drop(&mut self) {
        let mut s = self.line.lock();
        s.line.retain(|(t, _)| *t != self.ticket);
        if self.keep {
            s.keep(self.key, self.ticket);
        }
        if let Some((_, first)) = s.line.front() {
            first.unpark();
        }
    }
}

/// That a write may go ahead, and what it owes for it. If it does not after
/// all (this is dropped, not [`used`](Self::used)), it owes nothing, and its
/// place is kept.
#[must_use]
pub(super) struct Pass<'a>(Option<Owes<'a>>);

struct Owes<'a> {
    line: &'a Line,
    key: u64,
    ticket: u64,
    bytes: u64,
}

impl Pass<'_> {
    /// For a write that takes no room.
    pub fn free() -> Self {
        Self(None)
    }

    pub fn used(mut self) {
        self.0 = None;
    }
}

impl Drop for Pass<'_> {
    fn drop(&mut self) {
        if let Some(Owes { line, key, ticket, bytes }) = self.0.take() {
            let mut s = line.lock();
            s.bucket.owed = (s.bucket.owed - bytes as f64).max(0.0);
            s.keep(key, ticket);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, AtomicUsize};

    use super::*;
    use crate::distance::Metric;

    const MIB: f64 = (1u64 << 20) as f64;

    fn zone(limit_mb: u64, index_mb: u64, seal_mb: u64) -> (f64, f64) {
        let mut cfg = CollectionConfig::new(128, Metric::L2);
        cfg.index_bytes = index_mb << 20;
        cfg.seal_bytes = seal_mb << 20;
        let z = Zone::new(limit_mb << 20, &cfg);
        (z.from / MIB, z.to / MIB)
    }

    #[test]
    fn zones() {
        // The defaults, and the smallest limit that they leave as it is.
        assert_eq!(zone(1024, 64, 32), (768.0, 1024.0));
        assert_eq!(zone(128, 64, 32), (96.0, 128.0));
        // Below that, what index builds need.
        assert_eq!(zone(1, 64, 32), (96.0, 128.0));
        assert_eq!(zone(256, 256, 32), (288.0, 320.0));
        assert_eq!(zone(128, 1024, 32), (1056.0, 1280.0));
        // A start that depends on them, and one that does not.
        assert_eq!(zone(160, 100, 32), (132.0, 164.0));
        assert_eq!(zone(200, 100, 32), (150.0, 200.0));
        // Never, rather than wrapped around.
        let mut cfg = CollectionConfig::new(128, Metric::L2);
        cfg.index_bytes = u64::MAX;
        assert_eq!(Zone::new(1 << 20, &cfg).rate(1e18), None);
        cfg.index_bytes = u64::MAX / 2;
        cfg.seal_bytes = u64::MAX;
        assert_eq!(Zone::new(u64::MAX, &cfg).rate(1e18), None);
    }

    #[test]
    fn rates() {
        let z = Zone { from: 96.0 * MIB, to: 128.0 * MIB };
        let at = |mb: f64| z.rate(mb * MIB).map(|r| r / 1024.0);
        assert_eq!((at(0.0), at(96.0)), (None, None));
        // Halved by every 16th of the way, from 256 MiB to 4 KiB a second.
        for i in 1..16 {
            let want = (256 << 10) as f64 / (1u64 << i) as f64;
            assert!((at(96.0 + 2.0 * i as f64).unwrap() / want - 1.0).abs() < 1e-9, "{i}");
        }
        assert!((at(96.0 + 1e-6).unwrap() / 262144.0 - 1.0).abs() < 1e-3);
        assert!((at(128.0 - 1e-6).unwrap() / 4.0 - 1.0).abs() < 1e-3);
        assert_eq!((at(128.0), at(1e9)), (Some(0.0), Some(0.0)));
    }

    #[test]
    fn buckets() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let ms = Duration::from_millis;
        let mut b = Bucket::default();
        // The first write goes at once, whatever its size, and has to be paid for.
        assert_eq!(b.pay(at(0), 1000.0), ms(0));
        b.owed += 500.0;
        assert_eq!(b.pay(at(0), 1000.0), ms(500));
        assert_eq!(b.pay(at(100), 1000.0), ms(400));
        // At the rate there is now.
        assert_eq!(b.pay(at(200), 2000.0), ms(100));
        assert_eq!(b.pay(at(250), 1000.0), ms(150));
        // Nothing is paid, and nothing goes, at no rate.
        assert_eq!(b.pay(at(5000), 0.0), Duration::MAX);
        assert_eq!(b.pay(at(5050), 1000.0), ms(100));
        // After a pause, 2 ms may be made up for.
        assert_eq!(b.pay(at(9000), 1000.0), ms(0));
        b.owed += 3.0;
        assert_eq!(b.pay(at(9000), 1000.0), ms(1));
        // Not at a rate there is no more.
        assert_eq!(b.pay(at(20_000), 1e9), ms(0));
        assert_eq!(b.pay(at(20_000), 1000.0), ms(0));
        b.owed += 12.0;
        assert_eq!(b.pay(at(20_000), 1000.0), ms(10));
        // Nor does what is owed not go, however little there is.
        b.owed = -5.0;
        assert_eq!(b.pay(at(20_001), 0.0), Duration::MAX);
        // A time that has passed is not paid for twice.
        b.owed = 7.0;
        assert_eq!(b.pay(at(19_000), 1000.0), ms(7));
        assert_eq!(b.pay(at(20_001), 1000.0), ms(7));
        // More than a duration holds.
        b.owed = 1e30;
        assert_eq!(b.pay(at(20_001), 1e-9), Duration::MAX);
    }

    /// A rate that tests change, in bytes per second; `NONE`: any number.
    #[derive(Default)]
    struct Rate(AtomicU64);

    const NONE: u64 = u64::MAX;

    impl Rate {
        fn get(&self) -> Option<f64> {
            match self.0.load(Ordering::Relaxed) {
                NONE => None,
                r => Some(r as f64),
            }
        }

        fn set(&self, r: u64) {
            self.0.store(r, Ordering::Relaxed);
        }
    }

    fn until(what: &str, done: impl Fn() -> bool) {
        let t0 = Instant::now();
        while !done() {
            assert!(t0.elapsed() < Duration::from_secs(30), "waited too long: {what}");
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// The write goes ahead. How long it has waited.
    fn paid((time, pass): (Duration, Pass)) -> Duration {
        pass.used();
        time
    }

    fn waiting(line: &Line) -> usize {
        line.lock().line.len()
    }

    /// In the order they came, and at the rate: not before it allows.
    #[test]
    fn writes_go_in_order() {
        let (line, rate) = (Arc::new(Line::default()), Arc::new(Rate::default()));
        let order = Arc::new(Mutex::new(Vec::new()));
        let threads: Vec<_> = (0..8)
            .map(|i| {
                let (line, rate, order) = (line.clone(), rate.clone(), order.clone());
                until("the one before", || waiting(&line) == i);
                thread::spawn(move || {
                    let waited = line.wait(1000, || i as u64, None, || rate.get()).map(paid).unwrap();
                    order.lock().unwrap().push((i, Instant::now(), waited));
                })
            })
            .collect();
        until("all", || waiting(&line) == 8);
        thread::sleep(Duration::from_millis(50));
        assert!(order.lock().unwrap().is_empty());
        // 20 ms for each but the first.
        rate.set(50_000);
        threads.into_iter().for_each(|t| t.join().unwrap());
        let order = order.lock().unwrap();
        assert!(order.iter().map(|o| o.0).eq(0..8), "{order:?}");
        let took = order[7].1 - order[0].1;
        assert!(took >= Duration::from_millis(7 * 20 - 5), "{took:?}");
        assert!(order.iter().all(|o| o.2 >= Duration::from_millis(50)), "{order:?}");
        assert_eq!(waiting(&line), 0);
        // The last is still to be paid for.
        let owed = line.owed();
        assert!((0.0..=1000.0).contains(&owed), "{owed}");
    }

    /// Also one that comes when the first in the line could go, but has not
    /// woken up yet.
    #[test]
    fn none_overtakes() {
        let line = Line::default();
        let (asleep, wake) = (Mutex::new(()), AtomicUsize::new(0));
        thread::scope(|s| {
            let held = asleep.lock().unwrap();
            let first = s.spawn(|| {
                line.wait(1, || 1, None, || match wake.fetch_add(1, Ordering::Relaxed) {
                    0 | 1 => Some(0.0),
                    _ => {
                        drop(asleep.lock().unwrap());
                        Some(1e6)
                    }
                })
            });
            until("the first to sleep", || wake.load(Ordering::Relaxed) > 2);
            let soon = Instant::now() + Duration::from_millis(50);
            assert!(line.wait(1, || 2, Some(soon), || Some(1e6)).is_err());
            assert!(!first.is_finished());
            drop(held);
            first.join().unwrap().map(paid).unwrap();
        });
        assert!(line.wait(1, || 3, Some(Instant::now() + Duration::from_secs(5)), || Some(1e6)).map(paid).is_ok());
    }

    /// Those that are late go, also from the front and the middle, and have
    /// not been let through; the others stay in order.
    #[test]
    fn late_writes_leave_the_line() {
        let (line, rate) = (Line::default(), Rate::default());
        let through = AtomicUsize::new(0);
        thread::scope(|s| {
            let write = |ms: Option<u64>| {
                let before = waiting(&line);
                let (line, rate, through) = (&line, &rate, &through);
                let h = s.spawn(move || {
                    let t0 = Instant::now();
                    let deadline = ms.map(|ms| t0 + Duration::from_millis(ms));
                    let r = line.wait(100, || before as u64, deadline, || rate.get()).map(paid);
                    through.fetch_add(r.is_ok() as usize, Ordering::Relaxed);
                    (r, t0.elapsed())
                });
                until("it to wait", || waiting(line) == before + 1);
                h
            };
            let late = [write(Some(250)), write(None), write(Some(400)), write(None)];
            for (i, h) in late.into_iter().enumerate() {
                let ms = match i {
                    0 => 250,
                    2 => 400,
                    _ => continue,
                };
                let (r, took) = h.join().unwrap();
                let ms = Duration::from_millis(ms);
                assert!((ms..ms + Duration::from_millis(300)).contains(&took), "{took:?}");
                let waited = r.unwrap_err();
                assert!(waited <= took && waited + Duration::from_millis(50) > ms, "{waited:?} of {took:?}");
            }
            assert_eq!((waiting(&line), through.load(Ordering::Relaxed)), (2, 0));
            assert!(line.owed() <= 0.0);
            rate.set(1000);
        });
        assert_eq!((waiting(&line), through.load(Ordering::Relaxed)), (0, 2));
    }

    /// One that has given up is in nobody's way, and has its place again when
    /// it comes again, however often.
    #[test]
    fn a_place_is_kept() {
        let (line, rate) = (Line::default(), Rate::default());
        let order = Mutex::new(Vec::new());
        // Gives up at once.
        let late = |key: u64| assert!(line.wait(1000, || key, Some(Instant::now()), || rate.get()).is_err());
        (1..5).for_each(late);
        assert_eq!(line.places().1, [(1, 0), (2, 1), (3, 2), (4, 3)]);
        thread::scope(|s| {
            let write = |key: u64| {
                let before = waiting(&line);
                let (line, rate, order) = (&line, &rate, &order);
                s.spawn(move || {
                    let soon = Some(Instant::now() + Duration::from_secs(30));
                    let (_, pass) = line.wait(1000, || key, soon, || rate.get()).unwrap();
                    order.lock().unwrap().push(key);
                    pass.used();
                });
                until("it to wait", || waiting(line) == before + 1);
            };
            // A new one, one from the middle, and the first one, twice.
            write(5);
            write(3);
            late(1);
            write(1);
            // Another new one; and one that is known, but has no place.
            write(6);
            write(3);
            assert_eq!(line.places(), (vec![0, 2, 4, 5, 6], vec![(2, 1), (4, 3)]));
            // (A write every 100 ms.)
            rate.set(10_000);
            until("all", || waiting(&line) == 0);
        });
        assert_eq!(*order.lock().unwrap(), [1, 3, 5, 6, 3]);
        assert_eq!(line.places().1, [(2, 1), (4, 3)]);
    }

    /// Nor are there more than so many.
    #[test]
    fn places_are_not_kept_for_ever() {
        let mut s = State { next: 1 << 20, ..State::default() };
        for i in 0..MOST_KEPT as u64 + 10 {
            s.keep(1000 + i, i);
        }
        assert_eq!((s.kept.len(), s.kept[0].ticket), (MOST_KEPT, 10));
        assert_eq!(s.line_up(1009), 1 << 20);
        assert_eq!(s.line_up(1012), 12);
        // Once.
        assert_eq!(s.line_up(1012), (1 << 20) + 1);
        // Of two that are the same, the one that came first.
        s.keep(1, 5000);
        s.keep(1, 4000);
        assert_eq!((s.line_up(1), s.line_up(1), s.line_up(1)), (5000, 4000, (1 << 20) + 2));
        assert!(s.line.iter().map(|(t, _)| *t).is_sorted() && s.line.len() == 6);
        // Not after a while; and then they are removed.
        let n = s.kept.len();
        assert!(s.kept.iter().all(|k| k.until > Instant::now() + KEEP / 2 && k.until <= Instant::now() + KEEP));
        s.kept.iter_mut().for_each(|k| k.until = Instant::now());
        assert_eq!(s.line_up(1015), (1 << 20) + 3);
        assert_eq!(s.kept.len(), n);
        s.keep(7, 7);
        assert_eq!(s.kept.len(), 1);
    }

    /// One that was let through and has not gone owes nothing, and has its
    /// place.
    #[test]
    fn a_pass_that_is_not_used() {
        let line = &Line::default();
        let go = |key: u64| line.wait(1 << 30, || key, Some(Instant::now()), || Some(1000.0));
        let (_, pass) = go(1).unwrap();
        assert!(go(2).is_err());
        assert!(line.owed() > 1e9);
        drop(pass);
        assert_eq!(line.owed(), 0.0);
        assert_eq!(line.places().1, [(2, 1), (1, 0)]);
        let (_, pass) = go(3).unwrap();
        thread::scope(|s| {
            // Those that wait for it go.
            let soon = Some(Instant::now() + Duration::from_secs(20));
            let h = s.spawn(move || line.wait(1, || 2, soon, || Some(1000.0)).map(paid).unwrap());
            until("it to wait", || waiting(line) == 1);
            drop(pass);
            h.join().unwrap();
        });
        // One that is used is not given back, one that had nothing to pay
        // for has nothing to give back.
        let (_, pass) = line.wait(1 << 30, || 9, None, || Some(1000.0)).unwrap();
        pass.used();
        let owed = line.owed();
        assert!(owed > 1e9 && !line.places().1.iter().any(|k| k.0 == 9));
        let key = || panic!("not asked for");
        drop(line.wait(1 << 30, key, None, || None).unwrap());
        drop(Pass::free());
        assert_eq!(line.places().1, [(1, 0), (3, 2)]);
    }

    /// They need not wait for what is owed, nor for those before them, once
    /// there is no rate to keep to; soon.
    #[test]
    fn no_rate_lets_all_through() {
        let (line, rate) = (Line::default(), Rate::default());
        let hour = 3_600_000;
        let late = || Some(Instant::now());
        rate.set(1000);
        assert_eq!(line.wait(hour, || 1, None, || rate.get()).map(paid).unwrap(), Duration::ZERO);
        thread::scope(|s| {
            let write = || line.wait(1, || 2, None, || rate.get()).map(paid).unwrap();
            let all: Vec<_> = (0..32).map(|_| s.spawn(write)).collect();
            until("all", || waiting(&line) == 32);
            let t0 = Instant::now();
            rate.set(NONE);
            let waited: Vec<Duration> = all.into_iter().map(|h| h.join().unwrap()).collect();
            assert!(t0.elapsed() < Duration::from_millis(900), "{:?}", t0.elapsed());
            assert!(waited.iter().all(|w| !w.is_zero()));
        });
        assert_eq!(waiting(&line), 0);
        // What is owed is forgotten, then: here by those that waited, ...
        assert_eq!(line.owed(), 0.0);
        // ... here by one that does not come to the line.
        rate.set(1000);
        assert_eq!(line.wait(hour, || 1, late(), || rate.get()).map(paid), Ok(Duration::ZERO));
        assert!(line.wait(1, || 3, late(), || rate.get()).map(paid).is_err());
        rate.set(NONE);
        assert_eq!(line.wait(1, || 3, late(), || rate.get()).map(paid), Ok(Duration::ZERO));
        assert_eq!(line.lock().next, 35);
        rate.set(1000);
        assert_eq!(line.wait(hour, || 1, late(), || rate.get()).map(paid), Ok(Duration::ZERO));
        assert!(line.wait(1, || 3, late(), || rate.get()).map(paid).is_err());
    }

    /// Many small writes from many threads get the rate, whatever they take
    /// to wake up.
    #[test]
    fn the_rate_is_kept() {
        let line = Line::default();
        let bytes = AtomicUsize::new(0);
        let t0 = Instant::now();
        // 2 MB a second, 100 bytes at a time: one every 50 us.
        thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    while t0.elapsed() < Duration::from_millis(500) {
                        line.wait(100, || 1, None, || Some(2e6)).map(paid).unwrap();
                        bytes.fetch_add(100, Ordering::Relaxed);
                    }
                });
            }
        });
        let most = 2e6 * (t0.elapsed() + CREDIT).as_secs_f64() + 400.0;
        let got = bytes.load(Ordering::Relaxed) as f64;
        assert!(got <= most, "{got} of {most}");
        // (How close it comes depends on what else the machine does.)
        assert!(got >= most / 20.0, "{got} of {most}");
    }

    /// One whose thread panics leaves the line to the next.
    #[test]
    fn a_panic_does_not_block_the_line() {
        let line = Line::default();
        let calls = AtomicUsize::new(0);
        thread::scope(|s| {
            let bad = s.spawn(|| {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    line.wait(1, || 1, None, || match calls.fetch_add(1, Ordering::Relaxed) {
                        0..20 => Some(0.0),
                        _ => panic!("on purpose"),
                    })
                    .map(paid)
                }));
                assert!(r.is_err());
            });
            until("the first", || waiting(&line) == 1);
            line.wait(1, || 2, None, || Some(1e6)).map(paid).unwrap();
            bad.join().unwrap();
        });
        assert_eq!(waiting(&line), 0);
    }
}
