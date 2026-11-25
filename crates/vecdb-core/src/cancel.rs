//! Stopping long work early, sharing the CPUs between builds, and saying how
//! far one has come.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

use crate::error::{Error, Result};
use crate::par;

/// A flag that stops a build. Clones share it.
#[derive(Clone, Debug, Default)]
pub struct Cancel {
    stop: Arc<AtomicBool>,
    /// The builds that run now and share their threads, if any do.
    builds: Option<Arc<AtomicUsize>>,
    /// Who wants to know how far the work has come, and which part of it
    /// this is for.
    done: Option<(Arc<Done>, f32, f32)>,
}

/// How much of a piece of work is done, from 0 to 1, as far as it can tell.
#[derive(Debug, Default)]
pub(crate) struct Done(AtomicU32);

impl Done {
    pub fn get(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }

    pub fn set(&self, x: f32) {
        self.0.store(x.to_bits(), Ordering::Relaxed);
    }
}

impl Cancel {
    /// Asks the work to stop. It does so within a moment, with
    /// [`Error::Cancelled`].
    pub fn cancel(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    pub(crate) fn check(&self) -> Result<()> {
        if self.is_cancelled() { Err(Error::Cancelled) } else { Ok(()) }
    }

    /// For one of the `builds` that run now and share their threads.
    pub(crate) fn sharing(&self, builds: &Arc<AtomicUsize>) -> Self {
        Self { builds: Some(builds.clone()), ..self.clone() }
    }

    /// The threads for the next step of a build that may use `threads`
    /// (0 = one per CPU) while it runs alone. What a build makes does not
    /// depend on its threads, so their number may change along the way.
    pub(crate) fn threads(&self, threads: usize) -> usize {
        match self.builds.as_ref().map_or(1, |b| b.load(Ordering::Relaxed)) {
            0 | 1 => threads,
            n => (par::effective_threads(threads) / n).max(1),
        }
    }

    /// For work that says in `done` how far it has come.
    pub(crate) fn reporting(&self, done: &Arc<Done>) -> Self {
        Self { done: Some((done.clone(), 0.0, 1.0)), ..self.clone() }
    }

    /// For the step that takes this work from `from` to `to` (of 1).
    pub(crate) fn part(&self, from: f32, to: f32) -> Self {
        let at = |x: f32, (a, b): (f32, f32)| a + (b - a) * x;
        let done = self.done.as_ref().map(|(d, a, b)| (d.clone(), at(from, (*a, *b)), at(to, (*a, *b))));
        Self { done, ..self.clone() }
    }

    /// Says that `n` of `all` equal parts of this work are done.
    pub(crate) fn done(&self, n: usize, all: usize) {
        if let Some((done, from, to)) = &self.done {
            done.set(from + (to - from) * (n as f32 / all.max(1) as f32).min(1.0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parts_of_parts() {
        let done = Arc::new(Done::default());
        let work = Cancel::default().reporting(&done);
        assert_eq!(done.get(), 0.0);
        let half = work.part(0.5, 1.0);
        half.done(0, 4);
        assert_eq!(done.get(), 0.5);
        half.part(0.25, 0.75).done(1, 2);
        assert_eq!(done.get(), 0.75);
        half.part(0.25, 0.75).done(5, 2);
        assert_eq!(done.get(), 0.875);
        work.done(0, 0);
        assert_eq!(done.get(), 0.0);
        work.done(1, 1);
        assert_eq!(done.get(), 1.0);
        half.sharing(&Arc::default()).done(1, 4);
        assert_eq!(done.get(), 0.625);
        // The same work otherwise; and nobody has to want to know.
        work.part(0.0, 0.1).sharing(&Arc::default()).cancel();
        assert!(work.is_cancelled() && half.check().is_err());
        Cancel::default().part(0.0, 0.5).done(1, 2);
    }
}
