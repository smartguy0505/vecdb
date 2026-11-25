//! Stopping long work early, and sharing the CPUs between builds.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::error::{Error, Result};
use crate::par;

/// A flag that stops a build. Clones share it.
#[derive(Clone, Debug, Default)]
pub struct Cancel {
    stop: Arc<AtomicBool>,
    /// The builds that run now and share their threads, if any do.
    builds: Option<Arc<AtomicUsize>>,
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
        Self { stop: self.stop.clone(), builds: Some(builds.clone()) }
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
}
