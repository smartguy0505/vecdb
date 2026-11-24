//! Stopping long work early.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::{Error, Result};

/// A flag that stops a build. Clones share it.
#[derive(Clone, Debug, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    /// Asks the work to stop. It does so within a moment, with
    /// [`Error::Cancelled`].
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    pub(crate) fn check(&self) -> Result<()> {
        if self.is_cancelled() { Err(Error::Cancelled) } else { Ok(()) }
    }
}
