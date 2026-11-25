//! How many requests are worked on at a time, and how many may wait.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Lets a number of requests in at a time. More wait in line, up to a number
/// of them, and not for long: a request that cannot be answered soon is
/// better refused, so that its client can try elsewhere or later.
pub struct Gate {
    places: Arc<Semaphore>,
    size: usize,
    waiting: AtomicUsize,
    max_waiting: usize,
    timeout: Duration,
}

/// There is no place, and none to wait in, or none came free in time.
#[derive(Debug, PartialEq)]
pub struct Full;

/// One that waits (it may give up).
struct Waiting<'a>(&'a AtomicUsize);

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Gate {
    pub fn new(size: usize, max_waiting: usize, timeout: Duration) -> Self {
        let size = size.clamp(1, Semaphore::MAX_PERMITS.min(u32::MAX as usize));
        Self { places: Arc::new(Semaphore::new(size)), size, waiting: AtomicUsize::new(0), max_waiting, timeout }
    }

    /// Waits for a place, which is held until the result is dropped.
    pub async fn enter(&self) -> Result<OwnedSemaphorePermit, Full> {
        self.enter_many(1).await
    }

    /// The same for `n` places at once, or all there are.
    pub async fn enter_many(&self, n: usize) -> Result<OwnedSemaphorePermit, Full> {
        if let Ok(p) = self.try_enter_many(n) {
            return Ok(p);
        }
        if self.waiting.fetch_add(1, Ordering::Relaxed) >= self.max_waiting {
            self.waiting.fetch_sub(1, Ordering::Relaxed);
            return Err(Full);
        }
        let _waiting = Waiting(&self.waiting);
        let n = n.min(self.size) as u32;
        match tokio::time::timeout(self.timeout, self.places.clone().acquire_many_owned(n)).await {
            Ok(Ok(p)) => Ok(p),
            _ => Err(Full),
        }
    }

    /// The same without waiting.
    pub fn try_enter_many(&self, n: usize) -> Result<OwnedSemaphorePermit, Full> {
        self.places.clone().try_acquire_many_owned(n.min(self.size) as u32).map_err(|_| Full)
    }

    /// Places taken, and requests waiting.
    pub fn load(&self) -> (usize, usize) {
        (self.size - self.places.available_permits(), self.waiting.load(Ordering::Relaxed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn lets_in_queues_and_refuses() {
        let gate = Arc::new(Gate::new(2, 1, Duration::from_millis(200)));
        let a = gate.enter().await.unwrap();
        let b = gate.enter().await.unwrap();
        assert_eq!(gate.load(), (2, 0));
        // One may wait, a second not.
        let g = gate.clone();
        let c = tokio::spawn(async move { g.enter().await });
        while gate.load().1 == 0 {
            tokio::task::yield_now().await;
        }
        assert_eq!(gate.enter().await.unwrap_err(), Full);
        drop(a);
        let c = c.await.unwrap().unwrap();
        assert_eq!(gate.load(), (2, 0));
        // Not for longer than that.
        assert_eq!(gate.enter().await.unwrap_err(), Full);
        assert_eq!(gate.load(), (2, 0));
        // One that gives up leaves the line.
        let g = gate.clone();
        let d = tokio::spawn(async move { g.enter().await });
        while gate.load().1 == 0 {
            tokio::task::yield_now().await;
        }
        d.abort();
        let _ = d.await;
        assert_eq!(gate.load(), (2, 0));
        drop((b, c));
        assert_eq!(gate.load(), (0, 0));
        // More than there are is all there are.
        let all = gate.enter_many(100).await.unwrap();
        assert_eq!(gate.load(), (2, 0));
        drop(all);
    }
}
