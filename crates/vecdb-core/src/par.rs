//! Minimal data parallelism over row ranges using scoped std threads.

use std::ops::Range;

/// Splits `0..n` into at most `threads` contiguous ranges and runs `f` on
/// each in its own scoped thread. Results come back in range order, so a
/// reduction over them is deterministic for a given thread count.
/// `threads == 0` means one thread per available CPU.
pub fn map_ranges<T, F>(n: usize, threads: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(Range<usize>) -> T + Sync,
{
    let threads = effective_threads(threads).min(n.max(1));
    if threads <= 1 {
        return vec![f(0..n)];
    }
    let per = n.div_ceil(threads);
    std::thread::scope(|s| {
        let f = &f;
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let range = (t * per).min(n)..((t + 1) * per).min(n);
                s.spawn(move || f(range))
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("worker thread panicked")).collect()
    })
}

/// Like [`map_ranges`] over the rows of `data` (`dim` values each), giving
/// each thread mutable access to its rows: `f(first_row, rows)`.
pub fn map_rows_mut<T, R, F>(data: &mut [T], dim: usize, threads: usize, f: F) -> Vec<R>
where
    T: Send,
    R: Send,
    F: Fn(usize, &mut [T]) -> R + Sync,
{
    let n = data.len().checked_div(dim).unwrap_or(0);
    let threads = effective_threads(threads).min(n.max(1));
    if threads <= 1 {
        return vec![f(0, data)];
    }
    let per = n.div_ceil(threads);
    std::thread::scope(|s| {
        let f = &f;
        let handles: Vec<_> = data
            .chunks_mut(per * dim)
            .enumerate()
            .map(|(t, rows)| s.spawn(move || f(t * per, rows)))
            .collect();
        handles.into_iter().map(|h| h.join().expect("worker thread panicked")).collect()
    })
}

pub fn effective_threads(threads: usize) -> usize {
    if threads == 0 {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    } else {
        threads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_every_index_once_in_order() {
        for n in [0usize, 1, 7, 100] {
            for threads in [1usize, 2, 3, 8, 200] {
                let parts = map_ranges(n, threads, |r| r.collect::<Vec<_>>());
                let all: Vec<usize> = parts.into_iter().flatten().collect();
                assert_eq!(all, (0..n).collect::<Vec<_>>(), "n {n} threads {threads}");
            }
        }
    }
}
