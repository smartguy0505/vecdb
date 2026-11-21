//! Bounded top-k selection (k smallest distances).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchResult {
    pub id: u64,
    pub distance: f32,
}

/// Heap entry ordered by (distance, id), so ties break toward the smaller id
/// and results are deterministic.
#[derive(Clone, Copy, Debug)]
struct Entry(SearchResult);

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .distance
            .total_cmp(&other.0.distance)
            .then(self.0.id.cmp(&other.0.id))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Entry {}

/// Keeps the `k` best (smallest-distance) candidates seen so far.
/// Max-heap: the root is the current worst kept candidate.
#[derive(Debug)]
pub struct TopK {
    k: usize,
    heap: BinaryHeap<Entry>,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        Self { k, heap: BinaryHeap::with_capacity(k) }
    }

    #[inline]
    pub fn push(&mut self, id: u64, distance: f32) {
        if self.k == 0 {
            return;
        }
        // NaN (possible only from overflow, e.g. inf - inf) ranks last;
        // `+ 0.0` turns -0.0 into 0.0 so equal distances tie on id.
        let distance = if distance.is_nan() { f32::INFINITY } else { distance + 0.0 };
        let entry = Entry(SearchResult { id, distance });
        if self.heap.len() < self.k {
            self.heap.push(entry);
        } else if let Some(mut worst) = self.heap.peek_mut() {
            if entry < *worst {
                *worst = entry; // PeekMut re-sifts on drop
            }
        }
    }

    /// Distance a candidate must beat to enter, once the heap is full.
    #[inline]
    pub fn threshold(&self) -> Option<f32> {
        if self.heap.len() == self.k { self.heap.peek().map(|e| e.0.distance) } else { None }
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Results sorted closest first.
    pub fn into_sorted(self) -> Vec<SearchResult> {
        self.heap.into_sorted_vec().into_iter().map(|e| e.0).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    fn reference(cands: &[(u64, f32)], k: usize) -> Vec<SearchResult> {
        let mut v: Vec<SearchResult> =
            cands.iter().map(|&(id, distance)| SearchResult { id, distance }).collect();
        v.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.id.cmp(&b.id)));
        v.truncate(k);
        v
    }

    #[test]
    fn matches_sort_reference() {
        let mut rng = SplitMix64::new(1);
        for trial in 0..200 {
            let n = rng.below(300) as usize;
            let k = rng.below(40) as usize;
            // Few distinct values -> many ties.
            let cands: Vec<(u64, f32)> =
                (0..n).map(|i| (i as u64, rng.below(10) as f32)).collect();
            let mut top = TopK::new(k);
            for &(id, d) in &cands {
                top.push(id, d);
            }
            assert_eq!(top.into_sorted(), reference(&cands, k), "trial {trial}");
        }
    }

    #[test]
    fn k_zero_and_k_larger_than_n() {
        let mut top = TopK::new(0);
        top.push(1, 1.0);
        assert!(top.into_sorted().is_empty());

        let mut top = TopK::new(10);
        top.push(5, 2.0);
        top.push(3, 1.0);
        let r = top.into_sorted();
        assert_eq!(r.iter().map(|r| r.id).collect::<Vec<_>>(), vec![3, 5]);
    }

    #[test]
    fn nan_ranks_last_and_negative_zero_ties_on_id() {
        let mut top = TopK::new(3);
        top.push(1, f32::NAN);
        top.push(9, 0.0);
        top.push(2, -0.0);
        let r = top.into_sorted();
        assert_eq!(r.iter().map(|r| r.id).collect::<Vec<_>>(), vec![2, 9, 1]);
        assert_eq!(r[2].distance, f32::INFINITY);
    }
}
