//! Greedy beam search over a proximity graph.
//!
//! One loop serves every caller: index build (distances between stored
//! vectors), exact search (query to full vectors) and PQ-guided search
//! (query to PQ codes). The caller supplies the distance and the neighbor
//! lists; the loop owns the candidate list and the visited set.

use std::cmp::Ordering;

use super::visited::VisitedSet;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Candidate {
    pub dist: f32,
    pub node: u32,
    expanded: bool,
}

#[inline]
fn order(a_dist: f32, a_node: u32, b_dist: f32, b_node: u32) -> Ordering {
    a_dist.total_cmp(&b_dist).then(a_node.cmp(&b_node))
}

/// Per-thread buffers reused across searches.
pub(crate) struct Scratch {
    pub visited: VisitedSet,
    /// Best `l` candidates found so far, sorted by (distance, node).
    pub list: Vec<Candidate>,
    /// Nodes in the order they were expanded, with their distances.
    pub expanded: Vec<(f32, u32)>,
    fresh: Vec<u32>,
}

impl Scratch {
    pub fn new() -> Self {
        Self { visited: VisitedSet::new(), list: Vec::new(), expanded: Vec::new(), fresh: Vec::new() }
    }
}

/// Counters for one search.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchStats {
    /// Distance computations used to steer the search (full-vector or PQ).
    pub distances: usize,
    /// Nodes whose neighbor lists were read. On disk, each one is a read.
    pub expanded: usize,
}

/// Runs the search from `entry` with a candidate list of size `l` (>= 1).
///
/// `dist(node)` is the distance from the query to `node`; `neighbors(node)`
/// is its out-list; `prefetch(node)` starts loading what `dist` will read. Every candidate left in `scratch.list` has been
/// expanded when this returns, and `scratch.expanded` holds them all in
/// expansion order.
pub(crate) fn beam_search<'g>(
    scratch: &mut Scratch,
    entry: u32,
    l: usize,
    mut dist: impl FnMut(u32) -> f32,
    neighbors: impl Fn(u32) -> &'g [u32],
    prefetch: impl Fn(u32),
) -> SearchStats {
    debug_assert!(l >= 1);
    let Scratch { visited, list, expanded, fresh } = scratch;
    visited.clear();
    list.clear();
    expanded.clear();
    visited.insert(entry);
    list.push(Candidate { dist: dist(entry), node: entry, expanded: false });
    let mut stats = SearchStats { distances: 1, expanded: 0 };

    let mut cursor = 0; // first unexpanded candidate
    while cursor < list.len() {
        let current = &mut list[cursor];
        current.expanded = true;
        let node = current.node;
        expanded.push((current.dist, node));
        stats.expanded += 1;

        fresh.clear();
        fresh.extend(neighbors(node).iter().copied().filter(|&nb| visited.insert(nb)));
        fresh.iter().for_each(|&nb| prefetch(nb));
        let mut next = cursor + 1;
        for &nb in fresh.iter() {
            let d = dist(nb);
            stats.distances += 1;
            if list.len() == l {
                let last = list[l - 1];
                if order(d, nb, last.dist, last.node) != Ordering::Less {
                    continue;
                }
                list.pop();
            }
            let pos = list.partition_point(|c| order(c.dist, c.node, d, nb) == Ordering::Less);
            list.insert(pos, Candidate { dist: d, node: nb, expanded: false });
            next = next.min(pos);
        }
        cursor = next;
        while cursor < list.len() && list[cursor].expanded {
            cursor += 1;
        }
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Path graph 0 - 1 - ... - (n-1) with distance |node - target|.
    #[test]
    fn walks_to_the_target_and_keeps_l_best() {
        let n = 50u32;
        let adj: Vec<Vec<u32>> = (0..n)
            .map(|i| [i.checked_sub(1), (i + 1 < n).then_some(i + 1)].into_iter().flatten().collect())
            .collect();
        let mut s = Scratch::new();
        let stats = beam_search(&mut s, 0, 3, |v| (v as f32 - 40.0).abs(), |v| &adj[v as usize], |_| ());
        let found: Vec<u32> = s.list.iter().map(|c| c.node).collect();
        assert_eq!(found, vec![40, 39, 41]);
        assert!(s.list.iter().all(|c| c.expanded));
        assert_eq!(stats.expanded, s.expanded.len());
        assert!(stats.expanded >= 41);
        assert_eq!(s.expanded[0], (40.0, 0));
    }

    #[test]
    fn list_of_size_n_visits_the_whole_component() {
        let adj: Vec<Vec<u32>> = vec![vec![1, 2], vec![0], vec![3], vec![2], vec![]];
        let mut s = Scratch::new();
        let stats = beam_search(&mut s, 0, 5, |v| v as f32, |v| &adj[v as usize], |_| ());
        let mut seen: Vec<u32> = s.list.iter().map(|c| c.node).collect();
        seen.sort();
        assert_eq!(seen, vec![0, 1, 2, 3], "node 4 is unreachable");
        assert_eq!(stats, SearchStats { distances: 4, expanded: 4 });
    }
}
