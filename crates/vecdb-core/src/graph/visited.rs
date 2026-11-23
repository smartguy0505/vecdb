//! Visited-node set for graph search: open addressing over `u32` node ids,
//! with a generation tag per slot so clearing between queries is O(1).
//!
//! Its size follows the number of nodes a query touches (thousands), not the
//! collection size, so one set per thread stays small at any scale.

pub(crate) struct VisitedSet {
    /// Each slot is `(generation << 32) | node`; a slot is free when its
    /// generation is not the current one.
    slots: Vec<u64>,
    generation: u32,
    len: usize,
    /// log2 of `slots.len()`.
    bits: u32,
}

const MIN_BITS: u32 = 10;

impl VisitedSet {
    pub fn new() -> Self {
        Self { slots: vec![0; 1 << MIN_BITS], generation: 1, len: 0, bits: MIN_BITS }
    }

    /// Forgets every node.
    pub fn clear(&mut self) {
        self.len = 0;
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            // Wrapped: old tags could look current again.
            self.slots.fill(0);
            self.generation = 1;
        }
    }

    #[inline]
    fn slot_of(node: u32, bits: u32) -> usize {
        // Fibonacci hashing: the top bits of a multiplicative hash.
        (node.wrapping_mul(0x9E37_79B1) >> (32 - bits)) as usize
    }

    /// Inserts `node`; returns true if it was not in the set.
    #[inline]
    pub fn insert(&mut self, node: u32) -> bool {
        if (self.len + 1) * 2 > self.slots.len() {
            self.grow();
        }
        let tag = (self.generation as u64) << 32;
        let mask = self.slots.len() - 1;
        let mut i = Self::slot_of(node, self.bits);
        loop {
            let s = self.slots[i];
            if s >> 32 != self.generation as u64 {
                self.slots[i] = tag | node as u64;
                self.len += 1;
                return true;
            }
            if s as u32 == node {
                return false;
            }
            i = (i + 1) & mask;
        }
    }

    fn grow(&mut self) {
        let old = std::mem::replace(&mut self.slots, vec![0; 1 << (self.bits + 1)]);
        self.bits += 1;
        let (generation, len) = (self.generation, self.len);
        self.generation = 1;
        self.len = 0;
        for s in old {
            if s >> 32 == generation as u64 {
                self.insert(s as u32);
            }
        }
        debug_assert_eq!(self.len, len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;
    use std::collections::HashSet;

    #[test]
    fn matches_hash_set() {
        let mut rng = SplitMix64::new(4);
        let mut v = VisitedSet::new();
        for round in 0..20 {
            v.clear();
            let mut reference = HashSet::new();
            let n = 1 + round * 400;
            for _ in 0..n {
                let x = rng.below(3 * n as u64) as u32;
                assert_eq!(v.insert(x), reference.insert(x));
            }
            assert_eq!(v.len, reference.len());
        }
        v.clear();
        assert!(v.insert(u32::MAX) && !v.insert(u32::MAX) && v.insert(0));
    }

    #[test]
    fn generation_wrap_clears() {
        let mut v = VisitedSet::new();
        v.insert(7);
        v.generation = u32::MAX;
        v.insert(8);
        v.clear();
        assert!(v.insert(8) && v.insert(7));
    }
}
