//! A fixed-size bit set over a segment's positions: what a filter matches.

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Bitmap {
    words: Vec<u64>,
    len: usize,
}

impl Bitmap {
    pub fn empty(len: usize) -> Self {
        Self { words: vec![0; len.div_ceil(64)], len }
    }

    pub fn full(len: usize) -> Self {
        let mut b = Self { words: vec![!0; len.div_ceil(64)], len };
        b.mask_tail();
        b
    }

    #[inline]
    pub fn get(&self, i: usize) -> bool {
        self.words[i / 64] >> (i % 64) & 1 != 0
    }

    #[inline]
    pub fn set(&mut self, i: usize) {
        self.words[i / 64] |= 1 << (i % 64);
    }

    #[inline]
    pub fn clear(&mut self, i: usize) {
        self.words[i / 64] &= !(1 << (i % 64));
    }

    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn and(&mut self, other: &Bitmap) {
        debug_assert_eq!(self.len, other.len);
        self.words.iter_mut().zip(&other.words).for_each(|(a, b)| *a &= b);
    }

    pub fn or(&mut self, other: &Bitmap) {
        debug_assert_eq!(self.len, other.len);
        self.words.iter_mut().zip(&other.words).for_each(|(a, b)| *a |= b);
    }

    pub fn not(&mut self) {
        self.words.iter_mut().for_each(|w| *w = !*w);
        self.mask_tail();
    }

    /// Clears every bit that is set in `words` (same layout: bit `i` of
    /// word `i / 64`).
    pub fn and_not_words(&mut self, words: impl Iterator<Item = u64>) {
        self.words.iter_mut().zip(words).for_each(|(a, b)| *a &= !b);
    }

    /// Set bits, ascending.
    pub fn ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(i, &w)| {
            let mut w = w;
            std::iter::from_fn(move || {
                if w == 0 {
                    return None;
                }
                let b = w.trailing_zeros() as usize;
                w &= w - 1;
                Some(i * 64 + b)
            })
        })
    }

    fn mask_tail(&mut self) {
        let spare = self.len % 64;
        if spare != 0
            && let Some(last) = self.words.last_mut()
        {
            *last &= (1u64 << spare) - 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_ops_and_iteration() {
        for len in [0, 1, 63, 64, 65, 200] {
            let mut a = Bitmap::empty(len);
            let mut b = Bitmap::full(len);
            assert_eq!(b.count(), len);
            for i in (0..len).step_by(3) {
                a.set(i);
            }
            assert_eq!(a.ones().collect::<Vec<_>>(), (0..len).step_by(3).collect::<Vec<_>>());
            b.not();
            assert_eq!(b.count(), 0);
            b.or(&a);
            assert_eq!(b, a);
            a.not();
            assert_eq!(a.count(), len - len.div_ceil(3));
            a.and(&b);
            assert_eq!(a.count(), 0);
            let mut c = Bitmap::full(len);
            c.and_not_words(b.words.clone().into_iter());
            assert_eq!(c.count(), len - b.count());
        }
    }
}
