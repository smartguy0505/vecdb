//! Vector storage.
//!
//! Indexes read vectors through the [`VectorStore`] trait, so the in-memory
//! store can later be swapped for an mmap-backed or SSD-backed one without
//! touching the index code.

use std::collections::HashMap;

use crate::distance::check_finite;
use crate::error::{Error, Result};

/// Largest supported dimension. Guards against absurd allocations from
/// corrupt files or bad input.
pub const MAX_DIM: usize = 65_536;

pub trait VectorStore {
    fn dim(&self) -> usize;
    /// Number of stored rows.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// External id of `row`. Panics if `row >= len()`.
    fn id(&self, row: usize) -> u64;
    /// Vector of `row`, `dim()` values. Panics if `row >= len()`.
    fn vector(&self, row: usize) -> &[f32];
}

pub(crate) fn check_dim(dim: usize) -> Result<()> {
    if dim == 0 || dim > MAX_DIM {
        return Err(Error::InvalidArgument(format!("dimension {dim} not in 1..={MAX_DIM}")));
    }
    Ok(())
}

/// Append-only, contiguous in-memory store: ids in one array, vectors in one
/// row-major `f32` array.
#[derive(Debug, Clone)]
pub struct InMemoryStore {
    dim: usize,
    ids: Vec<u64>,
    data: Vec<f32>,
    rows_by_id: HashMap<u64, usize>,
}

impl InMemoryStore {
    pub fn new(dim: usize) -> Result<Self> {
        Self::with_capacity(dim, 0)
    }

    pub fn with_capacity(dim: usize, rows: usize) -> Result<Self> {
        check_dim(dim)?;
        Ok(Self {
            dim,
            ids: Vec::with_capacity(rows),
            data: Vec::with_capacity(rows.saturating_mul(dim)),
            rows_by_id: HashMap::with_capacity(rows),
        })
    }

    /// Rebuilds a store from raw parts (e.g. a loaded segment), validating them.
    pub fn from_parts(dim: usize, ids: Vec<u64>, data: Vec<f32>) -> Result<Self> {
        check_dim(dim)?;
        if Some(data.len()) != ids.len().checked_mul(dim) {
            return Err(Error::Corrupt(format!(
                "{} values do not form {} rows of dimension {dim}",
                data.len(),
                ids.len()
            )));
        }
        check_finite(&data)?;
        let mut rows_by_id = HashMap::with_capacity(ids.len());
        for (row, &id) in ids.iter().enumerate() {
            if rows_by_id.insert(id, row).is_some() {
                return Err(Error::DuplicateId(id));
            }
        }
        Ok(Self { dim, ids, data, rows_by_id })
    }

    /// Appends a vector as-is and returns its row. Metric-specific preparation
    /// (normalization) is the index's job, not the store's.
    pub fn push(&mut self, id: u64, vector: &[f32]) -> Result<usize> {
        if vector.len() != self.dim {
            return Err(Error::DimensionMismatch { expected: self.dim, got: vector.len() });
        }
        check_finite(vector)?;
        if self.rows_by_id.contains_key(&id) {
            return Err(Error::DuplicateId(id));
        }
        let row = self.ids.len();
        self.ids.push(id);
        self.data.extend_from_slice(vector);
        self.rows_by_id.insert(id, row);
        Ok(row)
    }

    pub fn row_of(&self, id: u64) -> Option<usize> {
        self.rows_by_id.get(&id).copied()
    }

    pub fn get(&self, id: u64) -> Option<&[f32]> {
        self.row_of(id).map(|row| self.vector(row))
    }

    pub fn ids(&self) -> &[u64] {
        &self.ids
    }

    /// All vectors, row-major.
    pub fn raw_vectors(&self) -> &[f32] {
        &self.data
    }
}

impl VectorStore for InMemoryStore {
    fn dim(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        self.ids.len()
    }

    #[inline]
    fn id(&self, row: usize) -> u64 {
        self.ids[row]
    }

    #[inline]
    fn vector(&self, row: usize) -> &[f32] {
        &self.data[row * self.dim..(row + 1) * self.dim]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_get() {
        let mut s = InMemoryStore::new(3).unwrap();
        assert_eq!(s.push(42, &[1.0, 2.0, 3.0]).unwrap(), 0);
        assert_eq!(s.push(7, &[4.0, 5.0, 6.0]).unwrap(), 1);
        assert_eq!(s.len(), 2);
        assert_eq!(s.get(7), Some(&[4.0, 5.0, 6.0][..]));
        assert_eq!(s.id(0), 42);
        assert_eq!(s.get(1), None);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(InMemoryStore::new(0).is_err());
        assert!(InMemoryStore::new(MAX_DIM + 1).is_err());
        let mut s = InMemoryStore::new(2).unwrap();
        assert!(matches!(
            s.push(1, &[1.0]),
            Err(Error::DimensionMismatch { expected: 2, got: 1 })
        ));
        assert!(matches!(s.push(1, &[1.0, f32::NAN]), Err(Error::NonFiniteValue)));
        s.push(1, &[1.0, 2.0]).unwrap();
        assert!(matches!(s.push(1, &[3.0, 4.0]), Err(Error::DuplicateId(1))));
        // Failed pushes must not leave partial state behind.
        assert_eq!(s.len(), 1);
        assert_eq!(s.raw_vectors().len(), 2);
    }

    #[test]
    fn from_parts_validates() {
        assert!(InMemoryStore::from_parts(2, vec![1, 2], vec![0.0; 4]).is_ok());
        assert!(InMemoryStore::from_parts(2, vec![1, 2], vec![0.0; 3]).is_err());
        assert!(matches!(
            InMemoryStore::from_parts(1, vec![5, 5], vec![0.0; 2]),
            Err(Error::DuplicateId(5))
        ));
    }
}
