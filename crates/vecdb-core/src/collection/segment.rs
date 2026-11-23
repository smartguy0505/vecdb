//! Segments: the growing one (in RAM, searched exactly) and sealed ones
//! (immutable files: flat, searched exactly in RAM, or an SSD index).
//! Rows are addressed by position: a row's place in its segment. Deletes
//! are tombstones per position, so a deleted row never hides a later
//! insert of the same id elsewhere.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use super::bytes::{Dec, Enc, write_file};
use super::meta::Attrs;
use crate::disk::DiskIndex;
use crate::distance::Metric;
use crate::error::{Error, Result};
use crate::kernels::Kernels;
use crate::store::{InMemoryStore, VectorStore};

pub(crate) fn flat_name(id: u64) -> String {
    format!("seg-{id:08}.flat")
}

pub(crate) fn disk_name(id: u64) -> String {
    format!("seg-{id:08}.vdx")
}

pub(crate) fn attrs_name(id: u64) -> String {
    format!("seg-{id:08}.attrs")
}

pub(crate) fn del_name(id: u64, generation: u64) -> String {
    format!("seg-{id:08}.del-{generation:08}")
}

/// Deleted positions of a sealed segment. Bits are only ever set, and
/// atomically, so searches can read them while a writer adds deletes.
pub(crate) struct Tombstones {
    words: Vec<AtomicU64>,
    len: usize,
    count: AtomicUsize,
}

const DEL_MAGIC: [u8; 8] = *b"VECDBDEL";
const DEL_VERSION: u32 = 1;

impl Tombstones {
    pub fn new(len: usize) -> Self {
        Self { words: (0..len.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(), len, count: AtomicUsize::new(0) }
    }

    #[inline]
    pub fn is_set(&self, pos: usize) -> bool {
        self.words[pos / 64].load(Ordering::Relaxed) & (1 << (pos % 64)) != 0
    }

    /// Marks `pos` deleted; false if it already was.
    pub fn set(&self, pos: usize) -> bool {
        let bit = 1 << (pos % 64);
        let fresh = self.words[pos / 64].fetch_or(bit, Ordering::Relaxed) & bit == 0;
        if fresh {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
        fresh
    }

    pub fn count(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Writes the tombstone file (fsynced; the caller fsyncs the directory).
    pub fn write(&self, path: &Path, segment: u64) -> Result<usize> {
        let mut e = Enc::default();
        e.buf.extend_from_slice(&DEL_MAGIC);
        e.u32(DEL_VERSION);
        e.u64(segment);
        e.u64(self.len as u64);
        let words: Vec<u64> = self.words.iter().map(|w| w.load(Ordering::Relaxed)).collect();
        let count: usize = words.iter().map(|w| w.count_ones() as usize).sum();
        e.u64(count as u64);
        words.iter().for_each(|&w| e.u64(w));
        e.crc();
        write_file(path, &e.buf, false)?;
        Ok(count)
    }

    /// Reads a tombstone file written for `segment` with `len` positions
    /// and `count` deletes (as the manifest recorded).
    pub fn read(path: &Path, segment: u64, len: usize, count: usize) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let mut d = Dec::checked(&bytes, "tombstone file")?;
        if d.take(8)? != DEL_MAGIC {
            return Err(d.corrupt("bad magic"));
        }
        let version = d.u32()?;
        if version != DEL_VERSION {
            return Err(Error::Unsupported(format!("tombstone file version {version}")));
        }
        if d.u64()? != segment || d.u64()? != len as u64 || d.u64()? != count as u64 {
            return Err(d.corrupt("does not match the manifest"));
        }
        let t = Self::new(len);
        let mut seen = 0;
        for w in &t.words {
            let v = d.u64()?;
            w.store(v, Ordering::Relaxed);
            seen += v.count_ones() as usize;
        }
        d.done()?;
        let spare = len % 64;
        if seen != count || (spare != 0 && t.words.last().unwrap().load(Ordering::Relaxed) >> spare != 0) {
            return Err(d.corrupt("bits do not match the count"));
        }
        t.count.store(count, Ordering::Relaxed);
        Ok(t)
    }
}

/// Attributes per position, as stored in `seg-*.attrs`. Kept only when the
/// collection declares fields.
#[derive(Default)]
pub(crate) struct AttrTable {
    pub rows: Vec<Attrs>,
}

const ATTR_MAGIC: [u8; 8] = *b"VECDBATR";
const ATTR_VERSION: u32 = 1;

impl AttrTable {
    pub fn write(&self, path: &Path) -> Result<()> {
        let mut e = Enc::default();
        e.buf.extend_from_slice(&ATTR_MAGIC);
        e.u32(ATTR_VERSION);
        e.u64(self.rows.len() as u64);
        for a in &self.rows {
            e.u32(a.encoded_len() as u32);
            a.encode(&mut e);
        }
        e.crc();
        write_file(path, &e.buf, false)
    }

    pub fn read(path: &Path, positions: usize) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let mut d = Dec::checked(&bytes, "attribute file")?;
        if d.take(8)? != ATTR_MAGIC {
            return Err(d.corrupt("bad magic"));
        }
        let version = d.u32()?;
        if version != ATTR_VERSION {
            return Err(Error::Unsupported(format!("attribute file version {version}")));
        }
        if d.u64()? != positions as u64 {
            return Err(d.corrupt("row count does not match the segment"));
        }
        let mut rows = Vec::with_capacity(positions);
        for _ in 0..positions {
            let n = d.u32()? as usize;
            rows.push(Attrs::decode(d.take(n)?)?);
        }
        d.done()?;
        Ok(Self { rows })
    }
}

/// Exact distances from a prepared query to `rows` (row-major, prepared),
/// a block at a time with the batch kernels; calls `push(row, distance)`
/// for every row `admit` accepts.
pub(crate) fn scan(
    kern: &Kernels,
    metric: Metric,
    q: &[f32],
    rows: &[f32],
    admit: impl Fn(usize) -> bool,
    mut push: impl FnMut(usize, f32),
) {
    const BLOCK: usize = 256;
    let dim = q.len();
    let n = rows.len() / dim;
    let mut out = [0.0f32; BLOCK];
    let mut start = 0;
    while start < n {
        let end = (start + BLOCK).min(n);
        if (start..end).any(&admit) {
            let o = &mut out[..end - start];
            match metric {
                Metric::L2 => (kern.l2_squared_batch)(q, &rows[start * dim..end * dim], o),
                Metric::InnerProduct | Metric::Cosine => (kern.dot_batch)(q, &rows[start * dim..end * dim], o),
            }
            for (j, &raw) in o.iter().enumerate() {
                if admit(start + j) {
                    let d = match metric {
                        Metric::L2 => raw,
                        Metric::InnerProduct => -raw,
                        Metric::Cosine => 1.0 - raw,
                    };
                    push(start + j, d);
                }
            }
        }
        start = end;
    }
}

/// The growing segment: new rows in RAM, searched exactly. Deleted and
/// replaced rows stay until the segment is sealed.
pub(crate) struct Growing {
    dim: usize,
    pub ids: Vec<u64>,
    /// Prepared vectors, row-major.
    pub data: Vec<f32>,
    /// Per row, when the collection has fields.
    pub attrs: Vec<Attrs>,
    dead: Vec<bool>,
    dead_count: usize,
    /// Live id -> row.
    rows: HashMap<u64, usize>,
}

impl Growing {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            ids: Vec::new(),
            data: Vec::new(),
            attrs: Vec::new(),
            dead: Vec::new(),
            dead_count: 0,
            rows: HashMap::new(),
        }
    }

    /// Rows, live or dead.
    pub fn rows(&self) -> usize {
        self.ids.len()
    }

    pub fn live(&self) -> usize {
        self.rows.len()
    }

    /// Bytes of vectors held (live or dead).
    pub fn bytes(&self) -> u64 {
        self.data.len() as u64 * 4
    }

    pub fn row_of(&self, id: u64) -> Option<usize> {
        self.rows.get(&id).copied()
    }

    #[inline]
    pub fn is_dead(&self, row: usize) -> bool {
        self.dead[row]
    }

    /// Appends a row; the id must not be live here.
    pub fn push(&mut self, id: u64, prepared: &[f32], attrs: Option<Attrs>) {
        debug_assert!(!self.rows.contains_key(&id));
        self.rows.insert(id, self.ids.len());
        self.ids.push(id);
        self.data.extend_from_slice(prepared);
        if let Some(a) = attrs {
            self.attrs.push(a);
        }
        self.dead.push(false);
    }

    pub fn delete(&mut self, id: u64) -> bool {
        match self.rows.remove(&id) {
            Some(row) => {
                self.dead[row] = true;
                self.dead_count += 1;
                true
            }
            None => false,
        }
    }

    pub fn vector(&self, row: usize) -> &[f32] {
        &self.data[row * self.dim..(row + 1) * self.dim]
    }
}

pub(crate) enum Body {
    /// Vectors in RAM (loaded from a flat segment file), searched exactly.
    Flat(InMemoryStore),
    /// An SSD index, plus its positions sorted by id for lookups.
    Disk { index: DiskIndex, by_id: Vec<u32> },
}

/// A sealed segment: immutable files plus tombstones.
pub(crate) struct Sealed {
    pub id: u64,
    pub body: Body,
    pub tomb: Tombstones,
    pub attrs: AttrTable,
    /// The files that make up the segment (not its tombstone file).
    pub files: Vec<PathBuf>,
}

impl Sealed {
    pub fn flat(id: u64, store: InMemoryStore, attrs: AttrTable, files: Vec<PathBuf>) -> Self {
        let tomb = Tombstones::new(store.len());
        Self { id, body: Body::Flat(store), tomb, attrs, files }
    }

    pub fn disk(id: u64, index: DiskIndex, attrs: AttrTable, files: Vec<PathBuf>) -> Self {
        let mut by_id: Vec<u32> = (0..index.len() as u32).collect();
        by_id.sort_unstable_by_key(|&p| index.id_at(p as usize));
        let tomb = Tombstones::new(index.len());
        Self { id, body: Body::Disk { index, by_id }, tomb, attrs, files }
    }

    pub fn positions(&self) -> usize {
        match &self.body {
            Body::Flat(s) => s.len(),
            Body::Disk { index, .. } => index.len(),
        }
    }

    pub fn live(&self) -> usize {
        self.positions() - self.tomb.count()
    }

    pub fn id_at(&self, pos: usize) -> u64 {
        match &self.body {
            Body::Flat(s) => s.id(pos),
            Body::Disk { index, .. } => index.id_at(pos),
        }
    }

    /// Position of `id` if it is here and not deleted.
    pub fn find(&self, id: u64) -> Option<usize> {
        let pos = match &self.body {
            Body::Flat(s) => s.row_of(id)?,
            Body::Disk { index, by_id } => {
                let i = by_id.binary_search_by_key(&id, |&p| index.id_at(p as usize)).ok()?;
                by_id[i] as usize
            }
        };
        (!self.tomb.is_set(pos)).then_some(pos)
    }

    /// Bytes of RAM beyond the index itself: tombstones and the id lookup.
    pub fn lookup_bytes(&self) -> usize {
        let by_id = match &self.body {
            Body::Flat(_) => 0,
            Body::Disk { by_id, .. } => by_id.len() * 4,
        };
        self.positions().div_ceil(8) + by_id
    }
}
