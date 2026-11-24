//! Segments: the growing one (in RAM, searched exactly) and sealed ones
//! (immutable files: flat, searched exactly in RAM, or an SSD index).
//! Rows are addressed by position: a row's place in its segment. Deletes
//! are tombstones per position, so a deleted row never hides a later
//! insert of the same id elsewhere.
//!
//! **Snapshots.** Every write has a sequence number, and a search reads as
//! of one: it sees the writes up to that number and none after. Growing
//! rows record the writes that inserted and deleted them. A sealed
//! segment's tombstone bits are shared by all searches, so each delete
//! also records its number until no open search predates it
//! ([`Tombstones::prune`]); [`AsOf`] reads the bits as of a snapshot.

use std::cell::OnceCell;
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use super::bytes::{Dec, Enc, write_file};
use super::columns::{ColumnBuilder, SegmentAttrs};
use super::meta::Field;
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
    /// Sequence number of the newest delete; stored before its bit.
    last_seq: AtomicU64,
    /// Deletes an open snapshot may predate, with their sequence numbers;
    /// recorded before their bits are set.
    recent: Mutex<Recent>,
}

#[derive(Default)]
struct Recent {
    /// Ascending sequence numbers.
    order: VecDeque<(u32, u64)>,
    seq_of: HashMap<u32, u64>,
}

const DEL_MAGIC: [u8; 8] = *b"VECDBDEL";
const DEL_VERSION: u32 = 1;

impl Tombstones {
    pub fn new(len: usize) -> Self {
        Self {
            words: (0..len.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(),
            len,
            count: AtomicUsize::new(0),
            last_seq: AtomicU64::new(0),
            recent: Mutex::new(Recent::default()),
        }
    }

    /// Whether `pos` is deleted now (by any write so far).
    #[inline]
    pub fn is_set(&self, pos: usize) -> bool {
        self.words[pos / 64].load(Ordering::Acquire) & (1 << (pos % 64)) != 0
    }

    /// Marks `pos` deleted by write `seq` (ascending from call to call);
    /// snapshots before `seq` still see the row. False if it already was
    /// deleted. Only the writer calls this.
    pub fn delete(&self, pos: usize, seq: u64) -> bool {
        if self.is_set(pos) {
            return false;
        }
        // The number first, then the bit: a search that sees the bit (an
        // acquire load of the word) also sees the number.
        self.last_seq.store(seq, Ordering::Relaxed);
        {
            let mut r = self.recent.lock().unwrap();
            r.order.push_back((pos as u32, seq));
            r.seq_of.insert(pos as u32, seq);
        }
        self.words[pos / 64].fetch_or(1 << (pos % 64), Ordering::Release);
        self.count.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Marks `pos` deleted for every snapshot: for deletes older than any
    /// search that can see this segment (carried into a new segment).
    pub fn settle(&self, pos: usize) -> bool {
        let bit = 1 << (pos % 64);
        let fresh = self.words[pos / 64].fetch_or(bit, Ordering::Release) & bit == 0;
        if fresh {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
        fresh
    }

    /// Whether `pos` was deleted as of write `seq` (see [`AsOf`] for many
    /// positions).
    pub fn deleted_as_of(&self, pos: usize, seq: u64) -> bool {
        if !self.is_set(pos) {
            return false;
        }
        if self.last_seq.load(Ordering::Acquire) <= seq {
            return true;
        }
        // A pruned record is older than every open snapshot.
        self.recent.lock().unwrap().seq_of.get(&(pos as u32)).is_none_or(|&d| d <= seq)
    }

    /// The deleted bits as of write `seq`, a word per 64 positions. Exact
    /// for good when taken after the snapshot: deletes up to `seq` were all
    /// applied before it, and later ones are left out.
    pub fn words_as_of(&self, seq: u64) -> Vec<u64> {
        let mut w: Vec<u64> = self.words.iter().map(|x| x.load(Ordering::Acquire)).collect();
        if self.last_seq.load(Ordering::Acquire) > seq {
            let r = self.recent.lock().unwrap();
            for &(p, d) in r.order.iter().rev() {
                if d <= seq {
                    break;
                }
                w[p as usize / 64] &= !(1 << (p % 64));
            }
        }
        w
    }

    /// Forgets the numbers of deletes up to `seq`, which no open snapshot
    /// predates.
    pub fn prune(&self, seq: u64) {
        if self.last_seq.load(Ordering::Relaxed) == 0 {
            return;
        }
        let mut r = self.recent.lock().unwrap();
        while let Some(&(p, d)) = r.order.front() {
            if d > seq {
                break;
            }
            r.order.pop_front();
            r.seq_of.remove(&p);
        }
    }

    /// Delete records kept for open snapshots.
    pub fn recent_len(&self) -> usize {
        self.recent.lock().unwrap().order.len()
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
        let words: Vec<u64> = self.words.iter().map(|w| w.load(Ordering::Acquire)).collect();
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

/// A segment's tombstones as of one snapshot, for many lookups: reads the
/// shared bits until a delete newer than the snapshot shows up, then a copy
/// of the bits as of the snapshot.
pub(crate) struct AsOf<'a> {
    tomb: &'a Tombstones,
    seq: u64,
    frozen: OnceCell<Vec<u64>>,
}

impl<'a> AsOf<'a> {
    pub fn new(tomb: &'a Tombstones, seq: u64) -> Self {
        Self { tomb, seq, frozen: OnceCell::new() }
    }

    #[inline]
    pub fn deleted(&self, pos: usize) -> bool {
        if let Some(w) = self.frozen.get() {
            return w[pos / 64] >> (pos % 64) & 1 != 0;
        }
        if !self.tomb.is_set(pos) {
            return false;
        }
        if self.tomb.last_seq.load(Ordering::Acquire) <= self.seq {
            return true;
        }
        let w = self.frozen.get_or_init(|| self.tomb.words_as_of(self.seq));
        w[pos / 64] >> (pos % 64) & 1 != 0
    }

    /// Clears the positions deleted as of the snapshot from `b`.
    pub fn remove_from(&self, b: &mut super::bitmap::Bitmap) {
        match self.frozen.get() {
            Some(w) => b.and_not_words(w.iter().copied()),
            None => b.and_not_words(self.tomb.words_as_of(self.seq).into_iter()),
        }
    }
}

/// Exact distances from a prepared query to `rows` (row-major, prepared),
/// a block at a time with the batch kernels; calls `push(row, distance)`
/// for every row `admit` accepts. Blocks where few rows are admitted (a
/// selective filter) are scored row by row. Returns the rows scored.
pub(crate) fn scan(
    kern: &Kernels,
    metric: Metric,
    q: &[f32],
    rows: &[f32],
    admit: impl Fn(usize) -> bool,
    mut push: impl FnMut(usize, f32),
) -> usize {
    const BLOCK: usize = 256;
    /// Below this many admitted rows a block is scored row by row.
    const SPARSE: usize = BLOCK / 8;
    let dim = q.len();
    let n = rows.len() / dim;
    let mut out = [0.0f32; BLOCK];
    let finish = |raw: f32| match metric {
        Metric::L2 => raw,
        Metric::InnerProduct => -raw,
        Metric::Cosine => 1.0 - raw,
    };
    let mut scored = 0;
    let mut start = 0;
    while start < n {
        let end = (start + BLOCK).min(n);
        let admitted = (start..end).filter(|&r| admit(r)).count();
        if admitted == 0 {
        } else if admitted < SPARSE {
            for r in (start..end).filter(|&r| admit(r)) {
                let v = &rows[r * dim..(r + 1) * dim];
                let raw = match metric {
                    Metric::L2 => (kern.l2_squared)(q, v),
                    Metric::InnerProduct | Metric::Cosine => (kern.dot)(q, v),
                };
                push(r, finish(raw));
            }
        } else {
            let o = &mut out[..end - start];
            match metric {
                Metric::L2 => (kern.l2_squared_batch)(q, &rows[start * dim..end * dim], o),
                Metric::InnerProduct | Metric::Cosine => (kern.dot_batch)(q, &rows[start * dim..end * dim], o),
            }
            for (j, &raw) in o.iter().enumerate() {
                if admit(start + j) {
                    push(start + j, finish(raw));
                }
            }
        }
        scored += admitted;
        start = end;
    }
    scored
}

/// The growing segment: new rows in RAM, searched exactly. Deleted and
/// replaced rows stay until the segment is sealed. Each row records the
/// write that inserted it and the one that deleted it, so a search sees
/// the segment as of its snapshot.
pub(crate) struct Growing {
    dim: usize,
    pub ids: Vec<u64>,
    /// Prepared vectors, row-major.
    pub data: Vec<f32>,
    /// Encoded attributes (see `columns`): row `r` is
    /// `attr_bytes[attr_off[r]..attr_off[r + 1]]`. Kept only when the
    /// collection has fields, as is the column index.
    attr_bytes: Vec<u8>,
    attr_off: Vec<usize>,
    pub cols: Option<ColumnBuilder>,
    fields: Vec<Field>,
    /// Sequence numbers of the writes that inserted each row (ascending)
    /// and deleted it (0: live).
    ins: Vec<u64>,
    del: Vec<u64>,
    live: usize,
    /// Id -> its newest row here, live or not; `prev[r]`: the row of the
    /// same id before `r` (`NONE` if none).
    latest: HashMap<u64, u32>,
    prev: Vec<u32>,
}

const NONE: u32 = u32::MAX;

impl Growing {
    pub fn new(dim: usize, fields: &[Field]) -> Self {
        Self {
            dim,
            ids: Vec::new(),
            data: Vec::new(),
            attr_bytes: Vec::new(),
            attr_off: vec![0],
            cols: (!fields.is_empty()).then(|| ColumnBuilder::new(fields)),
            fields: fields.to_vec(),
            ins: Vec::new(),
            del: Vec::new(),
            live: 0,
            latest: HashMap::new(),
            prev: Vec::new(),
        }
    }

    /// Encoded attributes of `row` (empty without fields).
    pub fn attrs(&self, row: usize) -> &[u8] {
        if self.cols.is_none() {
            return &[];
        }
        &self.attr_bytes[self.attr_off[row]..self.attr_off[row + 1]]
    }

    /// RAM held by attributes and their index.
    pub fn attr_ram(&self) -> usize {
        self.attr_bytes.capacity() + self.attr_off.capacity() * 8 + self.cols.as_ref().map_or(0, |c| c.ram_bytes())
    }

    /// Rows, live or dead.
    pub fn rows(&self) -> usize {
        self.ids.len()
    }

    /// Live rows now.
    pub fn live(&self) -> usize {
        self.live
    }

    /// Bytes of vectors held (live or dead).
    pub fn bytes(&self) -> u64 {
        self.data.len() as u64 * 4
    }

    /// The live row of `id` now.
    pub fn row_of(&self, id: u64) -> Option<usize> {
        let r = *self.latest.get(&id)? as usize;
        (self.del[r] == 0).then_some(r)
    }

    /// The row of `id` live as of write `seq`.
    pub fn row_at(&self, id: u64, seq: u64) -> Option<usize> {
        let mut r = *self.latest.get(&id)?;
        while r != NONE && self.ins[r as usize] > seq {
            r = self.prev[r as usize];
        }
        (r != NONE && self.live_at(r as usize, seq)).then_some(r as usize)
    }

    /// Whether `row` is deleted now.
    #[inline]
    pub fn is_dead(&self, row: usize) -> bool {
        self.del[row] != 0
    }

    /// The rows inserted up to write `seq` (a prefix: rows are appended in
    /// write order).
    pub fn rows_at(&self, seq: u64) -> usize {
        self.ins.partition_point(|&s| s <= seq)
    }

    /// Whether `row` (inserted up to `seq`) is not deleted as of `seq`.
    #[inline]
    pub fn live_at(&self, row: usize, seq: u64) -> bool {
        let d = self.del[row];
        d == 0 || d > seq
    }

    /// Appends a row inserted by write `seq`; the id must not be live here.
    /// `attrs` is the row's encoded attributes (ignored without fields).
    pub fn push(&mut self, id: u64, prepared: &[f32], attrs: &[u8], seq: u64) {
        debug_assert!(self.row_of(id).is_none());
        debug_assert!(self.ins.last().is_none_or(|&s| s <= seq));
        let row = self.ids.len();
        assert!(row < NONE as usize, "growing segment too large");
        self.prev.push(self.latest.insert(id, row as u32).unwrap_or(NONE));
        self.ids.push(id);
        self.data.extend_from_slice(prepared);
        if let Some(cols) = &mut self.cols {
            cols.add(row, attrs, &self.fields).expect("rows are encoded by encode_row");
            self.attr_bytes.extend_from_slice(attrs);
            self.attr_off.push(self.attr_bytes.len());
        }
        self.ins.push(seq);
        self.del.push(0);
        self.live += 1;
    }

    /// Deletes the live row of `id`, by write `seq`.
    pub fn delete(&mut self, id: u64, seq: u64) -> bool {
        match self.row_of(id) {
            Some(r) => {
                self.del[r] = seq;
                self.live -= 1;
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

/// A sealed segment: immutable files plus tombstones. Its files stay open
/// (the SSD index, and `raw` for whole-vector reads), so searches that
/// still hold it can read it after a build removed its files.
pub(crate) struct Sealed {
    pub id: u64,
    pub body: Body,
    pub tomb: Tombstones,
    /// When the collection has fields.
    pub attrs: Option<SegmentAttrs>,
    /// The files that make up the segment (not its tombstone file).
    pub files: Vec<PathBuf>,
    /// SSD segments: the index file, for buffered reads of whole vectors.
    pub raw: Option<File>,
}

impl Sealed {
    pub fn flat(id: u64, store: InMemoryStore, attrs: Option<SegmentAttrs>, files: Vec<PathBuf>) -> Self {
        let tomb = Tombstones::new(store.len());
        Self { id, body: Body::Flat(store), tomb, attrs, files, raw: None }
    }

    /// `files[0]` is the index file.
    pub fn disk(id: u64, index: DiskIndex, attrs: Option<SegmentAttrs>, files: Vec<PathBuf>) -> Result<Self> {
        let raw = File::open(&files[0])?;
        let mut by_id: Vec<u32> = (0..index.len() as u32).collect();
        by_id.sort_unstable_by_key(|&p| index.id_at(p as usize));
        let tomb = Tombstones::new(index.len());
        Ok(Self { id, body: Body::Disk { index, by_id }, tomb, attrs, files, raw: Some(raw) })
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

    /// Position of `id` if it is here, deleted or not.
    pub fn locate(&self, id: u64) -> Option<usize> {
        match &self.body {
            Body::Flat(s) => s.row_of(id),
            Body::Disk { index, by_id } => {
                let i = by_id.binary_search_by_key(&id, |&p| index.id_at(p as usize)).ok()?;
                Some(by_id[i] as usize)
            }
        }
    }

    /// Position of `id` if it is here and not deleted now.
    pub fn find(&self, id: u64) -> Option<usize> {
        self.locate(id).filter(|&pos| !self.tomb.is_set(pos))
    }

    /// RAM held by the attribute columns.
    pub fn attr_ram(&self) -> usize {
        self.attrs.as_ref().map_or(0, |a| a.columns.ram_bytes())
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
