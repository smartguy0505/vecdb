//! Attributes inside the engine: rows encoded with field numbers, and the
//! per-segment column index that filters read.
//!
//! A row's attributes are encoded as
//!
//! ```text
//! count u16, then per value: field u16, then  int: i64
//!                                             keyword: u16 length + UTF-8
//! ```
//!
//! (the WAL keeps field names: see `meta`). A sealed segment's attribute
//! file holds its rows and its columns:
//!
//! ```text
//! header (48 bytes): magic "VECDBATR", version u32 (2), fields u32,
//!                    positions u64, rows_len u64, columns_len u64,
//!                    reserved u32, crc u32 of bytes 0..44
//! rows:     per row, in any order: crc u32 (of the rest of the record),
//!           len u32, position u64, encoded row (len - 8 bytes)
//! offsets:  per position, u64: where its row starts in `rows`; crc u32
//! columns:  per field, in declaration order: type u8, then
//!             int:     n u64, n values i64 (ascending), n positions u32
//!             keyword: keys u64, then per key (ascending): u16 length +
//!                      bytes, count u32, count positions u32 (ascending)
//!           crc u32 of the columns
//! ```
//!
//! Only the columns are loaded into RAM; [`SegmentAttrs::row`] reads one
//! row back from the file (its checksum and position checked).

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use super::bitmap::Bitmap;
use super::bytes::{Dec, Enc};
use super::filter::{ColumnSource, merge_ranges};
use super::meta::{Attrs, Field, FieldType, Value};
use crate::crc32c::{Crc32c, crc32c};
use crate::error::{Error, Result};

/// One value of an encoded row.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ValueRef<'a> {
    Int(i64),
    Keyword(&'a [u8]),
}

/// Appends `attrs` (already validated against `fields`) to `out`.
pub(crate) fn encode_row(attrs: &Attrs, fields: &[Field], out: &mut Vec<u8>) {
    out.extend_from_slice(&(attrs.entries().len() as u16).to_le_bytes());
    for (name, v) in attrs.entries() {
        let f = fields.iter().position(|f| f.name == *name).expect("validated");
        out.extend_from_slice(&(f as u16).to_le_bytes());
        match v {
            Value::Int(i) => out.extend_from_slice(&i.to_le_bytes()),
            Value::Keyword(s) => {
                out.extend_from_slice(&(s.len() as u16).to_le_bytes());
                out.extend_from_slice(s.as_bytes());
            }
        }
    }
}

/// Calls `f(field, value)` for every value of an encoded row.
pub(crate) fn row_values<'a>(row: &'a [u8], fields: &[Field], mut f: impl FnMut(usize, ValueRef<'a>)) -> Result<()> {
    let mut d = Dec::new(row, "attribute row");
    for _ in 0..d.u16()? {
        let field = d.u16()? as usize;
        let Some(fd) = fields.get(field) else {
            return Err(d.corrupt("unknown field number"));
        };
        let v = match fd.ty {
            FieldType::Int => ValueRef::Int(d.i64()?),
            FieldType::Keyword => {
                let n = d.u16()? as usize;
                ValueRef::Keyword(d.take(n)?)
            }
        };
        f(field, v);
    }
    d.done()
}

/// The [`Attrs`] of an encoded row.
pub(crate) fn decode_row(row: &[u8], fields: &[Field]) -> Result<Attrs> {
    let mut entries = Vec::new();
    let mut bad = false;
    row_values(row, fields, |field, v| {
        let name = &fields[field].name;
        match v {
            ValueRef::Int(i) => entries.push((name.clone(), Value::Int(i))),
            ValueRef::Keyword(b) => match std::str::from_utf8(b) {
                Ok(s) => entries.push((name.clone(), Value::Keyword(s.into()))),
                Err(_) => bad = true,
            },
        }
    })?;
    if bad {
        return Err(Error::Corrupt("attribute row: keyword is not UTF-8".into()));
    }
    Ok(Attrs::from_entries(entries))
}

/// Columns being filled, one row at a time: the growing segment's index,
/// and the input of a sealed segment's [`Columns`].
pub(crate) struct ColumnBuilder {
    positions: usize,
    cols: Vec<BuildCol>,
}

enum BuildCol {
    Int(Vec<(i64, u32)>),
    Keyword(HashMap<Vec<u8>, Vec<u32>>),
}

impl ColumnBuilder {
    pub fn new(fields: &[Field]) -> Self {
        let cols = fields
            .iter()
            .map(|f| match f.ty {
                FieldType::Int => BuildCol::Int(Vec::new()),
                FieldType::Keyword => BuildCol::Keyword(HashMap::new()),
            })
            .collect();
        Self { positions: 0, cols }
    }

    /// Adds the values of the encoded row at `pos`.
    pub fn add(&mut self, pos: usize, row: &[u8], fields: &[Field]) -> Result<()> {
        let p = u32::try_from(pos).map_err(|_| Error::InvalidArgument("segment too large".into()))?;
        self.positions = self.positions.max(pos + 1);
        let cols = &mut self.cols;
        row_values(row, fields, |field, v| match (&mut cols[field], v) {
            (BuildCol::Int(pairs), ValueRef::Int(i)) => pairs.push((i, p)),
            (BuildCol::Keyword(map), ValueRef::Keyword(k)) => {
                let list = map.entry(k.to_vec()).or_default();
                if list.last() != Some(&p) {
                    list.push(p);
                }
            }
            _ => unreachable!("row_values reads values by their field's type"),
        })
    }

    pub fn set_positions(&mut self, n: usize) {
        self.positions = self.positions.max(n);
    }

    /// RAM held, roughly.
    pub fn ram_bytes(&self) -> usize {
        self.cols
            .iter()
            .map(|c| match c {
                BuildCol::Int(p) => p.capacity() * 16,
                BuildCol::Keyword(m) => m.iter().map(|(k, v)| k.capacity() + v.capacity() * 4 + 64).sum(),
            })
            .sum()
    }

    pub fn freeze(self) -> Columns {
        let cols = self
            .cols
            .into_iter()
            .map(|c| match c {
                BuildCol::Int(mut pairs) => {
                    pairs.sort_unstable();
                    pairs.dedup();
                    let pos: Vec<u32> = pairs.iter().map(|p| p.1).collect();
                    let single = one_each(&pos, self.positions);
                    Col::Int { values: pairs.iter().map(|p| p.0).collect(), pos, single }
                }
                BuildCol::Keyword(map) => {
                    let mut keys: Vec<(Vec<u8>, Vec<u32>)> = map.into_iter().collect();
                    keys.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                    let mut kw = Keywords::default();
                    for (k, mut list) in keys {
                        list.sort_unstable();
                        list.dedup();
                        kw.push(&k, &list);
                    }
                    kw.single = one_each(&kw.postings, self.positions);
                    Col::Keyword(kw)
                }
            })
            .collect();
        Columns { positions: self.positions, cols }
    }
}

impl ColumnSource for ColumnBuilder {
    fn positions(&self) -> usize {
        self.positions
    }

    fn keywords(&self, field: usize, keys: &[Vec<u8>]) -> Bitmap {
        let mut out = Bitmap::empty(self.positions);
        if let BuildCol::Keyword(map) = &self.cols[field] {
            for list in keys.iter().filter_map(|k| map.get(k)) {
                list.iter().for_each(|&p| out.set(p as usize));
            }
        }
        out
    }

    fn ints(&self, field: usize, ranges: &[(i64, i64)]) -> Bitmap {
        let mut out = Bitmap::empty(self.positions);
        if let BuildCol::Int(pairs) = &self.cols[field] {
            let ranges = merge_ranges(ranges);
            for &(v, p) in pairs {
                if ranges.iter().any(|&(lo, hi)| (lo..=hi).contains(&v)) {
                    out.set(p as usize);
                }
            }
        }
        out
    }
}

/// The column index of a sealed segment, held in RAM.
pub(crate) struct Columns {
    positions: usize,
    cols: Vec<Col>,
}

enum Col {
    /// Values ascending (ties by position), with their positions. `single`:
    /// every position has exactly one value.
    Int { values: Vec<i64>, pos: Vec<u32>, single: bool },
    Keyword(Keywords),
}

/// Whether `pos` holds every position below `n` exactly once.
fn one_each(pos: &[u32], n: usize) -> bool {
    if pos.len() != n {
        return false;
    }
    let mut seen = Bitmap::empty(n);
    for &p in pos {
        if seen.get(p as usize) {
            return false;
        }
        seen.set(p as usize);
    }
    true
}

/// Distinct keywords, ascending, each with its positions (ascending).
#[derive(Default)]
struct Keywords {
    bytes: Vec<u8>,
    /// Key `i` is `bytes[key_off[i]..key_off[i + 1]]` (one entry more than
    /// there are keys, once any key was pushed).
    key_off: Vec<u64>,
    /// Its positions are `postings[post_off[i]..post_off[i + 1]]`.
    post_off: Vec<u64>,
    postings: Vec<u32>,
    /// Every position has exactly one keyword.
    single: bool,
}

impl Keywords {
    fn push(&mut self, key: &[u8], list: &[u32]) {
        if self.key_off.is_empty() {
            self.key_off.push(0);
            self.post_off.push(0);
        }
        self.bytes.extend_from_slice(key);
        self.key_off.push(self.bytes.len() as u64);
        self.postings.extend_from_slice(list);
        self.post_off.push(self.postings.len() as u64);
    }

    fn len(&self) -> usize {
        self.key_off.len().saturating_sub(1)
    }

    fn key(&self, i: usize) -> &[u8] {
        &self.bytes[self.key_off[i] as usize..self.key_off[i + 1] as usize]
    }

    fn list(&self, i: usize) -> &[u32] {
        &self.postings[self.post_off[i] as usize..self.post_off[i + 1] as usize]
    }

    fn find(&self, key: &[u8]) -> Option<usize> {
        let (mut lo, mut hi) = (0, self.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            match self.key(mid).cmp(key) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }
}

impl Columns {
    /// RAM held.
    pub fn ram_bytes(&self) -> usize {
        self.cols
            .iter()
            .map(|c| match c {
                Col::Int { values, pos, .. } => values.len() * 8 + pos.len() * 4,
                Col::Keyword(k) => k.bytes.len() + (k.key_off.len() + k.post_off.len()) * 8 + k.postings.len() * 4,
            })
            .sum()
    }

    /// Values held (over all fields).
    pub fn values(&self) -> usize {
        self.cols
            .iter()
            .map(|c| match c {
                Col::Int { values, .. } => values.len(),
                Col::Keyword(k) => k.postings.len(),
            })
            .sum()
    }

    fn encode(&self, e: &mut Enc) {
        for c in &self.cols {
            match c {
                Col::Int { values, pos, .. } => {
                    e.u8(FieldType::Int.code());
                    e.u64(values.len() as u64);
                    values.iter().for_each(|&v| e.i64(v));
                    pos.iter().for_each(|&p| e.u32(p));
                }
                Col::Keyword(k) => {
                    e.u8(FieldType::Keyword.code());
                    e.u64(k.len() as u64);
                    for i in 0..k.len() {
                        let key = k.key(i);
                        e.u16(key.len() as u16);
                        e.buf.extend_from_slice(key);
                        let list = k.list(i);
                        e.u32(list.len() as u32);
                        list.iter().for_each(|&p| e.u32(p));
                    }
                }
            }
        }
    }

    fn decode(d: &mut Dec, fields: &[Field], positions: usize) -> Result<Self> {
        let mut cols = Vec::with_capacity(fields.len());
        for f in fields {
            if FieldType::from_code(d.u8()?) != Some(f.ty) {
                return Err(d.corrupt("column type does not match the field"));
            }
            let n = d.u64()? as usize;
            match f.ty {
                FieldType::Int => {
                    if n > d.remaining() / 12 {
                        return Err(d.corrupt("column truncated"));
                    }
                    let values: Vec<i64> = (0..n).map(|_| d.i64()).collect::<Result<_>>()?;
                    let pos: Vec<u32> = (0..n).map(|_| d.u32()).collect::<Result<_>>()?;
                    let sorted = (1..n).all(|i| (values[i - 1], pos[i - 1]) < (values[i], pos[i]));
                    if !sorted || pos.iter().any(|&p| p as usize >= positions) {
                        return Err(d.corrupt("int column out of order or range"));
                    }
                    let single = one_each(&pos, positions);
                    cols.push(Col::Int { values, pos, single });
                }
                FieldType::Keyword => {
                    if n > d.remaining() / 6 {
                        return Err(d.corrupt("column truncated"));
                    }
                    let mut k = Keywords::default();
                    let mut list = Vec::new();
                    for i in 0..n {
                        let kl = d.u16()? as usize;
                        let key = d.take(kl)?;
                        if i > 0 && k.key(i - 1) >= key {
                            return Err(d.corrupt("keywords out of order"));
                        }
                        let c = d.u32()? as usize;
                        if c > d.remaining() / 4 {
                            return Err(d.corrupt("column truncated"));
                        }
                        list.clear();
                        for _ in 0..c {
                            list.push(d.u32()?);
                        }
                        if c == 0
                            || (1..c).any(|j| list[j - 1] >= list[j])
                            || list.last().is_some_and(|&p| p as usize >= positions)
                        {
                            return Err(d.corrupt("keyword positions out of order or range"));
                        }
                        k.push(key, &list);
                    }
                    k.single = one_each(&k.postings, positions);
                    cols.push(Col::Keyword(k));
                }
            }
        }
        Ok(Self { positions, cols })
    }
}

impl ColumnSource for Columns {
    fn positions(&self) -> usize {
        self.positions
    }

    // When a column holds one value per position and more than half the
    // positions match, the bitmap starts full and the others are cleared:
    // the cost follows the smaller side.

    fn keywords(&self, field: usize, keys: &[Vec<u8>]) -> Bitmap {
        let n = self.positions;
        let Col::Keyword(k) = &self.cols[field] else { return Bitmap::empty(n) };
        let mut found: Vec<usize> = keys.iter().filter_map(|key| k.find(key)).collect();
        found.sort_unstable();
        found.dedup();
        let matches: usize = found.iter().map(|&i| k.list(i).len()).sum();
        if k.single && matches * 2 > n {
            let mut out = Bitmap::full(n);
            let mut next = found.iter().peekable();
            for i in 0..k.len() {
                if next.peek() == Some(&&i) {
                    next.next();
                } else {
                    k.list(i).iter().for_each(|&p| out.clear(p as usize));
                }
            }
            out
        } else {
            let mut out = Bitmap::empty(n);
            found.iter().for_each(|&i| k.list(i).iter().for_each(|&p| out.set(p as usize)));
            out
        }
    }

    fn ints(&self, field: usize, ranges: &[(i64, i64)]) -> Bitmap {
        let n = self.positions;
        let Col::Int { values, pos, single } = &self.cols[field] else { return Bitmap::empty(n) };
        // Index ranges of `values` that match.
        let spans: Vec<(usize, usize)> = merge_ranges(ranges)
            .iter()
            .map(|&(lo, hi)| (values.partition_point(|&v| v < lo), values.partition_point(|&v| v <= hi)))
            .filter(|(a, b)| a < b)
            .collect();
        let matches: usize = spans.iter().map(|(a, b)| b - a).sum();
        if *single && matches * 2 > n {
            let mut out = Bitmap::full(n);
            let mut from = 0;
            for &(a, b) in spans.iter().chain([(pos.len(), pos.len())].iter()) {
                pos[from..a].iter().for_each(|&p| out.clear(p as usize));
                from = b;
            }
            out
        } else {
            let mut out = Bitmap::empty(n);
            spans.iter().for_each(|&(a, b)| pos[a..b].iter().for_each(|&p| out.set(p as usize)));
            out
        }
    }
}

const MAGIC: [u8; 8] = *b"VECDBATR";
const VERSION: u32 = 2;
const HEADER: u64 = 48;
/// Record header: crc, len, position.
const REC_HEAD: usize = 16;

/// A sealed segment's attributes: the columns in RAM, rows in the file.
pub(crate) struct SegmentAttrs {
    path: PathBuf,
    positions: usize,
    rows_len: u64,
    pub columns: Columns,
}

impl SegmentAttrs {
    pub fn open(path: &Path, fields: &[Field], positions: usize) -> Result<Self> {
        let f = File::open(path)?;
        let file_len = f.metadata()?.len();
        let corrupt = |m: &str| Error::Corrupt(format!("attribute file {}: {m}", path.display()));
        if file_len < HEADER {
            return Err(corrupt("truncated"));
        }
        let mut h = [0u8; HEADER as usize];
        f.read_exact_at(&mut h, 0)?;
        let mut d = Dec::checked(&h, "attribute file header")?;
        if d.take(8)? != MAGIC {
            return Err(corrupt("bad magic"));
        }
        let version = d.u32()?;
        if version != VERSION {
            return Err(Error::Unsupported(format!("attribute file version {version}")));
        }
        let (nf, np, rows_len, cols_len) = (d.u32()? as usize, d.u64()?, d.u64()?, d.u64()?);
        if nf != fields.len() || np != positions as u64 {
            return Err(corrupt("does not match the segment"));
        }
        let offsets_end = HEADER.checked_add(rows_len).and_then(|x| x.checked_add(8 * np + 4));
        if offsets_end.and_then(|x| x.checked_add(cols_len)) != Some(file_len) || cols_len < 4 {
            return Err(corrupt("section lengths do not match the file"));
        }
        let mut cols = vec![0u8; cols_len as usize];
        f.read_exact_at(&mut cols, offsets_end.unwrap())?;
        let mut d = Dec::checked(&cols, "attribute columns")?;
        let columns = Columns::decode(&mut d, fields, positions)?;
        d.done()?;
        Ok(Self { path: path.to_path_buf(), positions, rows_len, columns })
    }

    /// Reads the row at `pos` from the file.
    pub fn row(&self, pos: usize, fields: &[Field]) -> Result<Attrs> {
        let f = File::open(&self.path)?;
        let mut b8 = [0u8; 8];
        f.read_exact_at(&mut b8, HEADER + self.rows_len + 8 * pos as u64)?;
        let off = u64::from_le_bytes(b8);
        let corrupt = |m: &str| Error::Corrupt(format!("attribute file {}: row {pos}: {m}", self.path.display()));
        if off.checked_add(REC_HEAD as u64).is_none_or(|e| e > self.rows_len) {
            return Err(corrupt("offset out of range"));
        }
        f.read_exact_at(&mut b8, HEADER + off)?;
        let crc = u32::from_le_bytes(b8[0..4].try_into().unwrap());
        let len = u32::from_le_bytes(b8[4..8].try_into().unwrap()) as u64;
        if len < 8 || off + 8 + len > self.rows_len {
            return Err(corrupt("bad length"));
        }
        let mut body = vec![0u8; len as usize];
        f.read_exact_at(&mut body, HEADER + off + 8)?;
        let mut c = Crc32c::new();
        c.update(&b8[4..8]);
        c.update(&body);
        if c.finalize() != crc {
            return Err(corrupt("checksum mismatch"));
        }
        if u64::from_le_bytes(body[0..8].try_into().unwrap()) != pos as u64 {
            return Err(corrupt("offset points at another row"));
        }
        decode_row(&body[8..], fields)
    }

    /// Calls `f(pos, row)` for every row, in file order (checksums checked,
    /// every position exactly once).
    pub fn for_each_row(&self, mut f: impl FnMut(usize, &[u8]) -> Result<()>) -> Result<()> {
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(HEADER))?;
        let mut r = BufReader::with_capacity(1 << 20, file.take(self.rows_len));
        let corrupt = |m: &str| Error::Corrupt(format!("attribute file {}: {m}", self.path.display()));
        let mut seen = Bitmap::empty(self.positions);
        let mut body = Vec::new();
        let mut head = [0u8; 8];
        let mut read = 0u64;
        while read < self.rows_len {
            r.read_exact(&mut head)?;
            let crc = u32::from_le_bytes(head[0..4].try_into().unwrap());
            let len = u32::from_le_bytes(head[4..8].try_into().unwrap()) as u64;
            if len < 8 || read + 8 + len > self.rows_len {
                return Err(corrupt("bad row length"));
            }
            body.resize(len as usize, 0);
            r.read_exact(&mut body)?;
            if crc32c_parts(&head[4..8], &body) != crc {
                return Err(corrupt("row checksum mismatch"));
            }
            let pos = u64::from_le_bytes(body[0..8].try_into().unwrap()) as usize;
            if pos >= self.positions || seen.get(pos) {
                return Err(corrupt("row position out of range or repeated"));
            }
            seen.set(pos);
            f(pos, &body[8..])?;
            read += 8 + len;
        }
        if seen.count() != self.positions {
            return Err(corrupt("rows missing"));
        }
        Ok(())
    }
}

fn crc32c_parts(a: &[u8], b: &[u8]) -> u32 {
    let mut c = Crc32c::new();
    c.update(a);
    c.update(b);
    c.finalize()
}

/// Writes a segment's attribute file: rows pushed in any order, each
/// position once.
pub(crate) struct AttrWriter {
    path: PathBuf,
    out: BufWriter<File>,
    fields: Vec<Field>,
    offsets: Vec<u64>,
    rows_len: u64,
    builder: ColumnBuilder,
    rec: Vec<u8>,
}

impl AttrWriter {
    pub fn create(path: &Path, fields: &[Field], positions: usize) -> Result<Self> {
        if positions > u32::MAX as usize {
            return Err(Error::InvalidArgument("segment too large for its attribute file".into()));
        }
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        let mut out = BufWriter::with_capacity(1 << 20, file);
        out.write_all(&[0u8; HEADER as usize])?;
        let mut builder = ColumnBuilder::new(fields);
        builder.set_positions(positions);
        Ok(Self {
            path: path.to_path_buf(),
            out,
            fields: fields.to_vec(),
            offsets: vec![u64::MAX; positions],
            rows_len: 0,
            builder,
            rec: Vec::new(),
        })
    }

    pub fn push(&mut self, pos: usize, row: &[u8]) -> Result<()> {
        if self.offsets.get(pos) != Some(&u64::MAX) {
            return Err(Error::InvalidArgument(format!("attribute row {pos} out of range or written twice")));
        }
        self.builder.add(pos, row, &self.fields)?;
        let len = 8 + row.len();
        self.rec.clear();
        self.rec.extend_from_slice(&[0; 4]);
        self.rec.extend_from_slice(&(len as u32).to_le_bytes());
        self.rec.extend_from_slice(&(pos as u64).to_le_bytes());
        self.rec.extend_from_slice(row);
        let crc = crc32c(&self.rec[4..]);
        self.rec[0..4].copy_from_slice(&crc.to_le_bytes());
        self.out.write_all(&self.rec)?;
        self.offsets[pos] = self.rows_len;
        self.rows_len += self.rec.len() as u64;
        Ok(())
    }

    /// Writes the offsets, columns and header and fsyncs the file (the
    /// caller fsyncs the directory).
    pub fn finish(mut self) -> Result<SegmentAttrs> {
        if let Some(p) = self.offsets.iter().position(|&o| o == u64::MAX) {
            return Err(Error::InvalidArgument(format!("attribute row {p} never written")));
        }
        let mut e = Enc::default();
        self.offsets.iter().for_each(|&o| e.u64(o));
        e.crc();
        self.out.write_all(&e.buf)?;
        let positions = self.offsets.len();
        let columns = self.builder.freeze();
        let mut e = Enc::default();
        columns.encode(&mut e);
        e.crc();
        self.out.write_all(&e.buf)?;
        let cols_len = e.buf.len() as u64;
        let mut h = Enc::default();
        h.buf.extend_from_slice(&MAGIC);
        h.u32(VERSION);
        h.u32(self.fields.len() as u32);
        h.u64(positions as u64);
        h.u64(self.rows_len);
        h.u64(cols_len);
        h.u32(0);
        h.crc();
        let mut file = self.out.into_inner().map_err(|e| e.into_error())?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&h.buf)?;
        file.sync_all()?;
        Ok(SegmentAttrs { path: self.path, positions, rows_len: self.rows_len, columns })
    }
}
