//! Readers and writers for the TEXMEX benchmark formats used by SIFT1M,
//! GIST1M and BIGANN (SIFT1B):
//! - `.fvecs`: per row, `i32` dim then `dim` x `f32`
//! - `.ivecs`: per row, `i32` dim then `dim` x `i32` (ground-truth neighbor ids)
//! - `.bvecs`: per row, `i32` dim then `dim` x `u8`
//!
//! All little-endian. The `.fbin`/`.u8bin` formats of big-ann-benchmarks can be
//! added here later.

use std::fs::File;
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::Path;

use crate::error::{Error, Result};
use crate::store::MAX_DIM;

/// Rows of equal dimension, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Vecs<T> {
    pub dim: usize,
    pub data: Vec<T>,
}

impl<T> Vecs<T> {
    pub fn len(&self) -> usize {
        self.data.len().checked_div(self.dim).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn row(&self, i: usize) -> &[T] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }
}

/// Reads `buf.len()` bytes. Returns Ok(false) on a clean EOF before the first
/// byte, and a Corrupt error on EOF partway through.
fn read_exact_or_eof(r: &mut impl Read, buf: &mut [u8], what: &str) -> Result<bool> {
    let mut filled = 0;
    while filled < buf.len() {
        match r.read(&mut buf[filled..]) {
            Ok(0) if filled == 0 => return Ok(false),
            Ok(0) => return Err(Error::Corrupt(format!("truncated {what}"))),
            Ok(n) => filled += n,
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(true)
}

fn read_vecs<T, const W: usize>(
    path: &Path,
    limit: Option<usize>,
    decode: fn([u8; W]) -> T,
) -> Result<Vecs<T>> {
    let mut r = BufReader::with_capacity(1 << 20, File::open(path)?);
    let limit = limit.unwrap_or(usize::MAX);
    let mut dim: Option<usize> = None;
    let mut data = Vec::new();
    let mut row_bytes = Vec::new();
    let mut rows = 0usize;
    let mut head = [0u8; 4];
    while rows < limit {
        if !read_exact_or_eof(&mut r, &mut head, "row header")? {
            break;
        }
        let d = i32::from_le_bytes(head);
        let d = usize::try_from(d)
            .ok()
            .filter(|&d| d > 0 && d <= MAX_DIM)
            .ok_or_else(|| Error::Corrupt(format!("row {rows}: bad dimension {d}")))?;
        match dim {
            None => {
                dim = Some(d);
                row_bytes.resize(d * W, 0);
            }
            Some(prev) if prev != d => {
                return Err(Error::Corrupt(format!("row {rows}: dimension {d}, earlier rows {prev}")));
            }
            Some(_) => {}
        }
        if !read_exact_or_eof(&mut r, &mut row_bytes, "row")? {
            return Err(Error::Corrupt(format!("row {rows}: truncated")));
        }
        let (chunks, _) = row_bytes.as_chunks::<W>();
        data.extend(chunks.iter().map(|c| decode(*c)));
        rows += 1;
    }
    Ok(Vecs { dim: dim.unwrap_or(0), data })
}

pub fn read_fvecs(path: &Path, limit: Option<usize>) -> Result<Vecs<f32>> {
    read_vecs::<f32, 4>(path, limit, f32::from_le_bytes)
}

pub fn read_ivecs(path: &Path, limit: Option<usize>) -> Result<Vecs<i32>> {
    read_vecs::<i32, 4>(path, limit, i32::from_le_bytes)
}

pub fn read_bvecs(path: &Path, limit: Option<usize>) -> Result<Vecs<u8>> {
    read_vecs::<u8, 1>(path, limit, |b| b[0])
}

/// Reads `.fvecs` or `.bvecs` (by extension) as `f32`.
pub fn read_vectors_f32(path: &Path, limit: Option<usize>) -> Result<Vecs<f32>> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("fvecs") => read_fvecs(path, limit),
        Some("bvecs") => {
            let b = read_bvecs(path, limit)?;
            Ok(Vecs { dim: b.dim, data: b.data.into_iter().map(f32::from).collect() })
        }
        _ => Err(Error::InvalidArgument(format!(
            "{}: expected a .fvecs or .bvecs file",
            path.display()
        ))),
    }
}

fn write_vecs<T: Copy, const W: usize>(
    path: &Path,
    dim: usize,
    data: &[T],
    encode: fn(T) -> [u8; W],
) -> Result<()> {
    if dim == 0 || dim > MAX_DIM || data.len() % dim != 0 {
        return Err(Error::InvalidArgument(format!(
            "{} values do not form rows of dimension {dim}",
            data.len()
        )));
    }
    let mut w = BufWriter::with_capacity(1 << 20, File::create(path)?);
    let head = (dim as i32).to_le_bytes();
    for row in data.chunks_exact(dim) {
        w.write_all(&head)?;
        for &x in row {
            w.write_all(&encode(x))?;
        }
    }
    w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    Ok(())
}

pub fn write_fvecs(path: &Path, dim: usize, data: &[f32]) -> Result<()> {
    write_vecs(path, dim, data, f32::to_le_bytes)
}

pub fn write_ivecs(path: &Path, dim: usize, data: &[i32]) -> Result<()> {
    write_vecs(path, dim, data, i32::to_le_bytes)
}
