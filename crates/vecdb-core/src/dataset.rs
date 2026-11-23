//! Readers and writers for the TEXMEX benchmark formats used by SIFT1M,
//! GIST1M and BIGANN (SIFT1B):
//! - `.fvecs`: per row, `i32` dim then `dim` x `f32`
//! - `.ivecs`: per row, `i32` dim then `dim` x `i32` (ground-truth neighbor ids)
//! - `.bvecs`: per row, `i32` dim then `dim` x `u8`
//!
//! All little-endian. [`VectorFile`] also reads the `.fbin` and `.u8bin`
//! formats of big-ann-benchmarks (`u32` rows, `u32` dim, then the rows), and
//! reads rows by position instead of loading the file, for builds larger
//! than RAM.

use std::fs::File;
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::Path;

use crate::error::{Error, Result};
use crate::store::{MAX_DIM, check_dim};

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

/// Rows read by position, for builds that do not hold every vector in
/// memory ([`DiskIndex::build_partitioned`](crate::DiskIndex::build_partitioned)).
pub trait VectorSource: Sync {
    fn dim(&self) -> usize;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Bytes one row takes in storage. Readers use it to decide when reading
    /// through rows they do not need beats a separate read.
    fn row_bytes(&self) -> usize;
    /// Reads rows `first..first + out.len() / dim()` into `out` as `f32`.
    fn read_rows(&self, first: usize, out: &mut [f32]) -> Result<()>;
}

impl VectorSource for Vecs<f32> {
    fn dim(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        Vecs::len(self)
    }

    fn row_bytes(&self) -> usize {
        self.dim * 4
    }

    fn read_rows(&self, first: usize, out: &mut [f32]) -> Result<()> {
        let start = first * self.dim;
        let rows = self.data.get(start..start + out.len()).ok_or_else(|| {
            Error::InvalidArgument(format!("rows {first}.. out of range ({} rows)", Vecs::len(self)))
        })?;
        out.copy_from_slice(rows);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FileFormat {
    /// `i32` dim before every row.
    Fvecs,
    Bvecs,
    /// One header (`u32` rows, `u32` dim), then bare rows.
    Fbin,
    U8bin,
}

impl FileFormat {
    fn value_bytes(self) -> usize {
        match self {
            FileFormat::Fvecs | FileFormat::Fbin => 4,
            FileFormat::Bvecs | FileFormat::U8bin => 1,
        }
    }

    fn row_prefix(self) -> usize {
        match self {
            FileFormat::Fvecs | FileFormat::Bvecs => 4,
            FileFormat::Fbin | FileFormat::U8bin => 0,
        }
    }
}

/// A `.fvecs`, `.bvecs`, `.fbin` or `.u8bin` file, read by position rather
/// than loaded. Rows are converted to `f32`.
#[derive(Debug)]
pub struct VectorFile {
    file: File,
    format: FileFormat,
    dim: usize,
    rows: usize,
    /// Offset of row 0, and bytes per row (with its prefix).
    start: u64,
    stride: usize,
}

/// Largest single read.
const READ_CHUNK: usize = 16 << 20;

impl VectorFile {
    /// Opens `path` (format by extension); `limit` keeps only the first rows.
    pub fn open(path: &Path, limit: Option<usize>) -> Result<Self> {
        let format = match path.extension().and_then(|e| e.to_str()) {
            Some("fvecs") => FileFormat::Fvecs,
            Some("bvecs") => FileFormat::Bvecs,
            Some("fbin") => FileFormat::Fbin,
            Some("u8bin") => FileFormat::U8bin,
            _ => {
                return Err(Error::InvalidArgument(format!(
                    "{}: expected .fvecs, .bvecs, .fbin or .u8bin",
                    path.display()
                )));
            }
        };
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        let mut head = [0u8; 8];
        let (dim, rows, start) = match format {
            FileFormat::Fvecs | FileFormat::Bvecs => {
                if len == 0 {
                    return Err(Error::Corrupt(format!("{}: empty file", path.display())));
                }
                crate::disk::read_exact_at(&file, &mut head[..4], 0)?;
                let dim = i32::from_le_bytes(head[..4].try_into().unwrap());
                let dim = usize::try_from(dim).map_err(|_| Error::Corrupt(format!("bad dimension {dim}")))?;
                check_dim(dim).map_err(|_| Error::Corrupt(format!("bad dimension {dim}")))?;
                let stride = (4 + dim * format.value_bytes()) as u64;
                if len % stride != 0 {
                    return Err(Error::Corrupt(format!(
                        "{}: {len} bytes is not a whole number of {stride}-byte rows",
                        path.display()
                    )));
                }
                (dim, len / stride, 0)
            }
            FileFormat::Fbin | FileFormat::U8bin => {
                if len < 8 {
                    return Err(Error::Corrupt(format!("{}: no header", path.display())));
                }
                crate::disk::read_exact_at(&file, &mut head, 0)?;
                let rows = u32::from_le_bytes(head[..4].try_into().unwrap()) as u64;
                let dim = u32::from_le_bytes(head[4..].try_into().unwrap()) as usize;
                check_dim(dim).map_err(|_| Error::Corrupt(format!("bad dimension {dim}")))?;
                let need = rows.checked_mul((dim * format.value_bytes()) as u64).and_then(|b| b.checked_add(8));
                if need.is_none_or(|need| need > len) {
                    return Err(Error::Corrupt(format!("{}: header says {rows} rows, file too short", path.display())));
                }
                (dim, rows, 8)
            }
        };
        let rows = usize::try_from(rows).map_err(|_| Error::InvalidArgument("too many rows".into()))?;
        let rows = limit.map_or(rows, |l| l.min(rows));
        let stride = format.row_prefix() + dim * format.value_bytes();
        Ok(Self { file, format, dim, rows, start, stride })
    }
}

impl VectorSource for VectorFile {
    fn dim(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        self.rows
    }

    fn row_bytes(&self) -> usize {
        self.stride
    }

    fn read_rows(&self, first: usize, out: &mut [f32]) -> Result<()> {
        let dim = self.dim;
        if out.len() % dim != 0 || first + out.len() / dim > self.rows {
            return Err(Error::InvalidArgument(format!("rows {first}.. out of range ({} rows)", self.rows)));
        }
        let per_read = (READ_CHUNK / self.stride).max(1);
        let mut buf = Vec::new();
        for (c, rows) in out.chunks_mut(per_read * dim).enumerate() {
            let row0 = first + c * per_read;
            let count = rows.len() / dim;
            buf.resize(count * self.stride, 0u8);
            crate::disk::read_exact_at(&self.file, &mut buf, self.start + (row0 * self.stride) as u64)?;
            for (i, (src, dst)) in buf.chunks_exact(self.stride).zip(rows.chunks_exact_mut(dim)).enumerate() {
                let (prefix, values) = src.split_at(self.format.row_prefix());
                if !prefix.is_empty() && i32::from_le_bytes(prefix.try_into().unwrap()) as i64 != dim as i64 {
                    return Err(Error::Corrupt(format!("row {}: dimension prefix is not {dim}", row0 + i)));
                }
                match self.format.value_bytes() {
                    4 => {
                        for (d, b) in dst.iter_mut().zip(values.as_chunks::<4>().0) {
                            *d = f32::from_le_bytes(*b);
                        }
                    }
                    _ => {
                        for (d, &b) in dst.iter_mut().zip(values) {
                            *d = b as f32;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_file_reads_every_format() {
        let dir = crate::disk::test_dir("dataset");
        let (dim, n) = (3usize, 7usize);
        let data: Vec<f32> = (0..dim * n).map(|i| i as f32).collect();
        let fvecs = dir.join("v.fvecs");
        write_fvecs(&fvecs, dim, &data).unwrap();
        let bytes: Vec<u8> = data.iter().map(|&x| x as u8).collect();
        let bvecs = dir.join("v.bvecs");
        let mut b = Vec::new();
        for row in bytes.chunks(dim) {
            b.extend((dim as i32).to_le_bytes());
            b.extend(row);
        }
        std::fs::write(&bvecs, b).unwrap();
        let mut head = (n as u32).to_le_bytes().to_vec();
        head.extend((dim as u32).to_le_bytes());
        let fbin = dir.join("v.fbin");
        std::fs::write(&fbin, [head.clone(), data.iter().flat_map(|x| x.to_le_bytes()).collect()].concat()).unwrap();
        let u8bin = dir.join("v.u8bin");
        std::fs::write(&u8bin, [head, bytes].concat()).unwrap();

        for path in [&fvecs, &bvecs, &fbin, &u8bin] {
            let f = VectorFile::open(path, None).unwrap();
            assert_eq!((f.dim(), f.len()), (dim, n), "{}", path.display());
            let mut all = vec![0.0; dim * n];
            f.read_rows(0, &mut all).unwrap();
            assert_eq!(all, data, "{}", path.display());
            let mut two = vec![0.0; 2 * dim];
            f.read_rows(4, &mut two).unwrap();
            assert_eq!(two, data[4 * dim..6 * dim]);
            assert!(f.read_rows(6, &mut two).is_err());
            assert_eq!(VectorFile::open(path, Some(2)).unwrap().len(), 2);
        }
        // A row whose prefix disagrees with the first row's.
        let mut b = std::fs::read(&fvecs).unwrap();
        let stride = 4 + 4 * dim;
        b[3 * stride] = 9;
        std::fs::write(&fvecs, b).unwrap();
        let f = VectorFile::open(&fvecs, None).unwrap();
        let mut all = vec![0.0; dim * n];
        assert!(matches!(f.read_rows(0, &mut all), Err(Error::Corrupt(_))));
        // A truncated file.
        std::fs::write(&fbin, [7u8, 0, 0, 0, 3, 0, 0, 0, 1, 2]).unwrap();
        assert!(matches!(VectorFile::open(&fbin, None), Err(Error::Corrupt(_))));
    }
}
