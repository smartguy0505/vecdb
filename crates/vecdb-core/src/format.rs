//! On-disk segment format, version 1 (flat: ids + raw vectors).
//!
//! All integers and floats are little-endian.
//!
//! ```text
//! offset  size  field
//!      0     8  magic "VECDBSEG"
//!      8     4  format version (1)
//!     12     4  metric code (Metric::to_code)
//!     16     4  dim
//!     20     4  flags (0; reserved)
//!     24     8  count (rows)
//!     32     8  ids_offset (64)
//!     40     8  vectors_offset (64-byte aligned, for SIMD loads / mmap)
//!     48     4  data_crc: CRC-32C of bytes [64, end of file)
//!     52     8  reserved (0)
//!     60     4  header_crc: CRC-32C of bytes [0, 60)
//!     64        ids: count x u64
//!               zero padding up to vectors_offset
//!               vectors: count x dim x f32, row-major
//! ```
//!
//! Vectors are stored already prepared for the metric (unit length for cosine).
//! Writes go to `<path>.tmp`, are fsynced, then renamed over `<path>` and the
//! directory is fsynced, so a crash leaves either the old file or the new one.

use std::fs::{self, File};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::crc32c::{Crc32c, crc32c};
use crate::distance::Metric;
use crate::error::{Error, Result};
use crate::store::{InMemoryStore, MAX_DIM, VectorStore};

pub const MAGIC: [u8; 8] = *b"VECDBSEG";
pub const VERSION: u32 = 1;
pub const HEADER_LEN: usize = 64;
const VECTOR_ALIGN: u64 = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentHeader {
    pub version: u32,
    pub metric: Metric,
    pub dim: u32,
    pub count: u64,
    pub ids_offset: u64,
    pub vectors_offset: u64,
    pub data_crc: u32,
}

impl SegmentHeader {
    fn encode(&self) -> [u8; HEADER_LEN] {
        let mut b = [0u8; HEADER_LEN];
        b[0..8].copy_from_slice(&MAGIC);
        b[8..12].copy_from_slice(&self.version.to_le_bytes());
        b[12..16].copy_from_slice(&self.metric.to_code().to_le_bytes());
        b[16..20].copy_from_slice(&self.dim.to_le_bytes());
        // 20..24 flags = 0
        b[24..32].copy_from_slice(&self.count.to_le_bytes());
        b[32..40].copy_from_slice(&self.ids_offset.to_le_bytes());
        b[40..48].copy_from_slice(&self.vectors_offset.to_le_bytes());
        b[48..52].copy_from_slice(&self.data_crc.to_le_bytes());
        // 52..60 reserved = 0
        let hcrc = crc32c(&b[0..60]);
        b[60..64].copy_from_slice(&hcrc.to_le_bytes());
        b
    }

    fn decode(b: &[u8]) -> Result<Self> {
        if b.len() < HEADER_LEN {
            return Err(Error::Corrupt("file shorter than segment header".into()));
        }
        if b[0..8] != MAGIC {
            return Err(Error::Corrupt("bad magic: not a vecdb segment".into()));
        }
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        if crc32c(&b[0..60]) != u32_at(60) {
            return Err(Error::Corrupt("header checksum mismatch".into()));
        }
        let version = u32_at(8);
        if version != VERSION {
            return Err(Error::Unsupported(format!("segment format version {version}")));
        }
        let metric = Metric::from_code(u32_at(12))
            .ok_or_else(|| Error::Corrupt(format!("unknown metric code {}", u32_at(12))))?;
        if u32_at(20) != 0 {
            return Err(Error::Unsupported(format!("segment flags {:#x}", u32_at(20))));
        }
        Ok(Self {
            version,
            metric,
            dim: u32_at(16),
            count: u64_at(24),
            ids_offset: u64_at(32),
            vectors_offset: u64_at(40),
            data_crc: u32_at(48),
        })
    }
}

fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

fn tmp_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".tmp");
    PathBuf::from(s)
}

/// Writes `store` as a flat segment at `path`, atomically replacing any existing file.
pub fn write_flat_segment<S: VectorStore>(path: &Path, metric: Metric, store: &S) -> Result<()> {
    let tmp = tmp_path(path);
    let result = write_to(&tmp, metric, store).and_then(|()| {
        fs::rename(&tmp, path)?;
        sync_parent_dir(path)
    });
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

fn write_to<S: VectorStore>(tmp: &Path, metric: Metric, store: &S) -> Result<()> {
    let dim = store.dim();
    let count = store.len() as u64;
    let ids_offset = HEADER_LEN as u64;
    let ids_end = ids_offset + count * 8;
    let vectors_offset = align_up(ids_end, VECTOR_ALIGN);

    let file = File::create(tmp)?;
    let mut w = BufWriter::with_capacity(1 << 20, file);
    w.write_all(&[0u8; HEADER_LEN])?; // placeholder; real header written last

    let mut crc = Crc32c::new();
    for row in 0..store.len() {
        let b = store.id(row).to_le_bytes();
        crc.update(&b);
        w.write_all(&b)?;
    }
    let pad = vec![0u8; (vectors_offset - ids_end) as usize];
    crc.update(&pad);
    w.write_all(&pad)?;

    let mut buf = Vec::with_capacity(dim * 4);
    for row in 0..store.len() {
        buf.clear();
        for &x in store.vector(row) {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        crc.update(&buf);
        w.write_all(&buf)?;
    }

    let mut file = w.into_inner().map_err(|e| e.into_error())?;
    let header = SegmentHeader {
        version: VERSION,
        metric,
        dim: dim as u32,
        count,
        ids_offset,
        vectors_offset,
        data_crc: crc.finalize(),
    };
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&header.encode())?;
    file.sync_all()?;
    Ok(())
}

fn sync_parent_dir(path: &Path) -> Result<()> {
    let dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    File::open(dir)?.sync_all()?;
    Ok(())
}

/// Reads and fully verifies a flat segment into memory.
pub fn read_flat_segment(path: &Path) -> Result<(Metric, InMemoryStore)> {
    let bytes = fs::read(path)?;
    let h = SegmentHeader::decode(&bytes)?;

    let dim = h.dim as usize;
    if dim == 0 || dim > MAX_DIM {
        return Err(Error::Corrupt(format!("dimension {dim} out of range")));
    }
    let corrupt = |m: &str| Error::Corrupt(m.to_string());
    let count = usize::try_from(h.count).map_err(|_| corrupt("row count too large"))?;
    let ids_len = count.checked_mul(8).ok_or_else(|| corrupt("row count too large"))?;
    let vec_len = count
        .checked_mul(dim)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| corrupt("row count too large"))?;
    let ids_off = usize::try_from(h.ids_offset).map_err(|_| corrupt("bad ids offset"))?;
    let vec_off = usize::try_from(h.vectors_offset).map_err(|_| corrupt("bad vectors offset"))?;
    if ids_off != HEADER_LEN {
        return Err(corrupt("bad ids offset"));
    }
    if vec_off < ids_off + ids_len || vec_off % 4 != 0 {
        return Err(corrupt("bad vectors offset"));
    }
    if vec_off.checked_add(vec_len) != Some(bytes.len()) {
        return Err(Error::Corrupt(format!(
            "file is {} bytes, header implies {}",
            bytes.len(),
            vec_off as u128 + vec_len as u128
        )));
    }
    if crc32c(&bytes[HEADER_LEN..]) != h.data_crc {
        return Err(corrupt("data checksum mismatch"));
    }

    let (id_chunks, _) = bytes[ids_off..ids_off + ids_len].as_chunks::<8>();
    let ids: Vec<u64> = id_chunks.iter().map(|c| u64::from_le_bytes(*c)).collect();
    let (val_chunks, _) = bytes[vec_off..].as_chunks::<4>();
    let data: Vec<f32> = val_chunks.iter().map(|c| f32::from_le_bytes(*c)).collect();

    let store = InMemoryStore::from_parts(dim, ids, data)?;
    Ok((h.metric, store))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip() {
        let h = SegmentHeader {
            version: VERSION,
            metric: Metric::Cosine,
            dim: 768,
            count: 12345,
            ids_offset: 64,
            vectors_offset: align_up(64 + 12345 * 8, 64),
            data_crc: 0xDEAD_BEEF,
        };
        let enc = h.encode();
        assert_eq!(SegmentHeader::decode(&enc).unwrap(), h);
        assert_eq!(h.vectors_offset % 64, 0);
    }

    #[test]
    fn header_rejects_flipped_bit() {
        let h = SegmentHeader {
            version: VERSION,
            metric: Metric::L2,
            dim: 4,
            count: 1,
            ids_offset: 64,
            vectors_offset: 128,
            data_crc: 0,
        };
        let mut enc = h.encode();
        enc[17] ^= 1;
        assert!(matches!(SegmentHeader::decode(&enc), Err(Error::Corrupt(_))));
    }
}
