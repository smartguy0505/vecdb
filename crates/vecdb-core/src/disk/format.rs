//! SSD index file format, version 1.
//!
//! One file, little-endian, made of 4096-byte sectors. Every section starts
//! on a sector boundary.
//!
//! ```text
//! sector 0      header (below), CRC-32C of bytes [0, 4092) in its last 4 bytes
//! graph         one record per node, packed into blocks (read from SSD)
//! vectors       separate layout only: one block per node (read from SSD)
//! codebooks     PQ centroids, KSUB x dim x f32              (loaded into RAM)
//! codes         PQ codes, nodes x m bytes                    (loaded into RAM)
//! id_start      (nodes + 1) x u64; absent when every node has one id (RAM)
//! ids           caller ids, u64, grouped by node             (loaded into RAM)
//! ```
//!
//! Records never straddle a block. A block is one sector holding
//! `4096 / record_len` records, or, for a record longer than a sector,
//! `ceil(record_len / 4096)` sectors holding one record. Search reads one
//! whole block per expanded node.
//!
//! Inline layout (full vector stored with the neighbors; each read gives the
//! exact distance of the node it expands):
//!
//! ```text
//! [vector: dim x codec][degree: u32][neighbors: max_degree x u32, unused = 0][crc: u32]
//! ```
//!
//! Separate layout (for long vectors; navigation reads only neighbor lists,
//! and full vectors are read only to re-rank the final candidates):
//!
//! ```text
//! graph record:  [degree: u32][neighbors: max_degree x u32][vector crc: u32][crc: u32]
//! vector block:  [vector: dim x codec][zero padding to a whole sector]
//! ```
//!
//! Every record ends with the CRC-32C of its other bytes. A vector block's
//! checksum lives in its node's graph record, so vectors need no padding
//! sector for it; re-ranked nodes have always been expanded first, so their
//! records are already in hand.
//!
//! Header fields (offsets in bytes):
//!
//! ```text
//!   0  8  magic "VECDBDSK"           52  4  PQ subspaces m
//!   8  4  version (1)                56  4  record_len   \
//!  12  4  metric code                60  4  records_per_block  } derived from
//!  16  4  dim                        64  4  block_bytes  }  the fields before;
//!  20  4  layout (1 inline, 2 sep.)  68  4  vector_block_bytes /  stored to cross-check
//!  24  4  vector codec (0 f32, 1 f16, 2 bf16)
//!  28  4  max_degree                 72  6 sections x (offset u64, len u64, crc u32, 0 u32):
//!  32  8  nodes                          graph, vectors, codebooks, codes, id_start, ids
//!  40  8  ids                       216  zeros up to 4092
//!  48  4  entry node               4092  4  header CRC-32C
//! ```
//!
//! Section CRCs cover the RAM sections and are checked when the file is
//! opened; the graph and vector sections are covered by their per-record
//! checksums instead (their header CRC field is 0).

use crate::crc32c::crc32c;
use crate::distance::Metric;
use crate::error::{Error, Result};
use crate::graph::MAX_DEGREE;
use crate::quant::Codec;
use crate::quant::codec::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16};
use crate::quant::pq::KSUB;
use crate::store::MAX_DIM;

pub const MAGIC: [u8; 8] = *b"VECDBDSK";
pub const VERSION: u32 = 1;
/// Unit of layout and of every read. Also the alignment O_DIRECT needs on
/// common SSDs (4 KiB logical blocks); 512-byte devices accept it too.
pub const SECTOR: usize = 4096;
const HEADER_CRC_AT: usize = SECTOR - 4;

/// Largest inline record [`Layout::auto`] keeps inline: four sectors, about
/// 4,030 f32 dimensions (8,060 at f16) with 64 neighbors.
///
/// Measured (W = 4, equal recall): at 960 dims (2-sector records) inline
/// needs fewer reads and fewer bytes; at 2048 dims (3 sectors) the two tie
/// on latency, separate reading 14-26% fewer bytes in 60-80% more reads;
/// at 8192 dims (9 sectors) separate is 10-18% faster. The cutoff sits
/// between the measured points.
pub const INLINE_MAX_RECORD: usize = 4 * SECTOR;

/// Where full vectors live. See the module docs and [`INLINE_MAX_RECORD`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Each node's vector is stored with its neighbor list.
    Inline,
    /// Neighbor lists and vectors are stored apart.
    Separate,
}

impl Layout {
    pub fn name(self) -> &'static str {
        match self {
            Layout::Inline => "inline",
            Layout::Separate => "separate",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "inline" => Some(Layout::Inline),
            "separate" => Some(Layout::Separate),
            _ => None,
        }
    }

    /// Inline while a record (vector + neighbor list) fits in
    /// [`INLINE_MAX_RECORD`] bytes, separate beyond that.
    pub fn auto(dim: usize, codec: Codec, max_degree: usize) -> Self {
        if Geometry::new(Layout::Inline, dim, codec, max_degree).record_len <= INLINE_MAX_RECORD {
            Layout::Inline
        } else {
            Layout::Separate
        }
    }

    pub(crate) fn code(self) -> u32 {
        match self {
            Layout::Inline => 1,
            Layout::Separate => 2,
        }
    }

    pub(crate) fn from_code(c: u32) -> Option<Self> {
        match c {
            1 => Some(Layout::Inline),
            2 => Some(Layout::Separate),
            _ => None,
        }
    }
}

/// Stable on-disk codes for the vector codecs the SSD index supports.
pub(crate) fn codec_code(c: Codec) -> Option<u32> {
    match c {
        Codec::F32 => Some(0),
        Codec::F16 => Some(1),
        Codec::BF16 => Some(2),
        Codec::SQ8 => None,
    }
}

pub(crate) fn codec_from_code(c: u32) -> Option<Codec> {
    match c {
        0 => Some(Codec::F32),
        1 => Some(Codec::F16),
        2 => Some(Codec::BF16),
        _ => None,
    }
}

/// Writes `v` into `out` (`v.len() * codec.bytes_per_dim()` bytes).
/// f16 rejects finite values beyond its range instead of storing infinity.
pub(crate) fn encode_vector(codec: Codec, v: &[f32], out: &mut [u8]) -> Result<()> {
    match codec {
        Codec::F32 => {
            for (o, x) in out.chunks_exact_mut(4).zip(v) {
                o.copy_from_slice(&x.to_le_bytes());
            }
        }
        Codec::F16 | Codec::BF16 => {
            let (to, from): (fn(f32) -> u16, fn(u16) -> f32) =
                if codec == Codec::F16 { (f32_to_f16, f16_to_f32) } else { (f32_to_bf16, bf16_to_f32) };
            for (o, &x) in out.chunks_exact_mut(2).zip(v) {
                let h = to(x);
                if x.is_finite() && !from(h).is_finite() {
                    return Err(Error::InvalidArgument(format!("value {x} is outside the {} range", codec.name())));
                }
                o.copy_from_slice(&h.to_le_bytes());
            }
        }
        Codec::SQ8 => return Err(Error::InvalidArgument("sq8 vectors are not supported in an SSD index".into())),
    }
    Ok(())
}

/// Reads a vector written by [`encode_vector`] into `out`.
#[inline]
pub(crate) fn decode_vector(codec: Codec, bytes: &[u8], out: &mut [f32]) {
    match codec {
        Codec::F32 => {
            for (o, b) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
                *o = f32::from_le_bytes(*b);
            }
        }
        Codec::F16 => {
            for (o, b) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
                *o = f16_to_f32(u16::from_le_bytes(*b));
            }
        }
        Codec::BF16 => {
            for (o, b) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
                *o = bf16_to_f32(u16::from_le_bytes(*b));
            }
        }
        Codec::SQ8 => unreachable!("rejected when the header is read"),
    }
}

/// Record and block sizes, all derived from (layout, dim, codec, max_degree).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Geometry {
    pub layout: Layout,
    pub vector_bytes: usize,
    pub max_degree: usize,
    /// Bytes of one graph record, its checksum included.
    pub record_len: usize,
    pub records_per_block: usize,
    /// Bytes of one graph block (a multiple of SECTOR).
    pub block_bytes: usize,
    /// Separate layout: bytes of one vector block (a multiple of SECTOR); 0 inline.
    pub vector_block_bytes: usize,
}

impl Geometry {
    pub fn new(layout: Layout, dim: usize, codec: Codec, max_degree: usize) -> Self {
        let vector_bytes = dim * codec.bytes_per_dim();
        let adjacency = 4 + 4 * max_degree;
        let record_len = match layout {
            Layout::Inline => vector_bytes + adjacency + 4,
            Layout::Separate => adjacency + 4 + 4,
        };
        let (records_per_block, block_bytes) = if record_len <= SECTOR {
            (SECTOR / record_len, SECTOR)
        } else {
            (1, record_len.div_ceil(SECTOR) * SECTOR)
        };
        let vector_block_bytes = match layout {
            Layout::Inline => 0,
            Layout::Separate => vector_bytes.div_ceil(SECTOR) * SECTOR,
        };
        Self { layout, vector_bytes, max_degree, record_len, records_per_block, block_bytes, vector_block_bytes }
    }

    pub fn graph_bytes(&self, nodes: u64) -> Option<u64> {
        nodes.div_ceil(self.records_per_block as u64).checked_mul(self.block_bytes as u64)
    }

    /// (offset of the node's block within the graph section, offset of its
    /// record within the block).
    #[inline]
    pub fn locate(&self, node: u32) -> (u64, usize) {
        let rpb = self.records_per_block as u64;
        let n = node as u64;
        ((n / rpb) * self.block_bytes as u64, (n % rpb) as usize * self.record_len)
    }

    /// Largest single read search issues.
    pub fn max_read(&self) -> usize {
        self.block_bytes.max(self.vector_block_bytes)
    }
}

pub(crate) const GRAPH: usize = 0;
pub(crate) const VECTORS: usize = 1;
pub(crate) const CODEBOOKS: usize = 2;
pub(crate) const CODES: usize = 3;
pub(crate) const ID_START: usize = 4;
pub(crate) const IDS: usize = 5;
pub(crate) const SECTIONS: usize = 6;
const SECTION_NAMES: [&str; SECTIONS] = ["graph", "vectors", "codebooks", "codes", "id_start", "ids"];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Section {
    pub offset: u64,
    pub len: u64,
    pub crc: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Header {
    pub metric: Metric,
    pub dim: usize,
    pub layout: Layout,
    pub codec: Codec,
    pub max_degree: usize,
    pub nodes: u64,
    pub ids: u64,
    pub entry: u32,
    pub pq_m: usize,
    pub sections: [Section; SECTIONS],
}

impl Header {
    pub fn geometry(&self) -> Geometry {
        Geometry::new(self.layout, self.dim, self.codec, self.max_degree)
    }

    /// Expected byte length of every section, or None on overflow.
    pub fn section_lens(&self) -> Option<[u64; SECTIONS]> {
        let g = self.geometry();
        let vectors = match self.layout {
            Layout::Inline => 0,
            Layout::Separate => self.nodes.checked_mul(g.vector_block_bytes as u64)?,
        };
        let id_start = if self.ids == self.nodes { 0 } else { self.nodes.checked_add(1)?.checked_mul(8)? };
        Some([
            g.graph_bytes(self.nodes)?,
            vectors,
            (KSUB * self.dim * 4) as u64,
            self.nodes.checked_mul(self.pq_m as u64)?,
            id_start,
            self.ids.checked_mul(8)?,
        ])
    }

    /// Places the sections one after another, each on a sector boundary.
    /// CRCs are left at 0 for the writer to fill in.
    pub fn place_sections(&mut self) -> Result<u64> {
        let lens = self.section_lens().ok_or_else(|| Error::InvalidArgument("index too large".into()))?;
        let mut at = SECTOR as u64;
        for (s, len) in self.sections.iter_mut().zip(lens) {
            *s = Section { offset: at, len, crc: 0 };
            at = at
                .checked_add(len.div_ceil(SECTOR as u64) * SECTOR as u64)
                .ok_or_else(|| Error::InvalidArgument("index too large".into()))?;
        }
        Ok(at)
    }

    pub fn encode(&self) -> Vec<u8> {
        let g = self.geometry();
        let mut b = vec![0u8; SECTOR];
        b[0..8].copy_from_slice(&MAGIC);
        let put32 = |b: &mut [u8], o: usize, v: u32| b[o..o + 4].copy_from_slice(&v.to_le_bytes());
        put32(&mut b, 8, VERSION);
        put32(&mut b, 12, self.metric.to_code());
        put32(&mut b, 16, self.dim as u32);
        put32(&mut b, 20, self.layout.code());
        put32(&mut b, 24, codec_code(self.codec).expect("writer checks the codec"));
        put32(&mut b, 28, self.max_degree as u32);
        b[32..40].copy_from_slice(&self.nodes.to_le_bytes());
        b[40..48].copy_from_slice(&self.ids.to_le_bytes());
        put32(&mut b, 48, self.entry);
        put32(&mut b, 52, self.pq_m as u32);
        put32(&mut b, 56, g.record_len as u32);
        put32(&mut b, 60, g.records_per_block as u32);
        put32(&mut b, 64, g.block_bytes as u32);
        put32(&mut b, 68, g.vector_block_bytes as u32);
        for (i, s) in self.sections.iter().enumerate() {
            let o = 72 + 24 * i;
            b[o..o + 8].copy_from_slice(&s.offset.to_le_bytes());
            b[o + 8..o + 16].copy_from_slice(&s.len.to_le_bytes());
            put32(&mut b, o + 16, s.crc);
        }
        let crc = crc32c(&b[..HEADER_CRC_AT]);
        put32(&mut b, HEADER_CRC_AT, crc);
        b
    }

    /// Parses and validates the header sector of a file of `file_len` bytes.
    pub fn decode(b: &[u8], file_len: u64) -> Result<Self> {
        let corrupt = |m: &str| Error::Corrupt(format!("SSD index header: {m}"));
        if b.len() < SECTOR {
            return Err(corrupt("file shorter than one sector"));
        }
        if b[0..8] != MAGIC {
            return Err(corrupt("bad magic: not a vecdb SSD index"));
        }
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        if crc32c(&b[..HEADER_CRC_AT]) != u32_at(HEADER_CRC_AT) {
            return Err(corrupt("checksum mismatch"));
        }
        let version = u32_at(8);
        if version != VERSION {
            return Err(Error::Unsupported(format!("SSD index format version {version} (this build reads {VERSION})")));
        }
        let metric = Metric::from_code(u32_at(12)).ok_or_else(|| corrupt("unknown metric"))?;
        let dim = u32_at(16) as usize;
        if dim == 0 || dim > MAX_DIM {
            return Err(corrupt("dimension out of range"));
        }
        let layout = Layout::from_code(u32_at(20)).ok_or_else(|| corrupt("unknown layout"))?;
        let codec = codec_from_code(u32_at(24)).ok_or_else(|| corrupt("unknown vector codec"))?;
        let max_degree = u32_at(28) as usize;
        if max_degree == 0 || max_degree > MAX_DEGREE {
            return Err(corrupt("max degree out of range"));
        }
        let nodes = u64_at(32);
        let ids = u64_at(40);
        let entry = u32_at(48);
        let pq_m = u32_at(52) as usize;
        if nodes == 0 || nodes >= u32::MAX as u64 {
            return Err(corrupt("node count out of range"));
        }
        if ids < nodes {
            return Err(corrupt("fewer ids than nodes"));
        }
        if entry as u64 >= nodes {
            return Err(corrupt("entry node out of range"));
        }
        if pq_m == 0 || pq_m > dim {
            return Err(corrupt("PQ subspace count out of range"));
        }
        let mut h = Header { metric, dim, layout, codec, max_degree, nodes, ids, entry, pq_m, sections: Default::default() };
        let g = h.geometry();
        if [u32_at(56), u32_at(60), u32_at(64), u32_at(68)]
            != [g.record_len as u32, g.records_per_block as u32, g.block_bytes as u32, g.vector_block_bytes as u32]
        {
            return Err(corrupt("record geometry does not match dim, codec and degree"));
        }
        let lens = h.section_lens().ok_or_else(|| corrupt("section sizes overflow"))?;
        let mut prev_end = SECTOR as u64;
        for i in 0..SECTIONS {
            let o = 72 + 24 * i;
            let s = Section { offset: u64_at(o), len: u64_at(o + 8), crc: u32_at(o + 16) };
            let name = SECTION_NAMES[i];
            if s.len != lens[i] {
                return Err(corrupt(&format!("{name} section has the wrong length")));
            }
            if s.offset % SECTOR as u64 != 0 || s.offset < prev_end {
                return Err(corrupt(&format!("{name} section is misplaced")));
            }
            let end = s.offset.checked_add(s.len).ok_or_else(|| corrupt("section end overflows"))?;
            if end > file_len {
                return Err(corrupt(&format!("file is truncated (the {name} section ends past it)")));
            }
            prev_end = end;
            h.sections[i] = s;
        }
        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(layout: Layout, dim: usize, codec: Codec, nodes: u64, ids: u64) -> Header {
        let mut h = Header {
            metric: Metric::L2,
            dim,
            layout,
            codec,
            max_degree: 64,
            nodes,
            ids,
            entry: 3,
            pq_m: 8,
            sections: Default::default(),
        };
        h.place_sections().unwrap();
        h
    }

    #[test]
    fn geometry_by_dimension() {
        // SIFT: 128 x 4 + 4 + 256 + 4 = 776 bytes, five per sector.
        let g = Geometry::new(Layout::Inline, 128, Codec::F32, 64);
        assert_eq!((g.record_len, g.records_per_block, g.block_bytes), (776, 5, 4096));
        // GIST at f32 misses one sector by 8 bytes; f16 fits.
        let g = Geometry::new(Layout::Inline, 960, Codec::F32, 64);
        assert_eq!((g.record_len, g.records_per_block, g.block_bytes), (4104, 1, 8192));
        let g = Geometry::new(Layout::Inline, 960, Codec::F16, 64);
        assert_eq!((g.record_len, g.block_bytes), (2184, 4096));
        // Separate: 268-byte records, 15 per sector; an 8192-dim f32 vector is 8 sectors.
        let g = Geometry::new(Layout::Separate, 8192, Codec::F32, 64);
        assert_eq!((g.record_len, g.records_per_block, g.vector_block_bytes), (268, 15, 32768));
        assert_eq!(g.locate(31), (2 * 4096, 268));
        assert_eq!(g.max_read(), 32768);
    }

    #[test]
    fn auto_layout_by_record_size() {
        assert_eq!(Layout::auto(128, Codec::F32, 64), Layout::Inline);
        assert_eq!(Layout::auto(960, Codec::F32, 64), Layout::Inline);
        assert_eq!(Layout::auto(4030, Codec::F32, 64), Layout::Inline); // 16120 + 264 = 16384 bytes
        assert_eq!(Layout::auto(4031, Codec::F32, 64), Layout::Separate);
        assert_eq!(Layout::auto(8060, Codec::F16, 64), Layout::Inline);
        assert_eq!(Layout::auto(8192, Codec::F16, 64), Layout::Separate);
        assert_eq!(Layout::auto(8192, Codec::F32, 64), Layout::Separate);
    }

    #[test]
    fn vectors_round_trip_through_every_codec() {
        let v: Vec<f32> = (0..37).map(|i| (i as f32 - 18.0) * 0.37).collect();
        for codec in [Codec::F32, Codec::F16, Codec::BF16] {
            let mut bytes = vec![0u8; v.len() * codec.bytes_per_dim()];
            encode_vector(codec, &v, &mut bytes).unwrap();
            let mut back = vec![0f32; v.len()];
            decode_vector(codec, &bytes, &mut back);
            let tol = if codec == Codec::F32 { 0.0 } else { 0.02 };
            for (a, b) in v.iter().zip(&back) {
                assert!((a - b).abs() <= tol * a.abs().max(1.0), "{} {a} {b}", codec.name());
            }
        }
        let mut bytes = vec![0u8; 2];
        assert!(encode_vector(Codec::F16, &[1e6], &mut bytes).is_err());
        assert!(encode_vector(Codec::BF16, &[1e6], &mut bytes).is_ok());
        assert!(encode_vector(Codec::SQ8, &[1.0], &mut bytes).is_err());
    }

    #[test]
    fn header_round_trips() {
        for (layout, codec, ids) in [(Layout::Inline, Codec::F32, 1000), (Layout::Separate, Codec::BF16, 1500)] {
            let h = header(layout, 100, codec, 1000, ids);
            let end = h.sections[IDS].offset + h.sections[IDS].len;
            let b = h.encode();
            assert_eq!(Header::decode(&b, end).unwrap(), h);
            assert_eq!(h.sections[ID_START].len, if ids == 1000 { 0 } else { 1001 * 8 });
        }
    }

    #[test]
    fn header_rejects_damage() {
        let h = header(Layout::Inline, 100, Codec::F32, 1000, 1000);
        let end = h.sections[IDS].offset + h.sections[IDS].len;
        let good = h.encode();
        assert!(matches!(Header::decode(&good, end - 1), Err(Error::Corrupt(_))), "truncated");
        assert!(matches!(Header::decode(&good[..100], end), Err(Error::Corrupt(_))));
        for at in [0usize, 9, 17, 40, 80, 300, 4093] {
            let mut b = good.clone();
            b[at] ^= 0x10;
            assert!(Header::decode(&b, end).is_err(), "flip at {at}");
        }
        // A consistent header (checksum fixed up) with a bad field is still rejected.
        let mut b = good.clone();
        b[48..52].copy_from_slice(&5000u32.to_le_bytes()); // entry >= nodes
        let crc = crc32c(&b[..HEADER_CRC_AT]);
        b[HEADER_CRC_AT..].copy_from_slice(&crc.to_le_bytes());
        assert!(matches!(Header::decode(&b, end), Err(Error::Corrupt(_))));
        let mut b = good;
        b[8..12].copy_from_slice(&2u32.to_le_bytes());
        let crc = crc32c(&b[..HEADER_CRC_AT]);
        b[HEADER_CRC_AT..].copy_from_slice(&crc.to_le_bytes());
        assert!(matches!(Header::decode(&b, end), Err(Error::Unsupported(_))));
    }
}
