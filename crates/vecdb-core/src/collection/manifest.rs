//! The manifest: the one file that says which other files make up a
//! collection. It is replaced atomically, so every change of the segment
//! set (a flush, an index build, a compaction) commits with one rename.
//!
//! ```text
//! magic "VECDBMAN", version u32 (1)
//! config: dim u32, metric u32, seal_bytes u64, index_bytes u64,
//!         compact_deleted f32, max_indexed_segments u32,
//!         index: max_degree u32, build_list u32, alpha f32, passes u32,
//!                seed u64, pq_m u32, codec u32, layout u32 (0 = auto)
//!         fields: count u16, then per field: name (u16 length + UTF-8), type u8
//! generation u64, flushed_seq u64, wal_gen u64, next_segment u64
//! segments: count u32, then per segment:
//!         id u64, kind u8 (1 flat, 2 SSD index), positions u64,
//!         deleted u64, del_gen u64 (0 = no tombstone file)
//! crc u32: CRC-32C of everything before it
//! ```

use std::path::Path;

use super::bytes::{Dec, Enc, write_file};
use super::meta::{Field, FieldType};
use super::{CollectionConfig, IndexConfig};
use crate::disk::{Layout, codec_code, codec_from_code};
use crate::distance::Metric;
use crate::error::{Error, Result};

pub(crate) const MANIFEST: &str = "MANIFEST";
const MAGIC: [u8; 8] = *b"VECDBMAN";
const VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SegmentKind {
    /// Vectors and ids in a flat segment file, searched exactly in RAM.
    Flat,
    /// An SSD index.
    Disk,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SegmentEntry {
    pub id: u64,
    pub kind: SegmentKind,
    /// Rows (id positions) in the segment, deleted or not.
    pub positions: u64,
    /// Rows deleted in the tombstone file `del_gen`.
    pub deleted: u64,
    /// Generation of the segment's tombstone file; 0 = none yet.
    pub del_gen: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Manifest {
    pub config: CollectionConfig,
    /// Increases with every manifest written.
    pub generation: u64,
    /// Every write with a sequence number up to this one is in the segment
    /// and tombstone files; the WAL holds the writes after it.
    pub flushed_seq: u64,
    pub wal_gen: u64,
    pub next_segment: u64,
    pub segments: Vec<SegmentEntry>,
}

impl Manifest {
    pub fn encode(&self) -> Vec<u8> {
        let mut e = Enc::default();
        e.buf.extend_from_slice(&MAGIC);
        e.u32(VERSION);
        let c = &self.config;
        e.u32(c.dim as u32);
        e.u32(c.metric.to_code());
        e.u64(c.seal_bytes);
        e.u64(c.index_bytes);
        e.f32(c.compact_deleted);
        e.u32(c.max_indexed_segments as u32);
        let i = &c.index;
        e.u32(i.max_degree as u32);
        e.u32(i.build_list as u32);
        e.f32(i.alpha);
        e.u32(i.passes as u32);
        e.u64(i.seed);
        e.u32(i.pq_m as u32);
        e.u32(codec_code(i.codec).expect("validated"));
        e.u32(i.layout.map_or(0, Layout::code));
        e.u16(c.fields.len() as u16);
        for f in &c.fields {
            e.str(&f.name);
            e.u8(f.ty.code());
        }
        e.u64(self.generation);
        e.u64(self.flushed_seq);
        e.u64(self.wal_gen);
        e.u64(self.next_segment);
        e.u32(self.segments.len() as u32);
        for s in &self.segments {
            e.u64(s.id);
            e.u8(match s.kind {
                SegmentKind::Flat => 1,
                SegmentKind::Disk => 2,
            });
            e.u64(s.positions);
            e.u64(s.deleted);
            e.u64(s.del_gen);
        }
        e.crc();
        e.buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut d = Dec::checked(bytes, "manifest")?;
        if d.take(8)? != MAGIC {
            return Err(d.corrupt("bad magic"));
        }
        let version = d.u32()?;
        if version != VERSION {
            return Err(Error::Unsupported(format!("manifest version {version}")));
        }
        let dim = d.u32()? as usize;
        let metric = Metric::from_code(d.u32()?).ok_or_else(|| d.corrupt("unknown metric"))?;
        let seal_bytes = d.u64()?;
        let index_bytes = d.u64()?;
        let compact_deleted = d.f32()?;
        let max_indexed_segments = d.u32()? as usize;
        let index = IndexConfig {
            max_degree: d.u32()? as usize,
            build_list: d.u32()? as usize,
            alpha: d.f32()?,
            passes: d.u32()? as usize,
            seed: d.u64()?,
            pq_m: d.u32()? as usize,
            codec: codec_from_code(d.u32()?).ok_or_else(|| d.corrupt("unknown codec"))?,
            layout: match d.u32()? {
                0 => None,
                c => Some(Layout::from_code(c).ok_or_else(|| d.corrupt("unknown layout"))?),
            },
        };
        let mut fields = Vec::new();
        for _ in 0..d.u16()? {
            let name = d.str()?.to_string();
            let ty = FieldType::from_code(d.u8()?).ok_or_else(|| d.corrupt("unknown field type"))?;
            fields.push(Field { name, ty });
        }
        let config =
            CollectionConfig { dim, metric, seal_bytes, index_bytes, compact_deleted, max_indexed_segments, index, fields };
        config.validate().map_err(|e| d.corrupt(&format!("config: {e}")))?;
        let generation = d.u64()?;
        let flushed_seq = d.u64()?;
        let wal_gen = d.u64()?;
        let next_segment = d.u64()?;
        let mut segments = Vec::new();
        for _ in 0..d.u32()? {
            let id = d.u64()?;
            let kind = match d.u8()? {
                1 => SegmentKind::Flat,
                2 => SegmentKind::Disk,
                _ => return Err(d.corrupt("unknown segment kind")),
            };
            let (positions, deleted, del_gen) = (d.u64()?, d.u64()?, d.u64()?);
            if id >= next_segment || deleted > positions || (del_gen == 0) != (deleted == 0) {
                return Err(d.corrupt("inconsistent segment entry"));
            }
            segments.push(SegmentEntry { id, kind, positions, deleted, del_gen });
        }
        d.done()?;
        Ok(Self { config, generation, flushed_seq, wal_gen, next_segment, segments })
    }

    pub fn read(dir: &Path) -> Result<Self> {
        Self::decode(&std::fs::read(dir.join(MANIFEST))?)
    }

    /// Replaces the manifest atomically and fsyncs the directory.
    pub fn write(&self, dir: &Path) -> Result<()> {
        write_file(&dir.join(MANIFEST), &self.encode(), true)
    }

    /// Writes the manifest under a temporary name, fsyncs it and renames it
    /// into place: the commit point of every change to the segment set.
    /// The caller then fsyncs the directory, which makes the rename (and
    /// the names of the files it refers to) durable.
    pub fn install(&self, dir: &Path) -> Result<()> {
        write_file(&dir.join(MANIFEST), &self.encode(), false)
    }
}
