//! Writes an SSD index file one node at a time, so the writer never needs
//! the whole graph or every vector in memory.

use std::fs::{self, File};
use std::path::{Path, PathBuf};

use super::format::{
    CODEBOOKS, CODES, GRAPH, Geometry, Header, ID_START, IDS, Layout, VECTORS, codec_code, encode_vector,
};
use super::io::drop_cache;
use crate::crc32c::{Crc32c, crc32c};
use crate::distance::Metric;
use crate::error::{Error, Result};
use crate::format::{sync_parent_dir, tmp_path};
use crate::graph::GraphIndex;
use crate::quant::Codec;

/// How to write an SSD index.
#[derive(Clone, Debug)]
pub struct DiskWriteOptions {
    /// `None` picks by record size ([`Layout::auto`]).
    pub layout: Option<Layout>,
    /// Storage for full vectors: `f32` (exact), or `f16` / `bf16` (half the
    /// bytes; distances are then exact for the rounded vectors).
    pub codec: Codec,
}

impl Default for DiskWriteOptions {
    fn default() -> Self {
        Self { layout: None, codec: Codec::F32 }
    }
}

/// What [`DiskIndex::write`](super::DiskIndex::write) produced.
#[derive(Clone, Debug)]
pub struct DiskWriteReport {
    pub layout: Layout,
    pub codec: Codec,
    pub file_bytes: u64,
    /// SSD bytes per node: its share of a graph block plus, for the separate
    /// layout, its vector block. (Search reads whole blocks: see
    /// [`DiskIndex::block_bytes`](super::DiskIndex::block_bytes).)
    pub ssd_bytes_per_node: f64,
    /// Bytes a search process keeps in RAM: codebooks, codes and ids.
    pub ram_bytes: u64,
}

/// Writes a built in-memory graph.
pub(crate) fn write(graph: &GraphIndex, path: &Path, opts: &DiskWriteOptions) -> Result<DiskWriteReport> {
    let (pq, codes) = graph
        .pq_parts()
        .ok_or_else(|| Error::InvalidArgument("train PQ on the graph before writing an SSD index".into()))?;
    let shape = Shape {
        metric: graph.metric(),
        dim: graph.dim(),
        max_degree: graph.max_degree(),
        nodes: graph.nodes(),
        ids: graph.len(),
        pq_m: pq.code_size(),
    };
    let mut w = IndexWriter::create(path, &shape, opts)?;
    for node in 0..graph.nodes() as u32 {
        w.push(graph.row(node), graph.neighbors(node))?;
    }
    w.finish(graph.entry(), pq.codebooks(), codes, &mut |node, ids| ids.extend_from_slice(graph.ids_of(node)))
}

/// What an index holds, known before its first node is written.
pub(crate) struct Shape {
    pub metric: Metric,
    pub dim: usize,
    pub max_degree: usize,
    pub nodes: usize,
    pub ids: usize,
    pub pq_m: usize,
}

/// The header of an index of this shape (entry and CRCs still unset), and
/// the file size.
pub(crate) fn header(shape: &Shape, opts: &DiskWriteOptions) -> Result<(Header, u64)> {
    if codec_code(opts.codec).is_none() {
        return Err(Error::InvalidArgument("SSD index vectors must be f32, f16 or bf16".into()));
    }
    if shape.nodes == 0 {
        return Err(Error::InvalidArgument("cannot write an empty SSD index".into()));
    }
    let layout = opts.layout.unwrap_or_else(|| Layout::auto(shape.dim, opts.codec, shape.max_degree));
    let mut h = Header {
        metric: shape.metric,
        dim: shape.dim,
        layout,
        codec: opts.codec,
        max_degree: shape.max_degree,
        nodes: shape.nodes as u64,
        ids: shape.ids as u64,
        entry: 0,
        pq_m: shape.pq_m,
        sections: Default::default(),
    };
    let file_bytes = h.place_sections()?;
    Ok((h, file_bytes))
}

/// Buffered writes to one section, at increasing offsets.
struct SectionOut {
    pos: u64,
    buf: Vec<u8>,
}

impl SectionOut {
    const FLUSH: usize = 1 << 20;

    fn at(pos: u64) -> Self {
        Self { pos, buf: Vec::with_capacity(Self::FLUSH) }
    }

    fn put(&mut self, file: &File, b: &[u8]) -> Result<()> {
        self.buf.extend_from_slice(b);
        if self.buf.len() >= Self::FLUSH {
            self.flush(file)?;
        }
        Ok(())
    }

    fn flush(&mut self, file: &File) -> Result<()> {
        write_at(file, &self.buf, self.pos)?;
        self.pos += self.buf.len() as u64;
        self.buf.clear();
        Ok(())
    }
}

/// Writes an index: [`push`](Self::push) every node in order, then
/// [`finish`](Self::finish). The file appears at its path only when
/// `finish` succeeds; on any error, or if the writer is dropped first, the
/// temporary file is removed.
///
/// The graph and vector sections are written as nodes arrive, each at its
/// own offset, so one pass over the vectors fills both.
pub(crate) struct IndexWriter {
    path: PathBuf,
    tmp: PathBuf,
    file: File,
    h: Header,
    g: Geometry,
    file_bytes: u64,
    graph: SectionOut,
    vectors: SectionOut,
    /// The graph block being filled, and how many records it holds.
    block: Vec<u8>,
    in_block: usize,
    /// One encoded vector (the separate layout's vector block).
    vector: Vec<u8>,
    pushed: usize,
    done: bool,
}

impl IndexWriter {
    pub fn create(path: &Path, shape: &Shape, opts: &DiskWriteOptions) -> Result<Self> {
        let (h, file_bytes) = header(shape, opts)?;
        let g = h.geometry();
        let tmp = tmp_path(path);
        let file = File::create(&tmp)?;
        let w = Self {
            path: path.to_path_buf(),
            tmp,
            file,
            graph: SectionOut::at(h.sections[GRAPH].offset),
            vectors: SectionOut::at(h.sections[VECTORS].offset),
            block: vec![0u8; g.block_bytes],
            in_block: 0,
            vector: vec![0u8; g.vector_block_bytes.max(g.vector_bytes)],
            pushed: 0,
            done: false,
            h,
            g,
            file_bytes,
        };
        // Bytes no section covers (padding, the header until `finish`) read as zeros.
        w.file.set_len(file_bytes)?;
        Ok(w)
    }

    /// Appends the next node: its vector (prepared for the metric) and its
    /// out-neighbors.
    pub fn push(&mut self, vector: &[f32], neighbors: &[u32]) -> Result<()> {
        let g = &self.g;
        if vector.len() != self.h.dim || neighbors.len() > g.max_degree || self.pushed as u64 >= self.h.nodes {
            return Err(Error::InvalidArgument(format!(
                "node {}: {} values, {} neighbors (index: dim {}, R {}, {} nodes)",
                self.pushed,
                vector.len(),
                neighbors.len(),
                self.h.dim,
                g.max_degree,
                self.h.nodes
            )));
        }
        let rec = &mut self.block[self.in_block * g.record_len..][..g.record_len];
        fill_record(g, self.h.codec, vector, neighbors, rec, &mut self.vector[..g.vector_bytes])?;
        if g.layout == Layout::Separate {
            // `fill_record` left the encoded vector in `self.vector`; the rest of the block stays zero.
            self.vectors.put(&self.file, &self.vector)?;
        }
        self.pushed += 1;
        self.in_block += 1;
        if self.in_block == g.records_per_block {
            self.graph.put(&self.file, &self.block)?;
            self.block.fill(0);
            self.in_block = 0;
        }
        Ok(())
    }

    /// Writes the RAM sections and the header, then moves the file into
    /// place. `ids_of(node, out)` appends the ids stored at `node`; it is
    /// called in node order, once or twice per node.
    pub fn finish(
        mut self,
        entry: u32,
        codebooks: &[f32],
        codes: &[u8],
        ids_of: &mut dyn FnMut(u32, &mut Vec<u64>),
    ) -> Result<DiskWriteReport> {
        let nodes = self.h.nodes as usize;
        if self.pushed != nodes {
            return Err(Error::InvalidArgument(format!("{} of {nodes} nodes written", self.pushed)));
        }
        if entry as usize >= nodes {
            return Err(Error::InvalidArgument(format!("entry {entry} out of range")));
        }
        if self.in_block > 0 {
            self.graph.put(&self.file, &self.block)?;
        }
        self.graph.flush(&self.file)?;
        self.vectors.flush(&self.file)?;
        self.h.entry = entry;

        let mut ids = Vec::new();
        let mut le_bytes = Vec::with_capacity(SectionOut::FLUSH);
        self.section(CODEBOOKS, &mut |put| {
            for chunk in codebooks.chunks(1024) {
                le_bytes.clear();
                le_bytes.extend(chunk.iter().flat_map(|x| x.to_le_bytes()));
                put(&le_bytes)?;
            }
            Ok(())
        })?;
        self.section(CODES, &mut |put| codes.chunks(1 << 20).try_for_each(&mut *put))?;
        if self.h.ids != self.h.nodes {
            self.section(ID_START, &mut |put| {
                let mut at = 0u64;
                put(&at.to_le_bytes())?;
                for node in 0..nodes as u32 {
                    ids.clear();
                    ids_of(node, &mut ids);
                    at += ids.len() as u64;
                    put(&at.to_le_bytes())?;
                }
                Ok(())
            })?;
        }
        self.section(IDS, &mut |put| {
            for node in 0..nodes as u32 {
                ids.clear();
                ids_of(node, &mut ids);
                le_bytes.clear();
                le_bytes.extend(ids.iter().flat_map(|id| id.to_le_bytes()));
                put(&le_bytes)?;
            }
            Ok(())
        })?;

        write_at(&self.file, &self.h.encode(), 0)?;
        self.file.sync_all()?;
        drop_cache(&self.file);
        fs::rename(&self.tmp, &self.path)?;
        self.done = true;
        sync_parent_dir(&self.path)?;

        let g = &self.g;
        Ok(DiskWriteReport {
            layout: self.h.layout,
            codec: self.h.codec,
            file_bytes: self.file_bytes,
            ssd_bytes_per_node: g.block_bytes as f64 / g.records_per_block as f64 + g.vector_block_bytes as f64,
            ram_bytes: [CODEBOOKS, CODES, ID_START, IDS].iter().map(|&s| self.h.sections[s].len).sum(),
        })
    }

    /// Writes section `i` from the bytes `fill` passes to its callback,
    /// checks its length and records its CRC.
    fn section(
        &mut self,
        i: usize,
        fill: &mut dyn FnMut(&mut dyn FnMut(&[u8]) -> Result<()>) -> Result<()>,
    ) -> Result<()> {
        let mut out = SectionOut::at(self.h.sections[i].offset);
        let mut crc = Crc32c::new();
        let mut len = 0u64;
        let file = &self.file;
        fill(&mut |b: &[u8]| {
            crc.update(b);
            len += b.len() as u64;
            out.put(file, b)
        })?;
        out.flush(file)?;
        if len != self.h.sections[i].len {
            return Err(Error::InvalidArgument(format!(
                "section {i}: {len} bytes written, {} expected",
                self.h.sections[i].len
            )));
        }
        self.h.sections[i].crc = crc.finalize();
        Ok(())
    }
}

impl Drop for IndexWriter {
    fn drop(&mut self) {
        if !self.done {
            let _ = fs::remove_file(&self.tmp);
        }
    }
}

fn write_at(file: &File, bytes: &[u8], offset: u64) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.write_all_at(bytes, offset)?;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < bytes.len() {
            done += file.seek_write(&bytes[done..], offset + done as u64)?;
        }
    }
    Ok(())
}

/// One graph record (see the layouts in `format.rs`). Leaves the encoded
/// vector in `vector`.
fn fill_record(
    g: &Geometry,
    codec: Codec,
    v: &[f32],
    neighbors: &[u32],
    rec: &mut [u8],
    vector: &mut [u8],
) -> Result<()> {
    encode_vector(codec, v, vector)?;
    let mut at = 0;
    if g.layout == Layout::Inline {
        rec[..g.vector_bytes].copy_from_slice(vector);
        at = g.vector_bytes;
    }
    rec[at..at + 4].copy_from_slice(&(neighbors.len() as u32).to_le_bytes());
    for (k, &x) in neighbors.iter().enumerate() {
        rec[at + 4 + 4 * k..][..4].copy_from_slice(&x.to_le_bytes());
    }
    at += 4 + 4 * g.max_degree;
    if g.layout == Layout::Separate {
        rec[at..at + 4].copy_from_slice(&crc32c(vector).to_le_bytes());
        at += 4;
    }
    debug_assert_eq!(at + 4, g.record_len);
    let crc = crc32c(&rec[..at]);
    rec[at..at + 4].copy_from_slice(&crc.to_le_bytes());
    Ok(())
}
