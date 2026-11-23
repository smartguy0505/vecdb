//! Writes an SSD index file from a built in-memory graph.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use super::format::{
    CODEBOOKS, CODES, GRAPH, Geometry, Header, ID_START, IDS, Layout, SECTOR, VECTORS, codec_code, encode_vector,
};
use super::io::drop_cache;
use crate::crc32c::{Crc32c, crc32c};
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

pub(crate) fn write(graph: &GraphIndex, path: &Path, opts: &DiskWriteOptions) -> Result<DiskWriteReport> {
    let (pq, codes) = graph
        .pq_parts()
        .ok_or_else(|| Error::InvalidArgument("train PQ on the graph before writing an SSD index".into()))?;
    if codec_code(opts.codec).is_none() {
        return Err(Error::InvalidArgument("SSD index vectors must be f32, f16 or bf16".into()));
    }
    if graph.nodes() == 0 {
        return Err(Error::InvalidArgument("cannot write an empty SSD index".into()));
    }
    let layout = opts.layout.unwrap_or_else(|| Layout::auto(graph.dim(), opts.codec, graph.max_degree()));
    let mut h = Header {
        metric: graph.metric(),
        dim: graph.dim(),
        layout,
        codec: opts.codec,
        max_degree: graph.max_degree(),
        nodes: graph.nodes() as u64,
        ids: graph.len() as u64,
        entry: graph.entry(),
        pq_m: pq.code_size(),
        sections: Default::default(),
    };
    let file_bytes = h.place_sections()?;

    let tmp = tmp_path(path);
    let result = write_to(&tmp, graph, &mut h, pq.codebooks(), codes).and_then(|()| {
        fs::rename(&tmp, path)?;
        sync_parent_dir(path)
    });
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result?;

    let g = h.geometry();
    let ram_bytes = [CODEBOOKS, CODES, ID_START, IDS].iter().map(|&s| h.sections[s].len).sum();
    Ok(DiskWriteReport {
        layout,
        codec: opts.codec,
        file_bytes,
        ssd_bytes_per_node: g.block_bytes as f64 / g.records_per_block as f64 + g.vector_block_bytes as f64,
        ram_bytes,
    })
}

/// Tracks the write position so each section can start where the header
/// says, and checksums the RAM sections as they are written.
struct Out<'a> {
    w: BufWriter<&'a File>,
    pos: u64,
}

impl Out<'_> {
    fn put(&mut self, b: &[u8]) -> Result<()> {
        self.w.write_all(b)?;
        self.pos += b.len() as u64;
        Ok(())
    }

    /// Zero-fills up to `offset`, where the next section starts.
    fn seek_to(&mut self, offset: u64) -> Result<()> {
        assert!(self.pos <= offset, "sections written out of order");
        let zeros = [0u8; SECTOR];
        while self.pos < offset {
            let n = ((offset - self.pos) as usize).min(SECTOR);
            self.put(&zeros[..n])?;
        }
        Ok(())
    }
}

fn write_to(tmp: &Path, graph: &GraphIndex, h: &mut Header, codebooks: &[f32], codes: &[u8]) -> Result<()> {
    let g = h.geometry();
    let nodes = graph.nodes();
    let file = File::create(tmp)?;
    let mut out = Out { w: BufWriter::with_capacity(1 << 20, &file), pos: 0 };
    out.put(&[0u8; SECTOR])?; // header placeholder, written last

    // Graph records, packed into blocks.
    out.seek_to(h.sections[GRAPH].offset)?;
    let mut block = vec![0u8; g.block_bytes];
    let mut vector = vec![0u8; g.vector_bytes];
    for first in (0..nodes).step_by(g.records_per_block) {
        block.fill(0);
        let last = (first + g.records_per_block).min(nodes);
        for (j, node) in (first..last).enumerate() {
            let rec = &mut block[j * g.record_len..][..g.record_len];
            fill_record(graph, &g, h, node as u32, rec, &mut vector)?;
        }
        out.put(&block)?;
    }

    // Vector blocks (separate layout).
    if g.vector_block_bytes > 0 {
        out.seek_to(h.sections[VECTORS].offset)?;
        let mut vb = vec![0u8; g.vector_block_bytes];
        for node in 0..nodes as u32 {
            encode_vector(h.codec, graph.row(node), &mut vb[..g.vector_bytes])?;
            out.put(&vb)?;
        }
    }

    // RAM sections, each with a CRC in the header.
    let mut section = |out: &mut Out, i: usize, bytes: &mut dyn FnMut(&mut dyn FnMut(&[u8]) -> Result<()>) -> Result<()>| {
        out.seek_to(h.sections[i].offset)?;
        let start = out.pos;
        let mut crc = Crc32c::new();
        bytes(&mut |b: &[u8]| {
            crc.update(b);
            out.put(b)
        })?;
        debug_assert_eq!(out.pos - start, h.sections[i].len);
        h.sections[i].crc = crc.finalize();
        Ok::<(), Error>(())
    };
    section(&mut out, CODEBOOKS, &mut |put| {
        for chunk in codebooks.chunks(1024) {
            let b: Vec<u8> = chunk.iter().flat_map(|x| x.to_le_bytes()).collect();
            put(&b)?;
        }
        Ok(())
    })?;
    section(&mut out, CODES, &mut |put| codes.chunks(1 << 20).try_for_each(&mut *put))?;
    if graph.len() != nodes {
        section(&mut out, ID_START, &mut |put| {
            let mut at = 0u64;
            put(&at.to_le_bytes())?;
            for node in 0..nodes as u32 {
                at += graph.ids_of(node).len() as u64;
                put(&at.to_le_bytes())?;
            }
            Ok(())
        })?;
    }
    section(&mut out, IDS, &mut |put| {
        for node in 0..nodes as u32 {
            for id in graph.ids_of(node) {
                put(&id.to_le_bytes())?;
            }
        }
        Ok(())
    })?;
    let end = h.sections[IDS].offset + h.sections[IDS].len;
    out.seek_to(end.div_ceil(SECTOR as u64) * SECTOR as u64)?;
    out.w.flush()?;
    drop(out);

    write_header(&file, &h.encode())?;
    file.sync_all()?;
    drop_cache(&file);
    Ok(())
}

fn write_header(file: &File, header: &[u8]) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.write_all_at(header, 0)?;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < header.len() {
            done += file.seek_write(&header[done..], done as u64)?;
        }
    }
    Ok(())
}

/// One graph record: see the layouts in `format.rs`.
fn fill_record(graph: &GraphIndex, g: &Geometry, h: &Header, node: u32, rec: &mut [u8], vector: &mut [u8]) -> Result<()> {
    let mut at = 0;
    if g.layout == Layout::Inline {
        encode_vector(h.codec, graph.row(node), &mut rec[..g.vector_bytes])?;
        at = g.vector_bytes;
    }
    let nb = graph.neighbors(node);
    rec[at..at + 4].copy_from_slice(&(nb.len() as u32).to_le_bytes());
    for (k, &x) in nb.iter().enumerate() {
        rec[at + 4 + 4 * k..][..4].copy_from_slice(&x.to_le_bytes());
    }
    at += 4 + 4 * g.max_degree;
    if g.layout == Layout::Separate {
        encode_vector(h.codec, graph.row(node), vector)?;
        rec[at..at + 4].copy_from_slice(&crc32c(vector).to_le_bytes());
        at += 4;
    }
    debug_assert_eq!(at + 4, g.record_len);
    let crc = crc32c(&rec[..at]);
    rec[at..at + 4].copy_from_slice(&crc.to_le_bytes());
    Ok(())
}
