//! The write-ahead log. Every write batch is appended as one record (and,
//! with [`Durability::Sync`], fsynced) before it changes anything in
//! memory, so a reopen replays exactly the batches that were acknowledged.
//!
//! ```text
//! header (32 bytes): magic "VECDBWAL", version u32 (1), dim u32,
//!                    generation u64, reserved u32, crc u32 of bytes 0..28
//! record:  crc u32   CRC-32C of the rest of the record, from `len` on
//!          len u32   bytes after this field
//!          first_seq u64, count u32, flags u8 (bit 0: written with Sync)
//!          count ops: tag u8, then
//!                     1 upsert: id u64, dim x f32, attrs (u32 length + bytes)
//!                     2 delete: id u64
//! ```
//!
//! A record is one batch, so a batch is replayed whole or not at all.
//! Sequence numbers run on without gaps from the manifest's `flushed_seq`.
//!
//! Recovery is point-in-time: replay stops at the first record that is
//! incomplete or fails its checksum, and the file is cut there. After a
//! crash that is the batch being written. Records written without an fsync
//! (`Buffered`) can also reach the disk out of order, so a bad record
//! followed by good ones is not proof of damage. But a good record written
//! with `Sync` after a bad one is: its fsync made everything before it
//! durable, so the bad record was acknowledged and later damaged. Then the
//! open fails instead of dropping acknowledged writes, naming the byte to
//! cut the log at to recover the writes before it. (The records after a bad
//! one are found through its length field; a damaged length hides them.)

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use super::bytes::{Dec, Enc};
use super::{Durability, WriteOp};
use crate::crc32c::crc32c;
use crate::error::{Error, Result};

const MAGIC: [u8; 8] = *b"VECDBWAL";
const VERSION: u32 = 1;
pub(crate) const HEADER_LEN: usize = 32;
/// Smallest record body: first_seq, count, flags and one delete.
const MIN_BODY: usize = 8 + 4 + 1 + 1 + 8;
const FLAG_SYNC: u8 = 1;
/// Largest batch, as a guard against absurd lengths in a damaged file.
pub(crate) const MAX_BATCH_BYTES: usize = 1 << 30;

const UPSERT: u8 = 1;
const DELETE: u8 = 2;

pub(crate) fn wal_name(generation: u64) -> String {
    format!("wal-{generation:08}.log")
}

/// One write read back from the log.
pub(crate) enum LoggedOp<'a> {
    Upsert { id: u64, vector: &'a [f32], attrs: &'a [u8] },
    Delete { id: u64 },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReplayReport {
    /// Batches and writes replayed.
    pub batches: u64,
    pub ops: u64,
    /// Bytes of the log read.
    pub bytes: u64,
    /// Bytes cut from the end of the log: an incomplete or damaged record
    /// and anything after it.
    pub cut_bytes: u64,
}

pub(crate) struct Wal {
    file: File,
    path: PathBuf,
    /// Length of the valid log.
    len: u64,
    batches: u64,
    /// Set when a write or fsync failed: the file's state is unknown, so no
    /// further writes are accepted until the collection is reopened.
    poisoned: bool,
    buf: Vec<u8>,
}

impl Wal {
    /// Creates an empty log, fsynced (the caller fsyncs the directory).
    pub fn create(dir: &Path, generation: u64, dim: usize) -> Result<Self> {
        let path = dir.join(wal_name(generation));
        let mut file = OpenOptions::new().read(true).write(true).create_new(true).open(&path)?;
        let mut e = Enc::default();
        e.buf.extend_from_slice(&MAGIC);
        e.u32(VERSION);
        e.u32(dim as u32);
        e.u64(generation);
        e.u32(0);
        e.crc();
        file.write_all(&e.buf)?;
        file.sync_all()?;
        Ok(Self {
            file,
            path,
            len: HEADER_LEN as u64,
            batches: 0,
            poisoned: false,
            buf: Vec::new(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn bytes(&self) -> u64 {
        self.len
    }

    /// Refuses further writes (until the collection is reopened).
    pub fn poison(&mut self) {
        self.poisoned = true;
    }

    /// Batches in the log.
    pub fn batches(&self) -> u64 {
        self.batches
    }

    /// Encoded size of `ops` as one record.
    pub fn record_bytes(dim: usize, ops: &[WriteOp]) -> usize {
        8 + 13
            + ops
                .iter()
                .map(|op| match op {
                    WriteOp::Upsert { attrs, .. } => 1 + 8 + dim * 4 + 4 + attrs.encoded_len(),
                    WriteOp::Delete { .. } => 1 + 8,
                })
                .sum::<usize>()
    }

    /// Appends `ops` as one record, numbered from `first_seq`. On error the
    /// log is cut back to where it was; if even that fails, or an fsync
    /// fails, the log refuses further writes.
    pub fn append(&mut self, first_seq: u64, ops: &[WriteOp], durability: Durability) -> Result<()> {
        if self.poisoned {
            return Err(Error::InvalidArgument("an earlier WAL write failed: reopen the collection".into()));
        }
        let mut e = Enc { buf: std::mem::take(&mut self.buf) };
        e.buf.clear();
        e.u32(0); // crc, filled below
        e.u32(0); // len
        e.u64(first_seq);
        e.u32(ops.len() as u32);
        e.u8(if durability == Durability::Sync { FLAG_SYNC } else { 0 });
        for op in ops {
            match op {
                WriteOp::Upsert { id, vector, attrs } => {
                    e.u8(UPSERT);
                    e.u64(*id);
                    e.f32s(vector);
                    let at = e.buf.len();
                    e.u32(0);
                    attrs.encode(&mut e);
                    let n = (e.buf.len() - at - 4) as u32;
                    e.buf[at..at + 4].copy_from_slice(&n.to_le_bytes());
                }
                WriteOp::Delete { id } => {
                    e.u8(DELETE);
                    e.u64(*id);
                }
            }
        }
        let len = (e.buf.len() - 8) as u32;
        e.buf[4..8].copy_from_slice(&len.to_le_bytes());
        let crc = crc32c(&e.buf[4..]);
        e.buf[0..4].copy_from_slice(&crc.to_le_bytes());
        self.buf = e.buf;

        let result = (|| -> std::io::Result<()> {
            self.file.seek(SeekFrom::Start(self.len))?;
            self.file.write_all(&self.buf)
        })();
        if let Err(err) = result {
            // Cut a partial record, so later records do not follow it.
            if self.file.set_len(self.len).is_err() {
                self.poisoned = true;
            }
            return Err(err.into());
        }
        if durability == Durability::Sync
            && let Err(err) = self.file.sync_data()
        {
            // After a failed fsync the kernel may have dropped the pages:
            // nothing about the file can be trusted any more.
            self.poisoned = true;
            return Err(err.into());
        }
        self.len += self.buf.len() as u64;
        self.batches += 1;
        Ok(())
    }

    /// Opens log `generation` and calls `apply` on every write in it, in
    /// order, with its sequence number. The first record must start at
    /// `next_seq`. Cuts an incomplete or damaged tail (see the module docs).
    pub fn replay(
        dir: &Path,
        generation: u64,
        dim: usize,
        next_seq: u64,
        mut apply: impl FnMut(u64, LoggedOp<'_>) -> Result<()>,
    ) -> Result<(Self, ReplayReport)> {
        let path = dir.join(wal_name(generation));
        let mut file = OpenOptions::new().read(true).write(true).open(&path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        let corrupt = |m: String| Error::Corrupt(format!("WAL {}: {m}", path.display()));
        if bytes.len() < HEADER_LEN {
            return Err(corrupt("header truncated".into()));
        }
        let mut h = Dec::checked(&bytes[..HEADER_LEN], "WAL header")?;
        if h.take(8)? != MAGIC {
            return Err(corrupt("bad magic".into()));
        }
        let version = h.u32()?;
        if version != VERSION {
            return Err(Error::Unsupported(format!("WAL version {version}")));
        }
        let (file_dim, file_gen) = (h.u32()? as usize, h.u64()?);
        if file_dim != dim || file_gen != generation {
            return Err(corrupt(format!("written for dim {file_dim}, generation {file_gen}")));
        }

        let mut report = ReplayReport::default();
        let mut pos = HEADER_LEN;
        let mut seq = next_seq;
        let mut vector = vec![0.0f32; dim];
        while let Some(body) = record_at(&bytes, pos) {
            let mut d = Dec::new(body, "WAL record");
            let first = d.u64()?;
            let count = d.u32()? as u64;
            d.u8()?; // flags
            if first != seq || count == 0 {
                // A record with a good checksum in the wrong place: not a
                // torn write.
                return Err(corrupt(format!("record at byte {pos} starts at write {first}, expected {seq}")));
            }
            for s in first..first + count {
                match d.u8()? {
                    UPSERT => {
                        let id = d.u64()?;
                        d.f32s(&mut vector)?;
                        let n = d.u32()? as usize;
                        let attrs = d.take(n)?;
                        apply(s, LoggedOp::Upsert { id, vector: &vector, attrs })?;
                    }
                    DELETE => apply(s, LoggedOp::Delete { id: d.u64()? })?,
                    t => return Err(corrupt(format!("unknown op {t} at write {s}"))),
                }
            }
            d.done()?;
            seq += count;
            report.batches += 1;
            report.ops += count;
            pos += 8 + body.len();
        }
        report.bytes = pos as u64;
        report.cut_bytes = (bytes.len() - pos) as u64;
        if synced_record_after(&bytes, pos) {
            return Err(corrupt(format!(
                "the record at byte {pos} is damaged and a write acknowledged with Sync follows it; \
                 cutting the log at byte {pos} recovers the writes before it"
            )));
        }
        if report.cut_bytes > 0 {
            file.set_len(pos as u64)?;
            file.sync_all()?;
        }
        let wal = Self {
            file,
            path,
            len: pos as u64,
            batches: report.batches,
            poisoned: false,
            buf: Vec::new(),
        };
        Ok((wal, report))
    }
}

/// Whether any good record after the bad record at `pos` was written with
/// `Sync`. The records after it are found through its length field, so a
/// damaged length hides them.
fn synced_record_after(bytes: &[u8], pos: usize) -> bool {
    let rest = &bytes[pos..];
    if rest.len() < 8 {
        return false;
    }
    let len = u32::from_le_bytes(rest[4..8].try_into().unwrap()) as usize;
    if !(MIN_BODY..=MAX_BATCH_BYTES).contains(&len) {
        return false;
    }
    let mut next = pos + 8 + len;
    while next < bytes.len()
        && let Some(body) = record_at(bytes, next)
    {
        if body[12] & FLAG_SYNC != 0 {
            return true;
        }
        next += 8 + body.len();
    }
    false
}

/// The body of the record at `pos` if it is complete and its checksum
/// holds; `None` at the end of the log or at a torn or damaged record.
fn record_at(bytes: &[u8], pos: usize) -> Option<&[u8]> {
    let rest = &bytes[pos..];
    if rest.len() < 8 {
        return None;
    }
    let crc = u32::from_le_bytes(rest[0..4].try_into().unwrap());
    let len = u32::from_le_bytes(rest[4..8].try_into().unwrap()) as usize;
    // A zero-filled tail has len 0 (and CRC 0, the CRC of nothing): the
    // length check rejects it before the checksum could pass.
    if !(MIN_BODY..=MAX_BATCH_BYTES).contains(&len) || rest.len() < 8 + len {
        return None;
    }
    if crc32c(&rest[4..8 + len]) != crc {
        return None;
    }
    Some(&rest[8..8 + len])
}
