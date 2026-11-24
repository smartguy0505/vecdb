//! The write-ahead log. Every write batch is appended as one record (and,
//! with [`Durability::Sync`], fsynced) before it changes anything in
//! memory, so a reopen replays exactly the batches that were acknowledged.
//!
//! ```text
//! header (32 bytes): magic "VECDBWAL", version u32 (1), dim u32,
//!                    generation u64, reserved u32, crc u32 of bytes 0..28
//! record:  crc u32   CRC-32C of the rest of the record, from `len` on
//!          len u32   bytes after this field
//!          first_seq u64, count u32, flags u8 (bit 0: written with Sync;
//!                     bit 1: written in a group, not its first record)
//!          count ops: tag u8, then
//!                     1 upsert: id u64, dim x f32, attrs (u32 length + bytes)
//!                     2 delete: id u64
//! ```
//!
//! A record is one batch, so a batch is replayed whole or not at all.
//! Sequence numbers run on without gaps from the manifest's `flushed_seq`.
//!
//! Each time the growing segment is frozen a new log is started, and the
//! old one is kept until the frozen rows are in a segment file. So there
//! may be several: the manifest names the first one still needed, and they
//! are replayed in order as one ([`replay_chain`]).
//!
//! Recovery is point-in-time: replay stops at the first record that is
//! incomplete or fails its checksum, and the file is cut there. After a
//! crash that is the batch being written. Records written without an fsync
//! (`Buffered`) can also reach the disk out of order, so a bad record
//! followed by good ones is not proof of damage. But a good record written
//! with `Sync` after a bad one is: its fsync made everything before it
//! durable, so the bad record was acknowledged and later damaged. (Batches
//! written together share one fsync, so only a record that starts a later
//! group counts.) Then the
//! open fails instead of dropping acknowledged writes, naming the byte to
//! cut the log at to recover the writes before it. (The records after a bad
//! one are found through its length field; a damaged length hides them.)
//!
//! The same holds from one log to the next. Buffered writes at the end of a
//! log can be lost while the next log, written later, survives: the next
//! log then starts past where the first ends, and it and those after it are
//! dropped. If any of them holds a `Sync` record, though, the lost writes
//! were durable before it, so something damaged them, and the open fails.

use std::fs::{File, OpenOptions};
#[cfg(test)]
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use super::bytes::{Dec, Enc};
use super::{Durability, WriteOp};
use crate::crc32c::crc32c;
use crate::error::{Error, Result};
use crate::format::sync_parent_dir;

const MAGIC: [u8; 8] = *b"VECDBWAL";
const VERSION: u32 = 1;
pub(crate) const HEADER_LEN: usize = 32;
/// Smallest record body: first_seq, count, flags and one delete.
const MIN_BODY: usize = 8 + 4 + 1 + 1 + 8;
const FLAG_SYNC: u8 = 1;
/// Not the first record of the group it was written (and fsynced) with.
const FLAG_IN_GROUP: u8 = 2;
/// Largest batch, as a guard against absurd lengths in a damaged file.
pub(crate) const MAX_BATCH_BYTES: usize = 1 << 30;

const UPSERT: u8 = 1;
const DELETE: u8 = 2;

pub(crate) fn wal_name(generation: u64) -> String {
    format!("wal-{generation:08}.log")
}

/// The generation in a log's file name (`None` for other names).
pub(crate) fn wal_generation(name: &str) -> Option<u64> {
    let digits = name.strip_prefix("wal-")?.strip_suffix(".log")?;
    (digits.len() == 8 && digits.bytes().all(|b| b.is_ascii_digit())).then(|| digits.parse().ok())?
}

/// One write read back from the log.
pub(crate) enum LoggedOp<'a> {
    Upsert { id: u64, vector: &'a [f32], attrs: &'a [u8] },
    Delete { id: u64 },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReplayReport {
    /// Logs replayed.
    pub logs: u64,
    /// Batches and writes replayed.
    pub batches: u64,
    pub ops: u64,
    /// Bytes of the log read.
    pub bytes: u64,
    /// Bytes dropped: an incomplete or damaged record and anything after
    /// it, and logs whose preceding writes were lost.
    pub cut_bytes: u64,
}

pub(crate) struct Wal {
    file: File,
    path: PathBuf,
    generation: u64,
    /// Length of the valid log.
    len: u64,
    /// Length known to be durable: fsynced, or written by an fsynced create.
    synced: u64,
    batches: u64,
    /// Set when a write or fsync failed: the file's state is unknown, so no
    /// further writes are accepted until the collection is reopened.
    poisoned: bool,
    buf: Vec<u8>,
}

impl Wal {
    /// Creates an empty log. With `sync` it is fsynced (the caller fsyncs
    /// the directory); without, a power loss may leave no log or one
    /// without a header, which a reopen takes for the end of the chain.
    pub fn create(dir: &Path, generation: u64, dim: usize, sync: bool) -> Result<Self> {
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
        if sync {
            file.sync_all()?;
        }
        Ok(Self {
            file,
            path,
            generation,
            len: HEADER_LEN as u64,
            synced: if sync { HEADER_LEN as u64 } else { 0 },
            batches: 0,
            poisoned: false,
            buf: Vec::new(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn generation(&self) -> u64 {
        self.generation
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

    /// Encodes `ops` as the WAL records them (a record's body after its
    /// header), appending to `out`.
    pub fn encode_ops(ops: &[WriteOp], out: &mut Vec<u8>) {
        let mut e = Enc { buf: std::mem::take(out) };
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
        *out = e.buf;
    }

    /// Appends a group of batches, each as one record: `(first_seq, count,
    /// ops encoded by encode_ops)`. They are written together and, with
    /// `Sync`, fsynced once. Records after the first are marked as part of
    /// a group: until the fsync they may reach the disk in any order. On
    /// error the log is cut back to where it was; if even that fails, or an
    /// fsync fails, the log refuses further writes.
    pub fn append_group(&mut self, batches: &[(u64, u32, &[u8])], durability: Durability) -> Result<()> {
        if self.poisoned {
            return Err(Error::InvalidArgument("an earlier WAL write failed: reopen the collection".into()));
        }
        let mut buf = std::mem::take(&mut self.buf);
        buf.clear();
        for (i, &(first_seq, count, ops)) in batches.iter().enumerate() {
            let start = buf.len();
            let mut flags = if durability == Durability::Sync { FLAG_SYNC } else { 0 };
            if i > 0 {
                flags |= FLAG_IN_GROUP;
            }
            buf.extend_from_slice(&[0; 8]); // crc, len: filled below
            buf.extend_from_slice(&first_seq.to_le_bytes());
            buf.extend_from_slice(&count.to_le_bytes());
            buf.push(flags);
            buf.extend_from_slice(ops);
            let len = (buf.len() - start - 8) as u32;
            buf[start + 4..start + 8].copy_from_slice(&len.to_le_bytes());
            let crc = crc32c(&buf[start + 4..]);
            buf[start..start + 4].copy_from_slice(&crc.to_le_bytes());
        }
        self.buf = buf;

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
        if durability == Durability::Sync {
            if let Err(err) = self.file.sync_data() {
                // After a failed fsync the kernel may have dropped the pages:
                // nothing about the file can be trusted any more.
                self.poisoned = true;
                return Err(err.into());
            }
        }
        self.len += self.buf.len() as u64;
        if durability == Durability::Sync {
            self.synced = self.len;
        }
        self.batches += batches.len() as u64;
        Ok(())
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// If records (or the header) may not be durable yet: the file, to
    /// fsync without holding the log, and the length that makes durable
    /// (then see [`mark_synced`](Self::mark_synced)). `None` under `Sync`.
    pub fn unsynced(&self) -> Result<Option<(File, u64)>> {
        Ok(if self.synced < self.len { Some((self.file.try_clone()?, self.len)) } else { None })
    }

    /// Records that the first `len` bytes were fsynced.
    pub fn mark_synced(&mut self, len: u64) {
        self.synced = self.synced.max(len);
    }

    /// Length of the log known to be durable.
    #[cfg(test)]
    pub fn synced(&self) -> u64 {
        self.synced
    }
}

/// What [`replay_chain`] hands its caller, in order.
pub(crate) enum Event<'a> {
    /// A write and its sequence number.
    Op(u64, LoggedOp<'a>),
    /// The end of a log that another one follows.
    EndOfLog { generation: u64 },
}

/// The logs of a collection, replayed.
pub(crate) struct Chain {
    /// All logs but the last, oldest first.
    pub closed: Vec<Wal>,
    /// The last one: new writes go there.
    pub active: Wal,
    pub report: ReplayReport,
    /// Names of logs dropped because the writes before them were lost.
    pub dropped: Vec<String>,
}

/// Replays the logs from generation `first` on, in order, as one sequence
/// of writes starting at `next_seq`. Cuts an incomplete or damaged tail and
/// drops the logs after lost writes (see the module docs). Everything kept
/// is fsynced, and so is the directory. No file is changed before all of them have been checked.
pub(crate) fn replay_chain(
    dir: &Path,
    first: u64,
    dim: usize,
    next_seq: u64,
    mut on: impl FnMut(Event<'_>) -> Result<()>,
) -> Result<Chain> {
    let mut gens: Vec<u64> = Vec::new();
    for e in std::fs::read_dir(dir)? {
        if let Some(g) = e?.file_name().to_str().and_then(wal_generation)
            && g >= first
        {
            gens.push(g);
        }
    }
    gens.sort_unstable();
    if gens.first() != Some(&first) {
        return Err(Error::Corrupt(format!("{} is missing in {}", wal_name(first), dir.display())));
    }
    let mut kept: Vec<(Loaded, End)> = Vec::new();
    let mut dropped = Vec::new();
    let mut report = ReplayReport::default();
    let mut seq = next_seq;
    let mut cur = Loaded::read(dir, first, dim)?;
    for (i, &g) in gens.iter().enumerate() {
        let end = cur.replay(seq, dim, |s, op| on(Event::Op(s, op)))?;
        seq += end.ops;
        report.logs += 1;
        report.batches += end.batches;
        report.ops += end.ops;
        report.bytes += end.pos as u64;
        report.cut_bytes += (cur.bytes.len() - end.pos) as u64;
        let Some(&g2) = gens.get(i + 1) else {
            kept.push((cur, end));
            break;
        };
        let next = if g2 == g + 1 {
            Loaded::read(dir, g2, dim)
        } else {
            Err(Error::Corrupt(format!("{} is missing", wal_name(g + 1))))
        };
        match next {
            Ok(next) if next.first_seq().is_none_or(|f| f == seq) => {
                if end.ops > 0 {
                    on(Event::EndOfLog { generation: g })?;
                }
                let mut log = std::mem::replace(&mut cur, next);
                // Only its length is needed from here on.
                log.bytes = Vec::new();
                kept.push((log, end));
            }
            Err(e @ (Error::Io(_) | Error::Unsupported(_))) => return Err(e),
            next => {
                // This log ends before the next one starts.
                if let Ok(n) = &next
                    && n.first_seq().is_some_and(|f| f < seq)
                {
                    return Err(n.corrupt(format!("starts before write {seq}, where {} ends", wal_name(g))));
                }
                for &g in &gens[i + 1..] {
                    let bytes = std::fs::read(dir.join(wal_name(g)))?;
                    if holds_synced_record(&bytes) {
                        return Err(cur.corrupt(format!(
                            "it ends at byte {} with write {}, but {} holds later writes acknowledged with Sync; \
                             cutting it at byte {0} and removing the logs after it recovers the writes before",
                            end.pos,
                            seq - 1,
                            wal_name(g)
                        )));
                    }
                    report.cut_bytes += bytes.len() as u64;
                    dropped.push(wal_name(g));
                }
                kept.push((cur, end));
                break;
            }
        }
    }
    // Gone for good before any write reuses their numbers.
    for name in &dropped {
        std::fs::remove_file(dir.join(name))?;
    }
    let mut logs =
        (first..).zip(kept).map(|(g, (log, end))| log.into_wal(g, &end)).collect::<Result<Vec<Wal>>>()?;
    // Also for the names of logs created without an fsync.
    sync_parent_dir(&dir.join(wal_name(first)))?;
    let active = logs.pop().expect("the first log");
    Ok(Chain { closed: logs, active, report, dropped })
}

/// A log read into memory.
struct Loaded {
    file: File,
    path: PathBuf,
    bytes: Vec<u8>,
    /// Length of the file.
    len: usize,
}

/// Where a log's good records end.
struct End {
    pos: usize,
    batches: u64,
    ops: u64,
}

impl Loaded {
    /// Reads log `generation` and checks its header.
    fn read(dir: &Path, generation: u64, dim: usize) -> Result<Self> {
        let path = dir.join(wal_name(generation));
        let mut file = OpenOptions::new().read(true).write(true).open(&path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        let log = Self { file, path, len: bytes.len(), bytes };
        if log.bytes.len() < HEADER_LEN {
            return Err(log.corrupt("header truncated".into()));
        }
        let mut h = Dec::checked(&log.bytes[..HEADER_LEN], "WAL header")?;
        if h.take(8)? != MAGIC {
            return Err(log.corrupt("bad magic".into()));
        }
        let version = h.u32()?;
        if version != VERSION {
            return Err(Error::Unsupported(format!("WAL version {version}")));
        }
        let (file_dim, file_gen) = (h.u32()? as usize, h.u64()?);
        if file_dim != dim || file_gen != generation {
            return Err(log.corrupt(format!("written for dim {file_dim}, generation {file_gen}")));
        }
        Ok(log)
    }

    fn corrupt(&self, m: String) -> Error {
        Error::Corrupt(format!("WAL {}: {m}", self.path.display()))
    }

    /// The first write in the log, if its first record is good.
    fn first_seq(&self) -> Option<u64> {
        let body = record_at(&self.bytes, HEADER_LEN)?;
        Some(u64::from_le_bytes(body[..8].try_into().unwrap()))
    }

    /// Calls `apply` on every write in the log, in order, with its sequence
    /// number. The first record must start at `next_seq`.
    fn replay(&self, next_seq: u64, dim: usize, mut apply: impl FnMut(u64, LoggedOp<'_>) -> Result<()>) -> Result<End> {
        let bytes = &self.bytes;
        let mut end = End { pos: HEADER_LEN, batches: 0, ops: 0 };
        let mut vector = vec![0.0f32; dim];
        while let Some(body) = record_at(bytes, end.pos) {
            let pos = end.pos;
            let mut d = Dec::new(body, "WAL record");
            let first = d.u64()?;
            let count = d.u32()? as u64;
            d.u8()?; // flags
            let seq = next_seq + end.ops;
            if first != seq || count == 0 {
                // A record with a good checksum in the wrong place: not a
                // torn write.
                return Err(self.corrupt(format!("record at byte {pos} starts at write {first}, expected {seq}")));
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
                    t => return Err(self.corrupt(format!("unknown op {t} at write {s}"))),
                }
            }
            d.done()?;
            end.batches += 1;
            end.ops += count;
            end.pos += 8 + body.len();
        }
        if synced_record_after(bytes, end.pos) {
            return Err(self.corrupt(format!(
                "the record at byte {} is damaged and a write acknowledged with Sync follows it; \
                 cutting the log at byte {0} recovers the writes before it",
                end.pos
            )));
        }
        Ok(end)
    }

    /// The log, cut after its last good record and made durable: records
    /// written without an fsync may be only in the page cache, and could
    /// otherwise be lost behind a later `Sync` record.
    fn into_wal(self, generation: u64, end: &End) -> Result<Wal> {
        if end.pos < self.len {
            self.file.set_len(end.pos as u64)?;
        }
        self.file.sync_all()?;
        Ok(Wal {
            file: self.file,
            path: self.path,
            generation,
            len: end.pos as u64,
            synced: end.pos as u64,
            batches: end.batches,
            poisoned: false,
            buf: Vec::new(),
        })
    }
}

/// Whether a log (its header may be damaged) holds any record written with
/// `Sync`. Records after a bad one are found through its length field.
fn holds_synced_record(bytes: &[u8]) -> bool {
    let mut pos = HEADER_LEN;
    while pos + 8 <= bytes.len() {
        if let Some(body) = record_at(bytes, pos) {
            if body[12] & FLAG_SYNC != 0 {
                return true;
            }
            pos += 8 + body.len();
        } else {
            let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
            if !(MIN_BODY..=MAX_BATCH_BYTES).contains(&len) {
                return false;
            }
            pos += 8 + len;
        }
    }
    false
}

/// Whether a good record after the bad record at `pos` proves that the bad
/// one was acknowledged: one that was written with `Sync` and starts a
/// group. Its group was written after the bad record's group was fsynced;
/// records of the bad record's own group prove nothing, as they share its
/// fsync. The records after it are found through its length field, so a
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
        if body[12] & FLAG_SYNC != 0 && body[12] & FLAG_IN_GROUP == 0 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disk::test_dir;

    /// Records of one group share an fsync, so a damaged record followed
    /// only by its own group's records may be a torn write: the log is
    /// cut. A later group, written after the damaged one's fsync, proves
    /// the damage happened after it was acknowledged: the open fails.
    #[test]
    fn only_a_later_group_proves_damage() {
        let dir = test_dir("wal-groups");
        let del = |id: u64| {
            let mut b = Vec::new();
            Wal::encode_ops(&[WriteOp::Delete { id }], &mut b);
            b
        };
        let (a, b, c, d) = (del(1), del(2), del(3), del(4));
        let mut wal = Wal::create(&dir, 1, 4, true).unwrap();
        wal.append_group(&[(1, 1, &a), (2, 1, &b), (3, 1, &c)], Durability::Sync).unwrap();
        let one_group = wal.bytes();
        wal.append_group(&[(4, 1, &d)], Durability::Sync).unwrap();
        drop(wal);
        let path = dir.join(wal_name(1));
        let good = fs::read(&path).unwrap();
        let rec = (good.len() - HEADER_LEN) / 4;
        let replay = |bytes: &[u8]| {
            fs::write(&path, bytes).unwrap();
            let mut ids = Vec::new();
            let r = replay_chain(&dir, 1, 4, 1, |e| {
                if let Event::Op(_, LoggedOp::Delete { id }) = e {
                    ids.push(id);
                }
                Ok(())
            });
            r.map(|c| (ids, c.report.cut_bytes))
        };
        assert_eq!(replay(&good).unwrap(), (vec![1, 2, 3, 4], 0));
        // The second record damaged, the next group intact: refused.
        let mut bad = good.clone();
        bad[HEADER_LEN + rec + 20] ^= 1;
        assert!(replay(&bad).is_err());
        // Without the next group: a torn group, cut after its first record.
        let torn = &bad[..one_group as usize];
        assert_eq!(replay(torn).unwrap(), (vec![1], 2 * rec as u64));
        // The first record of the second group damaged: nothing after it.
        let mut last = good.clone();
        last[HEADER_LEN + 3 * rec + 20] ^= 1;
        assert_eq!(replay(&last).unwrap(), (vec![1, 2, 3], rec as u64));
        let _ = fs::remove_dir_all(&dir);
    }
}
