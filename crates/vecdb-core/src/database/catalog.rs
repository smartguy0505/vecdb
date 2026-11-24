//! The catalog: the file that says which collections a database has.
//!
//! ```text
//! magic "VECDBCAT", version u32 (1)
//! next_id u64
//! collections: count u32, then per collection, by name:
//!         id u64, shards u32, name (u16 length + UTF-8)
//! dropped: count u32, then per collection: id u64
//! crc u32: CRC-32C of everything before it
//! ```
//!
//! It is replaced atomically, which is what creates or drops a collection.
//! `dropped` are the ones whose files may still be there.

use std::path::Path;

use crate::collection::bytes::{Dec, Enc, write_file};
use crate::error::{Error, Result};

pub(super) const CATALOG: &str = "CATALOG";
const MAGIC: [u8; 8] = *b"VECDBCAT";
const VERSION: u32 = 1;
/// Longest collection name, in bytes.
pub const MAX_NAME: usize = 128;
/// Most shards per collection.
pub const MAX_SHARDS: usize = 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Entry {
    pub id: u64,
    pub name: String,
    pub shards: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Catalog {
    /// Ids are not used again.
    pub next_id: u64,
    /// By name.
    pub entries: Vec<Entry>,
    pub dropped: Vec<u64>,
}

/// Names are letters, digits, `-`, `_` and `.`, and start with a letter or
/// digit.
pub(super) fn check_name(name: &str) -> Result<()> {
    let ok = |b: u8| b.is_ascii_alphanumeric() || b"-_.".contains(&b);
    if name.is_empty() || name.len() > MAX_NAME || !name.bytes().all(ok) || !name.as_bytes()[0].is_ascii_alphanumeric()
    {
        return Err(Error::InvalidArgument(format!(
            "collection name {name:?}: use up to {MAX_NAME} letters, digits, '-', '_' and '.', \
             starting with a letter or digit"
        )));
    }
    Ok(())
}

impl Catalog {
    pub fn find(&self, name: &str) -> Option<usize> {
        self.entries.binary_search_by(|e| e.name.as_str().cmp(name)).ok()
    }

    pub fn insert(&mut self, e: Entry) {
        let at = self.entries.partition_point(|x| x.name < e.name);
        self.entries.insert(at, e);
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut e = Enc::default();
        e.buf.extend_from_slice(&MAGIC);
        e.u32(VERSION);
        e.u64(self.next_id);
        e.u32(self.entries.len() as u32);
        for c in &self.entries {
            e.u64(c.id);
            e.u32(c.shards as u32);
            e.str(&c.name);
        }
        e.u32(self.dropped.len() as u32);
        self.dropped.iter().for_each(|&id| e.u64(id));
        e.crc();
        e.buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut d = Dec::checked(bytes, "catalog")?;
        if d.take(8)? != MAGIC {
            return Err(d.corrupt("bad magic"));
        }
        let version = d.u32()?;
        if version != VERSION {
            return Err(Error::Unsupported(format!("catalog version {version}")));
        }
        let mut c = Self { next_id: d.u64()?, ..Default::default() };
        for _ in 0..d.u32()? {
            let (id, shards) = (d.u64()?, d.u32()? as usize);
            let name = d.str()?.to_string();
            check_name(&name).map_err(|e| d.corrupt(&e.to_string()))?;
            let in_order = c.entries.last().is_none_or(|l: &Entry| l.name < name);
            if id >= c.next_id || !(1..=MAX_SHARDS).contains(&shards) || !in_order {
                return Err(d.corrupt("inconsistent entry"));
            }
            c.entries.push(Entry { id, name, shards });
        }
        for _ in 0..d.u32()? {
            c.dropped.push(d.u64()?);
        }
        let mut ids: Vec<u64> = c.entries.iter().map(|e| e.id).chain(c.dropped.iter().copied()).collect();
        ids.sort_unstable();
        if ids.windows(2).any(|w| w[0] == w[1]) || ids.last().is_some_and(|&l| l >= c.next_id) {
            return Err(d.corrupt("an id twice, or past the next one"));
        }
        d.done()?;
        Ok(c)
    }

    pub fn read(dir: &Path) -> Result<Self> {
        Self::decode(&std::fs::read(dir.join(CATALOG))?)
    }

    /// Writes the catalog under a temporary name, fsyncs it and renames it
    /// into place. The caller fsyncs the directory.
    pub fn install(&self, dir: &Path) -> Result<()> {
        write_file(&dir.join(CATALOG), &self.encode(), false)
    }
}
