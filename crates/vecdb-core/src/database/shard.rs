//! What a shard does, wherever it is: the part of a collection that holds
//! some of its rows. [`Collection`] is the one in this process. Requests and
//! replies are plain owned data, so a shard in another process can stand in
//! for it.

use crate::collection::{Attrs, Collection, CollectionStats, Fetch, SearchOptions, SearchStats, WriteOp};
use crate::error::{Error, Result};
use crate::topk::SearchResult;

/// One write.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// Inserts the row, replacing any row with the same id.
    Upsert(Row),
    Delete {
        id: u64,
    },
}

impl Op {
    pub fn id(&self) -> u64 {
        match self {
            Op::Upsert(r) => r.id,
            Op::Delete { id } => *id,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub id: u64,
    pub vector: Vec<f32>,
    pub attrs: Attrs,
}

impl Row {
    /// About its size.
    pub fn bytes(&self) -> usize {
        8 + self.vector.len() * 4 + self.attrs.encoded_len()
    }
}

#[derive(Clone, Debug)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    pub k: usize,
    pub options: SearchOptions,
    /// Of the rows found, in the same snapshot.
    pub fetch: Fetch,
    /// Most those may take; the search fails beyond it.
    pub max_row_bytes: usize,
}

impl SearchRequest {
    /// For ids and distances only.
    pub fn new(vector: Vec<f32>, k: usize, options: SearchOptions) -> Self {
        Self { vector, k, options, fetch: Fetch::NONE, max_row_bytes: usize::MAX }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SearchReply {
    /// Closest first.
    pub results: Vec<SearchResult>,
    /// Their rows, if any of them was asked for: what was not is empty.
    pub rows: Vec<Row>,
    /// Summed over segments and shards.
    pub stats: SearchStats,
}

/// One shard. Calls come from any number of threads.
pub trait Shard: Send + Sync {
    /// Applies `ops` in order, all or none. Per op, whether its id had a
    /// row before.
    fn write(&self, ops: &[Op]) -> Result<Vec<bool>>;
    fn search(&self, req: &SearchRequest) -> Result<SearchReply>;
    fn get(&self, id: u64) -> Result<Option<Row>>;
    fn stats(&self) -> Result<CollectionStats>;
    /// Writes what is in RAM to segment files.
    fn flush(&self) -> Result<()>;
    /// Merges all segments into one index.
    fn optimize(&self) -> Result<()>;
    /// Waits until background work has caught up.
    fn wait_idle(&self) -> Result<()>;
    /// The collection itself, if it is in this process: a search is then
    /// split into tasks for its segments along with the other shards'.
    fn local(&self) -> Option<&Collection> {
        None
    }
}

impl Shard for Collection {
    fn write(&self, ops: &[Op]) -> Result<Vec<bool>> {
        let ops: Vec<WriteOp> = ops
            .iter()
            .map(|op| match op {
                Op::Upsert(r) => WriteOp::Upsert { id: r.id, vector: &r.vector, attrs: &r.attrs },
                Op::Delete { id } => WriteOp::Delete { id: *id },
            })
            .collect();
        self.write_ops(&ops)
    }

    fn search(&self, req: &SearchRequest) -> Result<SearchReply> {
        let snap = self.snapshot();
        let (results, stats) = self.search_in(&snap, &req.vector, req.k, &req.options)?;
        let rows = fetch_rows(&results, req, |id| self.get_in(&snap, id, req.fetch))?;
        Ok(SearchReply { results, rows, stats })
    }

    fn get(&self, id: u64) -> Result<Option<Row>> {
        Ok((**self).get(id)?.map(|(vector, attrs)| Row { id, vector, attrs }))
    }

    fn stats(&self) -> Result<CollectionStats> {
        Ok((**self).stats())
    }

    fn flush(&self) -> Result<()> {
        (**self).flush()
    }

    fn optimize(&self) -> Result<()> {
        (**self).optimize().map(drop)
    }

    fn wait_idle(&self) -> Result<()> {
        (**self).wait_idle()
    }

    fn local(&self) -> Option<&Collection> {
        Some(self)
    }
}

/// The rows of `results`, by `get`, unless nothing is asked for.
pub(super) fn fetch_rows(
    results: &[SearchResult],
    req: &SearchRequest,
    get: impl Fn(u64) -> Result<Option<(Vec<f32>, Attrs)>>,
) -> Result<Vec<Row>> {
    if req.fetch == Fetch::NONE {
        return Ok(Vec::new());
    }
    let mut left = req.max_row_bytes;
    let row = |r: &SearchResult| {
        let Some((vector, attrs)) = get(r.id)? else {
            return Err(Error::Corrupt(format!("row {} was found by a search, but is not there", r.id)));
        };
        let row = Row { id: r.id, vector, attrs };
        left = left.checked_sub(row.bytes()).ok_or_else(|| {
            let m = format!("the rows found take more than {} bytes: ask for fewer", req.max_row_bytes);
            Error::InvalidArgument(m)
        })?;
        Ok(row)
    };
    results.iter().map(row).collect()
}

/// The shard, of `shards`, that has the row `id`. Part of the format: rows
/// are stored by it.
pub fn shard_of(id: u64, shards: usize) -> usize {
    // SplitMix64's mix, so that ids with a pattern spread evenly, then the
    // high bits scaled to the number of shards.
    let mut x = id.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    ((x as u128 * shards as u128) >> 64) as usize
}
