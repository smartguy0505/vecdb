//! What a shard does, wherever it is: the part of a collection that holds
//! some of its rows. [`Collection`] is the one in this process. Requests and
//! replies are plain owned data, so a shard in another process can stand in
//! for it.

use crate::collection::{Attrs, Collection, CollectionStats, SearchOptions, SearchStats, WriteOp};
use crate::error::Result;
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

#[derive(Clone, Debug)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    pub k: usize,
    pub options: SearchOptions,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SearchReply {
    /// Closest first.
    pub results: Vec<SearchResult>,
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
        let (results, stats) = self.search_with_stats(&req.vector, req.k, &req.options)?;
        Ok(SearchReply { results, stats })
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
