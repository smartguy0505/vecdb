//! vecdb-core: storage, distance kernels and indexes for the vecdb vector database.
//!
//! Conventions used everywhere in this crate:
//! - Every metric is a *distance*: smaller means closer.
//! - Vectors are `f32`, row-major, all of one dimension per index.
//! - Vectors are identified by caller-chosen `u64` ids; rows are internal positions.

#![deny(unsafe_code)]

pub mod cancel;
pub mod collection;
pub mod crc32c;
pub mod database;
pub mod dataset;
pub mod disk;
pub mod distance;
pub mod error;
pub mod flat;
pub mod format;
pub mod graph;
pub mod kernels;
mod par;
pub mod pool;
pub mod pq_index;
pub mod quant;
pub mod rng;
pub mod store;
pub mod topk;

pub use cancel::Cancel;
pub use collection::{
    Attrs, Collection, CollectionConfig, Durability, Field, FieldType, Filter, FilterMode, IndexConfig, OpenOptions,
    SearchOptions, Value, WriteOp,
};
pub use database::{
    Database, DatabaseOptions, Op, Row, SearchReply, SearchRequest, Shard, ShardedCollection,
};
pub use dataset::{VectorFile, VectorSource};
pub use disk::{
    DiskIndex, DiskOpenOptions, DiskSearchParams, DiskSearchStats, DiskWriteOptions, IoMode, Layout, MergePrune,
    PartitionedBuildParams, PartitionedBuildReport,
};
pub use distance::Metric;
pub use error::{Error, Result};
pub use flat::FlatIndex;
pub use graph::{GraphIndex, GraphParams, Navigation, SearchStats};
pub use pool::SearchPool;
pub use pq_index::PqIndex;
pub use store::{InMemoryStore, MAX_DIM, VectorStore};
pub use topk::{SearchResult, TopK};
