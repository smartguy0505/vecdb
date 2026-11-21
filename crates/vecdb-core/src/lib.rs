//! vecdb-core: storage, distance kernels and indexes for the vecdb vector database.
//!
//! Conventions used everywhere in this crate:
//! - Every metric is a *distance*: smaller means closer.
//! - Vectors are `f32`, row-major, all of one dimension per index.
//! - Vectors are identified by caller-chosen `u64` ids; rows are internal positions.

#![deny(unsafe_code)]

pub mod crc32c;
pub mod dataset;
pub mod distance;
pub mod error;
pub mod flat;
pub mod format;
pub mod kernels;
pub mod rng;
pub mod store;
pub mod topk;

pub use distance::Metric;
pub use error::{Error, Result};
pub use flat::FlatIndex;
pub use store::{InMemoryStore, MAX_DIM, VectorStore};
pub use topk::{SearchResult, TopK};
