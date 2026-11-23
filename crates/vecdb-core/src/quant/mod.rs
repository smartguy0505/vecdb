//! Vector compression: product quantization for compact in-memory codes,
//! and full-vector codecs (f16, bf16, int8) for re-ranking.

pub mod codec;
pub mod kmeans;
pub mod pq;

pub use codec::{Codec, EncodedVectors};
pub use pq::{DistanceTable, PqParams, ProductQuantizer};
