# vecdb

A vector database written from scratch in Rust, designed for billion-scale
approximate nearest neighbor search.

Status: step 1 of the roadmap is done (exact search, segment file format,
benchmark harness); step 2 (SIMD) is done for AVX2/FMA.

## Goals

- **Any dimension from 32 to 8192 and beyond**, chosen per collection. The
  engine accepts 1 to 65,536 dimensions; that cap is a sanity check and can be
  raised.
- **Single-node and distributed deployments** using the same engine and the
  same API. A small installation runs as one process; a large one spreads
  shards over many machines.
- **High recall at low latency.** Proposed targets, to be fixed once there are
  measurements on real hardware: recall@10 ≥ 0.95 by default, ≥ 0.99 when a
  query asks for it, at single-digit-millisecond p99 latency per node.
- **Billions of vectors.**

## Layout

```
crates/vecdb-core    library: SIMD distance kernels, top-k, stores, flat index, file formats
crates/vecdb-bench   CLI: synthetic data, ground truth, segment build, recall/latency runs,
                     kernel micro-benchmark
```

`vecdb-core` has no external dependencies yet.

## Build and test

```
cargo test --workspace
cargo build --release
```

Distance kernels are chosen at runtime for the CPU, so a default build is
portable and still uses AVX2/FMA where available. `VECDB_KERNELS=scalar` (or
`avx2-fma`) forces a kernel set, so A/B runs can use the same binary.

## Benchmarks

```
B=target/release/vecdb-bench
$B synth --out data/synth --n 100000 --nq 1000 --dim 128
$B groundtruth --base data/synth/base.fvecs --query data/synth/query.fvecs --out data/synth/gt.ivecs
$B build --base data/synth/base.fvecs --out data/synth/flat.vdb
$B search --segment data/synth/flat.vdb --query data/synth/query.fvecs --gt data/synth/gt.ivecs --k 10
```

The reader also loads the standard datasets: SIFT1M and GIST1M (`.fvecs` /
`.ivecs`) and BIGANN/SIFT1B (`.bvecs`).

The `recall@10 = 1.0` from that run is a self-check: `groundtruth` uses the
same flat index. It shows that the segment round trip and tie-breaking are
deterministic, but it does not independently show correctness. The independent
check is SIFT1M against its published ground truth.

Flat search (i7-8700, default release build, 128-dim L2, one thread; cost per
vector of one distance plus top-k; median of 5 back-to-back runs):

| build | 2k vectors (in cache) | 100k vectors (51 MB, out of cache) |
|---|---:|---:|
| step 1 (commit d4cebe4, auto-vectorized SSE2) | 22.0 ns | 35.3 ns |
| step 2, `VECDB_KERNELS=scalar` | 20.5 ns | 34.9 ns |
| step 2, `avx2-fma` (default on this CPU) | 13-14 ns | 29.6 ns |

About 1.5 ns of the in-cache gain comes from a faster top-k (a single float
compare rejects most candidates); the rest is the AVX2/FMA kernel.

Distance kernel alone (`vecdb-bench kernels`, ns per distance, in cache,
median of 5):

| dim | 32 | 128 | 1024 | 4096 | 8192 |
|---|---:|---:|---:|---:|---:|
| L2, `scalar` | 6.7 | 17.5 | 113 | 447 | 1063 |
| L2, `avx2-fma` | 3.0 | 9.3 | 54 | 212 | 578 |
| dot, `scalar` | 5.9 | 13.6 | 95 | 384 | 878 |
| dot, `avx2-fma` | 2.6 | 8.7 | 51 | 208 | 587 |

In cache, AVX2/FMA is 1.8-2.2x faster than the portable kernels for L2 and
1.5-2.3x for dot product across 32-8192 dims. Out of cache the scan is
memory-bound (about 15-17 GB/s from one core), so the gain there is small. One brute-force query over 1B x 128 vectors would
read 512 GB, which is why billion scale needs compression and a graph index,
not a faster scan.

Cross-check: ground truth computed with `scalar` and searched with `avx2-fma`
gives recall@10 = 1.0 for L2, cosine and inner product at 128, 1024 and 8192
dims. The two sets add floats in a different order, so a near-tie could
differ in the last bit on other data; that is expected, not a bug.

## Designing for billions

Raw `f32` vectors cost `n x dim x 4` bytes:

| dim | 1 billion vectors, raw f32 | PQ codes at 32 B/vector | PQ codes at 64 B/vector |
|----:|---------------------------:|------------------------:|------------------------:|
| 32 | 128 GB | 32 GB | 64 GB |
| 128 | 512 GB | 32 GB | 64 GB |
| 768 | 3.07 TB | 32 GB | 64 GB |
| 1536 | 6.14 TB | 32 GB | 64 GB |
| 8192 | 32.8 TB | 32 GB | 64 GB |

A graph with 64 `u32` neighbors per vector adds 256 GB per billion vectors.
Full vectors and the graph therefore live on NVMe SSD, and only compressed
codes stay in RAM. This is the DiskANN approach: a Vamana graph plus product
quantization (PQ). Sharding across nodes comes on top of that.

What the 32-8192 dimension range changes:

- **PQ code size is set per collection.** At a fixed number of bytes per
  vector, higher dimensions compress more and lose more recall. High-dimension
  collections need larger codes.
- **Re-ranking with full vectors is required for high recall.** The final
  top-k is recomputed exactly from full vectors, which can be stored as f32,
  f16/bf16 (half the size) or int8 (a quarter).
- **The disk layout depends on dimension.** DiskANN stores each node's full
  vector and neighbor list together in one 4 KB SSD page. With 64 neighbors
  (256 B), that fits up to about 950 f32 dimensions, or about 1,900 in f16. An
  8192-dim f32 vector alone is 32 KB, so high-dimension collections keep the
  graph and the full vectors in separate files.
- **Distance cost grows linearly with dimension,** so SIMD kernels matter more
  at 4096-8192 dims.

## Single-node and distributed modes

Proposed architecture. Not final: it depends on recall and latency targets and
on hardware.

```
client ─▶ query node (coordinator) ─▶ shard 1: segments ─┐
          plans, fans out,          ─▶ shard 2: segments ─┼─▶ per-shard top-k
          merges top-k              ─▶ shard N: segments ─┘
```

- **One engine, two transports.** A collection is split into shards and each
  shard into segments. A search fans out to the shards and merges their top-k
  lists. In single-node mode every shard is local and the fan-out is a
  function call. In distributed mode the same shard interface is served over
  gRPC by data nodes.
- **Roles in distributed mode:** stateless query nodes (coordinators), data
  nodes that own shard replicas, and a metadata service for the shard map
  (Raft-backed). Single-node mode runs all three roles in one process.
- **Rule for the code now:** `vecdb-core` contains no networking. Everything
  network-related lives in a separate crate behind the shard interface.

Within a shard:

- **Write path:** WAL → growing segment (in memory, exact search) → sealed
  immutable segment → background index build (PQ + Vamana) → SSD (later:
  object storage).
- **Read path:** the growing segment is searched by brute force. Sealed
  segments use PQ distances in RAM to walk the graph on SSD, then re-rank the
  candidates with full vectors. The per-segment top-k lists are then merged.
- **Deletes:** a tombstone bitmap per segment, with compaction to reclaim space.

## Roadmap

1. ✅ Exact search: distance kernels, top-k, flat index, segment format v1
   (checksummed, atomic writes), dataset I/O, recall harness
2. ◐ SIMD kernels with runtime CPU detection and a micro-benchmark at 32-8192
   dims: AVX2/FMA done. Still to do once there is hardware to test on: AVX-512
   and NEON (the portable kernels should auto-vectorize to NEON on aarch64, but
   that has not been built or tested yet), and batch kernels that score several
   rows per query load
3. Quantization: PQ with per-collection code size, ADC search with lookup
   tables; f16/bf16/int8 storage for full vectors; measure recall against bytes
   per vector
4. Vamana graph in memory: build, beam search, a search-time recall/latency
   knob; recall vs QPS on SIFT1M and a high-dimension dataset
5. Disk index: PQ in RAM, graph and full vectors on SSD (layout chosen by
   dimension), re-ranking, io_uring; target 100M vectors on one node
6. Segments, WAL, deletes, compaction, metadata filtering
7. Server and single-node mode: gRPC/HTTP API, collections, the shard
   interface, concurrent reads and writes
8. Distributed mode: remote shards, query nodes, metadata service,
   replication, rebalancing
