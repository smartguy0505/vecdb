# vecdb

A vector database written from scratch in Rust, designed for billion-scale
approximate nearest neighbor search.

Status: step 1 of the roadmap is done (exact search, segment file format,
benchmark harness).

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
crates/vecdb-core    library: distance kernels, top-k, stores, flat index, file formats
crates/vecdb-bench   CLI: synthetic data, ground truth, segment build, recall/latency runs
```

`vecdb-core` has no external dependencies yet.

## Build and test

```
cargo test --workspace
cargo build --release
```

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

Baseline (i7-8700, 128-dim L2, one thread, cost of one distance plus top-k):

| build | 2k vectors (in cache) | 100k vectors (51 MB, out of cache) |
|---|---:|---:|
| default x86-64 (SSE2) | 22 ns | 35 ns |
| `-C target-cpu=native` (AVX2/FMA) | 15.5 ns | 32 ns |

In cache the scan is compute-bound, and wider SIMD helps (roadmap step 2).
Out of cache it is memory-bound: about 15 GB/s from one core. One brute-force
query over 1B x 128 vectors would read 512 GB, which is why billion scale needs
compression and a graph index, not a faster scan.

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
2. SIMD kernels (AVX2/FMA, AVX-512, NEON) with runtime CPU detection, plus a
   micro-benchmark suite at 32-8192 dimensions
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
