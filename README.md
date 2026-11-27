# vecdb

A vector database written from scratch in Rust, designed for billion-scale
approximate nearest neighbor search.

Status: steps 1-4 are done: exact search, the segment file format and
benchmark harness; SIMD kernels (AVX2/FMA); quantization (PQ, plus
f16/bf16/int8 storage for re-ranking); and an in-memory graph index (Vamana)
with a search-time recall/latency knob. Step 5, the SSD index, works and is
measured at 1M scale: PQ codes in RAM, graph and full vectors on SSD, read
with direct I/O and io_uring, and a partitioned build that needs RAM for one
partition rather than the whole collection. The 100M-vectors-per-node target
has not been attempted yet: it needs a dedicated machine. Step 6 is done:
collections with a write-ahead log, segments, deletes, compaction, crash
recovery and attribute filters. Step 7 is done: a collection is shared
between threads, with snapshot reads, index builds beside writes and group
commit (7a); it flushes and builds in the background and searches its
segments in parallel, and databases hold named collections split into shards
(7b); a server offers them over HTTP (7c); writes that outrun index builds are
slowed down instead of stopped (7d), and one that gives up and comes back has
its place in the line again (7e); index builds that are behind take more at
once (7f). Step 8, the distributed mode, is next.

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
crates/vecdb-core    library: SIMD distance kernels, top-k, stores, flat index,
                     k-means, product quantization, vector codecs, PQ index,
                     graph index (Vamana), SSD index, collections (WAL,
                     segments, deletes, compaction), databases (named,
                     sharded collections), file formats
crates/vecdb-server  HTTP/JSON server for a database
crates/vecdb-bench   CLI: synthetic data, ground truth, segment build, recall/latency runs,
                     SSD index build and search, collection ingest and search, the
                     server against calls in process, kernel micro-benchmark
```

`vecdb-core` depends only on the standard library, plus `io-uring` and `libc`
on Linux (for the SSD index's reads). On other platforms the SSD index is
written to fall back to positioned reads through the page cache; that path
has not been built or tested outside Linux yet. The server adds `hyper`,
`tokio`, `serde` and `serde_json`, and runs on Unix only.

## Build and test

```
cargo test --workspace
cargo build --release
```

Distance kernels are chosen at runtime for the CPU, so a default build is
portable and still uses AVX2/FMA where available. `VECDB_KERNELS=scalar` (or
`avx2-fma`) forces a kernel set, so A/B runs can use the same binary.

[CONTRIBUTING.md](CONTRIBUTING.md) says how to work on this code, for people and
for coding agents.

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
same flat index. The independent check is SIFT1M against its published ground
truth. There, exact flat search gives recall@1 = 0.990, @10 = 0.9994 and
@100 = 0.9999. `vecdb-bench search` checks every difference and reports 0 real
misses: all of them are exact distance ties (SIFT values are integers) broken
in a different order.

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

## Quantization (step 3)

PQ splits each vector into `m` parts and stores each part as a 1-byte
centroid id, so a vector costs `m` bytes in RAM, whatever its dimension. A
search scans the codes with a per-query lookup table, then re-ranks the best
candidates with full vectors. Full vectors can be stored as `f32`, `f16`,
`bf16` or `sq8` (int8).

All numbers below are recall@10 against exact ground truth, with re-ranking
from `f32` unless a codec is named. The 1024- and 8192-dim sets are synthetic
and small (vectors from a 64-dim subspace plus noise); real embeddings must
be measured.

**SIFT1M, L2** (1M x 128, 1000 queries, published ground truth; exact search
reaches 0.9994 because of ties):

| PQ bytes/vector | no re-rank | re-rank 50 | re-rank 200 |
|---:|---:|---:|---:|
| 8 | 0.313 | 0.644 | 0.886 |
| 16 | 0.534 | 0.900 | 0.990 |
| 32 | 0.707 | 0.989 | 0.9993 |
| 64 | 0.868 | 0.9994 | 0.9994 |

**Collection size matters.** The same settings on the first 100k SIFT vectors
compared with all 1M:

| PQ bytes, re-rank | 100k vectors | 1M vectors |
|---|---:|---:|
| 16 B, re-rank 50 | 0.963 | 0.900 |
| 16 B, re-rank 200 | 0.998 | 0.990 |
| 32 B, re-rank 50 | 0.997 | 0.989 |
| 32 B, re-rank 200 | 1.000 | 0.9993 |

At a fixed code size and re-rank depth, recall falls as the collection grows.
The settings needed at 100M-1B vectors are not measured yet.

**1024 dims, 100k vectors, 200 queries:**

| PQ bytes/vector | L2, no re-rank | L2, re-rank 50 | L2, re-rank 200 | cosine, re-rank 50 | cosine, re-rank 200 |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.258 | 0.567 | 0.834 | | |
| 64 | 0.387 | 0.749 | 0.949 | 0.936 | 0.9995 |
| 128 | 0.540 | 0.935 | 0.9995 | 0.994 | 1.000 |
| 256 | 0.815 | 1.000 | 1.000 | | |

**8192 dims, 20k vectors, 100 queries, L2** (k-means `--iters 10`):

| PQ bytes/vector | no re-rank | re-rank 50 | re-rank 200 |
|---:|---:|---:|---:|
| 256 | 0.462 | 0.828 | 0.984 |
| 512 | 0.518 | 0.903 | 0.999 |

**Re-ranking codecs**, re-rank 200:
- 1024 dims, 128 B, L2: `f32` 0.9995, `f16` 0.999, `bf16` 0.999, `sq8` 0.992.
- 1024 dims, cosine: `f16` equals `f32`; `sq8` gives 0.994.
- 8192 dims: `f16` and `sq8` equal `f32`.
- SIFT: `f16` and `bf16` are lossless because its values are small integers;
  `sq8` gives 0.989.

What this shows, and what it doesn't:

- **Re-ranking is what delivers high recall.** PQ alone reached 0.26-0.87 here.
- **Recall at a given setting depends on collection size, metric and data.**
  Re-ranking 200 candidates is 0.02% of SIFT1M but 1% of the 8192-dim set,
  and cosine did better than L2 on the same 1024-dim vectors. So none of
  these numbers can be carried to a billion vectors; the parameters for
  billion-scale shards have to be measured at that scale.
- **RAM:** PQ codes cost `m` GB per billion vectors (for example, 128 GB for
  128-byte codes, against 4 TB for raw 1024-dim f32).
- **Speed:** the code scan is linear, about 0.4 ns per code byte plus 1.5 ns
  per vector, so 5-26 ms per query at 1M vectors. That is too slow for low
  latency on its own; the graph index (step 4) uses the same codes on a small
  fraction of the vectors. QPS in `vecdb-bench pq` output is noisy, and the
  first row of each `m` runs cold.

Commands for these runs (`B=target/release/vecdb-bench`, `S=` the SIFT1M
directory):

```
$B pq --base $S/sift_base.fvecs --query $S/sift_query.fvecs --gt $S/sift_groundtruth.ivecs \
      --nq 1000 --m 8,16,32,64 --rerank 0,10,50,200
$B groundtruth --base $S/sift_base.fvecs --limit 100000 --query $S/sift_query.fvecs --out gt100k.ivecs --k 10
$B pq --base $S/sift_base.fvecs --limit 100000 --query $S/sift_query.fvecs --gt gt100k.ivecs \
      --nq 1000 --m 16,32 --rerank 0,50,200 --codecs f32
$B synth --out emb1024 --dim 1024 --n 100000 --nq 200 --intrinsic 64 --noise 0.05 --seed 21
$B groundtruth --base emb1024/base.fvecs --query emb1024/query.fvecs --out emb1024/gt.ivecs --k 10 [--metric cosine]
$B pq --base emb1024/base.fvecs --query emb1024/query.fvecs --gt emb1024/gt.ivecs [--metric cosine] \
      --m 32,64,128,256 --rerank 0,50,200
$B synth --out emb8192 --dim 8192 --n 20000 --nq 100 --intrinsic 64 --noise 0.05 --seed 22
$B groundtruth --base emb8192/base.fvecs --query emb8192/query.fvecs --out emb8192/gt.ivecs --k 10
$B pq --base emb8192/base.fvecs --query emb8192/query.fvecs --gt emb8192/gt.ivecs \
      --m 256,512 --rerank 0,50,200 --codecs f32,f16,sq8 --iters 10
```

Known follow-ups: `f16` decoding is scalar, which costs about 3 ms per query
when re-ranking 200 x 8192 dims; F16C hardware conversion would fix that. A
SIMD "fast-scan" (4-bit PQ) could speed up the code scan severalfold. PQ
codebooks are not yet saved in the segment format (step 6).

## Graph index (step 4)

`GraphIndex` is a Vamana graph (the graph DiskANN uses) over full vectors in
memory, searched with a greedy beam search. The search list size `L` is the
recall/latency knob: a longer list expands more nodes and computes more
distances. Build parameters: maximum degree `R` (default 64), build list size
(100) and pruning factor α (1.2).

- **Build** inserts points in random order, in batches that grow from 1 to 2%
  of the collection (as in ParlayANN). Searches and pruning within a batch run
  in parallel, and the graph comes out identical for any thread count.
- **Reachability repair.** Vamana does not guarantee that every node can be
  reached from the entry point: pruning can remove all of a node's in-edges.
  On GIST1M that left 2,443 nodes (0.25%) that no search could return. After
  the build, each unreachable node gets an edge from the nearest reachable
  node with a free slot; all 982,694 GIST1M nodes are then reachable. (The
  repair can fail only if every reachable node is already full, which needs
  a very small `R`.)
- **Identical vectors share one graph node.** Otherwise copies prune each
  other (distance 0) and become unreachable. SIFT1M has 14,538 such copies,
  GIST1M 17,306.
- **Inner product** links vectors by angle (the cosine distance between
  rows) and ranks by −dot at search time. It is the weakest metric; see the
  1024-dim table below.
- **Two ways to navigate the same graph.** `exact` uses full vectors. `pq`
  uses PQ codes to steer and computes the exact distance of every expanded
  node. That is the disk index's search (step 5: codes in RAM, one SSD read
  per expanded node), run in memory so its recall can be measured before any
  I/O code exists.
- **Not yet:** inserts after the build and deletes (step 6). Two build
  passes (`passes = 2`, the original paper's schedule) were measured later,
  through the SSD index: on SIFT1M they raise recall at equal L by up to
  0.005 (0.965 to 0.970 at L 40) for twice the build time (see the
  partitioned build).

Results below are recall@10 and single-thread latency per query (median,
p99), all in RAM. `dist/q` is distance computations per query (PQ lookups
for `pq`); `hops/q` is nodes expanded, which becomes SSD reads in step 5.
Build parameters are the defaults and were not tuned per dataset.

**SIFT1M, L2** (1M x 128, 10,000 queries, published ground truth; exact search
reaches 0.9994). Built in 2-2.5 min on 6 threads; mean degree 45.5. PQ: 32 B/vector.

| L | recall, full vectors | dist/q | hops/q | p50 / p99 ms | recall, PQ-guided | p50 / p99 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.853 | 719 | 16 | 0.07 / 0.15 | 0.690 | 0.06 / 0.11 |
| 20 | 0.937 | 1,040 | 25 | 0.12 / 0.22 | 0.873 | 0.09 / 0.16 |
| 40 | 0.978 | 1,637 | 45 | 0.21 / 0.41 | 0.962 | 0.15 / 0.26 |
| 80 | 0.9945 | 2,710 | 84 | 0.37 / 0.72 | 0.991 | 0.23 / 0.39 |
| 160 | 0.9986 | 4,562 | 164 | 0.61 / 1.08 | 0.998 | 0.39 / 0.64 |
| 320 | 0.9993 | 7,675 | 324 | 1.08 / 2.14 | 0.9992 | 0.72 / 1.26 |

**GIST1M, L2** (1M x 960, 1,000 queries, published ground truth; exact search
reaches 0.9992). Built in 12-14 min on 6 threads; mean degree 33.4. PQ: 120 B/vector.

| L | recall, full vectors | dist/q | hops/q | p50 / p99 ms | recall, PQ-guided | p50 / p99 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.613 | 891 | 17 | 0.26 / 0.45 | 0.464 | 0.20 / 0.32 |
| 20 | 0.746 | 1,316 | 27 | 0.43 / 0.83 | 0.629 | 0.28 / 0.44 |
| 40 | 0.854 | 2,090 | 46 | 0.69 / 1.06 | 0.781 | 0.38 / 0.51 |
| 80 | 0.927 | 3,544 | 84 | 1.24 / 1.79 | 0.892 | 0.64 / 0.88 |
| 160 | 0.971 | 6,147 | 164 | 2.16 / 3.23 | 0.959 | 1.06 / 1.47 |
| 320 | 0.987 | 10,626 | 323 | 3.79 / 6.01 | 0.984 | 1.85 / 3.00 |

**8192 dims, 20k vectors, 100 queries, L2** (synthetic: a 64-dim subspace plus
noise, as in step 3). Built in 141 s; mean degree 53.7. PQ: 512 B/vector.

| L | recall, full vectors | p50 ms | recall, PQ-guided | p50 ms |
|---:|---:|---:|---:|---:|
| 40 | 0.909 | 4.1 | 0.842 | 1.8 |
| 80 | 0.977 | 7.0 | 0.956 | 2.6 |
| 160 | 0.998 | 11.3 | 0.992 | 3.8 |

**Metrics, 1024 dims, 100k vectors, 200 queries** (the synthetic set from
step 3; the same vectors under each metric, full-vector navigation):

| L | L2 | cosine | inner product |
|---:|---:|---:|---:|
| 40 | 0.7475 | 0.6965 | 0.6190 |
| 80 | 0.8830 | 0.8695 | 0.7960 |
| 160 | 0.9555 | 0.9650 | 0.9300 |

With PQ guidance (128 B) at L 160: 0.9645, 0.9605 and 0.9385. This set needs a
larger `L` than SIFT1M for the same recall.

**Inner product with uneven norms** is the hard case. Scaling each of those
vectors by a random factor exp(0.5·N(0,1)) keeps their directions but skews
their norms; recall then reaches only 0.50 at L 160 and 0.70 at L 320. The
usual reduction of inner product to L2 (an extra coordinate sqrt(M² − |x|²)
per vector) did worse on both sets (0.63 at L 160 on the plain set, 0.09 at
L 320 on the skewed one), although it did better on a small 32-dim test set.
For normalized embeddings, use cosine (it gives the same ranking as inner
product). Unnormalized inner-product search needs its own graph construction,
which is a follow-up.

What this shows, and what it doesn't:

- **The graph replaces the linear scan.** On SIFT1M, recall 0.9945 at L 80
  touches 2,710 vectors (0.27% of the collection) in about 0.4 ms, where the
  step 3 code scan took 5-26 ms.
- **Recall at a given L depends heavily on the data.** L 160 gives 0.9986 on
  SIFT1M but 0.971 on GIST1M, which needs L 320 for 0.987. `L` has to be
  chosen per collection, or set from a recall target measured on sample
  queries.
- **PQ guidance pays off more as dimension grows.** Its distances read `m`
  bytes instead of the full vector. At the same `L` it is about 1.5x faster
  on SIFT, 2x on GIST and 2.5-3x at 8192 dims, with somewhat lower recall
  that a larger `L` buys back. For example, on GIST PQ-guided L 320 beats
  full-vector L 160 on both recall (0.984 against 0.971) and latency (1.8
  against 2.2 ms).
- **What step 5 has to serve from SSD:** about 85 node reads per query at
  L 80 and 165 at L 160. That takes several reads in flight per step (beam
  width) to stay within milliseconds. At 8192 dims each full vector is 32 KB,
  so the disk index should re-score only the final candidates (as step 3
  does) rather than every expanded node.
- **Latency is noisy.** This is a shared desktop-class machine (i7-8700, 6
  cores) running other work. Between identical runs, median latency moved by
  a few percent, and p99 and multi-threaded QPS by up to 40%. Recall is
  deterministic. Nothing here is a billion-vector or SSD number.
- **Build time** is the cost: about 2.5 min for SIFT1M and 12-14 min for
  GIST1M on 6 threads, and 141 s for just 20k vectors at 8192 dims. Large
  collections are built per segment (steps 5-6).

Prefetching the first cache line of each neighbor's vector before computing
the distances cut median latency by about 7% on SIFT1M. That was measured in
one process on one graph, alternating prefetch off and on; whole-vector
prefetch gained less.

Commands for these runs (`B=target/release/vecdb-bench`, `S=` the SIFT1M
directory, `G=` the GIST1M directory):

```
$B graph --base $S/sift_base.fvecs --query $S/sift_query.fvecs --gt $S/sift_groundtruth.ivecs \
      --threads 6 --pq-m 32 --l 10,20,40,80,160,320
$B graph --base $G/gist_base.fvecs --query $G/gist_query.fvecs --gt $G/gist_groundtruth.ivecs \
      --threads 6 --pq-m 120 --l 10,20,40,80,160,320
$B graph --base emb8192/base.fvecs --query emb8192/query.fvecs --gt emb8192/gt.ivecs \
      --threads 6 --pq-m 512 --iters 10 --l 10,20,40,80,160
$B groundtruth --base emb1024/base.fvecs --query emb1024/query.fvecs --out emb1024/gt-ip.ivecs --k 10 --metric ip
$B graph --base emb1024/base.fvecs --query emb1024/query.fvecs --gt emb1024/gt.ivecs \
      --metric l2 --threads 6 --pq-m 128 --l 20,40,80,160   # also: --metric cosine / ip with gt-cos / gt-ip
```

## SSD index (step 5)

`DiskIndex` is the DiskANN design. PQ codes in RAM steer a beam search over
a Vamana graph stored on SSD, and full vectors give the exact distances. It
is written from a built `GraphIndex` (`DiskIndex::write`) and opened in any
process (`DiskIndex::open`).

- **RAM holds only** the PQ codebooks, the PQ codes (`m` bytes per vector)
  and the ids (8 bytes per vector, plus 8 per node when identical vectors
  were merged). Optionally it also caches the nodes nearest the entry point.
  Everything else is read on demand, one block per expanded node.
- **File:** one file in 4 KB sectors. Each node's record is its full vector,
  its degree and its neighbor list, followed by a CRC-32C. Records are packed
  into one-sector blocks (five per sector for SIFT), or into multi-sector
  blocks for long vectors (GIST: 8 KB; 8192 dims: 36 KB). The header and the
  RAM sections are checksummed too. Writes are atomic (temp file, fsync,
  rename).
- **Every record read is verified** (checksum, degree, neighbor ids), so a
  damaged file gives `Error::Corrupt` instead of a panic or a silently wrong
  answer. `verify()` scans the whole file. The CRC uses the SSE4.2
  instruction where available.
- **Search:**
  1. Keep the `L` best candidates by PQ distance.
  2. Each round, read the blocks of up to `W` (beam width) unexpanded
     candidates together.
  3. Score each expanded node exactly from its block, and add its
     neighbors by PQ distance.

  `L` trades recall for latency; `W` trades extra reads for fewer rounds.
  With `W = 1` this is exactly step 4's PQ-guided search, which a test
  checks.
- **I/O:** `O_DIRECT` on Linux, so latency is the SSD's and a large index does
  not evict the page cache. Reads go through io_uring, so the `W` reads of a
  round are in flight together. Where io_uring is unavailable, or on other
  platforms, it falls back to positioned reads. One `SearchContext` (ring
  plus buffers) per thread.
- **Vectors on SSD** can be stored as `f32`, `f16` or `bf16`.
- **Two layouts, chosen by record size:**
  - `inline`: the vector sits in the node's block. Used while a record fits
    in 4 sectors (16 KB: up to about 4,000 f32 dimensions).
  - `separate`: the graph and the vectors are stored apart, navigation reads
    only neighbor lists, and the `rerank` best candidates by PQ distance get
    their vectors read at the end. Used beyond that.

  The cutoff comes from the measurements below.
- **Building larger than RAM:** `DiskIndex::write` needs the whole graph and
  all vectors in memory; the partitioned build below does not.
- **Not yet:**
  - **Inserts and deletes:** step 6.
  - **Overlapping CPU work with reads:** see below.

Results: recall@10 against the published or exact ground truth, single
thread, beam width `W`, from a fresh process with direct I/O. In every run
the kernel read the bytes the index asked for (the `dev/app` column in the
logs is 1.00), so these are SSD reads, not page-cache hits. Hardware: two
Samsung PM981 NVMe drives in RAID 0 in a shared desktop-class machine
(i7-8700, 6 cores). The machine was heavily loaded (load average about 20
on 12 hardware threads) and the benchmark ran at the lowest CPU priority, so
CPU time is inflated and p99 is noisy. Reads were capped at 100k IOPS, which
single-thread runs do not reach.

**SIFT1M, inline f32** (0.85 GB file; RAM 47.5 B/vector; 1,000 queries).
With `W = 1`, recall and nodes expanded equal step 4's PQ-guided search on
all 10,000 queries: 0.9616 / 44.9 at L 40 and 0.9910 / 84.5 at L 80.

| W | L | recall | reads/q | rounds/q | p50 / p99 ms | CPU per query, user + kernel |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40 | 0.957 | 45 | 45 | 4.4 / 7.4 | 0.55 + 0.56 ms |
| 1 | 80 | 0.990 | 85 | 85 | 8.2 / 12.1 | 1.00 + 1.02 ms |
| 4 | 40 | 0.961 | 54 | 15 | 2.2 / 5.6 | 0.52 + 0.48 ms |
| 4 | 80 | 0.991 | 93 | 24 | 3.6 / 7.6 | 0.83 + 0.77 ms |
| 4 | 160 | 0.998 | 172 | 44 | 6.5 / 10.8 | 1.36 + 1.47 ms |
| 8 | 40 | 0.965 | 68 | 10 | 2.1 / 5.7 | 0.60 + 0.52 ms |
| 8 | 80 | 0.992 | 105 | 15 | 3.2 / 7.2 | 0.92 + 0.80 ms |
| 8 | 160 | 0.998 | 183 | 24 | 5.2 / 10.1 | 1.37 + 1.37 ms |

Caching the 100,000 nodes nearest the entry point (81 MB of RAM) cuts reads
by 18-40% (more at small L). At `W = 8` it gives 1.4 / 2.5 / 4.3 ms p50 at
L 40 / 80 / 160.
With positioned reads instead of io_uring, `W = 4` at L 80 takes 9.0 ms: the
four reads of a round then run one after another.

**GIST1M** (960 dims, 1,000 queries; RAM 135 B/vector). With `W = 1`, L 160
gives 0.9588 recall and 164.3 nodes expanded, equal to step 4's PQ-guided
search.

| file | W | L | recall | reads/q | KB/q | p50 / p99 ms |
|---|---:|---:|---:|---:|---:|---:|
| inline f32 (8 KB blocks, 8.2 GB) | 4 | 80 | 0.897 | 94 | 752 | 5.5 / 10.4 |
| | 4 | 160 | 0.959 | 172 | 1,376 | 10.0 / 52.5 |
| | 4 | 320 | 0.985 | 331 | 2,644 | 18.1 / 27.2 |
| | 8 | 160 | 0.961 | 183 | 1,463 | 8.4 / 14.9 |
| | 8 | 320 | 0.985 | 340 | 2,721 | 15.4 / 24.1 |
| inline f16 (4 KB blocks, 4.2 GB) | 4 | 160 | 0.959 | 172 | 688 | 8.5 / 15.5 |
| | 4 | 320 | 0.984 | 331 | 1,322 | 16.0 / 23.4 |
| separate f32, re-rank all | 4 | 160 | 0.959 | 344 | 1,376 | 9.9 / 17.4 |
| separate f32, re-rank 50 | 4 | 160 | 0.878 | 222 | 888 | 8.5 / 14.8 |
| separate f32, re-rank 130 | 4 | 320 | 0.972 | 461 | 1,842 | 16.4 / 24.7 |

**Higher dimensions** (synthetic, as in step 3: 64-dim intrinsic structure
plus noise; `W = 4`; PQ trained with 15 k-means iterations):

| data | file | L | recall | reads/q | KB/q | p50 ms |
|---|---|---:|---:|---:|---:|---:|
| 2048 dims, 50k vectors, PQ 256 B | inline f32 (12 KB blocks) | 80 | 0.941 | 88 | 1,050 | 7.0 |
| | | 160 | 0.985 | 165 | 1,985 | 12.4 |
| | separate f32, re-rank all | 160 | 0.985 | 331 | 1,985 | 12.6 |
| | separate f32, re-rank 130 | 160 | 0.984 | 295 | 1,702 | 11.9 |
| | separate f32, re-rank 100 | 160 | 0.983 | 265 | 1,462 | 11.8 |
| | separate f32, re-rank 50 | 160 | 0.962 | 215 | 1,062 | 10.3 |
| 8192 dims, 20k vectors, PQ 512 B | inline f32 (36 KB blocks) | 80 | 0.957 | 87 | 3,115 | 15.7 |
| | | 160 | 0.991 | 165 | 5,940 | 24.6 |
| | inline f16 (20 KB blocks) | 160 | 0.991 | 165 | 3,300 | 25.1 |
| | separate f32, re-rank all | 80 | 0.957 | 173 | 3,115 | 13.8 |
| | separate f32, re-rank all | 160 | 0.991 | 330 | 5,940 | 21.3 |
| | separate f32, re-rank 100 | 160 | 0.977 | 265 | 3,860 | 18.2 |
| | separate f32, re-rank 50 | 160 | 0.910 | 215 | 2,260 | 16.6 |

What this shows:

- **The disk search is the step 4 search, read from SSD.** At `W = 1` it
  matches step 4's PQ-guided recall and nodes expanded exactly on SIFT1M,
  GIST1M and the 8192-dim set (0.992 and 161.9 at L 160, when built with
  step 4's PQ settings: `--iters 10`). Latency is then about one read per hop: 76 µs of SSD time (the
  drives' own counters) plus about 25 µs of CPU.
- **The beam width is the main latency knob after `L`.** `W = 4` cuts SIFT1M
  at L 80 from 8.2 to 3.6 ms for 10% more reads. `W = 8` gains less (3.2 ms).
- **CPU, not the SSD, now limits wide beams.** Each read costs 16-25 µs of
  CPU, about half of it in the kernel. At `W = 8` that is about half the
  query time.
  - Part of the user-space cost comes from the thread sleeping during every
    read: SIFT1M at `W = 1` spends 12.7 µs of user CPU per hop with direct
    I/O, against 7.1 µs when the same reads come from the page cache without
    sleeping. Most likely other processes evict its caches while it sleeps;
    hardware counters would confirm it.
  - Kernel time per read is the same with io_uring and positioned reads,
    and with or without the IOPS cap. It is therefore spent in the shared
    block-I/O and wake-up path.
  - Ideas to try:
    - process each block while the round's other reads are still in
      flight;
    - io_uring registered buffers and files (a hypothesis for cutting the
      kernel cost, not yet measured);
    - lighter per-hop work.
- **The better layout depends on how many sectors a vector spans.**
  - **960 dims (2-sector records):** inline needs fewer reads and fewer
    bytes at equal recall.
  - **2048 dims (3 sectors):** about a tie. Separate re-ranking 100-130
    nodes reads 14-26% fewer bytes in 60-80% more reads, at about the
    same recall and latency.
  - **8192 dims (9 sectors):** separate is faster: 13.8 vs 15.7 ms at the
    same recall and bytes (L 80), and 21.3 vs 24.6 ms at L 160.

  Hence the 4-sector cutoff, which falls between measured points (nothing
  between 2048 and 8192 dims was tested). Separate needs re-ranking of
  about 0.8 x L nodes to stay within 0.01 recall of re-ranking all.
  Re-ranking only 50 loses up to 0.08, because PQ orders the candidates too
  coarsely to pick the final ones. The default (`rerank = 0`) re-ranks
  every expanded node, so the separate layout never returns lower recall
  than inline.
- **f16 storage halves SSD bytes per query at no measurable recall cost**
  (GIST1M 0.959 either way; 8192 dims 0.991 either way). At 8192 dims it is
  not yet faster, because f16 is decoded in software; F16C decoding is a
  step 3 follow-up.
- **RAM is codes plus ids:** 47.5 B/vector for SIFT1M, 135 B for GIST1M. At
  8192 dims and 20k vectors it is 939 B, of which 419 B is the shared PQ
  codebook (8 MB) spread over few vectors. The search process's RSS grows by
  exactly that.
- **Against the proposed targets** (recall@10 >= 0.95 at single-digit-ms
  p99, single thread):
  - SIFT1M meets them: 0.961 at 5.6 ms p99 (`W = 4`, L 40).
  - GIST1M, 2048 and 8192 dims do not on this machine: p99 is 15-22 ms at
    recall >= 0.95. Part of that is the loaded box (CPU time is inflated).
    The rest needs the CPU work above, and f16 with hardware decoding.
- **Not shown here:** 100M or more vectors, and many threads. 100M vectors
  needs a dedicated machine. Multi-thread QPS on this shared RAID would
  mostly measure the IOPS cap.

Commands (`S`, `G` as above; `--layout auto` is the default):

```
$B synth --out emb2048 --dim 2048 --n 50000 --nq 200 --intrinsic 64 --noise 0.05 --seed 23
$B groundtruth --base emb2048/base.fvecs --query emb2048/query.fvecs --out emb2048/gt.ivecs --k 10
$B disk-build --base emb2048/base.fvecs --out emb2048 --pq-m 256 --threads 6 --layout inline,separate
$B disk-build --base emb8192/base.fvecs --out emb8192 --pq-m 512 --threads 6 \
      --layout inline,separate --codec f32,f16
$B disk-build --base $S/sift_base.fvecs --out sift1m --pq-m 32 --threads 6
$B disk-search --index sift1m.inline.f32.vdx --query $S/sift_query.fvecs \
      --gt $S/sift_groundtruth.ivecs --nq 1000 --threads 1 --w 1,4,8 --l 40,80,160
$B disk-build --base $G/gist_base.fvecs --out gist1m --pq-m 120 --threads 6 \
      --layout inline,separate --codec f32,f16
$B disk-search --index gist1m.inline.f16.vdx --query $G/gist_query.fvecs \
      --gt $G/gist_groundtruth.ivecs --threads 1 --w 4 --l 80,160,320
$B disk-search --index gist1m.separate.f32.vdx ... --rerank 0,20,50   # 0 = re-rank all
$B disk-search ... --cache 100000            # keep 100k nodes near the entry in RAM
```

## Partitioned build (step 5)

`DiskIndex::build_partitioned` writes the SSD index straight from a vector
file. It never loads the file, and holds only one partition's vectors and
graph in RAM at a time. This is the merged-partition build of DiskANN. Files
are read by position: `.fvecs`, `.bvecs`, and the `.fbin` / `.u8bin` files
of big-ann-benchmarks. Any `VectorSource` works.

1. **Scan** every row: check it, prepare it for the metric (normalize for
   cosine) and hash it. Rows with equal hashes are compared. Identical rows
   share one node, and nodes are numbered in row order, as in the in-memory
   build.
2. **Train** PQ, and k-means centroids for the partitions, on one random
   sample of nodes. Given a memory budget, the number of partitions grows
   until the largest one, estimated on the sample with a 10% margin, fits.
3. **Encode:** a second read PQ-encodes every node and assigns it to its 2
   nearest centroids (`overlap`), so neighboring partitions share nodes.
4. **Build the partitions**, one at a time: read the partition's vectors,
   build its Vamana graph in RAM, and write its out-lists, with their exact
   distances, to a scratch file.
5. **Merge:** a node's two out-lists are joined and pruned back to R with
   RobustPrune. Distances to the node are exact (kept from the partition
   builds). Distances between candidates come from their PQ codes, because
   their vectors are not in RAM. A walk from the entry then finds the nodes
   the merged graph does not reach. Each gets an in-edge from its nearest
   reachable out-neighbor with a free slot.
6. **Write:** a third read streams every node, with its merged out-list,
   into the index file.

- **RAM** is the budget (one partition: about `4·dim + 8·R + 96` bytes per
  node), plus `m + 4` bytes per vector for the PQ codes and partition
  numbers, plus read buffers of up to 32 MB per thread. Before that, the
  scan needs 16 bytes per row; after it, the reachability walk needs up to
  4 bytes per node. The plan sizes partitions from a sample with a 10%
  margin. A partition that still comes out over the budget is built anyway,
  with a warning in the progress log.
- **Scratch disk** holds the partition out-lists and the merged graph (under
  1 GB for SIFT1M and GIST1M). It is removed when the build ends, also on
  error. Free space is checked first.
- **The same file for any thread count.** For this, k-means now sums each
  cluster in row order, so PQ codebooks no longer depend on the thread count
  either. SIFT1M's codebooks did not change: its values are integers, so the
  sums are exact in any order.
- **One partition reproduces the in-memory build byte for byte.** A test
  checks this for every metric, with identical rows and a sampled PQ. At
  full size, the 0.85 GB SIFT1M file is identical to the one measured above.

Results: recall@10 and SSD reads per query, `W = 4`, single thread.
SIFT1M was searched from the page cache on all 10,000 queries; the graph
alone decides recall and reads, so they do not depend on where the blocks
come from. GIST1M was searched with direct I/O on its 1,000 queries. Build
times are from the loaded machine and vary by up to 1.6x between runs: four
10-partition SIFT1M builds with the same partition work (the PQ merge twice,
two other merges) spent 187-300 s building partitions, and three runs of one
GIST1M budget build (with an earlier planner that made 23 partitions) took
1,131 to 1,483 s. The random-merge
rows were measured before the partition reads were spread over the
threads, which added up to 0.1 GB of read buffers.

**SIFT1M** (R 64, build L 100, PQ 32 B; recall / reads per query):

| build | mean degree | build time | peak RSS | L 40 | L 80 | L 160 |
|---|---:|---:|---:|---:|---:|---:|
| one partition (= in-memory build) | 45.5 | 135 s | 1.25 GB | 0.9653 / 54.4 | 0.9914 / 93.1 | 0.9981 / 172.0 |
| 10 partitions, PQ merge | 52.4 | 208 s | 0.50 GB | 0.9720 / 55.1 | 0.9944 / 93.9 | 0.9988 / 172.8 |
| 10 partitions, random merge, partition R 43 (DiskANN's) | 54.4 | 293 s | 0.40 GB | 0.9709 / 54.8 | 0.9937 / 93.5 | 0.9988 / 172.5 |
| 10 partitions, random merge, partition R 64 | 58.0 | 329 s | 0.44 GB | 0.9666 / 54.7 | 0.9924 / 93.5 | 0.9985 / 172.5 |
| one partition, two passes (`--passes 2`) | 45.1 | 256 s | 1.20 GB | 0.9700 / 53.8 | 0.9934 / 92.5 | 0.9987 / 171.5 |

**GIST1M** (960 dims, R 64, build L 100, PQ 120 B, inline f32 with 8 KB
blocks; recall / reads per query):

| build | mean degree | build time | peak RSS | L 80 | L 160 | L 320 |
|---|---:|---:|---:|---:|---:|---:|
| one partition (= in-memory build) | 33.4 | 1,013 s | 4.41 GB | 0.897 / 94.0 | 0.959 / 172.0 | 0.985 / 330.5 |
| 1 GB budget, in a cgroup capped at 1.6 GB: 20 partitions | 38.6 | 1,160 s | 1.10 GB | 0.911 / 94.3 | 0.966 / 172.3 | 0.990 / 330.9 |

What this shows:

- **RAM follows the budget, not the collection.** GIST1M's vectors take
  3.84 GB. The partitioned build ran in a cgroup capped at 1.6 GB and peaked
  at 1.10 GB, against 4.41 GB for the one-partition build. SIFT1M with 10
  partitions peaked at 0.50 GB, against 1.25 GB.
- **No recall is lost to partitioning.** Every node is inserted into two
  partitions, and the merged graph matches a two-pass in-memory build
  (SIFT1M: 0.9720 against 0.9700 at L 40, for 2% more reads). It therefore
  beats the default one-pass build: GIST1M reaches 0.966 against 0.959 at
  L 160, for the same reads. The merged graph has more edges (SIFT1M mean
  degree 52.4 against 45.5). That costs PQ lookups in RAM, not SSD reads.
- **The PQ merge is at least as good as DiskANN's random truncation,** with
  fewer edges, on SIFT1M: 0.001 higher recall at L 40 and L 80 and equal at
  L 160, at mean degree 52.4 against 54.4. Truncating full-degree partition
  lists at random did worst. `--merge random` remains available.
- **The cost is 1.1-1.7x the build time of a one-pass build** in these runs
  (SIFT1M 208-227 s against 135 s; GIST1M 1,131-1,483 s against 1,013 s),
  because each node is inserted twice, into smaller graphs. That is about
  what the two-pass build costs (SIFT1M: 256 s). Scratch disk stayed under
  1 GB.
- **The merge needs the reachability repair.** GIST1M's merged graph left
  2,455 nodes (0.25%) unreachable from the entry; 2,414 new in-edges made
  them all reachable. SIFT1M had none.
- **Partitions are uneven.** GIST1M's k-means partitions held 12k to 189k
  nodes (mean 98k, budget 225k), so the build needed 20 partitions where 9
  would hold the nodes at even sizes. Extra partitions cost little build
  time, since each node is still inserted twice. Each partition reads its
  rows, and gaps of up to 16 KB between them, so on a cold cache extra
  partitions add reads (not measured here: the input files were in the page
  cache). Splitting the largest clusters would even them out.
- **At 100M vectors** (estimated, not run): for 128 dims, R 64 and 32-byte
  PQ codes, RAM peaks at the budget plus about 3.6 GB for codes and
  partition numbers. The scan needs 1.6 GB before that, and the
  reachability walk up to 0.4 GB after the partitions. Scratch disk is up to
  about 130 GB and the index about 82 GB.

Commands (`S`, `G` as above):

```
$B disk-build --base $S/sift_base.fvecs --out sift1m-p10 --pq-m 32 --threads 6 --partitions 10
$B disk-build --base $G/gist_base.fvecs --out gist1m-1g --pq-m 120 --threads 6 --build-mem 1
$B disk-build ... --partitions 1          # = the in-memory build, byte for byte
$B disk-build ... --merge random --part-degree 43   # DiskANN's merge
```

## Collections (step 6)

A `Collection` is the storage engine of one shard: a directory of immutable
segment files plus a write-ahead log (WAL), tied together by a manifest.

```rust
let mut cfg = CollectionConfig::new(768, Metric::Cosine);
cfg.fields = vec![Field::keyword("lang"), Field::int("year")];
let c = Collection::create("data/docs", cfg, OpenOptions::default())?;
c.upsert(42, &vector, &Attrs::new().keyword("lang", "en").int("year", 2024))?;
c.delete(7)?;
let hits = c.search(&query, 10, &SearchOptions::default())?;
let filtered = SearchOptions {
    filter: Some(Filter::and([Filter::keyword_in("lang", &["en", "de"]), Filter::range("year", 2020..)])),
    ..Default::default()
};
let recent = c.search(&query, 10, &filtered)?;
```

Segments are flushed and indexed by background threads (step 7b); with
`OpenOptions { background: false, .. }` the write that fills a segment flushes
it, and `c.maintain()` runs the index builds and compactions that are due.

```
<dir>/MANIFEST            which files make up the collection (replaced atomically)
      LOCK                held while a process has the collection open
      wal-<gen>.log       writes not flushed yet (several, while a flush runs)
      seg-<id>.flat       sealed segment: ids + vectors, searched exactly in RAM
      seg-<id>.vdx        sealed segment: SSD index (step 5)
      seg-<id>.attrs      attributes, when the collection declares fields: rows,
                          and the column index that filters read
      seg-<id>.del-<gen>  tombstones: the segment's deleted rows
```

- **Write path.** A batch of upserts and deletes is checked, appended to the
  WAL as one checksummed record (fsynced with `Durability::Sync`), then
  applied. An upsert deletes the id's current row, wherever it is, and
  appends the new row to the growing segment, which lives in RAM and is
  searched exactly. Once it holds `seal_bytes` of vectors it is frozen, and a
  new growing segment and a new WAL start. A flush writes the frozen
  segment's live rows as a flat segment file and the tombstones of every
  segment that gained deletes, and commits all of it with a new manifest,
  which no longer needs the frozen segment's WAL.
- **Deletes** are tombstones per row position in a segment, not per id, so a
  deleted row never hides a later insert of the same id. Identical vectors
  share one graph node in an SSD index but keep their own positions, ids,
  attributes and tombstones.
- **Index builds and compaction** (`maintain`) replace sealed segments with
  one new segment built from their live rows, through the partitioned
  build. The flat segments become an SSD index once they hold `index_bytes`
  of live vectors. Flat segments that take as much, at least half of it
  deleted rows, are merged into one flat segment (step 7b). An SSD segment
  with 20% of its rows deleted is rebuilt without them, and the two smallest
  SSD segments are merged while there are more than `max_indexed_segments`.
  Rows are read back as stored, already
  prepared for the metric, and never normalized again. Rebuilding unchanged
  rows therefore writes the same file byte for byte (a test checks f32, f16
  and bf16), and rows do not drift over repeated compactions. For cosine,
  `get` returns the stored unit-length vector.
- **Search** runs on every segment and merges the top-k lists: every segment
  returns exact distances. Deleted rows are skipped when results are
  collected, but the graph walk still passes through them. So the
  candidate list size L of an SSD segment is scaled by 1 / (its live
  fraction), up to 8x.
- **Crash safety.** New files are written under new names and fsynced, then
  committed by renaming the manifest into place; old files are removed only
  after that. Opening replays the WAL on top of the manifest and removes
  files of ours that no manifest refers to; it leaves other files alone.
  WAL recovery is point-in-time: replay stops at the first record that is
  incomplete or fails its checksum, and the log is cut there. After a crash
  that is the batch being written. Records written without an fsync
  (`Buffered`) can also reach the disk out of order, so a bad record
  followed by good ones does not prove damage. A good record written with
  `Sync` after a bad one does: its fsync made the bad record durable before
  it was damaged. Then opening fails rather than dropping acknowledged
  writes, and names the byte to cut the log at. A failed fsync poisons the
  log until the collection is reopened. The WAL is also rotated once it
  reaches twice `seal_bytes`, so a workload of deletes alone flushes too.
- **Shared between threads** (step 7a, see [Concurrency](#concurrency-step-7a)):
  every method takes `&self`; searches read a snapshot, writers share WAL
  writes and fsyncs, and `maintain` runs beside both.
- **RAM per row of an SSD segment**, beyond the index: 4 bytes for the
  id lookup (positions sorted by id) and 1 bit of tombstone. Flat and
  growing segments hold their vectors. With fields declared, every sealed
  segment also holds its attribute columns: 12 bytes per integer value
  (measured), and 4 bytes per keyword value plus each distinct keyword once
  (see Filters below).

The tests check the engine against a plain `HashMap` reference:
- random upserts, deletes, flushes and reopens, with exact search results
  compared after every step;
- the WAL cut at every byte offset, keeping exactly the whole batches before
  the cut;
- the two directory states a crash can leave around the manifest commit, for
  both a flush and an index build, each opening to the same rows;
- that each id is live in at most one place.

Results on SIFT1M (ids = row numbers, one process, this loaded machine).
Ingest, with the growing segment sealed at 64 MiB:

| durability | batch | rows/s | batch latency p50 / p99 |
|---|---:|---:|---:|
| buffered | 1,000 | 405,000 | 0.84 / 3.6 ms |
| sync (fdatasync per batch) | 1 | 201 | 5.6 / 9.6 ms |
| sync | 10 | 1,850 | 5.7 / 9.5 ms |
| sync | 100 | 16,700 | 6.2 / 10.8 ms |
| sync | 1,000 | 90,800 | 8.9 / 16.8 ms |

- An fsync on this machine's RAID takes about 5.6 ms, so durable writes need
  batches, or several writers: since step 7a their batches share fsyncs
  (group commit, see [Concurrency](#concurrency-step-7a)).
- A flush (64 MiB written and fsynced) stalls its batch for 0.24-0.3 s.
- Reopening took 0.08 s with an empty WAL and 0.30 s to replay 76,000
  writes (40 MB).
- Index builds run inside `maintain`. In these step 6 runs they blocked
  writes (since step 7a they run beside them). Ingest with
  `maintain` after every batch (index at 128 MiB) averaged 4,950 rows/s:
  three builds of 264,000 rows took 199 of the 202 s. Merging all 1M rows
  into one SSD index (`optimize`) took 172 s; step 5's build of the same
  index took 135 s, a difference within this machine's run-to-run
  variation.

Search with deletes. One SSD segment of 1M rows, `W = 4`, page cache, one
thread, 1,000 queries, warm. Recall@10 is against exact search over the live
rows, followed by reads per query:

| deleted | L 40, fixed | L 40, adapted | L 80, fixed | L 80, adapted |
|---|---:|---:|---:|---:|
| 0% | 0.9648 / 54 | 0.9648 / 54 | 0.9907 / 93 | 0.9907 / 93 |
| 10% | 0.9587 / 54 | 0.9667 / 59 | 0.9893 / 93 | 0.9915 / 102 |
| 30% | 0.9403 / 54 | 0.9690 / 72 | 0.9851 / 93 | 0.9942 / 127 |
| 50% | 0.8977 / 54 | 0.9766 / 93 | 0.9766 / 93 | 0.9967 / 172 |
| 50%, then compacted (500k rows, 122 s) | 0.9684 / 53 | | 0.9943 / 92 | |

- **Scaling L by 1 / live fraction keeps recall at or above the undeleted
  index's at the same L,** for 1 / (live fraction) times the reads. The
  deletes here are uniformly random; deletes concentrated in one region of
  the space were not measured.
- **At equal reads, deleted rows still cost recall:** 0.9766 at 50% deleted
  against 0.9907 undeleted, both at 93 reads, because half the expanded
  nodes cannot be returned. Compaction removes the cost. The default
  rebuilds a segment at 20% deleted.

Several segments. The same 1M rows ingested with `maintain` after every
batch. With the index threshold at 128 MiB (and seal at 64 MiB) that gave
three SSD segments of 264,000 rows, a flat segment of 132,000 and 76,000
growing rows. At the defaults (seal 32 MiB, index 64 MiB) it gave seven SSD
segments of 132,000 rows, a flat segment of 66,000 and 10,000 growing rows;
that ingest averaged 5,550 rows/s, 177 of its 180 s in seven index builds.

| segments | rows scanned exactly | SSD reads / rounds per query | recall@10, L 40 | page cache: mean / p99 ms | direct I/O: mean / p99 ms |
|---|---:|---:|---:|---:|---:|
| 1 SSD | 0 | 54 / 14.8 | 0.9648 | 0.33 / 0.63 | 2.19 / 2.87 |
| 3 SSD + flat + growing | 208,000 | 155 / 42.4 | 0.9957 | 12.5 / 24.2 | not run |
| 7 SSD + flat + growing (defaults) | 76,000 | 356 / 97.2 | 0.9975 | 3.76 / 4.86 | 15.3 / 19.7 |
| flat + growing only (200k rows) | 200,000 | 0 | 1.0000 | 6.10 / 7.99 | |

Direct I/O here is capped at 100,000 reads/s, as in step 5.

- **The exact scan of the RAM segments costs about 30 ns per 128-dim row**
  on one core: 6.1 ms for 200,000 rows on their own. The defaults are
  therefore lower than in the first runs: seal at 32 MiB and index at
  64 MiB. With `maintain` run after each flush, at most about 96 MiB of
  vectors (196,000 SIFT rows, about 6 ms) is scanned per search; the
  defaults run had 76,000 rows in RAM.
- **On SSD, segments add latency, because they are searched one after
  another.** Each SSD segment waits for its own rounds of reads: 97 rounds
  for seven segments at L 40, 15.3 ms with direct I/O, against 2.2 ms for
  one segment. Each segment is searched with the full L, so more segments
  give higher recall at a fixed L, but one segment at L 160 (0.9989, 6.3 ms)
  beats seven at L 40 (0.9975, 15.3 ms). `max_indexed_segments` (default 8)
  trades rebuild work against this. Step 7b was to search segments
  concurrently, with their reads in flight together, to scan in parallel,
  and to build indexes in the background (done, see
  [below](#background-work-parallel-search-and-databases-step-7b)).

### Filters

A search can carry a filter on the collection's declared fields (see the
example above).

- **Model.** Fields are integers or keywords, declared per collection; a row
  may hold several values for a field. Conditions are `Eq` and `In`
  (integers and keywords) and `Range` (integers, inclusive), combined with
  `And`, `Or` and `Not`. A row matches a condition when any of its values
  does. A row with no value for the field matches no condition on it, and
  so matches its `Not`. An empty `In` or `Or` matches nothing, an empty
  `And` everything. An unknown field, a value of the wrong type, or a tree
  deeper than 32 is an error, not an empty result.
- **Storage.** The WAL keeps attributes as written, with field names;
  everywhere else rows are encoded with field numbers. A sealed segment's
  `.attrs` file holds its rows, each with its own checksum, and a column
  index: per keyword field, the distinct keywords with sorted position
  lists; per integer field, the values sorted with their positions. Only
  the columns are loaded into RAM: 12 bytes per integer value, 4 bytes per
  keyword value plus each distinct keyword once. `get` reads its row from
  the file. The growing segment keeps columns of its own, added to row by
  row.
- **Evaluation.** Per segment, the filter becomes a bitmap of matching
  positions, less the deleted ones. When a field holds one value per row
  and most rows match, the bitmap starts full and the other rows are
  cleared, so the work follows the smaller side. A filter matching every
  row then costs about what no filter costs on SIFT1M at L 40: a mean of
  0.29 against 0.27 ms per query over four interleaved pairs (each run
  0.23-0.34 ms). Before this path it cost 3.09 against 0.48 ms (measured
  under heavier load).
- **RAM segments** score only the matching rows, row by row in blocks where
  few match.
- **An SSD segment is searched one of two ways** (`FilterMode`):
  - *walk*: the graph search, returning only matching rows, with L scaled
    by 1 / (fraction matching), up to 16x;
  - *scan*: no graph. PQ distances to every matching row, from the codes in
    RAM, then the best L read from SSD and scored exactly.
- **`Auto`, the default, chooses per segment:**
  - it scans when fewer than 1/16 of the rows match;
  - otherwise it compares the walk's expected rounds (about L' / W + 4, at
    the scaled L') times what a round costs in scanned rows with the
    number of matching rows. A round is worth 3,500 rows with direct I/O
    and 400 from the page cache; both constants were measured, below;
  - it falls back to the scan when a walk expands fewer than L/2 matching
    nodes, as happens when the matches lie away from the query.

Tests:
- random filter trees over random multi-valued rows, and over rows with one
  value per field, evaluated by the growing segment's columns and by the
  columns written to a file and read back, against a direct evaluation
  (`Filter::matches`) row by row;
- filtered search on growing and flat segments equal to the reference over
  random writes, deletes, flushes and reopens;
- SSD segments in each mode and both layouts: no match, fewer matches than
  k, one matching row among identical vectors, deletes, every row matching
  (the same results as no filter), recall at 5% matching, and a reopen;
- damaged attribute files, invalid filters, and a WAL whose attributes do
  not fit the fields;
- the walk-or-scan choice at the measured points.

Results on SIFT1M: one SSD segment of 1M rows, `W = 4`, k = 10, 1 thread,
300 queries (200 with direct I/O, capped at 100,000 reads/s). Two integer
labels per row: `u`, uniform in 0..1000 and unrelated to the vector, and
`p`, the row's bucket (0..1000) by the rank of its projection onto a fixed
random direction. Filters are ranges matching a given fraction of rows:
- on `u`: matches spread over the space;
- on `p`, from one end: matches in one slab of the space, away from most
  queries (the hard case for a walk);
- on `p`, around each query's own bucket ("near"): matches near the query.

Recall@10 is against exact search over the matching rows. Without a filter,
L 40 gives 0.964 recall in about 0.25 ms from the page cache and 2.2 ms with
direct I/O.

Uniform labels, L 40, recall@10 / mean ms:

| matching | walk, page cache | scan, page cache | auto, page cache | walk, direct | scan, direct | auto, direct |
|---|---:|---:|---:|---:|---:|---:|
| 0.1% | 0.068 / 2.10 | 1.000 / 0.15 | 1.000 / 0.14 | 0.065 / 24.2 | 1.000 / 0.57 | 1.000 / 0.49 |
| 1% | 0.599 / 2.09 | 0.997 / 0.49 | 0.997 / 0.50 | 0.616 / 20.7 | 0.999 / 0.83 | 0.999 / 1.03 |
| 5% | 0.991 / 2.15 | 0.997 / 1.95 | 0.997 / 1.95 | 0.991 / 22.8 | 0.998 / 2.64 | |
| 10% | 0.989 / 1.61 | 0.991 / 3.44 | 0.989 / 1.57 | 0.991 / 13.7 | 0.995 / 3.75 | 0.995 / 3.82 |
| 20% | 0.988 / 1.00 | 0.990 / 6.11 | 0.988 / 1.10 | 0.990 / 7.94 | 0.991 / 6.63 | 0.990 / 7.69 |
| 50% | 0.976 / 0.87 | 0.986 / 14.2 | 0.976 / 0.98 | 0.981 / 4.09 | 0.988 / 15.1 | 0.981 / 3.82 |

(The direct 5% row is from the calibration run: the same code before the
final constants, at a load average of 5-10.)

Matches in one slab (L 40, recall@10 / mean ms; auto's share of queries
that fell back to a scan in brackets):

| matching | far slab: walk | far slab: auto | near slab: walk | near slab: auto |
|---|---:|---:|---:|---:|
| page cache, 1% | 0.177 / 2.05 | 0.998 / 0.46 (scan) | 0.803 / 2.16 | 0.998 / 0.63 (scan) |
| page cache, 10% | 0.455 / 1.48 | 0.992 / 3.58 (65%) | 0.993 / 1.49 | 0.995 / 1.52 (2%) |
| page cache, 20% | 0.566 / 0.99 | 0.990 / 4.50 (57%) | 0.991 / 1.01 | 0.993 / 1.06 (1%) |
| page cache, 50% | 0.792 / 0.93 | 0.975 / 5.27 (32%) | 0.989 / 1.10 | 0.989 / 1.22 (0%) |
| direct, 10% | 0.401 / 14.9 | 0.993 / 3.63 (scan) | 0.995 / 13.8 | 0.992 / 4.09 (scan) |
| direct, 20% | 0.523 / 7.90 | 0.990 / 12.1 (63%) | 0.995 / 7.77 | 0.997 / 7.60 (1%) |
| direct, 50% | 0.773 / 4.28 | 0.980 / 11.0 (35%) | 0.994 / 4.12 | 0.994 / 4.25 (0%) |

(The near-slab 50% times are means of three interleaved runs of walk and
auto. A single run had shown auto at 1.83 and 4.73 ms; the interleaved runs
put it within noise of the walk it chose.)

What this shows:
- **Neither way works alone.**
  - The walk fails on selective filters. Below about 5% matching, it
    cannot reach enough matching nodes (recall 0.07 at 0.1%, even at 16x
    L), and it takes the most rounds there (164 at L 40).
  - The scan's cost grows with the matching rows, at about 30-40 ns per
    row from RAM: 14 ms for 500,000 rows. So it loses once many rows
    match.
- **With direct I/O the scan wins up to about 20% matching.** It waits on
  the SSD one to three times; the walk waits 50 to 160 times. From the page
  cache the scan wins up to about 5-10%.
- **Auto chose the faster way at every measured point with uniform or
  query-near labels, except one:** 20% matching, direct I/O, L 40. There
  it walked for both label kinds, 16% slower than a scan.
- **At the same L, the scan's recall is higher:** at L 40 about 0.99 up to
  20% matching and 0.98-0.99 at 50%, against 0.964 for an unfiltered walk.
  It re-ranks the L nearest by PQ among all matching rows.
- **When the matches lie away from the query, the walk fails at every
  selectivity** (0.79 recall at 50%). The fallback restores 0.975-0.99, at
  the cost of a walk and a scan for those queries. At 20% from the page
  cache that is 4.5 ms, against 1.0 ms for uniform labels.
- **Two integer fields cost 24 MB of RAM for 1M rows** (12 bytes per
  value). Ingest with them ran at 248,000 rows/s, against 310,000 without
  attributes, in interleaved pairs. The difference is WAL encoding and the
  growing segment's columns.
- **Profiling found the scan's cost was in finding nodes, not distances.**
  A position's node was found by stepping through the 8 MB id table, a
  cache miss per step. Now it comes from the short list of nodes that hold
  identical vectors (SIFT1M has 14,538 duplicate rows), and PQ codes are
  fetched ahead. Measured back to back under the same load, the last of
  these changes took the scan at 5% matching from 8.2 to 4.1 ms, and at
  0.1% from 0.71 to 0.24 ms (the first version took about 1.8 ms there).

Limits:
- **Calibration.** The constants come from one dataset (128 dimensions,
  32-byte codes) on one machine, one thread. The machine (6 cores, 12
  threads) was shared and heavily loaded during the first runs. The tables
  above are from a quieter period, at a load average of 2-7.
- **Only integer range filters were benchmarked.** Keyword filters and
  `In`, `And`, `Or` and `Not` trees are tested for correctness against the
  reference, not for speed.
- **Bitmap per segment and query.** It costs the segment's positions / 8
  bytes per filter term, 12.5 MB at 100M rows, and time in proportion to
  the matches. Sorted position lists for selective terms, and checking the
  filter per expanded node for walks, are the next step for large
  segments.
- **Keyword dictionary in RAM.** It is kept per segment, so keywords unique
  to each row (ids, say) cost their bytes plus 20 bytes each.
- **The graph does not know the labels.** A filter whose matches lie away
  from the query is detected only after a walk, by the fallback. Per-label
  entry points or label-aware edges (as in Filtered-DiskANN) would avoid
  the wasted walk.

Not yet: inserting into an SSD index in place instead of rebuilding it.

Commands (`B`, `S` as above):

```
$B coll-ingest --base $S/sift_base.fvecs --dir sift1m --pq-m 32 --optimize true
$B coll-ingest ... --durability sync --batch 100 --secs 5
$B coll-ingest ... --index-mb 128 --max-ssd 16 --maintain true
$B coll-search --from sift1m --dir work --base $S/sift_base.fvecs --query $S/sift_query.fvecs \
   --l 40,80,160 --delete 0,0.1,0.3,0.5 --adapt both --compact true --direct false
$B coll-ingest --base $S/sift_base.fvecs --dir sift1m-labels --pq-m 32 --labels true --optimize true
$B coll-filter --dir sift1m-labels --base $S/sift_base.fvecs --query $S/sift_query.fvecs --l 40,160 \
   --filter u,p,near --select 0.001,0.01,0.05,0.1,0.2,0.5 --mode graph,scan,auto --direct false
$B coll-filter --dir sift1m-labels --base $S/sift_base.fvecs --query $S/sift_query.fvecs --l 40 \
   --filter none,u,none,u --select 1.0 --mode graph --direct false   # overhead of a filter
```

## Concurrency (step 7a)

A `Collection` is `Send + Sync` and every method takes `&self`, so threads
share one through an `Arc` (or a scoped borrow). Searches, writes and
`maintain` run at the same time:

```rust
let c = Arc::new(Collection::open("data/docs", OpenOptions::default())?.0);
let builder = { let c = c.clone(); std::thread::spawn(move || c.maintain()) };
c.upsert(42, &vector, &Attrs::default())?;               // during the build
let hits = c.search(&query, 10, &SearchOptions::default())?; // sees the upsert
```

- **Snapshot reads.** Every write batch gets a sequence number. A search
  reads the segment list and the last applied sequence number together and
  sees exactly the batches up to it: a batch is there whole or not at all,
  and each id at most once (an upsert that moves an id to the growing
  segment is one batch). Growing rows carry the sequence numbers that
  inserted and deleted them. A delete in a sealed segment sets a tombstone
  bit in place, but first records its sequence number in a side table, so a
  search older than the delete still reads the bit as clear. Searches
  register their snapshots, and the side table keeps an entry only while a
  search older than it is open; `stats().delete_records` shows its size.
- **Index builds beside writes.** A build takes the live rows of its input
  segments as of one moment and works without locks. It commits in two
  steps. First it writes the rows deleted meanwhile to the new segment's
  tombstone file, makes the logs durable and installs the manifest. Then,
  under the writer lock, it takes over the deletes made since and replaces
  the segments. Segments flushed in the meantime keep their place. Files the
  build
  replaces are removed from the directory but stay open, so searches that
  started before the commit finish on them.
- **Group commit.** A writer prepares its batch outside any lock (vectors
  normalized, rows encoded, the WAL record built) and queues it. One
  waiting writer takes every queued batch (up to 16 MiB), writes them with
  one WAL write and, under `Sync`, one fsync, applies them, and wakes the
  others. Each batch stays its own WAL record, so a crash keeps or drops
  batches whole. Records after the first of a group are flagged: the
  recovery rule that a good `Sync` record after a bad one proves damage
  counts only a later group, because the rest of the bad record's own group
  shared its fsync and may be the torn part of the same write.
- **Short lock holds.** A search scans the growing segment 1 MiB of vectors
  at a time, each chunk under its own read lock, so a write waits for one
  chunk rather than a whole scan. The price: a search pauses at its next
  chunk for every write that queues meanwhile (measured below). Reading the
  growing segment without a lock would avoid both waits.

The tests:
- an old snapshot keeps its view (search, filtered search by walk and by
  scan, `get`) after upserts and deletes in each kind of segment, a flush,
  and a rebuild that removed the files it reads; its delete records are
  dropped once it closes;
- two writers upserting groups of 8 ids, three searching threads and a
  `maintain` loop (flushes, builds, compactions, merges): every search
  returns each id once, each group's 8 rows from one batch, and no group
  from an older batch than an earlier search saw; then a reopen;
- deletes and upserts during an index build, with flushes in between: the
  new segment leaves the deleted rows out in memory, after a crash right
  after the commit, and after a reopen;
- 16 writers with `Sync`: fewer fsyncs than batches, and every
  acknowledged write survives a reopen;
- the WAL damage rule with groups.

Each was checked against a broken engine: reading tombstones without
snapshots, not carrying deletes into a build, carrying them without writing
the tombstone file, and the step 6 damage rule each make a test fail.

Results on SIFT1M: one SSD segment of 1M rows, L 40, W 4, page cache,
i7-8700 (6 cores, 12 threads), this loaded machine, 4-5 s per row.
Searches from several threads (recall@10 0.9648 throughout):

| search threads | QPS | p50 ms | p99 ms |
|---:|---:|---:|---:|
| 1 | 4,365 | 0.228 | 0.345 |
| 2 | 8,599 | 0.232 | 0.329 |
| 4 | 16,592 | 0.240 | 0.347 |
| 8 | 25,260 | 0.312 | 0.449 |

With fresh rows and a writer. First 25,000 rows upserted into the growing
segment (rows already in the index, so also 25,000 tombstones), then a
writer upserting 5,000 rows/s in batches of 100 (`Buffered`):

| search threads | 25k growing rows: QPS / p50 ms | + writer: QPS / p50 ms | write batch p50 / p99 ms |
|---:|---:|---:|---:|
| 1 | 1,059 / 0.913 | 782 / 1.271 | 0.76 / 1.61 |
| 4 | 3,613 / 1.074 | 2,641 / 1.502 | 0.43 / 0.64 |
| 8 | 6,194 / 1.239 | 4,616 / 1.685 | 0.51 / 1.41 |

- **The scan of the fresh rows is the cost, not the locks.** 25,000 rows
  are 12.8 MB, more than the 12 MiB L3, and are scanned at about 29 ns per
  row, as in step 2: 0.7 ms per search. During the writer's 4 s the growing
  segment grows to 45,100 rows, which accounts for most of the added
  0.36-0.45 ms (about 0.28 ms of scan at the midpoint).
- **Scanning in chunks trades a little search tail for much shorter
  writes.** Against one read lock per scan, in interleaved runs with 4
  searching threads (6 runs each): write batch p99 0.58-0.95 ms against
  1.84-2.02 ms, and search p99 2.11-2.63 ms (median 2.22) against 1.96-2.22
  ms (median 2.04); QPS and p50 the same within 2%. With 8 searching
  threads (3 runs each): write p99 0.84-1.43 against 2.21-2.52 ms, search
  p99 2.78-3.03 against 2.60-3.01 ms. Chunks of 8 MiB kept most of the
  write cost (p99 1.36-1.78 ms) without a better search p99.
- **A writer going flat out, with nothing building indexes, is a limit:**
  one writer at 59,600 rows/s (batches of 100, 5 s, starting from the
  indexed 1M) left 4 searching threads at 716 QPS, p99 20 ms, against
  16,592 alone, because every sealed segment stays flat and is scanned in
  full until `maintain` indexes it. Index builds ran at 4,000-5,800 rows/s
  with 6 threads in step 6, far slower than such a writer. Background builds
  (step 7b) bound the unindexed rows by making such a writer wait (see steps
  7b and 7c).
- Fresh rows are therefore the next search cost to cut: an interim index or
  int8 codes for growing and flat rows (a quarter of the memory traffic), or
  smaller seals.

Durable writes from several threads (`Sync`, batches of 100, 5 s each):

| writer threads | rows/s | batches per fsync | batch p50 / p99 ms |
|---:|---:|---:|---:|
| 1 | 15,137 | 1.0 | 6.6 / 10.8 |
| 2 | 16,265 | 1.1 | 12.5 / 19.4 |
| 4 | 27,632 | 2.0 | 14.0 / 21.9 |
| 8 | 45,913 | 4.0 | 16.2 / 145 |
| 16 | 73,353 | 8.0 | 19.0 / 161 |

- **Two writers gain little:** each prepares its next batch while the
  other's fsync runs, so each logs alone. From four on, a group holds the
  batches that arrived during the previous fsync.
- **The maximum of 120-190 ms at every count is the inline seal:** writing
  and fsyncing a 32 MiB segment stalls every waiting writer. With 8 writers
  and seals off (`--seal-mb 1024`), the maximum was 36-47 ms against
  149-188 ms with them (2 interleaved runs each). A seal stalls about 1% of
  the batches at 8 writers, so it reaches the p99 in some runs (145 ms in
  the table) and not in others (22-27 ms in the runs with and without
  seals). Step 7b moved it to a background thread.

Search during an index build. 25% of the rows deleted at random, then
`maintain` compacts the segment to 750,000 rows with 2 build threads while 4
threads search, measured in 25 s windows:

| | QPS | recall@10 | p50 ms | p99 ms |
|---|---:|---:|---:|---:|
| before the build | 13,986 | 0.9716 | 0.286 | 0.414 |
| during the build (10 windows) | 11,512-13,537 | 0.9716 | 0.290-0.343 | 0.446-0.517 |
| after (750,000 rows, no tombstones) | 20,565 | 0.9684 | 0.190 | 0.287 |

- Searches slowed by up to 18% while the build used two of the six cores;
  the build took 260 s with the searches running throughout (207 s in a run
  that searched during its first 30 s only).
- Recall before the build is higher than after it because L is scaled up
  for the deleted rows (step 6), and those searches read more.

Commands (`B`, `S` as above; `--direct false` = page cache):

```
$B coll-concurrent --from sift1m --dir work --base $S/sift_base.fvecs --query $S/sift_query.fvecs \
   --l 40 --direct false --readers 1,2,4,8
$B coll-concurrent ... --readers 1,4,8 --secs 4 --growing 25000
$B coll-concurrent ... --readers 4 --secs 4 --growing 25000 --writers 1 --write-rate 5000   # also 1, 8
$B coll-ingest --base $S/sift_base.fvecs --dir work --durability sync --batch 100 --secs 5 --writers 16
$B coll-concurrent ... --readers 4 --secs 25 --churn 0.25 --build true --windows 10 --threads 2
```

## Background work, parallel search and databases (step 7b)

```rust
let (db, report) = Database::open("data", &DatabaseOptions::default())?;
let c = db.create_collection("docs", CollectionConfig::new(128, Metric::L2), 4)?; // 4 shards
c.write(vec![Op::Upsert(Row { id: 42, vector, attrs: Attrs::default() })])?;
let reply = c.search(&SearchRequest::new(query, 10, SearchOptions::default()))?;
```

**Flushes.** The write that fills the growing segment only freezes it: the
segment moves to a list of frozen ones and an empty one and a new log take
its place. A thread per collection flushes the frozen segments, oldest first,
while writes and searches go on.

- A flush writes exactly the state as of the freeze, not the state as of its
  commit. Later deletes stay in the later logs, which a reopen replays on
  top. If a flush wrote them too, they would survive a power loss that takes
  the (buffered) writes that made them, and an upsert could lose its old row
  without gaining its new one.
- There may be several logs, then. The manifest names the first one still
  needed, and the recovery rule holds across them: buffered writes lost at
  the end of one log drop the logs after it, unless one of those holds a
  `Sync` record, which proves damage and fails the open.
- Two segments may wait to be flushed. When a third is full, writes wait.
  If flushes fail (a full disk, say) they are retried with growing pauses,
  the growing segment is left to grow to 4 times its size, and then writes
  get the error.

**Builds.** Two more threads per collection run the jobs that are due: one
those of flat segments (index builds, and merges of flat segments that are
mostly deleted), the other those of SSD segments (compactions and merges).
Writes are slowed down, and at `max_unindexed_bytes` (1 GiB) of vectors in no
index stopped, while an index build is due (see step 7d; in step 7b they ran
at full speed until they were stopped). A write that has waited for
`max_write_wait` (off by default) fails with `Unavailable`.
Dropping a collection stops its builds early. All collections of a database
share a limit of builds at once (default 1) per kind, and the builds that run
share the build threads (half of the CPUs unless told otherwise).

**Parallel search.** A search is split into tasks: one per SSD segment, one
per 4 MiB of vectors in RAM, and one per shard asked elsewhere. The calling
thread works through them, with help from a `SearchPool` shared by all
collections of a database:

- Tasks that mostly wait (walks with direct I/O, remote shards) take any idle
  thread of the pool (two per CPU).
- Tasks that compute share a number of CPUs, a quarter of the machine by
  default, evenly between the searches that have some. With as many searches
  as that, each is left to its own thread, as without a pool. Helpers check
  their share at every task and move on.
- A search never waits for the pool, and its results do not depend on it.

**Databases.** A directory with a catalog (names, ids, shard counts; replaced
atomically), and one directory per collection holding its shards. Rows go to
shards by a hash of their id. A collection is created under a temporary name
and committed by the catalog, and dropped the same way: its files go when the
last user lets go of it, or at the next open. A collection that cannot be
opened fails alone; the others work. `Shard` is the interface of one shard
(write, search, get, stats, flush, optimize, wait_idle) with owned requests
and replies, so step 8 can put it on a network. Shards in this process are
searched as if their segments were one collection's.

**Found and fixed on the way:**

- **Step 7a could lose rows** under `Buffered` after a power loss. A build
  leaves out rows deleted before it started, and its manifest became durable
  without the logs that held those deletes. An upsert whose new row was still
  in an unsynced log could thus lose both its rows. A build now fsyncs the
  logs before its manifest.
- Flat segments were never merged while most of their rows were deleted
  (their live rows did not reach `index_bytes`), so a steady stream of
  upserts of the same ids grew them, and every search, without end. The
  "shrink" job merges them.

The tests:

- simulated power losses: copies of the directory taken at the steps of
  flushes and builds and at random moments, with each log cut at a random
  point after its last fsync, and logs whose names were not durable yet
  dropped at random. A reopen finds whole batches, a prefix of them, with
  every `Sync` one that was acknowledged;
- a flush or a build that takes over deletes made meanwhile, with its commit
  step stopped in between: the rows stay right, also after a power loss
  right there;
- writes wait for a slow flush and for a slow build, and go on after; a
  flush that fails leaves its segment searchable, its writes succeed and it
  is retried; a drop stops a build;
- one pool thread or many, one shard or many, here or elsewhere, filters,
  deletes, concurrent writes: the same results, and no search sees part of a
  batch; a failing task fails its search, and a panic reaches its caller;
- the pool's shares: one search gets all the CPUs, two get half each and
  settle there, as many as there are CPUs get none, and those that wait get
  threads regardless;
- databases: creates and drops interrupted at each step, damaged
  collections, drops of collections in use, a shared build limit.

Some 70 mutations of this code were each run against the tests. The ones not
caught: two that make `wait_idle` return up to a second late in a narrow
race, one that removes a notification (latency only), one that changes
nothing (the directory is fsynced once more than needed), and one of the
pool's counters, which no longer matters once the helper that decrements it
is done. Not simulated: a log's name lost between its manifest's rename and
the fsync that makes that durable.

Results on SIFT1M (L 40, W 4, i7-8700: 6 cores, 12 threads, 2 memory
channels), on a machine shared with other work. Ranges are over interleaved
runs of the final build while that work was light, unless they say
otherwise. The pool has 24 threads and 3 CPUs, the default for 12 threads.

Searches of `sift1m-defaults` (7 SSD segments, some flat and growing rows),
page cache, recall@10 0.9974-0.9976 throughout:

| clients | serial: QPS | p50 / p99 ms | pool: QPS | p50 / p99 ms |
|---:|---:|---:|---:|---:|
| 1 | 252-261 | 3.8-3.9 / 5.3-5.4 | 461-469 | 2.1 / 2.9-3.1 |
| 2 | 415-453 | 4.4-4.7 / 5.6-7.5 | 438-439 | 4.5 / 6.5-6.6 |
| 4 | 840-876 | 4.2-4.4 / 8.8-9.1 | 826-850 | 4.3-4.5 / 8.9 |
| 8 | 1,183 | 6.4 / 11.3 | 1,139-1,168 | 6.5 / 11.3-14.3 |
| 12 | 1,167 | 9.2 / 27.9 | 1,031-1,151 | 9.3-10.3 / 27.9-32.2 |

CPU per search: 3.8-4.0 ms serial, 6.1-6.2 ms with the pool at one client,
the same as serial from two on (4.6-4.8 at two and four). Two runs each, one
for serial at 8 and 12 clients; a serial run of the second round there was
disturbed and is left out.

- **A pool helps a search when the machine is idle, and gets out of the way
  otherwise.** At one client it saves 45% of the latency for 1.6 times the
  CPU; from two clients on, it changes nothing within the noise. Letting
  every idle thread help (the first version of this code) halved the
  throughput at 8 clients, from 1,190 to 645 QPS, and tripled the p99. The
  limit is memory bandwidth: a scan of fresh rows reads 24-25 GB/s, which
  about 3 cores use up, so a helper there takes it from searches with one
  thread.
- The CPU per search rises by 60%, which costs nothing while the machine is
  idle. `search_cpus` sets how many CPUs a search may take.

One SSD segment, 25,000 growing rows and a writer at 5,000 rows/s (page
cache):

No clean serial run of the final build could be taken on the shared
machine. An earlier build of this step, on an idle one, gave 790-794 QPS
serial and 1,502-1,534 with the pool at one client (p50 1.23 against 0.62-0.64
ms), 2,208-2,304 against 2,217-2,245 at 4 and 3,710-3,939 against 3,639-3,866
at 8. The final build counts helpers against the CPUs differently, so these
numbers are indicative; its unit tests check who gets help (a search alone
gets all of the CPUs; as many searches as there are CPUs get none), not how
much it saves.

One SSD segment alone, 7a against 7b, with and without a pool (page cache):

| best of 3 runs, QPS | 1 reader | 8 readers |
|---|---:|---:|
| 7a | 4,407 | 24,163 |
| 7b, no pool | 4,365 | 24,681 |
| 7b, pool | 4,383 | 24,443 |

Direct I/O, capped at 100,000 reads/s (a cap put on the process, not the
drive's limit; about 356 reads per search, `sift1m-defaults`). 200 searches/s
are 71% of the cap:

| | 200 searches/s from 4 clients: p50 / p99 ms, CPU ms per search | 1 client: QPS / p50 ms |
|---|---:|---:|
| serial | 15.2-15.4 / 19-22, 6.3-6.6 | 64-68 / 14.3-15.1 |
| pool | 3.8-3.9 / 8.8-18, 9.6-10.4 | 250-260 / 3.75-3.81 |

- The pool cuts the latency to a quarter by reading the 7 SSD segments at
  once. Walks with direct I/O are not all waiting: a search spends about 6
  of its 15 ms on the CPU.
- At the cap and flat out (4 clients, an earlier build of this step) the
  pool has a worse tail: p99 58-62 ms against 19 ms serial, with 276-280 QPS
  against 241-245. Behaviour above the cap, with up to 24 threads in walks
  that use 40% of a CPU, is not measured.

Writes: 8 writers, `Sync`, batches of 100 (6 s, seal at 32 MiB), three runs:

| | p99 ms | p99.9 ms | max ms | rows/s |
|---|---:|---:|---:|---:|
| no seals | 23-28 | 29-59 | 30-59 | 45,800-55,600 |
| inline seal | 35-44 | 80-181 | 208-331 | 43,700-55,500 |
| background flush | 42-53 | 61-74 | 61-74 | 45,800-55,900 |

- **The background flush takes the 208-331 ms stall of the inline seal out
  of the writes** (max 61-74 ms). It does not help the p99: seals and logs
  share the disk (p99 42-53 ms, against 35-44 inline and 23-28 without
  seals), and their fsyncs overlap now.

Writes at a paced 20,000 rows/s, `Buffered`, one writer, batches of 100:
p99.9 1.8-5.7 ms and max 9.3-10.6 ms with the background flush, 6.7-11.4 and
135-183 ms with the inline one.

Ingesting the 1M rows from scratch, with index builds (6 build threads), one
writer (an earlier build of this step, with an idle machine):

| | rows/s | last batch done after | slowest batch | peak memory |
|---|---:|---:|---:|---:|
| inline seal and builds | 7,168 | 139.5 s | stalls for builds | 0.85 GB |
| background, 1 GiB of flat rows allowed | 466,000 | 2.1 s (builds go on for 123 s) | 96 ms | 2.06 GB |
| background, 128 MiB allowed | 10,800 | 92.9 s | 26.3 s | 1.09 GB |

- The default lets an ingest of up to 1 GiB of vectors go through at disk
  speed while the index catches up, and the fresh rows are searched by brute
  force until then.
- **The limit works by making a write wait for a build to finish.** As of
  step 7b that wait had no bound: the builds of a collection ran one at a time
  (and those of a database shared a limit, 1 by default), so a write waited
  behind whatever was running. In the ingest above it waited 26 s for an index
  build of 128 MiB of rows; with a 1M-row segment already there, upserts of
  its rows crossed its 20% of deletes, its compaction (787,344 rows) took 224
  s on this loaded machine, and the write at the 128 MiB limit waited 213 s
  for it (170 s in another run), although what it needed was a merge of the
  flat segments. Step 7c took that wait away (and left the one for index
  builds).

Search with a writer going flat out, 4 threads searching (12 s, page cache,
starting from the indexed 1M):

| | QPS | mean ms | p50 / p99 ms | rows/s written | slowest batch |
|---|---:|---:|---:|---:|---:|
| no background work | 102 | 39 | 26 / 132 | 115,000 | 412 ms |
| flushes in the background | 90 | 44 | 29 / 182 | 107,000 | 78 ms |
| and a pool | 91 | 44 | 32 / 154 | 119,000 | 78 ms |
| and a 128 MiB limit | (see above) | 45 | 26 / 176 | | 170 s (see 7c) |

- One run each, with 4.5-5.3 other cores in use, so only the large
  differences count. No index build was due in any of them (the collection's
  `index_bytes` is 512 MiB, more than 12 s of writing fills), and searching
  the piled-up flat rows costs the same whoever writes them: background
  flushes do not help a search. They do cut the longest write from 412 to 78
  ms.

Commands (`B`, `S` as above; add `--pool 24`):

```
$B coll-concurrent --from sift1m --dir work --base $S/sift_base.fvecs --query $S/sift_query.fvecs \
   --l 40 --direct false --readers 1,2,4,8,12 --pool 24
$B coll-concurrent ... --growing 25000 --writers 1 --write-rate 5000 --pool 24
$B coll-concurrent ... --direct true --readers 4 --read-rate 200 --pool 24
$B coll-ingest --base $S/sift_base.fvecs --dir work --durability sync --batch 100 --secs 6 --writers 8 \
   --index-mb 100000 --background true --seal-mb 32      # no builds; --background false: inline seal,
                                                         # --seal-mb 100000: no seals
$B coll-ingest --base $S/sift_base.fvecs --dir work --threads 6 --background true --unindexed-mb 1024
```

## Server (step 7c)

### Writes no longer wait behind compactions

In step 7b a write that had reached `max_unindexed_bytes` waited for the one
build thread of its collection, and so behind whatever that ran: minutes, if
it was a compaction of a large segment. Now:

- A collection has two build threads, or lanes, by what a job takes in. One
  runs the jobs of flat segments, which are what makes room (index builds,
  merges of flat segments that are mostly deleted); the other compacts and
  merges SSD segments. Neither takes a segment that the other works on.
- The limit of builds at once in a database (`builds`) holds per lane, so
  compactions of one collection do not keep another from making room either.
  Places are given in the order they were asked for.
- An index build takes the oldest flat segments, up to `index_bytes` or an
  eighth of `max_unindexed_bytes`, whichever is more (twice `index_bytes`
  since step 7f), instead of all there are. The price is more and smaller SSD
  segments, which merges put together later.
- `optimize` claims the segments it works on and leaves the lane of flat
  segments to others.
- `max_write_wait` bounds what is left of the wait: a write that has not got
  room by then fails with `Unavailable` and has changed nothing in the shard
  that made it wait (in no shard, since step 7e). (Off by default in the
  library, 10 s in the server.)

One writer upserts the rows of a 1M-row segment (`sift1m-1seg`, buffered, 20
s, limit 128 MiB, 4 build threads, 4 threads searching; page cache), on the
same shared machine as before. Its compaction becomes due, and then a merge of flat segments:

| | slowest batch | rows/s written | p50 / p99 ms of a batch | mean ms of a search |
|---|---:|---:|---:|---:|
| 7b | 126 s | 11,595 | 0.37 / 5.9 | 52.9 |
| 7c (before and after that run) | 2.3-2.4 s | 121,324-124,092 | 0.39-0.40 / 6.5-7.2 | 53.6-58.5 |

- The writer waited for the compaction (126 s; 170 and 213 s on a more
  loaded machine, step 7b) and waits for the merge of flat segments now. 7b's
  rows/s are that low because the run lasts until the writer is back.
- This is not what keeping to the limit costs. `index_bytes` of that
  collection is 512 MiB, so no index build came due, and a write only waits
  while a build is: both 7c runs ended with 829,321 flat rows (405 MiB). What
  it costs when index builds are due is in the next table.

Ingesting 1M rows from scratch, 6 build threads, one run each:

| limit | | slowest batch | last batch done after | until idle | SSD segments | peak memory |
|---|---|---:|---:|---:|---:|---:|
| 128 MiB | 7b | 26.4 s | 92 s | 138 s | 6 | 1.08 GB |
| | 7c | 20.7 s | 99 s | 136 s | 7 | 1.03 GB |
| 1 GiB | 7b | 0.16 s | 2.4 s | 125 s | 2 | 2.04 GB |
| | 7c | 0.16 s | 2.3 s | 122 s | 4 | 1.36 GB |

- **Not fixed in this step: a write waited for the index build of its own
  rows** (20.7 s at the 128 MiB limit, which is small for this rate), and with
  `builds` at 1 for those of other shards before it: the writer ran at full
  speed and then stopped. Step 7d slows it down instead.
- The default limit of 1 GiB takes this ingest (0.5 GB of vectors) without a
  wait. That limit is per shard, so RAM has to cover it times the shards that
  are written to.

### HTTP

`vecdb-server` offers a database over HTTP/1.1, with JSON requests and
replies. (gRPC comes with the shard interface in step 8.)

```
target/release/vecdb-server --data data/db          # http://127.0.0.1:7410; --help lists the flags

U=http://127.0.0.1:7410/v1/collections; J='Content-Type: application/json'
curl -X PUT $U/docs -H "$J" -d '{"dim": 4, "metric": "cosine", "shards": 2,
   "fields": {"year": "int", "tag": "keyword"}}'
curl $U/docs/upsert -H "$J" -d '{"rows": [
   {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4], "attrs": {"year": 2024, "tag": ["a", "b"]}},
   {"id": 2, "vector": "zczMPc3MTD6amZk+zczMPg=="}]}'
curl $U/docs/search -H "$J" -d '{"vector": [0.1, 0.2, 0.3, 0.4], "k": 10, "with_attrs": true,
   "filter": {"and": [{"field": "year", "gte": 2020}, {"field": "tag", "eq": "a"}]}}'
```

| request | body | reply |
|---|---|---|
| `GET /healthz` | | `{"status": "ok"}`, also without a key |
| `GET /metrics` | | counters, in the Prometheus text format |
| `GET /v1/collections` | | names and row counts |
| `PUT /v1/collections/{name}` | `dim`, `metric` (`l2`, `ip`, `cosine`); optional `shards` (1), `fields` and the settings of `CollectionConfig` | 201, all settings |
| `GET /v1/collections/{name}` | | settings (as `PUT` takes them), rows, segments, errors of background work |
| `DELETE /v1/collections/{name}` | | drops the collection and its data |
| `POST .../upsert` | `{"rows": [{"id", "vector", "attrs"}, ...]}` | how many, and how many of them replaced a row |
| `POST .../delete` | `{"ids": [...]}` | how many were there |
| `POST .../write` | `{"ops": [{"upsert": row}, {"delete": {"id"}}, ...]}` | per op, whether its id had a row before |
| `POST .../get` | `{"ids": [...]}` | the rows there are |
| `POST .../search` | `vector`; optional `k` (10), `l` (100), `beam_width`, `rerank`, `filter`, `filter_mode`, `with_attrs`, `with_vectors`, `with_stats` | `id` and `distance` of the closest, closest first |
| `POST .../flush` | | writes what is in RAM to segment files |
| `POST .../optimize` | | 202; merges each shard's segments in the background (`state.optimizing` says whether it runs) |

- **Vectors** are arrays of numbers, or base64 strings of little-endian `f32`
  values, which are shorter and faster to read. Every `f32` survives the trip
  in both. Collections with `cosine` store, and return, them normalized.
  Smaller distances are closer; with `ip` a distance is the negative product.
- **Ids** are unsigned 64-bit integers, as JSON numbers. Clients in
  languages whose numbers are doubles (JavaScript) have to stay below 2^53.
- **Attributes** are integers (signed, 64 bits) or strings, or arrays of
  them. A filter is `{"field": F, "eq": V}`, `{"field": F, "in": [...]}`,
  `{"field": F, ...}` with `gt`, `gte`, `lt`, `lte`, or `{"and": [...]}`,
  `{"or": [...]}`, `{"not": ...}`.
- Unknown keys, keys given twice and parameters in the path are errors, so
  that a typo does not change what a request does. An optional key with
  `null` is as if left out.
- **Errors** are `{"error": {"code", "message"}}`: 400 what was asked for is
  wrong, 401 no key, 403 a browser without a key, 404, 405, 408 the body came
  too slowly, 409 exists already (or runs already, or too many shards), 413,
  415 not JSON, 500 the server's fault (it also logs it), 503 with
  `Retry-After`: overloaded, or index builds are behind, or the collection
  could not be opened.

**Writes** are durable once answered (`--durability sync`; `buffered` leaves
the fsync to the operating system). The ops of a request apply in order, all
or none of those of a shard; across the shards of a collection they are not
atomic. A request that fails with 5xx, or whose reply is lost, may have been
applied in part or in whole. Sending it again is safe as long as nobody else
wrote the same ids meanwhile: upserts replace, and deleting what is gone does
nothing. A write that has to wait for index builds (see step 7d) gets 503
after `--write-wait-secs` (10), without having changed anything. Sent again as
it was, within a minute, it goes on from its place in the line (see step 7e).

**Overload.** The engine's calls block, so they run on threads of their own.
Searches, writes, requests that can take long and requests that only look
have separate limits of how many run at a time (two searches per CPU, 64
writes) and how many wait (1,024 each, 10 s at most). Beyond those, requests
get 503 at once. A search in progress is not cancelled when its client goes
away.

**Limits.** Bodies of writes may have 64 MiB, others 4 MiB; `k` and `l`
10,000; a request 100,000 writes; a reply about 64 MiB, which limits how many
rows with vectors one may ask for; a collection 64 shards, all of them 1,024
(each has three threads). Large bodies (over 256 KiB) share 512 MiB, held
until they have been written, and large replies another 512 MiB, held until
their clients have them. What is parsed from a body can take several times
its size (some 40 times, for rows with many attribute values). 4,096
connections at most; idle ones are closed after 75 s, clients that send a
body too slowly or do not take their reply as well.

**Security.** The server listens on 127.0.0.1 and has no TLS: to serve other
machines, put a reverse proxy in front of it. With `--api-key-file` (or
`VECDB_API_KEY`) every request but `/healthz` has to carry `Authorization:
Bearer <key>`; there is one key, and it allows everything. Without a key, the
server refuses to listen beyond the machine, and web pages in a browser on it
are kept out: requests with an `Origin` header or for a host name other than
`localhost` are refused (pages whose own name was pointed at 127.0.0.1 have
such a name), and bodies have to be `application/json`, which pages cannot
send elsewhere without asking. Any program on the machine may use it, and a
proxy is one: behind a proxy, give the server a key or let the proxy check who
is asking, or everyone who reaches the proxy has full access.
`--allow-unauthenticated true` answers anyone.

**Stopping.** On SIGTERM or SIGINT the server stops accepting, closes idle
connections, answers the requests it has for up to `--drain-secs` (30), then
closes the database: flushes that run are waited for and builds are stopped,
to be done after the next start. What is cut off (a request that takes
longer, an `optimize`) ends as in a crash, which loses nothing that was
answered.

### Tests

The tests talk to the server over real sockets, or run the program:

- a collection from its creation to its drop, against a model: upserts,
  deletes, writes in order, filters, flushes, index builds, `optimize`;
- what is wrong with a request: some 80 of them, each with its status, and
  none changes anything; nesting 50,000 deep;
- 64-bit ids and integers, and every bit of a float, unchanged; a request
  body in chunks of 1 byte, pipelined, after `Expect: 100-continue`;
- keys and their near misses; origins and host names;
- each limit at and beyond its value: bodies, replies, rows, places,
  connections, shards; bodies that come too slowly, replies that are not
  taken, clients that go away in between: what they held is given back, and
  not before the engine is done with it;
- overload: requests are answered or refused at once, and other kinds go on;
- a write that waits for an index build of another collection is refused in
  time, and accepted later;
- drops and creates under searches and writes; a damaged collection fails
  alone;
- SIGTERM, SIGINT and SIGKILL in the middle of use: every row that was
  answered is there after a restart.

76 mutations of this step's code (19 of the build lanes, 5 of the rows found,
52 of the server) were each run against the tests. Tests were added for those
that were not caught at first. Still not caught: two narrow races (a
`wait_idle` that does not look at one lane, missed in one run of two, and
`lane_segments` reading the view before the claims), and a `--network-threads`
without effect. Two were noticed because tests hung.

### Results

`sift1m-defaults` (page cache, k 10, L 40, recall@10 0.9975 throughout): the
same collection in the same process as its clients, and in a server on the
same machine, whose CPUs the clients share. One connection per client, 3
rounds; i7-8700, shared with other work, load 3-5 before.

| clients | in process: QPS | p50 / p99 ms | HTTP: QPS | p50 / p99 ms | HTTP, base64: QPS |
|---:|---:|---:|---:|---:|---:|
| 1 | 485-496 | 2.0 / 2.5-2.6 | 432-452 | 2.2-2.3 / 2.7-2.9 | 446-457 |
| 4 | 932-983 | 4.0-4.1 / 5.0-7.4 | 749-778 | 4.9-5.1 / 8.1-8.7 | 734-782 |
| 8 | 1,225-1,355 | 5.6-6.4 / 9.6-11.9 | 972-999 | 7.8-8.0 / 13.1-13.5 | 956-984 |

One SSD segment alone (`sift1m-1seg`, recall@10 0.9648), where a search takes
0.2 ms:

| clients | in process: QPS | p50 / p99 ms | HTTP: QPS | p50 / p99 ms | CPU ms per search |
|---:|---:|---:|---:|---:|---:|
| 1 | 4,367-4,535 | 0.22 / 0.31-0.48 | 3,009-3,479 | 0.28-0.33 / 0.40-0.43 | 0.22-0.23, 0.28-0.29 |
| 8 | 25,604-25,809 | 0.30-0.31 / 0.44-0.45 | 18,313-19,636 | 0.40-0.41 / 0.69-0.87 | 0.31-0.32, 0.39-0.40 |

Writes of 128-dim rows, 100 per request, one collection of one shard:

| clients | in process: rows/s | HTTP | HTTP, base64 | p99 ms: in process, HTTP, base64 |
|---:|---:|---:|---:|---|
| 1, buffered | 1,061,000-1,093,000 | 193,000-206,000 | 233,000-241,000 | 0.14-0.16, 0.64-0.69, 0.57-0.64 |
| 4, buffered | 1,472,000-1,515,000 | 569,000-621,000 | 763,000-781,000 | 0.95-1.9, 1.1-1.3, 0.99-1.3 |
| 8, buffered | 1,423,000-1,540,000 | 867,000-976,000 | 1,140,000-1,180,000 | 1.5-3.2, 1.7-2.3, 2.1-2.5 |
| 1, sync | 15,700-16,500 | 13,000-14,200 | 12,600-13,700 | 10-11, 13, 12-14 |
| 8, sync | 46,800-48,600 | 45,200-48,700 | 49,300-52,800 | 53-54, 25-27, 22-26 |

- **With one client HTTP adds 0.06-0.25 ms, and with 4 or 8 it takes 20-26%
  of the throughput** (client and server together on 12 threads). Why is not
  established. Most of it does not seem to be HTTP: in process, handing each
  search to one of 8 threads of a plain pool did as badly (`sift1m-defaults`
  at 4 clients, with `--pool-cpus 1` so that searches get no help: 759-762 QPS
  against 968-1,003 when each client searches itself, and 770-819 over HTTP).
  Two changes to the server were tried and are not in the code: searching on
  the network threads was worse (625-670 QPS, same setting), and an own pool
  that reuses the thread that was busy last made no difference on
  `sift1m-defaults` (753-755 at 4 clients, 938-970 at 8) and gained 2-4% on
  `sift1m-1seg`, which is within the noise. Fewer network threads
  (`--network-threads`) helped some runs and hurt others.
- **Base64 halves the bytes of a write** (35 KB against 71 for 100 rows of 128
  dimensions) **and takes about 20% off its time.** For searches it makes no
  difference.
- Durable writes lose 0-14% at one client (two runs: 11,800-15,900 rows/s in
  process against 12,800-13,600 in the first, the table's in the second) and
  nothing at 8, where the fsyncs are shared (the p99 in process is worse there;
  that is not explained).
- Not measured: single-row requests, hundreds of connections, anything but
  loopback.

Commands (`B`, `S` as above):

```
$B http --from sift1m-defaults --dir work --base $S/sift_base.fvecs --query $S/sift_query.fvecs \
   --server target/release/vecdb-server --l 40 --clients 1,4,8 --rounds 3 --rows 100000 --batch 100
$B http --from sift1m-1seg ... --clients 1,8 --rounds 3 --rows 50000 --durability sync
$B coll-concurrent --from sift1m-1seg --dir work ... --l 40 --direct false --readers 4 --secs 20 \
   --writers 1 --background true --pool 0 --unindexed-mb 128 --threads 4
$B coll-ingest --base $S/sift_base.fvecs --dir work --threads 6 --background true --unindexed-mb 128
```

## Slowing writes down (step 7d)

Up to step 7c a writer that outran the index builds wrote at full speed until
`max_unindexed_bytes` was reached, and then stood still until a build was done
(and, with several shards and `builds` at 1, those of the others before it).
Now it is slowed down to what the builds keep up with:

- **What counts** is the vectors that are in no SSD index: flat segments,
  those that wait to be flushed, and the growing one.
- **The rate of writes depends on that alone.** Up to three quarters of the
  limit they are free. Over the last quarter the rate is halved 16 times, from
  256 MiB to 4 KiB a second, and at the limit it is 0. The amount settles where
  writes are as fast as builds, whatever their rate is; nothing is estimated.
- **Builds give room back little by little.** A build frees what it took in at
  its end, all at once. So it says how far it is (PQ training, encoding and
  graphs, by their cost), and that share of its input counts as gone. The same
  goes for a build that waits for those of other shards: its way through the
  line of builds and its own work, by bytes, is one.
- **Writes wait in line**, per shard, in the order they came. Each goes once
  those before it are paid for at the rate there is: one that comes after a
  pause does not wait, however large it is. Deletes do not wait.
- Only while a build that makes room is due (or a flush that may make one due),
  and as long as builds do not fail. `max_write_wait` counts from when a write
  comes; one that has waited for longer fails and has changed nothing in that
  shard (in no shard, since step 7e).
- `CollectionStats` has the rate that writes are slowed down to and the time
  they have waited, the server `vecdb_shards_slowing_writes`,
  `vecdb_write_wait_seconds_total` and `state.slowing_writes`.

**Memory.** What counts as gone is in RAM until the build ends. So there may be
the limit and the input of one build (an eighth of the limit, or `index_bytes`
if that is more; twice `index_bytes` since step 7f), and the limit is taken to
be `index_bytes` and a quarter at least (or `index_bytes` and two segments):
1,152 MiB with the defaults, where it was 1,120 (the limit and three segments),
but up to twice as much as it was where the limit is not above `index_bytes`.
Writes that were let in and are under way (one per writer) come on top, as
before.

### Results

SIFT1M rows (128-dim, buffered WAL) written from scratch, 6 build threads, on
the same shared machine, which was loaded throughout (load 13-19 on 12
threads, other programs at higher priority). Builds took two to three times as
long as in the 7c tables above, which makes the waits of 7c longer. 7c and 7d
ran in turns, with the same benchmark program, one or two runs each (7c's
slowest batch in other runs, at loads up to 30: 54-56 s in the first case, 63-72
s in the third, 38 s in the fourth):

| what is written | | rows/s | batch p50 | batch p99 | slowest | batches over 1 / 10 s | most in no index | peak memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 writer, 1,000 a batch, limit 128 MiB | 7c | 3,713 | 3.0 ms | 0.02 s | **46.4 s** | 6 / 6 | 194 MiB | 0.99 GB |
| | 7d | 3,787-3,792 | 0.33 s | 0.48 s | **0.66-0.68 s** | 0 / 0 | 175 MiB | 1.00 GB |
| 8 writers, single rows, limit 128 MiB | 7c | 10,231-10,279 | 0.03 ms | 0.4 ms | **36.2-36.3 s** | 8 / 8 | 143-147 MiB | 0.63-0.64 GB |
| | 7d | 7,500-7,608 | 0.10-0.11 ms | 5.4-5.5 ms | **0.02 s** | 0 / 0 | 174 MiB | 0.64-0.67 GB |
| 4 shards, 8 writers, 1,000 a batch, limit 48 MiB a shard | 7c | 4,752 | 11 ms | 61 s | **61.2 s** | 24 / 24 | 261 MiB | 1.05 GB |
| (`index_bytes` 16, segments 8 MiB) | 7d | 5,050-5,131 | 0.04 s | 4.5-4.6 s | **4.6-4.8 s** | 253-254 / 0 | 191-193 MiB | 1.06-1.08 GB |
| the same, limit 64 MiB a shard | 7c | 5,074 | 14 ms | 22.1 s | **29.7 s** | 105 / 58 | 301 MiB | 1.56 GB |
| (`index_bytes` 8, segments 4 MiB) | 7d | 4,141-4,842 | 2.7-2.8 s | 3.5-4.3 s | **3.6-4.5 s** | 532-537 / 0 | 241-242 MiB | 1.50-1.52 GB |
| 64 writers, 1,000 a batch, limit 128 MiB | 7c | 10,417-11,414 | 0.14-0.15 s | 51-56 s | 51.3-56.2 s | 64 / 64 | 247-253 MiB | 1.09-1.11 GB |
| (600,000 rows, of which 7c has indexed a quarter when all are written) | 7d | 4,861-4,866 | 16.6-17.0 s | 26-28 s | 27.6-29.5 s | 344-348 / 342-344 | 188-189 MiB | 1.07-1.10 GB |
| the same, 1M rows, refused after 10 s and sent again a second later | 7c | 4,603 | 0.15 s | 104 s | 104 s | 147 / 147 | 280 MiB | 1.53 GB |
| (7c: 747 refusals, 7d: 586-587) | 7d | 4,544-4,546 | 9.8 s | 108 s | **185 s** | 756-757 / 247-254 | 196 MiB | 1.37-1.41 GB |
| 16 writers, 600,000 rows, refused after 10 s (7c: 66 refusals, 7d: none) | 7c | 7,178 | 0.03 s | 39 s | 42.8 s | 22 / 22 | 222 MiB | 0.93 GB |
| | 7d | 5,619 | 3.6 s | 5.5 s | **5.6 s** | 376 / 0 | 178 MiB | 0.89 GB |
| 1 writer, 1,000 a batch, limit 1 GiB (never reached) | 7c | 256,293-258,776 | 2.6 ms | 34-44 ms | 0.12-0.13 s | 0 / 0 | 483-484 MiB | 1.36-1.40 GB |
| | 7d | 251,889-252,989 | 2.5-2.7 ms | 30-47 ms | 0.16-0.18 s | 0 / 0 | 476-481 MiB | 1.37-1.39 GB |

Those 7d numbers are of the code before its last two fixes: jobs on SSD
segments, while they ran or waited for their turn, took back what an index
build had given; and a job that became an index build while it waited counted
as gone all at once when its turn came. They do not show in the fourth case,
run in turns (7d before, after, 7c, after, before) at a load of 22-33:

| 4 shards, limit 64 MiB a shard | to write | until all is indexed | batch p50 | batch p99 | slowest | batches over 10 s | most in no index |
|---|---:|---:|---:|---:|---:|---:|---:|
| 7c | 329 s | 896 s | 20 ms | 38 s | **50.1 s** | 105 | 299 MiB |
| 7d before the fixes | 346-360 s | 859-884 s | 4.3-4.6 s | 6.0-6.3 s | 6.4-6.5 s | 0 | 245-247 MiB |
| 7d | 348-364 s | 879-902 s | 4.2-4.4 s | 6.0-6.3 s | **6.2-6.6 s** | 0 | 244-245 MiB |

The others once more, at a load of 15-40, which makes everything slower (with
the first of the fixes; the last pair with both):

| what is written | | rows/s | batch p50 | batch p99 | slowest | batches over 1 / 10 s | most in no index | peak memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 writer, 1,000 a batch, limit 128 MiB | 7d | 3,586 | 0.30 s | 0.58 s | **0.60 s** | 0 / 0 | 175 MiB | 0.99 GB |
| 8 writers, single rows | 7d | 5,249 | 0.11 ms | 7.2 ms | **0.05 s** | 0 / 0 | 174 MiB | 0.64 GB |
| 4 shards, limit 48 MiB a shard | 7c | 3,710 | 11 ms | 64 s | **64.8 s** | 26 / 26 | 261 MiB | 1.04 GB |
| | 7d | 3,738 | 0.04 s | 6.8 s | **7.1 s** | 254 / 0 | 194 MiB | 1.04 GB |
| 64 writers, 600,000 rows | 7d | 4,119 | 18.7 s | 37 s | 38.8 s | 341 / 341 | 189 MiB | 1.07 GB |
| 16 writers, 600,000 rows, refused after 10 s (none was) | 7d | 5,395 | 3.6 s | 6.1 s | **6.2 s** | 376 / 0 | 179 MiB | 0.87 GB |
| 64 writers, 1M rows, refused after 10 s and sent again a second later | 7c | 4,896 | 0.20 s | 74 s | 73.6 s | 170 / 170 | 276 MiB | 1.62 GB |
| (7c: 956 refusals, 7d: 1,028) | 7d | 3,323 | 9.8 s | 252 s | **295 s** | 759 / 268 | 191 MiB | 1.44 GB |

- **Nobody stands still for a build any more.** Single rows wait 21 ms at most,
  (55 ms at the higher load) and a writer of 1,000-row batches 0.7 s, where 7c
  had some wait for 36 and 46 s. The price is that every batch waits a little:
  what 6 batches waited before, 780 of 1,000 share now (0.33 s each).
- **The rows/s of a long ingest are still those of the builds**: the first case
  took 314-315 s until all was indexed, against 313 s, and the fourth 879-902 s
  against 896 s. Where 7c's rows/s are higher, it had more rows left to index
  when the writers were done.
- **Shards that share one build place** are where counting the way through the
  line matters: with the progress of their own builds only, the third case had
  batches wait for 31.9-37.1 s, and the fourth for 13.0 s (more than the server
  lets them, 10 s).
- **Writers wait for each other**, and no pacing changes that: 64 of them with
  0.5 MiB each, at 2 MB/s, wait 17 s. **If each gives up after 10 s and comes
  back, 7d is worse than 7c:** the slowest batch took 185 s and 295 s to get
  through in the two series (7c: 104 s and 74 s), with 586-587 and 1,028
  refusals (747 and 956), and it took 274 s and 374 s until all was indexed (217
  s and 274 s). The wait of those who give up is for nothing, and they come back
  behind the others. And 7c let so much in at once (280 MiB, where the limit is
  128) that it flushed larger segments, and built 4 indexes where 7d built 7;
  larger ones probably cost less per row (the training does not grow with
  them). With 16 writers nobody is refused, where 7c refused 66 times. Step
  7e is about that case.
- Not slowed down, writes cost what they did: 8 writers of single rows, no
  index builds, 109,578-120,066 rows/s (7c: 108,348-121,602), one writer
  111,838-112,098 (110,536-115,844).
- **Not covered:** merges of SSD segments, which do not slow writes down; their
  number grows if they do not keep up. The server has one set of 64 places for
  writes: writers of a collection that is slowed down can take them all. Not
  measured: the default limit while builds are behind (SIFT1M is half of it),
  more dimensions, or a quiet machine.

The curve, and what counts as gone, were chosen by simulating them against
builds whose speed varies (not in the repository). With one shard and batches
of 1,000 rows, the slowest write of the simulation waited for 0.1-1.4 s as it
is now; for 2-11 s where a build did not count until it was done, and 125 s
where builds were 10 times slower; and for 12-159 s where how far a build is
was guessed from the time the last ones took, and that changed. For the third
case of the table it has 86-90 s without slowing down, 34-40 s if only a
build's own progress counts, and 6-8 s as it is now (measured: 61-72, 32-37 and
5-7 s). With 12, 16 and 20 halvings the waits were about the same, up to 8
shards; with 32 that share slow builds, 16 had the shortest.

### Tests

The rate, the line (its order, writes that are late, a panic in it, a rate that
goes away), the way of a build through the line of builds (with one place and
two, builds that leave it or fail), and what counts at each step of a build, a
failed build, a compaction (also one beside the build, or one that waits), a
merge, an `optimize`, a job that is another one by the time its turn comes; that
a build says how far it is in the right proportions and makes the same bytes as
before; writes that keep pace with builds that are much slower than they are,
also in four collections that share a place; deletes, batches of both, and
collections where nothing is to come do not wait. 108 mutations of this step's code were run
against the tests (tests were added for those not caught at first). Not caught:
two that change nothing, and one that lets a lost turn count for 20 ms longer.
The tests with timing ran 12 times (those of the server 5 times) on a loaded
machine without a failure.

Commands (`B`, `S` as above; 7c: the same benchmark, over its core):

```
$B coll-ingest --base $S/sift_base.fvecs --dir work --threads 6 --background true --unindexed-mb 128
$B coll-ingest ... --unindexed-mb 128 --writers 8 --batch 1 --limit 400000
$B coll-ingest ... --shards 4 --seal-mb 8 --index-mb 16 --unindexed-mb 48 --writers 8 --limit 600000
$B coll-ingest ... --shards 4 --seal-mb 4 --index-mb 8 --unindexed-mb 64 --writers 8
$B coll-ingest ... --unindexed-mb 128 --writers 64 --limit 600000       # or --write-wait 10, all rows
$B coll-ingest ... --unindexed-mb 1024 --index-mb 100000 --writers 8 --batch 1         # the cost of looking
```

## Writes that give up keep their place (step 7e)

Step 7d left one case worse than 7c: many writers that give up after a while
(in the server, after 10 s) and come back. 64 of them with 0.5 MiB each, where
builds keep up with 2 MB/s, are a line of 17 s and more. Who gave up was the
last again when they came back, and everyone who had been let through had lined
up before them. In a trace of that case the slowest batches were all from its
first second: about ten were refused time and again until there was nothing
else left to write, 250-293 s later.

- **A write that gives up keeps its place in the line**, for a minute. If it
  comes again by then, it goes on from where it was. Nobody waits for it
  meanwhile.
- **It is known by what is in it:** the length and the CRC-32C of what the log
  would have of it. A client has nothing to remember, and nothing to send but
  the same request again. What differs, or comes later, is a new one, as in 7d.
  Two writes with the same content (or checksum) may get each other's place,
  which changes nothing but their order. A shard keeps the last 1,024 places.
  Nothing is computed while writes are not slowed down.
- **Nothing is written before all shards have room.** The parts of a write, one
  for each shard it has rows of, wait for each other once they have been let
  in. If one is refused, the others go back: they owe nothing, and keep their
  places. Up to 7d they had been written by then, and were again, from the ends
  of their lines, when the request was sent again. Now a write that is refused
  because flushes or builds are behind has changed nothing in any shard. (One
  that fails later, as it is written, may still have in others.)
- For that the `Shard` interface has `begin`, the half of a write that can
  wait; what it returns is the other half. A shard that does not implement it
  does all in the second half, and may then have refused what others wrote.

**What else was tried**, by simulating the line and the builds (not in the
repository; 64 writers with batches of 0.5 MiB, 488 MiB in all, one shard, three
runs each with builds that take 2 and 2.8 times as long as on a quiet machine):

| a write that has waited for 10 s, or would | slowest batch | refusals (mean) |
|---|---:|---:|
| gives up, and is the last when it comes back (7d) | 226-228 s, 316-332 s | 633, 1,169 |
| is refused at once, if the line is as long as it may wait | 97-138 s, 160-216 s | 6,439, 12,062 |
| is refused at once by chance, the longer the line the likelier | 79-104 s, 112-179 s | 6,900-7,000, 12,400-12,500 |
| is refused if it loses its place to a new one by chance | 139-182 s, 151-187 s | 6,435, 12,059 |
| gives up, and keeps its place (7e) | 24-29 s, 34-41 s | 946, 1,518 |
| waits as long as it takes | 24-29 s, 33-41 s | 0 |

Refusing at once, which 7d had proposed, helps those who are let in and starves
others: who gets in is luck. With kept places everyone waits as if nobody gave
up, at about a refusal per batch, also when clients come back after 1-5 s (27-44
s) or 10-20 s (31-58 s). With several shards the places alone would have made it
worse. Simulated for 64 writers, 4 shards and the larger factor: 7d has the
slowest batch after 247-306 s and writes every row 1.4 times; with places, but
each shard writing its part at once, the shards took turns at refusing, and it
did not end (13 times, when it was stopped); with both parts of the change it is
40-49 s and every row once.

### Results

As in 7d: SIFT1M rows from scratch, 6 build threads, buffered WAL, on the same
shared and loaded machine; 64 writers with batches of 1,000 rows, which give up
after 10 s and try again a second later. The same benchmark program for all
(it now counts how often rows are written), in turns (7c, 7d, 7e, 7e, 7d, 7c;
the last block 7e, 7d, 7d, 7e), two runs each:

| what is written | | to write | until all is indexed | batch p50 | p99 | slowest | batches over 1 / 10 s | refused | each row written | most in no index | peak memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 shard, 1M rows, limit 128 MiB | 7c | 259-268 s | 259-268 s | 0.18 s | 118-133 s | **118-133 s** | 164-183 / 164-183 | 1,064-1,260 | 1.00 | 267-284 MiB | 1.49-1.58 GB |
| (load 16-26) | 7d | 297-303 s | 372-375 s | 9.8-9.9 s | 252-294 s | **292-300 s** | 752-760 / 265-324 | 998-1,038 | 1.00 | 193-199 MiB | 1.35-1.41 GB |
| | 7e | 301-302 s | 375-377 s | 22.4-22.5 s | 31-33 s | **33.5-33.9 s** | 755-759 / 748-749 | 1,301-1,318 | 1.00 | 193-197 MiB | 1.41-1.43 GB |
| 4 shards, 600,000 rows, limit 48 MiB a shard | 7c | 120-334 s | 233-335 s | 0.15-0.18 s | 118-333 s | **118-333 s** | 33-44 / 22-44 | 220-1,320 | 1.13-1.98 | 281-286 MiB | 1.38-1.68 GB |
| (`index_bytes` 16, segments 8 MiB, load 15-22) | 7d | 179-198 s | 258-312 s | 0.23 s | 171-175 s | **174-184 s** | 253-254 / 151-156 | 751-872 | 1.39-1.46 | 234-242 MiB | 1.52-1.62 GB |
| | 7e | 146-147 s | 335-343 s | 0.24-0.25 s | 41-47 s | **41-51 s** | 228-245 / 220-227 | 617-626 | 1.00 | 192-193 MiB | 1.32-1.33 GB |
| the same (load 14-16) | 7d | 120-128 s | 207-236 s | 0.19-0.21 s | 114-116 s | **118-125 s** | 251-252 / 114-136 | 429-476 | 1.23-1.30 | 225-237 MiB | 1.48-1.50 GB |
| | 7e | 106 s | 240-246 s | 0.16-0.25 s | 30-32 s | **31-35 s** | 229-239 / 217-220 | 413-415 | 1.00 | 192-195 MiB | 1.32-1.36 GB |

(The first two blocks are of the code before three simplifications, the last one
before one of them; the runs over HTTP below are of the final code. One more run
of the first case at a load of 16-17: 241 s, 303 s, p50 18.3 s, p99 25.0 s,
slowest 27.5 s, 993 refusals.)

- **Nobody starves any more.** The slowest batch waits about as long as without
  a deadline (28-39 s in the tables of 7d, which end after 600,000 rows): a
  quarter to a ninth of what it did in 7d. And what was written twice and more
  in several shards is written once.
- **What got worse:** with one shard the typical batch waits longer (22 s, where
  7d had 9.8 s and 7c 0.2 s), as the wait is the same for all now; and more are
  refused than in 7d (1.3 times a batch). That is the length of the line: who may
  wait for 10 s and has to for 22 is refused twice. With `--write-wait 45` the
  same run (at a load of 16) had no refusals, and its slowest batch took 23.3 s.
  And with 4 shards it takes longer than in 7d until all is indexed (335-343 s
  against 258-312 s, 240-246 s against 207-236 s), with 16 builds instead of
  13-15. The simulation had that the other way round. Why is not verified; it may
  be that flat segments with rows that were written again have builds take in
  three of them instead of two.
- **What did not get better: the time until all is indexed** with one shard,
  375-377 s where 7c took 259-268 s. It is that of the builds, 7 of 64 MiB or a
  little more where 7c had 4 larger ones.
- **Larger builds would help there**, but are not part of this step, as they
  change the RAM bound of small limits (they are step 7f). Tried with builds of
  up to twice `index_bytes` while the limit allows it, first case, load 15-17,
  two runs: 261-276 s until all is indexed (303 s without), slowest 25.6-29.2
  s, 789-793 refusals, most in no index 206 MiB (202).
- **Over HTTP** the same: 64 clients that send a request of 1,000 rows (base64)
  again a second after a 503, the server with `--max-unindexed-mb 128`, 1M rows,
  load 6-11, 7d, 7e, 7e, 7d: the slowest batch took 120 and 69 s in 7d, 17.9 and
  11.7 s in 7e, with 371 and 232, 700 and 257 refusals, and 187 and 160, 179 and
  120 s to write all (the load changed between runs). 4 shards as above, 600,000
  rows, load 11-14, 7e, 7d, 7d, 7e: 97 s twice in 7d, 24 s twice in 7e, 324-356
  and 382-389 refusals, 103-108 and 90 s to write all.
- **The other cases are as they were** (7d and 7e in turns, one run each, load
  13-19): 16
  writers with the deadline 5,244 and 5,124 rows/s, the slowest batch 5.9 and 6.4
  s, nobody refused; 64 writers without one 5,325 and 5,226 rows/s, 22.7 and 25.1
  s; 8 writers of single rows, slowed down, 6,909 and 6,787 rows/s, p99 5.6 ms
  both, slowest 27 and 23 ms; 4 shards, 8 writers, 5,725 and 5,762 rows/s, 4.7
  and 4.3 s. Not slowed down, two runs each: 8 writers of single rows
  117,404-118,688 and 117,283-117,435 rows/s; one writer 240,447-257,489 and
  252,577-272,439; 4 shards, 8 writers with batches of 100, 291,455-320,039 and
  292,353-306,472.
- **Not covered:** the 64 places of the server's writes. With more clients than
  that, the others wait for one of those (10 s at most, 1,024 of them), in no
  order that is kept. Not measured: clients that change their batches when they
  are refused (the order of rows or attributes is enough; they wait as in 7d
  then), writes that wait for flushes, more than 4 shards.

### Tests

The places: that they are kept, used once, in the order they were had, not for
ever and not too many; that one that is kept is in nobody's way; a write that
was let in and is not written owes nothing, has its place, and wakes the next.
A write of a collection is known again by its content and nothing else, also
one that had to wait for a flush. The shards: none writes before all have room,
with one that refuses, one that is slow and one that panics, each of three, and
with real ones where one waits for an index build; shards without `begin`. 49
mutations of the code as it is now were run against the tests (before that,
tests were added for three that were not caught, and three others led to simpler
code). One is not caught, in a line that cannot be reached; seven are only
noticed as tests that hang. The tests with timing ran 25 times in each profile
on a loaded machine without a failure; before that, two had failed now and then:
a new one that did not allow for what the bucket lets be made up for, and one of
7d (`writes_are_not_slowed_down_for_nothing`) that took a slow flush for the
limit of index builds.

Commands (`B`, `S` as above; 7c and 7d: the same benchmark, over their cores):

```
$B coll-ingest --base $S/sift_base.fvecs --dir work --threads 6 --background true --unindexed-mb 128 --writers 64 --write-wait 10
$B coll-ingest ... --shards 4 --seal-mb 8 --index-mb 16 --unindexed-mb 48 --writers 64 --write-wait 10 --limit 600000
$B coll-ingest ... --unindexed-mb 1024 --index-mb 100000 --shards 4 --writers 8 --batch 100         # the cost of it
```

## Larger index builds (step 7f)

While writes are slowed down, they go as fast as index builds take rows in, so
a long ingest takes as long as its builds. Up to 7e a build that was behind
took the oldest flat segments up to `index_bytes`, or an eighth of
`max_unindexed_bytes` if that is more. Below 16 times `index_bytes` (the
limits of all tables of 7d and 7e) that is `index_bytes`: 7 builds of 64 MiB
for SIFT1M. Part of what a build costs does not grow with it: PQ training
uses 65,536 rows at most, about 45% of a build of 64 MiB of 128-dimensional
vectors (see below). Now:

- **A build takes up to twice `index_bytes`**, if an eighth of the limit is
  less. It takes what there is when it starts, so it is larger only when
  builds are behind.
- **Not if it would be split into partitions.** A build of a little more
  than one partition of `build_memory` holds is split into three at least,
  with each row in two of them. So twice `index_bytes` is taken only as far as
  one partition holds it and the largest flat segment more (the last segment a
  build takes may go beyond; segments are `seal_bytes`, or more where they
  filled while flushes were behind). This comes from reading the build, not
  from a measurement. With the default 2 GiB, one partition holds about 980 MB
  of 128-dimensional vectors and 1.85 GB of 960-dimensional ones;
  `index_bytes` has to be above half of that for this to matter.
- **Nothing changes with the defaults:** an eighth of 1 GiB is twice 64 MiB.
- **Memory:** there may be the limit and twice `index_bytes` in RAM (and
  scanned by searches) where it was the limit and `index_bytes`: 256 MiB
  instead of 192 MiB for the limit of 128 MiB of the tables.

### Results

**One build by its size**, as the collection builds it (128 dimensions, PQ of
32 bytes, R 64, L 100, 6 threads, one partition), SIFT1M rows, two runs each in
turns (`vecdb-bench disk-build`). The two blocks ran at different loads, so
compare within a block:

| block (load) | rows | build | PQ training | graph | per million rows |
|---|---:|---:|---:|---:|---:|
| 1 (17-19) | 32,768 | 15.6-16.2 s | 11.3 s | 3.7-4.3 s | 476-494 s |
| | 65,536 | 27.9-28.6 s | 18.5-19.0 s | 8.2-8.5 s | 426-436 s |
| | 131,072 | 41.7-42.9 s | 18.1-19.2 s | 21.4 s | 318-327 s |
| | 196,608 | 58.1-59.9 s | 18.6-19.1 s | 35.8-37.6 s | 296-305 s |
| | 262,144 | 74.3-75.1 s | 18.2-18.8 s | 51.8 s | 283-286 s |
| 2 (11-14) | 262,144 | 55.0-57.3 s | 13.2-13.5 s | 38.3-39.8 s | 210-219 s |
| | 500,000 | 100.7-101.7 s | 13.2-13.6 s | 80.4-81.9 s | 201-203 s |
| | 524,288 | 104.1-105.6 s | 13.0-13.3 s | 83.8-85.3 s | 199-201 s |
| | 1,000,000 | 206.1-206.2 s | 13.0-13.2 s | 179.7-180.1 s | 206 s |

One build of twice the rows costs 11% less than two from 32,768 rows, 25%
from 65,536, 12% from 131,072 and 7% from 262,144; from 500,000 it costs 1-2%
more. Training stops growing at 65,536 rows, and the graph costs more per row
the larger it is: from 0.11-0.13 ms at 32,768 rows to 0.20 ms at 262,144 in
the first block, and from 0.15 ms to 0.18 ms between 262,144 rows and a
million in the second. That is why it is twice and not more: with
`index_bytes` of 256 MiB of 128-dimensional vectors or more, larger builds
take more memory for nothing.

**In collections**, as in 7d and 7e: SIFT1M rows from scratch, buffered WAL, 6
build threads, on the same shared and loaded machine; 7e and 7f in turns (7e,
7f, 7f, 7e), two runs each, with the same benchmark program. One shard, limit
128 MiB, `index_bytes` 64 MiB (load 10-18):

| what is written | | until all is indexed | to write | batch p50 | p99 | slowest | refused | most in no index | peak memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 writers, 1,000 a batch, refused after 10 s, 1M rows | 7e | 223-241 s | 178-191 s | 13.4-14.4 s | 18.1-22.8 s | 19.8-24.8 s | 734-747 | 190-198 MiB | 1.39-1.49 GB |
| | 7f | **197-211 s** | 170-171 s | 12.6-12.7 s | 16.7-20.0 s | 18.1-22.9 s | 703-713 | 206-207 MiB | 1.53-1.60 GB |
| 1 writer, 1,000 a batch, 1M rows | 7e | 220-236 s | 186-199 s | 0.23-0.24 s | 0.35-0.40 s | 0.48 s | 0 | 174-175 MiB | 0.99 GB |
| | 7f | **207-208 s** | 178 s | 0.22 s | 0.38 s | 0.54 s | 0 | 206 MiB | 1.12 GB |
| 8 writers, single rows, 400,000 | 7e | 93-96 s | 43-44 s | 0.07-0.10 ms | 4.5 ms | 15-20 ms | 0 | 173 MiB | 0.63-0.65 GB |
| | 7f | **70-71 s** | 41 s | 0.08 ms | 4.3-4.4 ms | 17-79 ms | 0 | 173 MiB | 0.68-0.72 GB |
| 16 writers, refused after 10 s, 600,000 | 7e | 123-124 s | 84-85 s | 2.9 s | 4.3-4.4 s | 4.4 s | 0 | 179 MiB | 0.84-0.85 GB |
| | 7f | **116 s** | 82 s | 2.8-2.9 s | 4.4-4.5 s | 4.5-4.6 s | 0 | 206-209 MiB | 0.96-0.97 GB |

Four shards, which share one build place (load 8-25; the first case also the
other way round, 7f, 7e, 7e, 7f, so four runs each):

| what is written | | until all is indexed | to write | batch p50 | p99 | slowest | refused | most in no index | peak memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 writers, limit 48 MiB a shard, `index_bytes` 16, 600,000 rows | 7e | 232-269 s | 109-122 s | 0.04-0.05 s | 4.4-5.1 s | 4.6-5.7 s | 0 | 188-191 MiB | 1.07-1.13 GB |
| | 7f | 250-268 s | 98-111 s | 0.04-0.05 s | 4.9-5.4 s | 5.0-5.6 s | 0 | 210-211 MiB | 1.15-1.30 GB |
| 8 writers, limit 64 MiB a shard, `index_bytes` 8, 600,000 rows | 7e | 219-276 s | 54-61 s | 0.02-0.04 s | 3.8-4.3 s | 3.8-4.4 s | 0 | 240-242 MiB | 0.96-0.97 GB |
| | 7f | **190-259 s** | 41-55 s | 0.02-0.03 s | 3.6-5.0 s | 3.7-5.3 s | 0 | 250-251 MiB | 1.08-1.09 GB |
| 64 writers, refused after 10 s, as the first | 7e | 258-263 s | 109-115 s | 0.25 s | 31-35 s | 34.8-37.8 s | 421-430 | 191-195 MiB | 1.34 GB |
| | 7f | **242-247 s** | 98-100 s | 0.23-0.26 s | 33-35 s | 33.5-35.5 s | 394-433 | 211-212 MiB | 1.41-1.45 GB |

(The 4-shard runs are of the code before its last change, which only sets how
much of a partition a segment may take: the same builds in all of these.)

- **All is indexed 5-18% sooner** with one shard, and 24-27% with 400,000
  single rows: a million rows took 5 builds of up to 128 MiB instead of 7 of
  64 MiB. With four shards 4-8% sooner where 64 writers give up after 10 s,
  and 6-13% with builds of 8 MiB.
- **With four shards and builds of 16 MiB that take turns, there is no clear
  effect:** in the first four runs of that case 7f took 9-12% longer (254-259
  s against 232-234 s); in the four that ran the other way round, 250-268 s
  against 247-269 s. The first four began at loads of 8.5 and 21.8 (7e), 14.5
  and 24.5 (7f). Built alone, two builds of 32,768 rows cost 11% more than one
  of twice that.
- **What got worse: memory, and the longest waits.** What is in no index, and
  what every search scans, is up to 18% more where builds are larger (206 MiB
  against 175 MiB with one writer), and the peak RSS 0.02-0.23 GB more. The
  slowest batch of one writer took 0.54 s, where it took 0.48 s; one of the runs
  with single rows had one of 79 ms (the other 17 ms, as in 7e). In a test
  whose limit is twice `index_bytes`, so that a build takes all there is, the
  slowest write took more of a round of builds in one of three comparisons on a
  loaded machine (below). Likely where writes wait at the end of a build, which
  tells nothing of how far it is (merging and writing): two waits of 61-69 ms
  were there, but that is all that shows it.
- **With the defaults nothing changes,** and nothing was measured.
- **Not covered:** other dimensions (all is 128-dimensional; at more
  dimensions training is more and the graph also costs more per row, so where
  doubling stops paying is not known), builds in several partitions, other
  data, a quiet machine, `index_bytes` above 64 MiB in a collection.

### Tests

The size of a build with an eighth of the limit more, less, with one
partition that holds twice `index_bytes` and a large segment besides, one
that holds less, and one that does not even hold `index_bytes`; and that a
build that waits for its turn counts for what it takes in by then, also when
that grows. Three tests had the old size in them: the RAM bound of writes that
keep pace with builds (the limit and `index_bytes`; now the limit and twice
that), the number of builds while a compaction waits (now a bound on what is
in no index, as builds are half as many), and what a slow build took in (now
computed from its inputs). 12 mutations of the new code were run against the
tests, and all are caught (two were not at first: all segments were about as
large, so which one was taken did not matter; a test now has one of twice the
size).

The tests of background work, databases and the server ran 10 times in each
profile, at once, on the loaded machine (load 14-23), without a failure. In the
first such loops, before the change below, `writes_keep_pace_with_builds`
failed 3 times in 10 in release (never in those of 7e): its slowest write took
51-62% of a round of builds, where it may take half. Those three ran beside
other work (a build, and other tests), which those of 7e did not. Like for
like, with one collection whose limit is twice `index_bytes`, the slowest write
took, of a round:

| | 7e: in the middle | at most | failed | 7f: in the middle | at most | failed |
|---|---:|---:|---:|---:|---:|---:|
| in turns (7e, 7f, 7f, 7e), load 8-20, 20 runs each | 0.14 | 0.39 | 0 | 0.15 | 0.69 | 1 |
| side by side, both profiles of both at once, load 21-32, 10 runs | 0.12 | 0.16 | 0 | 0.19 | 0.37 | 0 |
| the same, noting how far the build was at slow writes (which slows writes), 8 runs | 0.09 | 0.10 | 0 | 0.08 | 0.08 | 0 |

In the failure (557 ms), the other test of this kind, in the same process, had
its slowest write of 20 runs (475 ms): more likely a stall of the machine,
which 7f may leave less room for. So the test now checks what it is about:
writes that were stopped for each build would wait for most of one, 3 times; at
most one may wait for half a round, none for a whole one. (A mutation that has
them stopped is caught, with 3 long waits every time.) Other tests failed under
that load with 7e as well: `writes_that_wait` of the server (once with 7e,
twice with 7f) and `writes_are_not_slowed_down_for_nothing` (once each), which
are not changed, and `index_builds_do_not_wait_for_compactions` as 7e has it
(twice; 7f's checks a bound instead). They wait 100 ms or count builds, where a
loaded machine stops a thread for longer.

Commands (`B`, `S` as above; 7e: the same benchmark, over its core):

```
$B disk-build --base $S/sift_base.fvecs --out work/i --pq-m 32 --iters 25 --threads 6 --build-mem 2 --limit 262144
$B coll-ingest --base $S/sift_base.fvecs --dir work --threads 6 --background true --unindexed-mb 128 --writers 64 --write-wait 10
$B coll-ingest ... --shards 4 --seal-mb 8 --index-mb 16 --unindexed-mb 48 --writers 8 --limit 600000
$B coll-ingest ... --shards 4 --seal-mb 4 --index-mb 8 --unindexed-mb 64 --writers 8 --limit 600000
```

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
- **Record size grows with dimension.** DiskANN stores each node's full
  vector and neighbor list together in one 4 KB SSD page. With 64 neighbors
  (256 B), that fits up to about 950 f32 dimensions, or about 1,900 in f16.
  Longer records span several sectors (8 KB for GIST at f32, 36 KB at 8192
  dims). Storing the graph and the vectors apart pays off once a vector
  spans several sectors: in step 5 it tied at 2048 dims and was 10-18%
  faster at 8192.
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
  candidates with full vectors. Under a selective filter a segment skips the
  graph: PQ distances to its matching rows, then the best re-ranked. The
  per-segment top-k lists are then merged.
- **Deletes:** a tombstone bitmap per segment, with compaction to reclaim space.

## Roadmap

1. ✅ Exact search: distance kernels, top-k, flat index, segment format v1
   (checksummed, atomic writes), dataset I/O, recall harness
2. ◐ SIMD kernels with runtime CPU detection and a micro-benchmark at 32-8192
   dims: AVX2/FMA done. Still to do once there is hardware to test on: AVX-512
   and NEON (the portable kernels should auto-vectorize to NEON on aarch64, but
   that has not been built or tested yet), and batch kernels that score several
   rows per query load
3. ✅ Quantization: PQ with per-collection code size (k-means codebooks,
   uneven subspace splits, ADC lookup tables), `f16`/`bf16`/`sq8` storage for
   full vectors, re-ranking, and recall measured against bytes per vector on
   SIFT1M and 1024/8192-dim data. Follow-ups: F16C decoding, 4-bit fast-scan,
   persisting codebooks
4. ✅ Vamana graph in memory: parallel build (same graph for any thread
   count), beam search with the list size `L` as the recall/latency knob,
   PQ-guided search, identical-vector merging, reachability repair; recall
   and latency on SIFT1M, GIST1M and 8192-dim data. Follow-ups: a graph
   construction for inner product with uneven norms, tuning build parameters
   per dataset, inserts and deletes after the build (step 6), saving the
   graph (step 5)
5. ◐ SSD index. Done: PQ codes in RAM; graph and full vectors on SSD (f32,
   f16 or bf16; layout chosen by record size); checksummed records; direct
   I/O; io_uring beam search; hot-node cache; a partitioned build within a
   memory budget. Measured on SIFT1M, GIST1M and 2048- and 8192-dim data.
   Still to do:
   - a 100M-vector build and search on dedicated hardware (the target per
     node);
   - processing blocks while the round's other reads are in flight;
   - io_uring registered buffers;
   - F16C decoding;
   - more even partitions (split the largest k-means clusters)
6. ✅ Collections: WAL with point-in-time recovery, growing and sealed
   segments, manifest commits, tombstones per row position, index builds and
   compaction from live rows, crash-state and model tests; attribute filters
   (integer and keyword fields; eq, in, range, and, or, not) with a column
   index per segment and a walk-or-scan choice per SSD segment; ingest,
   delete, multi-segment and filtered search measured on SIFT1M.
   Follow-ups (background builds and group commit are done, step 7):
   - filters evaluated lazily, or as sorted lists, instead of a bitmap per
     segment (needed for segments of 100M rows);
   - a graph that knows the labels (per-label entry points or edges), for
     filters whose matches lie away from the query
7. ✅ Server and single-node mode. Done (7a): a collection shared between
   threads, with snapshot reads, index builds and compactions beside writes
   and searches (deletes made meanwhile carried into the new segment), and
   group commit; concurrent search, durable writes and search during a build
   measured on SIFT1M. Done (7b): flushes and index builds on background
   threads, with writes that wait when the builds fall behind; a thread pool
   that searches segments, and shards, in parallel; a catalog of named
   collections split into shards, behind one `Shard` interface.
   Done (7c): writes that no longer wait behind compactions, and whose wait
   can be bounded; an HTTP/JSON server. The shard interface over gRPC is part
   of step 8. Done (7d): writers that are slowed down to what index builds
   keep up with, instead of stopped for each. Done (7e): writes that give up
   keep their place, and have changed nothing in any shard. Done (7f): index
   builds of up to twice `index_bytes` while they are behind.

   Follow-ups:
   - faster index builds, which set the rate of a long ingest: PQ training is
     about 45% of one of 64 MiB, and the graph costs more per row in larger
     ones (see 7f); where larger builds stop paying at other dimensions;
   - builds that tell how far they are in all their steps: the scan for
     identical rows and the end (merging and writing) tell nothing, and writes
     are likely to wait for them (see 7f);
   - index builds of fewer than 256 different vectors, which fail (there are
     too few to train PQ with), again and again;
   - merges of SSD segments that keep up with a long ingest (nothing slows
     writes down for those);
   - places in the server per collection: the writers of one that is slowed
     down can take all 64;
   - a limit for all shards of a database, not for each;
   - stopping an `optimize` of a collection that is dropped;
   - server: TLS, a key per collection or user, streamed bulk imports, requests
     that are cancelled when their client is gone, or take too long, HTTP/2,
     compression, backups;
   - the cost of handing a search to another thread (see 7c);
   - rows with `with_vectors` read in batches;
   - cheaper scans of fresh rows (int8 codes, or scans shared by searches):
     memory bandwidth limits them;
   - growing rows read without a lock (7a measured about 10% of the search
     p99, so it was left out);
   - flush fsyncs in chunks, which share the disk with the logs better
8. Distributed mode: remote shards (which need a `begin` of their own: where
   all is done in the second half, parts of a refused write are written),
   query nodes, metadata service, replication, rebalancing
