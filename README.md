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
recovery and attribute filters. Step 7 is under way: a collection is shared
between threads, with snapshot reads, index builds beside writes and group
commit (7a); it flushes and builds in the background and searches its
segments in parallel, and databases hold named collections split into shards
(7b). A server is next.

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
crates/vecdb-bench   CLI: synthetic data, ground truth, segment build, recall/latency runs,
                     SSD index build and search, collection ingest and search,
                     kernel micro-benchmark
```

`vecdb-core` depends only on the standard library, plus `io-uring` and `libc`
on Linux (for the SSD index's reads). On other platforms the SSD index is
written to fall back to positioned reads through the page cache; that path
has not been built or tested outside Linux yet.

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
  (step 7b) bound the unindexed rows by making such a writer wait, which can
  take minutes (see below).
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
let reply = c.search(&SearchRequest { vector: query, k: 10, options: SearchOptions::default() })?;
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

**Builds.** Another thread per collection runs the index builds and
compactions that are due, on half of the CPUs unless told otherwise. All
collections of a database share a limit of builds at once (default 1).
Writes wait while the flat segments take more than `max_unindexed_bytes` (1
GiB) and an index build is due. Dropping a collection stops its build early.

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
- **The limit works by making a write wait for a build to finish, and the
  wait has no bound.** It is short only while the build that is due can
  start. The builds of a collection run one at a time (and those of a
  database share a limit, 1 by default), so a write waits behind whatever is
  running. In the ingest above it waited 26 s for an index build of 128 MiB
  of rows; with a 1M-row segment already there, upserts of its rows crossed
  its 20% of deletes, its compaction (787,344 rows) took 224 s on this
  loaded machine, and the write at the 128 MiB limit waited 213 s for it
  (170 s in another run), although what it needed was a merge of the flat
  segments. The wait grows with the size of the segments. This is the first
  open item (see the roadmap).

Search with a writer going flat out, 4 threads searching (12 s, page cache,
starting from the indexed 1M):

| | QPS | mean ms | p50 / p99 ms | rows/s written | slowest batch |
|---|---:|---:|---:|---:|---:|
| no background work | 102 | 39 | 26 / 132 | 115,000 | 412 ms |
| flushes in the background | 90 | 44 | 29 / 182 | 107,000 | 78 ms |
| and a pool | 91 | 44 | 32 / 154 | 119,000 | 78 ms |
| and a 128 MiB limit | (see above) | 45 | 26 / 176 | | 170 s |

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
7. ◐ Server and single-node mode. Done (7a): a collection shared between
   threads, with snapshot reads, index builds and compactions beside writes
   and searches (deletes made meanwhile carried into the new segment), and
   group commit; concurrent search, durable writes and search during a build
   measured on SIFT1M. Done (7b): flushes and index builds on background
   threads, with writes that wait when the builds fall behind; a thread pool
   that searches segments, and shards, in parallel; a catalog of named
   collections split into shards, behind one `Shard` interface.
   Still to do:
   - 7c: the server (proposed, not decided: HTTP/JSON first, gRPC for the
     shard interface in step 8)

   Follow-ups:
   - first: writes wait behind whatever build is running (170-213 s behind a
     compaction of about 780,000 rows; 26 s in an ingest that compacted
     nothing, above). Jobs that free the limit need a lane of their own, or
     the wait needs a bound;
   - cheaper scans of fresh rows (int8 codes, or scans shared by searches):
     memory bandwidth limits them;
   - growing rows read without a lock (7a measured about 10% of the search
     p99, so it was left out);
   - flush fsyncs in chunks, which share the disk with the logs better
8. Distributed mode: remote shards, query nodes, metadata service,
   replication, rebalancing
