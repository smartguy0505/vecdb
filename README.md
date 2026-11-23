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
has not been attempted yet: it needs a dedicated machine. Step 6 is under
way: collections with a write-ahead log, segments, deletes, compaction and
crash recovery work; attribute filters are next.

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
                     segments, deletes, compaction), file formats
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
let mut c = Collection::create("data/docs", cfg, OpenOptions::default())?;
c.upsert(42, &vector, &Attrs::new().keyword("lang", "en").int("year", 2024))?;
c.delete(7)?;
let hits = c.search(&query, 10, &SearchOptions::default())?;
c.maintain()?; // the index builds and compactions that are due
```

```
<dir>/MANIFEST            which files make up the collection (replaced atomically)
      LOCK                held while a process has the collection open
      wal-<gen>.log       writes since the last flush
      seg-<id>.flat       sealed segment: ids + vectors, searched exactly in RAM
      seg-<id>.vdx        sealed segment: SSD index (step 5)
      seg-<id>.attrs      attributes per row, when the collection declares fields
      seg-<id>.del-<gen>  tombstones: the segment's deleted rows
```

- **Write path.** A batch of upserts and deletes is checked, appended to the
  WAL as one checksummed record (fsynced with `Durability::Sync`), then
  applied. An upsert deletes the id's current row, wherever it is, and
  appends the new row to the growing segment, which lives in RAM and is
  searched exactly. Once it holds `seal_bytes` of vectors, a flush seals it.
  The flush writes the live rows as a flat segment file and the tombstones of
  every segment that gained deletes, starts a new WAL, and commits all of it
  with a new manifest.
- **Deletes** are tombstones per row position in a segment, not per id, so a
  deleted row never hides a later insert of the same id. Identical vectors
  share one graph node in an SSD index but keep their own positions, ids,
  attributes and tombstones.
- **Index builds and compaction** (`maintain`) replace sealed segments with
  one new segment built from their live rows, through the partitioned
  build. The flat segments become an SSD index once they hold `index_bytes`
  of live vectors. An SSD segment with 20% of its rows deleted is rebuilt
  without them, and the two smallest SSD segments are merged while there are
  more than `max_indexed_segments`. Rows are read back as stored, already
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
- **One writer.** Writes take `&mut self`, searches `&self`, and nothing runs
  in the background yet. Sealed segments are shared through `Arc` and only
  their tombstones change (atomically), so step 7 can search a snapshot of
  the segment list while a writer works.
- **RAM per row of an SSD segment**, beyond the index: 4 bytes for the
  id lookup (positions sorted by id) and 1 bit of tombstone. Flat and
  growing segments hold their vectors. With fields declared, every segment
  also keeps each row's attributes in RAM as written (field names
  included): a provisional format that the filter indexes are to replace.

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
  batches; combining concurrent writers' batches into one fsync (group
  commit) comes with step 7.
- A flush (64 MiB written and fsynced) stalls its batch for 0.24-0.3 s.
- Reopening took 0.08 s with an empty WAL and 0.30 s to replay 76,000
  writes (40 MB).
- Index builds run inside `maintain`, so they block writes. Ingest with
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
  trades rebuild work against this. Step 7 is to search segments
  concurrently, with their reads in flight together, to scan in parallel,
  and to build indexes in the background.

Not yet: attribute filters (the rows already store attributes), group
commit, and inserting into an SSD index in place instead of rebuilding it.

Commands (`B`, `S` as above):

```
$B coll-ingest --base $S/sift_base.fvecs --dir sift1m --pq-m 32 --optimize true
$B coll-ingest ... --durability sync --batch 100 --secs 5
$B coll-ingest ... --index-mb 128 --max-ssd 16 --maintain true
$B coll-search --from sift1m --dir work --base $S/sift_base.fvecs --query $S/sift_query.fvecs \
   --l 40,80,160 --delete 0,0.1,0.3,0.5 --adapt both --compact true --direct false
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
6. ◐ Collections. Done: WAL with point-in-time recovery, growing and sealed
   segments, manifest commits, tombstones per row position, index builds and
   compaction from live rows, crash-state and model tests; ingest, delete
   and multi-segment search measured on SIFT1M. Still to do: attribute
   filters
7. Server and single-node mode: gRPC/HTTP API, collections, the shard
   interface, concurrent reads and writes
8. Distributed mode: remote shards, query nodes, metadata service,
   replication, rebalancing
