# Contributing to vecdb

For people and for coding agents. It says how to get going, where things are,
what must stay true, and what a finished change looks like. What the code does
and why, with measurements, is in [README.md](README.md).

- [Quick start](#quick-start)
- [Where things are](#where-things-are)
- [What must stay true](#what-must-stay-true)
- [Making a change](#making-a-change)
- [Tests](#tests)
- [Benchmarks and numbers](#benchmarks-and-numbers)
- [Pull requests](#pull-requests)
- [For coding agents](#for-coding-agents)
- [What to work on](#what-to-work-on)
- [License](#license)

## Quick start

You need Linux and Rust (the declared minimum is 1.89; the code is developed
with 1.99). Nothing else has to be installed.

```
git clone https://github.com/smartguy0505/vecdb.git && cd vecdb
cargo test --workspace                  # a few minutes
cargo test --workspace --release        # the other profile: see "Tests"
cargo build --release
```

The server, on 127.0.0.1 only:

```
target/release/vecdb-server --data data/db      # --help lists the flags
curl http://127.0.0.1:7410/healthz
```

The README section "HTTP" has requests to try. To measure something without
downloading anything:

```
B=target/release/vecdb-bench
$B synth --out data/synth --n 100000 --nq 1000 --dim 128
$B groundtruth --base data/synth/base.fvecs --query data/synth/query.fvecs --out data/synth/gt.ivecs
$B build --base data/synth/base.fvecs --out data/synth/flat.vdb
$B search --segment data/synth/flat.vdb --query data/synth/query.fvecs --gt data/synth/gt.ivecs --k 10
```

`data/` and `target/` are ignored by git. Keep it that way: no data, indexes
or logs in commits.

## Where things are

| crate | what it is | may depend on |
|---|---|---|
| `vecdb-core` | the engine, a library | the standard library; `io-uring` and `libc` on Linux |
| `vecdb-server` | HTTP/JSON server over it | what is in its `Cargo.toml` (`hyper`, `tokio`, `serde`, ...) |
| `vecdb-bench` | command-line tools and benchmarks | `vecdb-core` only |

Inside `crates/vecdb-core/src`, from the bottom up:

| path | what it does |
|---|---|
| `kernels/`, `distance.rs`, `topk.rs` | distance kernels (portable and AVX2/FMA, chosen at run time), metrics, top-k |
| `store.rs`, `flat.rs`, `format.rs`, `dataset.rs` | vectors in RAM, exact search, the flat segment file, dataset files |
| `quant/`, `pq_index.rs` | k-means, product quantization, `f16`/`bf16`/`sq8` codecs |
| `graph/` | the Vamana graph in memory: build and search |
| `disk/` | the SSD index: format, partitioned build, direct I/O and io_uring, search |
| `collection/` | one shard: log (`wal.rs`), segments, flushes, index builds and compaction (`maintain.rs`), background threads (`background.rs`), slowing writes down (`pace.rs`), filters, search |
| `database/` | named collections, split into shards |
| `pool.rs`, `par.rs`, `cancel.rs` | the search thread pool, parallel loops, stopping builds and their progress |

Start with the module comment at the top of a file: that of
`collection/mod.rs` explains the whole engine (write path, read path, crash
safety, concurrency) on one page. Then read the README section of the step.

## What must stay true

A change that breaks one of these needs a discussion first, however good the
numbers are.

1. **Nothing that was acknowledged is lost, and no batch is kept in part:**
   with `Durability::Sync` whatever happens, with `Buffered` unless the machine
   goes down. Files are written under new names, fsynced, and committed by
   renaming the manifest into place; what they replace is removed after that.
   A crash, or a power loss, may come at any point in between.
2. **Damaged files are errors, never panics or wrong results.** Every format
   has a magic number, a version and checksums. Reading returns
   `Error::Corrupt`, or `Error::Unsupported` for versions it does not know.
3. **Formats have versions.** Who changes what is written raises the version
   and says what happens to files that are there already.
4. **A search sees a snapshot:** no batch in part, no id twice, no replaced id
   missing. Searches and writes go on during flushes and builds, which write
   their files without the locks those need.
5. **Smaller is closer**, for every metric, and the distances a search of a
   collection returns come from the stored vectors, not from PQ codes, so that
   those of several segments and shards compare.
6. **Builds are reproducible:** the same input and seed give the same bytes,
   for any number of threads. No clocks, no iteration over hash maps, no sums
   of floats in an order that depends on threads, in anything that is stored.
7. **The core has no networking and no new dependencies.** No serde, no async.
8. **`unsafe` stays where it is:** `disk/io.rs`, `kernels/x86.rs`, and one
   function in the server's `main.rs` (the limit on open files). The rest of
   the core and the server denies it. Every use has a `SAFETY:` comment that
   says why it is right.
9. **Bad input is an error.** Dimensions from 1 to 65,536, vectors without NaN
   or infinity, sizes that are checked before memory is taken for them.
10. **Memory and threads have limits,** which are options with documented
    defaults.
11. **Locks have an order.** The comments at the fields of `Engine`
    (`collection/mod.rs`), and where several are taken, say which. Keep them
    right.

## Making a change

For anything that is more than a fix, open an issue first and say what you
want to change, why, and how you will show that it works. Always ask before
you change a file format, the public API, the HTTP API, a default or a
dependency.

- **Small steps.** One change per pull request; no clean-ups on the side.
- **Understand first.** Reproduce a bug before you fix it. Measure before you
  optimize; what you expected is often not what is slow.
- **The smallest change that solves it.** No options or layers for later.
- **Style:** the code is formatted by hand, with lines of up to 120 columns.
  Do not run `cargo fmt` over it: its line length is another one, so it would
  change lines all over. Match what is around you. Names are short words; comments say why, and what is
  not obvious, in plain English. Public items have doc comments.
- **Errors** go to the caller as `Result`. `unwrap` and `expect` are for what
  cannot happen, not for what is unlikely (disks fill up, files go missing).
- **README.** It is also the project's notebook. A change that matters gets a
  paragraph or a section: what it does, the alternatives that were tried, what
  was measured, and what is *not* covered.

## Tests

Run them in **both profiles**. Debug checks overflows and assertions; release
has other timing, and finds other races.

```
cargo test --workspace
cargo test --workspace --release
cargo test -p vecdb-core --lib -- collection::tests::background      # some of them
VECDB_KERNELS=scalar cargo test --workspace                          # without AVX2
```

What is expected of tests for a change:

- **A test that fails without it.** Check that it does: undo the change, or
  break the line it is about, and see it fail. If nothing fails, the test does
  not test it. (That has happened here, more than once: two halves of a fix
  covered for each other, and the test passed with either one missing.)
- **Compare with a reference.** `collection/tests/mod.rs` has a `Model` (a hash
  map) and integer vectors whose distances are exact, so results are compared
  exactly, ties included. Approximate searches check recall.
- **Crashes:** `collection/tests/logs.rs` copies a collection as a power loss
  would leave it, at any point, and opens the copy. New files and new steps of
  a commit belong in there.
- **Damage:** cut every new format short and flip its bytes; each has to give
  an error.
- **Races by design, not by sleeping.** `Engine::at` marks points of flushes
  and builds; a test can stop a thread right there (`block_thread_at` in
  `collection/tests/background.rs`) and look at what others see meanwhile. If
  timing cannot be avoided, run the test a dozen times on a busy machine, in
  both profiles.
- **Files** go where `test_dir` puts them, and are removed at the end. Servers
  listen on 127.0.0.1, at a port the system chooses.

There is no CI yet, so say in the pull request what you ran.

## Benchmarks and numbers

Numbers decide here, so they have to be right.

- **Compare like with like:** the same benchmark program over the old and the
  new code, in turns (old, new, new, old), on the same machine. One run each,
  one after the other, shows little: other programs, caches and heat change
  from minute to minute.
- **Say what was measured:** data, dimension, number of rows, threads, how busy
  the machine was, cold or warm page cache, the command. Give ranges if you
  have several runs, and percentiles for latencies, not only averages.
- **Recall goes with latency.** A faster search with less recall is another
  point on the same curve. Compare at equal recall.
- **Show what got worse,** too, and say what you did not measure.
- Do not compile while a benchmark runs, and give builds fewer threads than
  there are CPUs (`--threads`) when you measure something beside them.
- Data: `vecdb-bench synth` is enough to try things and to find slowdowns.
  Numbers for recall come from real data, such as SIFT1M and GIST1M
  (<http://corpus-texmex.irisa.fr/>), read as they are (`.fvecs`, `.bvecs`,
  `.ivecs`). Random vectors have little structure, and say little about recall.

`vecdb-bench` without arguments lists its commands; the README shows the ones
behind each table.

## Pull requests

Before you open one:

- [ ] the tests pass in both profiles, and there are no new warnings
- [ ] a test fails without the change (say how you checked)
- [ ] nothing under "What must stay true" is broken
- [ ] performance: numbers from both sides, with the commands
- [ ] README and doc comments say what is true now
- [ ] no data, keys, addresses, host names or paths of your machine in the diff

The title says what changes, in a few words, in the style of the log
(`git log --oneline`). The text says why, what else you tried, what you ran,
and what is still open. Pushed commits are not rewritten: no force pushes to
`main`.

## For coding agents

All of the above applies. Also:

**Read in this order:** the status and "Roadmap" in the README, the README
section of the step you work on (the file is long: list its headings with
`grep -n '^##' README.md` and read what you need), the module comment, the
code, its tests.

**Ask the person you work for before you:** commit, push or open a pull
request, unless they have told you to; change a format, an API, a default or a
dependency; start something that loads a disk or all CPUs for long; delete
anything you did not make.

**Do not:**

- make a test pass by weakening it: no looser bounds, longer sleeps, `#[ignore]`
  or retries. If an expectation is wrong, say why in the pull request;
- run formatters or fix lints in files you do not change otherwise;
- report a number you have not measured, or a check you have not run;
- put text from the machine you run on into tracked files: paths, host names,
  addresses, user names, keys, e-mail addresses;
- open the server to a network in tests or benchmarks, or stop processes that
  are not yours.

**Before you say it is done:** run both profiles; break your change and see a
test fail; read your own diff once more, as if someone else had written it;
grep it for private text; list what you did not do or could not verify. Commits
that an agent wrote carry a `Co-Authored-By` line.

**Good to know:** the test profile is built with `opt-level = 1`, so debug tests
are slow but not unusable. Two checkouts must not share a target directory.
`cargo test -p vecdb-core --lib -- <module>` is the quick loop.

## What to work on

The "Roadmap" in the README has the list, with the follow-ups of each step.
Step 8, the distributed mode, is next, and large: ask first. Good places to
start, because they are small or stand alone:

| what | where | how to tell that it works |
|---|---|---|
| NEON kernels; AVX-512 kernels | `kernels/` | the kernel tests, which every set has to pass, and `vecdb-bench kernels` |
| F16C decoding of `f16` vectors | `quant/codec.rs` | the same values as now, and `vecdb-bench pq` |
| faster PQ training (about 40% of the index build of a 64 MiB segment) | `quant/pq.rs`, `quant/kmeans.rs` | the same recall, reproducible for any number of threads |
| building and testing on macOS or aarch64 | `disk/io.rs` has the fallback | the test suite |
| more even partitions in the partitioned build | `disk/build.rs` | time and peak memory of a build, at the same recall |
| CI that runs the tests in both profiles | `.github/workflows` | |

Found a bug? Open an issue with the commit, the command or a small program, and
what you saw and expected. If it may be a security problem, do not write how to
exploit it in a public issue; ask there for a private way to report it.

## License

There is no license file yet. Until there is one, nobody but the author has the
right to use, copy or change the code, and it is not settled under which terms
contributions are made. Ask in an issue before you put work into a larger one.
