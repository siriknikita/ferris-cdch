# Benchmarking

Performance numbers are engineering measurements of this implementation on one machine.
They say nothing about CDCH's scientific validity or predictive accuracy.

## 1. Tools

| Tool | Measures | Command |
|---|---|---|
| `ferris-bench` (`cdch-bench`) | single-image latency of every stage on decoded pixels in memory; batch throughput 1…N threads; peak RSS | `cargo run --release -p cdch-bench --bin ferris-bench -- --images DIR [--trials N] [--out FILE] [--only NAME]` |
| `scripts/bench_cli.sh` | the `cdch` CLI including PNG decoding, 1…14 threads | `scripts/bench_cli.sh PAIRS_DIR` |
| `python/benchmark_*.py` | the Python research pipeline end to end (v0.1.0 study) | see `PROJECT_STATE.md` |
| Studio `pipeline_bench` | decoding, preparation, CDCH, maps, pixel differences and exports in the desktop application | `cargo run --release -p studio-core --example pipeline_bench` (CDCH Research Studio repository) |

Method of `ferris-bench`: 50 warm-up calls, then 2,000 timed calls (fewer for slow
stages), cycling through the input images to avoid a single hot cache line; median and
p95 per call; batch throughput is the median of 5 runs over 4,000 images. Correctness is
not measured here; that is the job of the tests and `ferris-diff`. CI never asserts
absolute timings.

## 2. Environment of the reported numbers

Apple M4 Pro (14 logical CPUs), 24 GiB, macOS 27.2.1 arm64; rustc 1.91.1; release
profile `opt-level = 3`, fat LTO, `codegen-units = 1`; 80 decoded 128 × 128 DIV2K-derived
crops (kept outside the repository); archived protocol (centre (64, 64), 1°, 360 rays,
26,024 samples, 14,758 distinct pixels).

## 3. Results for v0.2.0 (2026-11-18)

Single-threaded, per image, µs (median / p95). Full data: `reports/perf/v0.2.0/`.

| Stage | Median | p95 | Baseline v0.1.0 median |
|---|---:|---:|---:|
| Core CDCH matrix into a reused buffer (`core_engine_matrix_into`) | 56.2 | 65.5 | 56.3 |
| Counts + matrix, single pass (`histograms_and_matrix_into`) | 58.0 | 67.0 | — |
| Public API, matrix only (`Analyzer::matrix`, cached plan) | 62.0 | 71.8 | — |
| Public API, full result (`Analyzer::compute`) | 65.9 | 74.3 | — |
| Public API, fresh plan every call (`ferris_cdch::compute`) | 480 | 493 | — |
| Sampling-plan construction (geometry only) | 428 | 589 | 431 |
| Representation A, fused, end to end | 56.0 | 58.1 | 56.1 |
| Literal reference port (test oracle only) | 356 | 363 | 240 |

| Throughput (features A, decoded images) | 1 thread | 14 threads |
|---|---:|---:|
| v0.2.0 | 17,757 img/s | 179,683 img/s |
| v0.1.0 | 17,508 img/s | 152,331 img/s |

The 14-thread figures saturate the machine and vary by about ±15 % between runs.

**Memory.** The plan for 128² holds 228 KB; a `CdchResult` for 360 rays holds about
1.1 MB (matrix 737 KB + counts 369 KB). Peak RSS of the benchmark process, which holds
4,000 decoded images, was 126 MiB. Allocation counts were not instrumented. By code
inspection, `Analyzer::compute` makes three allocations per call (matrix, counts, ray
lengths) plus one for the luma workspace.

**Image preparation and decoding** (from the CLI benchmark, v0.1.0, unchanged code):
decoding plus features runs at about 5,300 images/s on one thread, so PNG decoding of a
128² image costs about 130 µs, more than CDCH itself. For larger images in the desktop
application (CDCH Research Studio, rustc 1.100, same machine): decoding a 4000 × 3000 PNG
takes 35 ms and CDCH 1.6 ms (cached plan) or 20 ms (cold plan).

## 4. Interpretation

* The library extraction did not change the cost of any production path.
* The public API adds about 6 µs per call over the raw engine, for allocation and the
  plan-cache lookup; the full result with integer counts adds about 4 µs more.
* Reuse an `Analyzer`: building ray geometry costs about 7 times as much as the CDCH
  itself for 128² images.
* The literal reference port became slower through an LLVM inlining decision. Its source
  only changed formatting. The bisection is documented in
  `reports/perf/v0.2.0/README.md`. It is used only as a test oracle.

## 5. Reproduce

```bash
cargo run --release -p cdch-bench --bin ferris-bench -- \
    --images ../external/bench_test_png --out reports/perf/my_run.json
# Compare with another revision:
git worktree add /tmp/ferris-other <rev>
(cd /tmp/ferris-other && cargo build --release -p cdch-bench --bin ferris-bench)
/tmp/ferris-other/target/release/ferris-bench --images ../external/bench_test_png --out /tmp/other.json
```

Any directory of equally sized PNG files can replace the DIV2K crops; the absolute numbers
then refer to that image size.
