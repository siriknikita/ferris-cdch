# Rust performance report

**Summary.** On one Apple M4 Pro core, the complete `cdch_A_logistic` classifier pipeline drops
from **21.4 ms** (canonical JavaScript worker + Python transport) to **0.43 ms** with the fused
Rust backend, a **49.7×** end-to-end speedup with unchanged outputs. The CDCH stage itself goes
from 20.8 ms to 0.055 ms (**380×**). On this hardware the Rust CDCH-A pipeline is **2.4× faster
than the frequency-only pipeline** (1.01 ms). PNG decoding (0.31 ms, 72% of the latency) is now
the dominant cost.

All values are medians of measured runs on the hardware below. Machine-readable data:
`reports/perf/` (`pipelines/*.json`, `pipeline_benchmarks.csv`, `rust_microbench.json`,
`components.json`, `cli_throughput.csv`, `speedups.csv`) and `reports/results_summary.json`.

## 1. Environment

| Item | Value |
|---|---|
| CPU | Apple M4 Pro, 14 logical cores (arm64) |
| Memory | 24 GiB |
| OS | macOS 27.2.1 (Darwin 27.2) |
| Rust | rustc 1.91.1, release profile (`opt-level=3`, fat LTO, `codegen-units=1`), no `target-cpu=native` |
| Python | CPython 3.12.11 (uv); numpy 2.3.5 (Accelerate), scikit-learn 1.8.0, torch 2.7.1, Pillow 13.0.0 |
| Node | v22.23.4 (V8 12.4) |
| Threads | `OMP/OPENBLAS/MKL/VECLIB=1`, `torch.set_num_threads(1)`; Rust single-threaded unless stated |

**Historical context.** The archived study measured on an Intel Xeon Platinum 8573C (Linux,
Node 24): 84.1 ms (A logistic), 109.5 ms (B linear SVM), 2.89 ms (frequency), 5.31 ms (RGB CNN).
The same JavaScript pipeline takes 21.4 ms here, so absolute numbers are **not** comparable
across machines. Every speedup below compares baseline and Rust on this machine, in the same
session.

## 2. Method

- **End-to-end** (`python/benchmark_pipelines.py`) follows the archived `benchmark_native.py`
  protocol (config `native.json`):
  - holdout PNGs in the archived seeded order (rng 1611);
  - 8 warm-up images and 3 cold sessions;
  - batch size 1, one thread, a fresh subprocess per (model, backend);
  - 60 timed images × 5 passes = 300 trials per configuration.
  
  Per-stage clocks come from the pipeline. CPU % is (Python CPU + Node CPU) / wall time. Peak RSS
  is `getrusage` for Python plus `ps` RSS for Node (macOS has no `/proc`).
- **Rust microbenchmarks** (`ferris-bench`) time in-memory decoded buffers with no decoding and
  no Python: 50 warm-up and 2,000 timed calls per operation, cycling through the 80 holdout PNGs.
- **Components** (`python/benchmark_components.py`): Python→Rust crossing, batch throughput, and
  size scaling of JS vs Rust.
- **CLI** (`scripts/bench_cli.sh`): Rust-only decode + CDCH + linear model, warm file cache,
  3 repeats per thread count.
- **Not measured:** energy (no reliable instrumentation without privileged `powermetrics`), and
  allocations with a counting allocator. The allocation behaviour in §8 is from code inspection.

## 3. End-to-end classifier pipelines (single image, one thread)

| Model | Backend | Median ms | p95 ms | Cold first image ms | img/s | Peak RSS MiB | CPU % |
|---|---|---:|---:|---:|---:|---:|---:|
| cdch_A_logistic | canonical_js | 21.386 | 22.633 | 79.1 | 47 | 364 | 127 |
| cdch_A_logistic | rust_matrix | 0.625 | 0.678 | 4.4 | 1,597 | 257 | 100 |
| cdch_A_logistic | **rust** | **0.430** | 0.502 | 4.2 | 2,292 | 256 | 100 |
| cdch_B_linear_svm | canonical_js | 23.518 | 24.757 | 84.3 | 42 | 363 | 126 |
| cdch_B_linear_svm | rust_matrix | 2.592 | 2.832 | 9.9 | 381 | 263 | 100 |
| cdch_B_linear_svm | **rust** | **1.591** | 1.800 | 8.1 | 631 | 256 | 100 |
| cdch_C_logistic | canonical_js | 21.436 | 22.693 | 86.4 | 46 | 363 | 127 |
| cdch_C_logistic | **rust** | **0.415** | 0.467 | 7.2 | 2,407 | 256 | 100 |
| cdch_D_conv | canonical_js | 21.709 | 23.057 | 92.9 | 46 | 366 | 127 |
| cdch_D_conv | **rust** | **0.374** | 0.405 | 7.3 | 2,696 | 260 | 100 |
| frequency_cdch | canonical_js | 22.369 | 23.392 | 84.8 | 45 | 364 | 126 |
| frequency_cdch | **rust** | **1.076** | 1.126 | 8.0 | 930 | 257 | 100 |
| frequency_cdch_score_fusion | canonical_js | 22.635 | 23.684 | 89.9 | 44 | 360 | 126 |
| frequency_cdch_score_fusion | **rust** | **1.165** | 1.310 | 11.8 | 849 | 257 | 100 |
| frequency (no CDCH) | — | 1.012 | 1.070 | 8.0 | 989 | 257 | 100 |
| RGB CNN (no CDCH) | — | 1.453 | 1.639 | 6.4 | 680 | 269 | 100 |

The full table, including `rust_matrix` rows for every model, is in
`reports/perf/pipelines/pipeline_benchmarks.csv`. Peak RSS is dominated by the Python process
importing torch and sklearn (about 255 MiB in every configuration). The Node worker adds 98 MiB.

### Stage breakdown, `cdch_A_logistic` (median ms)

| Stage | canonical_js | rust_matrix | rust (fused) |
|---|---:|---:|---:|
| PNG decode + RGB array (PIL) | 0.425 | 0.309 | 0.308 |
| Python JSON packing of pixels | 1.200 | – | – |
| Node JSON parse | 0.338 | – | – |
| **ip-1.js execution** | **10.925** | – | – |
| Node Float64→base64 reply | 3.270 | – | – |
| Python base64 decode | 1.240 | – | – |
| pipe/scheduling (rest of IPC) | ≈3.5 | – | – |
| Rust CDCH matrix → NumPy | – | 0.059 | – |
| Representation (NumPy / Rust) | 0.294 | 0.185 | **0.054** (one call: CDCH + A) |
| Classifier (sklearn logistic) | 0.177 | 0.072 | 0.068 |
| **Total** | **21.386** | **0.625** | **0.430** |

The canonical CDCH stage (20.78 ms) splits into the JavaScript algorithm, **10.9 ms (53%)**, and
interoperability (serialization, pipes, parsing), **9.5 ms (46%)**, plus 0.3 ms for NumPy
features. The archived Xeon showed a similar split (47.9 ms JS, 33.8 ms IPC).

## 4. Speedups, S = T_baseline / T_Rust (same machine)

| Scope | Comparison | Baseline | Rust | S |
|---|---|---:|---:|---:|
| Core algorithm | 128² CDCH matrix: ip-1.js in Node vs Rust engine | 10.93 ms | 57.3 µs | **191×** |
| Core algorithm | 128² reduced 32-bin CDF (what A/C/D need) | 10.93 ms | 28.1 µs | **389×** |
| Core algorithm | 128² CDCH matrix: ip-1.js vs literal Rust reference port | 10.93 ms | 326 µs | 34× |
| Features | A from CDCH data: NumPy vs Rust | 294 µs | 28.0 µs | 10.5× |
| Features | B from CDCH data: NumPy vs Rust | 2.23 ms | 1.17 ms | 1.9× |
| Features | C from CDCH data: NumPy vs Rust | 43.1 µs | 1.1 µs | 40× |
| CDCH stage | pixels → A in Python (incl. transport / crossing) | 20.78 ms | 54.8 µs | **380×** |
| CDCH stage | pixels → B | 22.92 ms | 1.17 ms | 19.6× |
| CDCH stage | pixels → C / D | 20.8–21.1 ms | 31 µs | 666–684× |
| Full pipeline | cdch_A_logistic | 21.39 ms | 0.430 ms | **49.7×** |
| Full pipeline | cdch_B_linear_svm | 23.52 ms | 1.591 ms | 14.8× |
| Full pipeline | cdch_C_logistic / cdch_D_conv | 21.4–21.7 ms | 0.37–0.42 ms | 51.6× / 58.0× |
| Full pipeline | frequency_cdch / score fusion | 22.4–22.6 ms | 1.08–1.17 ms | 20.8× / 19.4× |
| Batch throughput | A features from decoded arrays, 1 thread vs one JS worker | 45.9 img/s | 17,122 img/s | 373× |
| Batch throughput | same, 14 threads | 45.9 img/s | 139,175 img/s | 3,032× |
| Batch throughput | decode + A + logistic, Rust CLI 1 thread vs JS pipeline | 46.6 img/s | 5,327 img/s | 114× |
| Batch throughput | same, CLI 14 threads | 46.6 img/s | 41,529 img/s | 892× |

The JavaScript batch baseline is one worker process. Running one Node worker per core would raise
it by up to about 14×, so the fair multi-core ratios are about 14× smaller than the 14-thread rows
(still about 64× for the CLI).

## 5. Scaling with image size (single image, natural content)

| Size | ip-1.js core (Node) | Rust engine matrix | Rust reference port | Rust fused A |
|---:|---:|---:|---:|---:|
| 64² | 6.18 ms | 51 µs | 272 µs | 45 µs |
| 128² | 11.35 ms | 63 µs | 331 µs | 57 µs |
| 256² | 20.53 ms | 95 µs | 442 µs | 85 µs |
| 512² | 38.97 ms | 158 µs | 670 µs | 148 µs |
| 1024² | 74.95 ms | 298 µs | 1,217 µs | 265 µs |

(Rust figures measured from Python, so they include the ~1 µs crossing.) Both implementations do
work proportional to the number of samples, 360 × (ray length) ≈ 360 · W / √2 to 360 · W / 2. The
Rust matrix path also has a fixed cost of writing 360 × 256 doubles (737 KB), which dominates
below 128².

## 6. Rust microbenchmarks (128², in-process, release)

| Operation | Median µs | p95 µs |
|---|---:|---:|
| Sampling-plan build (once per geometry) | 405 | 446 |
| Literal reference port, full matrix | 326 | 350 |
| Engine: full matrix into buffer | 57.3 | 67.4 |
| Engine: 32-bin checkpoint CDF | 28.1 | 32.6 |
| Engine: integer histograms (u32) | 27.8 | 31.2 |
| A from matrix (incl. validation) / from checkpoints | 54.2 / 28.0 | 60.5 / 30.5 |
| Matrix validation (descriptor checks) | 23.9 | 26.5 |
| B rich statistics (42) | 1,123 | 1,231 |
| C from matrix / from checkpoints | 14.9 / 1.1 | 17.2 / 1.2 |
| **Fused A / B / C, pixels → features** | **56.3 / 1,210 / 29.0** | 60.2 / 1,314 / 31.5 |
| Unfused A / B / C (matrix, then features) | 109.5 / 1,225 / 62.5 | 122.0 / 1,322 / 75.5 |

Batch throughput (in-process, `rayon`, plan shared):

| Threads | A img/s | B img/s | C img/s |
|---:|---:|---:|---:|
| 1 | 17,721 | 892 | 34,105 |
| 2 | 34,873 | 1,752 | 66,474 |
| 4 | 68,515 | 3,418 | 128,949 |
| 8 | 129,426 | 6,702 | 250,906 |
| 14 | 170,646 | 8,810 | 321,520 |

Scaling is close to linear up to the 10 performance cores. The last 4 threads land on efficiency
cores. Multithreading does not reduce single-image latency (one image is one task) and is used
only for batches.

### Optimization log (all changes exact; verified after each step)

| Step | Effect (median, 128²) |
|---|---|
| Literal port → precomputed sampling plan (unique-pixel luma, CSR sample lists) | 326 → 64 µs matrix |
| Table-driven luma (3 × 256 products, exhaustively proven equal) | 64 → 57 µs matrix; 43 → 35 µs checkpoints |
| Checkpoint CDF from per-length quotient tables `c/n` + 4 interleaved bucket counters | 35 → 28 µs checkpoints |
| Luma + checkpoint changes combined, end to end | fused A 74 → 56 µs; fused C 46 → 29 µs |
| Plan-level validity replaces the O(360×256) scan in fused B; single-pass `validate_matrix` | validation 73 → 24 µs; A from matrix 101 → 54 µs |
| Zero-PMF shortcut for `pow`/`log2` (bit-identical signed zeros) | B rich 1,388 → 1,123 µs |
| Branchless TwoSum in the angular DFT | A statistics 32 → 28 µs; NumPy agreement unchanged |
| *Rejected:* quotient tables for the 256-bin matrix, sparse histogram reset | slower (57 → 61–62 µs); reverted |
| *Rejected:* slice-based pairwise sums to vectorize `rich` | no change (libm-bound); kept for clarity |

No `unsafe`, SIMD intrinsics or target-specific flags are used. All crates except the PyO3
bindings use `#![forbid(unsafe_code)]`; the bindings contain no hand-written `unsafe` blocks.

## 7. Python/Rust crossing and interoperability

| Call from Python (128²) | Median µs |
|---|---:|
| `features(rgb, "A")` | 56.8 |
| `features(rgb, "C")` | 29.9 |
| `features(rgb, "B")` | 970.8 |
| `cdch(rgb)` (returns the 360×256 array) | 62.9 |
| `features_multi(rgb, ["A","B","C"])` | 1,052 |
| `features` on a 1×1 image (upper bound on fixed call cost) | 8.5 |

`features(…, "A")` from Python (56.8 µs) equals the in-process Rust time (56.3 µs) within noise,
so the PyO3 crossing costs about 1 µs. The 1×1 call is only an upper bound, since it still does
360 rays of work. Returning the full matrix to Python adds about 6 µs over the in-process matrix
(an allocation plus a 737 KB fill, with no serialization). In the canonical design the same data
crosses a pipe as 0.74 MB of float64, base64-encoded (about 0.98 MB) inside JSON, costing 9.5 ms.

## 8. Memory and allocation

| Configuration | Peak RSS |
|---|---:|
| Python + Node canonical pipeline (sum of process peaks) | 360–366 MiB (Node: 98 MiB) |
| Python + Rust extension pipeline | 255–263 MiB (torch/sklearn imports dominate) |
| Rust CLI, 400 images, 1 thread | 3.3 MiB |
| Rust CLI, 400 images, 14 threads | 10.6 MiB |
| Sampling plan, 128² archived geometry | 210 KB (26,024 samples, 14,758 unique pixels) |

By code inspection, the steady-state fused A/C/D path allocates nothing in Rust. The plan is
cached, and luma, checkpoint and scratch buffers live in a reused per-thread `FeatureExtractor`.
The only per-call allocations are the output vector and the NumPy array object. B reuses an
internal 737 KB matrix buffer. `cdch()` necessarily allocates its 737 KB result.

## 9. Cold start

The first image of a fresh session costs 79–93 ms with the canonical backend: Node spawn, `vm`
evaluation of `ip-1.js`, JIT warm-up, and model load. With Rust it costs 4–12 ms, mostly model
loading; the plan build itself is 0.4 ms. The Rust CLI processes 400 images, including process
start, in 75 ms on one thread.

## 10. Is Rust CDCH competitive with the frequency classifier?

On this hardware, yes for A, C and D. The A pipeline (0.43 ms) is 2.4× faster than the frequency
pipeline (1.01 ms), whose NumPy frequency features take 0.62 ms, against 0.054 ms for fused CDCH-A.
B (1.59 ms) remains slower than frequency because of its per-bin moment statistics. The
frequency + CDCH hybrid costs 1.08 ms, essentially the frequency pipeline plus 0.06 ms. The
frequency extractor was not ported to Rust, so this compares a Rust CDCH path against a NumPy
frequency path. A Rust FFT-based frequency extractor would likely be faster still. CDCH is no
longer the expensive half of the hybrid.

## 11. Remaining bottlenecks

For the Rust `cdch_A_logistic` pipeline (0.43 ms):

1. **PNG decoding, 0.31 ms (72%).** zlib inflate plus PIL. Decoding is shared by every method,
   including frequency and RGB.
2. **Python-side classifier call, 0.07 ms (16%).** sklearn `decision_function` overhead. The same
   linear model in Rust takes 0.08 µs (`cdch score`).
3. **CDCH + features, 0.054 ms (13%).** Half sampling and histogramming (luma of about 15k pixels,
   26k scattered increments), half the compensated 8-frequency DFT.

For B, about half the 1.1 ms in the rich statistics is libm `pow`, called on every nonzero PMF bin
(about 43k calls per image). It is kept deliberately: replacing `pow(c, 3)` with `c·c·c` would
break bitwise agreement with NumPy, and NumPy's own result is platform-specific.

The approach is therefore no longer limited by the CDCH algorithm, by interoperability, or by
feature extraction (for A/C/D), but by image decoding and per-call classifier overhead. Further
CDCH engineering such as SIMD or DFT folding could save at most about 13% of single-image latency.
That is not justified for the present research questions. The self-contained Rust CLI removes the
remaining Python overhead when deployment throughput matters.
