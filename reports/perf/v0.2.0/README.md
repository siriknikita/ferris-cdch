# Benchmark: v0.1.0 baseline vs v0.2.0 (library extraction)

Measured on 2026-11-18, back to back on the same machine, with `ferris-bench` (crate
`cdch-bench`) and the same 80 decoded 128×128 PNG crops (DIV2K-derived, kept outside the
repository in `../external/bench_test_png`).

| Item | Value |
|---|---|
| Hardware | Apple M4 Pro, 14 logical CPUs, 24 GiB |
| OS | macOS 27.2.1 (arm64) |
| Compiler | rustc 1.91.1 (ed61e7d7e 2025-11-07) |
| Profile | release: `opt-level = 3`, fat LTO, `codegen-units = 1`, `panic = "unwind"` |
| Parameters | archived protocol: centre (64, 64), angle step 1° (360 rays) |
| Plan | 26,024 samples, 14,758 unique pixels, 228 KB heap |
| Single-image method | 50 warm-up calls, 2,000 timed calls (fewer for slow cases), cycling over the images; median and p95 per call |
| Batch method | 4,000 images, rayon, 5 runs, median throughput |
| Baseline binary | commit `553ff8c` (v0.1.0) built in a temporary worktree |
| Refactored binary | branch `phase-k-library` (v0.2.0) |

Raw data: `microbench_baseline_553ff8c.json`, `microbench_v0.2.0.json`, logs `baseline.log`,
`refactored.log`. Decoding is excluded here (see the CLI throughput figures in
`../cli_throughput.csv` for decode-inclusive numbers).

## Single image (µs per call)

| Benchmark | Baseline median | v0.2.0 median | v0.2.0 p95 |
|---|---:|---:|---:|
| `plan_build` | 430.7 | 427.9 | 589.3 |
| `core_reference_matrix` | 239.9 | 356.2 | 362.7 |
| `core_engine_matrix_into` | 56.3 | 56.2 | 65.5 |
| `core_engine_matrix_alloc` | 61.5 | 62 | 70.8 |
| `core_engine_checkpoints_into` | 27.8 | 28.3 | 30.3 |
| `core_engine_histograms_u32` | 27.8 | 27.9 | 29.9 |
| `core_engine_histograms_and_matrix_into` | — | 58 | 67 |
| `ferris_cdch_analyzer_matrix` | — | 62 | 71.8 |
| `ferris_cdch_analyzer_compute` | — | 65.9 | 74.3 |
| `ferris_cdch_compute_cold_plan` | — | 480.4 | 493.3 |
| `features_A_from_matrix(validate+stats)` | 54 | 54.8 | 55.8 |
| `features_validate_matrix` | 23.9 | 23.9 | 24.6 |
| `features_A_from_checkpoints` | 27.8 | 28 | 28.6 |
| `features_rich42_from_matrix` | 1111.2 | 1128.7 | 1212.9 |
| `features_B_from_matrix` | 1156.5 | 1172.4 | 1252.3 |
| `features_C_from_matrix` | 14.8 | 14.6 | 16.3 |
| `features_C_from_checkpoints` | 1 | 1.2 | 1.3 |
| `e2e_A_fused` | 56.1 | 56 | 58.1 |
| `e2e_A_via_matrix` | 110.6 | 109.8 | 118.7 |
| `e2e_B_fused` | 1194 | 1205.1 | 1281.7 |
| `e2e_B_via_matrix` | 1213.1 | 1228.1 | 1305.2 |
| `e2e_C_fused` | 29 | 28.9 | 30.8 |
| `e2e_C_via_matrix` | 63.2 | 61.8 | 70.6 |

`—`: the benchmark did not exist in the baseline.

## Batch throughput (images/s, features, decoded images in memory)

| Rep. | Threads | Baseline | v0.2.0 |
|---|---:|---:|---:|
| A | 1 | 17508 | 17757 |
| B | 1 | 896 | 895 |
| C | 1 | 34477 | 33915 |
| A | 2 | 34988 | 34916 |
| B | 2 | 1760 | 1762 |
| C | 2 | 66312 | 64777 |
| A | 4 | 68753 | 68364 |
| B | 4 | 3422 | 3452 |
| C | 4 | 130260 | 129766 |
| A | 8 | 133451 | 131319 |
| B | 8 | 6737 | 6768 |
| C | 8 | 253718 | 249554 |
| A | 14 | 152331 | 179683 |
| B | 14 | 9304 | 9389 |
| C | 14 | 331030 | 320436 |

## Interpretation

* **Production paths are unchanged.** The optimized engine (`core_engine_*`), the fused
  feature paths (`e2e_*_fused`), feature construction and batch throughput agree with the
  baseline within run-to-run variation (about ±2 %; the 14-thread figures vary more because
  they saturate the machine).
* **Public API overhead.** `ferris_cdch_analyzer_matrix` (plan-cache lookup plus result
  allocation) costs the same as `core_engine_matrix_alloc`. `ferris_cdch_analyzer_compute`,
  which additionally returns integer counts and ray lengths through the single-pass
  `histograms_and_matrix_into`, adds about 4 µs. The one-off `ferris_cdch::compute` builds a
  fresh sampling plan each call (about 430 µs); reuse an `Analyzer` for repeated work.
* **Literal reference port: measured slowdown, not recovered.** `core_reference_matrix`
  (the deliberately simple transcription of `ip-1.js`, used only as a test oracle and never
  in production paths) went from 240 µs to 322–356 µs. Its source changed only in
  formatting. Bisection on the baseline tree showed that one semantically identical clippy
  rewrite alone (`(360 + step - 1) / step` → `360.div_ceil(step)` in `Params::n_rays`)
  reproduces the slowdown, which points to an LLVM inlining-heuristic effect under fat LTO.
  In the final tree, restoring that expression, `#[inline(always)]` on `js::round`,
  `js::luma`, `js::ray_point` and `RgbImage::rgb` did **not** recover it, so the extra code
  linked into the benchmark binary also influences the decision. The change was left as is
  because the reference port's purpose is correctness, not speed.
