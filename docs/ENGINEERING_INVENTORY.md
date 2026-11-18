# Engineering inventory (library extraction, November 2026)

Recorded on 2026-11-18 before and during the extraction of Ferris into a reusable
library and the creation of CDCH Research Studio. The baseline is commit `553ff8c`
("Merge phase J: reports"), branch `main`, with a clean working tree.

## Starting point

| Item | State at baseline |
|---|---|
| Repository | Git, branch `main` plus `phase-h-optimization`, `phase-i-benchmarks`, `phase-j-cli-mutants`, `phase-j-reports`; remote `origin` = `github.com/siriknikita/project-ferris-rust-cdch` (private); no tags; 13 commits |
| Workspace | `cdch-core`, `cdch-features`, `cdch-python`, `cdch-bench`, `cdch-cli`; version 0.1.0; `rust-version = 1.85`; MIT OR Apache-2.0 (licence texts were not in the repository) |
| Canonical reference | `oracle/ip-1.js` (Muzyka, SHA-256 `110732b4…aab44d8`), archived `oracle/oracle.js`, `oracle/diff_worker.js` |
| Canonical implementation | `cdch-core::reference` (literal port) and `cdch-core::{plan, engine}` (optimized), embedded V8 direction table |
| Public API | `cdch-core` types (`RgbImage`, `Params`, `Center`, `SamplingPlan`, `Engine`, `CdchMatrix`, `CdchError`); no facade intended for third parties |
| Bindings | PyO3 package `cdch_ferris` (`crates/cdch-python`), CLI `cdch` (`crates/cdch-cli`) |
| Image decoding | `cdch-cli::read_png` (PNG only, `png` crate, alpha dropped, grey replicated) |
| Threading | `rayon` in the CLI, bindings and benchmarks; the core is single-threaded and `Sync` |
| Platform-specific code | none |
| Evidence | `reports/differential` (713 cases), `reports/classifier_compatibility`, `reports/perf`, five `RUST_*.md` reports, `study/` archive with `SHA256SUMS` |
| HTML research interface | Not in the repository. The user supplied `../../../cdch-research/CDCH_Workbench_Clean.html` (outside both repositories; Ukrainian interface; PyScript + Chart.js loaded from CDNs) |

## Baseline checks (before any change)

Logs are in `reports/baseline_pre_extraction/`.

| Check | Result |
|---|---|
| `cargo fmt --all -- --check` | **failed** (pre-existing: 158 formatting differences) |
| `cargo clippy --workspace --exclude cdch-python --all-targets -- -D warnings` | **failed** (pre-existing: `manual_div_ceil` and two `unnecessary_map_or` in `cdch-core`, under rustc/clippy 1.91.1; later lints in other crates were hidden behind these) |
| `cargo test --release --workspace --exclude cdch-python` | passed (17 tests) |
| `ferris-diff --mutants` (synthetic fixtures only) | 91/91 cases bitwise, all 15 mutants detected |
| Microbenchmarks | re-run on the baseline binary for comparison (`reports/perf/v0.2.0/`) |

The formatting and lint failures predate this work; they were fixed mechanically
(formatting, `is_none_or`, an iterator loop with the same accumulation order) and the
full differential suite was re-run afterwards (713/713).

## The HTML Workbench (behavioural reference)

`CDCH_Workbench_Clean.html` has three screens:

1. **CDCH of one image** (`ip-1.js` embedded verbatim): image drawn onto a canvas
   scaled to fit 500 × 500; centre X/Y fields and sliders (click on the image to set the
   centre); angle step 1–360 (slider 1–30); "image centre" (`w/2`, `h/2`); "find centre"
   by luma range (`findIntensityCenter`); a greyscale column rendering of the CDCH with
   a 3-column display blur; PNG download.
2. **Image difference** (PyScript/NumPy): A − B with tolerance (levels or percent),
   Rec. 601 luminance or mean RGB difference, automatic bilinear resizing of B to A (or
   A to B) when sizes differ, red/blue overlay on A, statistics, histograms of |A − B|
   for red and blue pixels (fraction of all pixels), cumulative curves per colour.
3. **CDCH difference** (JavaScript): both images drawn onto 256 × 256 canvases, CDCH per
   image with the `ip-1.js` sampling rules but `Float32Array` storage and 0 (not NaN)
   for empty rays, Δ = B − A, MAE, RMSE, max |Δ|, fractions with |Δ| > 1e-7, mean |Δ| by
   angle and by intensity, JSON / CSV / PNG export.

## Inventory

### 1. What currently works

The canonical CDCH (reference port and optimized engine), checkpoint output, the A/B/C/D
representations, fused extraction, the Python bindings, the CLI, the differential and
mutation harness, and the benchmarks.

### 2. What is already verified

Bitwise equality with `ip-1.js` on 713 differential cases (18.0 M ray samples,
64.8 M matrix values); detection of 15 plausible wrong variants; bitwise reproduction of
640 archived matrices; identical AUROC across 840 classifier evaluations (see
`RUST_PORT_VALIDATION.md`, `RUST_CLASSIFIER_COMPATIBILITY.md`).

### 3. Reused unchanged

`cdch-core` algorithms (reference, plan, engine, JavaScript primitives, direction table),
`cdch-features`, the oracle files, all fixtures, the archived study, the Python package
and CLI behaviour. Only formatting and lint-driven rewrites with identical semantics
were applied to existing code.

### 4. Refactored or added

* New crate `ferris-cdch`: the stable public API (typed configuration, validated buffers,
  RGBA/grey conversion with an explicit alpha policy, structured results, CSV export,
  thread-safe analyzer, canonical intensity centroid, typed errors).
* `SamplingPlan::histograms_and_matrix_into`: counts and matrix from one pass, tested
  bitwise against the reference port and the oracle.
* `ferris-diff`: checks the new paths against `ip-1.js`; new `--centroid` comparison
  against `findIntensityCenter` (`oracle/centroid_worker.js`).
* Independent consumer project and script; CI; documentation; licence texts.

### 5. Missing before this work

A documented, typed public API; RGBA handling; the intensity centroid; any GUI-free
consumer proof; CI; licence files; tags or a versioning policy.

### 6. Evidence of mathematical equivalence

`reports/differential/differential_results.json` (v0.1.0) and
`reports/differential_v0.2.0/differential_results.json` (this version, including the
public API and centroid checks), `reports/baseline_pre_extraction/`, the property and
equivalence tests in `crates/*/tests`.

### 7. Implementable immediately

Everything in the library and the desktop application except the items in 8.

### 8. Requires platform-specific verification

* Linux and Windows execution of the test suite (pure Rust, expected to work; CI is
  configured but has not run).
* Bitwise cross-platform agreement of matrices (no host libm in the matrix path, so
  expected; unverified off macOS arm64).
* The Studio's native dialogs, packaging and the Windows window-geometry module.
* The direction table under Node 24 (the archived runtime); see `RUST_PORT_VALIDATION.md` §3.
