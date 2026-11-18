# Implementation report — Ferris 0.2.0 (library extraction)

Date: 2026-11-18. Machine: Apple M4 Pro, macOS 27.2.1 arm64. Toolchains: rustc 1.91.1
(default), 1.100.0 (lint check), 1.85.0 (MSRV check); Node 22.23.4.

This report covers the Ferris repository. The report for the whole assignment, including
CDCH Research Studio, its clean-checkout tests and the final hand-off, is
`../cdch-research-studio/docs/IMPLEMENTATION_REPORT.md`.

## 1. Initial state

Commit `553ff8c` on `main`, clean tree, remote `origin` (private GitHub), no tags; five
crates; no public facade, no licence files, no CI. Baseline: tests passed (17), synthetic
differential 91/91, but `cargo fmt --check` and `clippy -D warnings` failed (pre-existing).
Details: [ENGINEERING_INVENTORY.md](ENGINEERING_INVENTORY.md),
`reports/baseline_pre_extraction/`.

## 2. Final structure

New: `crates/ferris-cdch` (public API), `examples/independent-consumer`,
`oracle/centroid_worker.js`, `scripts/verify_independent_consumer.sh`,
`.github/workflows/ci.yml`, `docs/`, `CHANGELOG.md`, `CONTRIBUTING.md`, licence texts,
`reports/{baseline_pre_extraction,differential_v0.2.0,perf/v0.2.0}`. Existing crates
unchanged in behaviour; `cdch-core` gained `SamplingPlan::histograms_and_matrix_into`.

## 3. Implemented

Typed configuration, validated RGB / RGBA / grey buffers with explicit alpha policy,
structured results (matrix, counts, ray lengths), lossless CSV, thread-safe analyzer with
plan cache, canonical intensity centroid, typed errors without panics, provenance
(`build_info`), optional `representations` feature, two examples, independent consumer.

## 4. Preserved mathematical behaviour

Every rule of [MATHEMATICAL_COMPATIBILITY.md](MATHEMATICAL_COMPATIBILITY.md): V8
directions, ECMAScript rounding, left-to-right binary64 luma, width-limited radius,
multiplicity, integer counts with one division, `NaN` for empty rays. No algorithm code
was changed except formatting and lint rewrites with identical semantics.

## 5. Commands executed and results

| Command | Result |
|---|---|
| `cargo fmt --all -- --check` | pass (fail at baseline, fixed) |
| `cargo clippy --workspace --exclude cdch-python --all-targets --all-features -- -D warnings` (1.91.1 and 1.100.0) | pass |
| `PYO3_PYTHON=python/.venv/bin/python cargo clippy -p cdch-python --all-targets -- -D warnings` (1.91.1, 1.100.0) | pass |
| `cargo test --release --workspace --exclude cdch-python --all-features` | 36 passed, 0 failed |
| `cargo +1.85.0 test -p ferris-cdch --all-features --release` | 17 + 2 doctests passed |
| `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --exclude cdch-python --no-deps --all-features` | pass |
| `ferris-diff --mutants --centroid --div2k … --png-dir pairs=…` | 713/713 bitwise; 15/15 mutants; centroid 5,680/5,680 |
| `scripts/verify_independent_consumer.sh` | `INDEPENDENT_CONSUMER_OK` (fresh clone, commit `4c96576`) |
| `ferris-bench` baseline vs v0.2.0 | production paths unchanged; see §6 |
| `actionlint .github/workflows/ci.yml` | pass |
| `cargo build --release` | pass (part of the test and benchmark runs) |

## 6. Benchmarks

Per 128² image, one core, median: engine matrix 56.2 µs (baseline 56.3), public
`Analyzer::compute` 65.9 µs, fused representation A 56.0 µs (baseline 56.1); batch A
17,757 img/s per core. The literal reference port (test oracle) slowed from 240 to
322–356 µs through an LLVM inlining change (bisected, not recovered, documented).
[BENCHMARKING.md](BENCHMARKING.md), `reports/perf/v0.2.0/`.

## 7. Platforms

Verified: macOS arm64. Configured in CI but never run: Linux x86-64, Windows x86-64, macOS
runners. No platform-specific code exists in Ferris.

## 8. CI

`.github/workflows/ci.yml` (lint, docs, tests ×3 OS, MSRV, differential ×3 OS, consumer,
Python binding, audit). Validated syntax only; never executed.

## 9. Known limitations and risks

- Cross-platform bitwise identity is expected (no libm in the matrix path) but unverified.
- Direction table recorded with Node 22, archived study used Node 24 (not diffed).
- Translucent images: browsers' canvas behaviour is not reproduced; alpha is an explicit
  protocol choice.
- Reference-port benchmark regression (test-only code).
- Natural-image fixtures depend on licensed external data.

## 10. Git

Branch `phase-k-library` with commits `4c96576` (public API, single-pass output, centroid
oracle), `d597002` (benchmarks), `e19146d` (documentation, CI, licences) and this report,
merged into `main`. Annotated tag `v0.2.0` on the merge result. A lightweight `v0.2.0` tag
had been created earlier on `d597002`, so the Studio could be developed against a real tag.
It was never published and was replaced by the annotated tag on the final commit.
Nothing was pushed.

## 11. Actions requiring the owner's authorization

1. `git push origin main phase-k-library v0.2.0`. After that, CDCH Research Studio
   resolves Ferris from GitHub (`scripts/use-local-ferris.sh --remove`).
2. Optionally add a read-only token for the Studio's CI (`FERRIS_READ_TOKEN`).
