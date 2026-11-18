# Changelog

All notable changes to Ferris. Versions follow [Semantic Versioning](https://semver.org/);
any change to computed values is a breaking change (see docs/API.md §8).

## [0.2.0] — 2026-11-18

Library extraction. **No change to any computed value**: 713/713 differential cases are
bitwise equal to `ip-1.js`, as in 0.1.0.

### Added
- `ferris-cdch`, the stable public API: `CdchConfig`, `CenterSpec`, `Analyzer` (thread-safe
  plan cache), `CdchResult` (matrix, integer counts, ray lengths, lossless CSV export),
  `OwnedRgbImage`, `RgbaImage`, explicit `AlphaPolicy` (ignore / composite), grey
  replication, `intensity_centroid` (port of `findIntensityCenter`), typed
  `FerrisError::{InvalidInput, Internal}`, `build_info()`; optional `representations`
  feature; examples `basic` and `rgba`.
- `cdch-core`: `SamplingPlan::histograms_and_matrix_into` (counts and matrix in one pass).
- `ferris-diff`: checks of the single-pass path and the public API against `ip-1.js`;
  `--centroid` compares against `findIntensityCenter` (new `oracle/centroid_worker.js`).
- `examples/independent-consumer` and `scripts/verify_independent_consumer.sh`.
- GitHub Actions CI (not yet run), licence texts, documentation in `docs/`.
- Evidence: `reports/baseline_pre_extraction/`, `reports/differential_v0.2.0/`,
  `reports/perf/v0.2.0/`.

### Changed
- Workspace version 0.1.0 → 0.2.0.
- Code formatted with rustfmt; clippy warnings fixed with semantically identical rewrites
  (`is_none_or`, `div_ceil`, iterator loops with unchanged accumulation order).

### Known
- The literal reference port (`cdch-core::reference`, a test oracle) is about 35–50 %
  slower in the benchmark binary because of an inlining change; production paths are
  unchanged (reports/perf/v0.2.0/README.md).

## [0.1.0] — 2026-11-16

Phases A–J of the Rust port: reference port, optimized engine, representations A–D,
fused extraction, Python bindings, CLI, differential and mutation testing, classifier
reproduction, benchmarks. See `PROJECT_STATE.md` and the `RUST_*.md` reports.
