# Validation

How Ferris is shown to compute the canonical CDCH, and the evidence for version 0.2.0.
Earlier evidence (v0.1.0, phases A–J) is in `RUST_PORT_VALIDATION.md` and
`RUST_CLASSIFIER_COMPATIBILITY.md` and remains valid. The algorithm code it covers has
not changed.

## 1. Layers

| Layer | What | Where | Needs |
|---|---|---|---|
| Unit | JavaScript primitives (`Math.round` edge cases, luma bounds, exhaustive luma table over 2²⁴ colours, cardinal directions) | `cdch-core/src/js.rs` | — |
| Engine equivalence | Optimized engine, checkpoints and the single-pass output vs the literal reference port: 300 random geometries (1–300 px, all centre kinds, steps 1…360), the archived 128² protocol, strided views, NaN rows, invalid input | `cdch-core/tests/engine_vs_reference.rs` | — |
| Public API | 17 integration tests: reference equality through the facade (120 random cases, every centre kind), degenerate sizes (1×1, 1×N, N×1, 8×2048, 2048×8), radius limit = width, typed errors without panics, NaN rows, histogram properties (conservation, monotonicity, exact `F`), determinism, plan cache, thread safety (8 threads), strided views, RGBA policies, exact compositing on all colour/alpha pairs, grey replication, CSV round trip, centroid semantics, build info, the `representations` feature | `ferris-cdch/tests/public_api.rs` | — |
| Doctests | The examples in the crate documentation | `ferris-cdch/src/*.rs` | — |
| Features | Fused vs matrix representations, bitwise | `cdch-features/tests/fused_equivalence.rs` | — |
| **Differential** | Rust vs the **unmodified `ip-1.js`** executed by Node: coordinates, multiplicities, counts, matrices (reference, engine, single-pass, public API), checkpoints; centroid vs `findIntensityCenter` | `ferris-diff` (`cdch-bench`) + `oracle/*.js` | Node 22 |
| Mutation | 15 plausible wrong implementations must each be detected by at least one fixture | `ferris-diff --mutants` | Node 22 |
| Independent consumer | A separate Cargo project builds and runs against a fresh Git clone | `scripts/verify_independent_consumer.sh` | Git |
| Downstream | CDCH Research Studio: export round trips and a cross-check against the Workbench's own JavaScript | `cdch-research-studio` | — |

## 2. Commands

```bash
cargo test --release --workspace --exclude cdch-python --all-features
cargo run --release -p cdch-bench --bin ferris-diff -- --oracle-dir oracle \
    --out reports/differential_v0.2.0 --mutants --centroid \
    --div2k ../external/div2k --png-dir pairs=../external/pairs   # natural images: optional, licensed data
scripts/verify_independent_consumer.sh
```

## 3. Evidence for v0.2.0 (2026-11-18, macOS 27.2.1 arm64, rustc 1.91.1, Node 22.23.4)

| Check | Result |
|---|---|
| Unit + integration + doctests (`cargo test --release … --all-features`) | 36 passed, 0 failed (6 core unit, 5 engine, 4 + 2 features, 17 public API, 2 doctests) |
| Differential, full suite | **713 / 713 cases bitwise**, 18,003,700 samples, 64,832,000 matrix values, including the new single-pass and `ferris-cdch` checks |
| Mutation testing | 15 / 15 mutants detected |
| Centroid vs `findIntensityCenter` | **5,680 / 5,680** comparisons equal (1,658 of them "no pixel in range") |
| Independent consumer | built and ran against a fresh clone at commit `4c96576` (`INDEPENDENT_CONSUMER_OK`) |
| Clippy `-D warnings` | clean with rustc 1.91.1 and 1.100.0 (whole workspace; `cdch-python` with the project's Python 3.12 venv) |
| MSRV | `ferris-cdch` builds and its 17 integration tests and 2 doctests pass with Rust 1.85.0 |
| `cargo doc` with `-D warnings` | clean |
| Baseline (before changes) | `reports/baseline_pre_extraction/`: tests passed, 91/91 synthetic differential cases; fmt and clippy failed (pre-existing) |

Reports: `reports/differential_v0.2.0/differential_results.json` (per case, plus the
centroid section) and `reports/differential_v0.2.0/run.log`.

## 4. What the natural-image fixtures need

The 713-case suite includes 622 cases built from the DIV2K validation set and the
regenerated pilot pairs. These images are licensed third-party data and are not stored in
the repository (`../external/`, see `PROJECT_STATE.md` → Reproduction). Without them,
`ferris-diff` runs the 91 synthetic cases, which cover every category of adversarial
fixture (rounding ties, luma ties, libm/V8 disagreement, borders, rectangles, all
angle-step classes). CI runs the synthetic suite.

## 5. Not verified

* Execution on Linux and Windows, and therefore cross-platform bitwise identity: CI is
  configured (`.github/workflows/ci.yml`, validated with `actionlint`) but has **not run**,
  because the repository has not been pushed.
* MSRV 1.85 for the crates other than `ferris-cdch` (only `ferris-cdch` was built with 1.85.0).
* The direction table under Node 24 (see MATHEMATICAL_COMPATIBILITY.md §5).
* No face images were available for any differential test.

## 6. CI status

`.github/workflows/ci.yml` defines format, clippy, documentation, tests on Linux / macOS /
Windows, MSRV, differential and mutation tests (synthetic fixtures, Node 22) on all three
operating systems, the independent-consumer check, the Python-binding build and a RustSec
audit. The syntax is validated with `actionlint`; the workflow has never run.
