# Project state — Project Ferris (Rust-native CDCH)

**16 November 2026 · Phases A–J complete on macOS arm64**

The original study's own state is preserved unchanged in `study/PROJECT_STATE.md`.

## Completed work

| Phase | Result | Evidence |
|---|---|---|
| A. Audit | Archive checksums verified. `ip-1.js` recovered, digest `110732b4…aab44d8` matches. Pipeline mapped: Node worker → archived oracle → base64 float64 over a pipe → NumPy representations → sklearn/torch. Exact JavaScript semantics documented, including the radius limit = canvas width, `[w//2, h//2]` centre, `Math.round` ties, and NaN for empty rays. | RUST_PORT_VALIDATION.md §1–2 |
| B. Reference port | `cdch-core::reference`, a literal safe-Rust transcription. V8 direction bit patterns embedded. | `crates/cdch-core` |
| C. Numerical verification | `ferris-diff`: 713 cases, 18.0 M samples, 64.8 M matrix values. Bitwise agreement in coordinates, multiplicities, counts and CDFs. All 15 mutants detected. | `reports/differential/` |
| D. Feature port | A (56), B (98), C (768), D (24×32), following NumPy's evaluation order. | `crates/cdch-features` |
| E. Fused extraction | A/C/D from a 360×32 checkpoint CDF (no 256-bin matrix), B from an internal buffer. Fused output is bitwise equal to the matrix path. Full-matrix and base64 transport eliminated. | RUST_ARCHITECTURE.md §4 |
| F. Python integration | PyO3/maturin package `cdch_ferris`. `FerrisPipeline` with `canonical_js` / `rust_matrix` / `rust` backends on unchanged checkpoints. 23 binding tests plus a pipeline-equivalence test. | `crates/cdch-python`, `python/` |
| G. Reproducibility | 640/640 archived evaluation matrices bitwise equal. 840/840 AUROC values identical across extraction paths. Headline AUROCs reproduced exactly. Zero holdout label changes. | RUST_CLASSIFIER_COMPATIBILITY.md |
| H. Optimization | Exact optimizations only (table luma, quotient tables, split counters, plan-level validity, zero-PMF shortcut). Fused A 74 → 56 µs; B rich statistics 1.39 → 1.12 ms. | RUST_PERFORMANCE_REPORT.md §6 |
| I. Benchmarking | End-to-end (archived protocol), components, microbenchmarks, batch, CLI, memory, cold start. | `reports/perf/` |
| J. Reporting and extensions | Five reports, a self-contained `cdch` CLI with exported linear models, a wheel, the project archive. | this directory |

## Answers to the study questions

1. **Does Rust reproduce the canonical CDCH output?** Yes, bit for bit on every tested input
   (713 cases, including adversarial rounding, luma ties, empty rays, rectangular and full-size
   natural images). This holds because the V8 trigonometric values are embedded; with host libm
   it would not (8 adversarial fixtures).
2. **How much latency comes from the CDCH algorithm itself?** Under the canonical design,
   10.9 ms of the 20.8 ms CDCH stage (53%) on this machine is `ip-1.js` execution. The same
   algorithm in Rust takes 57 µs for the full matrix and 28 µs for what A/C/D need.
3. **How much came from JavaScript/Python interoperability?** 9.5 ms (46%): JSON packing of
   pixels, Node parsing, Float64→base64 encoding, pipe transfer, and base64 decoding. The PyO3
   crossing that replaces it costs about 1 µs.
4. **How much does fused extraction save?** Within Rust: 1.95× for A (109.5 → 56.3 µs) and 2.2×
   for C, but almost nothing for B (the matrix is needed). In the Python pipeline, fused A saves
   0.195 ms per image over Rust-matrix + NumPy features (0.625 → 0.430 ms).
5. **Can Rust CDCH achieve competitive throughput?** Yes. Single image: 2,292 img/s end to end in
   Python (A logistic). Batch: 17.7 k img/s per core for features, 171 k img/s on 14 threads. The
   Rust-only CLI including PNG decoding: 5.3 k img/s on one thread, about 42 k img/s on 14.
6. **Does CDCH remain less efficient than FFT-based classification?** Not on this hardware, for
   A/C/D. The Rust CDCH-A pipeline (0.43 ms) is faster than the NumPy frequency pipeline
   (1.01 ms). B (1.59 ms) remains slower. The frequency extractor itself was not ported. CDCH's
   accuracy remains well below frequency (AUROC 0.59–0.70 vs 0.86), and that is unchanged.
7. **Do accuracy and AUROC remain unchanged?** Yes. All 840 AUROC values are identical across
   JS, Rust-matrix, Rust-fused and Rust-from-JS-matrix extraction. Holdout scores change by at
   most 1.4·10⁻¹⁴ and no holdout label changes. No accuracy gain is claimed or expected.
8. **Is Rust CDCH suitable for real-time or batch forensics?** Computationally yes: about 56 µs
   per 128² image and roughly 270 µs per 1024² image, about 3 MiB resident for the CLI, no
   runtime dependencies, and linear multi-core batch scaling. Scientifically, the pilot's CDCH
   classifiers are weak (holdout AUROC ≤ 0.70) and fail the interpolation-specificity check
   (study/PROJECT_STATE.md). Speed removes a practical barrier, not a validity one.

## Limitations

- **One platform.** All measurements and tests ran on an Apple M4 Pro (macOS 27.2.1). Linux and
  Windows builds are expected to work (pure Rust) but were not run. The archived Linux/Xeon
  timings are not comparable.
- **Node version.** The archived matrices were made with Node v24.22.1. The embedded table and
  all oracle runs here use Node v22.23.4. Bitwise reproduction of archived C features (192,000
  values) is consistent with identical directions for the archived protocol, but the full table
  was not diffed against Node 24 (installing it was not permitted in this session).
- **Regenerated images.** The archive ships no images. Originals and bicubic crops are
  pixel-identical to the archive. Learned-SR outputs regenerated on arm64 PyTorch partly differ,
  so comparisons with archived values use only archive-equivalent images (250/400 PNG).
  Rust-vs-JS comparisons always use identical pixels.
- **Features are not bitwise equal to NumPy everywhere.** A's FFT harmonics and B's
  BLAS-dependent moments differ at ≤ 8.1·10⁻¹³ relative, the same order as NumPy's own
  cross-platform variation.
- **Calibration-set ties.** Validation-split images can flip at thresholds and score-fusion ECDFs
  under any rounding-level change, because the frozen rules are built from those images' own
  archived scores. Holdout results are unaffected.
- **No faces.** No face images were available for the differential tests.
- **Not measured:** energy per image, and allocation counts (behaviour stated from code
  inspection).
- **CLI models.** The CLI scores only the exported linear models. Other families still need
  Python.

## Reproduction (from `ferris/`)

```bash
# Rust: build, unit/property tests, differential + mutation suite (needs Node)
cargo test --release --workspace --exclude cdch-python
cargo run --release -p cdch-bench --bin ferris-diff -- --oracle-dir oracle --out reports/differential --mutants \
    --div2k ../external/div2k --png-dir pairs=../external/pairs

# Python environment (archived pins + the Rust extension)
cd python && uv sync
# Regenerate pilot pairs from pinned assets (the archived script; ~500 MB download)
PYTHONPATH=../study/pilot .venv/bin/python ../study/pilot/build_pairs.py --assets ../../external/assets --fetch --out ../../external/pairs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
FERRIS_PAIRS=../../external/pairs .venv/bin/python -m pytest -q tests
.venv/bin/python reproduce_classifiers.py --pairs ../../external/pairs --out ../reports/classifier_compatibility
.venv/bin/python benchmark_pipelines.py   --pairs ../../external/pairs --out ../reports/perf/pipelines
.venv/bin/python benchmark_components.py  --pairs ../../external/pairs --out ../reports/perf/components.json
.venv/bin/python export_linear_models.py  --out ../models
.venv/bin/python summarize_results.py
cd .. && cargo run --release -p cdch-bench --bin ferris-bench -- --images <dir of 128² PNGs> --out reports/perf/rust_microbench.json
scripts/bench_cli.sh ../external/pairs > reports/perf/cli_throughput.csv
```

## Next experiments

1. Diff `oracle/direction_table_v8.json` under Node 24 (the archived runtime) and on Linux x86-64.
   Run the differential suite on Linux to confirm platform-independent matrices.
2. Port the frequency extractor to Rust, giving a fully Rust hybrid. Then decide whether CDCH
   adds value at equal cost (archived +0.004 AUROC, CI −0.029…+0.035).
3. Use the fast extractor to explore CDCH at scale: multiple centres, angle steps, and
   per-channel histograms (the latter needs a separately defined, validated variant; the
   canonical CDCH is luma only). Register these as new methods, not as changes to CDCH.
4. Repeat the classifier study on an independent face cohort, now that feature extraction costs
   microseconds.
5. Optional engineering, only if decoding stops dominating: SIMD luma, DFT folding (k ≥ 8 have
   period 45), Rust inference for the non-linear model families.
