# Project Ferris — Rust-native CDCH

A validated Rust implementation of Muzyka's canonical CDCH algorithm (`oracle/ip-1.js`,
SHA-256 `110732b48bc6e8b2d5167eec30f7277245913bed98b9baaa3c65c0c22aab44d8`), with fused
feature extraction, NumPy bindings, a differential test harness against the original
JavaScript, and integration into the archived CDCH research pipeline.

> Status: work in progress. Phases A–G (audit, reference port, differential verification,
> feature port, fused extraction, Python integration, classifier reproduction) are done;
> optimization, benchmarking and the final reports are in progress.

## Layout

| Path | Contents |
|---|---|
| `crates/cdch-core` | Canonical CDCH: literal reference port + precomputed-sampling-plan engine |
| `crates/cdch-features` | Representations A (56), B (98), C (768), D (24×32); fused checkpoint path |
| `crates/cdch-python` | PyO3/maturin extension `cdch_ferris` |
| `crates/cdch-bench` | `ferris-diff` (differential test vs JS oracle), `ferris-bench` (microbenchmarks) |
| `crates/cdch-cli` | Standalone CLI (in progress) |
| `oracle/` | Canonical `ip-1.js`, archived `oracle.js`, differential worker, V8 direction table |
| `python/` | uv project: backend-selectable pipeline, reproduction and benchmark scripts |
| `study/` | The original CDCH research archive, unchanged (verify with `shasum -a 256 -c SHA256SUMS`) |
| `reports/` | Machine-readable differential and classifier-compatibility results |

## Quick start

```bash
cargo test --release --workspace --exclude cdch-python
cd python && uv sync            # builds the Rust extension via maturin
.venv/bin/python -c "import cdch_ferris, numpy as np; print(cdch_ferris.features(np.zeros((128,128,3),np.uint8), 'A')[:4])"
```

The differential harness needs Node.js (`node oracle/diff_worker.js` is spawned automatically):

```bash
cargo run --release -p cdch-bench --bin ferris-diff -- --oracle-dir oracle --out reports/differential
```

External inputs (DIV2K validation HR, SwinIR/Real-ESRGAN weights, regenerated pairs) are
not stored here; see `study/README.md` and `study/pilot/config/assets.json`.
