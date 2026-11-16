# Project Ferris — Rust-native CDCH

A validated Rust implementation of Muzyka's canonical CDCH algorithm (`oracle/ip-1.js`,
SHA-256 `110732b48bc6e8b2d5167eec30f7277245913bed98b9baaa3c65c0c22aab44d8`), with:

- fused feature extraction;
- NumPy bindings;
- a differential and mutation test harness against the original JavaScript;
- a standalone CLI;
- integration into the archived CDCH research pipeline.

| Result (Apple M4 Pro, one core) | Value |
|---|---|
| Canonical matrices reproduced bit for bit | 713/713 differential cases; 640/640 archived evaluation images |
| Classifier AUROC across extraction paths | 840/840 identical; headline 0.5925 / 0.70125 reproduced |
| `cdch_A_logistic` end to end | 21.4 ms → **0.43 ms** (49.7×) |
| CDCH stage (pixels → A) | 20.8 ms → **55 µs** (380×) |
| Batch, A features | 17.7 k img/s per core; 171 k img/s on 14 threads |

## Reports

| Report | Contents |
|---|---|
| [RUST_PORT_VALIDATION.md](RUST_PORT_VALIDATION.md) | Exact semantics, differential testing, mutation testing, feature arithmetic |
| [RUST_CLASSIFIER_COMPATIBILITY.md](RUST_CLASSIFIER_COMPATIBILITY.md) | Features, scores, labels and AUROC vs canonical JS and the archive |
| [RUST_PERFORMANCE_REPORT.md](RUST_PERFORMANCE_REPORT.md) | Latency, throughput, memory, speedups, bottlenecks |
| [RUST_ARCHITECTURE.md](RUST_ARCHITECTURE.md) | Crates, data structures, APIs, design decisions |
| [PROJECT_STATE.md](PROJECT_STATE.md) | Completed work, answers to the study questions, limitations, next steps |

Machine-readable results are in `reports/` (`results_summary.json` collects the headline numbers).

## Layout

| Path | Contents |
|---|---|
| `crates/cdch-core` | Canonical CDCH: literal reference port + precomputed-sampling-plan engine |
| `crates/cdch-features` | Representations A (56), B (98), C (768), D (24×32); fused checkpoint path |
| `crates/cdch-python` | PyO3/maturin extension `cdch_ferris` |
| `crates/cdch-cli` | `cdch` executable: features, matrices, linear-model scoring (no Python/Node) |
| `crates/cdch-bench` | `ferris-diff` (differential + mutation tests), `ferris-bench` (microbenchmarks) |
| `oracle/` | Canonical `ip-1.js`, archived `oracle.js`, differential worker, V8 direction table |
| `python/` | uv project: backend-selectable pipeline, reproduction, benchmarks, tests |
| `models/` | Archived linear checkpoints exported to JSON for the CLI |
| `study/` | The original CDCH research archive, unchanged (`shasum -a 256 -c SHA256SUMS`) |
| `reports/` | Differential, classifier-compatibility and performance results |

## Build

Requirements: stable Rust ≥ 1.85, Python 3.12 via `uv`, and Node.js (only for the canonical
oracle and differential tests).

```bash
cargo build --release --workspace --exclude cdch-python
cargo test  --release --workspace --exclude cdch-python
cd python && uv sync      # archived Python pins + builds cdch_ferris with maturin
```

A standalone wheel: `uvx maturin build --release -m crates/cdch-python/Cargo.toml -o dist`.

## Use

```python
import numpy as np, cdch_ferris as cf
from PIL import Image
rgb = np.asarray(Image.open("crop.png").convert("RGB"))        # (H, W, 3) uint8
cf.cdch(rgb)                      # (360, 256) canonical matrix, centre (W//2, H//2)
cf.features(rgb, "A")             # (56,)  fused, no matrix formed
cf.features_multi(rgb, ["A", "B", "C"])
cf.batch_features(stack, "A", threads=None)   # (n, 56), GIL released
```

```bash
target/release/cdch features --kind A crop.png
target/release/cdch score --model models/cdch_A_logistic.json --timing images/*.png
```

In the research pipeline:
`ferris_pipeline.load_deploy("cdch_A_logistic", backend="rust")`, where `backend` is one of
`canonical_js | rust_matrix | rust`.

## Verify

```bash
cargo run --release -p cdch-bench --bin ferris-diff -- --oracle-dir oracle --out reports/differential --mutants
cd python && FERRIS_PAIRS=<pairs dir> .venv/bin/python -m pytest -q tests
```

External inputs (the DIV2K validation HR archive, SwinIR/Real-ESRGAN weights, regenerated
pairs) are not stored here. See PROJECT_STATE.md → Reproduction, and
`study/pilot/config/assets.json` for the pinned sources.
