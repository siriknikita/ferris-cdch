# Development

## Requirements

* Rust (stable; minimum 1.85 for `ferris-cdch`). Tested with 1.85.0, 1.91.1 and 1.100.0.
* Node.js 22 for the differential tests (`node --version` → v22.x; the direction table was
  recorded with Node 22).
* Optional: Python 3.12 via `uv` for the bindings and the research pipeline.

## Everyday commands

```bash
cargo fmt --all
cargo clippy --workspace --exclude cdch-python --all-targets --all-features -- -D warnings
cargo test --release --workspace --exclude cdch-python --all-features   # release: exhaustive luma tests
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --exclude cdch-python --no-deps --all-features
cargo run --release -p cdch-bench --bin ferris-diff -- --oracle-dir oracle --out target/diff --mutants --centroid
scripts/verify_independent_consumer.sh
```

`cdch-python` needs a Python interpreter for PyO3. To check it, point PyO3 at the project
environment: `PYO3_PYTHON=$PWD/python/.venv/bin/python cargo clippy -p cdch-python`.

## Python bindings and the research pipeline

```bash
cd python && uv sync                       # archived pins + builds cdch_ferris with maturin
.venv/bin/python -c "import cdch_ferris, numpy as np; print(cdch_ferris.build_info())"
```

The pipeline, reproduction and benchmark commands are listed in `PROJECT_STATE.md` →
Reproduction.

## Changing the algorithm code

`cdch-core` reproduces `ip-1.js` exactly. Any change there must:

1. keep `cargo test --release -p cdch-core` green (engine vs reference, exhaustive luma);
2. keep `ferris-diff --mutants` at 0 mismatches with every mutant still detected, preferably
   on the full natural-image suite (see VALIDATION.md §4);
3. be benchmarked against the previous revision (BENCHMARKING.md §5);
4. if output values change for any input, be released as a breaking version (API.md §8).

Do not "simplify" arithmetic that looks redundant (for example, replacing the division by a
reciprocal multiply, or the luma formula by an integer formula); the mutation suite exists
because such changes are observable.

## Releasing

1. Update `CHANGELOG.md` and the workspace version in `Cargo.toml`.
2. Run the full check list above and the full differential suite; store the reports under
   `reports/`.
3. Commit, then tag: `git tag -a vX.Y.Z -m "Ferris X.Y.Z"`.
4. Publishing (pushing the tag to `origin`) is a separate, explicit decision; see below.

## Consuming an unpublished tag

Consumers declare `git = "https://github.com/siriknikita/project-ferris-rust-cdch.git", tag = "vX.Y.Z"`.
Until the tag is pushed, Cargo cannot fetch it. CDCH Research Studio solves this without
changing its manifest: its `scripts/use-local-ferris.sh` writes an untracked
`.cargo/config.toml` with a Cargo *source replacement* that reads the same tag from a
local clone, and `scripts/update-ferris-lock.sh` pins the commit in `Cargo.lock`. Once
the tag is on the remote, deleting the generated config is the only change needed.

To check that a commit can be consumed as a Git dependency at all, run
`scripts/verify_independent_consumer.sh [REPO] [REV]`.
