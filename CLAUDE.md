# Ferris — notes for Claude Code

Rust implementation of Muzyka's CDCH that must stay **bit-for-bit identical** to the
canonical JavaScript `oracle/ip-1.js`. Public API: `crates/ferris-cdch`. Consumer
application (separate repo, sibling directory): `../cdch-research-studio`.

## Invariants (do not break)
- Never change computed values silently. Any output change for any input = breaking
  version + new differential report (docs/API.md §8).
- Do not modify `oracle/ip-1.js`, `oracle/oracle.js`, `study/`, or existing `reports/*`;
  write new evidence into new report directories.
- `cdch-core` and `ferris-cdch`: no new dependencies, no `unsafe`.
- Do not "simplify" JS-exact arithmetic (division vs reciprocal, luma formula, rounding):
  `ferris-diff --mutants` exists because those changes are observable.
- Fixtures must be synthetic; licensed images (DIV2K, pilot pairs) live in `../external/`.

## Commands
```bash
cargo fmt --all -- --check
cargo clippy --workspace --exclude cdch-python --all-targets --all-features -- -D warnings
cargo test --release --workspace --exclude cdch-python --all-features
cargo run --release -p cdch-bench --bin ferris-diff -- --oracle-dir oracle --out target/diff --mutants --centroid   # needs Node 22
scripts/verify_independent_consumer.sh [REPO] [REV]
PYO3_PYTHON=$PWD/python/.venv/bin/python cargo clippy -p cdch-python --all-targets -- -D warnings
```
Full suite with natural images: add `--div2k ../external/div2k --png-dir pairs=../external/pairs`.

## Releasing
Bump workspace version + CHANGELOG, run everything above, annotated tag `vX.Y.Z`, push
`main` and the tag. Then in the Studio: change the tag in its Cargo.toml and run
`cargo update -p ferris-cdch`.

## Where things are documented
docs/API.md, docs/MATHEMATICAL_COMPATIBILITY.md, docs/VALIDATION.md (evidence),
docs/BENCHMARKING.md, docs/IMPLEMENTATION_REPORT.md, docs/adr/. Historical v0.1.0 reports:
RUST_*.md, PROJECT_STATE.md.
