# Contributing to Ferris

Ferris is research software whose central promise is **exact** agreement with the
canonical JavaScript CDCH. Contributions are welcome if they keep that promise.

## Ground rules

1. **Never change computed values silently.** If a change alters any output bit for any
   input, it is a new, documented algorithm version (docs/API.md §8), not a fix.
2. **Run the oracle.** Changes to `cdch-core` or `ferris-cdch` must pass `ferris-diff
   --mutants --centroid` with zero mismatches, and each mutant must still be detected.
3. **Do not weaken tests** to make them pass. If a test is wrong, explain why in the commit.
4. **Do not modify** `oracle/ip-1.js`, `oracle/oracle.js`, `study/` or historical reports.
   New evidence goes into new report directories.
5. **No new runtime dependencies** in `cdch-core` or `ferris-cdch` without an ADR.
6. Keep `unsafe` out (`#![forbid(unsafe_code)]`).

## Checklist for a pull request

```bash
cargo fmt --all -- --check
cargo clippy --workspace --exclude cdch-python --all-targets --all-features -- -D warnings
cargo test --release --workspace --exclude cdch-python --all-features
cargo run --release -p cdch-bench --bin ferris-diff -- --oracle-dir oracle --out target/diff --mutants --centroid
scripts/verify_independent_consumer.sh
```

Add a `CHANGELOG.md` entry. For performance changes, attach `ferris-bench` results for the
old and new revision on the same machine (docs/BENCHMARKING.md).

## Data

Do not commit images of unknown licence, personal data or large generated files.
Fixtures must be synthetic (see `crates/cdch-bench/src/synth.rs`) or have a recorded licence.
