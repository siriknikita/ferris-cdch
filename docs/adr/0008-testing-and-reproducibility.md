# ADR 0008: Layered testing with an executable oracle

**Status:** accepted (2026-11-18)

## Context
Correctness claims must be backed by tests that would fail if the implementation were
wrong, and must be reproducible by others.

## Decision
- Keep the **unmodified, checksum-verified** `ip-1.js` as an executable oracle, run by
  Node through small worker scripts; compare outputs bitwise.
- Measure the sensitivity of that comparison with **mutation testing**: 15 plausible wrong
  implementations must each be caught.
- Keep a **literal reference port** in Rust, so the optimized engine and the public API
  can be tested exhaustively without Node.
- Test the public API from the outside (integration tests and an **independent consumer
  project** built from a fresh Git clone).
- Keep **performance measurement separate** from correctness; never assert timings in CI.
- Record environment, versions and raw results of every evidence run under `reports/`.
- Use only synthetic fixtures in the repository; licensed images stay outside.

## Alternatives considered
Golden files only (cannot detect issues outside the stored cases and are not
self-explaining); tolerance-based comparisons (see ADR 0005).

## Consequences
The full suite needs Node 22, and the natural-image cases need external data; CI runs the
synthetic suite.
