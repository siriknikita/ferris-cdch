# ADR 0001: Ferris as a standalone library with a public facade crate

**Status:** accepted (2026-11-18)

## Context
Ferris began as a research port: a workspace whose crates served the CDCH study's
Python pipeline, CLI and verification harness. A desktop application (CDCH Research
Studio) and future research projects need the same algorithm.

## Problem
Without a stable, documented library boundary, each consumer would either copy code or
depend on internal crates whose APIs were shaped by benchmarks and bindings. Copies drift,
and drift in a scientific algorithm silently invalidates comparisons between studies.

## Decision
Keep Ferris as its own Git repository, and add one public crate, `ferris-cdch`, as the
stable API. It is a thin facade over the verified `cdch-core`, with typed configuration,
validated buffers, structured results and typed errors. Applications (including the
Studio) live in separate repositories and depend on Ferris; Ferris depends on none of
them, and on nothing beyond `std`.

## Alternatives considered
- *Monorepo with the application:* simpler local development, but it couples library
  releases to GUI changes and invites GUI-driven changes to the algorithm.
- *Expose `cdch-core` directly:* no new crate, but its API mixes test-oracle, benchmark
  and performance-tuning entry points, and it has no RGBA, centroid or result types.
- *Rewrite the algorithm behind a new API:* rejected; the existing code is the verified
  artefact.

## Rationale
One canonical implementation, one versioned interface. The facade can evolve
ergonomically while the numerical core stays frozen and verified.

## Consequences
- Consumers pin a Git tag; Cargo.lock pins the commit.
- The facade is covered by its own tests and by the differential suite.
- Two layers must be kept in sync (facade docs vs core behaviour).

## Risks and trade-offs
A thin facade adds about 6 µs per call (allocation, cache lookup). Consumers needing every
microsecond can use `ferris_cdch::core`, which is less stable.
