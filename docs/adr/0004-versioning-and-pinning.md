# ADR 0004: Versioning, tags and dependency pinning

**Status:** accepted (2026-11-18)

## Context
Ferris is not published to crates.io (`publish = false`), and its remote repository is
private. Consumers need reproducible builds and an unambiguous record of which algorithm
version produced a result.

## Problem
How should consumers refer to Ferris so that builds are reproducible, upgrades are
deliberate, and provenance can be recorded?

## Decision
- Semantic Versioning of `ferris-cdch`, with a stricter rule: any change of computed
  values is breaking.
- Releases are annotated Git tags `vX.Y.Z`.
- Consumers declare `{ git = "<remote URL>", tag = "vX.Y.Z" }` and commit `Cargo.lock`,
  which pins the exact commit. Applications should record that source string (URL, tag,
  commit) with their results; CDCH Research Studio does so in every export.
- Before a tag is published, consumers use Cargo source replacement to read the same tag
  from a local clone, leaving manifest and lockfile identical to the published state.

## Alternatives considered
- *crates.io:* public publication was not authorized, and is unnecessary for the use case.
- *Path dependencies:* hide the dependency, break clean checkouts, and record no version.
- *Branch dependencies:* not immutable.
- *Git submodules / vendoring:* duplicate the source in the consumer.

## Consequences
Upgrading is an explicit change of the tag plus a lockfile update. The repository is public
(since 2026-11-18, as `github.com/siriknikita/ferris-cdch`), so fetching needs no credentials.

## Risks and trade-offs
Tags are mutable in Git; the lockfile's commit hash, which Cargo verifies, is the real pin.
