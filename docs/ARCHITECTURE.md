# Architecture

Ferris is a Cargo workspace. One crate, `ferris-cdch`, is the public API for other
software; the remaining crates are the verified implementation, research bindings and
the verification tools.

```
                     ┌─────────────────────────────┐
  other Rust apps ──▶│ ferris-cdch (public API)    │◀── CDCH Research Studio (separate repo)
                     │ config · images · results · │
                     │ errors · centroid · CSV     │
                     └──────────┬──────────────────┘
                                │ (optional feature "representations")
          ┌─────────────────────┼──────────────────────────────┐
          ▼                     ▼                              ▼
   ┌──────────────┐     ┌────────────────┐              ┌──────────────┐
   │ cdch-core    │◀────│ cdch-features  │◀─────────────│ cdch-python  │ PyO3 → cdch_ferris
   │ reference    │     │ A / B / C / D  │              └──────────────┘
   │ plan, engine │     └────────────────┘              ┌──────────────┐
   │ js primitives│◀────────────────────────────────────│ cdch-cli     │ `cdch` executable
   └──────────────┘                                     └──────────────┘
          ▲                                             ┌──────────────┐
          └─────────────────────────────────────────────│ cdch-bench   │ ferris-diff, ferris-bench
                                                        └──────┬───────┘
                                                               │ stdin/stdout frames
                                                        ┌──────▼───────┐
                                                        │ oracle/*.js  │ unmodified ip-1.js (Node)
                                                        └──────────────┘
```

## Crates

| Crate | Role | Depends on |
|---|---|---|
| `ferris-cdch` | Stable API: `CdchConfig`, `CenterSpec`, `RgbImage` / `OwnedRgbImage` / `RgbaImage`, `AlphaPolicy`, `Analyzer`, `CdchResult`, `intensity_centroid`, `FerrisError`. Re-exports `cdch-core` as `core`. | `cdch-core`; optionally `cdch-features` |
| `cdch-core` | The algorithm. `reference`: a line-by-line transcription of `ip-1.js`. `plan`: `SamplingPlan` (image-independent ray geometry). `engine`: per-image evaluation (`matrix_into`, `histograms`, `histograms_and_matrix_into`, `checkpoints_into`) and the plan cache `Engine`. `js`: JavaScript-exact `round`, `luma`, direction table. | none |
| `cdch-features` | Representations A/B/C/D of the CDCH study, with NumPy-compatible arithmetic and fused extraction from 32-bin checkpoints. | `cdch-core` |
| `cdch-python` | NumPy bindings used by the research pipeline (`python/`). | `pyo3`, `numpy`, `rayon` |
| `cdch-cli` | `cdch` executable: features, matrices, linear-model scoring, PNG decoding. | `png`, `rayon`, `serde_json` |
| `cdch-bench` | Differential and mutation testing against Node, centroid oracle, microbenchmarks. | `ferris-cdch`, `serde_json`, `png`, `rayon` |

`ferris-cdch` has no third-party dependencies, so linking Ferris adds nothing beyond `std`.

## Data flow of one computation

1. The caller decodes an image and creates an `RgbImage` (borrowed, zero copy) or
   converts RGBA/grey with an explicit `AlphaPolicy`.
2. `CdchConfig::params(w, h)` resolves the centre and validates the step.
3. `Analyzer` fetches or builds the `SamplingPlan` for `(w, h, centre, step)`. Building
   evaluates every ray with the JavaScript-exact arithmetic and stores the distinct pixels,
   the per-sample indices (multiplicity preserved) and the ray offsets.
4. `histograms_and_matrix_into` computes the canonical luma of each distinct pixel once,
   counts per ray and divides the integer prefix sums by the ray length.
5. `CdchResult` owns the matrix, counts and ray lengths.

## Design decisions

The ADRs in [adr/](adr) record the decisions of the library extraction:

| ADR | Decision |
|---|---|
| [0001](adr/0001-standalone-library.md) | Ferris is a standalone library repository with a small public facade crate |
| [0004](adr/0004-versioning-and-pinning.md) | Versioning, Git tags and pinning for consumers |
| [0005](adr/0005-mathematical-equivalence.md) | Bitwise equivalence with `ip-1.js` is the contract |
| [0008](adr/0008-testing-and-reproducibility.md) | Layered testing with an executable oracle |

The older design table (V8 direction table, integer counts with one division, quotient
tables, unique-pixel luma, checkpoint output, NumPy-order arithmetic) is in
`RUST_ARCHITECTURE.md` §7 and still describes the implementation.

## Boundaries

* Ferris never decodes files, never touches the network and has no GUI or platform code.
* The image-difference analysis of the desktop application is **not** part of Ferris; it
  is a different analysis and lives in the application's `image-diff` crate.
* Research scripts (`python/`, `study/`) depend on Ferris, never the reverse.
