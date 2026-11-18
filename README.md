# Ferris — canonical CDCH in Rust

Ferris is a reusable Rust library that computes the **CDCH** (cumulative directional
colour histogram) exactly as Muzyka's original JavaScript does (`oracle/ip-1.js`,
SHA-256 `110732b48bc6e8b2d5167eec30f7277245913bed98b9baaa3c65c0c22aab44d8`). Bit for bit.

It exists so that CDCH research is not limited by the speed of the original browser
implementation: large statistical experiments, parameter sweeps, image comparisons and
machine-learning studies become practical when one CDCH costs about 60 µs instead of
about 20 ms. Speed is the only thing Ferris changes. It does not make CDCH a better
descriptor, and no accuracy claim follows from it.

| | |
|---|---|
| Public crate | [`ferris-cdch`](crates/ferris-cdch) (version 0.2.0) |
| Exactness | 713/713 differential cases bitwise equal to `ip-1.js` (18.0 M samples, 64.8 M values); 15/15 deliberately wrong variants detected; intensity centroid equal in 5,680/5,680 comparisons |
| Speed (Apple M4 Pro, one core) | 128 × 128 image, 360 rays: 62 µs for the matrix, 66 µs with integer counts |
| Dependencies | none beyond `std` (optional `representations` feature adds the A/B/C/D feature extractors) |
| Licence | MIT OR Apache-2.0 |

## Use it from another Rust project

Ferris is not on crates.io. Depend on it through Git, pinned to a release tag:

```toml
[dependencies]
ferris-cdch = { git = "https://github.com/siriknikita/ferris-cdch.git", tag = "v0.2.0" }
```

`Cargo.lock` then records the exact commit of the tag. For working against an unreleased
local tag, see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#consuming-an-unpublished-tag).

```rust
use ferris_cdch::{Analyzer, CdchConfig, CenterSpec, OwnedRgbImage};

// Decode the image with any library (Ferris has no codecs) into packed 8-bit RGB.
let image = OwnedRgbImage::new(width, height, rgb_bytes)?;

let analyzer = Analyzer::new();                       // reuse: caches ray geometry
let config = CdchConfig::default()                     // archived protocol:
    .with_center(CenterSpec::Archived)                 //   centre (w // 2, h // 2)
    .with_angle_step(1);                               //   360 rays
let result = analyzer.compute(&image.view(), &config)?;

let cdf: &[f64] = result.cdf(90);                      // ray at 90°, 256 values
let counts: &[u32] = result.counts(90);                // its integer histogram
result.write_matrix_csv(std::fs::File::create("cdch.csv")?)?;  // lossless CSV
```

The complete guide is [docs/API.md](docs/API.md). Runnable examples are
`cargo run -p ferris-cdch --example basic` and `--example rgba`, plus the separate
project in [`examples/independent-consumer`](examples/independent-consumer).

## Documentation

| Document | Contents |
|---|---|
| [docs/API.md](docs/API.md) | Public API, errors, threading, versioning |
| [docs/MATHEMATICAL_COMPATIBILITY.md](docs/MATHEMATICAL_COMPATIBILITY.md) | Exactly what is computed, and every compatibility assumption |
| [docs/VALIDATION.md](docs/VALIDATION.md) | How equivalence is verified, and the current evidence |
| [docs/BENCHMARKING.md](docs/BENCHMARKING.md) | Methodology and measured performance |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Crates, data flow, design decisions |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Building, testing, releasing |
| [docs/adr/](docs/adr) | Architectural decision records |
| [docs/ENGINEERING_INVENTORY.md](docs/ENGINEERING_INVENTORY.md) | State of the project before the library extraction |
| [docs/IMPLEMENTATION_REPORT.md](docs/IMPLEMENTATION_REPORT.md) | What was done in v0.2.0 and the evidence for it |
| [CHANGELOG.md](CHANGELOG.md), [CONTRIBUTING.md](CONTRIBUTING.md) | History, contribution rules |
| `RUST_*.md`, `PROJECT_STATE.md` | Historical reports of the v0.1.0 port (phases A–J) |

## Repository layout

| Path | Contents |
|---|---|
| `crates/ferris-cdch` | **The public library API** |
| `crates/cdch-core` | Canonical CDCH: literal reference port and optimized sampling-plan engine |
| `crates/cdch-features` | Statistical representations A (56), B (98), C (768), D (24 × 32) |
| `crates/cdch-python` | PyO3 / NumPy bindings (`cdch_ferris`) |
| `crates/cdch-cli` | `cdch` command-line tool |
| `crates/cdch-bench` | `ferris-diff` (differential and mutation tests), `ferris-bench` |
| `oracle/` | Canonical `ip-1.js`, archived oracle and Node workers |
| `examples/independent-consumer` | A separate Cargo project that uses Ferris only through Git |
| `python/`, `models/`, `study/` | Research pipeline, exported models, the original study archive (unchanged) |
| `reports/` | Machine-readable evidence |

## Related project

[CDCH Research Studio](../cdch-research-studio) is a native desktop application that uses
Ferris as an external dependency. Ferris does not depend on it.
