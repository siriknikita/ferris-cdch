# Ferris API guide (`ferris-cdch` 0.2.0)

Generate the reference documentation with `cargo doc -p ferris-cdch --all-features --open`.
Every code block below is taken from compiled examples, doctests or tests.

## 1. Adding the dependency

```toml
[dependencies]
ferris-cdch = { git = "https://github.com/siriknikita/ferris-cdch.git", tag = "v0.2.0" }
# Optional: the validated A/B/C/D feature extractors of the CDCH study.
# ferris-cdch = { git = "…", tag = "v0.2.0", features = ["representations"] }
```

`Cargo.lock` records the exact commit of the tag, so builds are reproducible. Ferris has
no runtime dependencies apart from `std` and needs Rust 1.85 or newer.

## 2. Images

CDCH reads 8-bit **R, G, B** values of a row-major image. Ferris never decodes files;
use `image`, `png`, Pillow or anything else and pass the pixels.

| Type | Use |
|---|---|
| `RgbImage<'a>` | Borrowed packed or strided RGB (zero copy). `RgbImage::new(w, h, &bytes)`, `RgbImage::with_stride(w, h, stride, &bytes)`. |
| `rgb_view(w, h, &bytes)` | Same as `RgbImage::new`, but also rejects empty images. |
| `OwnedRgbImage` | Owned packed RGB. `new(w, h, vec)`, `from_gray(w, h, &grey)`, `from_rgba(&rgba, policy)`, `.view()`. |
| `RgbaImage<'a>` | Borrowed straight-alpha RGBA (packed or strided). `is_opaque()`. |
| `AlphaPolicy` | `Ignore` (default: keep stored RGB, like Pillow's `convert("RGB")` used by the archived pipeline), `CompositeOver { background }`, `OVER_WHITE`, `OVER_BLACK`. |

Compositing is `round((c·a + bg·(255 − a)) / 255)`, computed exactly in integers. Because
the canonical algorithm has no alpha semantics, the alpha policy is part of the
experimental protocol. Record it with your results.

```rust
use ferris_cdch::{compute, AlphaPolicy, CdchConfig, OwnedRgbImage, RgbaImage};
let src = RgbaImage::new(w, h, &rgba)?;
let rgb = OwnedRgbImage::from_rgba(&src, AlphaPolicy::OVER_WHITE);
let result = compute(&rgb.view(), &CdchConfig::default().with_angle_step(90))?;
```

## 3. Configuration

```rust
pub enum CenterSpec { Archived, Midpoint, Explicit { x: f64, y: f64 } }
pub struct CdchConfig { pub center: CenterSpec, pub angle_step: u32 }
```

| Item | Meaning |
|---|---|
| `CenterSpec::Archived` (default) | `(w // 2, h // 2)`, the archived research protocol |
| `CenterSpec::Midpoint` | `(w / 2, h / 2)`, the oracle default and the "image centre" button of `ip-1.js` |
| `CenterSpec::Explicit { x, y }` | Any finite point, in pixel coordinates (x right, y down, pixel centres at integers) |
| `angle_step` | Whole degrees, 1..=360. Rays at `0, step, … < 360`, so `ceil(360 / step)` rays. 0° points up, clockwise on screen. |

`CdchConfig::default()` is the archived protocol (archived centre, 1°, 360 rays).
`config.params(w, h)` resolves the concrete centre, and `config.n_rays()` gives the row count.

## 4. Computing

```rust
let analyzer = Analyzer::new();                // Send + Sync; clone to share the cache
let result = analyzer.compute(&img, &config)?; // matrix + counts + ray lengths
let matrix = analyzer.matrix(&img, &config)?;  // matrix only (fastest)
let once   = ferris_cdch::compute(&img, &config)?; // fresh analyzer, no reuse
```

The geometry of the rays depends only on `(width, height, centre, step)`. `Analyzer`
caches one sampling plan per geometry, so a batch of equally sized images pays the
geometry cost (about 0.43 ms for 128²) once. Results are deterministic: the same input
gives the same bits on any thread and in any order.

## 5. Results

`CdchResult` exposes:

| Method | Value |
|---|---|
| `n_rays()`, `angle(i)` | number of rays, angle of ray `i` in degrees |
| `cdf(i)` | 256 canonical values `F[b] = (counts[0] + … + counts[b]) / ray_length` |
| `counts(i)`, `counts_flat()` | integer histograms of canonical luma |
| `ray_length(i)`, `ray_lengths()`, `total_samples()` | samples per ray (with multiplicity) |
| `matrix()`, `into_matrix()` | the `rays × 256` matrix (`CdchMatrix`, row-major `f64`) |
| `is_complete()`, `empty_rays()` | whether any ray is empty (rows of `NaN`) |
| `params()`, `width()`, `height()` | the resolved parameters and image size |
| `write_matrix_csv(w)`, `write_counts_csv(w)` | lossless CSV (shortest round-trip decimal; `NaN` as `NaN`) |

A ray is empty exactly when the **rounded centre** lies outside the image. Every ray
starts at the centre, so in that case all rays are empty. `ip-1.js` then produces
`0/0 = NaN`, and Ferris does the same.

## 6. Other functions

* `intensity_centroid(&img, min, max) -> Result<Option<Centroid>>`: the "find centre"
  feature of `ip-1.js` (`findIntensityCenter`). It returns the centroid of the pixels
  whose canonical luma lies in `[min, max]`, rounded with JavaScript `Math.round`.
  `None` means that no pixel qualifies.
* `luma(r, g, b)`, `luma_plane(&img)`: canonical luma
  `Math.round(0.2125 R + 0.7154 G + 0.0721 B)`.
* `build_info()`, `VERSION`, `CANONICAL_SHA256`: provenance.
* `ferris_cdch::core`: the full `cdch-core` API (reference port, sampling plans,
  checkpoint output, JavaScript primitives). It is less stable; see §8.
* `ferris_cdch::representations` (feature `representations`): `FeatureExtractor` and the
  representations A/B/C/D of `study/CDCH_REPRESENTATIONS.md`.

## 7. Errors

```rust
pub enum FerrisError { InvalidInput(InvalidInput), Internal(String) }
pub enum InvalidInput { EmptyImage{..}, BufferSize{..}, RowStride{..}, TooLarge{..},
                        NonFiniteCenter{..}, AngleStep(u32), IntensityRange{..} }
```

Invalid input never panics; it returns `FerrisError::InvalidInput` (`is_invalid_input()`).
`Internal` means a broken invariant inside Ferris, which is a bug. Images are limited to
`width · height ≤ u32::MAX` and sides `≤ u32::MAX / 4`. The size check happens before any
buffer is read, so oversized declarations are rejected without allocation.

## 8. Stability and versioning

Ferris follows Semantic Versioning for the **top-level items of `ferris-cdch`**:

* Patch releases (0.2.x) fix bugs and never change numerical output.
* Minor releases before 1.0 (0.3.0) may change the API. They are listed in `CHANGELOG.md`.
* **Any change to the numbers Ferris produces for the same input and configuration is a
  breaking change.** It requires a new minor version (pre-1.0) or major version, and a
  new differential report.
* `ferris_cdch::core` and `ferris_cdch::representations` re-export internal crates; their
  API may change in minor releases.

Releases are annotated Git tags `vX.Y.Z` on this repository.

## 9. Thread safety and performance notes

* `Analyzer`, `CdchResult`, `OwnedRgbImage` and the configuration types are `Send + Sync`.
* Each `compute` call allocates its result (about 360 · 256 · 12 bytes) plus a small
  workspace. For maximum throughput in tight loops, use `ferris_cdch::core` directly
  (`SamplingPlan::matrix_into` with a reused `Workspace` and output buffer).
* Ferris is single-threaded per call. Parallelize across images (for example with rayon) by
  sharing one `Analyzer`.
