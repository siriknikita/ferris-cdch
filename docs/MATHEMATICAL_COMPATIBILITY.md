# Mathematical compatibility

The canonical definition of CDCH is the JavaScript function `drawRaysAndGenerateHistograms`
and its helper `calculateCumulativeHistogram` in `oracle/ip-1.js`
(SHA-256 `110732b48bc6e8b2d5167eec30f7277245913bed98b9baaa3c65c0c22aab44d8`). Ferris
reproduces the values that function produces, **bit for bit**, for every input the
JavaScript accepts on an opaque canvas. This document states exactly what that means and
which assumptions it rests on. Evidence is in [VALIDATION.md](VALIDATION.md); the detailed
history is in `RUST_PORT_VALIDATION.md`.

## 1. Definition

For an image of width `W`, height `H`, centre `(x, y)` and integer angle step `s`:

1. **Rays.** For `angle = 0, s, 2s, … < 360`:
   `radians = (angle − 90) · (π / 180)`, direction `(cos radians, sin radians)` with the
   **V8 results** of `Math.cos` and `Math.sin`. 0° points up (−y); angles increase
   clockwise on screen. Number of rays: `⌈360 / s⌉`.
2. **Samples.** For `r = 0, 1, 2, … while r < W` (**the image width, also for tall
   images**): `px = Math.round(x + r·cos)`, `py = Math.round(y + r·sin)`, evaluated in
   binary64 with the products and sums rounded separately (no fused multiply–add). The ray
   stops at the first sample with `px < 0 ∨ px ≥ W ∨ py < 0 ∨ py ≥ H`. `−0` passes `px < 0`.
   Coordinates repeated by rounding are counted again (multiplicity is kept).
3. **Intensity.** `Math.round(R·0.2125 + G·0.7154 + B·0.0721)`, evaluated left to right
   in binary64. The result always lies in 0..=255.
4. **Histogram.** `counts[b]` is the number of samples of the ray with intensity `b`.
5. **Normalization.** `cumulative[b] = counts[0] + … + counts[b]` (exact integers),
   `F[b] = cumulative[b] / cumulative[255]`, with one IEEE binary64 division per bin.
   An empty ray gives `0/0 = NaN` in all 256 bins.

`Math.round(v)` is ECMAScript rounding: the nearest integer, with ties toward +∞ and
`−0` preserved for `v ∈ [−0.5, −0]`.

## 2. What is preserved, and how

| Aspect | Canonical behaviour | Ferris |
|---|---|---|
| Channel order | R, G, B as returned by `getImageData`; alpha unused | `RgbImage::rgb(x, y)` returns `[R, G, B]`; alpha is never read by the algorithm |
| Colour space | none: raw 8-bit channel values, no gamma or ICC handling | same; decoding and colour management are the caller's |
| Numeric type | JavaScript Number (binary64) | `f64` for coordinates and outputs; integer counts (exact, since all partial sums are integers below 2⁵³) |
| Trigonometry | V8 `Math.cos/sin` | 360 embedded V8 bit patterns (`cdch-core/src/directions_v8.rs`); host libm differs at 37 of 360 angles on macOS |
| Rounding | ECMAScript `Math.round` | `js::round`, verified on 19 edge values against Node |
| Luma | left-to-right binary64 | `js::luma` and the table form `js::luma_fast`, proven equal on all 2²⁴ colours |
| Traversal order | angle-major, radius-minor, early exit | same in the reference port; the optimized engine precomputes the identical sample list per geometry |
| Accumulation order | integer increments, exact | exact integers, so order cannot matter |
| Division | `value / max` per bin | one `f64` division per bin (never a reciprocal multiply, which differs: mutant `NormReciprocal`) |
| Parameter defaults | the archived pipeline sends `[w // 2, h // 2]`; the oracle default and the UI button use `w / 2` | `CenterSpec::Archived` (default) and `CenterSpec::Midpoint` |
| Angle step | integers 1..=360 (the UI's validation) | `angle_step: u32` validated to 1..=360 |
| Boundary | early exit; `NaN` for empty rays | identical |

## 3. Inputs the public API treats differently

The public crate rejects some inputs the JavaScript would accept silently:

| Input | JavaScript | `ferris-cdch` |
|---|---|---|
| `NaN` / infinite centre | rows of `NaN` | `InvalidInput::NonFiniteCenter` |
| Non-integer or out-of-range step | iterates anyway / UI alert | `InvalidInput::AngleStep` |
| Zero-sized image | rows of `NaN` | `InvalidInput::EmptyImage` |
| Centre outside the image | rows of `NaN` | **accepted**; returns `NaN` rows and `is_complete() == false` |

## 4. Additional canonical function: intensity centroid

`intensity_centroid` reproduces `findIntensityCenter`: pixels are visited row by row; a
pixel counts if its canonical luma lies in `[min, max]`; `sumX`, `sumY` and `count` are
accumulated as binary64 (as JavaScript Numbers); the result is
`(Math.round(sumX / count), Math.round(sumY / count))`. If no pixel qualifies, the
JavaScript shows an alert and Ferris returns `None`.

## 5. Assumptions and their status

1. **Opaque canvas pixels.** `ip-1.js` reads pixels from a canvas into which the browser
   has drawn the image. For opaque images the canvas returns the decoded RGB values. For
   translucent images browsers store premultiplied alpha, so the values returned for
   pixels with alpha < 255 are browser-dependent approximations (fully transparent
   pixels become `(0, 0, 0)`). Ferris therefore does not claim to reproduce a browser on
   translucent images. It offers explicit `AlphaPolicy` choices instead, and the default
   `Ignore` matches the archived research pipeline (Pillow `convert("RGB")`).
2. **Canvas resampling.** The Workbench HTML draws images onto canvases scaled to fit
   500 × 500 (screen 1) or 256 × 256 (screen 3). That resampling is done by the browser
   and is not specified. Ferris computes on the pixels it is given; applications must
   state any resampling they apply (CDCH Research Studio records it in every export).
3. **Direction table provenance.** The table was recorded with Node 22.23.4 (V8 12.4) on
   darwin-arm64. The archived study used Node 24 on Linux x86-64. Evidence (192,000
   archived values reproduced bitwise) is consistent with identical directions for the
   archived protocol, but the full table has not been diffed against Node 24.
4. **Platform independence.** The matrix path uses no host math functions (only IEEE
   `+ − × ÷`, comparisons, `floor`, and the embedded table), so results should be
   identical on every platform. This has been **verified only on macOS arm64**. CI is
   configured for Linux, macOS and Windows but has not yet run.

## 6. Where tolerances are used

None for CDCH matrices, counts, coordinates, checkpoints or centroids: all comparisons
are exact (`f64::to_bits`, with `NaN == NaN`).

Tolerances appear only for the *derived feature representations* (`cdch-features`),
where NumPy's FFT and BLAS evaluation orders cannot be reproduced in general: at most
2.1·10⁻¹⁵ relative (A) and 8.1·10⁻¹³ relative (B). These are rounding differences of
mathematically equal expressions summed in a different order, bounded by
`n · ε · Σ|terms|` for `n ≤ 256` terms and `ε = 2⁻⁵³`. See `RUST_PORT_VALIDATION.md` §7.

The CDCH Research Studio cross-check against the Workbench's own screen-3 JavaScript
shows differences up to 2.96·10⁻⁸. That is the rounding of the Workbench's
`Float32Array` storage (half an ulp of a float32 near 1 is 2⁻²⁵ ≈ 2.98·10⁻⁸). It is not
a difference in the algorithm.
