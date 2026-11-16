# Rust architecture

## 1. Three tiers, kept distinct

| Tier | Where | Status |
|---|---|---|
| **Canonical reference** | `oracle/ip-1.js` (Muzyka, SHA-256 `110732b4…aab44d8`), run through the archived `oracle.js` | Defines CDCH. Never modified. |
| **Validated Rust implementation** | `cdch-core::reference` (literal port), `cdch-core::engine` + `plan` (optimized), `cdch-features` | Bitwise equal to the canonical matrices on 713 differential cases. Used for all reported results. |
| **Experimental / diagnostic** | `cdch-bench::mutants`, `reference::Trig::Libm`, `js::direction_libm` | Deliberately non-canonical. Used only to measure harness sensitivity and libm effects. |

The research pipeline in `study/` is unchanged. The Rust path is an additional backend
(`python/ferris_pipeline.py`), and the canonical backend remains selectable (`canonical_js`).

## 2. Workspace

```
ferris/
├── Cargo.toml                  workspace, release profile (fat LTO, codegen-units = 1)
├── crates/
│   ├── cdch-core/              canonical CDCH (no dependencies)
│   ├── cdch-features/          representations A/B/C/D (depends on cdch-core only)
│   ├── cdch-python/            PyO3 + NumPy bindings → Python package `cdch_ferris`
│   ├── cdch-cli/               `cdch` executable + PNG decoding + linear-model inference
│   └── cdch-bench/             `ferris-diff` (differential/mutation tests), `ferris-bench`
├── oracle/                     ip-1.js, archived oracle.js, diff_worker.js, V8 direction table
├── python/                     uv project: backends, reproduction, benchmarks, tests
├── models/                     archived linear checkpoints exported to JSON (for the CLI)
├── scripts/                    direction-table generator, CLI throughput benchmark
├── reports/                    machine-readable results
└── study/                      original research archive (verbatim)
```

### Dependencies

| Crate | Dependency | Why |
|---|---|---|
| cdch-core | none | The algorithm needs only `std`. |
| cdch-features | cdch-core | — |
| cdch-python | `pyo3` 0.30 (abi3-py310), `numpy` 0.30, `rayon` 1.12 | Python bindings, zero-copy NumPy views and arrays, parallel batch extraction. abi3 gives one wheel per platform for CPython ≥ 3.10. |
| cdch-cli | `png` 0.18, `rayon`, `serde_json` (`float_roundtrip`) | Pure-Rust lossless decoding, batch parallelism, model files. `float_roundtrip` makes weight parsing correctly rounded. |
| cdch-bench | `png` (via cdch-cli), `rayon`, `serde_json` (`float_roundtrip`) | Fixtures, throughput, reports and the oracle wire protocol. |

The build uses no C/C++ code and no system libraries. `maturin` builds the extension; `uv`
manages the Python environment, with pins identical to the archived pilot.

## 3. `cdch-core`

### Types

- `RgbImage<'a>`: borrowed `height × width × 3` u8 pixels with a row stride, so cropped views
  need no copy. Checks buffer length and size limits (u32 pixel indices).
- `Center { x, y }`: any finite f64. `Center::archived(w, h)` = `(w // 2, h // 2)` (research
  protocol, the default); `Center::midpoint(w, h)` = `(w / 2, h / 2)` (oracle default / UI button).
- `Params { center, angle_step }`: integer steps 1..=360; `n_rays() = ⌈360 / step⌉`.
- `CdchMatrix` / `MatrixView`: row-major `rows × 256` f64.
- `CdchError`: buffer, stride, size, non-finite centre, angle step, ray count, invalid matrix,
  dimension mismatch. No panics on user input.

### `js` — JavaScript-exact primitives

- `round`: ECMAScript `Math.round` (ties toward +∞, `−0` preserved).
- `luma`: the literal formula, evaluated left to right in binary64. `luma_fast` uses three
  256-entry product tables built at compile time (`const fn`), with the same products and order,
  and `floor(v + 0.5)` by truncation. Proven equal to `luma` for all 2²⁴ colours by an
  exhaustive test.
- `direction(angle)`: `(cos, sin)` from `directions_v8.rs`, the IEEE bit patterns V8 produces for
  `Math.cos/sin((angle − 90) * (Math.PI / 180))`. Regenerate with
  `node oracle/direction_table.js > oracle/direction_table_v8.json && scripts/gen_direction_table.sh`.
  This removes the dependence on the host libm, which differs from V8 at 37 of 360 angles on macOS.
- `ray_point`: `(round(x + r·cos), round(y + r·sin))`, two separately rounded operations. Rust
  never contracts these into an FMA.

### `reference` — correctness-first port

A line-by-line transcription of `drawRaysAndGenerateHistograms` and
`calculateCumulativeHistogram`, with the JavaScript quoted in comments. It allocates freely and is
kept simple on purpose. `cdch_traced` also returns per-ray coordinates and counts.
`Trig::Libm` exists only for diagnostics.

### `plan` — `SamplingPlan`

The rounded coordinates depend only on `(width, height, centre, step)`. A plan evaluates them
once, with the same arithmetic as the reference, and stores:

| Field | Content |
|---|---|
| `unique: Vec<u32>` | distinct sampled pixels `py·w + px`, ascending (streams through image memory) |
| `samples: Vec<u32>` | per sample in emission order, an index into `unique` (multiplicity preserved) |
| `ray_starts: Vec<u32>` | CSR offsets of each ray in `samples` |
| `quotients`, `quotient_starts` | per distinct ray length `n`: `c / n` for `c = 0..=n`, computed with the same IEEE division |

For 128² with the archived centre: 26,024 samples, 14,758 unique pixels, 210 KB.

### `engine` — per-image evaluation

```
luma of each unique pixel (table form)
  ├── matrix_into:      per ray: 256-bin u32 histogram → integer prefix → cum / n
  ├── histograms:       raw u32 counts (scientific inspection)
  └── checkpoints_into: luma >> 3 = bucket in 0..32; per ray: 4 interleaved 32-bin counters
                        → prefix → quotient[n][cum] (== cum / n bit for bit)
```

The checkpoint CDF holds `F(θ, 8j + 7)` for `j = 0..32`. These are exactly the intensities
representations A (even `j`), C and D consume, so A/C/D never need the 256-bin matrix.

`Engine` is a thread-safe plan cache (`RwLock<HashMap<geometry, Arc<SamplingPlan>>>`).
`Workspace` holds the reusable luma buffer.

## 4. `cdch-features`

### Mathematical dependencies (what can be fused)

| Representation | Needs | Fused path |
|---|---|---|
| A means, standard deviations, angular differences | `F(θ, 16m + 7)` for all θ, 16 columns | yes, from the checkpoints |
| A harmonics (rfft over θ of the demeaned sampled CDF) | all 360 rays of those 16 columns, jointly | yes, after the per-ray pass; global over θ, so the 360 × 16 values must be retained |
| A validity (`descriptor` checks) | finite, ≥ 0, ends at 1, monotone | yes, equivalent to "no empty ray", a property of the plan |
| B moments, entropy, concentration, tails, max mass, mean \|ΔP\|, quantiles | the full 256-bin PMF of each ray, from **float** CDF differences | per ray only. P must be `F[b] − F[b−1]` in binary64 to match, so integer counts alone cannot reproduce the values; the 256 F values of a ray are needed. |
| B angular lags 1, 15, 90, 180 | full CDF rows of rays up to 180° apart | no, needs (almost) the whole matrix |
| B summary over rays (mean, std of 21 quantities) | all 360 per-ray values | after the per-ray pass |
| C / D (24 sectors × 32 columns) | `F(θ, 8j + 7)` | yes, from the checkpoints |

So A, C and D are computed from a 360 × 32 reduced array (92 KB, never 737 KB). B forms the
matrix in a reused internal buffer and reduces it in place; it never reaches Python.

### NumPy-order arithmetic (`numpy_compat`)

To reproduce NumPy's results bitwise wherever possible: NumPy's pairwise summation for
contiguous sums, sequential accumulation for axis-0 reductions, libm `pow`/`log2` where NumPy
calls them, `linspace` as `i·(1/255)`, and population standard deviation via NumPy's `_var`
steps. The A harmonics use a compensated (TwoSum) direct DFT for the 8 needed frequencies. That is
not bitwise equal to pocketfft, but agrees to ≤ 2.1·10⁻¹⁵ relative.

### `FeatureExtractor`

A per-thread object with a shared `Arc<Engine>` and reusable buffers (checkpoints, matrix,
sampled, scratch). `extract(img, params, rep, out)` is fused; `extract_via_matrix` is the unfused
comparison path.

## 5. Python API (`cdch_ferris`)

| Function | Returns |
|---|---|
| `cdch(rgb, center=None, angle_step=1)` | `(n_rays, 256)` float64 canonical matrix |
| `cdch_reference(rgb, …)` | same, via the literal port |
| `histograms(rgb, …)` | `(n_rays, 256)` uint32 counts |
| `sample_coordinates(w, h, center=None, angle_step=1)` | `(ray_lengths, (N, 2) coordinates)` |
| `features(rgb, kind="A", center=None, fused=True)` | A `(56,)`, B `(98,)`, C `(768,)`, D `(24, 32)` |
| `features_multi(rgb, kinds, center=None)` | `{kind: array}` from one call |
| `features_from_matrix(cdf, kind)` | representation of an existing matrix |
| `batch_features(images, kind, center=None, threads=None)` | `(n, dim)` |
| `batch_cdch(images, center=None, angle_step=1, threads=None)` | `(n, n_rays, 256)` |
| `build_info()` | version, target, canonical digest, rayon threads, cached plans |

**Input validation.** Inputs must be NumPy arrays (`TypeError` otherwise), with dtype `uint8`
(`TypeError`), shape `(H, W, 3)` in RGB order (`ValueError`), a finite centre, an angle step in
1..=360 and a known representation (`ValueError`). Empty rays raise `ValueError` for A and B, as
`descriptor()` does. C and D propagate NaN, as the archived code does.

**Memory and GIL policy.**
- Single-image calls read C-contiguous arrays in place (zero copy) and keep the GIL, so no Python
  thread can mutate the buffer mid-computation.
- Batch calls copy the stack once (about 49 KB per 128² image) and then release the GIL for
  `rayon`.
- Fortran-ordered and strided inputs are copied once into C order.
- Outputs are moved into NumPy without copying (`into_pyarray`).
- Rayon pools for explicit thread counts are cached.

**Pipeline integration.** `python/ferris_pipeline.py` provides
`FerrisPipeline(models, rgb_path, outputs, backend)` with `backend ∈ {"canonical_js",
"rust_matrix", "rust"}`. It subclasses the archived `Pipeline` and keeps its classifier section
verbatim. The default centre is the archived `(w // 2, h // 2)`. Checkpoints are loaded only after
their SHA-256 matches `study/SHA256SUMS`.

## 6. CLI (`cdch`)

```
cdch info
cdch features --kind A|B|C|D [--center X,Y] [--threads N] [--json] IMAGE.png...
cdch matrix   [--center X,Y] [--step S] IMAGE.png
cdch score    --model models/cdch_A_logistic.json [--threads N] [--timing] IMAGE.png...
```

`score` evaluates `((x − mean) / scale) · coef + intercept` with the archived weights (exported
by `python/export_linear_models.py`) and the frozen threshold. Only linear models are exported;
RBF SVM, forest, boosting, MLP and the CNN still run in Python.

## 7. Design decisions

| Decision | Reason |
|---|---|
| Embed V8's trig bit patterns instead of calling libm | libm differs from V8 at 37 angles here and varies by OS. Adversarial centres show the difference reaches sampled pixels. |
| Integer counts plus one IEEE division per value | identical to JavaScript's exact-integer Number sums; a reciprocal multiply is detectably different (mutant `NormReciprocal`: 679 cases) |
| Precomputed plan per geometry | coordinates are image-independent; this moves all trigonometry and rounding out of the per-image path |
| Unique-pixel luma | each sampled pixel's luma is computed once despite repeated sampling |
| 32-bin checkpoint output | exactly the information A/C/D use; 8× less output than the matrix |
| Quotient tables for checkpoints, direct division for the matrix | measured: lookups beat division for 32-wide rows, but lose to vectorized division for 256-wide rows |
| Mirror NumPy's evaluation order | bitwise agreement wherever NumPy's order is reproducible; residual differences stay explainable |
| Keep libm `pow`/`log2` in B | bit-equality with NumPy over speed (about 0.5 ms per image) |
| Safe Rust only, no intrinsics or target flags | portability and auditability; measured headroom from SIMD is ≤ 13% of single-image latency |
| Separate literal reference port | an independent second implementation for engine testing, without Node |

## 8. Building and portability

```bash
cargo build --release --workspace --exclude cdch-python      # core, features, CLI, bench
cargo test  --release --workspace --exclude cdch-python
cd python && uv sync                                         # builds cdch_ferris via maturin
maturin build --release -m crates/cdch-python/Cargo.toml     # standalone wheel (dist/)
```

The code is pure Rust with no platform-specific code. It was built and tested on macOS arm64
only. Linux and Windows builds are expected to work, since all dependencies are cross-platform,
but they have not been run. On Linux x86-64 the reported performance figures do not transfer.

The CDCH matrix and checkpoint outputs use no host-libm function (directions are embedded, and
everything else is IEEE `+ − × ÷` and comparisons), so they should be bitwise identical on every
platform. Feature arithmetic does call libm: DFT twiddles and `hypot` in A, `pow` and `log2` in B.
A and B can therefore differ across platforms at the ulp level, exactly as NumPy's own results do
(RUST_PORT_VALIDATION.md §7).
