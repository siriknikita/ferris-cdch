# Classifier compatibility

**Verdict.** On every archived evaluation image the Rust CDCH matrix is bitwise identical to the
canonical JavaScript matrix (640/640). Through the unchanged archived checkpoints:

- holdout scores agree within 1.4·10⁻¹⁴, with zero holdout label changes;
- all 840 AUROC values (4 encodings × 7 tasks × 30 models) are identical across the four
  extraction paths, and features differ from NumPy by at most 8.1·10⁻¹³ relative;
- the headline archived AUROCs reproduce exactly.

No model was retrained or modified.

Evidence: `reports/classifier_compatibility/` (`feature_agreement.csv`, `score_agreement.csv`,
`auroc_comparison.csv`, `per_image_scores.csv`, `image_equivalence.csv`, `summary.json`), produced
by `python/reproduce_classifiers.py` on the final build.

## 1. Inputs

The archived study does not ship its images. They were regenerated with the archived
`pilot/build_pairs.py`, using the SHA-256-pinned DIV2K validation archive and the SwinIR and
Real-ESRGAN weights (all digests verified), on this machine's CPU PyTorch (macOS arm64).

| Evidence | Result |
|---|---|
| Source archive members, source digests, split assignment | identical (100/100) |
| Original 128² crops, pixel digest from archived `inventory.json` | identical (100/100) |
| PNG file bytes | all differ: different zlib/libpng in the macOS Pillow wheel (pixel-irrelevant) |
| Bicubic, SwinIR and Real-ESRGAN pixels | not archived. Judged by whether canonical-JS A, B and C (rtol = atol = 10⁻¹²) and frequency features reproduce `feature_cache/` |

Images whose regenerated pixels reproduce the archived features ("archive-equivalent"):

| Encoding | Images | Matrices JS == Rust (bitwise) | Archive-equivalent | of which original / bicubic / SwinIR / Real-ESRGAN |
|---|---:|---:|---:|---|
| png (train+val+test) | 400 | 400 | 250 | 100 / 100 / 45 / 5 |
| jpeg90 (test) | 80 | 80 | 63 | 20 / 20 / 17 / 6 |
| jpeg75 (test) | 80 | 80 | 74 | 20 / 20 / 18 / 16 |
| resize96 (test) | 80 | 80 | 54 | 20 / 20 / 14 / 0 |

The learned-SR outputs differ slightly from the archived x86 run, because CPU PyTorch kernels are
not bitwise reproducible across platforms. This affects which pixels are tested, not the Rust/JS
comparison, which always uses identical pixels. Comparisons with the archive are restricted to
archive-equivalent images. A first, weaker criterion (A + frequency only) admitted one image that
differs in a single pixel visible only on C's finer bin grid; the A + B + C criterion excludes it.

## 2. Extraction paths compared

| Label | Matrix | Representations |
|---|---|---|
| `js` | canonical `ip-1.js` (archived Node worker, base64 float64 transport) | archived NumPy (`representation`, `descriptor`, `rich`) |
| `rust_m` | `cdch_ferris.cdch` | archived NumPy |
| `rust` | (fused; not returned) | `cdch_ferris.features_multi` |
| `rust_n` | canonical JS | `cdch_ferris.features_from_matrix` |

All four feed the same classifier code: the archived `Pipeline.predict` classifier section,
vectorized, with `native_score`, sklearn `decision_function` and the fusion rules. Checkpoints are
loaded only after their SHA-256 matches `study/SHA256SUMS`.

`python/tests/test_pipeline_equivalence.py` checks that `FerrisPipeline(backend="canonical_js")`
reproduces the archived `native_pipeline.Pipeline` outputs bit for bit, and that the Rust backends
agree within 10⁻¹². Models tested: A logistic, B linear SVM, C MLP, D conv, frequency_cdch, two
score fusions, frequency, rgb.

## 3. Features

The feature order and dimensions are unchanged: A 56, B 98 (= A + 42), C 768, and D the C vector
as a 24 × 32 tensor. Agreement on all images:

| Encoding | Rep. | rust vs js bitwise | max rel | rust_m vs js | rust vs rust_n |
|---|---|---:|---:|---|---|
| png | A | 20,728 / 22,400 | 2.1·10⁻¹⁵ | bitwise 100% | bitwise 100% |
| png | B | 36,503 / 39,200 | 3.0·10⁻¹³ | bitwise 100% | bitwise 100% |
| png | C | 307,200 / 307,200 | 0 | bitwise 100% | bitwise 100% |
| jpeg90 | A / B / C | 4,150 / 7,280 / 61,440 of 4,480 / 7,840 / 61,440 | A 1.0·10⁻¹⁵, B 7.8·10⁻¹⁴ | bitwise 100% | bitwise 100% |
| jpeg75 | A / B / C | 4,157 / 7,292 / 61,440 of the same | A 1.5·10⁻¹⁵, B 1.1·10⁻¹³ | bitwise 100% | bitwise 100% |
| resize96 | A / B / C | 4,159 / 7,275 / 61,440 of the same | A 1.5·10⁻¹⁵, B 8.1·10⁻¹³ | bitwise 100% | bitwise 100% |

The fused path is bitwise equal to Rust-from-matrix, and the Rust matrix is bitwise equal to the
JS matrix. The only differences are in feature arithmetic (FFT harmonics, BLAS dot), as explained
in RUST_PORT_VALIDATION.md §7.

Against the **archived** feature cache (archive-equivalent PNGs, values computed on Linux):

| Rep. | Canonical JS here, bitwise | Rust, bitwise | Max rel (Rust) |
|---|---:|---:|---:|
| A | 13,082 / 14,000 | 12,939 / 14,000 | 1.9·10⁻¹⁵ |
| B | 22,898 / 24,500 | 22,752 / 24,500 | 3.9·10⁻¹³ |
| C | 192,000 / 192,000 | 192,000 / 192,000 | 0 |

## 4. Scores and labels

Rust fused vs canonical JS, all 30 models, all encodings:

| Quantity | Holdout (test) images | All images (incl. train/val) |
|---|---:|---:|
| max \|score difference\| | 1.4·10⁻¹⁴ (png `cdch_B_rbf_svm`) | 0.0125 (validation rows of score-fusion models) |
| label changes at the frozen thresholds | **0** | 5 |

All 5 non-holdout label changes, and every non-negligible score difference, occur on
**validation** images. They follow from how the archived decision rules were built:

- `choose_threshold` sets each threshold to one of the validation scores. For example, the
  `cdch_A_mlp` threshold coincides exactly with the Rust score of `val/0812_swinir`, so that image
  sits on its threshold and a 2·10⁻¹⁶ difference (local NumPy) decides `≥`.
- Score-fusion rules map scores through empirical CDFs whose references are the archived
  validation scores of the same images. A validation image is tied with its own reference entry,
  so any rounding-level change moves its ECDF value by ½ / 80 = 0.00625 per member.

Holdout images are not in those references and show none of this. The effect is a property of
evaluating a model on its own calibration set, not of the CDCH implementation; canonical JS run on
another platform has the same sensitivity.

Against the archived holdout predictions (archive-equivalent test images):

| Model group | max \|Rust − archived\| | max \|JS here − archived\| | Holdout label changes |
|---|---:|---:|---:|
| float64 CDCH models (A/B/C × 6 families) | 1.95·10⁻¹⁴ | 1.95·10⁻¹⁴ | 0 |
| float32 angular CNNs (D, D permuted) | 2.4·10⁻⁷ | 2.4·10⁻⁷ | 0 |
| frequency_cdch | 3.6·10⁻⁶ | 3.6·10⁻⁶ | 0 |

The CNN and frequency gaps are identical for JS and Rust, so they come from float32 PyTorch and
frequency-feature kernels differing across platforms, not from CDCH.

## 5. AUROC

`statistics()` from the archived `experiment.py`, on the archived holdout tasks:

| Quantity | Result |
|---|---|
| AUROC rows (4 encodings × 7 tasks × 30 models) | 840 |
| rows where any extraction path differs | **0** |
| rows equal to archived `CLASSIFIER_COMPARISON.csv` | 782 |

PNG, original vs learned SR (the archived headline):

| Model | canonical JS | Rust matrix | Rust fused | Rust from JS matrix | Archived |
|---|---:|---:|---:|---:|---:|
| cdch_A_logistic | 0.5925 | 0.5925 | 0.5925 | 0.5925 | 0.5925 |
| cdch_B_linear_svm | 0.70125 | 0.70125 | 0.70125 | 0.70125 | 0.70125 |
| cdch_C_logistic | 0.5025 | 0.5025 | 0.5025 | 0.5025 | 0.5025 |
| cdch_D_conv | 0.5125 | 0.5125 | 0.5125 | 0.5125 | 0.5125 |
| frequency_cdch | 0.86125 | 0.86125 | 0.86125 | 0.86125 | 0.86125 |
| frequency | 0.8575 | 0.8575 | 0.8575 | 0.8575 | 0.8575 |
| rgb | 0.54375 | 0.54375 | 0.54375 | 0.54375 | 0.54375 |

208 of the 210 PNG rows match the archive. All 58 rows (of 840) that differ from the archive
(jpeg90 38, jpeg75 12, resize96 6, png 2) are in tasks that contain a learned-SR class, whose
regenerated pixels partly differ (§1). Every `original_vs_bicubic` row matches, because the
original and bicubic pixels are archive-identical. The 58 rows still agree across all four
extraction paths, so **CDCH extraction contributes zero AUROC difference**.

## 6. Self-contained inference

`cdch score` (Rust only: PNG decode, fused features, exported scaler + linear weights) on the 80
PNG holdout images with `cdch_A_logistic`:

- max \|CLI − Python Rust backend\| = 2.7·10⁻¹⁵, the same against canonical JS;
- zero label changes.

The weights are exported, not refit, by `python/export_linear_models.py`, which records each
source checkpoint's SHA-256.

## 7. Conclusion

Replacing the JavaScript CDCH with the Rust implementation leaves classifier behaviour unchanged
on the archived holdout. The CDCH matrices are bitwise identical, features differ at most at the
10⁻¹³ relative level in a few NumPy-order-dependent statistics, holdout scores change by ≤ 10⁻¹⁴,
and no label or AUROC value changes. No accuracy change should be attributed to the rewrite.
