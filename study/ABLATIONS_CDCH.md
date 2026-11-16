# CDCH ablations

All figures use held-out independent source photographs. Pooled PNG original versus learned SR is the primary source task. The confidence intervals below resample source groups jointly. Model search and diagnostics on the previously reported holdout are exploratory. The original validation selection remains fixed.

## Representation and architecture

| Paired PNG comparison | Δ AUROC | 95% CI |
|---|---:|---:|
| B logistic − A logistic | +0.099 | [−0.001, +0.211] |
| B linear SVM − A linear SVM | +0.095 | [−0.011, +0.211] |
| B linear SVM − A logistic | +0.109 | [+0.005, +0.226] |
| B linear SVM − frequency logistic | −0.156 | [−0.260, −0.038] |
| A RBF SVM − A logistic | +0.034 | [−0.055, +0.141] |
| C logistic − A logistic | −0.090 | [−0.211, +0.013] |
| D circular CNN − permuted ordering | −0.006 | [−0.020, +0.005] |

Three alternate development splits gave B-minus-A AUROC improvements of +0.015, +0.008 and +0.060 for logistic and +0.014, +0.011 and +0.069 for linear SVM. The splits share source photographs across repetitions, so the directions are descriptive. The original validation partition selected A logistic at 0.674. Its seed-independent formulation returned the same result across seeds. MLP, forest and D seed variation appears in `results/stability.csv`. Three MLP runs and the C linear SVM reached their initial iteration caps. See `source_run/model_inventory.csv` for exact fit details. The B linear SVM converged.

## Interpolation, compression and resizing

| Pipeline | Original vs pooled AI PNG | Original vs bicubic PNG | Bicubic vs pooled AI PNG | Primary JPEG 90 | Primary JPEG 75 | Primary resize 96 |
|---|---:|---:|---:|---:|---:|---:|
| CDCH A logistic | 0.593 | 0.668 | 0.413 | 0.581 | 0.594 | 0.580 |
| CDCH B logistic | 0.691 | 0.693 | 0.513 | 0.696 | 0.670 | 0.640 |
| CDCH B linear SVM | 0.701 | 0.710 | 0.509 | 0.694 | 0.660 | 0.643 |
| CDCH D angular CNN | 0.513 | 0.548 | 0.463 | 0.513 | 0.519 | 0.510 |
| Frequency logistic | 0.858 | 0.990 | 0.080 | 0.867 | 0.810 | 0.834 |
| RGB CNN | 0.544 | 0.560 | 0.481 | 0.544 | 0.540 | 0.536 |

Bicubic carries the positive label in original versus bicubic only to test the trained direction. In bicubic versus AI, bicubic is negative and the positive class remains learned SR. The original-versus-bicubic frequency response and the low score for bicubic-versus-AI show that the detector assigns strong learned-SR scores to the conventional control. The standalone enriched model reduces the extreme response but ranks bicubic versus AI close to chance. Within-source ordering is a separate diagnostic whose paired rates appear in `results/within_source_ordering.csv`.

## Cross-upscaler transfer

Training and validation kept the authentic crops and only the indicated SR positive variant. Test matched held-out originals to each generator. All numbers are PNG AUROC. The full table includes bicubic controls, transform robustness, and intervals.

| Pipeline | SwinIR → SwinIR | SwinIR → Real-ESRGAN | Real-ESRGAN → SwinIR | Real-ESRGAN → Real-ESRGAN |
|---|---:|---:|---:|---:|
| CDCH A logistic | 0.598 | 0.540 | 0.565 | 0.625 |
| CDCH B logistic | 0.663 | 0.603 | 0.590 | 0.620 |
| CDCH B linear SVM | 0.660 | 0.608 | 0.585 | 0.618 |
| Frequency logistic | 0.975 | 0.588 | 0.753 | 0.845 |

## Fusion

Validation chose CDCH ranks at weights of 0.25 alongside frequency, 0.75 alongside RGB, and 0.25 alongside the full ensemble. The remaining score weights were equal within the respective baseline. The frequency + CDCH concatenation estimates one logistic model on a joint feature vector. Scores from the newly fitted validation-selected concatenation match the earlier concatenation because both use representation A. PNG fusion point estimates and the paired latency changes appear in the research report. The holdout is too small to establish a materially positive hybrid effect.
