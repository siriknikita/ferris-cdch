# Pilot ablation tables

All estimates use the same 20 holdout source images. Intervals are 95% percentile source bootstraps. Average precision is stepwise AUPRC. Fixed-threshold counts are TN / FP / FN / TP.

## Original versus pooled learned SR, PNG

| Model | AUROC (95% interval) | Average precision | Balanced accuracy | TN / FP / FN / TP |
|---|---:|---:|---:|---:|
| RGB CNN | 0.544 (0.541–0.598) | 0.710 | 0.512 | 19 / 1 / 37 / 3 |
| Frequency | 0.858 (0.767–0.938) | 0.920 | 0.787 | 14 / 6 / 5 / 35 |
| CDCH | 0.593 (0.496–0.688) | 0.760 | 0.500 | 11 / 9 / 22 / 18 |
| Frequency + CDCH | 0.861 (0.777–0.939) | 0.927 | 0.787 | 16 / 4 / 9 / 31 |
| RGB + frequency | 0.840 (0.749–0.933) | 0.887 | 0.675 | 19 / 1 / 24 / 16 |
| RGB + frequency + CDCH | 0.785 (0.694–0.884) | 0.864 | 0.725 | 17 / 3 / 16 / 24 |
| Authentic one-class | 0.639 (0.541–0.741) | 0.737 | 0.575 | 17 / 3 / 28 / 12 |
| AI reconstruction one-class | 0.565 (0.479–0.654) | 0.705 | 0.525 | 16 / 4 / 30 / 10 |
| Ensemble | 0.853 (0.758–0.938) | 0.904 | 0.662 | 19 / 1 / 25 / 15 |
| Ensemble + CDCH | 0.821 (0.730–0.911) | 0.895 | 0.700 | 17 / 3 / 18 / 22 |

## Matched interpolation and compression sensitivity

Values are AUROC. Bicubic is positive in the original-versus-bicubic column. Learned SR is positive in the bicubic-versus-learned-SR column.

| Model | Original / bicubic PNG | Bicubic / learned SR PNG | Original / learned SR JPEG Q90 | Original / learned SR JPEG Q75 |
|---|---:|---:|---:|---:|
| RGB CNN | 0.560 | 0.481 | 0.544 | 0.540 |
| Frequency | 0.990 | 0.080 | 0.868 | 0.810 |
| CDCH | 0.667 | 0.412 | 0.581 | 0.594 |
| Frequency + CDCH | 0.998 | 0.080 | 0.886 | 0.846 |
| RGB + frequency | 0.953 | 0.168 | 0.834 | 0.807 |
| RGB + frequency + CDCH | 0.948 | 0.196 | 0.783 | 0.765 |
| Authentic one-class | 0.880 | 0.095 | 0.566 | 0.562 |
| AI reconstruction one-class | 0.155 | 0.877 | 0.647 | 0.620 |
| Ensemble | 0.948 | 0.249 | 0.856 | 0.794 |
| Ensemble + CDCH | 0.954 | 0.275 | 0.812 | 0.765 |

## Generator crossover, PNG

| Frequency training cohort | Originals vs. SwinIR AUROC (95% interval) | Originals vs. Real-ESRGAN AUROC (95% interval) |
|---|---:|---:|
| Both upscalers | 0.925 (0.847–0.990) | 0.790 (0.635–0.920) |
| SwinIR only | 0.975 (0.932–1.000) | 0.588 (0.412–0.745) |
| Real-ESRGAN only | 0.752 (0.665–0.863) | 0.845 (0.732–0.955) |

The full estimates, intervals and fixed-threshold counts for each combination are in [holdout metrics](results/holdout_metrics.csv). Paired uncertainty for all fusion and CDCH comparisons is in [paired AUROC deltas](results/paired_auroc_deltas.csv).
