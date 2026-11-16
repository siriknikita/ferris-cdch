# CDCH-native classification and computational efficiency

**16 November 2026 · Exploratory follow-up on a 100-source DIV2K pilot**

## Findings

Enriching the CDCH representation increased the best observed standalone AUROC from 0.593 to 0.701 on the existing PNG holdout. The top holdout candidate used 98 CDCH statistics and a linear SVM. Validation selected the original 56-feature logistic model. When classifier family was held constant, both linear comparisons favored the enriched features, but the paired 95% intervals included zero. The best standalone CDCH score remained below the 0.858 frequency baseline. On bicubic versus learned SR, the enriched linear SVM scored 0.509 AUROC.

The standalone CDCH implementation was slower and used more resident memory than the frequency baseline and the small RGB CNN. For CDCH A, the canonical JavaScript and its oracle plus serialization/transport accounted for nearly all steady-state time. The classifier itself took a median 0.62 ms. A source-matched, interleaved benchmark confirmed a median 79.8 ms cost above frequency-only inference. CDCH fusion with frequency features improved PNG AUROC by 0.004 with a paired interval spanning both gains and losses. FaceForensics++ and Celeb-DF v2 transfer remain unmeasured.

## Design and provenance

The archive `SR_Zero_Shot_Pilot.zip` matched the copy in the original workspace. All 39 historical files were recovered including the checksum manifest. Each of its 38 listed hashes passed. The original image manifest matched by SHA-256. The extractor checked all 500 derivative files, including the 100 LR images. The pilot baseline was rerun using those images and the locked dependencies. Both checkpoints, the protocol, and all five evaluation tables reproduced byte for byte. The canonical JavaScript matched the specified digest. The new timed bridge reproduced four full oracle matrices exactly. Its compact outputs matched all 560 previously archived descriptors at tolerance 10⁻¹². Another 80 matrices were generated for a common resize perturbation.

The original data are 100 DIV2K validation photographs [1] with a matched 128 × 128 reference crop and reconstructions from bicubic, SwinIR ×2 [2], and Real-ESRGAN ×2 [3]. Each reconstruction shares a source with its authentic crop. The original 60/20/20 source-disjoint partitions were preserved. Primary training inputs were original photographs and the two pooled learned SR variants, 180 training examples. There were 60 validation examples and 60 holdout examples for the primary contrast. Bicubic was an inference-only control. All transforms, model selection, operating thresholds, and rank-fusion weights were determined using training and validation sources. Compression applied JPEG 90 or 75 with 4:2:0 subsampling to all variants, and the resizing control applied Lanczos 128→96→128. The PNG-trained scoring pipelines were held fixed.

CDCH A is the pilot 56-feature vector. B adds 42 luma distribution and angular CDF descriptors. C preserves coarse angle and intensity structure as a 768-element vector. D uses those same values with a circular angular CNN. A, B and C each entered logistic regression, linear and RBF SVMs, random forest, histogram boosting and a small MLP. D had a permuted-order control. Source-only validation selected the highest prespecified candidate AUROC with deterministic tie resolution. The design estimated source-clustered, percentile intervals from 4,000 bootstrap draws. Average precision estimates AUPRC through the stepwise precision-recall area. The positive prevalence is 2/3 in the primary pooled task. The validation-chosen thresholds are kept fixed across all contrasts.

The 20-source holdout had already been used by the previous project phase and was reused to compare with the published-in-project baseline. All follow-up intervals are exploratory and unadjusted for multiplicity. The intervals quantify sampling uncertainty under this cohort and model fit. Three fixed alternate splits of the 80 development sources and three seeds for selected stochastic models probe stability. Those alternate splits overlap and do not add independent holdout observations.

## Source classification

| Pipeline | Validation AUROC | PNG holdout AUROC (95% source CI) | Average precision | Balanced accuracy |
|---|---:|---:|---:|---:|
| CDCH A · logistic, validation-selected | 0.674 | 0.593 [0.504, 0.690] | 0.760 | 0.500 |
| CDCH A · RBF SVM | 0.573 | 0.626 [0.578, 0.709] | 0.774 | 0.550 |
| CDCH B · logistic | 0.626 | 0.691 [0.616, 0.801] | 0.847 | 0.638 |
| CDCH B · linear SVM, highest native holdout score | 0.629 | 0.701 [0.621, 0.816] | 0.850 | 0.650 |
| CDCH D · angular CNN | 0.516 | 0.513 [0.491, 0.549] | 0.706 | 0.513 |
| Frequency · logistic | See model inventory | 0.858 [0.767, 0.939] | 0.920 | 0.788 |
| RGB CNN | See model inventory | 0.544 [0.541, 0.594] | 0.710 | 0.513 |
| Frequency + CDCH A · jointly trained logistic | See model inventory | 0.861 [0.780, 0.938] | 0.927 | 0.788 |

The holdout-leading B linear SVM gained 0.109 relative to A logistic, with an unadjusted paired interval of +0.005 to +0.226. This comparison changes both representation and classifier and highlights a holdout winner. Matched changes were +0.099 [−0.001, +0.211] for logistic and +0.095 [−0.011, +0.211] for linear SVM. B outscored A within both matched families in three alternate development partitions. A led B within these families on the original validation partition, showing dependence on the source split. The strongest compact nonlinear candidate gained 0.034 [−0.055, +0.141] over A logistic. Neither the richer CDF vectors nor the angular CNN gave a measurable advantage in this setup. D minus the permuted D control was −0.006 [−0.020, +0.005]. Results for the tested pooled CDF design leave open the value of other raw or higher-resolution representations.

The selected A classifier achieved 0.668 AUROC on original versus bicubic and 0.413 on bicubic versus learned SR. The enriched B linear model scored 0.710 and 0.509 on the same controls. Frequency reached 0.990 and 0.080 in the learned-SR score direction. These results show substantial dependence on reconstruction and interpolation artifacts. B's 0.701 fell to 0.660 after JPEG 75 and 0.643 after resize. Refit B linear models trained on one upscaler reached 0.608 from SwinIR to Real-ESRGAN and 0.585 in reverse. The frequency classifier showed an asymmetric 0.588 and 0.753. See [ABLATIONS_CDCH.md](ABLATIONS_CDCH.md) for the complete controls.

## End-to-end CPU inference

Batch one used 60 ordered trials and eight warm-up inputs for every pipeline. Inputs were the same checksum-verified pilot PNGs, sampled without replacement from the 80 four-variant holdout crops. Fresh Python processes isolated each model. OpenMP, OpenBLAS, and PyTorch inference used one thread and CPU affinity was pinned on an Intel Xeon Platinum 8573C. Node.js was 24.22.1 and PyTorch was 2.7.1+cpu. The shared environment showed run-to-run timing variability, reported as three consecutive 20-image block medians. A second experiment interleaved 40 identical images across the key pipelines, with randomized execution order and a reused Node worker. Its paired differences avoid comparing separately timed medians for incremental cost.

| Pipeline | Median / p95 (ms) | Images/s | Sampled peak RSS (MiB) | Checkpoint size (KiB) |
|---|---:|---:|---:|---:|
| Frequency | 2.89 / 5.05 | 306.05 | 291.34 | 2.10 |
| RGB CNN | 5.31 / 6.92 | 183.47 | 310.84 | 99.25 |
| CDCH A · logistic | 84.10 / 160.65 | 10.31 | 520.06 | 3.11 |
| CDCH B · linear SVM | 109.54 / 166.02 | 8.54 | 519.66 | 4.29 |
| CDCH D · angular CNN | 82.67 / 125.55 | 11.17 | 525.39 | 7.30 |
| Frequency + CDCH A · jointly fitted | 135.62 / 247.14 | 6.78 | 518.95 | 3.85 |

Totals include decode, feature construction and prediction. Frequency extraction groups the FFT and spectral descriptors in one measured interval. CDCH A spent median 47.88 ms in canonical JS plus oracle setup, 33.79 ms in transport and serialization, 1.00 ms in feature construction and 0.62 ms in classification. Entry-level timings use a persistent worker and perform the complete raster-to-score path for every input. The bridge packs RGB by JSON and returns one 737,280-byte float64 matrix encoded as base64. The IPC residual includes packaging on both ends, pipe/format overhead and scheduling. These measurements describe the implementation in this package. Median stages need not sum to the median total. CDCH B's additional statistics took 20.72 ms at the median. First-image cold-session time includes loading one model and launching the Node worker where necessary, after the Python runtime and libraries are imported. It was 279.0 ms for CDCH A, 3.47 ms for frequency and 11.47 ms for RGB.

RSS combines contemporaneous benchmark-process and Node readings after each predicted image. The CSV additionally reports the sum of each process's individual RSS high-water marks as an upper bound. Common imported libraries influence the absolute footprint. Measured training costs are classifier fitting from feature caches. These were approximately 13.4 ms wall / 13.4 ms process CPU for A logistic, 4.9 / 4.9 ms for B linear SVM and 2.24 / 2.24 s for D. The archived frequency and RGB checkpoints were reproduced without separate fitting-resource instrumentation. The single-process statistical pipeline requires Python and FFT/scikit-learn, the model weights for the RGB path require PyTorch, and the CDCH paths require Node plus the original JS and bridge. CUDA and differential power instrumentation were unavailable. No energy estimate or fitting RSS is reported.

## Fusion and Pareto comparison

| Hybrid relative to deployed baseline | Δ AUROC [95% paired CI] | Paired median extra latency [95% interval] |
|---|---:|---:|
| Retrained frequency + CDCH A features vs frequency | +0.004 [−0.027, +0.034] | +81.9 ms [79.8, 85.3] |
| Frequency/CDCH separate prediction fusion vs frequency | −0.003 [−0.032, +0.031] | +79.3 ms [77.9, 82.4] |
| RGB/CDCH separate prediction fusion vs RGB | +0.041 [−0.059, +0.110] | +83.5 ms [80.3, 84.8] |
| CDCH added to full ensemble vs ensemble | −0.058 [−0.116, +0.006] | +78.2 ms [75.1, 81.6] |

Feature concatenation refits the downstream logistic model. Score fusion uses separately fitted classifiers and adds execution at inference with fixed validation-rank transforms and weights. Validation selected CDCH A so the new concatenation reused the original A feature definition. Its refit reproduced the archived concatenated model's scores. Isolated latency variation between the two equivalent concatenations should be treated as timing variation. The Pareto frontier by AUROC, latency and sampled RSS consisted of the frequency baseline and the frequency + CDCH concatenation after treating the duplicate implementations as one configuration. All standalone CDCH models were dominated by frequency in those three trial metrics. The hybrid enters the point-estimate frontier on an uncertain 0.004 AUROC increase. The standalone checkpoint sizes and their small score-only times did not translate into lower pipeline cost.

## Decision and next experiment

CDCH contains modest standalone signal for original versus the tested learned reconstructions, with enriched luma statistics the best holdout candidate. Classifier nonlinearity and explicit angular locality gave no reliable gains under this design. The frequency model provided the strongest measured end-to-end accuracy and efficiency trade-off. The interpolation and cross-upscaler controls leave AI specificity unresolved. External deepfake transfer is unknown. The extended freeze, blind scoring, label join and tamper test were exercised on a 16-image SR-only integrity fixture.

The next experiment with highest expected scientific value is an independent authentic-face cohort with source or identity grouping, fixed A logistic, B linear, and frequency-only scores, plus bicubic and a second conventional kernel. Precommit original versus neural SR and bicubic versus neural SR paired contrasts and a fixed crop and codec protocol before using the new holdout. Size the cohort to resolve a scientifically meaningful source-paired AUROC difference. Include a modern author-released model only after its geometry, terms and runtime pass the feasibility check. Evaluate sealed scores on the actual target videos once authorized. The pilot gives no basis to treat a detector of reconstruction as a detector of deepfakes.

## References

[1] E. Agustsson and R. Timofte, “NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study,” CVPR Workshops, 2017. [Online]. Available: [DIV2K release](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

[2] J. Liang et al., “SwinIR: Image Restoration Using Swin Transformer,” ICCV Workshops, 2021. [Online]. Available: [model release](https://github.com/JingyunLiang/SwinIR).

[3] X. Wang et al., “Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data,” ICCV Workshops, 2021. [Online]. Available: [model release](https://github.com/xinntao/Real-ESRGAN).
