# Research Stage: CDCH-Native Classification and Computational Efficiency Analysis

## Research role

Act as a scientific machine-learning researcher specializing in image forensics, efficient statistical classification, computer vision, experimental design, and reproducible software.

Continue the existing **Zero-Shot Deepfake Detection Through AI Super-Resolution Artifacts** project.

This stage investigates whether CDCH can serve as an effective, computationally efficient standalone image representation, rather than merely an auxiliary feature appended to another classifier.

Do not assume that CDCH improves accuracy or computational efficiency. Both must be measured empirically.

## 1. Recover and audit the existing project

Locate and inspect `SR_Zero_Shot_Pilot.zip` and its contents, including:

- `PROJECT_STATE.md`
- `RESEARCH_REPORT.md`
- `ABLATIONS.md`
- `DATASET_INVENTORY.md`
- `experiment.py`
- `cdch_features.py`
- `build_pairs.py`
- `config/pilot.json`
- `config/target_protocol.json`
- Existing model checkpoints, result tables, and verification records.

Reuse the existing reproducible infrastructure wherever possible.

The previous pilot used 100 DIV2K source photographs, with 60 training, 20 validation, and 20 independent holdout sources. The paired reconstruction variants included bicubic interpolation, SwinIR ×2, and Real-ESRGAN ×2.

Previously reported PNG holdout results:

| Model | AUROC |
|---|---:|
| RGB CNN | 0.544 |
| Frequency-only classifier | 0.858 |
| CDCH-only logistic regression | 0.593 |
| Frequency + CDCH logistic regression | 0.861 |
| Ensemble | 0.853 |
| Ensemble + CDCH | 0.821 |

The paired AUROC change from adding CDCH to frequency features was +0.004, with a reported 95% confidence interval of −0.029 to +0.035.

These results do not establish the usefulness or limitations of CDCH as a standalone representation.

**Do not confuse a weak CDCH logistic-regression baseline with evidence that CDCH cannot support more effective classifiers.**

Preserve all previous results. Do not overwrite or retrospectively modify the earlier experiment.

## 2. Scientific research questions

Investigate the following questions independently.

**RQ1 — Representation**

Can CDCH-derived representations distinguish authentic photographic images from AI-upscaled reconstructions?

**RQ2 — Classification architecture**

Does CDCH become more informative when used with nonlinear statistical classifiers rather than a linear model?

**RQ3 — Representation fidelity**

Did the existing 56-dimensional CDCH summary discard useful information contained in the original CDCH histograms or cumulative distributions?

**RQ4 — Computational efficiency**

Can CDCH-native classifiers achieve a favorable accuracy–latency–memory–energy trade-off compared with frequency-based and RGB-based classifiers?

**RQ5 — Feature fusion**

Does combining CDCH with frequency or RGB features improve predictive performance enough to justify additional computation?

**RQ6 — Transferability**

Can CDCH-native classifiers trained without deepfake examples transfer to real-versus-deepfake detection on previously unseen datasets?

Do not claim that RQ6 has been answered without evaluating actual real and deepfake media.

## 3. CDCH implementation requirements

Use Muzyka's original JavaScript implementation for canonical CDCH histogram generation.

The existing project records the following SHA-256 digest for the canonical `ip-1.js`:

`110732b48bc6e8b2d5167eec30f7277245913bed98b9baaa3c65c0c22aab44d8`

Verify the recovered source file against this digest before performing experiments.

Python may be used for:

- Consuming canonical CDCH output.
- Feature engineering.
- Statistical analysis.
- Classifier training.
- Benchmarking.
- Plotting and report generation.

Do not silently replace the canonical JavaScript CDCH algorithm with a Python approximation.

Investigate the computational cost of the JavaScript execution and any inter-process communication separately from downstream classification.

## 4. CDCH representation experiments

Develop and compare several representations derived from the same canonical CDCH output.

### Representation A — Existing compact summary

Use the current 56-dimensional feature representation as the baseline.

### Representation B — Rich statistical descriptors

Investigate additional descriptors derived from CDCH histograms, including:

- Distribution moments.
- Entropy and concentration.
- Quantiles and tail characteristics.
- Histogram shape and slope.
- Distances between color-channel distributions.
- Differences between cumulative distribution functions.
- Other mathematically justified descriptors.

Avoid adding arbitrary statistics without explaining their potential forensic relevance.

### Representation C — Higher-dimensional CDCH representation

Investigate whether the original histogram structure contains predictive information lost during aggregation.

Evaluate suitable fixed-dimensional representations that preserve more of the CDCH output.

Consider regularized feature vectors, dimensionality reduction, or small neural classifiers where justified.

### Representation D — Structured CDCH representation

If the CDCH output has meaningful ordering or relationships among its bins, investigate architectures that preserve those relationships instead of flattening everything into a vector of unrelated values.

Do not assume a convolutional architecture is appropriate unless the representation has a defensible structure.

For all representations:

- Prevent training/holdout leakage.
- Fit transformations on training data only.
- Record dimensionality and memory requirements.
- Avoid unjustified hyperparameter searches.
- Preserve source-disjoint evaluation.

## 5. Train standalone classifiers

Train and evaluate the following classifiers on CDCH-derived representations.

### Baseline models

1. Logistic regression.
2. Linear SVM.
3. RBF-kernel SVM.
4. Random forest.
5. Gradient-boosted decision trees.
6. Small multilayer perceptron.

Use regularization and validation to control overfitting.

Where technically justified, evaluate a compact architecture operating directly on the structured CDCH representation.

Do not automatically prefer a more complex model.

The central objective is to determine whether CDCH contains useful discriminative information and how much computation is required to exploit it.

Compare against the existing frequency-only classifier and RGB CNN using identical data partitions and clearly documented evaluation conditions.

## 6. Modern AI-upscaling considerations

The existing pilot used SwinIR and Real-ESRGAN.

These are valid neural upscalers, but they do not adequately represent the full range of modern generative super-resolution systems.

Investigate the feasibility of including modern generative restoration models such as:

- SUPIR.
- DiffBIR.
- SeeSR.
- Other relevant open-weight diffusion-based super-resolution systems.
- Commercial production-grade systems where outputs can legally and reproducibly be obtained.

Do not assume that a commercially popular model is scientifically superior or more representative without evidence.

For each candidate, evaluate:

- Model availability and licensing.
- Whether pretrained weights can be obtained.
- Reconstruction quality and model provenance.
- GPU/CPU memory requirements.
- Execution time.
- Output resolution.
- Whether matched original/reconstruction pairs can be constructed.
- Feasibility within the available computational environment.

Do not download enormous checkpoints or datasets before establishing feasibility.

If modern generative SR is infeasible in the current environment, proceed with the existing data and clearly distinguish findings about older neural SR systems from hypotheses about newer generative systems.

Do not delay the CDCH efficiency study solely because a more advanced upscaler is unavailable.

## 7. Controlled experimental design

Use authentic photographs as the negative class and neural AI-upscaled reconstructions of those same photographs as the positive class.

Maintain identical image dimensions, consistent color handling, and source-disjoint splits.

Retain conventional interpolation as a control.

The earlier frequency classifier achieved 0.990 AUROC when distinguishing originals from bicubic reconstructions, compared with 0.858 on originals versus learned SR.

This suggests an important resampling shortcut.

Therefore, evaluate:

1. Original versus AI-upscaled.
2. Original versus bicubic.
3. Bicubic versus AI-upscaled.
4. Cross-upscaler transfer.
5. Sensitivity to image compression and resizing.
6. Generalization to independent source images.

Distinguish generic reconstruction artifacts from AI-specific characteristics.

Keep the primary training task unchanged. Introducing bicubic reconstructions as a separate class or negative training class must be clearly identified as an additional experimental condition.

Do not use deepfake images for training, feature selection, or hyperparameter tuning in the primary zero-shot experiment.

## 8. Computational efficiency benchmarking

This section is a primary research objective, not an optional supplement.

Benchmark complete inference pipelines, including feature extraction.

For CDCH:

\[
T_{\mathrm{CDCH,total}}
=
T_{\mathrm{decode}}
+
T_{\mathrm{CDCH}}
+
T_{\mathrm{feature}}
+
T_{\mathrm{classifier}}
\]

For frequency classification:

\[
T_{\mathrm{frequency,total}}
=
T_{\mathrm{decode}}
+
T_{\mathrm{FFT}}
+
T_{\mathrm{feature}}
+
T_{\mathrm{classifier}}
\]

For RGB classification:

\[
T_{\mathrm{RGB,total}}
=
T_{\mathrm{decode}}
+
T_{\mathrm{preprocess}}
+
T_{\mathrm{CNN}}
\]

Avoid double-counting shared preprocessing when comparing these pipelines.

Record the following metrics:

- AUROC and AUPRC.
- Balanced accuracy.
- Latency per image.
- Throughput in images per second.
- Peak resident memory.
- Model checkpoint size.
- Feature representation size.
- CPU utilization where measurable.
- Energy per inference where reliable instrumentation is available.
- Training time and training resource consumption.

Measure performance under controlled hardware and software conditions.

Use appropriate warm-up runs and repeated timed measurements.

Report median, p95, and uncertainty or run-to-run variability where practical.

Distinguish cold-start and steady-state inference.

Benchmark CPU execution as a priority, and GPU execution if suitable resources are available.

Do not claim energy measurements without reliable power instrumentation.

Do not infer total pipeline efficiency from the number of trainable model parameters alone.

## 9. Accuracy–efficiency analysis

Construct comparisons of predictive quality versus computational cost.

Identify Pareto-efficient configurations.

For each model, report:

- Predictive accuracy.
- End-to-end latency.
- Peak memory.
- Training cost.
- Model size.
- Deployment complexity.

Investigate whether CDCH-native classifiers can provide materially lower inference costs without unacceptable deterioration in predictive performance.

Do not use arbitrary combined scores unless their weighting assumptions are explicitly justified.

Include plots showing AUROC versus latency, throughput, and memory consumption.

## 10. Feature-fusion experiments

After evaluating standalone CDCH classifiers, compare them with hybrid approaches.

Investigate:

1. Frequency-only.
2. CDCH-only.
3. RGB-only.
4. Frequency + CDCH feature-level fusion.
5. RGB + CDCH prediction-level fusion.
6. Full ensemble, where computationally feasible.

Distinguish:

- Retraining a classifier using concatenated features.
- Combining separately trained classifier outputs.
- Adding CDCH inference to an already deployed classifier.

These approaches have different computational costs and scientific interpretations.

Measure incremental accuracy and incremental computation for every fusion experiment.

Use source-validation data only for choosing fusion parameters.

Do not optimize fusion against holdout labels.

## 11. Statistical methodology

Use independent source images as the primary sampling units.

Preserve the dependency structure among reconstructions derived from the same photograph.

Report:

- AUROC.
- AUPRC.
- Balanced accuracy.
- Confidence intervals.
- Paired AUROC differences.
- Variability across repeated training seeds.
- Robustness to dataset splitting.
- Cross-upscaler evaluation.

Use source-level bootstrap procedures rather than treating correlated reconstructions as independent observations.

Predefine the primary comparisons and distinguish exploratory analyses from confirmatory tests.

Assess whether the original 20-source holdout is sufficient for reliable conclusions.

If the sample is inadequate, report the limitation and prioritize a larger source-disjoint cohort before making strong claims.

## 12. Zero-shot deepfake transfer

Preserve the original research hypothesis:

> A classifier trained exclusively on authentic photographs versus AI-upscaled reconstructions may learn representations that distinguish authentic photographs from deepfakes without encountering deepfake examples during training.

After source-domain training:

- Freeze all models.
- Freeze feature transformations.
- Freeze normalization parameters.
- Freeze thresholds.
- Freeze ensemble weights.

Then evaluate on authentic and deepfake images from external datasets, prioritizing FaceForensics++ and Celeb-DF v2.

Reuse the existing blind scoring and label-joining infrastructure.

No deepfake supervision is permitted in the primary zero-shot experiment.

If target media remain inaccessible, do not manufacture surrogate results or claim zero-shot performance.

Document the blocked evaluation and produce a reproducible evaluation package.

Keep source-domain classification and zero-shot deepfake detection results clearly separated.

## 13. Execution priorities

Work in the following order.

**Phase 1 — Audit and baseline reproduction**

Verify existing code, features, CDCH source, models, dataset splits, and reported results.

**Phase 2 — Standalone CDCH classifier study**

Compare compact, enriched, and structured CDCH representations using lightweight classifiers.

**Phase 3 — Computational efficiency**

Benchmark CDCH, frequency, and RGB pipelines under controlled conditions.

**Phase 4 — Statistical validation**

Conduct ablations, cross-validation or repeated source-disjoint splits, and uncertainty analysis.

**Phase 5 — Modern SR feasibility**

Investigate improved training datasets and generative super-resolution systems without disrupting the core CDCH experiments.

**Phase 6 — Zero-shot preparation or evaluation**

Run the frozen real/deepfake evaluation if legitimate target datasets are accessible. Otherwise preserve the evaluation protocol and report the access blocker.

Do not expand into computationally expensive model training until the corresponding hypothesis and expected scientific value are clearly defined.

## 14. Deliverables

Produce a reproducible research package containing:

1. `CDCH_RESEARCH_REPORT.md` — methodology, results, interpretation, and limitations.
2. `CDCH_REPRESENTATIONS.md` — exact mathematical definitions and implementation details for every feature representation.
3. `CLASSIFIER_COMPARISON.csv` — model-by-model performance metrics.
4. `EFFICIENCY_BENCHMARKS.csv` — full inference costs and hardware details.
5. `ABLATIONS_CDCH.md` — controlled comparisons and uncertainty estimates.
6. `SR_MODEL_FEASIBILITY.md` — feasibility analysis of advanced AI-upscalers.
7. `ZERO_SHOT_STATUS.md` — frozen models, evaluation readiness, and real target results if available.
8. Reproducible source code, configuration files, and dependency lockfile.
9. Plots of accuracy versus computation and representation-level comparisons.
10. Updated `PROJECT_STATE.md`, retaining all previous verified findings.

Include IEEE-style citations for external research claims.

Package the deliverables into a downloadable ZIP archive.

Do not fabricate results, performance measurements, completed training runs, hardware capabilities, or dataset access.

## 15. Final decision criteria

Conclude by answering:

1. Is CDCH informative enough to support a standalone classifier?
2. Which CDCH representation performs best, and is the improvement reliable?
3. Which classifier provides the strongest accuracy–efficiency trade-off?
4. How much computational overhead does CDCH add to hybrid systems?
5. Do nonlinear or structured CDCH classifiers outperform the initial logistic regression baseline?
6. Are the findings robust to interpolation, compression, and upscaler architecture?
7. What remains unknown about zero-shot deepfake transfer?
8. Which next experiment has the highest expected scientific value?

Keep conclusions proportional to the evidence.

**The goal is not to demonstrate that CDCH works. The goal is to determine when, why, and at what computational cost CDCH is useful — or to establish rigorously that it provides insufficient benefit under the tested conditions.**