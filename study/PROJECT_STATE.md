# Project state

**November 16, 2026 · DIV2K pilot complete**

| Stage | Verified progress | Pending |
|---|---|---|
| 1. Literature and dataset investigation | Verified 20 references or official release/code records. Compared DIV2K, RealSR, DRealSR, FFHQ, CelebA, FF++, Celeb-DF v2 and DFDC. Acquired and audited the official DIV2K validation HR archive. | Acquire optical and face-source archives. Document file-level terms, camera/identity groups, near duplicates, codecs and alignments. |
| 2. Controlled pair construction | Verified 100 sources and generated 500 artifacts. Each source has the reference, LR, bicubic, SwinIR and Real-ESRGAN versions in one split. | Repeat on an authentic face cohort with identity blocking and independent LR conditions. |
| 3. Classifier experiments | Trained RGB, frequency, authentic-only and reconstruction-only Isolation Forest models, two generator-specific frequency models, CDCH and combined variants. Saved checkpoints and fixed ensemble rules. | Test repeatability across independent datasets and repeated source or identity splits. |
| 4. In-domain validation | Evaluated seven contrasts and three encodings on 20 holdout sources. Audited confusion counts, within-source ordering, bootstrap metrics and paired ablations. | Extend optical and codec controls. Reserve an independent confirmatory cohort before refining features. |
| 5. Zero-shot target evaluation | Implemented and smoke-tested source-only freeze, blind scoring, label join, frame/video aggregation and tamper rejection. | Secure authorized target media. Pin the real target crop pipeline, establish label custody, seal actual scores, and evaluate official target cohorts. Target AUROC, AUPRC and confusion counts remain unmeasured. |
| 6. Analysis | Measured a strong bicubic response, asymmetric generator crossover, uncertain marginal CDCH contribution and an absence of clear ensemble gain in the pilot. | Measure target transfer, within-method compression effects, and target CDCH/fusion effects. Run supervised and self-blended reference scores on the same target examples when permitted. |

## Verified findings

- Frequency-only AUROC was 0.858 for originals versus learned SR, 0.990 for originals versus bicubic, and 0.080 for bicubic versus learned SR with the learned-SR score direction. Bicubic reached the AI-positive threshold on all 20 reference-matched holdout sources.
- The learned-reconstruction one-class score showed 0.877 AUROC against bicubic and 0.565 against originals. The ordering is a source-domain diagnostic.
- On original versus pooled learned SR, adding CDCH to frequency features changed PNG AUROC by +0.004 (paired 95% interval −0.029 to +0.035). The same directional interpolation response persisted.
- Canonical CDCH source digest is `110732b48bc6e8b2d5167eec30f7277245913bed98b9baaa3c65c0c22aab44d8`. The original JavaScript and archived oracle executed directly. Histograms passed dimension, monotonicity, finiteness and terminal-CDF checks for 560 images or re-encodes.
- Target CLI fixture evaluations used 16 SR crops assigned to pseudo-videos. They verify control flow and integrity checks.

## Next experimental decision

The main source classifier failed the interpolation-specificity check. A full-scale expansion would amplify an ambiguous surrogate signal. The next bounded test is a face-source replication with the same crossed treatments and independently chosen interpolation kernels, plus explicit original-versus-AI and AI-versus-bicubic contrasts. Freeze the existing frequency score as the direct baseline and the reconstruction one-class score as a secondary comparison before opening target labels. Reserve any feature-selection changes for source-only validation and an untouched confirmatory target cohort.

Open questions include whether the SR shortcut is driven by LR low-pass filtering, the reconstruction kernel, codec interactions, or more than one of these effects. Available pilot evidence cannot decide whether any score will transfer to deepfakes.

## CDCH-native follow-up · 16 November 2026

| Stage | Verified progress | Pending |
|---|---|---|
| Historical reproduction | All historical archive checksums passed. Full baseline rerun on the hash-matched image pairs reproduced both original checkpoints, the protocol, and five CSVs byte for byte. | Extend to an independent cohort with a new manifest. |
| Standalone representations | Trained A/B/C with six model families and a structured D CNN plus order control. Validation selected A logistic. B linear SVM led the reused PNG holdout at 0.701 AUROC. | Confirm the B-versus-A representation effect on an independent source cohort. |
| End-to-end cost | Benchmark covers 30 CPU configurations. The canonical Node+oracle and IPC dominate CDCH inference. A logistic required 84.1 ms median end to end, frequency required 2.89 ms. Interleaved paired timing confirmed hybrid overhead of roughly 78–84 ms. | Assess at additional input sizes and on deployment hardware. Obtain reliable energy instrumentation if energy is an endpoint. |
| Controlled transfer | Added common resize, generator-specific native controls, source-paired intervals, random-seed and alternate development split checks. Bicubic-versus-AI remains near chance for enriched CDCH. | Add a second conventional kernel, face identity cohort, and independent generative upscaler. |
| Modern SR feasibility | Assessed SUPIR, DiffBIR, SeeSR, HYPIR and Topaz Gigapixel as distinct conditions. No new weight downloads or model execution. | Choose a permitted candidate, pin checkpoint terms, and verify matched ×2 geometry and GPU resource usage. |
| Deepfake evaluation | Sealed source-only model parameters and fixed thresholds. Ran blind scoring and the label join on 16 real DIV2K crop files used as an SR-only integrity fixture. | Acquire authorized official FF++ and Celeb-DF v2 media and freeze the real target crop pipeline before releasing labels. Real target metrics remain unavailable. |
