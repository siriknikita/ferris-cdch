# Zero-shot deepfake detection through super-resolution artifacts

This package contains the 100-source DIV2K pilot, its trained models, scored holdout, provenance, and a separate target-evaluation workflow. The target workflow passed a smoke test on SR fixtures. The real deepfake evaluation remains pending.

- [Research report](RESEARCH_REPORT.md)
- [Dataset inventory](DATASET_INVENTORY.md)
- [Project state](PROJECT_STATE.md)
- [IEEE references](REFERENCES.md)
- [Ablation summary](ABLATIONS.md)

## Reproduce the pilot

Use Python 3.12, `uv`, and Node.js 24. The measured run used Node 24.22.1, PyTorch 2.7.1+cpu, Pillow 13.0.0 and Linux x86-64. The asset downloader retrieves the official DIV2K validation archive (429 MiB) and public author-released model weights, pins code and verifies all SHA-256 digests. The dataset owner limits DIV2K to academic research [2]. Budget approximately 2 GiB for the environment, assets, and generated files. Run from the package root.

```bash
uv sync --locked
uv run --locked python build_pairs.py --assets assets --fetch --out pairs
mkdir -p generated
uv run --locked python cdch_features.py --pairs pairs --out generated/cdch-png.json
uv run --locked python cdch_features.py --pairs pairs --quality 90 --out generated/cdch-jpeg90.json
uv run --locked python cdch_features.py --pairs pairs --quality 75 --out generated/cdch-jpeg75.json
uv run --locked python experiment.py --pairs pairs --out reproduced-run \
  --cdch generated/cdch-png.json \
  --cdch-jpeg 90 generated/cdch-jpeg90.json \
  --cdch-jpeg 75 generated/cdch-jpeg75.json
uv run --locked python plot_results.py --run reproduced-run --out reproduced-plots
uv run --locked python smoke_target.py --pairs pairs --run reproduced-run --out reproduced-fixture
```

The pilot code uses a single context crop per source. It keeps the 80 × 80 LR crop for audit. It crops the 160 × 160 HR reference and x2 outputs identically to 128 × 128. Results include the source manifest and exact file hashes. File-level reproducibility can vary across architectures and numeric libraries. Check versions and compare metric values and manifests on the intended platform.

The author JavaScript was recovered from the project's earlier CDCH notebook, verified byte-for-byte against the recorded SHA-256 and included in `cdch/ip-1.js` with author attribution. The archived adapters execute it directly in Node. Python computes 56 downstream statistics from its raw 360 × 256 output. The included code and checkpoints support local academic research. Obtain the respective rightsholders' terms before redistributing author code, assets, or data.

## Files

| Location | Contents |
|---|---|
| `config/` | Pilot configuration, upstream URL and digest manifest, target evaluation plan |
| `results/manifest.json` and `results/inventory.json` | 500 pair-file entries and 100 source-file entries with dimensions, split assignments, hashes, and crop coordinates |
| `results/holdout_metrics.csv` | 252 holdout comparisons with estimates, source-bootstrap intervals, prevalence, and confusion counts |
| `results/holdout_scores.csv`, `results/holdout_disagreement.csv` | Model scores and dispersion of member percentile scores, by source, variant, and encoding |
| `results/within_source_ordering.csv`, `results/paired_auroc_deltas.csv` | Matched source-level ordering and paired feature/fusion comparisons |
| `results/rgb.pt`, `results/statistical.joblib`, `results/protocol.json` | CNN and statistical checkpoints, preprocessing provenance, validation CDFs, and fixed thresholds |
| `results/cdch-*.json` | Derived canonical CDCH descriptor sidecars with image and code hashes |
| `results/verification.txt` | Executed verification checks and scope |
| `plots/` | Reproducible control, compression, and score-distribution figures |

Load only checkpoint files with trusted provenance. The CNN loads with PyTorch weights-only deserialization. The statistical checkpoint uses Joblib.

## Run the target evaluation once media access is available

Choose and version the decoder, frame sample selection, face tracker, ROI, and crop-resize policy with `config/target_protocol.json` before scoring. Record this as `target_preprocessing.json`. A separate custodian should map opaque video IDs to dataset metadata and keep that mapping unavailable to the training and scoring operator until scores are sealed. A target pipeline must compile its real-data manifest from official media and split files. The package provides the crop scoring and evaluation interface. Video decoding, detection and tracking remain dataset integration work.

Prepare `blind.csv` with columns `item_id,video_id,path,sha256`. Use globally unique opaque IDs and normalized RGB 128 × 128 PNGs under `target_crops/`. Record the number of videos for which crops failed and record frame coverage by video. Use distinct opaque video IDs for different encodings of the same video and consistent dependence groups in the eventual label mapping. Store an independently timestamped copy of the freeze and score digests before receiving target labels.

```bash
uv run --locked python zero_shot.py freeze --run reproduced-run --pairs pairs \
  --preprocessing target_preprocessing.json --out frozen
uv run --locked python cdch_features.py --blind blind.csv --crops target_crops --out target-cdch.json
uv run --locked python zero_shot.py score --frozen frozen --blind blind.csv \
  --crops target_crops --cdch target-cdch.json --out target-scored
```

After sealing, provide `labels.csv` with columns `video_id,dataset,compression,manipulation,label,cluster_id`. Use `real` with label 0 and the named manipulation with label 1. Each row covers one scored video. Declare dependence groups before the join and document unresolved overlaps. Inspect frame and video attrition by class after release.

```bash
uv run --locked python zero_shot.py evaluate --frozen frozen --scored target-scored \
  --labels labels.csv --out target-evaluated
```

The evaluator emits direct frame and mean-video analyses for each dataset, codec, and manipulation, paired model deltas, and a separately named exploratory reversed-ranking table. Declare any gaps in dependence grouping and small strata when interpreting intervals. The hashes detect changes relative to the recorded manifests. Independent timestamping and label custody establish when those manifests were sealed.
