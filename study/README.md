# CDCH-native classification study

Start with [CDCH_RESEARCH_REPORT.md](CDCH_RESEARCH_REPORT.md). The verified historical pilot appears under `pilot/` with its own original checksums and project state. The stage includes a locked environment, verified derived caches, source checkpoints, per-image predictions and benchmark samples. The DIV2K photographs, generated crops, and pretrained upscaler weights are external inputs.

## Reproduce the source experiment

Use Python 3.12, Node.js 24 and uv. Run from this directory. The pilot's pinned lockfile is shared with the new stage. On Linux, pin `taskset` to one available CPU for benchmark runs. The measurement environment used CPU 6 in its affinity set.

```bash
uv sync --project pilot --locked
export PYTHONPATH="$PWD/pilot"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P="$PWD/pilot/.venv/bin/python"

(cd pilot && sha256sum -c SHA256SUMS)
"$P" pilot/build_pairs.py --assets assets --fetch --out pairs
"$P" extract_native.py --pairs pairs --out recomputed-features
"$P" train_native.py --pairs pairs --features recomputed-features --out recomputed-study
"$P" pilot/experiment.py --pairs pairs --out recomputed-baseline \
  --cdch pilot/results/cdch-png.json --cdch-jpeg 90 pilot/results/cdch-jpeg90.json \
  --cdch-jpeg 75 pilot/results/cdch-jpeg75.json
"$P" audit_native.py --pairs pairs --features recomputed-features --run recomputed-study \
  --baseline recomputed-baseline --out recomputed-audit
taskset -c 6 "$P" benchmark_native.py --pairs pairs --run recomputed-study --out recomputed-benchmark
taskset -c 6 "$P" benchmark_paired.py --pairs pairs --run recomputed-study --out recomputed-paired
"$P" plot_native.py --run recomputed-study --bench recomputed-benchmark --out recomputed-plots
```

The pipeline independently checks all 500 historical image hashes before extracting features. If the official asset download is unavailable, place the assets listed in `pilot/config/assets.json` in the `assets` directory and omit `--fetch`. If the study's matched PNGs and original manifest are already available, use that directory as `--pairs`. Asset-level verification and hardware variation details appear in the historical README. Only load trusted local checkpoints. All script output directories above should be fresh.

## Package map

| File or directory | Contents |
|---|---|
| `CDCH_RESEARCH_REPORT.md`, `ABLATIONS_CDCH.md` | Measurements and scientific interpretation |
| `CDCH_REPRESENTATIONS.md` | Mathematical definitions and model capacity |
| `CLASSIFIER_COMPARISON.csv` | Per-encoding, per-contrast held-out metrics and source-bootstrap intervals |
| `EFFICIENCY_BENCHMARKS.csv` | End-to-end CPU measurements and model resource inventory |
| `results/` | Raw predictions, paired deltas, crossover, stability, checks, isolated timing samples and paired timings |
| `source_run/` | Fitted candidate models and fixed source decisions |
| `feature_cache/` | A/B/C and frequency features for exact pilot PNG, JPEG and resize conditions |
| `ZERO_SHOT_STATUS.md`, `target_native.py` | Freeze, blind score and post-seal label join |
| `fixture/` | SR-only CLI smoke manifests and execution evidence |
| `SR_MODEL_FEASIBILITY.md` | Modern SR release and hardware review |
| `plots/` | Accuracy versus latency, throughput and RSS, representation comparison, and timing stages |
| `pilot/` | Original verified experiment, checkpoints, plots, asset pins and uv lockfile |
