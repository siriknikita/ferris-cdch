"""Regenerate the pilot plots from the reported metrics and scores."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = {
    "rgb": "RGB CNN",
    "frequency": "Frequency",
    "cdch": "CDCH",
    "frequency_cdch": "Frequency + CDCH",
    "rgb_frequency": "RGB + frequency",
    "rgb_frequency_cdch": "RGB + frequency + CDCH",
    "authentic_anomaly": "Authentic one-class",
    "reconstruction_similarity": "AI one-class",
    "ensemble": "Ensemble",
    "ensemble_cdch": "Ensemble + CDCH",
}
ENCODING = {"png": "PNG", "jpeg90": "JPEG Q90", "jpeg75": "JPEG Q75"}
CONTRASTS = {
    "original_vs_ai": "Original / learned SR",
    "original_vs_bicubic": "Original / bicubic",
    "bicubic_vs_ai": "Bicubic / learned SR",
}
COLORS = ["#237d8d", "#cb7044", "#624d9b"]
VARIANTS = {"original": "Original", "bicubic": "Bicubic", "swinir": "SwinIR", "realesrgan": "Real-ESRGAN"}


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def interval_plot(index, models, series, output, title):
    fig, ax = plt.subplots(figsize=(9.0, max(3.5, 0.47 * len(models) + 1.7)), layout="constrained")
    for i, (keys, label) in enumerate(series):
        y = np.arange(len(models)) + (i - (len(series) - 1) / 2) * 0.19
        rows = [index[keys + (model,)] for model in models]
        point, lower, upper = ([float(row[k]) for row in rows] for k in ("auroc", "auroc_lo", "auroc_hi"))
        ax.hlines(y, lower, upper, linewidth=1.4, color=COLORS[i], alpha=.77)
        ax.scatter(point, y, s=30, color=COLORS[i], label=label, zorder=3)
    ax.axvline(.5, linestyle="--", color="#65717a", linewidth=1)
    ax.set(yticks=np.arange(len(models)), yticklabels=[MODELS[k] for k in models],
           xlim=(0, 1.02), xlabel="AUROC  ·  95% source-bootstrap intervals",
           title=title)
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.14), frameon=False, ncols=len(series))
    ax.grid(axis="x", alpha=.18)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def score_plot(rows, disagreement, output):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    series = [
        ([r for r in rows if r["encoding"] == "png" and r["model"] == "frequency"],
         "Frequency classifier", "Decision score (higher = learned SR)", "score"),
        ([r for r in rows if r["encoding"] == "png" and r["model"] == "reconstruction_similarity"],
         "AI one-class model", "Similarity score (higher = learned SR)", "score"),
        ([r for r in disagreement if r["encoding"] == "png" and r["ensemble"] == "ensemble_cdch"],
         "Ensemble with CDCH", "SD of member validation-percentile scores", "rank_score_sd"),
    ]
    for ax, (data, title, ylabel, field) in zip(axes, series):
        values = [[float(r[field]) for r in data if r["variant"] == variant] for variant in VARIANTS]
        ax.boxplot(values, tick_labels=list(VARIANTS.values()), showfliers=False,
                   medianprops=dict(color="#16283f"), patch_artist=True,
                   boxprops=dict(facecolor="#d9e8ec", edgecolor="#237d8d"))
        rng = np.random.default_rng(16)
        for x, group in enumerate(values, 1):
            ax.scatter(x + rng.uniform(-.12, .12, len(group)), group, color="#237d8d", s=10, alpha=.48)
        ax.set(title=title, ylabel=ylabel)
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=.18)
    fig.suptitle("Holdout score distributions by treatment")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    index = {(r["encoding"], r["task"], r["model"]): r for r in read(args.run / "holdout_metrics.csv")}
    interval_plot(index, ["frequency", "frequency_cdch", "cdch", "authentic_anomaly",
                          "reconstruction_similarity", "ensemble", "ensemble_cdch"],
                  [(("png", k), v) for k, v in CONTRASTS.items()],
                  args.out / "01_interpolation_controls.png", "Response to matched interpolation controls")
    interval_plot(index, list(MODELS), [((k, "original_vs_ai"), v) for k, v in ENCODING.items()],
                  args.out / "02_compression_ablation.png", "Original versus learned SR after matched re-encoding")
    score_plot(read(args.run / "holdout_scores.csv"), read(args.run / "holdout_disagreement.csv"),
               args.out / "03_score_distributions.png")


if __name__ == "__main__":
    main()
