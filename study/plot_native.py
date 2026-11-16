"""Produce scope-matched accuracy and efficiency figures from measured CSVs."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILIES = ("logistic", "linear_svm", "rbf_svm", "forest", "boosting", "mlp")
COLORS = {"A": "#455fb7", "B": "#078070", "C": "#836ba6", "D": "#b17424",
          "baseline": "#2b334b", "fusion": "#b55145"}
NAMES = {"cdch_A_logistic": "CDCH A · logistic", "cdch_B_linear_svm": "CDCH B · linear SVM",
         "cdch_D_conv": "CDCH D · angular CNN", "frequency": "Frequency",
         "frequency_cdch": "Frequency + CDCH", "rgb": "RGB CNN"}


def table(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def family_for(name):
    if name.startswith("cdch_"):
        return name.split("_")[1]
    if name in {"frequency", "rgb", "authentic_anomaly", "reconstruction_similarity", "ensemble"}:
        return "baseline"
    return "fusion"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180})
    records = table(args.run / "CLASSIFIER_COMPARISON.csv")
    primary = {r["model"]: float(r["auroc"]) for r in records
               if r["encoding"] == "png" and r["task"] == "original_vs_ai"}
    bench = {r["model"]: r for r in table(args.bench / "EFFICIENCY_BENCHMARKS.csv")}

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.7))
    panels = [("median_ms", "Median latency (ms)", True),
              ("throughput_img_s", "Throughput (images/s)", True),
              ("sampled_combined_peak_rss_mib", "Sampled peak RSS (MiB)", False)]
    for ax, (metric, label, log) in zip(axes, panels):
        for kind, color in COLORS.items():
            keys = [name for name in bench if family_for(name) == kind]
            ax.scatter([float(bench[name][metric]) for name in keys],
                       [primary[name] for name in keys], s=29, alpha=0.83,
                       color=color, label=f"CDCH {kind}" if len(kind) == 1 else kind.capitalize())
        if log:
            ax.set_xscale("log")
        for name, short in {"frequency": "Frequency", "rgb": "RGB", "cdch_B_linear_svm": "B · linear SVM"}.items():
            ax.annotate(short, (float(bench[name][metric]), primary[name]), xytext=(4, 6),
                        textcoords="offset points", fontsize=8)
        ax.set(xlabel=label, ylim=(0.45, 0.93))
        ax.grid(axis="y", color="#ebecf2")
    axes[0].set_ylabel("Holdout AUROC")
    fig.suptitle("Accuracy and end-to-end CPU cost", x=0.075, ha="left", fontweight="bold", fontsize=15)
    fig.text(0.075, 0.88, "DIV2K pilot · originals vs. learned ×2 SR · 20 independent sources · PNG",
             fontsize=10, color="#454d64")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9)
    fig.subplots_adjust(top=0.81, bottom=0.19, wspace=0.28)
    fig.savefig(args.out / "01_accuracy_efficiency.png", bbox_inches="tight")
    plt.close(fig)

    fig, (ax, right) = plt.subplots(1, 2, figsize=(12.2, 5.3), gridspec_kw={"width_ratios": [1.27, 1]})
    x = np.arange(len(FAMILIES))
    for i, kind in enumerate("ABC"):
        cells = [f"cdch_{kind}_{name}" for name in FAMILIES]
        ax.scatter(x + (i - 1) * 0.17, [primary[name] for name in cells],
                   color=COLORS[kind], label=f"CDCH {kind}", s=32)
    ax.scatter([6], [primary["cdch_D_conv"]], color=COLORS["D"], label="CDCH D", s=33)
    ax.set_xticks(np.arange(7), ["Logistic", "Linear SVM", "RBF SVM", "Forest", "Boosting", "MLP", "Angular CNN"],
                  rotation=30, ha="right")
    ax.set_ylabel("PNG holdout AUROC")
    ax.set_ylim(0.44, 0.91)
    ax.grid(axis="y", color="#ebecf2")
    ax.axhline(primary["frequency"], color="#626974", linestyle="--", linewidth=1, label="Frequency")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.29), frameon=False, ncol=3, fontsize=8)
    ax.set_title("Classifiers by representation", loc="left", weight="bold")
    encodings = ("png", "jpeg90", "jpeg75", "resize96")
    for name in ("frequency", "cdch_B_linear_svm", "cdch_A_logistic", "cdch_D_conv", "rgb"):
        by_encoding = {r["encoding"]: float(r["auroc"]) for r in records
                       if r["model"] == name and r["task"] == "original_vs_ai"}
        right.plot(range(4), [by_encoding[e] for e in encodings], marker="o", markersize=4,
                   label=NAMES[name], color="#7d818d" if name == "rgb" else COLORS[family_for(name)])
    right.set_xticks(range(4), ["PNG", "JPEG 90", "JPEG 75", "Resize 96"], fontsize=9)
    right.set_ylim(0.44, 0.91)
    right.grid(axis="y", color="#ebecf2")
    right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.29), frameon=False, ncol=2, fontsize=8)
    right.set_title("Frozen scores after common transforms", loc="left", weight="bold")
    fig.tight_layout()
    fig.savefig(args.out / "02_representation_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    names = ["frequency", "rgb", "cdch_A_logistic", "cdch_B_linear_svm", "cdch_D_conv", "frequency_cdch"]
    stages = (("decode", "Decode", "#8191ab"), ("frequency", "FFT + descriptors", "#31649d"),
              ("rgb_preprocess", "RGB preprocessing", "#a3b7d1"), ("cnn", "RGB CNN", "#5043aa"),
              ("js", "Canonical JavaScript + oracle", "#168375"), ("ipc", "IPC + serialization", "#79ae9b"),
              ("feature", "CDCH descriptors", "#bc8c53"), ("classifier_other", "Other inference", "#ad594c"))
    bottom = np.zeros(len(names))
    for key, label, color in stages:
        x = np.array([float(bench[n][f"{key}_median_ms"]) for n in names])
        ax.barh(np.arange(len(names)), x, left=bottom, label=label, color=color, height=0.62)
        bottom += x
    ax.set_yticks(np.arange(len(names)), [NAMES[n] for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Sum of stage medians (ms)")
    ax.set_title("Where inference time goes", loc="left", weight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "03_latency_breakdown.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
