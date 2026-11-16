"""Independently verify point metrics and check packaged benchmark coverage."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
import json


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path, default=Path.cwd(), nargs="?")
    root = p.parse_args().root
    all_predictions = defaultdict(list)
    for row in rows(root / "results/holdout_predictions.csv"):
        all_predictions[row["encoding"], row["model"]].append(row)
    thresholds = json.loads((root / "source_run/SOURCE_DECISIONS.json").read_text())["thresholds"]
    computed = rows(root / "CLASSIFIER_COMPARISON.csv")
    for row in computed:
        key = row["encoding"], row["model"]
        negative, positive = row["task"].split("_vs_")
        candidates = all_predictions[key]
        positive_set = {"swinir", "realesrgan"} if positive == "ai" else {positive}
        selected = [r for r in candidates if r["variant"] in positive_set | {negative}]
        y = np.array([int(r["variant"] in positive_set) for r in selected])
        scores = np.array([float(r["score"]) for r in selected])
        prediction = scores >= thresholds[row["model"]]
        expected = [roc_auc_score(y, scores), average_precision_score(y, scores),
                    balanced_accuracy_score(y, prediction)]
        np.testing.assert_allclose(expected,
                                  [float(row[k]) for k in ("auroc", "auprc_ap", "balanced_accuracy")], atol=1e-12)
        if len({r["source_id"] for r in selected}) != int(row["groups"]):
            raise ValueError("Wrong independent group count")
    benchmark = rows(root / "EFFICIENCY_BENCHMARKS.csv")
    if {r["model"] for r in benchmark} != set(thresholds):
        raise ValueError("Timing coverage differs from source fits")
    counts = {len(json.loads((root / f"results/timing_isolated/{r['model']}.json").read_text())["raw_ns"])
              for r in benchmark}
    if counts != {60}:
        raise ValueError("Unexpected number of timing trials")
    print(f"PASS: {len(computed)} independent metric checks and {len(benchmark)} benchmark configurations")


if __name__ == "__main__":
    main()
