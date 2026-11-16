"""Repeat key pipeline timings in random order on identical images."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from build_pairs import ROOT, sha
from experiment import write_csv
from native_features import verify_js, worker
from native_pipeline import load_deploy

KEYS = ("frequency", "rgb", "ensemble", "cdch_A_logistic", "cdch_B_linear_svm", "frequency_cdch",
        "frequency_cdch_score_fusion", "rgb_cdch_score_fusion", "ensemble_cdch_score_fusion")
PAIRS = (("cdch_A_logistic", "frequency"), ("cdch_B_linear_svm", "frequency"),
         ("cdch_B_linear_svm", "cdch_A_logistic"), ("frequency_cdch", "frequency"),
         ("frequency_cdch_score_fusion", "frequency"), ("rgb_cdch_score_fusion", "rgb"),
         ("ensemble_cdch_score_fusion", "ensemble"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pairs", "run", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    verify_js()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest = args.pairs / "manifest.json"
    if sha(manifest) != sha(ROOT / "results/manifest.json"):
        raise ValueError("Wrong timing manifest")
    rows = [r for r in json.loads(manifest.read_text())
            if r["split"] == "test" and r["variant"] != "lr"]
    # Two treatments from every independent source, in randomized order.
    groups = sorted({r["source_id"] for r in rows})
    rng = np.random.default_rng(135)
    picks = [row for group in groups for row in rng.choice([r for r in rows if r["source_id"] == group],
                                                          2, replace=False)]
    rng.shuffle(picks)
    decision = json.loads((args.run / "SOURCE_DECISIONS.json").read_text())
    model = decision["selected"]
    key = "C" if model.startswith("cdch_D_") else model.split("_")[1]
    pipelines = {name: load_deploy(name, args.run, ROOT / "results/rgb.pt", model, key) for name in KEYS}
    trials = []
    with worker() as process:
        for i, row in enumerate([*picks[:6], *picks]):
            path = args.pairs / row["path"]
            if sha(path) != row["sha256"]:
                raise ValueError("Wrong timing input")
            times = {}
            for name in rng.permutation(KEYS):
                _, clock = pipelines[name].predict(path, process)
                times[name] = clock["total"] / 1e6
            if i >= 6:
                trials.append(dict(source_id=row["source_id"], variant=row["variant"], **times))
    args.out.mkdir(parents=True)
    write_csv(args.out / "timings_ms.csv", trials)
    per_group = {g: [i for i, r in enumerate(trials) if r["source_id"] == g] for g in groups}
    draw = np.array(list(per_group.values()))[rng.integers(len(groups), size=(4000, len(groups)))].reshape(4000, -1)
    estimates = []
    for left, right in PAIRS:
        diff = np.array([r[left] - r[right] for r in trials])
        lo, hi = np.quantile(np.median(diff[draw], axis=1), [0.025, 0.975])
        estimates.append(dict(candidate=left, reference=right, trials=len(trials),
                              paired_median_delta_ms=float(np.median(diff)), ci_lo_ms=lo, ci_hi_ms=hi))
    write_csv(args.out / "paired_latency_delta.csv", estimates)


if __name__ == "__main__":
    main()
