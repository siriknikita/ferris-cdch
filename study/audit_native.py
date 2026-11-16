"""Verify source parity and quantify secondary representation comparisons."""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from build_pairs import ROOT, sha
from experiment import bootstrap_weights, write_csv
from native_features import matrix, read_rgb, verify_js, worker
from native_models import native_score
from train_native import auc, load_set, train_entry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("pairs", "features", "run", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    verify_js()
    args.out.mkdir(parents=True)
    mf = args.pairs / "manifest.json"
    all_rows = json.loads(mf.read_text())
    if sha(mf) != sha(ROOT / "results/manifest.json"):
        raise ValueError("Unexpected pair manifest")
    with worker() as process:
        for row in [r for r in all_rows if r["variant"] != "lr"][:4]:
            rgb = read_rgb(args.pairs / row["path"])
            payload = {"samples": [{"height": 128, "width": 128, "pixels": rgb.ravel().tolist(),
                                    "center": [64, 64], "angle_step": 1}]}
            oracle = subprocess.check_output(["node", ROOT / "cdch/oracle.js"],
                                             input=json.dumps(payload).encode())
            cdf, _ = matrix(process, rgb)
            np.testing.assert_array_equal(cdf, json.loads(oracle)["results"][0]["histograms"])
    verification = {"canonical_matrix_oracle_exact": 4, "pair_manifest_sha256": sha(mf),
                    "canonical_sha256": sha(ROOT / "cdch/ip-1.js"),
                    "reproduced_pilot_outputs": {}}
    if args.baseline:
        baseline_names = ("rgb.pt", "statistical.joblib", "protocol.json", "holdout_metrics.csv",
                          "holdout_scores.csv", "paired_auroc_deltas.csv",
                          "within_source_ordering.csv", "holdout_disagreement.csv")
        for file in baseline_names:
            expected, actual = sha(ROOT / "results" / file), sha(args.baseline / file)
            verification["reproduced_pilot_outputs"][file] = {"expected": expected, "actual": actual,
                                                               "identical": expected == actual}
    (args.out / "VERIFICATION.json").write_text(json.dumps(verification, indent=2) + "\n")

    with (args.run / "holdout_predictions.csv").open() as stream:
        pred = list(csv.DictReader(stream))
    comparisons = [("cdch_B_logistic", "cdch_A_logistic"),
                   ("cdch_B_linear_svm", "cdch_A_linear_svm"),
                   ("cdch_B_linear_svm", "cdch_A_logistic"),
                   ("cdch_B_linear_svm", "frequency"),
                   ("cdch_A_rbf_svm", "cdch_A_logistic"),
                   ("cdch_C_logistic", "cdch_A_logistic")]
    output = []
    for encoding in dict.fromkeys(r["encoding"] for r in pred):
        score = {}
        for model in {m for pair in comparisons for m in pair}:
            score[model] = {(r["source_id"], r["variant"]): float(r["score"]) for r in pred
                            if r["encoding"] == encoding and r["model"] == model and r["variant"] != "bicubic"}
        items = sorted(score["cdch_A_logistic"])
        y = np.array([k[1] != "original" for k in items], dtype=int)
        boot = bootstrap_weights([k[0] for k in items], 4000, 20261116)
        for a, b in comparisons:
            lhs, rhs = (np.array([score[m][i] for i in items]) for m in (a, b))
            lo, hi = np.quantile(auc(y, lhs, boot) - auc(y, rhs, boot), [0.025, 0.975])
            output.append(dict(encoding=encoding, task="original_vs_ai", candidate=a, reference=b,
                               auroc_delta=float(auc(y, lhs)[0] - auc(y, rhs)[0]), ci_lo=lo, ci_hi=hi,
                               status="secondary, unadjusted"))
    write_csv(args.out / "secondary_deltas.csv", output)

    rows = [r for r in all_rows if r["variant"] != "lr"]
    dev = np.array([r["split"] != "test" for r in rows])
    groups = np.array([r["source_id"] for r in rows])
    variants = np.array([r["variant"] for r in rows])
    use = variants != "bicubic"
    y = np.isin(variants, ["swinir", "realesrgan"]).astype(int)
    features = load_set(args.features, rows, "png")
    result = []
    for seed in (17, 73, 101):
        train = np.isin(groups, np.random.default_rng(seed).permutation(np.unique(groups[dev]))[:60])
        for model in ("cdch_A_logistic", "cdch_B_logistic", "cdch_A_linear_svm", "cdch_B_linear_svm"):
            entry, cost = train_entry(model, features, y, train & use, 20261116)
            valid = dev & ~train & use
            score = native_score(entry, features[entry["feature"]][valid])
            result.append(dict(seed=seed, model=model, auroc=float(auc(y[valid], score)[0]), **cost))
    write_csv(args.out / "secondary_development_splits.csv", result)


if __name__ == "__main__":
    main()
