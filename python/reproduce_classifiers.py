"""Phase G: reproduce archived CDCH features, classifier scores, labels and AUROC.

For every archived evaluation image (400 PNG crops; 80 test crops under jpeg90, jpeg75 and
resize96) the CDCH representations are computed four ways:

  js      canonical ip-1.js (archived Node worker) -> archived NumPy representations
  rust_m  Rust canonical matrix (cdch_ferris.cdch) -> archived NumPy representations
  rust    Rust fused extraction (cdch_ferris.features_multi)
  rust_n  Rust representations of the JS matrix (cdch_ferris.features_from_matrix)

All archived checkpoints are scored with each feature set through one shared, unmodified
classifier code path. Results are compared with each other and, where the regenerated pixels
are equivalent to the archived ones, with the archived feature cache and predictions.

    python reproduce_classifiers.py --pairs ../../external/pairs --out ../reports/classifier_compatibility
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import cdch_ferris
from ferris_pipeline import (ROOT, RGB_PATH, STUDY, FerrisPipeline, archived_center, decisions,
                             load_models, model_names)
from experiment import batch_input, frequency_features, statistics  # archived
from native_features import matrix, read_rgb, representation, verify_js, worker  # archived
from native_models import fused, native_score, score_rgb  # archived

KINDS = ("A", "B", "C")
BACKENDS = ("js", "rust_m", "rust", "rust_n")
ENCODINGS = ("png", "jpeg90", "jpeg75", "resize96")


def numpy_reps(cdf):
    a = representation(cdf, "A")
    return {"A": np.asarray(a), "B": representation(cdf, "B", a), "C": representation(cdf, "C")}


def score_batch(pipe, feats, frequency, rgb_scores):
    """The archived classifier section of Pipeline.predict, vectorized over images."""
    predictions = {n: native_score(entry, feats[entry["feature"]], pipe.networks[n])
                   for n, entry in pipe.native.items()}
    predictions["rgb"] = rgb_scores
    for n, (fit, direction) in pipe.statistical.items():
        vectors = frequency
        if n in {"frequency_cdch", "frequency_plus_selected_features"}:
            vectors = np.c_[frequency, feats["A" if n == "frequency_cdch" else pipe.hybrid_key]]
        predictions[n] = (fit.decision_function(vectors) if hasattr(fit, "named_steps")
                          else direction * fit.score_samples(vectors))
    for n, rule in pipe.rules.items():
        predictions[n] = fused(predictions, rule)
    return {n: np.asarray(predictions[n], dtype=np.float64) for n in pipe.outputs}


def tasks(variants):
    original, bicubic = variants == "original", variants == "bicubic"
    out = {"original_vs_ai": (original | ~bicubic, ~(original | bicubic)),
           "original_vs_bicubic": (original | bicubic, bicubic),
           "bicubic_vs_ai": (~original, ~(original | bicubic))}
    for g in ("swinir", "realesrgan"):
        chosen = variants == g
        out[f"original_vs_{g}"] = (original | chosen, chosen)
        out[f"bicubic_vs_{g}"] = (bicubic | chosen, chosen)
    return out


def agreement(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    diff = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    rel = np.divide(diff, scale, out=np.zeros_like(diff), where=scale > 0)
    return {"values": int(a.size), "bitwise_equal": int(np.sum(a.view(np.uint64) == b.view(np.uint64))),
            "max_abs": float(diff.max(initial=0)), "max_rel": float(rel.max(initial=0)),
            "allclose_1e-12": bool(np.allclose(a, b, rtol=1e-12, atol=1e-12))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    verify_js()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((ROOT / "results/manifest.json").read_text())
    rows = [r for r in manifest if r["variant"] != "lr"]
    names = model_names()
    pipe = FerrisPipeline(load_models(names), RGB_PATH, names, backend="rust")
    thresholds = decisions()["thresholds"]

    archived_pred = defaultdict(dict)
    with (STUDY / "results/holdout_predictions.csv").open(newline="") as stream:
        for r in csv.DictReader(stream):
            archived_pred[(r["encoding"], r["model"])][(r["source_id"], r["variant"])] = float(r["score"])
    archived_auc = {}
    with (STUDY / "CLASSIFIER_COMPARISON.csv").open(newline="") as stream:
        for r in csv.DictReader(stream):
            archived_auc[(r["encoding"], r["task"], r["model"])] = float(r["auroc"])

    # Archived pixel evidence: the original crops carry pixel digests in inventory.json.
    inv_arch = {s["source_id"]: s for s in json.loads((ROOT / "results/inventory.json").read_text())["sources"]}
    inv_new = {s["source_id"]: s for s in json.loads((args.pairs / "inventory.json").read_text())["sources"]}

    feature_rows, score_rows, auc_rows, image_rows, score_image_rows = [], [], [], [], []
    summary = {"encodings": {}}
    with worker() as process:
        for encoding in ENCODINGS:
            selected = rows if encoding == "png" else [r for r in rows if r["split"] == "test"]
            with np.load(STUDY / "feature_cache" / f"{encoding}.npz") as cache:
                if list(cache["paths"]) != [r["path"] for r in selected]:
                    raise ValueError("feature cache order differs from manifest")
                cached = {k: cache[k] for k in ("A", "B", "C", "frequency")}
            feats = {b: {k: [] for k in KINDS} for b in BACKENDS}
            frequency, rgbs, matrix_equal = [], [], []
            for row in selected:
                rgb = read_rgb(args.pairs / row["path"], encoding)
                cdf_js, _ = matrix(process, rgb)
                cdf_rs = cdch_ferris.cdch(rgb, center=archived_center(rgb))
                matrix_equal.append(cdf_js.tobytes() == cdf_rs.tobytes())
                per = {"js": numpy_reps(cdf_js), "rust_m": numpy_reps(cdf_rs),
                       "rust": cdch_ferris.features_multi(rgb, list(KINDS), center=archived_center(rgb)),
                       "rust_n": {k: cdch_ferris.features_from_matrix(cdf_js, k) for k in KINDS}}
                for b in BACKENDS:
                    for k in KINDS:
                        feats[b][k].append(np.asarray(per[b][k], np.float64))
                frequency.append(frequency_features(rgb.astype(np.float32) / 255))
                rgbs.append(rgb)
            feats = {b: {k: np.stack(v) for k, v in d.items()} for b, d in feats.items()}
            frequency = np.asarray(frequency)
            n = len(selected)

            # Archive equivalence per image: canonical-JS A, B and C and the frequency features all
            # reproduce the archived cache (A/B/C at the archived extract_native.py tolerance,
            # rtol = atol = 1e-12). A alone is insufficient: C samples twice as many intensity bins.
            equivalent = np.array([
                all(np.allclose(feats["js"][k][i], cached[k][i], rtol=1e-12, atol=1e-12) for k in KINDS)
                and np.allclose(frequency[i], cached["frequency"][i], rtol=1e-6, atol=1e-7)
                for i in range(n)])
            for i, row in enumerate(selected):
                pixel_digest = (inv_arch[row["source_id"]]["reference_pixel_sha256"]
                                == inv_new[row["source_id"]]["reference_pixel_sha256"]) if row["variant"] == "original" else None
                image_rows.append(dict(encoding=encoding, path=row["path"], split=row["split"], variant=row["variant"],
                                       matrix_bitwise_js_vs_rust=matrix_equal[i], archive_equivalent=bool(equivalent[i]),
                                       original_pixel_digest_match=pixel_digest))

            # Feature agreement.
            for k in KINDS:
                comps = {"rust_vs_js": ("rust", "js"), "rust_m_vs_js": ("rust_m", "js"),
                         "rust_n_vs_js": ("rust_n", "js"), "rust_vs_rust_n": ("rust", "rust_n")}
                for label, (x, y) in comps.items():
                    feature_rows.append(dict(encoding=encoding, kind=k, comparison=label, images=n,
                                             **agreement(feats[x][k], feats[y][k])))
                if equivalent.any():
                    for b in ("js", "rust"):
                        feature_rows.append(dict(encoding=encoding, kind=k, comparison=f"{b}_vs_archived_cache",
                                                 images=int(equivalent.sum()),
                                                 **agreement(feats[b][k][equivalent], cached[k][equivalent])))
                # Per-column maximum relative difference rust vs js (identifies which features differ).
                diff = np.abs(feats["rust"][k] - feats["js"][k])
                scale = np.maximum(np.abs(feats["rust"][k]), np.abs(feats["js"][k]))
                rel = np.divide(diff, scale, out=np.zeros_like(diff), where=scale > 0).max(axis=0)
                summary.setdefault("per_feature_max_rel_rust_vs_js", {}).setdefault(encoding, {})[k] = rel.tolist()

            # Scores.
            rgb_scores = score_rgb(pipe.rgb, batch_input(np.stack(rgbs).astype(np.float32) / 255))
            scores = {b: score_batch(pipe, feats[b], frequency, rgb_scores) for b in BACKENDS}
            variants = np.array([r["variant"] for r in selected])
            test = np.array([r["split"] == "test" for r in selected])
            for name in names:
                ref = scores["js"][name]
                key = (encoding, name)
                for i, r in enumerate(selected):
                    arch = archived_pred.get(key, {}).get((r["source_id"], r["variant"])) if r["split"] == "test" else None
                    score_image_rows.append(dict(encoding=encoding, model=name, path=r["path"], split=r["split"],
                                                 variant=r["variant"], archive_equivalent=bool(equivalent[i]),
                                                 threshold=thresholds.get(name), archived=arch,
                                                 **{b: repr(float(scores[b][name][i])) for b in BACKENDS}))
                for b in ("rust_m", "rust", "rust_n"):
                    s = scores[b][name]
                    t = thresholds.get(name)
                    flip = ((s >= t) != (ref >= t)) if t is not None else np.zeros(n, bool)
                    score_rows.append(dict(encoding=encoding, model=name, comparison=f"{b}_vs_js", images=n,
                                           max_abs_score_diff=float(np.max(np.abs(s - ref))),
                                           test_max_abs_score_diff=float(np.max(np.abs(s - ref)[test])),
                                           bitwise_equal=int(np.sum(s.view(np.uint64) == ref.view(np.uint64))),
                                           label_flips=int(flip.sum()) if t is not None else None,
                                           test_label_flips=int((flip & test).sum()) if t is not None else None))
                key = (encoding, name)
                if key in archived_pred:
                    mask = test & equivalent
                    if mask.any():
                        arch = np.array([archived_pred[key][(r["source_id"], r["variant"])]
                                         for r, m in zip(selected, mask) if m])
                        for b in ("js", "rust"):
                            t = thresholds.get(name)
                            d = np.abs(scores[b][name][mask] - arch)
                            score_rows.append(dict(encoding=encoding, model=name, comparison=f"{b}_vs_archived_predictions",
                                                   images=int(mask.sum()),
                                                   max_abs_score_diff=float(np.max(d)),
                                                   test_max_abs_score_diff=float(np.max(d)),
                                                   bitwise_equal=int(np.sum(scores[b][name][mask].view(np.uint64) == arch.view(np.uint64))),
                                                   label_flips=None,
                                                   test_label_flips=int(np.sum((scores[b][name][mask] >= t) != (arch >= t))) if t is not None else None))
            # AUROC on the archived holdout tasks.
            tv = variants[test]
            for task, (include, target) in tasks(tv).items():
                y = target[include].astype(int)
                for name in names:
                    row = dict(encoding=encoding, task=task, model=name, n=int(include.sum()))
                    for b in BACKENDS:
                        row[f"auroc_{b}"] = float(statistics(y, scores[b][name][test][include])["auroc"][0])
                    row["auroc_archived"] = archived_auc.get((encoding, task, name))
                    auc_rows.append(row)
            summary["encodings"][encoding] = {
                "images": n, "matrix_bitwise_js_vs_rust": int(np.sum(matrix_equal)),
                "archive_equivalent_images": int(equivalent.sum()),
                "archive_equivalent_by_variant": {v: int(np.sum(equivalent & (variants == v))) for v in np.unique(variants)},
                "test_images": int(test.sum()),
            }
            print(f"{encoding}: {n} images, matrices bitwise-equal {sum(matrix_equal)}/{n}, "
                  f"archive-equivalent {int(equivalent.sum())}/{n}", flush=True)

    def write(name, data):
        with (args.out / name).open("w", newline="") as stream:
            w = csv.DictWriter(stream, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    write("feature_agreement.csv", feature_rows)
    write("score_agreement.csv", score_rows)
    write("auroc_comparison.csv", auc_rows)
    write("image_equivalence.csv", image_rows)
    write("per_image_scores.csv", score_image_rows)
    summary["environment"] = {"cdch_ferris": cdch_ferris.build_info(), "numpy": np.__version__,
                              "torch": torch.__version__}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
