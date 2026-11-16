"""Fit CDCH models, freeze validation decisions, and evaluate matched controls."""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np
import torch
from build_pairs import ROOT, sha
from experiment import (TinyRGB, bootstrap_weights, choose_threshold, metrics,
                        rgb_scores, statistics, write_csv)
from native_features import read_rgb
from native_models import CANDIDATES, feature_key, fit_model, fused, native_score

HERE = Path(__file__).resolve().parent


def load_set(root, rows, encoding):
    with np.load(root / f"{encoding}.npz") as cache:
        if list(cache["paths"]) != [r["path"] for r in rows]:
            raise ValueError("Feature cache order differs from verified manifest")
        data = {k: cache[k] for k in ("A", "B", "C", "frequency")}
    for key, dim in (("A", 56), ("B", 98), ("C", 768), ("frequency", 24)):
        if data[key].shape != (len(rows), dim) or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid {key} cache for {encoding}")
    return data


def auc(y, scores, weights=None):
    return statistics(y, scores, weights=weights)["auroc"]


def train_entry(name, features, y, train, seed):
    key = feature_key(name)
    fitted, cost = fit_model(name, features[key][train], y[train], seed)
    return dict(name=name, feature=key, fit=fitted), cost


def select_weight(members, weights, reference, y):
    options = []
    for w in weights:
        rule = [(k, a, reference[k]) for k, a in zip(members, w) if a > 0]
        scores = fused(reference, rule)
        options.append((float(auc(y, scores)[0]), rule, scores))
    return max(options, key=lambda result: result[0])


def read_archived_scores(test_rows, encoding, name):
    table = ROOT / "results/holdout_scores.csv"
    with table.open(newline="") as stream:
        lookup = {(r["source_id"], r["variant"]): float(r["score"]) for r in csv.DictReader(stream)
                  if r["encoding"] == encoding and r["model"] == name}
    return np.array([lookup[(r["source_id"], r["variant"])] for r in test_rows])


def evaluate(scores, thresholds, variants, groups, encoding, seed, reps, comparisons):
    original, bicubic = variants == "original", variants == "bicubic"
    tasks = {
        "original_vs_ai": (original | ~bicubic, ~(original | bicubic)),
        "original_vs_bicubic": (original | bicubic, bicubic),
        "bicubic_vs_ai": (~original, ~(original | bicubic)),
    }
    for generator in ("swinir", "realesrgan"):
        chosen = variants == generator
        tasks[f"original_vs_{generator}"] = (original | chosen, chosen)
        tasks[f"bicubic_vs_{generator}"] = (bicubic | chosen, chosen)
    results, deltas = [], []
    for task, (include, target) in tasks.items():
        y = target[include].astype(int)
        weights = bootstrap_weights(groups[include], reps, seed)
        for name, values in scores.items():
            results.append(dict(encoding=encoding, task=task, model=name, groups=len(set(groups[include])),
                                **metrics(y, values[include], thresholds[name], weights)))
        for left, right in comparisons:
            a, b = scores[left][include], scores[right][include]
            diff = auc(y, a, weights) - auc(y, b, weights)
            lo, hi = np.quantile(diff, [0.025, 0.975])
            deltas.append(dict(encoding=encoding, task=task, candidate=left, reference=right,
                               auroc_delta=float(auc(y, a)[0] - auc(y, b)[0]), ci_lo=lo, ci_hi=hi))
    return results, deltas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pairs", "features", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = json.loads((HERE / "config/native.json").read_text())
    prior = ROOT / "results"
    source_manifest = args.pairs / "manifest.json"
    if sha(source_manifest) != sha(prior / "manifest.json"):
        raise ValueError("Manifest mismatch")
    rows = [r for r in json.loads(source_manifest.read_text()) if r["variant"] != "lr"]
    groups = np.array([r["source_id"] for r in rows])
    variants = np.array([r["variant"] for r in rows])
    splits = np.array([r["split"] for r in rows])
    y = (~np.isin(variants, ["original", "bicubic"])).astype(int)
    primary = variants != "bicubic"
    train, val, test = (splits == split for split in ("train", "val", "test"))
    primary_train, primary_val = train & primary, val & primary
    test_rows = [r for r in rows if r["split"] == "test"]
    args.out.mkdir(parents=True)
    models = args.out / "models"
    models.mkdir()
    raw_features = load_set(args.features, rows, "png")

    native, costs, validation = {}, {}, {}
    seed = config["seed"]
    diagnostic = "cdch_D_permuted_conv"
    for name in [*CANDIDATES, diagnostic]:
        native[name], costs[name] = train_entry(name, raw_features, y, primary_train, seed)
        validation[name] = native_score(native[name], raw_features[native[name]["feature"]][primary_val])
    selected = max(CANDIDATES, key=lambda model: auc(y[primary_val], validation[model])[0])

    fitted = joblib.load(prior / "statistical.joblib")
    archived = json.loads((prior / "protocol.json").read_text())
    cnn = TinyRGB()
    cnn.load_state_dict(torch.load(prior / "rgb.pt", weights_only=True, map_location="cpu"))
    png_val = [rows[i] for i in np.flatnonzero(primary_val)]
    validation["rgb"] = rgb_scores(cnn, np.stack([read_rgb(args.pairs / r["path"]) for r in png_val])
                                   .astype(np.float32) / 255)
    for name, (estimator, feature, direction) in fitted.items():
        x_freq, x_a = raw_features["frequency"][primary_val], raw_features["A"][primary_val]
        vectors = {"frequency": x_freq, "cdch": x_a, "frequency_cdch": np.c_[x_freq, x_a]}
        x = vectors[feature]
        validation[name if name != "cdch" else "archived_cdch"] = (
            estimator.decision_function(x) if hasattr(estimator, "named_steps")
            else direction * estimator.score_samples(x))
    for name in ["rgb", "frequency", "cdch", "frequency_cdch", "authentic_anomaly", "reconstruction_similarity"]:
        values = validation[name if name != "cdch" else "archived_cdch"]
        tolerance = 1e-5 if name == "rgb" else 1e-10
        np.testing.assert_allclose(values, archived["cdf"][name], atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(validation["cdch_A_logistic"], validation["archived_cdch"], atol=1e-10)

    from time import perf_counter, process_time
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    statistical = {}
    for name in ("frequency", "frequency_cdch", "authentic_anomaly", "reconstruction_similarity"):
        statistical[name] = (fitted[name][0], fitted[name][2])
    key = native[selected]["feature"]
    hybrid = "frequency_plus_selected_features"
    wall, cpu = perf_counter(), process_time()
    train_x = np.c_[raw_features["frequency"][primary_train], raw_features[key][primary_train]]
    statistical[hybrid] = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
                           .fit(train_x, y[primary_train]), 1)
    costs[hybrid] = {"fit_wall_s": perf_counter() - wall, "fit_cpu_s": process_time() - cpu}
    validation[hybrid] = statistical[hybrid][0].decision_function(
        np.c_[raw_features["frequency"][primary_val], raw_features[key][primary_val]])

    rules, fusion_validation = {}, {}
    ensemble = ["rgb", "frequency", "authentic_anomaly", "reconstruction_similarity"]
    for model, members in {
        "ensemble": ensemble, "frequency_cdch_score_fusion": ["frequency", selected],
        "rgb_cdch_score_fusion": ["rgb", selected], "ensemble_cdch_score_fusion": [*ensemble, selected],
    }.items():
        grid = [0] if model == "ensemble" else config["fusion_cdch_weights"]
        candidates = []
        for weight in grid:
            n = len(members) if model == "ensemble" else len(members) - 1
            candidates.append([(1 - weight) / n] * n + ([] if model == "ensemble" else [weight]))
        _, rules[model], fusion_validation[model] = select_weight(members, candidates, validation, y[primary_val])

    thresholds = {name: choose_threshold(y[primary_val], validation[name]) for name in [*native, hybrid]}
    for name in ["rgb", *statistical]:
        if name in archived["thresholds"]:
            thresholds[name] = archived["thresholds"][name]
    for name, scores in fusion_validation.items():
        thresholds[name] = (archived["thresholds"][name] if name == "ensemble"
                            else choose_threshold(y[primary_val], scores))
    source_models = {"native": native, "statistical": statistical, "rules": rules,
                     "thresholds": thresholds, "selected": selected}
    joblib.dump(source_models, args.out / "source_models.joblib")
    info = {}
    for name, entry in {**native, **statistical}.items():
        checkpoint = models / f"{name}.joblib"
        joblib.dump(entry, checkpoint)
        info[name] = {"model": name, "feature_dim": (int(raw_features[native[name]["feature"]].shape[1])
                      if name in native else (24 + (56 if name == "frequency_cdch" else raw_features[key].shape[1])
                                               if name in {"frequency_cdch", hybrid} else 24)),
                      "model_bytes": checkpoint.stat().st_size,
                      "fit_wall_s": costs.get(name, {}).get("fit_wall_s", ""),
                      "fit_cpu_s": costs.get(name, {}).get("fit_cpu_s", ""),
                      "converged": costs.get(name, {}).get("converged", ""),
                      "validation_auroc": float(auc(y[primary_val], validation[name])[0])}
    for name, rule in rules.items():
        rule_file = models / f"{name}_rule.joblib"
        joblib.dump(rule, rule_file)
        leaves = {k for k, _, _ in rule}
        dims = (24 if leaves & set(statistical) else 0) + (info[selected]["feature_dim"] if selected in leaves else 0)
        info[name] = {"model": name, "feature_dim": dims + (128 * 128 * 3 if "rgb" in leaves else 0),
                      "model_bytes": rule_file.stat().st_size + sum(
                          (prior / "rgb.pt").stat().st_size if k == "rgb" else info[k]["model_bytes"] for k in leaves),
                      "fit_wall_s": "", "fit_cpu_s": "", "converged": "",
                      "validation_auroc": float(auc(y[primary_val], fusion_validation[name])[0])}
    info["rgb"] = {"model": "rgb", "feature_dim": 128 * 128 * 3, "model_bytes": (prior / "rgb.pt").stat().st_size,
                   "fit_wall_s": "", "fit_cpu_s": "", "converged": "", "validation_auroc": float(auc(y[primary_val], validation["rgb"])[0])}
    write_csv(args.out / "model_inventory.csv", list(info.values()))
    frozen = {"selected": selected, "thresholds": thresholds,
              "fusion_weights": {name: {k: w for k, w, _ in rule} for name, rule in rules.items()},
              "manifest_sha256": sha(source_manifest), "config_sha256": sha(HERE / "config/native.json")}
    (args.out / "SOURCE_DECISIONS.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print(f"Source validation selected {selected}", flush=True)

    comparison_specs = [(selected, "cdch_A_logistic"), (selected, "frequency"),
                        ("cdch_D_conv", diagnostic), (hybrid, "frequency"),
                        ("frequency_cdch", "frequency"), ("frequency_cdch_score_fusion", "frequency"),
                        ("rgb_cdch_score_fusion", "rgb"), ("ensemble_cdch_score_fusion", "ensemble")]
    comparison_specs = list(dict.fromkeys(comparison_specs))
    result, deltas, predictions, ordering, crossover = [], [], [], [], []
    for encoding in config["holdout_encodings"]:
        features = ({k: v[test] for k, v in raw_features.items()} if encoding == "png"
                    else load_set(args.features, test_rows, encoding))
        scores = {name: native_score(entry, features[entry["feature"]]) for name, entry in native.items()}
        x = {"frequency_cdch": np.c_[features["frequency"], features["A"]],
             hybrid: np.c_[features["frequency"], features[key]]}
        scores.update({name: (fit.decision_function(x.get(name, features["frequency"]))
                              if hasattr(fit, "named_steps") else direction * fit.score_samples(features["frequency"]))
                       for name, (fit, direction) in statistical.items()})
        inputs = np.stack([read_rgb(args.pairs / r["path"], encoding) for r in test_rows]).astype(np.float32) / 255
        scores["rgb"] = rgb_scores(cnn, inputs)
        if encoding != "resize96":
            # Retain the archived four-thread RGB prediction for bit-level baseline continuity.
            archived_rgb = read_archived_scores(test_rows, encoding, "rgb")
            np.testing.assert_allclose(scores["rgb"], archived_rgb, atol=1e-5, rtol=1e-5)
            scores["rgb"] = archived_rgb
            for name in ("frequency", "frequency_cdch", "cdch_A_logistic"):
                old_key = "cdch" if name == "cdch_A_logistic" else name
                np.testing.assert_allclose(scores[name], read_archived_scores(test_rows, encoding, old_key), atol=1e-10)
        scores.update({n: fused(scores, rule) for n, rule in rules.items()})
        comparison, paired = evaluate(scores, thresholds, variants[test], groups[test], encoding, seed,
                                      config["bootstrap_replicates"], comparison_specs)
        result.extend(comparison)
        deltas.extend(paired)
        for name, values in scores.items():
            predictions.extend(dict(encoding=encoding, model=name, source_id=r["source_id"],
                                    variant=r["variant"], score=float(v)) for r, v in zip(test_rows, values))
            for negative in ("original", "bicubic"):
                for positive in ("swinir", "realesrgan"):
                    diff = values[variants[test] == positive] - values[variants[test] == negative]
                    fraction = (diff > 0).astype(float) + 0.5 * (diff == 0)
                    draws = bootstrap_weights(groups[test][variants[test] == positive],
                                              config["bootstrap_replicates"], seed)
                    lo, hi = np.quantile(draws @ fraction / len(diff), [0.025, 0.975])
                    ordering.append(dict(encoding=encoding, model=name, positive=positive, negative=negative,
                                         fraction=fraction.mean(), ci_lo=lo, ci_hi=hi))

        for trained in config["generators"]:
            train_mask = train & np.isin(variants, ["original", trained])
            valid_mask = val & np.isin(variants, ["original", trained])
            frequency_name = f"frequency_{trained}_only"
            for name in dict.fromkeys((selected, "cdch_A_logistic", "cdch_B_logistic", "cdch_B_linear_svm", "frequency")):
                if name == "frequency":
                    predictor = fitted[frequency_name][0].decision_function
                    evaluation_x = features["frequency"]
                    cutoff = archived["thresholds"][frequency_name]
                else:
                    entry, _ = train_entry(name, raw_features, y, train_mask, seed)
                    key_x = entry["feature"]
                    evaluation_x = features[key_x]
                    predictor = lambda x: native_score(entry, x)
                    cutoff = choose_threshold(y[valid_mask], predictor(raw_features[key_x][valid_mask]))
                values = predictor(evaluation_x)
                for tested in config["generators"]:
                    for control in ("original", "bicubic"):
                        mask = np.isin(variants[test], [control, tested])
                        label = (variants[test][mask] == tested).astype(int)
                        weights = bootstrap_weights(groups[test][mask], config["bootstrap_replicates"], seed)
                        crossover.append(dict(encoding=encoding, model=name, train_generator=trained,
                                              test_generator=tested, negative=control,
                                              **metrics(label, values[mask], cutoff, weights)))

    stability = []
    for repetition in config["training_repeat_seeds"]:
        for name in dict.fromkeys([selected, "cdch_A_logistic", "cdch_D_conv", "cdch_A_forest", "cdch_A_mlp"]):
            entry, cost = train_entry(name, raw_features, y, primary_train, repetition)
            for split_name, mask in (("validation", primary_val), ("holdout", test & primary)):
                stability.append(dict(check="training_seed", seed=repetition, evaluation=split_name,
                                      model=name, auroc=float(auc(y[mask], native_score(
                                          entry, raw_features[entry["feature"]][mask]))[0]), **cost))
    development_groups = np.unique(groups[~test])
    for repetition in config["development_split_seeds"]:
        rng = np.random.default_rng(repetition)
        subtrain = np.isin(groups, rng.permutation(development_groups)[:60])
        subval = ~test & ~subtrain & primary
        for name in dict.fromkeys([selected, "cdch_A_logistic", "cdch_D_conv", "frequency"]):
            if name == "frequency":
                model, cost = fit_model("cdch_A_logistic", raw_features["frequency"][subtrain & primary],
                                        y[subtrain & primary], seed)
                scores_sub = model.decision_function(raw_features["frequency"][subval])
            else:
                entry, cost = train_entry(name, raw_features, y, subtrain & primary, seed)
                scores_sub = native_score(entry, raw_features[entry["feature"]][subval])
            stability.append(dict(check="development_split", seed=repetition, evaluation="validation",
                                  model=name, auroc=float(auc(y[subval], scores_sub)[0]), **cost))

    for filename, collection in (("CLASSIFIER_COMPARISON", result), ("paired_deltas", deltas),
                                 ("holdout_predictions", predictions), ("within_source_ordering", ordering),
                                 ("cross_upscaler", crossover), ("stability", stability)):
        write_csv(args.out / f"{filename}.csv", collection)


if __name__ == "__main__":
    main()
