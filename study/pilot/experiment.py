"""Train SR-only classifiers and evaluate matched controls on held-out sources."""
import argparse
import csv
import io
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from build_pairs import sha


class TinyRGB(nn.Sequential):
    def __init__(self):
        layers = []
        for incoming, outgoing in [(3, 16), (16, 32), (32, 64)]:
            layers.extend([nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                           nn.BatchNorm2d(outgoing), nn.SiLU(), nn.AvgPool2d(2)])
        super().__init__(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1))


def pixels(path, quality=None):
    with Image.open(path) as source:
        if source.mode != "RGB" or source.format != "PNG":
            raise ValueError(f"Expected a normalized RGB PNG at {path}")
        if quality is None:
            return np.asarray(source, dtype=np.float32) / 255
        buffer = io.BytesIO()
        source.save(buffer, format="JPEG", quality=quality, subsampling=2, optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as jpeg:
            return np.asarray(jpeg, dtype=np.float32) / 255


def logmean(values):
    return np.log10(np.mean(values) + 1e-14)


def frequency_features(rgb):
    y = rgb @ np.array([0.2126, 0.7152, 0.0722])
    window = np.hanning(y.shape[0])[:, None] * np.hanning(y.shape[1])
    power = np.abs(np.fft.fft2((y - y.mean()) * window)) ** 2
    power /= max(power.sum(), 1e-14)
    fy, fx = np.meshgrid(np.fft.fftfreq(y.shape[0]), np.fft.fftfreq(y.shape[1]), indexing="ij")
    radius = np.hypot(fx, fy)
    bands = np.linspace(0, np.sqrt(0.5) + 1e-6, 17)
    out = [logmean(power[(radius >= left) & (radius < right)])
           for left, right in zip(bands, bands[1:])]
    for frequencies in (abs(fy), abs(fx)):
        for center in (0.125, 0.25, 0.375, 0.5):
            out.append(logmean(power[abs(frequencies - center) < 0.012]))
    return out


def batch_input(images):
    return torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()


def rgb_scores(model, images):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(batch).flatten() for batch in batch_input(images).split(32)]).numpy()


def statistics(y, scores, threshold=None, weights=None):
    """Weighted AUROC and stepwise AUPRC with exact tie handling."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=np.float64)
    weights = np.ones((1, len(y))) if weights is None else np.atleast_2d(weights).astype(float, copy=False)
    order = np.argsort(-scores, kind="stable")
    y, scores, weights = y[order], scores[order], weights[:, order]
    start = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]
    pos = np.add.reduceat(weights * y, start, axis=1)
    neg = np.add.reduceat(weights * (1 - y), start, axis=1)
    tp, fp = pos.cumsum(axis=1), neg.cumsum(axis=1)
    p, n = tp[:, -1], fp[:, -1]
    wins = (neg * (tp - 0.5 * pos)).sum(axis=1)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = {"auroc": wins / (p * n), "auprc_ap": (precision * pos).sum(axis=1) / p}
        if threshold is not None:
            above = scores >= threshold
            tpos = (weights[:, above] * y[above]).sum(axis=1)
            fpos = (weights[:, above] * (1 - y[above])).sum(axis=1)
            result["balanced_accuracy"] = 0.5 * (tpos / p + 1 - fpos / n)
    return result


def bootstrap_weights(groups, replicates, seed):
    labels, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    group_weights = rng.multinomial(len(labels), np.full(len(labels), 1 / len(labels)), size=replicates)
    return group_weights[:, inverse]


def metrics(y, scores, threshold, weights):
    estimate = statistics(y, scores, threshold)
    batches = [weights] if isinstance(weights, np.ndarray) else weights
    draws = [statistics(y, scores, threshold, batch) for batch in batches]
    bootstrap = {key: np.concatenate([draw[key] for draw in draws]) for key in estimate}
    result = {}
    for key, value in estimate.items():
        valid = bootstrap[key][np.isfinite(bootstrap[key])]
        if not len(valid):
            raise ValueError("Bootstrap has no draws with both classes")
        result[key] = float(value[0])
        result[f"{key}_lo"], result[f"{key}_hi"] = np.quantile(valid, [0.025, 0.975]).tolist()
        result[f"{key}_bootstrap_valid"] = len(valid)
    y, pred = np.asarray(y), np.asarray(scores, dtype=np.float64) >= threshold
    for label, truth, positive in [("tn", 0, False), ("fp", 0, True),
                                   ("fn", 1, False), ("tp", 1, True)]:
        result[label] = int(np.count_nonzero((y == truth) & (pred == positive)))
    return {"n": len(y), "positive_prevalence": float(y.mean()), **result}


def choose_threshold(y, scores):
    # In a tie, prefer the highest threshold (higher specificity).
    scores = np.asarray(scores, dtype=np.float64)
    grid = np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores)[::-1]]
    values = [statistics(y, scores, t)["balanced_accuracy"][0] for t in grid]
    return float(grid[np.argmax(np.round(values, 12))])


def train_rgb(images, y, val_images, val_y, config):
    torch.manual_seed(config["seed"])
    model = TinyRGB()
    dataset = TensorDataset(batch_input(images), torch.from_numpy(y.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(config["seed"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()
    best_auc, best_state, best_epoch = -1.0, None, None
    for epoch in range(1, config["cnn_epochs"] + 1):
        model.train()
        for batch, label in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch).flatten(), label)
            loss.backward()
            optimizer.step()
        auc = statistics(val_y, rgb_scores(model, val_images))["auroc"][0]
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, {"rgb_epoch": best_epoch, "rgb_val_auroc": float(best_auc)}


def cdf(scores, reference):
    sorted_reference = np.sort(reference)
    left = np.searchsorted(sorted_reference, scores, side="left")
    right = np.searchsorted(sorted_reference, scores, side="right")
    return (left + right) / (2 * len(reference))


def features(images, cdch=None):
    result = {"rgb": images, "frequency": np.asarray([frequency_features(image) for image in images])}
    if cdch is not None:
        result["cdch"] = cdch
        result["frequency_cdch"] = np.column_stack([result["frequency"], cdch])
    return result


def raw_scores(rgb, fitted, inputs):
    result = {"rgb": rgb_scores(rgb, inputs["rgb"])}
    for name, (model, feature, direction) in fitted.items():
        if isinstance(model, IsolationForest):
            result[name] = direction * model.score_samples(inputs[feature])
        else:
            result[name] = model.decision_function(inputs[feature])
    return result


def scores(rgb, fitted, protocol, inputs):
    raw = raw_scores(rgb, fitted, inputs)
    for name, members in protocol["fusion"].items():
        raw[name] = np.mean([cdf(raw[k], protocol["cdf"][k]) for k in members], axis=0)
    return raw


def read_cdch(path, rows, expected_provenance=None, quality=None):
    if path is None:
        if expected_provenance is not None:
            raise ValueError("Frozen CDCH model requires a feature sidecar")
        return None, None
    from cdch_features import CANONICAL, METHOD
    from build_pairs import ROOT

    payload = json.loads(path.read_text())
    provenance = {key: payload[key] for key in ("canonical_sha256", "adapter_sha256",
                                               "descriptor_code_sha256", "method", "node_version")}
    if (provenance["canonical_sha256"] != CANONICAL or provenance["method"] != METHOD
            or provenance["descriptor_code_sha256"] != sha(ROOT / "cdch_features.py")
            or provenance["adapter_sha256"] != {name: sha(ROOT / "cdch" / name)
                                               for name in ("worker.js", "oracle.js")}
            or payload["jpeg_quality"] != quality):
        raise ValueError("CDCH generator, descriptor or JPEG provenance mismatch")
    if expected_provenance is not None and provenance != expected_provenance:
        raise ValueError("CDCH sidecar generator differs from frozen provenance")
    if payload["image_sha256"] != {row["path"]: row["sha256"] for row in rows}:
        raise ValueError("CDCH sidecar is for a different image manifest")
    data = payload["features"]
    values = np.asarray(list(data.values()), dtype=float)
    if set(data) != set(payload["image_sha256"]) or values.shape != (len(data), 56) or not np.isfinite(values).all():
        raise ValueError("CDCH sidecar contains invalid vectors")
    return data, provenance


def read_inputs(rows, directory, cdch=None, quality=None):
    sidecar = None if cdch is None else np.asarray([cdch[r["path"]] for r in rows], dtype=float)
    return features(np.stack([pixels(directory / r["path"], quality) for r in rows]), sidecar)


def setup_models(rows, directory, output, config, cdch, cdch_provenance, manifest_hash):
    positions = {}
    y = {}
    data = {}
    for split in ("train", "val"):
        positions[split] = [row for row in rows if row["split"] == split
                            and row["variant"] != "bicubic"]
        y[split] = np.array([int(row["variant"] != "original") for row in positions[split]])
        data[split] = read_inputs(positions[split], directory, cdch)

    rgb, details = train_rgb(data["train"]["rgb"], y["train"], data["val"]["rgb"], y["val"], config)
    fitted = {}
    for name, matrix in data["train"].items():
        if name != "rgb":
            fitted[name] = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
                            .fit(matrix, y["train"]), name, 1)
    frequency = data["train"]["frequency"]
    for label, name, direction in [(0, "authentic_anomaly", -1), (1, "reconstruction_similarity", 1)]:
        fitted[name] = (IsolationForest(n_estimators=128, random_state=config["seed"])
                        .fit(frequency[y["train"] == label]), "frequency", direction)

    for generator in config["generator_ids"]:
        provenance = np.array([row["variant"] for row in positions["train"]])
        include = (provenance == generator) | (provenance == "original")
        fitted[f"frequency_{generator}_only"] = (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            .fit(frequency[include], y["train"][include]), "frequency", 1,
        )

    fusion = {"rgb_frequency": ["rgb", "frequency"],
              "ensemble": ["rgb", "frequency", "authentic_anomaly", "reconstruction_similarity"]}
    if cdch is not None:
        fusion["rgb_frequency_cdch"] = ["rgb", "frequency", "cdch"]
        fusion["ensemble_cdch"] = fusion["ensemble"] + ["cdch"]
    raw = raw_scores(rgb, fitted, data["val"])
    protocol = {"input_px": config["input_px"], "fusion": fusion,
                "training_manifest_sha256": manifest_hash,
                "cdf": {name: values.tolist() for name, values in raw.items()},
                "mapping": {"original": 0, "real": 0, "ai_reconstruction": 1, "fake": 1},
                "video_aggregation": "mean of all fixed sampled frame scores",
                "cdch_provenance": cdch_provenance, "config": config,
                "real_fake_score_direction": "higher_is_fake", **details}
    validation_scores = scores(rgb, fitted, protocol, data["val"])
    validation_variants = np.array([row["variant"] for row in positions["val"]])
    protocol["thresholds"] = {}
    for name, values in validation_scores.items():
        include = np.ones(len(values), dtype=bool)
        for generator in config["generator_ids"]:
            if name == f"frequency_{generator}_only":
                include = np.isin(validation_variants, ["original", generator])
        protocol["thresholds"][name] = choose_threshold(y["val"][include], values[include])
    torch.save(rgb.state_dict(), output / "rgb.pt")
    joblib.dump(fitted, output / "statistical.joblib")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return rgb, fitted, protocol


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_holdout(rows, directory, output, rgb, fitted, protocol, cdch, jpeg_paths):
    config = protocol["config"]
    holdout = [row for row in rows if row["split"] == "test"]
    variant = np.array([row["variant"] for row in holdout])
    group = np.array([row["source_id"] for row in holdout])
    original, bicubic = variant == "original", variant == "bicubic"
    ai = ~(original | bicubic)
    tasks = {"original_vs_ai": (original | ai, ai),
             "original_vs_bicubic": (original | bicubic, bicubic),
             "bicubic_vs_ai": (bicubic | ai, ai)}
    for generator in config["generator_ids"]:
        match = variant == generator
        tasks[f"original_vs_{generator}"] = (original | match, match)
        tasks[f"bicubic_vs_{generator}"] = (bicubic | match, match)

    results, per_item, deltas, conditional, disagreement = [], [], [], [], []
    for quality in [None, *config["jpeg_quality"]]:
        cdch_inputs = cdch
        if quality is not None and cdch is not None:
            cdch_inputs, _ = read_cdch(jpeg_paths[quality], holdout, protocol["cdch_provenance"], quality)
        inputs = read_inputs(holdout, directory, cdch_inputs, quality)
        predictions = scores(rgb, fitted, protocol, inputs)
        encoding = "png" if quality is None else f"jpeg{quality}"
        for name, values in predictions.items():
            for row, value in zip(holdout, values):
                per_item.append(dict(encoding=encoding, model=name, source_id=row["source_id"],
                                     variant=row["variant"], score=float(value)))
        for fusion, members in protocol["fusion"].items():
            deviation = np.std([cdf(predictions[k], protocol["cdf"][k]) for k in members], axis=0)
            for row, value in zip(holdout, deviation):
                disagreement.append(dict(encoding=encoding, ensemble=fusion, source_id=row["source_id"],
                                         variant=row["variant"], rank_score_sd=float(value)))
        lookup = {(r["source_id"], r["variant"]): i for i, r in enumerate(holdout)}
        for name, values in predictions.items():
            for test in ("bicubic", *config["generator_ids"]):
                positives = values[[lookup[(g, test)] for g in np.unique(group)]]
                for control in ("original", "bicubic"):
                    if test == control:
                        continue
                    negatives = values[[lookup[(g, control)] for g in np.unique(group)]]
                    ordered = (positives > negatives) + 0.5 * (positives == negatives)
                    rng = np.random.default_rng(config["seed"])
                    resampled = ordered[rng.integers(len(ordered), size=(config["bootstrap_replicates"], len(ordered)))].mean(axis=1)
                    lo, hi = np.quantile(resampled, [0.025, 0.975])
                    conditional.append(dict(encoding=encoding, model=name, positive=test, negative=control,
                                            n_sources=len(ordered), ordered_fraction=ordered.mean(), ci_lo=lo, ci_hi=hi))
        for task, (include, positives) in tasks.items():
            y = positives[include].astype(int)
            weights = bootstrap_weights(group[include], config["bootstrap_replicates"], config["seed"])
            for name, values in predictions.items():
                results.append(dict(encoding=encoding, task=task, model=name,
                                    groups=len(set(group[include])),
                                    **metrics(y, values[include], protocol["thresholds"][name], weights)))
            for fusion, members in comparisons(protocol).items():
                for baseline in members:
                    lhs, rhs = predictions[fusion][include], predictions[baseline][include]
                    point = statistics(y, lhs)["auroc"][0] - statistics(y, rhs)["auroc"][0]
                    dist = statistics(y, lhs, weights=weights)["auroc"] - statistics(y, rhs, weights=weights)["auroc"]
                    lo, hi = np.quantile(dist, [0.025, 0.975])
                    deltas.append(dict(encoding=encoding, task=task, ensemble=fusion, member=baseline,
                                       auroc_delta=point, ci_lo=lo, ci_hi=hi))
    write_csv(output / "holdout_metrics.csv", results)
    write_csv(output / "holdout_scores.csv", per_item)
    write_csv(output / "paired_auroc_deltas.csv", deltas)
    write_csv(output / "within_source_ordering.csv", conditional)
    write_csv(output / "holdout_disagreement.csv", disagreement)


def comparisons(protocol):
    pairs = dict(protocol["fusion"])
    if protocol["cdch_provenance"] is not None:
        pairs["frequency_cdch"] = ["frequency", "cdch"]
        pairs["rgb_frequency_cdch"] = [*pairs["rgb_frequency_cdch"], "rgb_frequency"]
        pairs["ensemble_cdch"] = [*pairs["ensemble_cdch"], "ensemble"]
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cdch", type=Path)
    parser.add_argument("--cdch-jpeg", nargs=2, metavar=("QUALITY", "SIDECAR"), action="append", default=[])
    args = parser.parse_args()
    rows = json.loads((args.pairs / "manifest.json").read_text())
    config = json.loads((args.pairs / "inventory.json").read_text())["config"]
    torch.set_num_threads(config["torch_threads"])
    torch.use_deterministic_algorithms(True)
    expected = {"original", "lr", "bicubic", *config["generator_ids"]}
    variant_split = {}
    for row in rows:
        key = row["source_id"]
        split, variants = variant_split.setdefault(key, (row["split"], set()))
        if row["split"] != split or row["variant"] in variants:
            raise ValueError("Source crossover or duplicate variant")
        variants.add(row["variant"])
        if sha(args.pairs / row["path"]) != row["sha256"]:
            raise ValueError("Pair checksum mismatch")
    if any(variants != expected for _, variants in variant_split.values()):
        raise ValueError("Missing or unexpected variants")
    counts = Counter(split for split, _ in variant_split.values())
    if [counts[k] for k in ("train", "val", "test")] != config["split_sources"]:
        raise ValueError("Unexpected split counts")
    rows = [row for row in rows if row["variant"] != "lr"]
    cdch, provenance = read_cdch(args.cdch, rows)
    jpeg_paths = {int(quality): Path(path) for quality, path in args.cdch_jpeg}
    if set(jpeg_paths) != (set(config["jpeg_quality"]) if cdch is not None else set()):
        parser.error("Supply one CDCH JPEG sidecar for each configured quality when CDCH is active")
    args.out.mkdir(parents=True, exist_ok=False)
    rgb, fitted, protocol = setup_models(rows, args.pairs, args.out, config, cdch, provenance,
                                         sha(args.pairs / "manifest.json"))
    run_holdout(rows, args.pairs, args.out, rgb, fitted, protocol, cdch, jpeg_paths)



if __name__ == "__main__":
    main()
