"""Seal source-only models, score blind target crops, then join target labels."""
import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image

from build_pairs import ROOT, sha
from experiment import (
    TinyRGB, bootstrap_weights, cdf, comparisons, metrics, read_cdch, read_inputs,
    scores, statistics, write_csv,
)

MODELS = ("rgb.pt", "statistical.joblib", "protocol.json")
CODE = ("build_pairs.py", "experiment.py", "zero_shot.py",
        "pyproject.toml", "uv.lock", "config/target_protocol.json", "cdch_features.py",
        "cdch/ip-1.js", "cdch/worker.js", "cdch/oracle.js")


def read_json(path):
    return json.loads(path.read_text())


def emit(path, payload):
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def csv_rows(path, fields):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != set(fields):
            raise ValueError(f"Wrong fields in {path}; expected {fields}")
        rows = list(reader)
    if not rows or any(not row[x] for row in rows for x in fields):
        raise ValueError(f"Empty fields or rows in {path}")
    return rows


def freeze(args):
    (args.out / "code").mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in MODELS:
        path = args.out / name
        shutil.copy2(args.run / name, path)
        hashes[name] = sha(path)
    for name in CODE:
        path = args.out / "code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
        hashes[f"code/{name}"] = sha(path)
    for path, name in [(args.pairs / "manifest.json", "pair_manifest.json"),
                       (args.pairs / "inventory.json", "pair_inventory.json"),
                       (args.preprocessing, "target_preprocessing.json")]:
        target = args.out / name
        read_json(path)
        shutil.copy2(path, target)
        hashes[name] = sha(target)
    if read_json(args.out / "protocol.json")["training_manifest_sha256"] != hashes["pair_manifest.json"]:
        raise ValueError("Pair manifest differs from the one used for training")
    emit(args.out / "freeze.json", {"status": "source_only", "sha256": hashes})
    print(f"Record this freeze SHA-256 before label release: {sha(args.out / 'freeze.json')}")


def verify_frozen(directory):
    freeze_path = directory / "freeze.json"
    frozen = read_json(freeze_path)
    if frozen["status"] != "source_only":
        raise ValueError("Unsupported frozen status")
    for name, digest in frozen["sha256"].items():
        if sha(directory / name) != digest:
            raise ValueError(f"Frozen digest mismatch in {name}")
    for name in CODE:
        if sha(ROOT / name) != frozen["sha256"][f"code/{name}"]:
            raise ValueError(f"Current code differs from frozen copy: {name}")
    return read_json(directory / "protocol.json"), sha(freeze_path)


def score(args):
    protocol, freeze_digest = verify_frozen(args.frozen)
    torch.set_num_threads(protocol["config"]["torch_threads"])
    rows = csv_rows(args.blind, ("item_id", "video_id", "path", "sha256"))
    seen = set()
    counts = defaultdict(int)
    for row in rows:
        if row["item_id"] in seen:
            raise ValueError("Repeated blind item ID")
        seen.add(row["item_id"])
        counts[row["video_id"]] += 1
        path = args.crops / row["path"]
        if sha(path) != row["sha256"]:
            raise ValueError(f"Input SHA-256 mismatch for {row['item_id']}")
        with Image.open(path) as image:
            if image.size != (protocol["input_px"],) * 2 or image.mode != "RGB" or image.format != "PNG":
                raise ValueError(f"Invalid crop {row['item_id']}")
    if max(counts.values()) > 16:
        raise ValueError("Target protocol allows at most 16 frames per video")
    cdch, provenance = read_cdch(args.cdch, rows, protocol["cdch_provenance"])
    if provenance != protocol["cdch_provenance"]:
        raise ValueError("CDCH presence does not match frozen model")
    rgb = TinyRGB()
    rgb.load_state_dict(torch.load(args.frozen / "rgb.pt", weights_only=True, map_location="cpu"))
    # Only load self-produced or otherwise trusted model files.
    fitted = joblib.load(args.frozen / "statistical.joblib")
    model_names = list(protocol["thresholds"])
    args.out.mkdir(parents=True, exist_ok=False)
    output_path = args.out / "scores.csv"
    disagreement_path = args.out / "disagreement.csv"
    with output_path.open("x", newline="") as stream, disagreement_path.open("x", newline="") as disagreement:
        disagreements = csv.DictWriter(disagreement, fieldnames=["item_id", "video_id", "ensemble", "rank_score_sd"])
        disagreements.writeheader()
        writer = csv.DictWriter(stream, fieldnames=["item_id", "video_id", "model", "score"])
        writer.writeheader()
        for start in range(0, len(rows), 64):
            batch = rows[start:start + 64]
            prediction = scores(rgb, fitted, protocol, read_inputs(batch, args.crops, cdch))
            for name, members in protocol["fusion"].items():
                rank_scores = [cdf(prediction[k], protocol["cdf"][k]) for k in members]
                for row, value in zip(batch, np.std(rank_scores, axis=0)):
                    disagreements.writerow(dict(item_id=row["item_id"], video_id=row["video_id"],
                                                ensemble=name, rank_score_sd=float(value)))
            for model in model_names:
                for row, value in zip(batch, prediction[model]):
                    if not np.isfinite(value):
                        raise ValueError("Nonfinite blind score")
                    writer.writerow(dict(item_id=row["item_id"], video_id=row["video_id"],
                                         model=model, score=float(value)))
    emit(args.out / "seal.json", dict(freeze_sha256=freeze_digest,
                                      blind_manifest_sha256=sha(args.blind),
                                      scores_sha256=sha(output_path),
                                      disagreement_sha256=sha(disagreement_path), item_count=len(rows)))
    print(f"Record this score seal SHA-256 before label release: {sha(args.out / 'seal.json')}")


def subsets(labels):
    videos = defaultdict(list)
    for video, row in labels.items():
        videos[(row["dataset"], row["compression"])].append(video)
    for key, all_videos in sorted(videos.items()):
        yield (*key, "all"), set(all_videos)
        real = {video for video in all_videos if labels[video]["label"] == "0"}
        manipulations = {labels[v]["manipulation"] for v in all_videos if labels[v]["label"] == "1"}
        for manipulation in sorted(manipulations):
            fake = {v for v in all_videos if labels[v]["label"] == "1"
                    and labels[v]["manipulation"] == manipulation}
            yield (*key, manipulation), real | fake


def target_bootstrap(groups, count, seed):
    # Keep memory bounded on frame-level benchmarks.
    generator = np.random.default_rng(seed)
    for start in range(0, count, 16):
        yield bootstrap_weights(groups, min(16, count - start),
                                int(generator.integers(0, 2 ** 32)))


def evaluate(args):
    protocol, freeze_digest = verify_frozen(args.frozen)
    seal = read_json(args.scored / "seal.json")
    score_path = args.scored / "scores.csv"
    if (freeze_digest != seal["freeze_sha256"] or sha(score_path) != seal["scores_sha256"]
            or sha(args.scored / "disagreement.csv") != seal["disagreement_sha256"]):
        raise ValueError("Score seal mismatch")
    rows = csv_rows(score_path, ("item_id", "video_id", "model", "score"))
    truth = csv_rows(args.labels, ("video_id", "dataset", "compression", "manipulation", "label", "cluster_id"))
    labels = {r["video_id"]: r for r in truth}
    if len(labels) != len(truth) or set(labels) != {r["video_id"] for r in rows}:
        raise ValueError("Labels must cover exactly the scored videos once each")
    if any(r["label"] not in {"0", "1"} or (r["label"] == "0") != (r["manipulation"] == "real")
           or r["manipulation"] == "all" for r in truth):
        raise ValueError("Use real/0 and distinct named manipulations/1")

    score_by_model = defaultdict(dict)
    item_to_video = {}
    for row in rows:
        item, video, name = row["item_id"], row["video_id"], row["model"]
        if item in score_by_model[name] or item_to_video.setdefault(item, video) != video:
            raise ValueError("Duplicate or inconsistent score key")
        score_by_model[name][item] = float(row["score"])
    if (set(score_by_model) != set(protocol["thresholds"]) or len(item_to_video) != seal["item_count"]
            or any(scores.keys() != item_to_video.keys() for scores in score_by_model.values())):
        raise ValueError("Scoring coverage mismatch")

    output, reversed_output, deltas, skipped = [], [], [], []
    config = protocol["config"]
    for (dataset, codec, manipulation), videos in subsets(labels):
        group_items = sorted(i for i, v in item_to_video.items() if v in videos)
        for level in ("frame", "video"):
            unit_videos = [item_to_video[i] for i in group_items] if level == "frame" else sorted(videos)
            y = np.array([int(labels[v]["label"]) for v in unit_videos])
            groups = np.array([labels[v]["cluster_id"] for v in unit_videos])
            base = dict(dataset=dataset, compression=codec, manipulation=manipulation,
                        level=level, groups=len(set(groups)))
            if len(np.unique(y)) != 2 or len(set(groups)) < 2:
                skipped.append({**base, "reason": "requires both classes and at least two declared clusters"})
                continue
            unit_scores = {}
            for model, scores_by_item in score_by_model.items():
                per_video = defaultdict(list)
                for item in group_items:
                    per_video[item_to_video[item]].append(scores_by_item[item])
                values = (np.array([scores_by_item[i] for i in group_items]) if level == "frame"
                          else np.array([np.mean(per_video[v]) for v in unit_videos]))
                unit_scores[model] = values
                weights = target_bootstrap(groups, config["bootstrap_replicates"], config["seed"])
                result = metrics(y, values, protocol["thresholds"][model], weights)
                output.append(dict(**base, model=model, **result))
                rev = {k: float(v[0]) for k, v in statistics(y, -values).items()}
                reversed_output.append(dict(**base, model=model, **rev))
            for fusion, members in comparisons(protocol).items():
                lhs = unit_scores[fusion]
                for member in members:
                    rhs = unit_scores[member]
                    point = statistics(y, lhs)["auroc"][0] - statistics(y, rhs)["auroc"][0]
                    draws = target_bootstrap(groups, config["bootstrap_replicates"], config["seed"])
                    delta = np.concatenate([statistics(y, lhs, weights=w)["auroc"] -
                                            statistics(y, rhs, weights=w)["auroc"] for w in draws])
                    lo, hi = np.nanquantile(delta, [0.025, 0.975])
                    deltas.append(dict(**base, ensemble=fusion, member=member,
                                       auroc_delta=point, ci_lo=lo, ci_hi=hi))

    args.out.mkdir(parents=True, exist_ok=False)
    for name, results in [("direct", output), ("exploratory_reversed", reversed_output),
                          ("paired_auroc_deltas", deltas)]:
        if results:
            write_csv(args.out / f"{name}.csv", results)
    emit(args.out / "evaluation.json", dict(freeze_sha256=freeze_digest,
                                           score_seal_sha256=sha(args.scored / "seal.json"),
                                           label_sha256=sha(args.labels), skipped=skipped))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = [
        ("freeze", freeze, ("run", "pairs", "preprocessing", "out"), ()),
        ("score", score, ("frozen", "blind", "crops", "out"), ("cdch",)),
        ("evaluate", evaluate, ("frozen", "scored", "labels", "out"), ()),
    ]
    for name, action, required, optional in commands:
        command = sub.add_parser(name)
        for argument in (*required, *optional):
            command.add_argument(f"--{argument}", required=argument in required, type=Path)
        command.set_defaults(action=action)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
