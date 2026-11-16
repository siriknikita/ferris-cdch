"""Freeze source-only models, score blind face crops, and reuse the pilot label join."""
import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import torch

import zero_shot as pilot
from build_pairs import ROOT, sha
from native_features import verify_js
from native_pipeline import Pipeline

HERE = Path(__file__).resolve().parent
CODE = ("native_features.py", "native_models.py", "native_worker.js", "native_pipeline.py",
        "target_native.py", "config/native.json")


def freeze(args):
    run, frozen = args.run, args.out
    decision = pilot.read_json(run / "SOURCE_DECISIONS.json")
    models = joblib.load(run / "source_models.joblib")
    if decision["thresholds"] != models["thresholds"] or decision["selected"] != models["selected"]:
        raise ValueError("Frozen source decisions disagree with checkpoints")
    if decision["manifest_sha256"] != sha(args.pairs / "manifest.json"):
        raise ValueError("Target freeze must refer to the original training manifest")
    if decision["config_sha256"] != sha(HERE / "config/native.json"):
        raise ValueError("Target freeze config differs from training config")
    protocol = {
        "config": pilot.read_json(HERE / "config/native.json"),
        "thresholds": decision["thresholds"],
        "fusion": {name: list(weights) for name, weights in decision["fusion_weights"].items()},
        "cdch_provenance": None,
        "input_px": 128,
        "training_manifest_sha256": decision["manifest_sha256"],
        "primary": ["frequency", decision["selected"]],
        "video_aggregation": "mean of prespecified frame scores",
        "score_direction": "higher_is_fake",
    }
    with TemporaryDirectory() as directory:
        temp = Path(directory)
        for name in ("rgb.pt", "statistical.joblib"):
            shutil.copy2(ROOT / "results" / name, temp / name)
        pilot.emit(temp / "protocol.json", protocol)
        pilot.freeze(argparse.Namespace(run=temp, pairs=args.pairs, preprocessing=args.preprocessing, out=frozen))
    record = pilot.read_json(frozen / "freeze.json")
    for name in CODE:
        saved = frozen / "code" / "native" / name
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / name, saved)
        record["sha256"][f"code/native/{name}"] = sha(saved)
    for name in ("source_models.joblib", "SOURCE_DECISIONS.json"):
        shutil.copy2(run / name, frozen / name)
        record["sha256"][name] = sha(frozen / name)
    (frozen / "freeze.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Use final extended freeze digest: {sha(frozen / 'freeze.json')}")


def verify_current(directory):
    protocol, digest = pilot.verify_frozen(directory)
    for name in CODE:
        if sha(HERE / name) != sha(directory / "code" / "native" / name):
            raise ValueError(f"Native scoring code differs from frozen copy: {name}")
    verify_js()
    return protocol, digest


def score(args):
    torch.set_num_threads(1)
    protocol, digest = verify_current(args.frozen)
    rows = pilot.csv_rows(args.blind, ("item_id", "video_id", "path", "sha256"))
    if len({r["item_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate blind ID")
    if max(Counter(r["video_id"] for r in rows).values()) > 16:
        raise ValueError("Each video may contain at most 16 sampled frames")
    base = args.crops.resolve()
    paths = []
    for row in rows:
        path = (base / row["path"]).resolve()
        if not path.is_relative_to(base) or sha(path) != row["sha256"]:
            raise ValueError(f"Unsafe path or mismatched image digest: {row['item_id']}")
        paths.append(path)
    pipe = Pipeline(joblib.load(args.frozen / "source_models.joblib"),
                    args.frozen / "rgb.pt", list(protocol["thresholds"]))
    args.out.mkdir(parents=True)
    output, disagreements = args.out / "scores.csv", args.out / "disagreement.csv"
    with output.open("x", newline="") as stream, pipe.worker() as worker:
        writer = csv.DictWriter(stream, ["item_id", "video_id", "model", "score"])
        writer.writeheader()
        from math import isfinite
        for row, path in zip(rows, paths):
            scores, _ = pipe.predict(path, worker)
            for model, value in scores.items():
                if not isfinite(value):
                    raise ValueError(f"Nonfinite score from {model}")
                writer.writerow(dict(item_id=row["item_id"], video_id=row["video_id"], model=model, score=value))
    disagreements.write_text("item_id,video_id,ensemble,rank_score_sd\n")
    pilot.emit(args.out / "seal.json", dict(freeze_sha256=digest, blind_manifest_sha256=sha(args.blind),
                                          scores_sha256=sha(output), disagreement_sha256=sha(disagreements),
                                          item_count=len(rows)))
    print(f"Seal before opening labels: {sha(args.out / 'seal.json')}")


def evaluate(args):
    verify_current(args.frozen)
    pilot.evaluate(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for func, required in ((freeze, ("run", "pairs", "preprocessing")),
                           (score, ("frozen", "blind", "crops")),
                           (evaluate, ("frozen", "scored", "labels"))):
        tool = sub.add_parser(func.__name__)
        for key in (*required, "out"):
            tool.add_argument(f"--{key}", required=True, type=Path)
        tool.set_defaults(action=func)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
