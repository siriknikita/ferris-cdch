"""Exercise the locked target workflow using only holdout paired SR crops."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from build_pairs import ROOT, sha
from experiment import write_csv


def run(arguments):
    subprocess.run([sys.executable, *map(str, arguments)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = json.loads((args.pairs / "manifest.json").read_text())
    held = [r for r in rows if r["split"] == "test" and r["variant"] in ("original", "swinir")]
    ids = sorted({r["source_id"] for r in held})[:8]
    held = [r for r in held if r["source_id"] in ids]
    blind, labels = [], []
    for i, row in enumerate(held, 1):
        video = f"fixture-{i:03d}"
        blind.append(dict(item_id=f"{video}-001", video_id=video,
                          path=row["path"], sha256=row["sha256"]))
        is_sr = row["variant"] == "swinir"
        labels.append(dict(video_id=video, dataset="SR-only CLI fixture", compression="png",
                           manipulation="swinir-treatment" if is_sr else "real", label=int(is_sr),
                           cluster_id=row["source_id"]))
    manifest = args.out / "blind.csv"
    write_csv(manifest, blind)
    prep = args.out / "preprocessing.json"
    prep.write_text(json.dumps({"scope": "CLI fixture using existing SR holdout crops",
                               "dataset": "DIV2K validation partition",
                               "decoding": "one normalized RGB PNG per pseudo-video"}, indent=2))
    target = ROOT / "zero_shot.py"
    frozen, scored = args.out / "frozen", args.out / "scored"
    feature_file = args.out / "cdch.json"
    run([target, "freeze", "--run", args.run, "--pairs", args.pairs, "--preprocessing", prep, "--out", frozen])
    run([ROOT / "cdch_features.py", "--blind", manifest, "--crops", args.pairs, "--out", feature_file])
    run([target, "score", "--frozen", frozen, "--blind", manifest, "--crops", args.pairs,
         "--cdch", feature_file, "--out", scored])
    labels_file = args.out / "fixture_labels.csv"
    write_csv(labels_file, labels)
    evaluated = args.out / "evaluated"
    run([target, "evaluate", "--frozen", frozen, "--scored", scored, "--labels", labels_file, "--out", evaluated])
    with (evaluated / "direct.csv").open() as stream:
        results = list(csv.DictReader(stream))
    if not results or {int(row["n"]) for row in results} != {len(held)}:
        raise ValueError("Unexpected fixture evaluation coverage")
    path = scored / "scores.csv"
    with path.open("a") as stream:
        stream.write("\n")
    rejected = subprocess.run([sys.executable, str(target), "evaluate", "--frozen", str(frozen),
                               "--scored", str(scored), "--labels", str(labels_file),
                               "--out", str(args.out / "tampered")], text=True, capture_output=True)
    if rejected.returncode == 0 or "Score seal mismatch" not in rejected.stderr:
        raise ValueError("Target reader accepted modified scores")
    path.write_bytes(path.read_bytes().removesuffix(b"\n"))
    seal = json.loads((scored / "seal.json").read_text())
    if sha(path) != seal["scores_sha256"]:
        raise ValueError("Could not restore the fixture score file")
    print(f"PASS: freeze, blind score, post-seal evaluation, tamper rejection ({len(held)} SR fixture items)")


if __name__ == "__main__":
    main()
