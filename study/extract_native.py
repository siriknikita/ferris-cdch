"""Build verified CDCH and frequency features on the unchanged pair manifest."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from build_pairs import ROOT, sha
from experiment import frequency_features, read_cdch
from native_features import matrix, read_rgb, representation, verify_js, worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    verify_js()
    archived = ROOT / "results"
    manifest = args.pairs / "manifest.json"
    if sha(manifest) != sha(archived / "manifest.json"):
        raise ValueError("Pair manifest differs from archived pilot")
    all_rows = json.loads(manifest.read_text())
    for row in all_rows:
        if sha(args.pairs / row["path"]) != row["sha256"]:
            raise ValueError(f"Pair digest mismatch: {row['path']}")
    rows = [row for row in all_rows if row["variant"] != "lr"]
    test = [row for row in rows if row["split"] == "test"]
    args.out.mkdir(parents=True)
    with worker() as process:
        for encoding in ("png", "jpeg90", "jpeg75", "resize96"):
            selected = rows if encoding == "png" else test
            previous = None
            if encoding != "resize96":
                quality = None if encoding == "png" else int(encoding[4:])
                previous, _ = read_cdch(archived / f"cdch-{encoding}.json", selected, quality=quality)
            arrays = {name: [] for name in ("A", "B", "C", "frequency")}
            for row in selected:
                rgb = read_rgb(args.pairs / row["path"], encoding)
                cdf, _ = matrix(process, rgb)
                a = representation(cdf, "A")
                if previous is not None:
                    np.testing.assert_allclose(a, previous[row["path"]], rtol=1e-12, atol=1e-12)
                arrays["A"].append(a)
                for name in ("B", "C"):
                    arrays[name].append(representation(cdf, name, a))
                arrays["frequency"].append(frequency_features(rgb.astype(np.float32) / 255))
            np.savez_compressed(args.out / f"{encoding}.npz",
                                paths=[row["path"] for row in selected], **arrays)
            print(f"{encoding}: extracted {len(selected)} matrices", flush=True)


if __name__ == "__main__":
    main()
