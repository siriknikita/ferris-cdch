"""Execute canonical Muzyka JS and summarize its output matrices in Python."""
import argparse
import base64
import json
import subprocess
from pathlib import Path

import numpy as np

from build_pairs import ROOT, sha
from experiment import pixels

CANONICAL = "110732b48bc6e8b2d5167eec30f7277245913bed98b9baaa3c65c0c22aab44d8"
METHOD = "geometric-center-360x256-cdf-summary-v1"


def descriptor(cdf):
    if (cdf.shape != (360, 256) or not np.isfinite(cdf).all() or cdf.min() < 0
            or not np.allclose(cdf[:, -1], 1) or np.any(np.diff(cdf, axis=1) < -1e-12)):
        raise ValueError("Invalid canonical cumulative histogram matrix")
    sampled = cdf[:, 7:255:16]
    # Downstream statistics on the canonical output, no ray or histogram generation.
    directional_change = np.abs(np.roll(sampled, 1, axis=0) - sampled)
    directional_power = np.abs(np.fft.rfft(sampled - sampled.mean(axis=0), axis=0)) / 360
    harmonics = [directional_power[i].mean() for i in (1, 2, 4, 8, 16, 32, 64, 128)]
    return np.r_[sampled.mean(axis=0), sampled.std(axis=0),
                 directional_change.mean(axis=0), harmonics].tolist()


def generate(rows, directory, quality=None):
    script_dir = ROOT / "cdch"
    digest = sha(script_dir / "ip-1.js")
    if digest != CANONICAL:
        raise ValueError(f"Canonical JS checksum mismatch: {digest}")
    adapter_hashes = {name: sha(script_dir / name) for name in ("worker.js", "oracle.js")}
    result = {}
    with subprocess.Popen(["node", str(script_dir / "worker.js")], text=True,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
        try:
            for index, row in enumerate(rows, 1):
                path = directory / row["path"]
                if row["sha256"] != sha(path) or row["path"] in result:
                    raise ValueError(f"Invalid or repeated input path {path}")
                rgb = np.rint(pixels(path, quality) * 255).astype(np.uint8)
                height, width, _ = rgb.shape
                request = {"width": width, "height": height,
                           "pixels": rgb.reshape(-1).tolist(),
                           "center": [width // 2, height // 2], "angle_step": 1}
                process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
                process.stdin.flush()
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError("Canonical Node worker stopped before producing a matrix")
                response = json.loads(line)
                if response.get("error") or response["rows"] != 360:
                    raise ValueError(response)
                if response["center"] != request["center"]:
                    raise ValueError("Canonical oracle reported an unexpected center")
                matrix = np.frombuffer(base64.b64decode(response["binary"], validate=True),
                                       dtype="<f8").reshape(360, 256)
                result[row["path"]] = descriptor(matrix)
                if index % 50 == 0:
                    print(f"CDCH matrices processed: {index}/{len(rows)}", flush=True)
        except BaseException:
            process.kill()
            raise
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("Canonical Node worker failed")
    return {"canonical_sha256": digest, "adapter_sha256": adapter_hashes,
            "descriptor_code_sha256": sha(Path(__file__)), "method": METHOD,
            "node_version": subprocess.check_output(["node", "--version"], text=True).strip(),
            "jpeg_quality": quality, "image_sha256": {r["path"]: r["sha256"] for r in rows},
            "features": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pairs", type=Path)
    source.add_argument("--blind", type=Path)
    parser.add_argument("--crops", type=Path)
    parser.add_argument("--quality", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a fresh output path")
    if args.blind is not None:
        from zero_shot import csv_rows
        if args.crops is None or args.quality is not None:
            parser.error("Blind inputs require --crops and use the supplied PNG pixels")
        rows = csv_rows(args.blind, ("item_id", "video_id", "path", "sha256"))
        directory = args.crops
    else:
        rows = json.loads((args.pairs / "manifest.json").read_text())
        rows = [r for r in rows if r["variant"] != "lr" and (args.quality is None or r["split"] == "test")]
        directory = args.pairs
    payload = generate(rows, directory, args.quality)
    with args.out.open("x") as stream:
        json.dump(payload, stream, separators=(",", ":"))
        stream.write("\n")


if __name__ == "__main__":
    main()
