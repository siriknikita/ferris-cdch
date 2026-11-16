"""Canonical CDCH transport and downstream representations."""
import base64
import io
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from PIL import Image

from build_pairs import ROOT, sha
from cdch_features import CANONICAL, descriptor

HERE = Path(__file__).resolve().parent


def verify_js():
    if sha(ROOT / "cdch/ip-1.js") != CANONICAL:
        raise ValueError("Canonical Muzyka JavaScript checksum mismatch")


@contextmanager
def worker():
    # Node independently verifies the same digest before accepting input.
    cmd = ["node", str(HERE / "native_worker.js"), str(ROOT / "cdch"), CANONICAL]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
        try:
            yield process
        except BaseException:
            process.kill()
            raise
        finally:
            process.stdin.close()
        if process.wait():
            raise RuntimeError("Canonical CDCH worker exited unsuccessfully")


def read_rgb(path, encoding="png"):
    with Image.open(path) as original:
        if original.mode != "RGB" or original.format != "PNG" or original.size != (128, 128):
            raise ValueError(f"Expected normalized RGB 128 x 128 PNG: {path}")
        if encoding in ("jpeg90", "jpeg75"):
            buffer = io.BytesIO()
            original.save(buffer, format="JPEG", quality=int(encoding[4:]), subsampling=2, optimize=False)
            buffer.seek(0)
            with Image.open(buffer) as jpeg:
                return np.array(jpeg)
        if encoding == "resize96":
            return np.array(original.resize((96, 96), Image.Resampling.LANCZOS)
                            .resize((128, 128), Image.Resampling.LANCZOS))
        if encoding != "png":
            raise ValueError(f"Unknown encoding {encoding}")
        return np.array(original)


def matrix(process, rgb):
    height, width, _ = rgb.shape
    start = perf_counter_ns()
    request = {"height": height, "width": width, "pixels": rgb.reshape(-1).tolist(),
               "center": [width // 2, height // 2], "angle_step": 1}
    wire = (json.dumps(request, separators=(",", ":")) + "\n").encode()
    packed = perf_counter_ns()
    process.stdin.write(wire)
    process.stdin.flush()
    line = process.stdout.readline()
    received = perf_counter_ns()
    if not line:
        raise RuntimeError("Canonical CDCH worker stopped before replying")
    response = json.loads(line)
    if response.get("error") or response["center"] != request["center"] or response["rows"] != 360:
        raise ValueError(f"Invalid canonical CDCH worker response: {response.get('error')}")
    cdf = np.frombuffer(base64.b64decode(response["binary"], validate=True), "<f8").reshape(360, 256)
    end = perf_counter_ns()
    return cdf, {
        "js": response["js_ns"],
        "ipc": end - start - response["js_ns"],
        "python_pack": packed - start,
        "python_unpack": end - received,
        "node_json_parse": response["parse_ns"],
        "node_reply_pack": response["pack_ns"],
        "roundtrip": end - start,
    }


def rich(cdf):
    pmf = np.diff(cdf, axis=1, prepend=0)
    bins = np.linspace(0, 1, 256)
    mean = pmf @ bins
    centered = bins - mean[:, None]
    variance = np.sum(pmf * centered**2, axis=1)
    measures = [mean, np.sqrt(variance)]
    for k in (3, 4):
        central = np.sum(pmf * centered**k, axis=1)
        measures.append(np.divide(central, variance**(k / 2), out=np.zeros_like(central),
                                  where=variance > 1e-10))
    measures += [-np.sum(pmf * np.log2(np.maximum(pmf, 1e-300)), axis=1) / 8,
                 np.sum(pmf**2, axis=1), cdf[:, 15], 1 - cdf[:, 239],
                 np.max(pmf, axis=1), np.mean(np.abs(np.diff(pmf, axis=1)), axis=1)]
    quantiles = [(cdf >= p).argmax(axis=1) / 255 for p in (0.1, 0.25, 0.5, 0.75, 0.9)]
    measures.extend(quantiles)
    measures.extend([quantiles[3] - quantiles[1], quantiles[4] - quantiles[0]])
    for lag in (1, 15, 90, 180):
        measures.append(np.mean(np.abs(cdf - np.roll(cdf, lag, axis=0)), axis=1))
    return np.concatenate([np.array([r.mean(), r.std()]) for r in measures])


def representation(cdf, kind, compact=None):
    if kind == "A":
        return np.asarray(descriptor(cdf) if compact is None else compact)
    if kind == "B":
        return np.r_[representation(cdf, "A", compact), rich(cdf)]
    if kind in {"C", "D", "D_permuted"}:
        return cdf[:, 7::8].reshape(24, 15, 32).mean(axis=1).ravel()
    raise ValueError(f"Unknown CDCH representation {kind}")
