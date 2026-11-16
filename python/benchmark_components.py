"""Phase I: component benchmarks — Python/Rust crossing, batch throughput, size scaling.

    python benchmark_components.py --pairs ../../external/pairs --out ../reports/perf/components.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

import cdch_ferris as cf
from ferris_pipeline import ROOT, archived_center
from native_features import matrix, read_rgb, worker  # archived JS transport


def timed(fn, reps, warmup=20):
    for _ in range(warmup):
        fn()
    ns = []
    for _ in range(reps):
        t = perf_counter_ns()
        fn()
        ns.append(perf_counter_ns() - t)
    a = np.array(ns) / 1e3
    return {"median_us": float(np.median(a)), "p95_us": float(np.quantile(a, 0.95)), "reps": reps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [r for r in json.loads((ROOT / "results/manifest.json").read_text()) if r["variant"] != "lr"]
    paths = [args.pairs / r["path"] for r in rows]
    result = {}

    # 1. Decode cost (PIL PNG -> uint8 array), the archived read_rgb.
    t = perf_counter_ns()
    images = [read_rgb(p) for p in paths]
    result["decode_png_per_image_us"] = (perf_counter_ns() - t) / 1e3 / len(paths)
    stack = np.stack(images)
    one = images[0]
    c = archived_center(one)

    # 2. Python <-> Rust crossing: identical work called from Python, and near-zero work.
    tiny = np.zeros((1, 1, 3), np.uint8)
    result["crossing"] = {
        "features_A_128": timed(lambda: cf.features(one, "A", center=c), 5000),
        "features_C_128": timed(lambda: cf.features(one, "C", center=c), 5000),
        "features_B_128": timed(lambda: cf.features(one, "B", center=c), 1000),
        "cdch_matrix_128": timed(lambda: cf.cdch(one, center=c), 5000),
        "features_multi_ABC_128": timed(lambda: cf.features_multi(one, ["A", "B", "C"], center=c), 1000),
        "near_zero_work_call_1x1_C": timed(lambda: cf.features(tiny, "C", center=(0.0, 0.0)), 20000),
    }

    # 3. Batch throughput from decoded arrays, by thread count (GIL released).
    batch = []
    for kind in ("A", "B", "C"):
        for threads in (1, 2, 4, 8, None):
            cf.batch_features(stack, kind, threads=threads)
            rates = []
            for _ in range(5):
                t = perf_counter_ns()
                cf.batch_features(stack, kind, threads=threads)
                rates.append(len(stack) / ((perf_counter_ns() - t) / 1e9))
            batch.append({"kind": kind, "threads": threads or cf.build_info()["rayon_threads"], "images": len(stack),
                          "median_img_s": float(np.median(rates))})
    result["batch_from_arrays"] = batch

    # 4. Batch including decoding, single process (decode is serial PIL).
    t = perf_counter_ns()
    decoded = np.stack([read_rgb(p) for p in paths])
    feats = cf.batch_features(decoded, "A", threads=1)
    result["batch_decode_plus_A_single_thread_img_s"] = len(paths) / ((perf_counter_ns() - t) / 1e9)
    t = perf_counter_ns()
    with worker() as process:
        for im in images[:100]:
            matrix(process, im)
    result["js_worker_matrix_throughput_img_s"] = 100 / ((perf_counter_ns() - t) / 1e9)
    assert feats.shape == (len(paths), 56)

    # 5. Size scaling: JS core (inside Node, excluding transport) vs Rust matrix / fused A.
    big = read_rgb(paths[0])  # natural content, tiled for larger sizes
    scaling = []
    with worker() as process:
        for size in (64, 128, 256, 512, 1024):
            reps = max(5, 200 // (size // 64))
            img = np.ascontiguousarray(np.tile(big, (size // 128 + 1, size // 128 + 1, 1))[:size, :size])
            js_core, roundtrip = [], []
            for _ in range(reps):
                _, stats = matrix(process, img)
                js_core.append(stats["js"] / 1e3)
                roundtrip.append(stats["roundtrip"] / 1e3)
            ctr = archived_center(img)
            row = {"size": size, "reps": reps, "js_core_median_us": float(np.median(js_core)),
                   "js_roundtrip_median_us": float(np.median(roundtrip)),
                   "rust_matrix": timed(lambda: cf.cdch(img, center=ctr), reps * 10, 5),
                   "rust_reference_matrix": timed(lambda: cf.cdch_reference(img, center=ctr), reps, 3),
                   "rust_fused_A": timed(lambda: cf.features(img, "A", center=ctr), reps * 10, 5)}
            scaling.append(row)
    result["size_scaling"] = scaling
    result["build"] = cf.build_info()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("size_scaling", "batch_from_arrays")}, indent=1))
    for r in scaling:
        print(r["size"], "js_core", round(r["js_core_median_us"]), "us; rust matrix", round(r["rust_matrix"]["median_us"], 1),
              "us; rust ref", round(r["rust_reference_matrix"]["median_us"], 1), "us; fused A",
              round(r["rust_fused_A"]["median_us"], 1))
    for b in batch:
        print(b)


if __name__ == "__main__":
    main()
