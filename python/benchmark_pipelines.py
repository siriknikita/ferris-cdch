"""Phase I: end-to-end CPU latency of the archived classifiers with each CDCH backend.

Follows the archived benchmark_native.py protocol (config/native.json: warm-up, cold
sessions, single thread, the same seeded holdout image order), with portable resource
measurement (macOS has no /proc). Every (model, backend) runs in a fresh subprocess.

    python benchmark_pipelines.py --pairs ../../external/pairs --out ../reports/perf/pipelines
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns, process_time_ns

import numpy as np

HERE = Path(__file__).resolve().parent
MODELS_CDCH = ["cdch_A_logistic", "cdch_B_linear_svm", "cdch_C_logistic", "cdch_D_conv",
               "frequency_cdch", "frequency_cdch_score_fusion"]
MODELS_OTHER = ["frequency", "rgb"]
STAGES = ("decode", "js", "ipc", "python_pack", "python_unpack", "node_json_parse", "node_reply_pack",
          "roundtrip", "rust_matrix", "rust_fused", "feature", "cdch_total", "frequency", "rgb_preprocess",
          "cnn", "classifier_other", "total")


def rss_bytes(pid):
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) * 1024 if out else 0


def cpu_seconds(pid):
    """Cumulative CPU time of another process (ps TIME, [[dd-]hh:]mm:ss.ss)."""
    out = subprocess.run(["ps", "-o", "time=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    if not out:
        return 0.0
    days, _, rest = out.rpartition("-")
    parts = [float(p) for p in rest.split(":")]
    seconds = sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
    return seconds + (int(days) * 86400 if days else 0)


def inputs(pairs, config):
    from ferris_pipeline import ROOT
    rows = [r for r in json.loads((ROOT / "results/manifest.json").read_text())
            if r["split"] == "test" and r["variant"] != "lr"]
    order = np.random.default_rng(1611).permutation(len(rows))[:config["warmup"] + config["trials"]]
    return [pairs / rows[i]["path"] for i in order]


def run_one(args):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import ferris_pipeline as fp

    config = json.loads((fp.STUDY / "config/native.json").read_text())["benchmark"]
    paths = inputs(args.pairs, config)
    warmup, cycles = config["warmup"], args.cycles

    cold = []
    for path in paths[:config["cold_trials"]]:
        start = perf_counter_ns()
        pipe = fp.load_deploy(args.model, args.backend)
        with pipe.worker() as w:
            pipe.predict(path, w)
        cold.append((perf_counter_ns() - start) / 1e6)

    pipe = fp.load_deploy(args.model, args.backend)
    samples, py_cpu, node_cpu, node_rss = [], 0, 0.0, 0
    with pipe.worker() as worker:
        pid = worker.pid if worker is not None else None
        for path in paths[:warmup]:
            pipe.predict(path, worker)
        node_start = cpu_seconds(pid) if pid else 0.0
        trials = paths[warmup:] * cycles
        for path in trials:
            c0 = process_time_ns()
            _, clock = pipe.predict(path, worker)
            py_cpu += process_time_ns() - c0
            samples.append(clock)
        if pid:
            node_cpu = cpu_seconds(pid) - node_start
            node_rss = rss_bytes(pid)
    total = np.array([s["total"] for s in samples]) / 1e6
    peak_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
    if sys.platform.startswith("linux"):
        peak_self *= 1024
    wall_s = total.sum() / 1e3
    summary = {
        "model": args.model, "backend": args.backend, "trials": len(samples), "warmup": warmup,
        "distinct_images": len(paths) - warmup, "cycles": cycles,
        "median_ms": float(np.median(total)), "p95_ms": float(np.quantile(total, 0.95)),
        "mean_ms": float(total.mean()), "min_ms": float(total.min()),
        "block_median_min_ms": float(min(np.median(c) for c in np.array_split(total, 3))),
        "block_median_max_ms": float(max(np.median(c) for c in np.array_split(total, 3))),
        "throughput_img_s": float(len(total) / wall_s),
        "cold_session_first_image_median_ms": statistics.median(cold), "cold_ms": cold,
        "python_peak_rss_mib": peak_self / 2**20, "node_rss_mib": node_rss / 2**20,
        "combined_peak_rss_upper_bound_mib": (peak_self + node_rss) / 2**20,
        "cpu_percent_one_core_approx": 100 * (py_cpu / 1e9 + node_cpu) / wall_s,
        "energy_j_per_image": None,
    }
    summary.update({f"{s}_median_ms": float(np.median([c.get(s, 0) for c in samples])) / 1e6 for s in STAGES})
    args.out.write_text(json.dumps({"summary": summary, "raw_ns": samples}) + "\n")


def environment():
    import cdch_ferris
    import sklearn
    import torch
    sh = lambda *c: subprocess.run(c, capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {
        "cpu": sh("sysctl", "-n", "machdep.cpu.brand_string") or platform.processor(),
        "logical_cpus": os.cpu_count(), "memory_bytes": int(sh("sysctl", "-n", "hw.memsize") or 0),
        "os": f"{platform.system()} {platform.mac_ver()[0] or platform.release()}", "machine": platform.machine(),
        "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "sklearn": sklearn.__version__, "node": sh("node", "--version"), "rustc": sh("rustc", "--version"),
        "cdch_ferris": cdch_ferris.build_info(),
        "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                                      "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--backend")
    parser.add_argument("--cycles", type=int, default=5, help="passes over the 60 timed holdout images")
    args = parser.parse_args()
    if args.model:
        run_one(args)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "environment.json").write_text(json.dumps(environment(), indent=2) + "\n")
    jobs = [(m, b) for m in MODELS_CDCH for b in ("canonical_js", "rust_matrix", "rust")]
    jobs += [(m, "rust") for m in MODELS_OTHER]  # no CDCH: backend-independent
    rows = []
    for model, backend in jobs:
        dest = args.out / f"{model}__{backend}.json"
        subprocess.run([sys.executable, __file__, "--pairs", str(args.pairs), "--out", str(dest),
                        "--model", model, "--backend", backend, "--cycles", str(args.cycles)], check=True)
        s = json.loads(dest.read_text())["summary"]
        rows.append(s)
        print(f"{model:<30} {backend:<13} median {s['median_ms']:8.3f} ms  p95 {s['p95_ms']:8.3f} ms  "
              f"cold {s['cold_session_first_image_median_ms']:8.1f} ms", flush=True)
    import csv
    keys = list(dict.fromkeys(k for r in rows for k in r if k != "cold_ms"))
    with (args.out / "pipeline_benchmarks.csv").open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
