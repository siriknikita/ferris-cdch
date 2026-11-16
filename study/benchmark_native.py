"""Run isolated CPU inference benchmarks on the verified pilot holdout PNGs."""
import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns, process_time_ns

import joblib
import numpy as np
import torch

from build_pairs import ROOT, sha
from experiment import write_csv
from native_features import verify_js
from native_pipeline import load_deploy

HERE = Path(__file__).resolve().parent
STAGES = ("decode", "js", "ipc", "python_pack", "python_unpack", "node_json_parse",
          "node_reply_pack", "roundtrip", "feature", "frequency", "rgb_preprocess", "cnn", "classifier_other")


def memory(pid):
    text = Path(f"/proc/{pid}/status").read_text().splitlines()
    return {parts[0].rstrip(":"): int(parts[1]) for ln in text if ln.startswith(("VmRSS:", "VmHWM:"))
            for parts in [ln.split()]}


def node_cpu(pid):
    if not pid:
        return 0.0
    parts = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].strip().split()
    return (int(parts[11]) + int(parts[12])) / os.sysconf("SC_CLK_TCK")


def per_model(args, config, inputs):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    verify_js()
    decisions = json.loads((args.run / "SOURCE_DECISIONS.json").read_text())
    selected = decisions["selected"]
    key = "C" if selected.startswith("cdch_D_") else selected.split("_")[1]

    def load():
        return load_deploy(args.model, args.run, ROOT / "results/rgb.pt", selected, key)

    cold = []
    for path in inputs[:config["cold_trials"]]:
        start = perf_counter_ns()
        pipe = load()
        with pipe.worker() as worker:
            pipe.predict(path, worker)
            cold.append((perf_counter_ns() - start) / 1e6)
    pipe = load()
    warmup = config["warmup"]
    sample, variability = [], []
    steady_rss = 0
    with pipe.worker() as worker:
        pid = worker.pid if worker is not None else None
        for path in inputs[:warmup]:
            pipe.predict(path, worker)
        node_start = node_cpu(pid)
        py_cpu = 0
        trials = inputs[warmup:]
        for path in trials:
            cpu_start = process_time_ns()
            _, clock = pipe.predict(path, worker)
            py_cpu += process_time_ns() - cpu_start
            child = memory(pid) if pid else {}
            me = memory(os.getpid())
            steady_rss = max(steady_rss, me["VmRSS"] + child.get("VmRSS", 0))
            sample.append(clock)
        combined_hwm_bound = memory(os.getpid())["VmHWM"] + (memory(pid)["VmHWM"] if pid else 0)
        cpu = 100 * ((py_cpu / 1e9) + (node_cpu(pid) - node_start)) / (sum(r["total"] for r in sample) / 1e9)
    total = np.asarray([row["total"] for row in sample]) / 1e6
    variability = [np.median(chunk) for chunk in np.array_split(total, 3)]
    cpu_name = next((line.split(":", 1)[-1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("model name")), platform.machine())
    result = {
        "model": args.model, "trials": len(sample), "warmup": warmup, "cold_trials": len(cold),
        "median_ms": float(np.median(total)), "p95_ms": float(np.quantile(total, 0.95)),
        "block_median_min_ms": float(min(variability)), "block_median_max_ms": float(max(variability)),
        "throughput_img_s": 1000 * len(total) / total.sum(),
        "cold_session_first_image_median_ms": statistics.median(cold),
        "sampled_combined_peak_rss_mib": steady_rss / 1024,
        "sum_of_process_rss_highwater_upper_bound_mib": combined_hwm_bound / 1024,
        "cpu_percent_one_core_approx": cpu, "energy_j_per_image": "",
        "cpu": cpu_name, "affinity": ",".join(map(str, sorted(os.sched_getaffinity(0)))),
        "python": platform.python_version(), "platform": platform.platform(),
        "node": subprocess.check_output(["node", "--version"], text=True).strip(),
        "torch": str(torch.__version__), "batch": 1, "torch_threads": torch.get_num_threads(),
        "encoding": "png", "input_px": 128,
    }
    result.update({f"{stage}_median_ms": float(np.median([r.get(stage, 0) for r in sample])) / 1e6
                   for stage in STAGES})
    with args.out.open("x") as stream:
        json.dump({"summary": result, "raw_ns": sample, "cold_ms": cold}, stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pairs", "run", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--model")
    args = parser.parse_args()
    config = json.loads((HERE / "config/native.json").read_text())["benchmark"]
    manifest = args.pairs / "manifest.json"
    if sha(manifest) != sha(ROOT / "results/manifest.json"):
        raise ValueError("Wrong benchmark manifest")
    rows = [r for r in json.loads(manifest.read_text()) if r["split"] == "test" and r["variant"] != "lr"]
    order = np.random.default_rng(1611).permutation(len(rows))[:config["warmup"] + config["trials"]]
    inputs = [args.pairs / rows[i]["path"] for i in order]
    for i in order:
        if sha(args.pairs / rows[i]["path"]) != rows[i]["sha256"]:
            raise ValueError("Benchmark pair failed checksum")
    if args.model:
        per_model(args, config, inputs)
        return
    args.out.mkdir(parents=True)
    with (args.run / "model_inventory.csv").open(newline="") as stream:
        inventory = list(csv.DictReader(stream))
    result = []
    for row in inventory:
        dest = args.out / f"{row['model']}.json"
        subprocess.run([sys.executable, __file__, "--pairs", str(args.pairs), "--run", str(args.run),
                        "--out", str(dest), "--model", row["model"]], check=True)
        result.append({**json.loads(dest.read_text())["summary"], **row})
        print(f"{row['model']}: {result[-1]['median_ms']:.2f} ms", flush=True)
    write_csv(args.out / "EFFICIENCY_BENCHMARKS.csv", result)


if __name__ == "__main__":
    main()
