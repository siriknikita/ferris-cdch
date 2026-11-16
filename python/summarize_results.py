"""Collect the headline numbers behind the RUST_* reports into reports/results_summary.json
and print the Markdown tables used in them. Reads only files under ../reports and ../study.

    python summarize_results.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"
MODELS_CDCH = ["cdch_A_logistic", "cdch_B_linear_svm", "cdch_C_logistic", "cdch_D_conv",
               "frequency_cdch", "frequency_cdch_score_fusion"]


def rows(path):
    with path.open(newline="") as s:
        return list(csv.DictReader(s))


def main():
    out = {}
    diff = json.loads((REP / "differential/differential_results.json").read_text())
    out["differential"] = {k: diff["summary"][k] for k in ("cases", "passed", "failed", "total_samples", "total_matrix_values",
                                                         "libm_variant_cases_with_coordinate_differences",
                                                         "libm_variant_rays_with_coordinate_differences")}
    out["differential"]["by_category"] = diff["summary"]["by_category"]
    out["mutants"] = {k: v["cases_detecting"] for k, v in diff["summary"]["mutant_detection"].items()}

    comp = json.loads((REP / "classifier_compatibility/summary.json").read_text())
    out["classifier_images"] = comp["encodings"]
    fa = rows(REP / "classifier_compatibility/feature_agreement.csv")
    out["feature_agreement"] = [r for r in fa]
    sa = rows(REP / "classifier_compatibility/score_agreement.csv")
    rust_js = [r for r in sa if r["comparison"] == "rust_vs_js"]
    out["scores_rust_vs_js"] = {
        "max_abs_test": max(float(r["test_max_abs_score_diff"]) for r in rust_js),
        "test_label_flips": sum(int(r["test_label_flips"] or 0) for r in rust_js),
        "all_split_label_flips": sum(int(r["label_flips"] or 0) for r in rust_js),
    }
    arch = [r for r in sa if r["comparison"] == "rust_vs_archived_predictions"]
    cdch_arch = [r for r in arch if r["model"].startswith("cdch_")]
    out["scores_rust_vs_archived_cdch_models"] = {
        "max_abs": max(float(r["max_abs_score_diff"]) for r in cdch_arch),
        "test_label_flips": sum(int(r["test_label_flips"] or 0) for r in cdch_arch)}
    auc = rows(REP / "classifier_compatibility/auroc_comparison.csv")
    out["auroc"] = {
        "rows": len(auc),
        "rows_any_backend_differs": sum(len({r[f"auroc_{b}"] for b in ("js", "rust_m", "rust", "rust_n")}) > 1 for r in auc),
        "rows_js_equals_archived": sum(abs(float(r["auroc_js"]) - float(r["auroc_archived"])) < 1e-12 for r in auc),
    }
    headline = ["cdch_A_logistic", "cdch_B_linear_svm", "cdch_C_logistic", "cdch_D_conv", "frequency_cdch", "frequency", "rgb"]
    out["auroc_headline_png_original_vs_ai"] = {
        r["model"]: {k: float(r[k]) for k in ("auroc_js", "auroc_rust_m", "auroc_rust", "auroc_rust_n", "auroc_archived")}
        for r in auc if r["encoding"] == "png" and r["task"] == "original_vs_ai" and r["model"] in headline}

    pipe = {(r["model"], r["backend"]): r for r in rows(REP / "perf/pipelines/pipeline_benchmarks.csv")}
    out["pipelines"] = {f"{m}|{b}": {k: float(v) for k, v in r.items() if k.endswith("_ms") or k in (
        "throughput_img_s", "combined_peak_rss_upper_bound_mib", "python_peak_rss_mib", "node_rss_mib",
        "cpu_percent_one_core_approx")} for (m, b), r in pipe.items()}
    out["environment"] = json.loads((REP / "perf/pipelines/environment.json").read_text())
    micro = json.loads((REP / "perf/rust_microbench.json").read_text())
    out["rust_microbench_us"] = {b["name"]: b["median_us"] for b in micro["single_image"] if "median_us" in b}
    out["rust_batch_img_s"] = {f'{b["representation"]}|{b["threads"]}': b["median_img_s"] for b in micro["batch"]}
    out["components"] = json.loads((REP / "perf/components.json").read_text())
    out["cli_throughput"] = rows(REP / "perf/cli_throughput.csv")
    # ---- Speedups S = T_baseline / T_rust (same machine, same session) ----
    mb = out["rust_microbench_us"]
    comp = out["components"]
    def stage(m, b, s):
        return float(pipe[(m, b)][f"{s}_median_ms"])
    sp = []
    def add(scope, what, base_label, base, rust_label, rust):
        sp.append(dict(scope=scope, comparison=what, baseline=base_label, baseline_value=base,
                       rust=rust_label, rust_value=rust, speedup=base / rust))
    js_core_ms = stage("cdch_A_logistic", "canonical_js", "js")
    add("core", "CDCH matrix, 128^2 (ms)", "ip-1.js in Node (js stage)", js_core_ms, "engine matrix_into", mb["core_engine_matrix_into"] / 1e3)
    add("core", "CDCH reduced output, 128^2 (ms)", "ip-1.js in Node (js stage)", js_core_ms, "engine checkpoints_into", mb["core_engine_checkpoints_into"] / 1e3)
    add("core", "CDCH matrix, 128^2 (ms)", "ip-1.js in Node (js stage)", js_core_ms, "literal reference port", mb["core_reference_matrix"] / 1e3)
    for kind, np_model, rust_bench in (("A", "cdch_A_logistic", "features_A_from_checkpoints"),
                                       ("B", "cdch_B_linear_svm", "features_B_from_matrix"),
                                       ("C", "cdch_C_logistic", "features_C_from_checkpoints")):
        add("features", f"representation {kind} from CDCH data (ms)", "NumPy (feature stage)",
            stage(np_model, "canonical_js", "feature"), rust_bench, mb[rust_bench] / 1e3)
    for m in MODELS_CDCH:
        add("cdch_stage", f"{m}: pixels -> representation in Python (ms)", "canonical_js roundtrip+NumPy",
            stage(m, "canonical_js", "cdch_total"), "rust fused call", stage(m, "rust", "cdch_total"))
    for m in MODELS_CDCH:
        add("pipeline", f"{m}: end-to-end median (ms)", "canonical_js", stage(m, "canonical_js", "total"),
            "rust", stage(m, "rust", "total"))
    js_tp = comp["js_worker_matrix_throughput_img_s"]
    for b in comp["batch_from_arrays"]:
        if b["kind"] == "A":
            add("batch", f"A features from decoded arrays, {b['threads']} threads (img/s)", "JS worker, 1 process",
                1 / js_tp, f"batch_features threads={b['threads']}", 1 / b["median_img_s"])
    cli = {}
    for r in out["cli_throughput"]:
        cli.setdefault(int(r["threads"]), []).append(float(r["img_per_s"]))
    for t, v in sorted(cli.items()):
        add("batch", f"decode + A + logistic, CLI {t} threads (img/s)", "canonical_js pipeline (img/s)",
            1 / float(pipe[("cdch_A_logistic", "canonical_js")]["throughput_img_s"]), f"cdch score --threads {t}",
            1 / (sorted(v)[len(v) // 2]))
    with (REP / "perf/speedups.csv").open("w", newline="") as s:
        w = csv.DictWriter(s, fieldnames=list(sp[0]))
        w.writeheader()
        w.writerows(sp)
    out["speedups"] = sp
    (REP / "results_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\n### Speedups\n")
    print("| Scope | Comparison | Baseline | Rust | S |")
    print("|---|---|---:|---:|---:|")
    for r in sp:
        print(f"| {r['scope']} | {r['comparison']} | {r['baseline_value']:.4g} | {r['rust_value']:.4g} | {r['speedup']:.1f}× |")

    # ---- Markdown tables ----
    print("\n### Pipelines (median / p95 ms, cold ms, img/s, peak RSS MiB, CPU %)\n")
    print("| Model | Backend | Median | p95 | Cold first image | Throughput | Peak RSS | CPU |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for (m, b), r in pipe.items():
        print(f"| {m} | {b} | {float(r['median_ms']):.3f} | {float(r['p95_ms']):.3f} | "
              f"{float(r['cold_session_first_image_median_ms']):.1f} | {float(r['throughput_img_s']):.0f} | "
              f"{float(r['combined_peak_rss_upper_bound_mib']):.0f} | {float(r['cpu_percent_one_core_approx']):.0f} |")
    print("\n### Stages (median ms)\n")
    stages = ["decode", "js", "ipc", "python_pack", "python_unpack", "node_json_parse", "node_reply_pack", "rust_matrix",
              "rust_fused", "feature", "frequency", "cnn", "classifier_other", "total"]
    print("| Model | Backend | " + " | ".join(stages) + " |")
    print("|---|---|" + "---:|" * len(stages))
    for (m, b), r in pipe.items():
        print(f"| {m} | {b} | " + " | ".join(f"{float(r[s + '_median_ms']):.3f}" for s in stages) + " |")
    print("\n### Feature agreement (png)\n")
    print("| Kind | Comparison | Values | Bitwise equal | Max abs | Max rel |")
    print("|---|---|---:|---:|---:|---:|")
    for r in fa:
        if r["encoding"] == "png":
            print(f"| {r['kind']} | {r['comparison']} | {r['values']} | {r['bitwise_equal']} | {float(r['max_abs']):.3g} | {float(r['max_rel']):.3g} |")
    print(json.dumps({k: out[k] for k in ("differential", "mutants", "scores_rust_vs_js", "scores_rust_vs_archived_cdch_models", "auroc")}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
