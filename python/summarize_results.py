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
    (REP / "results_summary.json").write_text(json.dumps(out, indent=2) + "\n")

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
