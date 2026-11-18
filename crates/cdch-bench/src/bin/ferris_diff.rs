//! Differential test: canonical JavaScript (archived oracle running ip-1.js) versus
//! the Rust reference port and the optimized engine, on identical decoded pixels.
//!
//! ```text
//! ferris-diff --oracle-dir oracle --out reports/differential \
//!     [--div2k DIR] [--png-dir LABEL=DIR]... [--png-limit N] [--mutants]
//! ```
//! `--mutants` also evaluates deliberately wrong variants (see `cdch_bench::mutants`) and
//! records which fixtures expose each one, measuring the harness's sensitivity.

#![forbid(unsafe_code)]

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use cdch_bench::mutants;
use cdch_bench::synth::{self, Case};
use cdch_bench::{host_info, png_files, read_png, CentroidOracle, Oracle, OwnedRgb};
use cdch_core::engine::{checkpoint_bin, CHECKPOINTS};
use cdch_core::reference::{self, Trig};
use cdch_core::{Center, Params, SamplingPlan, Workspace, BINS};
use serde_json::{json, Value};

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

struct Args {
    oracle_dir: PathBuf,
    out: PathBuf,
    div2k: Option<PathBuf>,
    png_dirs: Vec<(String, PathBuf)>,
    png_limit: usize,
    skip_synthetic: bool,
    mutants: bool,
    centroid: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        oracle_dir: PathBuf::from("oracle"),
        out: PathBuf::from("reports/differential"),
        div2k: None,
        png_dirs: vec![],
        png_limit: usize::MAX,
        skip_synthetic: false,
        mutants: false,
        centroid: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        match k.as_str() {
            "--oracle-dir" => a.oracle_dir = it.next().unwrap().into(),
            "--out" => a.out = it.next().unwrap().into(),
            "--div2k" => a.div2k = Some(it.next().unwrap().into()),
            "--png-dir" => {
                let v = it.next().unwrap();
                let (label, dir) = v.split_once('=').expect("--png-dir LABEL=DIR");
                a.png_dirs.push((label.to_string(), dir.into()));
            }
            "--png-limit" => a.png_limit = it.next().unwrap().parse().unwrap(),
            "--skip-synthetic" => a.skip_synthetic = true,
            "--mutants" => a.mutants = true,
            "--centroid" => a.centroid = true,
            other => panic!("unknown argument {other}"),
        }
    }
    a
}

fn natural_cases(args: &Args) -> Vec<Case> {
    let mut out = Vec::new();
    if let Some(dir) = &args.div2k {
        for (i, path) in png_files(dir).iter().enumerate() {
            let img = read_png(path).expect("decode DIV2K");
            let stem = path.file_stem().unwrap().to_string_lossy().to_string();
            let (w, h) = (img.width, img.height);
            // Pilot-style centre crop (128 px), archived centre.
            let c = img.crop((w - 128) / 2, (h - 128) / 2, 128, 128);
            out.push(Case {
                id: format!("div2k_{stem}_crop128"),
                category: "natural",
                img: c,
                center: Some((64.0, 64.0)),
                angle_step: 1,
            });
            if i % 10 == 0 {
                let c = img.crop((w - 500) / 2, (h - 333) / 2, 500, 333);
                out.push(Case {
                    id: format!("div2k_{stem}_crop500x333"),
                    category: "natural",
                    img: c,
                    center: Some((250.0, 166.0)),
                    angle_step: 1,
                });
                let c = img.crop(0, 0, 255, 255);
                out.push(Case {
                    id: format!("div2k_{stem}_crop255_halfcentre"),
                    category: "natural",
                    img: c,
                    center: Some((127.5, 127.5)),
                    angle_step: 1,
                });
            }
            if i % 50 == 0 {
                out.push(Case {
                    id: format!("div2k_{stem}_full_{w}x{h}"),
                    category: "natural_full",
                    img: img.clone(),
                    center: Some(((w / 2) as f64, (h / 2) as f64)),
                    angle_step: 1,
                });
            }
        }
    }
    for (label, dir) in &args.png_dirs {
        for path in png_files(dir).into_iter().take(args.png_limit) {
            let img = read_png(&path).expect("decode PNG");
            let rel = path
                .strip_prefix(dir)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();
            let center = Some(((img.width / 2) as f64, (img.height / 2) as f64));
            out.push(Case {
                id: format!("{label}:{rel}"),
                category: "pilot_pairs",
                img,
                center,
                angle_step: 1,
            });
        }
    }
    out
}

fn run_case(oracle: &mut Oracle, case: &Case, with_mutants: bool) -> Value {
    let img: &OwnedRgb = &case.img;
    let view = img.view();
    let (w, h) = (img.width, img.height);
    let center = match case.center {
        Some((x, y)) => Center::new(x, y),
        None => Center::midpoint(w, h),
    };
    let params = Params {
        center,
        angle_step: case.angle_step,
    };

    let js = oracle
        .compute(img, case.center, case.angle_step, true)
        .expect("oracle");
    let trace = js.trace.as_ref().expect("trace");

    let t0 = Instant::now();
    let (ref_m, ref_traces) = reference::cdch_traced(&view, &params, Trig::V8Table).unwrap();
    let ref_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = Instant::now();
    let plan = SamplingPlan::new(w, h, params).unwrap();
    let plan_ms = t1.elapsed().as_secs_f64() * 1e3;
    let mut ws = Workspace::new();
    let t2 = Instant::now();
    let eng_m = plan.matrix(&view, &mut ws).unwrap();
    let eng_ms = t2.elapsed().as_secs_f64() * 1e3;
    let cp = plan.checkpoints(&view, &mut ws).unwrap();
    // Single-pass counts + matrix (used by the public facade).
    let mut fused_counts = vec![0u32; plan.n_rays() * BINS];
    let mut fused_m = vec![0f64; plan.n_rays() * BINS];
    plan.histograms_and_matrix_into(&view, &mut ws, &mut fused_counts, &mut fused_m)
        .unwrap();
    // The public API (`ferris-cdch`) with the same resolved centre.
    let facade = ferris_cdch::compute(
        &view,
        &ferris_cdch::CdchConfig {
            center: ferris_cdch::CenterSpec::Explicit {
                x: center.x,
                y: center.y,
            },
            angle_step: case.angle_step,
        },
    )
    .expect("facade accepts every differential case");
    let (_, libm_traces) = reference::cdch_traced(&view, &params, Trig::Libm).unwrap();
    let libm_m = reference::cdch_with(&view, &params, Trig::Libm, None).unwrap();

    let rows = params.n_rays();
    let mut mismatches: Vec<String> = Vec::new();
    let center_ok = same(js.center.0, center.x) && same(js.center.1, center.y);
    if !center_ok {
        mismatches.push(format!(
            "centre js={:?} rust=({}, {})",
            js.center, center.x, center.y
        ));
    }
    let shape_ok = js.rows == rows && js.matrix.len() == rows * BINS;
    if !shape_ok {
        mismatches.push(format!("shape js rows={} rust rows={rows}", js.rows));
    }
    let harness_ok = trace.matrix.len() == js.matrix.len()
        && trace
            .matrix
            .iter()
            .zip(&js.matrix)
            .all(|(a, b)| same(*a, *b));
    if !harness_ok {
        mismatches.push("tracing harness matrix differs from archived oracle".into());
    }

    // Coordinates and multiplicities.
    let ray_lengths_ok = trace.ray_lengths.len() == rows
        && trace
            .ray_lengths
            .iter()
            .zip(&ref_traces)
            .all(|(n, t)| *n == t.coords.len());
    let mut coords_ok = ray_lengths_ok;
    let mut offset = 0usize;
    let (mut total_samples, mut duplicate_samples, mut consecutive_dups) = (0usize, 0usize, 0usize);
    let mut counts_ok = ray_lengths_ok;
    let mut first_coord_mismatch: Option<String> = None;
    for (i, t) in ref_traces.iter().enumerate() {
        let n = *trace.ray_lengths.get(i).unwrap_or(&0);
        let js_coords: Vec<(i64, i64)> = (0..n)
            .map(|k| {
                (
                    trace.coords[2 * (offset + k)] as i64,
                    trace.coords[2 * (offset + k) + 1] as i64,
                )
            })
            .collect();
        offset += n;
        if js_coords != t.coords {
            coords_ok = false;
            if first_coord_mismatch.is_none() {
                first_coord_mismatch = Some(format!("ray {i} (angle {})", t.angle));
            }
        }
        total_samples += t.coords.len();
        let distinct: HashSet<_> = t.coords.iter().collect();
        duplicate_samples += t.coords.len() - distinct.len();
        consecutive_dups += t.coords.windows(2).filter(|p| p[0] == p[1]).count();
        // Integer counts recovered exactly from the JS CDF: c[b] = cdf[b] * n.
        if js.matrix.len() >= (i + 1) * BINS {
            let row = &js.matrix[i * BINS..(i + 1) * BINS];
            let mut prev = 0i64;
            for (b, &value) in row.iter().enumerate() {
                let c = if n == 0 { 0.0 } else { value * n as f64 };
                let ci = c.round();
                if (c - ci).abs() > 1e-6 || (ci as i64 - prev) as u32 != t.counts[b] {
                    counts_ok = false;
                }
                prev = ci as i64;
            }
        }
    }
    if !coords_ok {
        mismatches.push(format!(
            "coordinates differ, first at {}",
            first_coord_mismatch.unwrap_or_default()
        ));
    }
    if !counts_ok {
        mismatches.push("histogram counts differ".into());
    }

    let cmp = |m: &[f64]| -> (usize, f64) {
        let mut n = 0;
        let mut max_abs = 0f64;
        for (a, b) in m.iter().zip(&js.matrix) {
            if !same(*a, *b) {
                n += 1;
                if a.is_finite() && b.is_finite() {
                    max_abs = max_abs.max((a - b).abs());
                } else {
                    max_abs = f64::INFINITY;
                }
            }
        }
        (n, max_abs)
    };
    let (ref_diff, ref_max) = if shape_ok {
        cmp(ref_m.as_slice())
    } else {
        (usize::MAX, f64::NAN)
    };
    let (eng_diff, eng_max) = if shape_ok {
        cmp(eng_m.as_slice())
    } else {
        (usize::MAX, f64::NAN)
    };
    let (libm_diff, _) = if shape_ok {
        cmp(libm_m.as_slice())
    } else {
        (usize::MAX, f64::NAN)
    };
    if ref_diff != 0 {
        mismatches.push(format!(
            "reference matrix: {ref_diff} values differ (max abs {ref_max:e})"
        ));
    }
    if eng_diff != 0 {
        mismatches.push(format!(
            "engine matrix: {eng_diff} values differ (max abs {eng_max:e})"
        ));
    }
    let mut cp_diff = 0;
    if shape_ok {
        for i in 0..rows {
            for j in 0..CHECKPOINTS {
                if !same(
                    cp.values[i * CHECKPOINTS + j],
                    js.matrix[i * BINS + checkpoint_bin(j)],
                ) {
                    cp_diff += 1;
                }
            }
        }
    }
    if cp_diff != 0 {
        mismatches.push(format!("checkpoint matrix: {cp_diff} values differ"));
    }
    let (fused_diff, _) = if shape_ok {
        cmp(&fused_m)
    } else {
        (usize::MAX, f64::NAN)
    };
    if fused_diff != 0 {
        mismatches.push(format!("single-pass matrix: {fused_diff} values differ"));
    }
    let (facade_diff, _) = if shape_ok {
        cmp(facade.matrix().as_slice())
    } else {
        (usize::MAX, f64::NAN)
    };
    if facade_diff != 0 {
        mismatches.push(format!("ferris-cdch matrix: {facade_diff} values differ"));
    }
    let ref_counts: Vec<u32> = ref_traces.iter().flat_map(|t| t.counts.clone()).collect();
    let facade_counts_ok = facade.counts_flat() == ref_counts.as_slice()
        && fused_counts == ref_counts
        && facade
            .ray_lengths()
            .iter()
            .zip(&trace.ray_lengths)
            .all(|(a, b)| *a as usize == *b);
    if !facade_counts_ok {
        mismatches.push("single-pass / ferris-cdch counts or ray lengths differ".into());
    }
    let libm_coord_rays = ref_traces
        .iter()
        .zip(&libm_traces)
        .filter(|(a, b)| a.coords != b.coords)
        .count();
    let nan_rows = (0..rows)
        .filter(|&i| js.matrix.get(i * BINS).is_some_and(|v| v.is_nan()))
        .count();
    let detected: Vec<String> = if with_mutants && shape_ok {
        mutants::ALL
            .iter()
            .filter(|&&m| {
                let mm = mutants::cdch(&view, &params, m);
                mm.len() != js.matrix.len() || mm.iter().zip(&js.matrix).any(|(a, b)| !same(*a, *b))
            })
            .map(|m| format!("{m:?}"))
            .collect()
    } else {
        Vec::new()
    };

    json!({
        "id": case.id, "category": case.category, "width": w, "height": h,
        "center": [center.x, center.y], "center_from_oracle_default": case.center.is_none(),
        "angle_step": case.angle_step, "rays": rows,
        "samples": total_samples, "unique_pixels": plan.unique_pixels().len(),
        "within_ray_duplicate_samples": duplicate_samples, "consecutive_duplicate_samples": consecutive_dups,
        "nan_rows": nan_rows,
        "checks": {
            "centre": center_ok, "shape": shape_ok, "tracing_harness_equals_oracle": harness_ok,
            "ray_lengths": ray_lengths_ok, "coordinates": coords_ok, "counts": counts_ok,
            "reference_matrix_bitwise": ref_diff == 0, "engine_matrix_bitwise": eng_diff == 0,
            "checkpoints_bitwise": cp_diff == 0,
            "single_pass_matrix_bitwise": fused_diff == 0,
            "ferris_cdch_matrix_bitwise": facade_diff == 0,
            "ferris_cdch_counts": facade_counts_ok,
        },
        "pass": mismatches.is_empty(),
        "mismatches": mismatches,
        "diagnostic_libm": {"rays_with_different_coordinates": libm_coord_rays, "matrix_values_different": libm_diff},
        "mutants_detected": detected,
        "timing_ms": {"js_oracle": js.js_ns as f64 / 1e6, "rust_reference_traced": ref_ms, "rust_plan_build": plan_ms, "rust_engine_matrix": eng_ms},
    })
}

fn main() {
    let args = parse_args();
    let mut cases = if args.skip_synthetic {
        vec![]
    } else {
        synth::cases()
    };
    cases.extend(natural_cases(&args));
    eprintln!("ferris-diff: {} cases", cases.len());
    let mut oracle = Oracle::spawn(&args.oracle_dir).expect("spawn node oracle");
    let mut results = Vec::with_capacity(cases.len());
    for (i, case) in cases.iter().enumerate() {
        let r = run_case(&mut oracle, case, args.mutants);
        if !r["pass"].as_bool().unwrap() {
            eprintln!("MISMATCH {}: {}", case.id, r["mismatches"]);
        }
        if (i + 1) % 50 == 0 {
            eprintln!("  {}/{}", i + 1, cases.len());
        }
        results.push(r);
    }
    let passed = results
        .iter()
        .filter(|r| r["pass"].as_bool().unwrap())
        .count();
    let mut by_cat: std::collections::BTreeMap<String, (usize, usize, usize, usize)> =
        Default::default();
    let mut libm_cases = 0;
    let mut libm_rays = 0u64;
    for r in &results {
        let e = by_cat
            .entry(r["category"].as_str().unwrap().to_string())
            .or_default();
        e.0 += 1;
        e.1 += r["pass"].as_bool().unwrap() as usize;
        e.2 += r["samples"].as_u64().unwrap() as usize;
        e.3 += r["within_ray_duplicate_samples"].as_u64().unwrap() as usize;
        let lr = r["diagnostic_libm"]["rays_with_different_coordinates"]
            .as_u64()
            .unwrap();
        libm_rays += lr;
        libm_cases += (lr > 0) as usize;
    }
    let mutant_summary: serde_json::Map<String, Value> = mutants::ALL
        .iter()
        .map(|m| {
            let name = format!("{m:?}");
            let hits: Vec<&str> = results
                .iter()
                .filter(|r| r["mutants_detected"].as_array().is_some_and(|a| a.iter().any(|x| x == name.as_str())))
                .map(|r| r["id"].as_str().unwrap())
                .collect();
            (name, json!({"cases_detecting": hits.len(), "examples": hits.iter().take(5).collect::<Vec<_>>()}))
        })
        .collect();
    let summary = json!({
        "mutants_evaluated": args.mutants,
        "mutant_detection": mutant_summary,
        "cases": results.len(), "passed": passed, "failed": results.len() - passed,
        "total_samples": results.iter().map(|r| r["samples"].as_u64().unwrap()).sum::<u64>(),
        "total_matrix_values": results.iter().map(|r| r["rays"].as_u64().unwrap() * BINS as u64).sum::<u64>(),
        "libm_variant_cases_with_coordinate_differences": libm_cases,
        "libm_variant_rays_with_coordinate_differences": libm_rays,
        "by_category": by_cat.iter().map(|(k, v)| (k.clone(), json!({"cases": v.0, "passed": v.1, "samples": v.2, "within_ray_duplicate_samples": v.3}))).collect::<serde_json::Map<_, _>>(),
    });
    let centroid_report = if args.centroid {
        Some(centroid_cases(&args, &cases))
    } else {
        None
    };
    let centroid_failed = centroid_report
        .as_ref()
        .is_some_and(|r| r["failed"].as_u64() != Some(0));
    std::fs::create_dir_all(&args.out).unwrap();
    let report = json!({"host": host_info(), "canonical_sha256": cdch_core::CANONICAL_SHA256, "summary": summary, "centroid": centroid_report, "cases": results});
    std::fs::write(
        args.out.join("differential_results.json"),
        serde_json::to_string_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("{}", serde_json::to_string_pretty(&summary).unwrap());
    if let Some(c) = &report["centroid"].as_object() {
        println!(
            "centroid: {} comparisons, {} failed",
            c["comparisons"], c["failed"]
        );
    }
    if passed != results.len() || centroid_failed {
        std::process::exit(1);
    }
}

/// `findIntensityCenter` of the unmodified `ip-1.js` versus `ferris_cdch::intensity_centroid`
/// on every fixture up to 512x512 pixels, for intensity ranges that cover the full range,
/// narrow bands, single levels, the centre pixel's own level and ranges matching nothing.
fn centroid_cases(args: &Args, cases: &[Case]) -> Value {
    let mut oracle = CentroidOracle::spawn(&args.oracle_dir).expect("spawn centroid oracle");
    let (mut comparisons, mut failed, mut none_cases) = (0u64, 0u64, 0u64);
    let mut failures = Vec::new();
    for case in cases
        .iter()
        .filter(|c| c.img.width * c.img.height <= 512 * 512)
    {
        let view = case.img.view();
        let (w, h) = (case.img.width, case.img.height);
        let [r, g, b] = view.rgb(w / 2, h / 2);
        let own = ferris_cdch::luma(r, g, b);
        let ranges = [
            (0u8, 255u8),
            (0, 0),
            (255, 255),
            (100, 180),
            (250, 255),
            (own, own),
            (own.saturating_sub(3), own.saturating_add(3)),
            (128, 128),
        ];
        for (min, max) in ranges {
            let js = oracle
                .centroid(&case.img, min, max)
                .expect("centroid oracle");
            let rust = ferris_cdch::intensity_centroid(&view, min, max)
                .expect("valid range")
                .map(|c| (c.x, c.y));
            comparisons += 1;
            none_cases += js.is_none() as u64;
            let ok = match (js, rust) {
                (Some(a), Some(b)) => same(a.0, b.0) && same(a.1, b.1),
                (None, None) => true,
                _ => false,
            };
            if !ok {
                failed += 1;
                failures.push(json!({"id": case.id, "min": min, "max": max, "js": format!("{js:?}"), "rust": format!("{rust:?}")}));
                eprintln!(
                    "CENTROID MISMATCH {} [{min}, {max}]: js {js:?} rust {rust:?}",
                    case.id
                );
            }
        }
    }
    json!({"comparisons": comparisons, "failed": failed, "no_pixel_cases": none_cases, "failures": failures})
}
