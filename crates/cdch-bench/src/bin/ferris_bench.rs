//! Rust-side microbenchmarks and throughput measurements.
//!
//! ```text
//! ferris-bench --images DIR [--warmup 50] [--trials 2000] [--out reports/perf/rust_microbench.json]
//!              [--only SUBSTRING]   # run matching single-image benchmarks only (for profiling)
//! ```
//! Times single-image operations on decoded 8-bit RGB buffers already in memory
//! (no decoding, no Python), cycling through the supplied images, then measures
//! batch throughput with 1..N rayon threads.

#![forbid(unsafe_code)]

use std::hint::black_box;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use cdch_bench::{host_info, peak_rss_bytes, png_files, quantile, read_png, OwnedRgb};
use cdch_core::engine::CHECKPOINTS;
use cdch_core::{reference, Engine, MatrixView, Params, SamplingPlan, Workspace, BINS};
use cdch_features::{FeatureExtractor, Representation};
use rayon::prelude::*;
use serde_json::{json, Value};

struct Args {
    images: PathBuf,
    warmup: usize,
    trials: usize,
    out: PathBuf,
    batch_images: usize,
    only: Option<String>,
}

static ONLY: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();

fn parse() -> Args {
    let mut a = Args {
        images: PathBuf::new(),
        warmup: 50,
        trials: 2000,
        out: PathBuf::from("reports/perf/rust_microbench.json"),
        batch_images: 4000,
        only: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        match k.as_str() {
            "--images" => a.images = it.next().unwrap().into(),
            "--warmup" => a.warmup = it.next().unwrap().parse().unwrap(),
            "--trials" => a.trials = it.next().unwrap().parse().unwrap(),
            "--out" => a.out = it.next().unwrap().into(),
            "--batch-images" => a.batch_images = it.next().unwrap().parse().unwrap(),
            "--only" => a.only = Some(it.next().unwrap()),
            o => panic!("unknown argument {o}"),
        }
    }
    a
}

/// Time `f(image_index)` per call; returns summary in microseconds.
fn bench(
    name: &str,
    n_images: usize,
    warmup: usize,
    trials: usize,
    mut f: impl FnMut(usize),
) -> Value {
    if let Some(Some(only)) = ONLY.get() {
        if !name.contains(only.as_str()) {
            return json!({"name": name, "skipped": true});
        }
    }
    for i in 0..warmup {
        f(i % n_images);
    }
    let mut us = Vec::with_capacity(trials);
    for i in 0..trials {
        let t = Instant::now();
        f(i % n_images);
        us.push(t.elapsed().as_nanos() as f64 / 1e3);
    }
    let median = quantile(&us, 0.5);
    let v = json!({
        "name": name, "trials": trials, "warmup": warmup,
        "median_us": median, "p95_us": quantile(&us, 0.95), "p05_us": quantile(&us, 0.05),
        "mean_us": us.iter().sum::<f64>() / us.len() as f64, "min_us": us.iter().cloned().fold(f64::INFINITY, f64::min),
        "throughput_img_s_at_median": 1e6 / median,
    });
    eprintln!(
        "{name:<44} median {median:>9.2} µs   p95 {:>9.2} µs",
        v["p95_us"].as_f64().unwrap()
    );
    v
}

fn main() {
    let args = parse();
    ONLY.set(args.only.clone()).unwrap();
    let images: Vec<OwnedRgb> = png_files(&args.images)
        .iter()
        .map(|p| read_png(p).unwrap())
        .collect();
    assert!(
        !images.is_empty(),
        "no PNG images in {}",
        args.images.display()
    );
    let (w, h) = (images[0].width, images[0].height);
    let images: Vec<OwnedRgb> = images
        .into_iter()
        .filter(|i| i.width == w && i.height == h)
        .collect();
    let n = images.len();
    eprintln!("ferris-bench: {n} images of {w}x{h}");
    let params = Params::archived(w, h);
    let (warm, trials) = (args.warmup, args.trials);
    let mut results = Vec::new();

    // Cold plan construction (geometry only).
    results.push(bench("plan_build", 1, 3, 200, |_| {
        black_box(SamplingPlan::new(w, h, params).unwrap());
    }));
    let plan = SamplingPlan::new(w, h, params).unwrap();
    let plan_info = json!({"rays": plan.n_rays(), "total_samples": plan.total_samples(), "unique_pixels": plan.unique_pixels().len(), "heap_bytes": plan.heap_bytes()});

    // Correctness-first reference port (allocates its output).
    results.push(bench(
        "core_reference_matrix",
        n,
        warm.min(10),
        trials.min(500),
        |i| {
            black_box(reference::cdch(&images[i].view(), &params).unwrap());
        },
    ));

    let mut ws = Workspace::new();
    let mut matrix = vec![0.0; plan.n_rays() * BINS];
    results.push(bench("core_engine_matrix_into", n, warm, trials, |i| {
        plan.matrix_into(&images[i].view(), &mut ws, &mut matrix)
            .unwrap();
        black_box(&matrix);
    }));
    results.push(bench("core_engine_matrix_alloc", n, warm, trials, |i| {
        black_box(plan.matrix(&images[i].view(), &mut ws).unwrap());
    }));
    let mut cp = vec![0.0; plan.n_rays() * CHECKPOINTS];
    results.push(bench(
        "core_engine_checkpoints_into",
        n,
        warm,
        trials,
        |i| {
            plan.checkpoints_into(&images[i].view(), &mut ws, &mut cp)
                .unwrap();
            black_box(&cp);
        },
    ));
    results.push(bench("core_engine_histograms_u32", n, warm, trials, |i| {
        black_box(plan.histograms(&images[i].view(), &mut ws).unwrap());
    }));
    let mut counts = vec![0u32; plan.n_rays() * BINS];
    results.push(bench(
        "core_engine_histograms_and_matrix_into",
        n,
        warm,
        trials,
        |i| {
            plan.histograms_and_matrix_into(&images[i].view(), &mut ws, &mut counts, &mut matrix)
                .unwrap();
            black_box((&counts, &matrix));
        },
    ));

    // Public API (`ferris-cdch`): plan cache lookup, allocation of the result, and
    // (for `compute`) counts plus ray lengths in addition to the matrix.
    let analyzer = ferris_cdch::Analyzer::new();
    let config = ferris_cdch::CdchConfig::archived();
    results.push(bench("ferris_cdch_analyzer_matrix", n, warm, trials, |i| {
        black_box(analyzer.matrix(&images[i].view(), &config).unwrap());
    }));
    results.push(bench(
        "ferris_cdch_analyzer_compute",
        n,
        warm,
        trials,
        |i| {
            black_box(analyzer.compute(&images[i].view(), &config).unwrap());
        },
    ));
    results.push(bench(
        "ferris_cdch_compute_cold_plan",
        n,
        warm.min(10),
        trials.min(300),
        |i| {
            black_box(ferris_cdch::compute(&images[i].view(), &config).unwrap());
        },
    ));

    // Feature construction alone, from precomputed matrices / checkpoints.
    let matrices: Vec<Vec<f64>> = images
        .iter()
        .map(|im| plan.matrix(&im.view(), &mut ws).unwrap().into_vec())
        .collect();
    let cps: Vec<Vec<f64>> = images
        .iter()
        .map(|im| plan.checkpoints(&im.view(), &mut ws).unwrap().values)
        .collect();
    let mut out_a = vec![0.0; 56];
    let mut out_b = vec![0.0; 98];
    let mut out_c = vec![0.0; 768];
    let mut out_r = vec![0.0; 42];
    let (mut sampled, mut scratch) = (Vec::new(), Vec::new());
    results.push(bench(
        "features_A_from_matrix(validate+stats)",
        n,
        warm,
        trials,
        |i| {
            cdch_features::compact(&MatrixView::new(360, &matrices[i]).unwrap(), &mut out_a)
                .unwrap();
            black_box(&out_a);
        },
    ));
    results.push(bench("features_validate_matrix", n, warm, trials, |i| {
        let validated =
            cdch_features::validate_matrix(&MatrixView::new(360, &matrices[i]).unwrap());
        black_box(&validated);
        validated.unwrap();
    }));
    results.push(bench("features_A_from_checkpoints", n, warm, trials, |i| {
        cdch_features::compact_from_checkpoints(&cps[i], &mut sampled, &mut scratch, &mut out_a)
            .unwrap();
        black_box(&out_a);
    }));
    results.push(bench("features_rich42_from_matrix", n, warm, trials, |i| {
        cdch_features::rich(&MatrixView::new(360, &matrices[i]).unwrap(), &mut out_r);
        black_box(&out_r);
    }));
    results.push(bench("features_B_from_matrix", n, warm, trials, |i| {
        cdch_features::enriched(&MatrixView::new(360, &matrices[i]).unwrap(), &mut out_b).unwrap();
        black_box(&out_b);
    }));
    results.push(bench("features_C_from_matrix", n, warm, trials, |i| {
        cdch_features::sector(&MatrixView::new(360, &matrices[i]).unwrap(), &mut out_c).unwrap();
        black_box(&out_c);
    }));
    results.push(bench("features_C_from_checkpoints", n, warm, trials, |i| {
        cdch_features::sector_from_checkpoints(&cps[i], &mut out_c).unwrap();
        black_box(&out_c);
    }));

    // End-to-end single image (pixels -> representation), fused vs unfused.
    let mut fx = FeatureExtractor::new();
    for (rep, out) in [
        (Representation::A, &mut out_a),
        (Representation::B, &mut out_b),
        (Representation::C, &mut out_c),
    ] {
        results.push(bench(&format!("e2e_{rep:?}_fused"), n, warm, trials, |i| {
            fx.extract(&images[i].view(), &params, rep, out).unwrap();
            black_box(&*out);
        }));
        results.push(bench(
            &format!("e2e_{rep:?}_via_matrix"),
            n,
            warm,
            trials,
            |i| {
                fx.extract_via_matrix(&images[i].view(), &params, rep, out)
                    .unwrap();
                black_box(&*out);
            },
        ));
    }

    // Batch throughput (A, fused), 1..N threads.
    let engine = Arc::new(Engine::new());
    let total = args.batch_images;
    let mut batch = Vec::new();
    let max_threads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);
    let mut counts: Vec<usize> = vec![1, 2, 4, 8];
    counts.push(max_threads);
    counts.retain(|&t| t <= max_threads);
    counts.dedup();
    if args.only.is_some() {
        counts.clear();
    }
    for &threads in &counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for rep in [Representation::A, Representation::B, Representation::C] {
            let dim = rep.dim();
            let mut out = vec![0.0; total * dim];
            let run = |out: &mut Vec<f64>| {
                pool.install(|| {
                    out.par_chunks_mut(dim).enumerate().for_each_init(
                        || FeatureExtractor::with_engine(Arc::clone(&engine)),
                        |fx, (i, dst)| {
                            fx.extract(&images[i % n].view(), &params, rep, dst)
                                .unwrap()
                        },
                    )
                })
            };
            run(&mut out); // warm-up
            let mut rates = Vec::new();
            for _ in 0..5 {
                let t = Instant::now();
                run(&mut out);
                rates.push(total as f64 / t.elapsed().as_secs_f64());
            }
            let med = quantile(&rates, 0.5);
            eprintln!("batch {rep:?} threads {threads:>2}: {med:>10.0} img/s");
            batch.push(json!({"representation": format!("{rep:?}"), "threads": threads, "images": total, "median_img_s": med, "runs_img_s": rates}));
        }
    }

    let report = json!({
        "host": host_info(), "image_size": [w, h], "images": n, "params": {"center": [params.center.x, params.center.y], "angle_step": params.angle_step},
        "plan": plan_info, "single_image": results, "batch": batch, "peak_rss_bytes": peak_rss_bytes(),
        "build": {"profile": if cfg!(debug_assertions) { "debug" } else { "release" }},
    });
    if let Some(dir) = args.out.parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    std::fs::write(&args.out, serde_json::to_string_pretty(&report).unwrap()).unwrap();
}
