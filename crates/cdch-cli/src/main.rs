//! `cdch` — standalone CDCH feature extraction and linear-model scoring.
//!
//! ```text
//! cdch info
//! cdch features --kind A|B|C|D [--center X,Y] [--threads N] [--json] IMAGE.png...
//! cdch matrix   [--center X,Y] [--step S] IMAGE.png            (CSV, n_rays x 256)
//! cdch score    --model MODEL.json [--threads N] [--timing] IMAGE.png...
//! ```
//! Defaults follow the archived protocol: centre `(width // 2, height // 2)`, 360 rays.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use cdch_cli::{read_png, LinearModel};
use cdch_core::{Center, Engine, Params};
use cdch_features::{FeatureExtractor, Representation};
use rayon::prelude::*;
use serde_json::json;

const USAGE: &str = "usage:
  cdch info
  cdch features --kind A|B|C|D [--center X,Y] [--threads N] [--json] IMAGE.png...
  cdch matrix   [--center X,Y] [--step S] IMAGE.png
  cdch score    --model MODEL.json [--threads N] [--timing] IMAGE.png...";

struct Opts {
    kind: Representation,
    center: Option<(f64, f64)>,
    step: u32,
    threads: Option<usize>,
    json: bool,
    timing: bool,
    model: Option<PathBuf>,
    files: Vec<PathBuf>,
}

fn parse(args: &[String]) -> Result<Opts, String> {
    let mut o = Opts { kind: Representation::A, center: None, step: 1, threads: None, json: false, timing: false, model: None, files: vec![] };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = || it.next().cloned().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--kind" => o.kind = Representation::parse(&value()?).ok_or("--kind must be A, B, C or D")?,
            "--center" => {
                let v = value()?;
                let (x, y) = v.split_once(',').ok_or("--center X,Y")?;
                o.center = Some((x.parse().map_err(|_| "bad centre x")?, y.parse().map_err(|_| "bad centre y")?));
            }
            "--step" => o.step = value()?.parse().map_err(|_| "bad --step")?,
            "--threads" => o.threads = Some(value()?.parse().map_err(|_| "bad --threads")?),
            "--json" => o.json = true,
            "--timing" => o.timing = true,
            "--model" => o.model = Some(value()?.into()),
            f if f.starts_with("--") => return Err(format!("unknown option {f}")),
            f => o.files.push(f.into()),
        }
    }
    Ok(o)
}

fn params(o: &Opts, w: usize, h: usize) -> Params {
    Params { center: o.center.map_or(Center::archived(w, h), |(x, y)| Center::new(x, y)), angle_step: o.step }
}

/// Run `f` over the files on `threads` workers (serial for 1), preserving order.
fn map_files<T: Send>(o: &Opts, f: impl Fn(&mut FeatureExtractor, &Path) -> Result<T, String> + Sync) -> Result<Vec<T>, String> {
    let engine = Arc::new(Engine::new());
    if o.threads == Some(1) {
        let mut fx = FeatureExtractor::with_engine(engine);
        return o.files.iter().map(|p| f(&mut fx, p)).collect();
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(o.threads.unwrap_or(0)).build().map_err(|e| e.to_string())?;
    pool.install(|| {
        o.files
            .par_iter()
            .map_init(|| FeatureExtractor::with_engine(Arc::clone(&engine)), |fx, p| f(fx, p))
            .collect()
    })
}

fn run(cmd: &str, o: Opts) -> Result<(), String> {
    match cmd {
        "info" => {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "version": env!("CARGO_PKG_VERSION"), "canonical_sha256": cdch_core::CANONICAL_SHA256,
                    "target": format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
                    "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
                }))
                .unwrap()
            );
            Ok(())
        }
        "features" => {
            let kind = o.kind;
            let rows = map_files(&o, |fx, p| {
                let img = read_png(p)?;
                let mut out = vec![0.0; kind.dim()];
                fx.extract(&img.view(), &params(&o, img.width, img.height), kind, &mut out).map_err(|e| format!("{}: {e}", p.display()))?;
                Ok(out)
            })?;
            for (p, v) in o.files.iter().zip(rows) {
                if o.json {
                    println!("{}", json!({"path": p, "kind": format!("{kind:?}"), "features": v}));
                } else {
                    let vals: Vec<String> = v.iter().map(|x| format!("{x:?}")).collect();
                    println!("{},{}", p.display(), vals.join(","));
                }
            }
            Ok(())
        }
        "matrix" => {
            let [path] = o.files.as_slice() else { return Err("matrix takes exactly one image".into()) };
            let img = read_png(path)?;
            let p = params(&o, img.width, img.height);
            let mut fx = FeatureExtractor::new();
            let mut m = vec![0.0; p.n_rays() * cdch_core::BINS];
            fx.matrix_into(&img.view(), &p, &mut m).map_err(|e| e.to_string())?;
            for row in m.chunks(cdch_core::BINS) {
                let vals: Vec<String> = row.iter().map(|x| format!("{x:?}")).collect();
                println!("{}", vals.join(","));
            }
            Ok(())
        }
        "score" => {
            let model = LinearModel::load(o.model.as_deref().ok_or("--model is required")?)?;
            let start = Instant::now();
            let rows = map_files(&o, |fx, p| {
                let t0 = Instant::now();
                let img = read_png(p)?;
                let t1 = Instant::now();
                let mut x = vec![0.0; model.feature.dim()];
                fx.extract(&img.view(), &params(&o, img.width, img.height), model.feature, &mut x).map_err(|e| format!("{}: {e}", p.display()))?;
                let t2 = Instant::now();
                let s = model.decision(&x);
                let t3 = Instant::now();
                Ok((s, [(t1 - t0).as_secs_f64(), (t2 - t1).as_secs_f64(), (t3 - t2).as_secs_f64()]))
            })?;
            let wall = start.elapsed().as_secs_f64();
            for (p, (s, t)) in o.files.iter().zip(&rows) {
                let label = (*s >= model.threshold) as u8;
                if o.timing {
                    println!("{},{:?},{},{:.1},{:.1},{:.2}", p.display(), s, label, t[0] * 1e6, t[1] * 1e6, t[2] * 1e6);
                } else {
                    println!("{},{:?},{}", p.display(), s, label);
                }
            }
            if o.timing {
                eprintln!("{} images in {:.3} s ({:.0} img/s) using model {}", rows.len(), wall, rows.len() as f64 / wall, model.name);
            }
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    match parse(&args[1..]).and_then(|o| run(cmd, o)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("cdch: {e}");
            ExitCode::FAILURE
        }
    }
}
