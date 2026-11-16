//! Shared utilities for the Ferris differential harness and benchmarks.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{json, Value};

pub mod mutants;
pub mod synth;

/// Deterministic SplitMix64 generator (no external RNG dependency).
#[derive(Clone)]
pub struct SplitMix(pub u64);

impl SplitMix {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
    pub fn fill(&mut self, buf: &mut [u8]) {
        for b in buf {
            *b = self.next_u64() as u8;
        }
    }
}

pub use cdch_cli::{read_png, OwnedRgb};

/// All `*.png` files below `dir`, sorted for determinism.
pub fn png_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&d) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("png")) {
                    out.push(p);
                }
            }
        }
    }
    out.sort();
    out
}

/// Minimal standard base64 decoder (the oracle encodes Float64Array/Int32Array buffers).
pub fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    fn val(c: u8) -> Result<u32, String> {
        Ok(match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a' + 26) as u32,
            b'0'..=b'9' => (c - b'0' + 52) as u32,
            b'+' => 62,
            b'/' => 63,
            _ => return Err(format!("invalid base64 byte {c}")),
        })
    }
    let bytes: Vec<u8> = s.bytes().filter(|&c| c != b'=').collect();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        let mut acc = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            acc |= val(c)? << (18 - 6 * i);
        }
        out.push((acc >> 16) as u8);
        if chunk.len() > 2 {
            out.push((acc >> 8) as u8);
        }
        if chunk.len() > 3 {
            out.push(acc as u8);
        }
    }
    Ok(out)
}

pub fn f64s_le(bytes: &[u8]) -> Vec<f64> {
    bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

pub fn i32s_le(bytes: &[u8]) -> Vec<i32> {
    bytes.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Result of one oracle evaluation.
pub struct OracleReply {
    pub center: (f64, f64),
    pub rows: usize,
    pub matrix: Vec<f64>,
    pub trace: Option<OracleTrace>,
    pub js_ns: u64,
}

pub struct OracleTrace {
    pub ray_lengths: Vec<usize>,
    pub coords: Vec<i32>,
    pub matrix: Vec<f64>,
}

/// Persistent Node process running `oracle/diff_worker.js` (canonical ip-1.js via the
/// archived oracle).
pub struct Oracle {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl Oracle {
    pub fn spawn(oracle_dir: &Path) -> std::io::Result<Self> {
        let mut child = Command::new("node")
            .arg("--max-old-space-size=8192")
            .arg(oracle_dir.join("diff_worker.js"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        let stdin = child.stdin.take().expect("stdin");
        let stdout = BufReader::new(child.stdout.take().expect("stdout"));
        Ok(Self { child, stdin, stdout, next_id: 0 })
    }

    pub fn compute(
        &mut self,
        img: &OwnedRgb,
        center: Option<(f64, f64)>,
        angle_step: u32,
        trace: bool,
    ) -> Result<OracleReply, String> {
        self.next_id += 1;
        let header = json!({
            "id": self.next_id, "width": img.width, "height": img.height,
            "center": center.map(|(x, y)| json!([x, y])), "angle_step": angle_step, "trace": trace,
        })
        .to_string();
        let mut frame = Vec::with_capacity(8 + header.len() + img.data.len());
        frame.extend_from_slice(&(header.len() as u32).to_le_bytes());
        frame.extend_from_slice(header.as_bytes());
        frame.extend_from_slice(&(img.data.len() as u32).to_le_bytes());
        frame.extend_from_slice(&img.data);
        self.stdin.write_all(&frame).map_err(|e| e.to_string())?;
        self.stdin.flush().map_err(|e| e.to_string())?;
        let mut line = String::new();
        self.stdout.read_line(&mut line).map_err(|e| e.to_string())?;
        if line.is_empty() {
            return Err("oracle exited".into());
        }
        let v: Value = serde_json::from_str(&line).map_err(|e| e.to_string())?;
        if let Some(err) = v.get("error") {
            return Err(format!("oracle error: {err}"));
        }
        if v["id"].as_u64() != Some(self.next_id) {
            return Err("oracle reply out of order".into());
        }
        let num = |x: &Value| x.as_f64().unwrap_or(f64::NAN);
        let trace = if v["trace"].is_object() {
            let t = &v["trace"];
            Some(OracleTrace {
                ray_lengths: t["ray_lengths"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect(),
                coords: i32s_le(&base64_decode(t["coords"].as_str().unwrap())?),
                matrix: f64s_le(&base64_decode(t["matrix"].as_str().unwrap())?),
            })
        } else {
            None
        };
        Ok(OracleReply {
            center: (num(&v["center"][0]), num(&v["center"][1])),
            rows: v["rows"].as_u64().unwrap_or(0) as usize,
            matrix: f64s_le(&base64_decode(v["matrix"].as_str().unwrap())?),
            trace,
            js_ns: v["js_ns"].as_u64().unwrap_or(0),
        })
    }
}

impl Drop for Oracle {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Nearest-rank quantile on a sorted copy.
pub fn quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Linear interpolation (NumPy default) for comparability with the archived benchmarks.
    let pos = q * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    v[lo] + (v[hi] - v[lo]) * (pos - lo as f64)
}

/// Resident set size in bytes without libc bindings: the high-water mark (`VmHWM`) on
/// Linux, otherwise the current RSS reported by `ps`.
pub fn peak_rss_bytes() -> Option<u64> {
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                return rest.trim().trim_end_matches(" kB").trim().parse::<u64>().ok().map(|k| k * 1024);
            }
        }
    }
    let out = Command::new("ps").args(["-o", "rss=", "-p", &std::process::id().to_string()]).output().ok()?;
    String::from_utf8(out.stdout).ok()?.trim().parse::<u64>().ok().map(|k| k * 1024)
}


/// Host description for reports.
pub fn host_info() -> Value {
    let sh = |cmd: &str, args: &[&str]| {
        Command::new(cmd).args(args).output().ok().and_then(|o| String::from_utf8(o.stdout).ok()).map(|s| s.trim().to_string())
    };
    json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "cpu": sh("sysctl", &["-n", "machdep.cpu.brand_string"]).or_else(|| sh("sh", &["-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"])),
        "logical_cpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
        "memory_bytes": sh("sysctl", &["-n", "hw.memsize"]),
        "os_version": sh("sw_vers", &["-productVersion"]).or_else(|| sh("uname", &["-r"])),
        "rustc": sh("rustc", &["--version"]),
        "node": sh("node", &["--version"]),
    })
}
