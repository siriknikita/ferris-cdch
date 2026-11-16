//! # cdch-features
//!
//! Rust implementations of the four archived CDCH representations
//! (`CDCH_REPRESENTATIONS.md`; Python reference `pilot/cdch_features.descriptor` and
//! `native_features.rich` / `representation`):
//!
//! | ID | Dim | Definition |
//! |----|-----|------------|
//! | A  | 56  | compact pilot descriptor |
//! | B  | 98  | A followed by 42 enriched statistics |
//! | C  | 768 | 24 x 32 angular-sector CDF means, sector-major |
//! | D  | 768 | the C values, interpreted as a 24 x 32 tensor |
//!
//! Two input forms are supported:
//!
//! * the full canonical `360 x 256` matrix ([`MatrixView`]);
//! * the reduced `360 x 32` checkpoint matrix produced by
//!   [`cdch_core::SamplingPlan::checkpoints_into`]. A, C and D depend only on the
//!   CDF at bins `8j + 7`, so they can be computed from it exactly (fused path). B
//!   needs the full per-ray PMF and full-row angular lags, so it uses the matrix.

#![forbid(unsafe_code)]

pub mod numpy_compat;
mod extractor;

pub use extractor::FeatureExtractor;

use cdch_core::engine::CHECKPOINTS;
use cdch_core::{CdchError, MatrixView, Result, BINS, RAYS_ARCHIVED};
use numpy_compat::{linspace_0_1_256, pairwise_sum, pairwise_sum_by};

pub const A_DIM: usize = 56;
pub const RICH_DIM: usize = 42;
pub const B_DIM: usize = A_DIM + RICH_DIM;
pub const C_DIM: usize = 768;
pub const D_SHAPE: (usize, usize) = (24, 32);

/// Intensity columns sampled by A: `cdf[:, 7:255:16]` → 7, 23, …, 247.
pub const A_COLUMNS: usize = 16;
/// Harmonics retained by A.
pub const A_HARMONICS: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// The archived representations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Representation {
    A,
    B,
    C,
    D,
}

impl Representation {
    pub fn dim(self) -> usize {
        match self {
            Representation::A => A_DIM,
            Representation::B => B_DIM,
            Representation::C | Representation::D => C_DIM,
        }
    }
    /// Accepts the archived keys `A`, `B`, `C`, `D` and `D_permuted` (same input as D).
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "A" | "a" => Some(Representation::A),
            "B" | "b" => Some(Representation::B),
            "C" | "c" => Some(Representation::C),
            "D" | "d" | "D_permuted" => Some(Representation::D),
            _ => None,
        }
    }
    /// True when the representation can be computed from the 32-bin checkpoint matrix.
    pub fn fusable(self) -> bool {
        !matches!(self, Representation::B)
    }
}

/// The checks `cdch_features.descriptor` applies before computing A:
/// shape `360 x 256`, all finite, `min >= 0`, `np.allclose(cdf[:, -1], 1)`
/// (`|x - 1| <= 1e-8 + 1e-5`), and `np.diff(cdf, axis=1) >= -1e-12`.
pub fn validate_matrix(m: &MatrixView<'_>) -> Result<()> {
    if m.rows() != RAYS_ARCHIVED {
        return Err(CdchError::RayCount { expected: RAYS_ARCHIVED, actual: m.rows() });
    }
    let d = m.as_slice();
    if !d.iter().all(|v| v.is_finite()) {
        return Err(CdchError::InvalidMatrix("non-finite values (empty ray?)".into()));
    }
    if d.iter().any(|&v| v < 0.0) {
        return Err(CdchError::InvalidMatrix("negative values".into()));
    }
    for i in 0..m.rows() {
        let row = m.row(i);
        if (row[BINS - 1] - 1.0).abs() > 1e-8 + 1e-5 {
            return Err(CdchError::InvalidMatrix(format!("row {i} does not end at 1")));
        }
        if row.windows(2).any(|w| w[1] - w[0] < -1e-12) {
            return Err(CdchError::InvalidMatrix(format!("row {i} is not monotone")));
        }
    }
    Ok(())
}

/// Twiddle table `(cos, sin)(2πm/360)` for the angular DFT, built from the first
/// octant by exact symmetries so the table is exactly symmetric.
fn twiddles() -> &'static [(f64, f64); RAYS_ARCHIVED] {
    use std::sync::OnceLock;
    static T: OnceLock<[(f64, f64); RAYS_ARCHIVED]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [(0.0, 0.0); RAYS_ARCHIVED];
        let base = |m: usize| -> (f64, f64) {
            // m in 0..=45 degrees.
            let a = m as f64 * (std::f64::consts::PI / 180.0);
            (a.cos(), a.sin())
        };
        for (m, slot) in t.iter_mut().enumerate() {
            let q = m / 90;
            let r = m % 90;
            let (c, s) = if r <= 45 {
                base(r)
            } else {
                let (c2, s2) = base(90 - r);
                (s2, c2)
            };
            *slot = match q {
                0 => (c, s),
                1 => (-s, c),
                2 => (-c, -s),
                _ => (s, -c),
            };
        }
        t
    })
}

/// `|rfft(x, axis=0)[k]|` for the 16 columns of a demeaned `360 x 16` array,
/// using compensated (Neumaier) summation. NumPy uses pocketfft; agreement is
/// within a few ulps (quantified in the validation report), not bitwise.
fn harmonic_magnitudes(centered: &[f64], k: usize, out: &mut [f64; A_COLUMNS]) {
    let tw = twiddles();
    let mut re = [0f64; A_COLUMNS];
    let mut im = [0f64; A_COLUMNS];
    let mut cre = [0f64; A_COLUMNS];
    let mut cim = [0f64; A_COLUMNS];
    for n in 0..RAYS_ARCHIVED {
        let (c, s) = tw[(k * n) % RAYS_ARCHIVED];
        let row = &centered[n * A_COLUMNS..(n + 1) * A_COLUMNS];
        for j in 0..A_COLUMNS {
            let a = row[j] * c;
            let t = re[j] + a;
            cre[j] += if re[j].abs() >= a.abs() { (re[j] - t) + a } else { (a - t) + re[j] };
            re[j] = t;
            let b = -(row[j] * s);
            let t = im[j] + b;
            cim[j] += if im[j].abs() >= b.abs() { (im[j] - t) + b } else { (b - t) + im[j] };
            im[j] = t;
        }
    }
    for j in 0..A_COLUMNS {
        out[j] = (re[j] + cre[j]).hypot(im[j] + cim[j]);
    }
}

/// Representation A from the sampled `360 x 16` array `sampled[i * 16 + m] = F(θ_i, 16m + 7)`.
///
/// ```python
/// directional_change = np.abs(np.roll(sampled, 1, axis=0) - sampled)
/// directional_power = np.abs(np.fft.rfft(sampled - sampled.mean(axis=0), axis=0)) / 360
/// harmonics = [directional_power[i].mean() for i in (1, 2, 4, 8, 16, 32, 64, 128)]
/// return np.r_[sampled.mean(axis=0), sampled.std(axis=0), directional_change.mean(axis=0), harmonics]
/// ```
pub fn compact_from_sampled(sampled: &[f64], scratch: &mut Vec<f64>, out: &mut [f64]) {
    const N: usize = RAYS_ARCHIVED;
    const K: usize = A_COLUMNS;
    assert_eq!(sampled.len(), N * K);
    assert_eq!(out.len(), A_DIM);
    let nf = N as f64;
    // Axis-0 reductions are sequential in NumPy.
    let mut mean = [0f64; K];
    for i in 0..N {
        for j in 0..K {
            mean[j] += sampled[i * K + j];
        }
    }
    for m in mean.iter_mut() {
        *m /= nf;
    }
    scratch.clear();
    scratch.extend((0..N * K).map(|idx| sampled[idx] - mean[idx % K]));
    let centered = &scratch[..];
    let mut var = [0f64; K];
    let mut change = [0f64; K];
    for i in 0..N {
        let prev = if i == 0 { N - 1 } else { i - 1 };
        for j in 0..K {
            let d = centered[i * K + j];
            var[j] += d * d;
            change[j] += (sampled[prev * K + j] - sampled[i * K + j]).abs();
        }
    }
    for j in 0..K {
        out[j] = mean[j];
        out[K + j] = (var[j] / nf).sqrt();
        out[2 * K + j] = change[j] / nf;
    }
    let mut mags = [0f64; K];
    for (h, &k) in A_HARMONICS.iter().enumerate() {
        harmonic_magnitudes(centered, k, &mut mags);
        let power: [f64; K] = std::array::from_fn(|j| mags[j] / nf);
        out[3 * K + h] = pairwise_sum(&power) / K as f64;
    }
}

/// Representation A from the full matrix (validates first, like `descriptor`).
pub fn compact(m: &MatrixView<'_>, out: &mut [f64]) -> Result<()> {
    validate_matrix(m)?;
    let mut sampled = Vec::with_capacity(RAYS_ARCHIVED * A_COLUMNS);
    for i in 0..RAYS_ARCHIVED {
        let row = m.row(i);
        sampled.extend((0..A_COLUMNS).map(|c| row[16 * c + 7]));
    }
    compact_from_sampled(&sampled, &mut Vec::new(), out);
    Ok(())
}

/// Representation A from the `360 x 32` checkpoint matrix (fused path). The caller
/// must establish validity (every ray non-empty), which for the canonical engine
/// is a property of the sampling plan.
pub fn compact_from_checkpoints(cp: &[f64], sampled: &mut Vec<f64>, scratch: &mut Vec<f64>, out: &mut [f64]) -> Result<()> {
    if cp.len() != RAYS_ARCHIVED * CHECKPOINTS {
        return Err(CdchError::RayCount { expected: RAYS_ARCHIVED, actual: cp.len() / CHECKPOINTS });
    }
    sampled.clear();
    for i in 0..RAYS_ARCHIVED {
        let row = &cp[i * CHECKPOINTS..(i + 1) * CHECKPOINTS];
        // Checkpoint j holds bin 8j + 7; A uses bins 16m + 7 = checkpoint 2m.
        sampled.extend((0..A_COLUMNS).map(|c| row[2 * c]));
    }
    compact_from_sampled(sampled, scratch, out);
    Ok(())
}

/// Representation C (and D) from the full matrix:
/// `cdf[:, 7::8].reshape(24, 15, 32).mean(axis=1).ravel()` (sequential sum over 15 rays).
pub fn sector(m: &MatrixView<'_>, out: &mut [f64]) -> Result<()> {
    if m.rows() != RAYS_ARCHIVED {
        return Err(CdchError::RayCount { expected: RAYS_ARCHIVED, actual: m.rows() });
    }
    sector_by(|i, j| m.at(i, 8 * j + 7), out);
    Ok(())
}

/// Representation C (and D) from the checkpoint matrix (fused path).
pub fn sector_from_checkpoints(cp: &[f64], out: &mut [f64]) -> Result<()> {
    if cp.len() != RAYS_ARCHIVED * CHECKPOINTS {
        return Err(CdchError::RayCount { expected: RAYS_ARCHIVED, actual: cp.len() / CHECKPOINTS });
    }
    sector_by(|i, j| cp[i * CHECKPOINTS + j], out);
    Ok(())
}

#[inline]
fn sector_by(f: impl Fn(usize, usize) -> f64, out: &mut [f64]) {
    assert_eq!(out.len(), C_DIM);
    for s in 0..24 {
        for j in 0..32 {
            let mut acc = 0.0;
            for t in 0..15 {
                acc += f(15 * s + t, j);
            }
            out[s * 32 + j] = acc / 15.0;
        }
    }
}

/// The 42 enriched statistics of representation B (`native_features.rich`).
/// Works for any number of rays; the archived protocol uses 360.
pub fn rich(m: &MatrixView<'_>, out: &mut [f64]) {
    assert_eq!(out.len(), RICH_DIM);
    let rows = m.rows();
    let bins = linspace_0_1_256();
    // 21 per-ray quantities, stored measure-major: q[k * rows + i].
    let mut q = vec![0f64; 21 * rows];
    let mut pmf = [0f64; BINS];
    let mut centered = [0f64; BINS];
    for i in 0..rows {
        let cdf = m.row(i);
        // pmf = np.diff(cdf, axis=1, prepend=0)
        pmf[0] = cdf[0] - 0.0;
        for b in 1..BINS {
            pmf[b] = cdf[b] - cdf[b - 1];
        }
        // mean = pmf @ bins (BLAS in NumPy; pairwise here).
        let mean = pairwise_sum_by(BINS, &|b| pmf[b] * bins[b]);
        for b in 0..BINS {
            centered[b] = bins[b] - mean;
        }
        let variance = pairwise_sum_by(BINS, &|b| pmf[b] * (centered[b] * centered[b]));
        let c3 = pairwise_sum_by(BINS, &|b| pmf[b] * centered[b].powf(3.0));
        let c4 = pairwise_sum_by(BINS, &|b| pmf[b] * centered[b].powf(4.0));
        let (skew, kurt) = if variance > 1e-10 {
            (c3 / variance.powf(1.5), c4 / (variance * variance))
        } else {
            (0.0, 0.0)
        };
        let entropy = -pairwise_sum_by(BINS, &|b| pmf[b] * pmf[b].max(1e-300).log2()) / 8.0;
        let concentration = pairwise_sum_by(BINS, &|b| pmf[b] * pmf[b]);
        let max_mass = pmf.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mad = pairwise_sum_by(BINS - 1, &|b| (pmf[b + 1] - pmf[b]).abs()) / (BINS - 1) as f64;
        let quantile = |p: f64| -> f64 {
            // (cdf >= p).argmax() / 255 — 0 when no bin qualifies.
            cdf.iter().position(|&v| v >= p).unwrap_or(0) as f64 / 255.0
        };
        let qs = [quantile(0.1), quantile(0.25), quantile(0.5), quantile(0.75), quantile(0.9)];
        let vals = [
            mean,
            variance.sqrt(),
            skew,
            kurt,
            entropy,
            concentration,
            cdf[15],
            1.0 - cdf[239],
            max_mass,
            mad,
            qs[0],
            qs[1],
            qs[2],
            qs[3],
            qs[4],
            qs[3] - qs[1],
            qs[4] - qs[0],
        ];
        for (k, v) in vals.iter().enumerate() {
            q[k * rows + i] = *v;
        }
        // Angular lags: mean |cdf[i] - cdf[(i - lag) mod rows]| over intensity.
        for (n, &lag) in [1usize, 15, 90, 180].iter().enumerate() {
            let other = m.row((i + rows - lag % rows) % rows);
            q[(17 + n) * rows + i] = pairwise_sum_by(BINS, &|b| (cdf[b] - other[b]).abs()) / BINS as f64;
        }
    }
    for k in 0..21 {
        let r = &q[k * rows..(k + 1) * rows];
        out[2 * k] = numpy_compat::mean(r);
        out[2 * k + 1] = numpy_compat::std(r);
    }
}

/// Representation B from the full matrix: A followed by [`rich`].
pub fn enriched(m: &MatrixView<'_>, out: &mut [f64]) -> Result<()> {
    assert_eq!(out.len(), B_DIM);
    compact(m, &mut out[..A_DIM])?;
    rich(m, &mut out[A_DIM..]);
    Ok(())
}

/// Any representation from the full matrix.
pub fn from_matrix(m: &MatrixView<'_>, rep: Representation, out: &mut [f64]) -> Result<()> {
    if out.len() != rep.dim() {
        return Err(CdchError::BufferSize { expected: rep.dim(), actual: out.len() });
    }
    match rep {
        Representation::A => compact(m, out),
        Representation::B => enriched(m, out),
        Representation::C | Representation::D => sector(m, out),
    }
}
