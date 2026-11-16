//! Deliberately wrong CDCH variants ("mutants") used to measure the sensitivity of the
//! differential harness. Each mutant changes one plausible implementation detail of
//! `ip-1.js`. A harness that cannot tell a mutant from the canonical algorithm on any
//! fixture would be too weak to support the equivalence claim.

use cdch_core::js::{self, B_COEF, G_COEF, R_COEF};
use cdch_core::{Params, RgbImage, BINS};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mutant {
    /// `Math.round` as `floor(x + 0.5)` (wrong for 0.49999999999999994 and large odd values).
    RoundFloorHalf,
    /// Round half to even (Rust `round_ties_even`, NumPy `rint`).
    RoundHalfEven,
    /// Round half away from zero (Rust `f64::round`, C `round`).
    RoundHalfAway,
    /// Truncation toward zero (`x as i64`).
    Truncate,
    /// Integer luma `(2125 R + 7154 G + 721 B + 5000) / 10000`.
    LumaInteger,
    /// Luma re-associated: `R*r + (G*g + B*b)` (changes the bin of 42 of the 2^24 colours).
    LumaReassociated,
    /// Luma with Rec. 601 weights (0.299, 0.587, 0.114).
    LumaRec601,
    /// Radius limit `max(width, height)` instead of `width`.
    RadiusMaxDim,
    /// Radius limit `min(width, height)`.
    RadiusMinDim,
    /// Normalize by multiplying with the reciprocal of the sample count.
    NormReciprocal,
    /// Directions from the host libm instead of V8.
    LibmTrig,
    /// 0° pointing right (no `- 90`).
    ZeroDegreesRight,
    /// Angles traversed clockwise-negative (`angle + 90`).
    MirroredAngles,
    /// Duplicate rounded coordinates within a ray counted once.
    DedupCoordinates,
    /// Radius starting at 1 (centre pixel skipped).
    SkipCentre,
}

pub const ALL: [Mutant; 15] = [
    Mutant::RoundFloorHalf,
    Mutant::RoundHalfEven,
    Mutant::RoundHalfAway,
    Mutant::Truncate,
    Mutant::LumaInteger,
    Mutant::LumaReassociated,
    Mutant::LumaRec601,
    Mutant::RadiusMaxDim,
    Mutant::RadiusMinDim,
    Mutant::NormReciprocal,
    Mutant::LibmTrig,
    Mutant::ZeroDegreesRight,
    Mutant::MirroredAngles,
    Mutant::DedupCoordinates,
    Mutant::SkipCentre,
];

fn round_with(m: Mutant, x: f64) -> f64 {
    match m {
        Mutant::RoundFloorHalf => (x + 0.5).floor(),
        Mutant::RoundHalfEven => x.round_ties_even(),
        Mutant::RoundHalfAway => x.round(),
        Mutant::Truncate => x.trunc(),
        _ => js::round(x),
    }
}

fn luma_with(m: Mutant, r: u8, g: u8, b: u8) -> usize {
    let (r, g, b) = (r as f64, g as f64, b as f64);
    match m {
        Mutant::LumaInteger => ((2125 * r as u32 + 7154 * g as u32 + 721 * b as u32 + 5000) / 10000) as usize,
        Mutant::LumaReassociated => js::round(r * R_COEF + (g * G_COEF + b * B_COEF)) as usize,
        Mutant::LumaRec601 => js::round(r * 0.299 + g * 0.587 + b * 0.114) as usize,
        _ => js::round(r * R_COEF + g * G_COEF + b * B_COEF) as usize,
    }
    .min(255)
}

fn direction_with(m: Mutant, angle: u32) -> (f64, f64) {
    match m {
        Mutant::LibmTrig => js::direction_libm(angle),
        Mutant::ZeroDegreesRight => {
            let r = angle as f64 * (std::f64::consts::PI / 180.0);
            (r.cos(), r.sin())
        }
        Mutant::MirroredAngles => {
            let r = (angle as f64 + 90.0) * (std::f64::consts::PI / 180.0);
            (r.cos(), r.sin())
        }
        _ => js::direction(angle),
    }
}

/// The mutated CDCH matrix (`n_rays x 256`).
pub fn cdch(img: &RgbImage<'_>, params: &Params, m: Mutant) -> Vec<f64> {
    let (x, y) = (params.center.x, params.center.y);
    let (w, h) = (img.width() as f64, img.height() as f64);
    let limit = match m {
        Mutant::RadiusMaxDim => w.max(h),
        Mutant::RadiusMinDim => w.min(h),
        _ => w,
    };
    let mut out = Vec::with_capacity(params.n_rays() * BINS);
    for angle in params.angles() {
        let (c, s) = direction_with(m, angle);
        let mut hist = [0u32; BINS];
        let mut seen: Vec<(i64, i64)> = Vec::new();
        let mut r: u32 = if m == Mutant::SkipCentre { 1 } else { 0 };
        while (r as f64) < limit {
            let rf = r as f64;
            let px = round_with(m, x + rf * c);
            let py = round_with(m, y + rf * s);
            if px < 0.0 || px >= w || py < 0.0 || py >= h {
                break;
            }
            r += 1;
            if m == Mutant::DedupCoordinates {
                let key = (px as i64, py as i64);
                if seen.contains(&key) {
                    continue;
                }
                seen.push(key);
            }
            let [rr, gg, bb] = img.rgb(px as usize, py as usize);
            hist[luma_with(m, rr, gg, bb)] += 1;
        }
        let total: u32 = hist.iter().sum();
        let mut cum = 0u32;
        let inv = 1.0 / total as f64;
        for &v in &hist {
            cum += v;
            out.push(if m == Mutant::NormReciprocal { cum as f64 * inv } else { cum as f64 / total as f64 });
        }
    }
    out
}
