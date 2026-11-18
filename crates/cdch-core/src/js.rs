//! JavaScript-exact numeric primitives used by `ip-1.js`.

use crate::directions_v8::{COS_BITS, RADIANS_BITS, SIN_BITS};

/// `const R_COEF = 0.2125;`
pub const R_COEF: f64 = 0.2125;
/// `const G_COEF = 0.7154;`
pub const G_COEF: f64 = 0.7154;
/// `const B_COEF = 0.0721;`
pub const B_COEF: f64 = 0.0721;

/// ECMAScript `Math.round`: the integral Number closest to `x`, ties toward +∞,
/// preserving `-0` for inputs in `[-0.5, -0]`.
///
/// `x - floor(x)` is exact whenever the comparison with 0.5 can be decided by it
/// (see `RUST_PORT_VALIDATION.md`, "Rounding"), and `|x| >= 2^52` is already integral.
#[inline]
pub fn round(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let f = x.floor();
    let r = if x - f >= 0.5 { f + 1.0 } else { f };
    if r == 0.0 && x.is_sign_negative() {
        -0.0
    } else {
        r
    }
}

/// `Math.round(pixelData[0] * R_COEF + pixelData[1] * G_COEF + pixelData[2] * B_COEF)`
/// evaluated left to right in binary64 without contraction. Rust never fuses
/// `a * b + c` into an FMA unless asked to, so this matches V8.
#[inline]
pub fn luma(r: u8, g: u8, b: u8) -> u8 {
    let v = r as f64 * R_COEF + g as f64 * G_COEF + b as f64 * B_COEF;
    // v ∈ [0, 255.00000000000003]; the rounded value is at most 255.
    round(v) as u8
}

const fn product_table(coef: f64) -> [f64; 256] {
    let mut t = [0f64; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = i as f64 * coef;
        i += 1;
    }
    t
}

/// `v * R_COEF` for every 8-bit `v` (IEEE products are exact-rounded identically at
/// compile time and run time).
pub const LUMA_R: [f64; 256] = product_table(R_COEF);
pub const LUMA_G: [f64; 256] = product_table(G_COEF);
pub const LUMA_B: [f64; 256] = product_table(B_COEF);

/// Table-driven [`luma`]: the same three products, added in the same order, then
/// `floor(v + 0.5)` via truncation. `luma_fast_matches_luma_exhaustively` proves this
/// equals [`luma`] for all 2^24 colours (for these reachable `v`, adding 0.5 never
/// rounds across an integer and `v >= 0`).
#[inline]
pub fn luma_fast(r: u8, g: u8, b: u8) -> u8 {
    let v = (LUMA_R[r as usize] + LUMA_G[g as usize]) + LUMA_B[b as usize];
    (v + 0.5) as u8
}

/// `(Math.cos(radians), Math.sin(radians))` with
/// `radians = (angle - 90) * (Math.PI / 180)`, reproduced from the V8 bit patterns
/// recorded by `oracle/direction_table.js`.
///
/// The table removes any dependence on the host libm. Using `f64::cos`/`f64::sin`
/// instead is available as [`direction_libm`] for comparison.
#[inline]
pub fn direction(angle_deg: u32) -> (f64, f64) {
    let a = angle_deg as usize;
    (f64::from_bits(COS_BITS[a]), f64::from_bits(SIN_BITS[a]))
}

/// The JavaScript `radians` value for an integer angle, from the V8 table.
#[inline]
pub fn radians(angle_deg: u32) -> f64 {
    f64::from_bits(RADIANS_BITS[angle_deg as usize])
}

/// The same expression evaluated with the host's libm (`f64::cos`, `f64::sin`).
/// Used only to quantify libm/V8 disagreement; never used by the validated path.
pub fn direction_libm(angle_deg: u32) -> (f64, f64) {
    let adjusted = angle_deg as f64 - 90.0;
    let radians = adjusted * (std::f64::consts::PI / 180.0);
    (radians.cos(), radians.sin())
}

/// Rounded sample coordinate of radius `r` on the ray with direction `(cos, sin)`:
/// `Math.round(x + r * Math.cos(radians))`, `Math.round(y + r * Math.sin(radians))`.
#[inline]
pub fn ray_point(cx: f64, cy: f64, r: u32, cos: f64, sin: f64) -> (f64, f64) {
    let rf = r as f64;
    (round(cx + rf * cos), round(cy + rf * sin))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn same(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits()
    }

    #[test]
    fn math_round_semantics() {
        // Values checked against V8 (`node -p "Math.round(x)"`).
        let cases: &[(f64, f64)] = &[
            (0.5, 1.0),
            (1.5, 2.0),
            (2.5, 3.0),
            (-0.5, -0.0),
            (-1.5, -1.0),
            (-2.5, -2.0),
            (-0.4, -0.0),
            (-0.6, -1.0),
            (0.49999999999999994, 0.0),
            (-0.49999999999999994, -0.0),
            (4503599627370495.5, 4503599627370496.0),
            (4503599627370497.0, 4503599627370497.0),
            (-4503599627370495.5, -4503599627370495.0),
            (63.5, 64.0),
            (63.49999999999999, 63.0),
            (-1e-300, -0.0),
            (1e-300, 0.0),
            (0.0, 0.0),
            (-0.0, -0.0),
        ];
        for &(x, want) in cases {
            assert!(
                same(round(x), want),
                "round({x:e}) = {:e}, want {want:e}",
                round(x)
            );
        }
        assert!(round(f64::NAN).is_nan());
        assert_eq!(round(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn luma_bounds() {
        assert_eq!(luma(0, 0, 0), 0);
        assert_eq!(luma(255, 255, 255), 255);
        let v = 255.0 * R_COEF + 255.0 * G_COEF + 255.0 * B_COEF;
        assert!(v < 255.5);
    }

    #[test]
    fn luma_exhaustive_is_bounded() {
        // Every one of the 2^24 colours rounds into 0..=255 (no overflow of the 256-bin histogram).
        for r in 0..=255u32 {
            for g in 0..=255u32 {
                for b in 0..=255u32 {
                    let v = r as f64 * R_COEF + g as f64 * G_COEF + b as f64 * B_COEF;
                    assert!(round(v) <= 255.0);
                }
            }
        }
    }

    #[test]
    fn luma_fast_matches_luma_exhaustively() {
        for r in 0..=255u8 {
            for g in 0..=255u8 {
                for b in 0..=255u8 {
                    assert_eq!(luma_fast(r, g, b), luma(r, g, b), "colour ({r}, {g}, {b})");
                }
            }
        }
    }

    #[test]
    fn table_cardinal_directions() {
        assert_eq!(direction(90), (1.0, 0.0));
        assert_eq!(direction(0).1, -1.0);
        assert_eq!(direction(270).0, -1.0);
        assert!(direction(0).0 > 0.0 && direction(0).0 < 1e-16);
    }
}

#[cfg(test)]
mod libm_comparison {
    use super::*;

    /// Diagnostic: how many of the 360 directions differ between V8 and host libm.
    /// Printed with `cargo test -- --nocapture`; never fails.
    #[test]
    fn report_v8_vs_libm() {
        let mut diffs = Vec::new();
        for a in 0..360 {
            let (c8, s8) = direction(a);
            let (cl, sl) = direction_libm(a);
            let radians = (a as f64 - 90.0) * (std::f64::consts::PI / 180.0);
            assert_eq!(
                radians.to_bits(),
                super::radians(a).to_bits(),
                "radians differ at {a}"
            );
            if c8.to_bits() != cl.to_bits() || s8.to_bits() != sl.to_bits() {
                diffs.push((a, c8, cl, s8, sl));
            }
        }
        println!("V8_VS_LIBM_DIRECTION_DIFFS {}", diffs.len());
        for d in &diffs {
            println!(
                "  angle {} cos v8={:e} libm={:e} sin v8={:e} libm={:e}",
                d.0, d.1, d.2, d.3, d.4
            );
        }
    }
}
