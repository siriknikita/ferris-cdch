//! Deterministic synthetic differential-test fixtures.

use crate::{OwnedRgb, SplitMix};
use cdch_core::js::{B_COEF, G_COEF, R_COEF};

#[derive(Clone, Debug)]
pub struct Case {
    pub id: String,
    pub category: &'static str,
    pub img: OwnedRgb,
    /// `None` exercises the oracle's own default centre (`w/2`, `h/2`).
    pub center: Option<(f64, f64)>,
    pub angle_step: u32,
}

fn image(w: usize, h: usize, f: impl Fn(usize, usize) -> [u8; 3]) -> OwnedRgb {
    let mut data = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            data.extend_from_slice(&f(x, y));
        }
    }
    OwnedRgb { width: w, height: h, data }
}

fn random(w: usize, h: usize, seed: u64) -> OwnedRgb {
    let mut rng = SplitMix(seed);
    let mut data = vec![0u8; w * h * 3];
    rng.fill(&mut data);
    OwnedRgb { width: w, height: h, data }
}

/// Colours whose binary64 luma lies within `tol` of a half-integer: the cases where
/// `Math.round` tie-breaking and the exact evaluation order of
/// `R*0.2125 + G*0.7154 + B*0.0721` decide the bin.
pub fn luma_tie_colours(tol: f64, limit: usize) -> Vec<[u8; 3]> {
    let mut out = Vec::new();
    'outer: for r in 0..=255u32 {
        for g in 0..=255u32 {
            for b in 0..=255u32 {
                let v = r as f64 * R_COEF + g as f64 * G_COEF + b as f64 * B_COEF;
                if (v - v.floor() - 0.5).abs() <= tol {
                    out.push([r as u8, g as u8, b as u8]);
                    if out.len() >= limit {
                        break 'outer;
                    }
                }
            }
        }
    }
    out
}

pub fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    let mut push = |id: String, category: &'static str, img: OwnedRgb, center: Option<(f64, f64)>, step: u32| {
        v.push(Case { id, category, img, center, angle_step: step });
    };
    let archived = |w: usize, h: usize| Some(((w / 2) as f64, (h / 2) as f64));

    // Uniform images.
    push("black_128".into(), "uniform", image(128, 128, |_, _| [0, 0, 0]), archived(128, 128), 1);
    push("white_128".into(), "uniform", image(128, 128, |_, _| [255, 255, 255]), archived(128, 128), 1);
    for c in [[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 64, 200], [17, 230, 99]] {
        push(format!("const_{}_{}_{}_128", c[0], c[1], c[2]), "uniform", image(128, 128, move |_, _| c), archived(128, 128), 1);
    }

    // Gradients and patterns.
    push("hgrad_128".into(), "gradient", image(128, 128, |x, _| [(2 * x) as u8; 3]), archived(128, 128), 1);
    push("vgrad_128".into(), "gradient", image(128, 128, |_, y| [(2 * y) as u8; 3]), archived(128, 128), 1);
    push("hgrad_rgb_256".into(), "gradient", image(256, 64, |x, y| [x as u8, (255 - x) as u8, (4 * y) as u8]), archived(256, 64), 1);
    push("diag_grad_128".into(), "gradient", image(128, 128, |x, y| [(x + y) as u8, x as u8, y as u8]), archived(128, 128), 1);
    push("radial_128".into(), "gradient", image(128, 128, |x, y| {
        let d = (((x as f64 - 64.0).powi(2) + (y as f64 - 64.0).powi(2)).sqrt() * 2.0).min(255.0) as u8;
        [d, 255 - d, d / 2]
    }), archived(128, 128), 1);
    push("checker1_128".into(), "pattern", image(128, 128, |x, y| if (x + y) % 2 == 0 { [0; 3] } else { [255; 3] }), archived(128, 128), 1);
    push("checker8_128".into(), "pattern", image(128, 128, |x, y| if (x / 8 + y / 8) % 2 == 0 { [30, 60, 90] } else { [220, 180, 140] }), archived(128, 128), 1);
    push("stripes_v_128".into(), "pattern", image(128, 128, |x, _| if x % 3 == 0 { [250, 10, 10] } else { [10, 10, 250] }), archived(128, 128), 1);

    // Random RGB.
    for seed in 0..8u64 {
        push(format!("random_s{seed}_128"), "random", random(128, 128, 1000 + seed), archived(128, 128), 1);
    }

    // Luma tie colours (floating-point rounding of the intensity).
    let ties = luma_tie_colours(1e-9, 4096);
    let tie_img = {
        let mut rng = SplitMix(4242);
        let t = ties.clone();
        let picks: Vec<[u8; 3]> = (0..128 * 128).map(|_| t[rng.below(t.len() as u64) as usize]).collect();
        image(128, 128, move |x, y| picks[y * 128 + x])
    };
    push(format!("luma_ties_{}colours_128", ties.len()), "fp_luma", tie_img, archived(128, 128), 1);

    // Small and degenerate sizes.
    for (w, h) in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3), (3, 5), (5, 3), (7, 1), (1, 7), (4, 4)] {
        push(format!("small_{w}x{h}"), "small", random(w, h, (w * 31 + h) as u64), archived(w, h), 1);
        push(format!("small_{w}x{h}_midpoint"), "small", random(w, h, (w * 37 + h) as u64), Some((w as f64 / 2.0, h as f64 / 2.0)), 1);
    }

    // Rectangular (the radius limit is the canvas WIDTH, even for tall images).
    for (w, h) in [(200, 50), (50, 200), (128, 64), (64, 128), (300, 17), (17, 300), (129, 65), (127, 127), (500, 500)] {
        push(format!("rect_{w}x{h}"), "rectangular", random(w, h, (w * 7919 + h) as u64), archived(w, h), 1);
    }
    push("rect_1024x768".into(), "rectangular", random(1024, 768, 77), archived(1024, 768), 1);
    push("rect_127x127_midpoint".into(), "rounding", random(127, 127, 5), Some((63.5, 63.5)), 1);
    push("oracle_default_center_129x65".into(), "rounding", random(129, 65, 6), None, 1);

    // Centres designed to stress coordinate rounding and boundaries (on a 128x128 random image).
    let base = random(128, 128, 31337);
    let centres: &[(&str, f64, f64, &'static str)] = &[
        ("half_63.5_63.5", 63.5, 63.5, "rounding"),
        ("half_64.5_64.5", 64.5, 64.5, "rounding"),
        ("half_64.5_64", 64.5, 64.0, "rounding"),
        ("quarter_10.25_99.75", 10.25, 99.75, "rounding"),
        ("third_33.3_77.7", 33.3, 77.7, "rounding"),
        ("near_half_0.49999999999999994", 0.49999999999999994, 0.5, "rounding"),
        ("tiny_offset_64+1e-13", 64.0 + 1e-13, 64.0 - 1e-13, "rounding"),
        ("corner_0_0", 0.0, 0.0, "border"),
        ("corner_127_127", 127.0, 127.0, "border"),
        ("edge_127.5_0", 127.5, 0.0, "border"),
        ("edge_0_64", 0.0, 64.0, "border"),
        ("negative_half_-0.5_-0.5", -0.5, -0.5, "border"),
        ("outside_-0.6_5", -0.6, 5.0, "border"),
        ("outside_200_64", 200.0, 64.0, "border"),
        ("outside_64_-40", 64.0, -40.0, "border"),
    ];
    for &(name, x, y, cat) in centres {
        push(format!("centre_{name}"), cat, base.clone(), Some((x, y)), 1);
    }

    // Angle steps (integer steps accepted by ip-1.js's UI validation).
    let small = random(64, 48, 99);
    for step in [2u32, 3, 5, 7, 10, 45, 90, 120, 359, 360] {
        push(format!("step_{step}_64x48"), "angle_step", small.clone(), archived(64, 48), step);
    }
    push("step_7_half_centre".into(), "angle_step", small.clone(), Some((31.5, 23.5)), 7);
    v
}
