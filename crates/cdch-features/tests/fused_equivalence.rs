//! Fused (checkpoint) extraction must be bit-identical to matrix-based extraction.

use cdch_core::{Center, Params, RgbImage};
use cdch_features::{FeatureExtractor, Representation};

fn rng_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        })
        .collect()
}

#[test]
fn fused_equals_matrix_path() {
    let mut fx = FeatureExtractor::new();
    for seed in 1..40u64 {
        let (w, h) = match seed % 4 {
            0 => (128, 128),
            1 => (97, 140),
            2 => (200, 64),
            _ => (33, 33),
        };
        let mut px = rng_bytes(seed, w * h * 3);
        if seed % 5 == 0 {
            // smooth image: few distinct intensities, many repeated bins
            for (i, p) in px.iter_mut().enumerate() {
                *p = ((i / 3) % w) as u8;
            }
        }
        let img = RgbImage::new(w, h, &px).unwrap();
        for params in [
            Params::archived(w, h),
            Params {
                center: Center::new(w as f64 / 2.0 - 0.5, h as f64 / 3.0),
                angle_step: 1,
            },
        ] {
            for rep in [
                Representation::A,
                Representation::B,
                Representation::C,
                Representation::D,
            ] {
                let mut a = vec![0.0; rep.dim()];
                let mut b = vec![0.0; rep.dim()];
                fx.extract(&img, &params, rep, &mut a).unwrap();
                fx.extract_via_matrix(&img, &params, rep, &mut b).unwrap();
                for (k, (x, y)) in a.iter().zip(&b).enumerate() {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "{rep:?} feature {k} seed {seed}: {x} vs {y}"
                    );
                }
            }
        }
    }
}

#[test]
fn empty_ray_is_rejected_for_a_like_python() {
    let px = vec![0u8; 16 * 16 * 3];
    let img = RgbImage::new(16, 16, &px).unwrap();
    let params = Params {
        center: Center::new(-5.0, 3.0),
        angle_step: 1,
    };
    let mut fx = FeatureExtractor::new();
    let mut out = vec![0.0; 56];
    assert!(fx
        .extract(&img, &params, Representation::A, &mut out)
        .is_err());
    assert!(fx
        .extract_via_matrix(&img, &params, Representation::A, &mut out)
        .is_err());
    // C does not validate in the Python pipeline; NaN propagates identically.
    let mut c1 = vec![0.0; 768];
    let mut c2 = vec![0.0; 768];
    fx.extract(&img, &params, Representation::C, &mut c1)
        .unwrap();
    fx.extract_via_matrix(&img, &params, Representation::C, &mut c2)
        .unwrap();
    assert!(c1
        .iter()
        .zip(&c2)
        .all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())));
}
