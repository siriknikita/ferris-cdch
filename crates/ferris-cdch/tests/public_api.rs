//! Integration tests of the public `ferris-cdch` API. Only public items are used.
//! The literal reference port (`ferris_cdch::core::reference`) serves as the
//! independent oracle; it is itself verified against `ip-1.js` by `ferris-diff`.

use std::sync::Arc;

use ferris_cdch::core::reference;
use ferris_cdch::{
    compute, intensity_centroid, luma, luma_plane, rgb_view, AlphaPolicy, Analyzer, CdchConfig,
    CenterSpec, FerrisError, InvalidInput, OwnedRgbImage, RgbImage, RgbaImage, INTENSITY_BINS,
};

struct SplitMix(u64);
impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn pixels(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

fn bits_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Facade output must equal the literal reference port bit for bit, and its
/// counts and ray lengths must be consistent with the matrix.
fn assert_matches_reference(img: &RgbImage<'_>, cfg: &CdchConfig) {
    let analyzer = Analyzer::new();
    let got = analyzer.compute(img, cfg).unwrap();
    let params = cfg.params(img.width(), img.height()).unwrap();
    let (want, traces) = reference::cdch_traced(img, &params, reference::Trig::V8Table).unwrap();
    assert!(
        bits_eq(got.matrix().as_slice(), want.as_slice()),
        "matrix differs from reference for {}x{} {cfg:?}",
        img.width(),
        img.height()
    );
    assert_eq!(got.n_rays(), traces.len());
    for (i, t) in traces.iter().enumerate() {
        assert_eq!(got.angle(i), t.angle);
        assert_eq!(got.counts(i), &t.counts[..]);
        assert_eq!(got.ray_length(i) as usize, t.coords.len());
    }
    let fast = analyzer.matrix(img, cfg).unwrap();
    assert!(bits_eq(fast.as_slice(), want.as_slice()));
}

#[test]
fn random_images_match_reference_for_all_center_kinds() {
    let mut rng = SplitMix(0x00FE_8815);
    for case in 0..120 {
        let w = 1 + rng.below(if case % 7 == 0 { 200 } else { 48 }) as usize;
        let h = 1 + rng.below(if case % 7 == 3 { 200 } else { 48 }) as usize;
        let px = rng.pixels(w * h * 3);
        let img = rgb_view(w, h, &px).unwrap();
        let center = match case % 5 {
            0 => CenterSpec::Archived,
            1 => CenterSpec::Midpoint,
            2 => CenterSpec::Explicit {
                x: rng.below(w as u64) as f64,
                y: rng.below(h as u64) as f64,
            },
            3 => CenterSpec::Explicit {
                x: rng.below(w as u64 * 10) as f64 / 10.0 + 0.05,
                y: rng.below(h as u64 * 10) as f64 / 10.0 - 0.45,
            },
            _ => CenterSpec::Explicit { x: -0.5, y: -0.5 },
        };
        let step = [1, 1, 2, 3, 5, 7, 10, 30, 45, 90, 359, 360][case % 12];
        assert_matches_reference(
            &img,
            &CdchConfig {
                center,
                angle_step: step,
            },
        );
    }
}

#[test]
fn default_is_archived_protocol() {
    let cfg = CdchConfig::default();
    assert_eq!(cfg, CdchConfig::archived());
    assert_eq!(cfg.n_rays().unwrap(), 360);
    let p = cfg.params(129, 64).unwrap();
    assert_eq!((p.center.x, p.center.y), (64.0, 32.0));
    let p = CdchConfig::default()
        .with_center(CenterSpec::Midpoint)
        .params(129, 64)
        .unwrap();
    assert_eq!((p.center.x, p.center.y), (64.5, 32.0));
}

#[test]
fn degenerate_sizes() {
    // 1x1: every ray samples only the centre pixel once.
    let px = [10u8, 200, 30];
    let img = rgb_view(1, 1, &px).unwrap();
    let r = compute(&img, &CdchConfig::default()).unwrap();
    assert!(r.is_complete());
    let l = luma(10, 200, 30) as usize;
    for i in 0..r.n_rays() {
        assert_eq!(r.ray_length(i), 1);
        assert_eq!(r.counts(i)[l], 1);
        assert!(r.cdf(i)[..l].iter().all(|&v| v == 0.0));
        assert!(r.cdf(i)[l..].iter().all(|&v| v == 1.0));
    }
    // Thin strips, both orientations, and the radius limit = width for tall images.
    let mut rng = SplitMix(5);
    for (w, h) in [(1, 37), (37, 1), (2, 300), (300, 2), (8, 2048), (2048, 8)] {
        let px = rng.pixels(w * h * 3);
        let img = rgb_view(w, h, &px).unwrap();
        for center in [CenterSpec::Archived, CenterSpec::Midpoint] {
            assert_matches_reference(
                &img,
                &CdchConfig {
                    center,
                    angle_step: 1,
                },
            );
        }
    }
    // Tall image: the downward ray stops after `width` samples, not at the bottom.
    let (w, h) = (8usize, 200usize);
    let px = vec![0u8; w * h * 3];
    let r = compute(
        &rgb_view(w, h, &px).unwrap(),
        &CdchConfig::default()
            .with_center(CenterSpec::Explicit { x: 4.0, y: 0.0 })
            .with_angle_step(90),
    )
    .unwrap();
    assert_eq!(
        r.ray_length(2),
        w as u32,
        "180° ray is limited to width samples"
    );
}

#[test]
fn invalid_inputs_are_typed_errors_not_panics() {
    let ok = vec![0u8; 4 * 4 * 3];
    let img = rgb_view(4, 4, &ok).unwrap();
    let expect_invalid = |r: Result<_, FerrisError>, want: InvalidInput| match r {
        Err(FerrisError::InvalidInput(e)) => assert_eq!(e, want),
        other => panic!("expected {want:?}, got {other:?}"),
    };
    expect_invalid(
        rgb_view(0, 4, &ok).map(|_| ()),
        InvalidInput::EmptyImage {
            width: 0,
            height: 4,
        },
    );
    expect_invalid(
        rgb_view(4, 4, &ok[..10]).map(|_| ()),
        InvalidInput::BufferSize {
            expected: 48,
            actual: 10,
        },
    );
    expect_invalid(
        OwnedRgbImage::new(4, 4, vec![0; 47]).map(|_| ()),
        InvalidInput::BufferSize {
            expected: 48,
            actual: 47,
        },
    );
    // Too large is detected from the geometry alone, before any buffer is touched.
    expect_invalid(
        rgb_view(100_000, 100_000, &ok).map(|_| ()),
        InvalidInput::TooLarge {
            width: 100_000,
            height: 100_000,
        },
    );
    expect_invalid(
        RgbaImage::with_stride(4, 4, 15, &ok).map(|_| ()),
        InvalidInput::RowStride {
            stride: 15,
            width: 4,
            channels: 4,
        },
    );
    for step in [0, 361, u32::MAX] {
        expect_invalid(
            compute(&img, &CdchConfig::default().with_angle_step(step)).map(|_| ()),
            InvalidInput::AngleStep(step),
        );
    }
    for (x, y) in [
        (f64::NAN, 1.0),
        (1.0, f64::INFINITY),
        (f64::NEG_INFINITY, 0.0),
    ] {
        let r = compute(
            &img,
            &CdchConfig::default().with_center(CenterSpec::Explicit { x, y }),
        );
        assert!(matches!(
            r,
            Err(FerrisError::InvalidInput(
                InvalidInput::NonFiniteCenter { .. }
            ))
        ));
        assert!(r.unwrap_err().is_invalid_input());
    }
    // An empty image built through the low-level type is also rejected by the analyzer.
    let empty = RgbImage::new(0, 0, &[]).unwrap();
    expect_invalid(
        compute(&empty, &CdchConfig::default()).map(|_| ()),
        InvalidInput::EmptyImage {
            width: 0,
            height: 0,
        },
    );
}

#[test]
fn centre_outside_image_gives_nan_rows_like_javascript() {
    let px = vec![77u8; 5 * 5 * 3];
    let img = rgb_view(5, 5, &px).unwrap();
    let r = compute(
        &img,
        &CdchConfig::default().with_center(CenterSpec::Explicit { x: -3.0, y: 2.0 }),
    )
    .unwrap();
    // Every ray's first sample (r = 0) is the centre itself, so the JavaScript breaks
    // out of every radius loop immediately: all rays are empty and all bins NaN.
    assert!(!r.is_complete());
    assert_eq!(r.empty_rays().len(), r.n_rays());
    assert_eq!(r.total_samples(), 0);
    for i in 0..r.n_rays() {
        assert!(r.cdf(i).iter().all(|v| v.is_nan()));
        assert!(r.counts(i).iter().all(|&c| c == 0));
    }
    // A centre of (-0.5, -0.5) rounds to (-0, -0), which passes `px < 0`, so it is inside.
    let r = compute(
        &img,
        &CdchConfig::default().with_center(CenterSpec::Explicit { x: -0.5, y: -0.5 }),
    )
    .unwrap();
    assert!(r.is_complete());
}

#[test]
fn histogram_properties() {
    let mut rng = SplitMix(42);
    let (w, h) = (97, 61);
    let px = rng.pixels(w * h * 3);
    let r = compute(&rgb_view(w, h, &px).unwrap(), &CdchConfig::default()).unwrap();
    assert!(r.is_complete());
    assert_eq!(
        r.total_samples(),
        r.ray_lengths().iter().map(|&n| n as u64).sum::<u64>()
    );
    for i in 0..r.n_rays() {
        let n = r.ray_length(i);
        assert_eq!(r.counts(i).iter().sum::<u32>(), n, "counts are conserved");
        let cdf = r.cdf(i);
        assert_eq!(cdf.len(), INTENSITY_BINS);
        assert!(cdf.windows(2).all(|p| p[0] <= p[1]), "monotone");
        assert!(cdf.iter().all(|&v| (0.0..=1.0).contains(&v)));
        assert_eq!(cdf[255], 1.0);
        let mut cum = 0u32;
        for (&value, &count) in cdf.iter().zip(r.counts(i)) {
            cum += count;
            assert_eq!(value.to_bits(), (cum as f64 / n as f64).to_bits());
        }
    }
}

#[test]
fn deterministic_and_cached() {
    let mut rng = SplitMix(9);
    let (w, h) = (64, 64);
    let px = rng.pixels(w * h * 3);
    let img = rgb_view(w, h, &px).unwrap();
    let analyzer = Analyzer::new();
    let first = analyzer.compute(&img, &CdchConfig::default()).unwrap();
    for _ in 0..20 {
        assert_eq!(
            analyzer.compute(&img, &CdchConfig::default()).unwrap(),
            first
        );
    }
    assert_eq!(analyzer.cached_plans(), 1, "one geometry, one plan");
    // A different configuration in between does not disturb later results.
    analyzer
        .compute(&img, &CdchConfig::default().with_angle_step(7))
        .unwrap();
    assert_eq!(
        analyzer.compute(&img, &CdchConfig::default()).unwrap(),
        first
    );
    assert_eq!(analyzer.cached_plans(), 2);
    // A fresh analyzer agrees.
    assert_eq!(compute(&img, &CdchConfig::default()).unwrap(), first);
}

#[test]
fn analyzer_is_thread_safe() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Analyzer>();
    assert_send_sync::<ferris_cdch::CdchResult>();

    let mut rng = SplitMix(1234);
    let images: Arc<Vec<(usize, usize, Vec<u8>)>> = Arc::new(
        (0..6)
            .map(|k| {
                let (w, h) = (20 + 7 * k, 30 + 5 * k);
                (w, h, rng.pixels(w * h * 3))
            })
            .collect(),
    );
    let sequential: Vec<_> = images
        .iter()
        .map(|(w, h, px)| compute(&rgb_view(*w, *h, px).unwrap(), &CdchConfig::default()).unwrap())
        .collect();
    let shared = Analyzer::new();
    let handles: Vec<_> = (0..8)
        .map(|t| {
            let (analyzer, images) = (shared.clone(), Arc::clone(&images));
            std::thread::spawn(move || {
                (0..30)
                    .map(|k| {
                        let idx = (k + t) % images.len();
                        let (w, h, px) = &images[idx];
                        let img = rgb_view(*w, *h, px).unwrap();
                        (idx, analyzer.compute(&img, &CdchConfig::default()).unwrap())
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    for h in handles {
        for (idx, r) in h.join().unwrap() {
            assert_eq!(r, sequential[idx]);
        }
    }
    assert_eq!(shared.cached_plans(), images.len());
}

#[test]
fn strided_borrowed_views_need_no_copy_and_agree() {
    let mut rng = SplitMix(77);
    let (w, h, pad) = (31usize, 17usize, 9usize);
    let packed = rng.pixels(w * h * 3);
    let stride = 3 * w + pad;
    let mut strided = vec![0xAAu8; stride * h];
    for y in 0..h {
        strided[y * stride..y * stride + 3 * w]
            .copy_from_slice(&packed[y * 3 * w..(y + 1) * 3 * w]);
    }
    let a = compute(&rgb_view(w, h, &packed).unwrap(), &CdchConfig::default()).unwrap();
    let b = compute(
        &RgbImage::with_stride(w, h, stride, &strided).unwrap(),
        &CdchConfig::default(),
    )
    .unwrap();
    assert_eq!(a, b);
}

#[test]
fn rgba_conversion_policies() {
    let mut rng = SplitMix(2024);
    let (w, h) = (23usize, 19usize);
    let rgb = rng.pixels(w * h * 3);

    // Fully opaque RGBA: every policy reproduces the RGB image exactly.
    let opaque: Vec<u8> = rgb
        .chunks_exact(3)
        .flat_map(|p| [p[0], p[1], p[2], 255])
        .collect();
    let src = RgbaImage::new(w, h, &opaque).unwrap();
    assert!(src.is_opaque());
    for policy in [
        AlphaPolicy::Ignore,
        AlphaPolicy::OVER_WHITE,
        AlphaPolicy::OVER_BLACK,
    ] {
        assert_eq!(OwnedRgbImage::from_rgba(&src, policy).as_bytes(), &rgb[..]);
    }

    // Partially transparent: Ignore keeps stored colours; compositing changes results.
    let alpha = rng.pixels(w * h);
    let translucent: Vec<u8> = rgb
        .chunks_exact(3)
        .zip(&alpha)
        .flat_map(|(p, &a)| [p[0], p[1], p[2], a])
        .collect();
    let src = RgbaImage::new(w, h, &translucent).unwrap();
    assert!(!src.is_opaque());
    let ignored = OwnedRgbImage::from_rgba(&src, AlphaPolicy::Ignore);
    assert_eq!(ignored.as_bytes(), &rgb[..]);
    let white = OwnedRgbImage::from_rgba(&src, AlphaPolicy::OVER_WHITE);
    let cfg = CdchConfig::default();
    assert_ne!(
        compute(&ignored.view(), &cfg).unwrap(),
        compute(&white.view(), &cfg).unwrap()
    );

    // Strided RGBA equals packed RGBA.
    let stride = 4 * w + 5;
    let mut padded = vec![0u8; stride * h];
    for y in 0..h {
        padded[y * stride..y * stride + 4 * w]
            .copy_from_slice(&translucent[y * 4 * w..(y + 1) * 4 * w]);
    }
    let strided = RgbaImage::with_stride(w, h, stride, &padded).unwrap();
    assert_eq!(
        OwnedRgbImage::from_rgba(&strided, AlphaPolicy::OVER_BLACK).as_bytes(),
        OwnedRgbImage::from_rgba(&src, AlphaPolicy::OVER_BLACK).as_bytes()
    );
}

#[test]
fn compositing_is_exact_round_to_nearest_for_all_inputs() {
    // Compare the integer formula with exact rational arithmetic for every colour,
    // alpha and a set of backgrounds (no ties exist because 255 is odd).
    for bg in [0u8, 1, 127, 128, 254, 255] {
        let policy = AlphaPolicy::CompositeOver {
            background: [bg, bg, bg],
        };
        for c in 0..=255u8 {
            for a in 0..=255u8 {
                let num = c as u32 * a as u32 + bg as u32 * (255 - a as u32);
                // nearest integer to num / 255
                let (q, r) = (num / 255, num % 255);
                let want = if 2 * r > 255 { q + 1 } else { q } as u8;
                assert_eq!(policy.apply([c, c, c, a]), [want; 3], "c={c} a={a} bg={bg}");
            }
        }
        assert_eq!(
            policy.apply([9, 9, 9, 0]),
            [bg; 3],
            "transparent shows background"
        );
        assert_eq!(
            policy.apply([9, 10, 11, 255]),
            [9, 10, 11],
            "opaque keeps colour"
        );
    }
}

#[test]
fn gray_replication() {
    let gray = [0u8, 100, 255, 7];
    let img = OwnedRgbImage::from_gray(2, 2, &gray).unwrap();
    assert_eq!(
        img.as_bytes(),
        &[0, 0, 0, 100, 100, 100, 255, 255, 255, 7, 7, 7]
    );
    assert!(OwnedRgbImage::from_gray(3, 2, &gray).is_err());
}

#[test]
fn csv_round_trips_exactly() {
    let mut rng = SplitMix(31337);
    let (w, h) = (41, 29);
    let px = rng.pixels(w * h * 3);
    let r = compute(
        &rgb_view(w, h, &px).unwrap(),
        &CdchConfig::default()
            .with_center(CenterSpec::Explicit { x: 40.3, y: 1.7 })
            .with_angle_step(3),
    )
    .unwrap();
    let mut buf = Vec::new();
    r.write_matrix_csv(&mut buf).unwrap();
    let text = String::from_utf8(buf).unwrap();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    assert_eq!(header.len(), 2 + 256);
    assert_eq!(
        (header[0], header[2], header[257]),
        ("angle_deg", "F_0", "F_255")
    );
    for (i, line) in lines.enumerate() {
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(fields[0].parse::<u32>().unwrap(), r.angle(i));
        assert_eq!(fields[1].parse::<u32>().unwrap(), r.ray_length(i));
        for (b, f) in fields[2..].iter().enumerate() {
            let v: f64 = f.parse().unwrap();
            let want = r.cdf(i)[b];
            assert!(v.to_bits() == want.to_bits() || (v.is_nan() && want.is_nan()));
        }
    }
    let mut buf = Vec::new();
    r.write_counts_csv(&mut buf).unwrap();
    let text = String::from_utf8(buf).unwrap();
    let second = text.lines().nth(1).unwrap();
    let counts: Vec<u32> = second
        .split(',')
        .skip(2)
        .map(|f| f.parse().unwrap())
        .collect();
    assert_eq!(counts, r.counts(0));
}

#[test]
fn intensity_centroid_semantics() {
    // Two qualifying pixels at x = 2 and x = 3 on row 1: mean 2.5 rounds up (JS ties to +inf).
    let (w, h) = (6usize, 3usize);
    let mut px = vec![0u8; w * h * 3];
    for x in [2usize, 3] {
        let i = (w + x) * 3;
        px[i..i + 3].copy_from_slice(&[255, 255, 255]);
    }
    let img = rgb_view(w, h, &px).unwrap();
    let c = intensity_centroid(&img, 200, 255).unwrap().unwrap();
    assert_eq!((c.x, c.y, c.count), (3.0, 1.0, 2));
    // Black pixels: everything except those two.
    let c = intensity_centroid(&img, 0, 0).unwrap().unwrap();
    assert_eq!(c.count, (w * h - 2) as u64);
    // No pixel qualifies.
    assert_eq!(intensity_centroid(&img, 10, 20).unwrap(), None);
    // Invalid range.
    assert!(matches!(
        intensity_centroid(&img, 9, 8),
        Err(FerrisError::InvalidInput(InvalidInput::IntensityRange {
            min: 9,
            max: 8
        }))
    ));
    // Full range: centroid of all pixels = geometric centre, JS-rounded.
    let c = intensity_centroid(&img, 0, 255).unwrap().unwrap();
    assert_eq!((c.x, c.y), (3.0, 1.0)); // mean x 2.5 -> 3, mean y 1.0
}

#[test]
fn luma_plane_uses_canonical_formula() {
    let mut rng = SplitMix(55);
    let (w, h) = (13, 7);
    let px = rng.pixels(w * h * 3);
    let img = rgb_view(w, h, &px).unwrap();
    let plane = luma_plane(&img);
    for (k, p) in px.chunks_exact(3).enumerate() {
        assert_eq!(plane[k], ferris_cdch::core::js::luma(p[0], p[1], p[2]));
    }
}

#[test]
fn build_info_identifies_canonical_source() {
    let info = ferris_cdch::build_info();
    assert_eq!(info.version, env!("CARGO_PKG_VERSION"));
    assert_eq!(
        info.canonical_sha256,
        "110732b48bc6e8b2d5167eec30f7277245913bed98b9baaa3c65c0c22aab44d8"
    );
}

#[cfg(feature = "representations")]
#[test]
fn representations_feature_is_reexported() {
    use ferris_cdch::representations::{FeatureExtractor, Representation};
    let mut rng = SplitMix(3);
    let px = rng.pixels(64 * 64 * 3);
    let img = rgb_view(64, 64, &px).unwrap();
    let params = CdchConfig::default().params(64, 64).unwrap();
    let mut fx = FeatureExtractor::new();
    let mut out = vec![0.0; Representation::A.dim()];
    fx.extract(&img, &params, Representation::A, &mut out)
        .unwrap();
    assert!(out.iter().all(|v| v.is_finite()));
}
