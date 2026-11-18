//! The optimized engine must be bit-identical to the literal reference port.

use cdch_core::engine::{checkpoint_bin, CHECKPOINTS};
use cdch_core::{reference, Center, Params, RgbImage, SamplingPlan, Workspace, BINS};

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
}

fn bits_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn check(w: usize, h: usize, pixels: &[u8], params: Params) {
    let img = RgbImage::new(w, h, pixels).unwrap();
    let want = reference::cdch(&img, &params).unwrap();
    let plan = SamplingPlan::new(w, h, params).unwrap();
    let mut ws = Workspace::new();
    let got = plan.matrix(&img, &mut ws).unwrap();
    assert!(
        bits_eq(want.as_slice(), got.as_slice()),
        "matrix mismatch {w}x{h} {params:?}"
    );
    let cp = plan.checkpoints(&img, &mut ws).unwrap();
    for i in 0..want.rows() {
        for j in 0..CHECKPOINTS {
            let (a, b) = (
                want.row(i)[checkpoint_bin(j)],
                cp.values[i * CHECKPOINTS + j],
            );
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "checkpoint mismatch ray {i} j {j}"
            );
        }
    }
    // Trace coordinates agree with the plan, and counts reconstruct the CDF.
    let (_, traces) = reference::cdch_traced(&img, &params, reference::Trig::V8Table).unwrap();
    let hists = plan.histograms(&img, &mut ws).unwrap();
    for (i, t) in traces.iter().enumerate() {
        let coords: Vec<(i64, i64)> = plan
            .ray_coords(i)
            .map(|(x, y)| (x as i64, y as i64))
            .collect();
        assert_eq!(coords, t.coords);
        assert_eq!(&hists[i * BINS..(i + 1) * BINS], &t.counts[..]);
    }
    // The fused single-pass output equals both separate outputs bit for bit.
    // Buffers start dirty to prove every value is overwritten.
    let mut counts = vec![7u32; plan.n_rays() * BINS];
    let mut fused = vec![-1.0f64; plan.n_rays() * BINS];
    plan.histograms_and_matrix_into(&img, &mut ws, &mut counts, &mut fused)
        .unwrap();
    assert_eq!(counts, hists, "fused counts mismatch {w}x{h} {params:?}");
    assert!(
        bits_eq(want.as_slice(), &fused),
        "fused matrix mismatch {w}x{h} {params:?}"
    );
}

#[test]
fn random_geometries_and_pixels() {
    let mut rng = SplitMix(20261116);
    for case in 0..300 {
        let w = 1 + rng.below(if case % 10 == 0 { 300 } else { 40 }) as usize;
        let h = 1 + rng.below(if case % 10 == 5 { 300 } else { 40 }) as usize;
        let pixels: Vec<u8> = (0..w * h * 3).map(|_| rng.next() as u8).collect();
        let center = match case % 6 {
            0 => Center::archived(w, h),
            1 => Center::midpoint(w, h),
            2 => Center::new(
                rng.below(w as u64) as f64 + 0.5,
                rng.below(h as u64) as f64 + 0.5,
            ),
            3 => Center::new(
                (rng.next() % 4000) as f64 / 100.0 - 5.0,
                (rng.next() % 4000) as f64 / 100.0 - 5.0,
            ),
            4 => Center::new(-3.0, h as f64 + 2.0), // outside: NaN rows
            _ => Center::new(0.0, 0.0),
        };
        let step = [1u32, 1, 1, 2, 7, 45, 90, 359, 360][case % 9];
        check(
            w,
            h,
            &pixels,
            Params {
                center,
                angle_step: step,
            },
        );
    }
}

#[test]
fn archived_128_protocol() {
    let mut rng = SplitMix(7);
    let pixels: Vec<u8> = (0..128 * 128 * 3).map(|_| rng.next() as u8).collect();
    check(128, 128, &pixels, Params::archived(128, 128));
}

#[test]
fn strided_view_matches_packed() {
    let (w, h, pad) = (37usize, 23usize, 11usize);
    let mut rng = SplitMix(99);
    let packed: Vec<u8> = (0..w * h * 3).map(|_| rng.next() as u8).collect();
    let stride = 3 * w + pad;
    let mut strided = vec![0u8; stride * h];
    for y in 0..h {
        strided[y * stride..y * stride + 3 * w]
            .copy_from_slice(&packed[y * 3 * w..(y + 1) * 3 * w]);
    }
    let params = Params::archived(w, h);
    let plan = SamplingPlan::new(w, h, params).unwrap();
    let mut ws = Workspace::new();
    let a = plan
        .matrix(&RgbImage::new(w, h, &packed).unwrap(), &mut ws)
        .unwrap();
    let b = plan
        .matrix(
            &RgbImage::with_stride(w, h, stride, &strided).unwrap(),
            &mut ws,
        )
        .unwrap();
    assert!(bits_eq(a.as_slice(), b.as_slice()));
}

#[test]
fn empty_rays_are_nan_like_javascript() {
    let pixels = vec![10u8; 4 * 4 * 3];
    let img = RgbImage::new(4, 4, &pixels).unwrap();
    let m = reference::cdch(
        &img,
        &Params {
            center: Center::new(-1.0, -1.0),
            angle_step: 1,
        },
    )
    .unwrap();
    assert!(m.as_slice().iter().all(|v| v.is_nan()));
}

#[test]
fn invalid_inputs_are_errors() {
    let pixels = vec![0u8; 10];
    assert!(RgbImage::new(4, 4, &pixels).is_err());
    let ok = vec![0u8; 48];
    let img = RgbImage::new(4, 4, &ok).unwrap();
    assert!(reference::cdch(
        &img,
        &Params {
            center: Center::new(f64::NAN, 1.0),
            angle_step: 1
        }
    )
    .is_err());
    assert!(reference::cdch(
        &img,
        &Params {
            center: Center::new(1.0, 1.0),
            angle_step: 0
        }
    )
    .is_err());
    assert!(reference::cdch(
        &img,
        &Params {
            center: Center::new(1.0, 1.0),
            angle_step: 361
        }
    )
    .is_err());
}
