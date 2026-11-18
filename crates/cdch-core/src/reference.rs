//! Correctness-first port of `drawRaysAndGenerateHistograms` and
//! `calculateCumulativeHistogram` from `ip-1.js`.
//!
//! Kept intentionally literal. Each statement corresponds to a line of the
//! JavaScript (quoted in comments). It is the baseline the optimized
//! [`crate::engine`] is tested against.

use crate::{js, CdchMatrix, Params, Result, RgbImage, BINS};

/// Which trigonometric source to use for the ray directions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trig {
    /// V8 bit patterns (canonical; default).
    V8Table,
    /// Host libm `f64::cos`/`f64::sin` (diagnostic only).
    Libm,
}

/// Per-ray trace: the rounded coordinates visited, in order, and the integer histogram.
#[derive(Clone, Debug, PartialEq)]
pub struct RayTrace {
    pub angle: u32,
    pub coords: Vec<(i64, i64)>,
    pub counts: Vec<u32>,
}

/// Canonical CDCH with the V8 direction table.
pub fn cdch(img: &RgbImage<'_>, params: &Params) -> Result<CdchMatrix> {
    cdch_with(img, params, Trig::V8Table, None)
}

/// Canonical CDCH plus a full per-ray trace (coordinates, multiplicities, counts).
pub fn cdch_traced(
    img: &RgbImage<'_>,
    params: &Params,
    trig: Trig,
) -> Result<(CdchMatrix, Vec<RayTrace>)> {
    let mut traces = Vec::with_capacity(params.n_rays());
    let m = cdch_with(img, params, trig, Some(&mut traces))?;
    Ok((m, traces))
}

pub fn cdch_with(
    img: &RgbImage<'_>,
    params: &Params,
    trig: Trig,
    mut trace: Option<&mut Vec<RayTrace>>,
) -> Result<CdchMatrix> {
    params.validate()?;
    let (x, y) = (params.center.x, params.center.y);
    // rayCanvas.width / rayCanvas.height equal the image size under the oracle.
    let width = img.width() as f64;
    let height = img.height() as f64;
    let mut data = Vec::with_capacity(params.n_rays() * BINS);

    // for (let angle = 0; angle < 360; angle += step) {
    let mut angle: u32 = 0;
    while angle < 360 {
        // const adjustedAngle = angle - 90; const radians = adjustedAngle * (Math.PI / 180);
        let (cos, sin) = match trig {
            Trig::V8Table => js::direction(angle),
            Trig::Libm => js::direction_libm(angle),
        };
        // let rayHistogram = new Array(256).fill(0);
        let mut ray_histogram = [0u32; BINS];
        let mut coords = Vec::new();

        // for (let r = 0; r < rayCanvas.width; r++) {
        let mut r: u32 = 0;
        while (r as f64) < width {
            // let px = Math.round(x + r * Math.cos(radians));
            // let py = Math.round(y + r * Math.sin(radians));
            let (px, py) = js::ray_point(x, y, r, cos, sin);
            // if (px < 0 || px >= rayCanvas.width || py < 0 || py >= rayCanvas.height) break;
            if px < 0.0 || px >= width || py < 0.0 || py >= height {
                break;
            }
            // let pixelData = hiddenContext.getImageData(px, py, 1, 1).data;
            let [red, green, blue] = img.rgb(px as usize, py as usize);
            // let intensity = Math.round(pixelData[0] * R_COEF + pixelData[1] * G_COEF + pixelData[2] * B_COEF);
            let intensity = js::luma(red, green, blue);
            // rayHistogram[intensity]++;
            ray_histogram[intensity as usize] += 1;
            if trace.is_some() {
                coords.push((px as i64, py as i64));
            }
            r += 1;
        }

        // const cumulativeHist = calculateCumulativeHistogram(rayHistogram);
        calculate_cumulative_histogram(&ray_histogram, &mut data);
        if let Some(t) = trace.as_deref_mut() {
            t.push(RayTrace {
                angle,
                coords,
                counts: ray_histogram.to_vec(),
            });
        }
        angle += params.angle_step;
    }
    CdchMatrix::from_vec(params.n_rays(), data)
}

/// ```js
/// function calculateCumulativeHistogram(hist) {
///   let cumulative = new Array(256).fill(0);
///   cumulative[0] = hist[0];
///   for (let i = 1; i < hist.length; i++) cumulative[i] = cumulative[i - 1] + hist[i];
///   let max = cumulative[255];
///   return cumulative.map(value => value / max);
/// }
/// ```
/// The JavaScript accumulates in Number (binary64); every partial sum is an exact
/// integer below 2^53, so accumulating in f64 is identical. An empty ray yields
/// `0 / 0 = NaN` in every bin, as in JavaScript.
pub fn calculate_cumulative_histogram(hist: &[u32; BINS], out: &mut Vec<f64>) {
    let mut cumulative = [0f64; BINS];
    cumulative[0] = hist[0] as f64;
    for i in 1..BINS {
        cumulative[i] = cumulative[i - 1] + hist[i] as f64;
    }
    let max = cumulative[255];
    out.extend(cumulative.iter().map(|&value| value / max));
}
