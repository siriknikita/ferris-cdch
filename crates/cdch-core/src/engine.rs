//! Optimized CDCH evaluation on a precomputed [`SamplingPlan`].
//!
//! Per image the work is: luma of each distinct sampled pixel (same formula as
//! `ip-1.js`), one histogram per ray, an integer prefix sum, and one binary64
//! division per output value (`cumulative / max`, never a reciprocal multiply, so
//! results stay bit-identical to JavaScript).

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::{js, CdchError, CdchMatrix, Params, Result, RgbImage, SamplingPlan, BINS};

/// Number of checkpoint bins in the reduced output: `b = 8j + 7`, `j = 0..32`.
/// These are exactly the intensities used by representations A (even `j`), C and D.
pub const CHECKPOINTS: usize = 32;

/// Intensity bin of checkpoint `j`.
#[inline]
pub const fn checkpoint_bin(j: usize) -> usize {
    8 * j + 7
}

/// Reusable per-thread buffers. Allocation happens only when the plan grows.
#[derive(Default, Debug)]
pub struct Workspace {
    lum: Vec<u8>,
}

impl Workspace {
    pub fn new() -> Self {
        Self::default()
    }
}

/// The canonical CDF sampled at the 32 checkpoint bins: `rows x 32`, row-major.
/// `values[i * 32 + j]` is bit-identical to `matrix[i][8j + 7]`.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckpointMatrix {
    pub rows: usize,
    pub values: Vec<f64>,
}

fn check_dims(plan: &SamplingPlan, img: &RgbImage<'_>) -> Result<()> {
    if img.width() != plan.width() || img.height() != plan.height() {
        return Err(CdchError::Dimensions {
            plan: (plan.width(), plan.height()),
            image: (img.width(), img.height()),
        });
    }
    Ok(())
}

impl SamplingPlan {
    /// Luma of every distinct sampled pixel, in `unique` order.
    fn fill_luma(&self, img: &RgbImage<'_>, ws: &mut Workspace) {
        let unique = self.unique_pixels();
        ws.lum.clear();
        ws.lum.reserve(unique.len());
        let data = img.data();
        if img.is_packed() {
            ws.lum.extend(unique.iter().map(|&p| {
                let i = 3 * p as usize;
                js::luma_fast(data[i], data[i + 1], data[i + 2])
            }));
        } else {
            let (w, stride) = (self.width(), img.row_stride());
            ws.lum.extend(unique.iter().map(|&p| {
                let p = p as usize;
                let i = (p / w) * stride + 3 * (p % w);
                js::luma_fast(data[i], data[i + 1], data[i + 2])
            }));
        }
    }

    /// Full canonical matrix into `out` (`n_rays * 256` values).
    pub fn matrix_into(
        &self,
        img: &RgbImage<'_>,
        ws: &mut Workspace,
        out: &mut [f64],
    ) -> Result<()> {
        check_dims(self, img)?;
        let rows = self.n_rays();
        if out.len() != rows * BINS {
            return Err(CdchError::BufferSize {
                expected: rows * BINS,
                actual: out.len(),
            });
        }
        self.fill_luma(img, ws);
        let lum = &ws.lum[..];
        for (ray, row) in out.chunks_exact_mut(BINS).enumerate() {
            let samples = self.ray_samples(ray);
            let mut hist = [0u32; BINS];
            for &k in samples {
                hist[lum[k as usize] as usize] += 1;
            }
            // Direct division vectorizes better here than the quotient-table lookup used
            // by the 32-checkpoint path (measured in RUST_PERFORMANCE_REPORT.md).
            let max = samples.len() as f64;
            let mut cum = 0u32;
            for (dst, &h) in row.iter_mut().zip(hist.iter()) {
                cum += h;
                *dst = cum as f64 / max;
            }
        }
        Ok(())
    }

    /// Full canonical matrix (allocating).
    pub fn matrix(&self, img: &RgbImage<'_>, ws: &mut Workspace) -> Result<CdchMatrix> {
        let mut data = vec![0.0; self.n_rays() * BINS];
        self.matrix_into(img, ws, &mut data)?;
        CdchMatrix::from_vec(self.n_rays(), data)
    }

    /// Raw integer histograms (`n_rays * 256` counts), for scientific inspection.
    pub fn histograms(&self, img: &RgbImage<'_>, ws: &mut Workspace) -> Result<Vec<u32>> {
        check_dims(self, img)?;
        self.fill_luma(img, ws);
        let mut out = vec![0u32; self.n_rays() * BINS];
        for (ray, hist) in out.chunks_exact_mut(BINS).enumerate() {
            for &k in self.ray_samples(ray) {
                hist[ws.lum[k as usize] as usize] += 1;
            }
        }
        Ok(out)
    }

    /// Integer histograms and the canonical matrix from one pass over the image.
    ///
    /// `counts` receives `n_rays * 256` raw counts and `matrix` the `n_rays * 256`
    /// canonical values. Each matrix value is produced exactly as in
    /// [`Self::matrix_into`]: an integer cumulative count divided once (IEEE binary64)
    /// by the ray length, so both outputs are bit-identical to the separate calls.
    pub fn histograms_and_matrix_into(
        &self,
        img: &RgbImage<'_>,
        ws: &mut Workspace,
        counts: &mut [u32],
        matrix: &mut [f64],
    ) -> Result<()> {
        check_dims(self, img)?;
        let n = self.n_rays() * BINS;
        for len in [counts.len(), matrix.len()] {
            if len != n {
                return Err(CdchError::BufferSize {
                    expected: n,
                    actual: len,
                });
            }
        }
        self.fill_luma(img, ws);
        let lum = &ws.lum[..];
        for (ray, (hist, row)) in counts
            .chunks_exact_mut(BINS)
            .zip(matrix.chunks_exact_mut(BINS))
            .enumerate()
        {
            hist.fill(0);
            let samples = self.ray_samples(ray);
            for &k in samples {
                hist[lum[k as usize] as usize] += 1;
            }
            let max = samples.len() as f64;
            let mut cum = 0u32;
            for (dst, &h) in row.iter_mut().zip(hist.iter()) {
                cum += h;
                *dst = cum as f64 / max;
            }
        }
        Ok(())
    }

    /// The canonical CDF at the 32 checkpoint bins only (`n_rays * 32` values),
    /// without materializing the 256-bin matrix. Each value is computed as
    /// `cumulative[8j+7] / cumulative[255]` with the same integer cumulative count
    /// and the same single division as the full path, so it is bit-identical.
    pub fn checkpoints_into(
        &self,
        img: &RgbImage<'_>,
        ws: &mut Workspace,
        out: &mut [f64],
    ) -> Result<()> {
        check_dims(self, img)?;
        let rows = self.n_rays();
        if out.len() != rows * CHECKPOINTS {
            return Err(CdchError::BufferSize {
                expected: rows * CHECKPOINTS,
                actual: out.len(),
            });
        }
        self.fill_luma(img, ws);
        // Bucket j collects intensities in (8j - 1, 8j + 7]; i.e. bucket = luma >> 3.
        for v in ws.lum.iter_mut() {
            *v >>= 3;
        }
        let bucket = &ws.lum[..];
        for (ray, row) in out.chunks_exact_mut(CHECKPOINTS).enumerate() {
            // Four interleaved count tables break the store-to-load dependency chain when
            // consecutive samples fall in the same bucket (common in smooth regions).
            let mut counts = [[0u32; CHECKPOINTS]; 4];
            let samples = self.ray_samples(ray);
            let mut quads = samples.chunks_exact(4);
            for q in &mut quads {
                counts[0][bucket[q[0] as usize] as usize] += 1;
                counts[1][bucket[q[1] as usize] as usize] += 1;
                counts[2][bucket[q[2] as usize] as usize] += 1;
                counts[3][bucket[q[3] as usize] as usize] += 1;
            }
            for &k in quads.remainder() {
                counts[0][bucket[k as usize] as usize] += 1;
            }
            let quotient = self.ray_quotients(ray);
            let mut cum = 0u32;
            for (j, dst) in row.iter_mut().enumerate() {
                cum += counts[0][j] + counts[1][j] + counts[2][j] + counts[3][j];
                *dst = quotient[cum as usize];
            }
        }
        Ok(())
    }

    pub fn checkpoints(&self, img: &RgbImage<'_>, ws: &mut Workspace) -> Result<CheckpointMatrix> {
        let mut values = vec![0.0; self.n_rays() * CHECKPOINTS];
        self.checkpoints_into(img, ws, &mut values)?;
        Ok(CheckpointMatrix {
            rows: self.n_rays(),
            values,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct PlanKey {
    width: usize,
    height: usize,
    cx: u64,
    cy: u64,
    step: u32,
}

/// Thread-safe cache of sampling plans keyed by geometry. Typical research batches
/// use a single image size, so the plan is built once and shared.
#[derive(Default, Debug)]
pub struct Engine {
    plans: RwLock<HashMap<PlanKey, Arc<SamplingPlan>>>,
}

impl Engine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn plan(&self, width: usize, height: usize, params: Params) -> Result<Arc<SamplingPlan>> {
        params.validate()?;
        let key = PlanKey {
            width,
            height,
            cx: params.center.x.to_bits(),
            cy: params.center.y.to_bits(),
            step: params.angle_step,
        };
        if let Some(p) = self.plans.read().expect("plan cache poisoned").get(&key) {
            return Ok(Arc::clone(p));
        }
        let plan = Arc::new(SamplingPlan::new(width, height, params)?);
        let mut guard = self.plans.write().expect("plan cache poisoned");
        Ok(Arc::clone(guard.entry(key).or_insert(plan)))
    }

    pub fn cached_plans(&self) -> usize {
        self.plans.read().expect("plan cache poisoned").len()
    }
}
