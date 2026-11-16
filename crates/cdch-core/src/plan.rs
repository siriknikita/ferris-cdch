//! Precomputed sampling geometry.
//!
//! The rounded ray coordinates in `ip-1.js` depend only on the canvas size, the
//! centre and the angle step, never on pixel values. A [`SamplingPlan`] evaluates
//! them once, with the same JavaScript-exact arithmetic as [`crate::reference`],
//! and stores:
//!
//! * `unique`: the distinct pixels any ray touches, as ascending linear indices
//!   `py * width + px` (ascending order streams through image memory);
//! * `samples`: for every sample in emission order, its index into `unique`
//!   (duplicate coordinates produced by rounding appear as repeated indices and
//!   keep their canonical multiplicity);
//! * `ray_starts`: CSR offsets of each ray within `samples`.

use crate::{js, CdchError, Params, Result};

#[derive(Clone, Debug)]
pub struct SamplingPlan {
    width: usize,
    height: usize,
    params: Params,
    ray_starts: Vec<u32>,
    samples: Vec<u32>,
    unique: Vec<u32>,
}

impl SamplingPlan {
    pub fn new(width: usize, height: usize, params: Params) -> Result<Self> {
        params.validate()?;
        if width.checked_mul(height).map_or(true, |n| n > u32::MAX as usize) {
            return Err(CdchError::TooLarge { width, height });
        }
        let (x, y) = (params.center.x, params.center.y);
        let (wf, hf) = (width as f64, height as f64);
        let n_rays = params.n_rays();
        let mut ray_starts = Vec::with_capacity(n_rays + 1);
        let mut linear: Vec<u32> = Vec::new();
        ray_starts.push(0u32);
        for angle in params.angles() {
            let (cos, sin) = js::direction(angle);
            let mut r: u32 = 0;
            while (r as f64) < wf {
                let (px, py) = js::ray_point(x, y, r, cos, sin);
                if px < 0.0 || px >= wf || py < 0.0 || py >= hf {
                    break;
                }
                linear.push(py as u32 * width as u32 + px as u32);
                r += 1;
            }
            if linear.len() > u32::MAX as usize {
                return Err(CdchError::TooLarge { width, height });
            }
            ray_starts.push(linear.len() as u32);
        }
        let mut unique = linear.clone();
        unique.sort_unstable();
        unique.dedup();
        let samples = linear
            .iter()
            .map(|p| unique.binary_search(p).expect("pixel present") as u32)
            .collect();
        Ok(Self { width, height, params, ray_starts, samples, unique })
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn params(&self) -> &Params {
        &self.params
    }
    pub fn n_rays(&self) -> usize {
        self.ray_starts.len() - 1
    }
    /// Total samples over all rays (with multiplicity).
    pub fn total_samples(&self) -> usize {
        self.samples.len()
    }
    /// Distinct pixels touched by any ray.
    pub fn unique_pixels(&self) -> &[u32] {
        &self.unique
    }
    /// Sample indices (into [`Self::unique_pixels`]) of ray `i`, in emission order.
    #[inline]
    pub fn ray_samples(&self, i: usize) -> &[u32] {
        &self.samples[self.ray_starts[i] as usize..self.ray_starts[i + 1] as usize]
    }
    /// Number of samples on ray `i` (the JavaScript `cumulative[255]`).
    #[inline]
    pub fn ray_len(&self, i: usize) -> usize {
        (self.ray_starts[i + 1] - self.ray_starts[i]) as usize
    }
    /// Coordinates `(px, py)` of ray `i`, in emission order.
    pub fn ray_coords(&self, i: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        let w = self.width;
        self.ray_samples(i).iter().map(move |&k| {
            let p = self.unique[k as usize] as usize;
            (p % w, p / w)
        })
    }
    /// True when every ray has at least one sample, i.e. no row of the canonical
    /// matrix is `0 / 0 = NaN`. Depends on geometry only.
    pub fn all_rays_nonempty(&self) -> bool {
        (0..self.n_rays()).all(|i| self.ray_len(i) > 0)
    }
    /// Heap bytes held by the plan.
    pub fn heap_bytes(&self) -> usize {
        4 * (self.ray_starts.capacity() + self.samples.capacity() + self.unique.capacity())
    }
}
