//! # Ferris — canonical CDCH for Rust
//!
//! `ferris-cdch` is the stable public API of Project Ferris, a validated Rust
//! implementation of Muzyka's canonical CDCH (cumulative directional colour
//! histogram) algorithm. The reference definition is the original JavaScript
//! `ip-1.js` (SHA-256 [`CANONICAL_SHA256`]); the implementation underneath
//! (`cdch-core`) reproduces it **bit for bit**, including JavaScript `Math.round`
//! semantics, the V8 trigonometric values, the radius limit equal to the image
//! width, and `NaN` rows for empty rays.
//!
//! ## Quick start
//!
//! ```
//! use ferris_cdch::{Analyzer, CdchConfig, CenterSpec, OwnedRgbImage};
//!
//! // A 64x48 image (R, G, B per pixel, row-major). Decoding files is up to you.
//! let (w, h) = (64, 48);
//! let pixels: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();
//! let image = OwnedRgbImage::new(w, h, pixels)?;
//!
//! let analyzer = Analyzer::new();
//! let config = CdchConfig::default().with_angle_step(10); // archived centre, 36 rays
//! let result = analyzer.compute(&image.view(), &config)?;
//!
//! assert_eq!(result.n_rays(), 36);
//! assert_eq!(result.params().center.x, 32.0); // w // 2
//! let cdf = result.cdf(0); // ray pointing up
//! assert_eq!(cdf[255], 1.0);
//! assert!(cdf.windows(2).all(|p| p[0] <= p[1]));
//! # Ok::<(), ferris_cdch::FerrisError>(())
//! ```
//!
//! ## Guarantees
//!
//! * **Exactness.** Matrices are bit-identical to `ip-1.js` on every tested input
//!   (see `docs/VALIDATION.md`). Nothing in the matrix path depends on the host
//!   math library.
//! * **Determinism.** Identical inputs and configuration give identical bits, on
//!   any thread, in any call order.
//! * **No panics on invalid input.** Invalid input returns
//!   [`FerrisError::InvalidInput`]; [`FerrisError::Internal`] signals a bug.
//! * **Thread safety.** [`Analyzer`] is `Send + Sync`; share one across threads.
//! * **No runtime dependencies** beyond `std` (the optional `representations`
//!   feature adds the validated A/B/C/D feature extractors).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod centroid;
mod config;
mod error;
mod image;
mod result;

use std::sync::Arc;

use cdch_core::{CdchMatrix, Engine, SamplingPlan, Workspace, BINS};

pub use centroid::{intensity_centroid, Centroid};
pub use config::{CdchConfig, CenterSpec};
pub use error::{FerrisError, InvalidInput, Result};
pub use image::{rgb_view, AlphaPolicy, OwnedRgbImage, RgbImage, RgbaImage};
pub use result::CdchResult;

pub use cdch_core::{Center, Params, BINS as INTENSITY_BINS, CANONICAL_SHA256};

/// Full access to the verified low-level implementation (`cdch-core`): the literal
/// reference port, sampling plans, checkpoint outputs and JavaScript-exact primitives.
/// Its API may change between minor versions of Ferris; prefer the top-level items.
pub use cdch_core as core;

/// The validated statistical representations A/B/C/D of the CDCH research study.
#[cfg(feature = "representations")]
pub use cdch_features as representations;

/// Version of this crate (`CARGO_PKG_VERSION`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Canonical luma of one pixel: `Math.round(R * 0.2125 + G * 0.7154 + B * 0.0721)`,
/// evaluated exactly as `ip-1.js` does (binary64, left to right, ties toward +∞).
#[inline]
pub fn luma(r: u8, g: u8, b: u8) -> u8 {
    cdch_core::js::luma_fast(r, g, b)
}

/// Canonical luma of every pixel, row-major (`width * height` values).
pub fn luma_plane(img: &RgbImage<'_>) -> Vec<u8> {
    let mut out = Vec::with_capacity(img.width() * img.height());
    for y in 0..img.height() {
        for x in 0..img.width() {
            let [r, g, b] = img.rgb(x, y);
            out.push(luma(r, g, b));
        }
    }
    out
}

/// Reusable, thread-safe CDCH evaluator.
///
/// Ray geometry depends only on image size, centre and angle step, so the
/// analyzer caches one sampling plan per geometry. Batches of equally sized
/// images therefore pay the geometry cost once. Clone it cheaply to share the
/// cache.
#[derive(Clone, Debug, Default)]
pub struct Analyzer {
    engine: Arc<Engine>,
}

impl Analyzer {
    /// An analyzer with an empty plan cache.
    pub fn new() -> Self {
        Self::default()
    }

    fn plan(&self, img: &RgbImage<'_>, config: &CdchConfig) -> Result<Arc<SamplingPlan>> {
        let (w, h) = (img.width(), img.height());
        if w == 0 || h == 0 {
            return Err(InvalidInput::EmptyImage {
                width: w,
                height: h,
            }
            .into());
        }
        let params = config.params(w, h)?;
        Ok(self.engine.plan(w, h, params)?)
    }

    /// Compute the canonical matrix together with ray lengths and integer counts.
    pub fn compute(&self, img: &RgbImage<'_>, config: &CdchConfig) -> Result<CdchResult> {
        let plan = self.plan(img, config)?;
        let rows = plan.n_rays();
        let mut counts = vec![0u32; rows * BINS];
        let mut matrix = vec![0f64; rows * BINS];
        plan.histograms_and_matrix_into(img, &mut Workspace::new(), &mut counts, &mut matrix)?;
        let ray_lengths = (0..rows).map(|i| plan.ray_len(i) as u32).collect();
        Ok(CdchResult {
            width: img.width(),
            height: img.height(),
            params: *plan.params(),
            ray_lengths,
            counts,
            matrix: CdchMatrix::from_vec(rows, matrix)?,
        })
    }

    /// Compute only the canonical matrix (the fastest full-matrix path).
    pub fn matrix(&self, img: &RgbImage<'_>, config: &CdchConfig) -> Result<CdchMatrix> {
        let plan = self.plan(img, config)?;
        Ok(plan.matrix(img, &mut Workspace::new())?)
    }

    /// Number of cached sampling plans.
    pub fn cached_plans(&self) -> usize {
        self.engine.cached_plans()
    }
}

/// One-off computation with a fresh [`Analyzer`].
pub fn compute(img: &RgbImage<'_>, config: &CdchConfig) -> Result<CdchResult> {
    Analyzer::new().compute(img, config)
}

/// Build and algorithm identification, for provenance records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuildInfo {
    /// Version of `ferris-cdch`.
    pub version: &'static str,
    /// SHA-256 of the canonical `ip-1.js` the implementation reproduces.
    pub canonical_sha256: &'static str,
    /// Name of the reproduced algorithm.
    pub algorithm: &'static str,
}

/// Identification of this build of Ferris.
pub fn build_info() -> BuildInfo {
    BuildInfo {
        version: VERSION,
        canonical_sha256: CANONICAL_SHA256,
        algorithm: "CDCH (Muzyka, ip-1.js drawRaysAndGenerateHistograms)",
    }
}
