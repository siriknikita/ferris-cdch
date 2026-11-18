//! # cdch-core
//!
//! A validated Rust implementation of the canonical CDCH computation defined by
//! Muzyka's original JavaScript (`ip-1.js`, SHA-256
//! `110732b48bc6e8b2d5167eec30f7277245913bed98b9baaa3c65c0c22aab44d8`).
//!
//! The JavaScript remains the authoritative definition. This crate provides two
//! implementations that must agree with it bit for bit:
//!
//! * [`mod@reference`] — a line-by-line port of `drawRaysAndGenerateHistograms` and
//!   `calculateCumulativeHistogram`, kept deliberately simple.
//! * [`engine`] — the optimized path: a precomputed [`SamplingPlan`] (pixel
//!   indices per ray, which depend only on geometry) applied to each image, with
//!   reusable buffers and an optional reduced "checkpoint" output used by fused
//!   feature extraction.
//!
//! See `RUST_ARCHITECTURE.md` and `RUST_PORT_VALIDATION.md` at the workspace root.

#![forbid(unsafe_code)]

mod directions_v8;
pub mod engine;
pub mod js;
pub mod plan;
pub mod reference;

pub use engine::{CheckpointMatrix, Engine, Workspace};
pub use plan::SamplingPlan;

use std::fmt;

/// SHA-256 of the canonical `ip-1.js` this crate reproduces.
pub const CANONICAL_SHA256: &str =
    "110732b48bc6e8b2d5167eec30f7277245913bed98b9baaa3c65c0c22aab44d8";

/// Number of intensity bins (`new Array(256)` in `ip-1.js`).
pub const BINS: usize = 256;

/// Number of rays under the archived protocol (angle step 1°).
pub const RAYS_ARCHIVED: usize = 360;

/// Errors reported for invalid input. The canonical algorithm itself never fails;
/// these guard against inputs the JavaScript would mis-handle silently.
#[derive(Debug, Clone, PartialEq)]
pub enum CdchError {
    /// The pixel buffer is too short for the declared geometry.
    BufferSize { expected: usize, actual: usize },
    /// The row stride is smaller than `3 * width`.
    RowStride { stride: usize, width: usize },
    /// Width or height exceed the supported range (u32 pixel indices).
    TooLarge { width: usize, height: usize },
    /// The ray centre is NaN or infinite.
    NonFiniteCenter { x: f64, y: f64 },
    /// `ip-1.js` accepts integer steps in 1..=360 (its UI validates `step > 0 && step <= 360`).
    AngleStep(u32),
    /// A derived representation requires a different number of rays.
    RayCount { expected: usize, actual: usize },
    /// The canonical matrix failed the archived validity checks (empty ray, NaN, ...).
    InvalidMatrix(String),
    /// An image was evaluated with a sampling plan built for another size.
    Dimensions {
        plan: (usize, usize),
        image: (usize, usize),
    },
}

impl fmt::Display for CdchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CdchError::BufferSize { expected, actual } => {
                write!(
                    f,
                    "pixel buffer has {actual} bytes, expected at least {expected}"
                )
            }
            CdchError::RowStride { stride, width } => {
                write!(
                    f,
                    "row stride {stride} is smaller than 3 * width ({})",
                    3 * width
                )
            }
            CdchError::TooLarge { width, height } => {
                write!(f, "image {width}x{height} exceeds the supported size")
            }
            CdchError::NonFiniteCenter { x, y } => write!(f, "ray centre ({x}, {y}) is not finite"),
            CdchError::AngleStep(s) => write!(f, "angle step {s} outside 1..=360"),
            CdchError::RayCount { expected, actual } => {
                write!(f, "representation requires {expected} rays, got {actual}")
            }
            CdchError::InvalidMatrix(why) => write!(f, "invalid canonical CDCH matrix: {why}"),
            CdchError::Dimensions { plan, image } => write!(
                f,
                "image is {}x{} but the sampling plan was built for {}x{}",
                image.0, image.1, plan.0, plan.1
            ),
        }
    }
}

impl std::error::Error for CdchError {}

pub type Result<T> = std::result::Result<T, CdchError>;

/// A borrowed, decoded 8-bit RGB image in row-major order (R, G, B interleaved).
///
/// `row_stride` is the distance in bytes between rows. It equals `3 * width` for
/// tightly packed buffers and may be larger for cropped views.
#[derive(Clone, Copy, Debug)]
pub struct RgbImage<'a> {
    width: usize,
    height: usize,
    row_stride: usize,
    data: &'a [u8],
}

impl<'a> RgbImage<'a> {
    /// A tightly packed `height x width x 3` buffer.
    pub fn new(width: usize, height: usize, data: &'a [u8]) -> Result<Self> {
        Self::with_stride(width, height, 3 * width, data)
    }

    pub fn with_stride(
        width: usize,
        height: usize,
        row_stride: usize,
        data: &'a [u8],
    ) -> Result<Self> {
        if width > u32::MAX as usize / 4
            || height > u32::MAX as usize / 4
            || width
                .checked_mul(height)
                .is_none_or(|n| n > u32::MAX as usize)
        {
            return Err(CdchError::TooLarge { width, height });
        }
        if row_stride < 3 * width {
            return Err(CdchError::RowStride {
                stride: row_stride,
                width,
            });
        }
        let expected = if height == 0 {
            0
        } else {
            (height - 1) * row_stride + 3 * width
        };
        if data.len() < expected {
            return Err(CdchError::BufferSize {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            width,
            height,
            row_stride,
            data,
        })
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }
    #[inline]
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }
    #[inline]
    pub fn data(&self) -> &'a [u8] {
        self.data
    }
    #[inline]
    pub fn is_packed(&self) -> bool {
        self.row_stride == 3 * self.width
    }

    /// `(R, G, B)` at column `x`, row `y`. Mirrors the oracle's `getImageData(x, y)`.
    #[inline]
    pub fn rgb(&self, x: usize, y: usize) -> [u8; 3] {
        let i = y * self.row_stride + 3 * x;
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }
}

/// Ray origin in canvas pixel coordinates (x to the right, y downward).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Center {
    pub x: f64,
    pub y: f64,
}

impl Center {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Archived research protocol: `[width // 2, height // 2]` (Python integer division),
    /// as sent by `native_features.matrix` and `cdch_features.generate`.
    pub fn archived(width: usize, height: usize) -> Self {
        Self {
            x: (width / 2) as f64,
            y: (height / 2) as f64,
        }
    }

    /// The oracle's fallback when no centre is supplied and `ip-1.js`'s "centre"
    /// button: `width / 2`, `height / 2` as floating-point values.
    pub fn midpoint(width: usize, height: usize) -> Self {
        Self {
            x: width as f64 / 2.0,
            y: height as f64 / 2.0,
        }
    }
}

/// Parameters of one CDCH evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    pub center: Center,
    /// Integer angle increment in degrees, 1..=360. The archived protocol uses 1.
    pub angle_step: u32,
}

impl Params {
    /// The archived protocol: centre `[w // 2, h // 2]`, 360 rays.
    pub fn archived(width: usize, height: usize) -> Self {
        Self {
            center: Center::archived(width, height),
            angle_step: 1,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if !self.center.x.is_finite() || !self.center.y.is_finite() {
            return Err(CdchError::NonFiniteCenter {
                x: self.center.x,
                y: self.center.y,
            });
        }
        if !(1..=360).contains(&self.angle_step) {
            return Err(CdchError::AngleStep(self.angle_step));
        }
        Ok(())
    }

    /// Number of rays: angles `0, step, 2*step, ... < 360`.
    pub fn n_rays(&self) -> usize {
        360_usize.div_ceil(self.angle_step as usize)
    }

    /// The angles (degrees) in emission order.
    pub fn angles(&self) -> impl Iterator<Item = u32> {
        (0..360).step_by(self.angle_step as usize)
    }
}

/// A canonical CDCH matrix: `rows x 256` float64 values in row-major order. Row `i`
/// is the normalized cumulative histogram of ray `i` (angle `i * step` degrees).
#[derive(Clone, Debug, PartialEq)]
pub struct CdchMatrix {
    rows: usize,
    data: Vec<f64>,
}

impl CdchMatrix {
    pub fn from_vec(rows: usize, data: Vec<f64>) -> Result<Self> {
        if data.len() != rows * BINS {
            return Err(CdchError::BufferSize {
                expected: rows * BINS,
                actual: data.len(),
            });
        }
        Ok(Self { rows, data })
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
    pub fn into_vec(self) -> Vec<f64> {
        self.data
    }
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * BINS..(i + 1) * BINS]
    }
    pub fn view(&self) -> MatrixView<'_> {
        MatrixView {
            rows: self.rows,
            data: &self.data,
        }
    }
}

/// Borrowed view of a `rows x 256` canonical matrix (e.g. from NumPy).
#[derive(Clone, Copy, Debug)]
pub struct MatrixView<'a> {
    rows: usize,
    data: &'a [f64],
}

impl<'a> MatrixView<'a> {
    pub fn new(rows: usize, data: &'a [f64]) -> Result<Self> {
        if data.len() != rows * BINS {
            return Err(CdchError::BufferSize {
                expected: rows * BINS,
                actual: data.len(),
            });
        }
        Ok(Self { rows, data })
    }
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }
    #[inline]
    pub fn as_slice(&self) -> &'a [f64] {
        self.data
    }
    #[inline]
    pub fn row(&self, i: usize) -> &'a [f64] {
        &self.data[i * BINS..(i + 1) * BINS]
    }
    #[inline]
    pub fn at(&self, i: usize, b: usize) -> f64 {
        self.data[i * BINS + b]
    }
}
