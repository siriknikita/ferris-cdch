//! Error types of the public API.
//!
//! Every error is either an [`InvalidInput`] (the caller supplied something the
//! canonical algorithm cannot meaningfully evaluate) or an internal failure (a
//! broken invariant inside Ferris, which should never happen and is a bug).
//! Expected invalid input never panics.

use std::fmt;

use cdch_core::CdchError;

/// Why an input was rejected.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum InvalidInput {
    /// Width or height is zero. The canonical JavaScript would return rows of `NaN`;
    /// the public API rejects such images instead.
    EmptyImage {
        /// Declared width.
        width: usize,
        /// Declared height.
        height: usize,
    },
    /// The pixel buffer is shorter than the declared geometry requires.
    BufferSize {
        /// Minimum number of bytes required.
        expected: usize,
        /// Number of bytes supplied.
        actual: usize,
    },
    /// The row stride is smaller than `channels * width`.
    RowStride {
        /// Supplied stride in bytes.
        stride: usize,
        /// Image width in pixels.
        width: usize,
        /// Bytes per pixel of the buffer type.
        channels: usize,
    },
    /// The image exceeds the supported size (pixel indices must fit in `u32`).
    TooLarge {
        /// Declared width.
        width: usize,
        /// Declared height.
        height: usize,
    },
    /// The ray centre is `NaN` or infinite.
    NonFiniteCenter {
        /// Supplied x.
        x: f64,
        /// Supplied y.
        y: f64,
    },
    /// The angle step is outside `1..=360` degrees.
    AngleStep(u32),
    /// The intensity range of a centroid search has `min > max`.
    IntensityRange {
        /// Lower bound.
        min: u8,
        /// Upper bound.
        max: u8,
    },
}

impl fmt::Display for InvalidInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InvalidInput::EmptyImage { width, height } => {
                write!(f, "image has no pixels ({width}x{height})")
            }
            InvalidInput::BufferSize { expected, actual } => {
                write!(
                    f,
                    "pixel buffer has {actual} bytes, expected at least {expected}"
                )
            }
            InvalidInput::RowStride {
                stride,
                width,
                channels,
            } => write!(
                f,
                "row stride {stride} is smaller than {channels} * width ({})",
                channels * width
            ),
            InvalidInput::TooLarge { width, height } => {
                write!(f, "image {width}x{height} exceeds the supported size")
            }
            InvalidInput::NonFiniteCenter { x, y } => {
                write!(f, "ray centre ({x}, {y}) is not finite")
            }
            InvalidInput::AngleStep(s) => {
                write!(f, "angle step {s} is outside 1..=360 degrees")
            }
            InvalidInput::IntensityRange { min, max } => {
                write!(f, "intensity range is empty: min {min} > max {max}")
            }
        }
    }
}

/// Error returned by the public API.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum FerrisError {
    /// The caller's input was rejected; fix the input and retry.
    InvalidInput(InvalidInput),
    /// An internal invariant failed. This indicates a bug in Ferris.
    Internal(String),
}

impl FerrisError {
    /// True for errors caused by caller input.
    pub fn is_invalid_input(&self) -> bool {
        matches!(self, FerrisError::InvalidInput(_))
    }
}

impl fmt::Display for FerrisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FerrisError::InvalidInput(e) => write!(f, "invalid input: {e}"),
            FerrisError::Internal(e) => write!(f, "internal error in Ferris: {e}"),
        }
    }
}

impl std::error::Error for FerrisError {}

impl From<InvalidInput> for FerrisError {
    fn from(e: InvalidInput) -> Self {
        FerrisError::InvalidInput(e)
    }
}

impl From<CdchError> for FerrisError {
    fn from(e: CdchError) -> Self {
        match e {
            CdchError::BufferSize { expected, actual } => {
                InvalidInput::BufferSize { expected, actual }.into()
            }
            CdchError::RowStride { stride, width } => InvalidInput::RowStride {
                stride,
                width,
                channels: 3,
            }
            .into(),
            CdchError::TooLarge { width, height } => {
                InvalidInput::TooLarge { width, height }.into()
            }
            CdchError::NonFiniteCenter { x, y } => InvalidInput::NonFiniteCenter { x, y }.into(),
            CdchError::AngleStep(s) => InvalidInput::AngleStep(s).into(),
            // The facade always builds plans for the image it evaluates and sizes every
            // buffer itself, so the remaining variants indicate a Ferris bug.
            other => FerrisError::Internal(other.to_string()),
        }
    }
}

/// Result alias for the public API.
pub type Result<T> = std::result::Result<T, FerrisError>;
