//! Image buffers accepted by Ferris.
//!
//! The canonical CDCH reads 8-bit **R, G, B** values (in that order) of a
//! row-major image; it never reads alpha. Ferris therefore computes on
//! [`RgbImage`] (borrowed, zero-copy, optionally strided) and offers explicit,
//! documented conversions from RGBA and greyscale buffers into
//! [`OwnedRgbImage`]. Decoding files (PNG, JPEG, ...) is deliberately left to the
//! caller so the core carries no codec dependencies.

use crate::error::{InvalidInput, Result};
pub use cdch_core::RgbImage;

/// Validate geometry shared by every buffer type.
fn check_geometry(
    width: usize,
    height: usize,
    channels: usize,
    row_stride: usize,
    len: usize,
) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(InvalidInput::EmptyImage { width, height }.into());
    }
    if width > u32::MAX as usize / 4
        || height > u32::MAX as usize / 4
        || width
            .checked_mul(height)
            .is_none_or(|n| n > u32::MAX as usize)
    {
        return Err(InvalidInput::TooLarge { width, height }.into());
    }
    let min_stride = channels
        .checked_mul(width)
        .ok_or(InvalidInput::TooLarge { width, height })?;
    if row_stride < min_stride {
        return Err(InvalidInput::RowStride {
            stride: row_stride,
            width,
            channels,
        }
        .into());
    }
    let expected = (height - 1)
        .checked_mul(row_stride)
        .and_then(|n| n.checked_add(min_stride))
        .ok_or(InvalidInput::TooLarge { width, height })?;
    if len < expected {
        return Err(InvalidInput::BufferSize {
            expected,
            actual: len,
        }
        .into());
    }
    Ok(())
}

/// Borrow a tightly packed RGB buffer, rejecting empty images.
///
/// Equivalent to [`RgbImage::new`] plus the public API's non-empty check.
pub fn rgb_view(width: usize, height: usize, data: &[u8]) -> Result<RgbImage<'_>> {
    check_geometry(width, height, 3, 3 * width, data.len())?;
    Ok(RgbImage::new(width, height, data)?)
}

/// An owned, tightly packed 8-bit RGB image (`height * width * 3` bytes, R G B order).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedRgbImage {
    width: usize,
    height: usize,
    data: Vec<u8>,
}

impl OwnedRgbImage {
    /// Take ownership of a packed RGB buffer. The length must be exactly
    /// `width * height * 3`.
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Result<Self> {
        check_geometry(width, height, 3, 3 * width, data.len())?;
        if data.len() != 3 * width * height {
            return Err(InvalidInput::BufferSize {
                expected: 3 * width * height,
                actual: data.len(),
            }
            .into());
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Replicate an 8-bit greyscale buffer into R = G = B (the same conversion as
    /// Pillow's `Image.convert("RGB")` for mode `L`).
    pub fn from_gray(width: usize, height: usize, gray: &[u8]) -> Result<Self> {
        check_geometry(width, height, 1, width, gray.len())?;
        let data = gray[..width * height]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect();
        Self::new(width, height, data)
    }

    /// Convert an RGBA image under an explicit [`AlphaPolicy`].
    pub fn from_rgba(img: &RgbaImage<'_>, policy: AlphaPolicy) -> Self {
        let mut data = Vec::with_capacity(img.width * img.height * 3);
        for y in 0..img.height {
            let row = &img.data[y * img.row_stride..y * img.row_stride + 4 * img.width];
            for p in row.chunks_exact(4) {
                data.extend_from_slice(&policy.apply([p[0], p[1], p[2], p[3]]));
            }
        }
        Self {
            width: img.width,
            height: img.height,
            data,
        }
    }

    /// Width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }
    /// Height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }
    /// The packed RGB bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }
    /// Consume the image, returning the packed RGB bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }

    /// Zero-copy view for computation.
    pub fn view(&self) -> RgbImage<'_> {
        // Geometry was validated on construction.
        RgbImage::new(self.width, self.height, &self.data)
            .expect("OwnedRgbImage invariants guarantee a valid view")
    }
}

/// A borrowed 8-bit RGBA image (R, G, B, A interleaved, row-major, straight
/// (non-premultiplied) alpha).
#[derive(Clone, Copy, Debug)]
pub struct RgbaImage<'a> {
    width: usize,
    height: usize,
    row_stride: usize,
    data: &'a [u8],
}

impl<'a> RgbaImage<'a> {
    /// A tightly packed `height * width * 4` buffer.
    pub fn new(width: usize, height: usize, data: &'a [u8]) -> Result<Self> {
        Self::with_stride(width, height, 4 * width, data)
    }

    /// A buffer whose rows start `row_stride` bytes apart.
    pub fn with_stride(
        width: usize,
        height: usize,
        row_stride: usize,
        data: &'a [u8],
    ) -> Result<Self> {
        check_geometry(width, height, 4, row_stride, data.len())?;
        Ok(Self {
            width,
            height,
            row_stride,
            data,
        })
    }

    /// Width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }
    /// Height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// True when every pixel has alpha 255. For such images every
    /// [`AlphaPolicy`] produces the same RGB values.
    pub fn is_opaque(&self) -> bool {
        (0..self.height).all(|y| {
            self.data[y * self.row_stride..y * self.row_stride + 4 * self.width]
                .chunks_exact(4)
                .all(|p| p[3] == 255)
        })
    }
}

/// How alpha is removed before CDCH, which only reads R, G and B.
///
/// The canonical algorithm is defined on opaque canvas pixels and has no alpha
/// semantics of its own, so this choice is part of the experiment and should be
/// recorded with results.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AlphaPolicy {
    /// Keep the stored R, G, B values and discard alpha (default). This equals
    /// Pillow's `Image.convert("RGB")`, which the archived research pipeline
    /// used to prepare its inputs.
    #[default]
    Ignore,
    /// Composite the straight-alpha colour over an opaque background:
    /// `out = round((c * a + bg * (255 - a)) / 255)`, evaluated exactly in integers
    /// as `(c * a + bg * (255 - a) + 127) / 255`. Because 255 is odd the exact
    /// quotient is never a tie, so this is round-to-nearest.
    CompositeOver {
        /// Opaque background colour `[R, G, B]`.
        background: [u8; 3],
    },
}

impl AlphaPolicy {
    /// Composite over white (`[255, 255, 255]`).
    pub const OVER_WHITE: AlphaPolicy = AlphaPolicy::CompositeOver {
        background: [255, 255, 255],
    };
    /// Composite over black (`[0, 0, 0]`).
    pub const OVER_BLACK: AlphaPolicy = AlphaPolicy::CompositeOver {
        background: [0, 0, 0],
    };

    /// Apply the policy to one straight-alpha RGBA pixel.
    #[inline]
    pub fn apply(self, [r, g, b, a]: [u8; 4]) -> [u8; 3] {
        match self {
            AlphaPolicy::Ignore => [r, g, b],
            AlphaPolicy::CompositeOver { background } => {
                let a = a as u32;
                let mix =
                    |c: u8, bg: u8| ((c as u32 * a + bg as u32 * (255 - a) + 127) / 255) as u8;
                [
                    mix(r, background[0]),
                    mix(g, background[1]),
                    mix(b, background[2]),
                ]
            }
        }
    }
}
