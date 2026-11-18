//! Port of `findIntensityCenter` from `ip-1.js` ("centre by intensity").

use cdch_core::js;

use crate::error::{InvalidInput, Result};
use crate::RgbImage;

/// Result of an intensity-centroid search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Centroid {
    /// `Math.round(sumX / count)`.
    pub x: f64,
    /// `Math.round(sumY / count)`.
    pub y: f64,
    /// Number of pixels whose canonical luma lies in `[min, max]`.
    pub count: u64,
}

/// Centroid of the pixels whose canonical luma lies in `[min, max]` (inclusive),
/// rounded with JavaScript `Math.round`, exactly as `findIntensityCenter` in
/// `ip-1.js`. Returns `Ok(None)` when no pixel qualifies (the JavaScript shows an
/// alert in that case).
///
/// The sums are accumulated in binary64 in row-major order like the JavaScript,
/// so results agree even for images large enough that the sums exceed 2^53.
pub fn intensity_centroid(img: &RgbImage<'_>, min: u8, max: u8) -> Result<Option<Centroid>> {
    if min > max {
        return Err(InvalidInput::IntensityRange { min, max }.into());
    }
    let (mut sum_x, mut sum_y, mut count) = (0f64, 0f64, 0f64);
    for y in 0..img.height() {
        for x in 0..img.width() {
            let [r, g, b] = img.rgb(x, y);
            let intensity = js::luma_fast(r, g, b);
            if intensity >= min && intensity <= max {
                sum_x += x as f64;
                sum_y += y as f64;
                count += 1.0;
            }
        }
    }
    if count > 0.0 {
        Ok(Some(Centroid {
            x: js::round(sum_x / count),
            y: js::round(sum_y / count),
            count: count as u64,
        }))
    } else {
        Ok(None)
    }
}
