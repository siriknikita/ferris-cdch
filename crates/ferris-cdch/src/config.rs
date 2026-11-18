//! Strongly typed CDCH configuration.

use cdch_core::{Center, Params};

use crate::error::{InvalidInput, Result};

/// Where the rays start.
///
/// Coordinates are canvas pixel coordinates: `x` to the right, `y` downward,
/// `(0, 0)` at the centre of the top-left pixel. Any finite value is accepted.
/// The first sample of every ray is the rounded centre itself, so if it lies
/// outside the image every ray is empty and every row is `NaN`, exactly as in the
/// canonical JavaScript (see [`crate::CdchResult::is_complete`]).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum CenterSpec {
    /// `(width / 2, height / 2)` rounded down to integers (Python `//`), as sent by
    /// the archived research pipeline. **Default.**
    #[default]
    Archived,
    /// `(width / 2, height / 2)` as floating-point values: the oracle's default
    /// when no centre is given and the "image centre" button of `ip-1.js`.
    Midpoint,
    /// An explicit centre.
    Explicit {
        /// Horizontal coordinate (pixels, to the right).
        x: f64,
        /// Vertical coordinate (pixels, downward).
        y: f64,
    },
}

impl CenterSpec {
    /// The concrete centre for an image of the given size.
    pub fn resolve(&self, width: usize, height: usize) -> Center {
        match *self {
            CenterSpec::Archived => Center::archived(width, height),
            CenterSpec::Midpoint => Center::midpoint(width, height),
            CenterSpec::Explicit { x, y } => Center::new(x, y),
        }
    }
}

/// Parameters of a CDCH computation, independent of any particular image.
///
/// The default is the archived research protocol: [`CenterSpec::Archived`] and a
/// 1° angle step (360 rays).
///
/// ```
/// use ferris_cdch::{CdchConfig, CenterSpec};
/// let cfg = CdchConfig::default()
///     .with_center(CenterSpec::Explicit { x: 10.0, y: 12.5 })
///     .with_angle_step(5);
/// assert_eq!(cfg.n_rays().unwrap(), 72);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CdchConfig {
    /// Ray origin.
    pub center: CenterSpec,
    /// Angle increment in whole degrees, `1..=360`. Rays are emitted at
    /// `0, step, 2*step, ... < 360` degrees, 0° pointing up, clockwise on screen.
    pub angle_step: u32,
}

impl Default for CdchConfig {
    fn default() -> Self {
        Self::archived()
    }
}

impl CdchConfig {
    /// The archived protocol: centre `(w // 2, h // 2)`, step 1° (360 rays).
    pub const fn archived() -> Self {
        Self {
            center: CenterSpec::Archived,
            angle_step: 1,
        }
    }

    /// Replace the centre specification.
    pub fn with_center(mut self, center: CenterSpec) -> Self {
        self.center = center;
        self
    }

    /// Replace the angle step (validated when the configuration is used).
    pub fn with_angle_step(mut self, angle_step: u32) -> Self {
        self.angle_step = angle_step;
        self
    }

    /// Check the image-independent parts of the configuration.
    pub fn validate(&self) -> Result<()> {
        if !(1..=360).contains(&self.angle_step) {
            return Err(InvalidInput::AngleStep(self.angle_step).into());
        }
        if let CenterSpec::Explicit { x, y } = self.center {
            if !x.is_finite() || !y.is_finite() {
                return Err(InvalidInput::NonFiniteCenter { x, y }.into());
            }
        }
        Ok(())
    }

    /// Number of rays, `ceil(360 / angle_step)`.
    pub fn n_rays(&self) -> Result<usize> {
        self.validate()?;
        Ok(360_usize.div_ceil(self.angle_step as usize))
    }

    /// Resolve into the canonical core parameters for an image size.
    pub fn params(&self, width: usize, height: usize) -> Result<Params> {
        self.validate()?;
        let params = Params {
            center: self.center.resolve(width, height),
            angle_step: self.angle_step,
        };
        params.validate()?;
        Ok(params)
    }
}
