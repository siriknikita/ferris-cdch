//! Library side of the `cdch` CLI: PNG decoding and exported linear-model inference.

#![forbid(unsafe_code)]

use std::io::BufReader;
use std::path::Path;

use serde_json::Value;

/// A decoded, owned, tightly packed 8-bit RGB image.
#[derive(Clone, Debug)]
pub struct OwnedRgb {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl OwnedRgb {
    pub fn view(&self) -> cdch_core::RgbImage<'_> {
        cdch_core::RgbImage::new(self.width, self.height, &self.data).expect("valid owned image")
    }

    /// Copy of the rectangle `[x0, x0+w) x [y0, y0+h)`.
    pub fn crop(&self, x0: usize, y0: usize, w: usize, h: usize) -> OwnedRgb {
        let mut data = Vec::with_capacity(w * h * 3);
        for y in y0..y0 + h {
            let s = (y * self.width + x0) * 3;
            data.extend_from_slice(&self.data[s..s + 3 * w]);
        }
        OwnedRgb { width: w, height: h, data }
    }
}

/// Decode an 8-bit PNG to packed RGB (alpha dropped, grey replicated), matching Pillow's
/// `Image.convert("RGB")` for these colour types. PNG is lossless, so the pixels equal Pillow's.
pub fn read_png(path: &Path) -> Result<OwnedRgb, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut decoder = png::Decoder::new(BufReader::new(file));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().map_err(|e| format!("{}: {e}", path.display()))?;
    let size = reader.output_buffer_size().ok_or("png too large")?;
    let mut buf = vec![0u8; size];
    let info = reader.next_frame(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
    buf.truncate(info.buffer_size());
    let (w, h) = (info.width as usize, info.height as usize);
    let data = match info.color_type {
        png::ColorType::Rgb => buf,
        png::ColorType::Rgba => buf.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect(),
        png::ColorType::Grayscale => buf.iter().flat_map(|&g| [g, g, g]).collect(),
        png::ColorType::GrayscaleAlpha => buf.chunks_exact(2).flat_map(|p| [p[0], p[0], p[0]]).collect(),
        other => return Err(format!("{}: unsupported colour type {other:?}", path.display())),
    };
    if data.len() != w * h * 3 {
        return Err(format!("{}: unexpected decoded size", path.display()));
    }
    Ok(OwnedRgb { width: w, height: h, data })
}

/// An archived StandardScaler + linear classifier, exported by `python/export_linear_models.py`.
#[derive(Clone, Debug)]
pub struct LinearModel {
    pub name: String,
    pub feature: cdch_features::Representation,
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
    pub coef: Vec<f64>,
    pub intercept: f64,
    pub threshold: f64,
}

impl LinearModel {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let vec = |k: &str| -> Result<Vec<f64>, String> {
            v[k].as_array()
                .ok_or(format!("missing {k}"))?
                .iter()
                .map(|x| x.as_f64().ok_or(format!("bad number in {k}")))
                .collect()
        };
        let feature = cdch_features::Representation::parse(v["feature"].as_str().unwrap_or(""))
            .ok_or("unknown feature key")?;
        let m = LinearModel {
            name: v["name"].as_str().unwrap_or("").to_string(),
            feature,
            mean: vec("scaler_mean")?,
            scale: vec("scaler_scale")?,
            coef: vec("coef")?,
            intercept: v["intercept"].as_f64().ok_or("missing intercept")?,
            threshold: v["threshold"].as_f64().ok_or("missing threshold")?,
        };
        let d = feature.dim();
        if m.mean.len() != d || m.scale.len() != d || m.coef.len() != d {
            return Err(format!("{}: dimensions do not match representation {feature:?}", path.display()));
        }
        Ok(m)
    }

    /// sklearn `decision_function`: `((x - mean) / scale) @ coef + intercept`. The dot
    /// product order differs from BLAS, so agreement is to rounding error, not bitwise.
    pub fn decision(&self, x: &[f64]) -> f64 {
        let mut acc = 0.0;
        for i in 0..x.len() {
            acc += (x[i] - self.mean[i]) / self.scale[i] * self.coef[i];
        }
        acc + self.intercept
    }
}
