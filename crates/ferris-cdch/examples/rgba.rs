//! Explicit alpha handling before CDCH.
//!
//! ```text
//! cargo run -p ferris-cdch --example rgba
//! ```

use ferris_cdch::{compute, AlphaPolicy, CdchConfig, OwnedRgbImage, RgbaImage};

fn main() -> Result<(), ferris_cdch::FerrisError> {
    // 32x32 red square whose right half is 25 % opaque.
    let (w, h) = (32usize, 32usize);
    let mut rgba = Vec::with_capacity(w * h * 4);
    for _y in 0..h {
        for x in 0..w {
            let a = if x < w / 2 { 255 } else { 64 };
            rgba.extend_from_slice(&[200, 30, 30, a]);
        }
    }
    let src = RgbaImage::new(w, h, &rgba)?;
    let config = CdchConfig::default().with_angle_step(90);

    for (name, policy) in [
        ("ignore alpha", AlphaPolicy::Ignore),
        ("over white", AlphaPolicy::OVER_WHITE),
        ("over black", AlphaPolicy::OVER_BLACK),
    ] {
        let rgb = OwnedRgbImage::from_rgba(&src, policy);
        let result = compute(&rgb.view(), &config)?;
        // Ray 1 points right (90°), into the translucent half.
        let first_bin = result.cdf(1).iter().position(|&f| f > 0.0).unwrap();
        println!("{name:<13} lowest luma on the 90° ray: {first_bin}");
    }
    Ok(())
}
