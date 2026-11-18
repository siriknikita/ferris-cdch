//! Compute a canonical CDCH for a synthetic image and print a summary.
//!
//! ```text
//! cargo run -p ferris-cdch --example basic
//! ```

use ferris_cdch::{Analyzer, CdchConfig, CenterSpec, OwnedRgbImage};

fn main() -> Result<(), ferris_cdch::FerrisError> {
    // A 128x96 radial gradient: dark centre, bright edges.
    let (w, h) = (128usize, 96usize);
    let mut pixels = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let d = ((x as f64 - 64.0).powi(2) + (y as f64 - 48.0).powi(2)).sqrt();
            let v = (d * 3.0).min(255.0) as u8;
            pixels.extend_from_slice(&[v, v / 2, 255 - v]);
        }
    }
    let image = OwnedRgbImage::new(w, h, pixels)?;

    let analyzer = Analyzer::new();
    let config = CdchConfig::default()
        .with_center(CenterSpec::Explicit { x: 64.0, y: 48.0 })
        .with_angle_step(45);
    let result = analyzer.compute(&image.view(), &config)?;

    println!(
        "ferris-cdch {} | {}x{} | centre ({}, {}) | {} rays",
        ferris_cdch::VERSION,
        result.width(),
        result.height(),
        result.params().center.x,
        result.params().center.y,
        result.n_rays()
    );
    for i in 0..result.n_rays() {
        let cdf = result.cdf(i);
        // The intensity below which half of the ray's samples lie.
        let median = cdf.iter().position(|&f| f >= 0.5).unwrap_or(255);
        println!(
            "ray {:>3}°  samples {:>3}  median luma {:>3}  F(127) = {:.4}",
            result.angle(i),
            result.ray_length(i),
            median,
            cdf[127]
        );
    }
    Ok(())
}
