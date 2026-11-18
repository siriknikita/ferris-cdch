//! Independent consumer of the public Ferris API. Exits non-zero if any check fails.

use ferris_cdch::{
    intensity_centroid, AlphaPolicy, Analyzer, CdchConfig, CenterSpec, FerrisError,
    OwnedRgbImage, RgbaImage,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let info = ferris_cdch::build_info();
    println!("ferris-cdch {} (canonical ip-1.js {})", info.version, info.canonical_sha256);

    // A synthetic RGBA test card: horizontal gradient, translucent lower half.
    let (w, h) = (120usize, 80usize);
    let mut rgba = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            let v = (x * 255 / (w - 1)) as u8;
            rgba.extend_from_slice(&[v, 255 - v, 128, if y < h / 2 { 255 } else { 128 }]);
        }
    }
    let src = RgbaImage::new(w, h, &rgba)?;
    let rgb = OwnedRgbImage::from_rgba(&src, AlphaPolicy::OVER_WHITE);

    let analyzer = Analyzer::new();
    let config = CdchConfig::default().with_angle_step(15);
    let result = analyzer.compute(&rgb.view(), &config)?;
    assert_eq!(result.n_rays(), 24);
    assert!(result.is_complete());
    for i in 0..result.n_rays() {
        let cdf = result.cdf(i);
        assert_eq!(cdf[255], 1.0, "every CDF ends at 1");
        assert!(cdf.windows(2).all(|p| p[0] <= p[1]), "CDFs are monotone");
        assert_eq!(result.counts(i).iter().sum::<u32>(), result.ray_length(i));
    }
    // Deterministic: a second analyzer gives the same bits.
    assert_eq!(Analyzer::new().compute(&rgb.view(), &config)?, result);

    let c = intensity_centroid(&rgb.view(), 0, 255)?.expect("all pixels qualify");
    println!("centroid of all pixels: ({}, {}) over {} pixels", c.x, c.y, c.count);

    // Invalid input is a typed error, not a panic.
    match analyzer.compute(&rgb.view(), &config.with_center(CenterSpec::Explicit { x: f64::NAN, y: 0.0 })) {
        Err(FerrisError::InvalidInput(e)) => println!("rejected as expected: {e}"),
        other => panic!("expected InvalidInput, got {other:?}"),
    }

    let mut csv = Vec::new();
    result.write_matrix_csv(&mut csv)?;
    println!("CSV export: {} bytes, {} rows", csv.len(), result.n_rays() + 1);
    println!("INDEPENDENT_CONSUMER_OK");
    Ok(())
}
