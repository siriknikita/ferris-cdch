# ferris-cdch

The public Rust API of Project Ferris: canonical CDCH computation that reproduces
Muzyka's original JavaScript (`ip-1.js`) bit for bit.

```rust
use ferris_cdch::{Analyzer, CdchConfig, OwnedRgbImage};

let image = OwnedRgbImage::new(width, height, rgb_bytes)?;
let result = Analyzer::new().compute(&image.view(), &CdchConfig::default())?;
let cdf_of_first_ray: &[f64] = result.cdf(0);
```

See the repository's `docs/API.md` for the full guide and `docs/MATHEMATICAL_COMPATIBILITY.md`
for the exact semantics.
