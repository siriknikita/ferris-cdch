//! Structured CDCH results.

use std::io::{self, Write};

use cdch_core::{CdchMatrix, Params, BINS};

/// The complete output of one CDCH computation.
///
/// Row `i` corresponds to the ray at [`Self::angle`]`(i)` degrees. For each ray the
/// result holds the integer histogram of canonical luma values (`counts`) and the
/// canonical normalized cumulative histogram (`cdf`), where
/// `cdf[b] = (counts[0] + ... + counts[b]) / ray_length`, computed with one IEEE
/// binary64 division per value exactly as `ip-1.js` does.
///
/// An empty ray (`ray_length == 0`) has `NaN` in all 256 CDF bins, as in the
/// canonical JavaScript. This happens exactly when the rounded centre lies outside
/// the image, and then it happens to every ray.
#[derive(Clone, Debug, PartialEq)]
pub struct CdchResult {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) params: Params,
    pub(crate) ray_lengths: Vec<u32>,
    pub(crate) counts: Vec<u32>,
    pub(crate) matrix: CdchMatrix,
}

impl CdchResult {
    /// Image width the result was computed on.
    pub fn width(&self) -> usize {
        self.width
    }
    /// Image height the result was computed on.
    pub fn height(&self) -> usize {
        self.height
    }
    /// The resolved parameters (concrete centre and angle step).
    pub fn params(&self) -> &Params {
        &self.params
    }
    /// Number of rays (matrix rows).
    pub fn n_rays(&self) -> usize {
        self.ray_lengths.len()
    }
    /// Angle of ray `i` in degrees (`i * angle_step`).
    pub fn angle(&self, i: usize) -> u32 {
        i as u32 * self.params.angle_step
    }
    /// Number of samples on ray `i` (with multiplicity), the canonical `cumulative[255]`.
    pub fn ray_length(&self, i: usize) -> u32 {
        self.ray_lengths[i]
    }
    /// All ray lengths in ray order.
    pub fn ray_lengths(&self) -> &[u32] {
        &self.ray_lengths
    }
    /// Total samples over all rays.
    pub fn total_samples(&self) -> u64 {
        self.ray_lengths.iter().map(|&n| n as u64).sum()
    }
    /// Canonical normalized cumulative histogram of ray `i` (256 values).
    pub fn cdf(&self, i: usize) -> &[f64] {
        self.matrix.row(i)
    }
    /// Raw integer histogram of ray `i` (256 counts).
    pub fn counts(&self, i: usize) -> &[u32] {
        &self.counts[i * BINS..(i + 1) * BINS]
    }
    /// All counts, row-major `n_rays x 256`.
    pub fn counts_flat(&self) -> &[u32] {
        &self.counts
    }
    /// The canonical matrix, row-major `n_rays x 256`.
    pub fn matrix(&self) -> &CdchMatrix {
        &self.matrix
    }
    /// Consume the result, keeping only the canonical matrix.
    pub fn into_matrix(self) -> CdchMatrix {
        self.matrix
    }
    /// Indices of rays with no samples (their CDF rows are `NaN`).
    pub fn empty_rays(&self) -> Vec<usize> {
        (0..self.n_rays())
            .filter(|&i| self.ray_lengths[i] == 0)
            .collect()
    }
    /// True when every ray has at least one sample, so the matrix contains no `NaN`.
    pub fn is_complete(&self) -> bool {
        self.ray_lengths.iter().all(|&n| n > 0)
    }

    /// Write the matrix as CSV: header `angle_deg,ray_length,F_0,...,F_255`, one row
    /// per ray. Values use Rust's shortest round-trip formatting, so parsing the text
    /// back yields the identical binary64 values; `NaN` is written as `NaN`.
    pub fn write_matrix_csv<W: Write>(&self, mut w: W) -> io::Result<()> {
        write!(w, "angle_deg,ray_length")?;
        for b in 0..BINS {
            write!(w, ",F_{b}")?;
        }
        writeln!(w)?;
        for i in 0..self.n_rays() {
            write!(w, "{},{}", self.angle(i), self.ray_lengths[i])?;
            for v in self.cdf(i) {
                write!(w, ",{v:?}")?;
            }
            writeln!(w)?;
        }
        Ok(())
    }

    /// Write the integer histograms as CSV: header
    /// `angle_deg,ray_length,count_0,...,count_255`.
    pub fn write_counts_csv<W: Write>(&self, mut w: W) -> io::Result<()> {
        write!(w, "angle_deg,ray_length")?;
        for b in 0..BINS {
            write!(w, ",count_{b}")?;
        }
        writeln!(w)?;
        for i in 0..self.n_rays() {
            write!(w, "{},{}", self.angle(i), self.ray_lengths[i])?;
            for c in self.counts(i) {
                write!(w, ",{c}")?;
            }
            writeln!(w)?;
        }
        Ok(())
    }
}
