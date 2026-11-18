//! Fused image → representation extraction with reusable buffers.

use std::sync::Arc;

use cdch_core::engine::CHECKPOINTS;
use cdch_core::{
    CdchError, Engine, MatrixView, Params, Result, RgbImage, SamplingPlan, Workspace, BINS,
    RAYS_ARCHIVED,
};

use crate::{
    compact_from_checkpoints, compact_prevalidated, from_matrix, rich, sector_from_checkpoints,
    Representation, A_DIM,
};

/// Per-thread extractor. Holds reusable buffers and shares a plan cache.
///
/// * A, C, D: fused — the 256-bin matrix is never formed; only the 32 checkpoint
///   CDF values per ray are computed (bit-identical to the matrix columns).
/// * B: the matrix is formed in a reused internal buffer (needed by the PMF
///   moments and full-row angular lags) and reduced in place; it never leaves Rust.
#[derive(Debug)]
pub struct FeatureExtractor {
    engine: Arc<Engine>,
    ws: Workspace,
    cp: Vec<f64>,
    matrix: Vec<f64>,
    sampled: Vec<f64>,
    scratch: Vec<f64>,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtractor {
    pub fn new() -> Self {
        Self::with_engine(Arc::new(Engine::new()))
    }

    pub fn with_engine(engine: Arc<Engine>) -> Self {
        Self {
            engine,
            ws: Workspace::new(),
            cp: Vec::new(),
            matrix: Vec::new(),
            sampled: Vec::new(),
            scratch: Vec::new(),
        }
    }

    pub fn engine(&self) -> &Arc<Engine> {
        &self.engine
    }

    pub fn plan(&self, img: &RgbImage<'_>, params: &Params) -> Result<Arc<SamplingPlan>> {
        self.engine.plan(img.width(), img.height(), *params)
    }

    /// Full canonical matrix into `out` (`n_rays * 256`).
    pub fn matrix_into(
        &mut self,
        img: &RgbImage<'_>,
        params: &Params,
        out: &mut [f64],
    ) -> Result<()> {
        let plan = self.plan(img, params)?;
        plan.matrix_into(img, &mut self.ws, out)
    }

    /// The plan-level equivalent of `descriptor()`'s matrix checks. The engine's matrix is
    /// non-finite, non-terminal or non-monotone only if some ray is empty, so checking the
    /// geometry replaces the O(360 x 256) scan, exactly.
    fn require_valid_archived(plan: &SamplingPlan) -> Result<()> {
        if plan.n_rays() != RAYS_ARCHIVED {
            return Err(CdchError::RayCount {
                expected: RAYS_ARCHIVED,
                actual: plan.n_rays(),
            });
        }
        if !plan.all_rays_nonempty() {
            return Err(CdchError::InvalidMatrix(
                "non-finite values (empty ray?)".into(),
            ));
        }
        Ok(())
    }

    /// Checkpoint matrix into the internal buffer; returns the plan used.
    fn checkpoints(&mut self, img: &RgbImage<'_>, params: &Params) -> Result<Arc<SamplingPlan>> {
        let plan = self.plan(img, params)?;
        if plan.n_rays() != RAYS_ARCHIVED {
            return Err(CdchError::RayCount {
                expected: RAYS_ARCHIVED,
                actual: plan.n_rays(),
            });
        }
        self.cp.resize(plan.n_rays() * CHECKPOINTS, 0.0);
        plan.checkpoints_into(img, &mut self.ws, &mut self.cp)?;
        Ok(plan)
    }

    /// One representation, fused where possible.
    pub fn extract(
        &mut self,
        img: &RgbImage<'_>,
        params: &Params,
        rep: Representation,
        out: &mut [f64],
    ) -> Result<()> {
        if out.len() != rep.dim() {
            return Err(CdchError::BufferSize {
                expected: rep.dim(),
                actual: out.len(),
            });
        }
        match rep {
            Representation::A => {
                let plan = self.checkpoints(img, params)?;
                Self::require_valid_archived(&plan)?;
                compact_from_checkpoints(&self.cp, &mut self.sampled, &mut self.scratch, out)
            }
            Representation::C | Representation::D => {
                self.checkpoints(img, params)?;
                sector_from_checkpoints(&self.cp, out)
            }
            Representation::B => {
                let plan = self.plan(img, params)?;
                Self::require_valid_archived(&plan)?;
                self.matrix.resize(plan.n_rays() * BINS, 0.0);
                plan.matrix_into(img, &mut self.ws, &mut self.matrix)?;
                let m = MatrixView::new(plan.n_rays(), &self.matrix)?;
                compact_prevalidated(&m, &mut out[..A_DIM]);
                rich(&m, &mut out[A_DIM..]);
                Ok(())
            }
        }
    }

    /// Unfused reference path: full matrix, then the representation from the matrix.
    pub fn extract_via_matrix(
        &mut self,
        img: &RgbImage<'_>,
        params: &Params,
        rep: Representation,
        out: &mut [f64],
    ) -> Result<()> {
        let plan = self.plan(img, params)?;
        self.matrix.resize(plan.n_rays() * BINS, 0.0);
        plan.matrix_into(img, &mut self.ws, &mut self.matrix)?;
        from_matrix(&MatrixView::new(plan.n_rays(), &self.matrix)?, rep, out)
    }
}
