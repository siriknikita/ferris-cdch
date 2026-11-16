//! `cdch_ferris._native` — NumPy bindings for the validated Rust CDCH implementation.
//!
//! Inputs are validated (type, dtype, shape, channel count) and reported as Python
//! exceptions, never panics.
//!
//! Memory/GIL policy:
//! * single-image calls read C-contiguous `uint8` arrays in place (zero copy) and
//!   keep the GIL, so no Python thread can mutate the buffer mid-computation;
//! * batch calls copy the stack once (≈49 KB per 128² image) and then release the
//!   GIL for the parallel computation, which is sound because Rust owns the copy;
//! * non-contiguous inputs are copied once into packed order.

use std::cell::RefCell;
use std::sync::{Arc, OnceLock};

use cdch_core::{reference, CdchError, Center, Engine, MatrixView, Params, RgbImage, BINS};
use cdch_features::{FeatureExtractor, Representation};
use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArray4, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

fn engine() -> &'static Arc<Engine> {
    static ENGINE: OnceLock<Arc<Engine>> = OnceLock::new();
    ENGINE.get_or_init(|| Arc::new(Engine::new()))
}

thread_local! {
    static EXTRACTOR: RefCell<FeatureExtractor> = RefCell::new(FeatureExtractor::with_engine(Arc::clone(engine())));
}

fn err(e: CdchError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Borrowed (zero-copy) or owned RGB pixels plus geometry, extracted from a NumPy array.
struct Pixels<'py> {
    width: usize,
    height: usize,
    data: PixelData<'py>,
}

enum PixelData<'py> {
    Borrowed(numpy::PyReadonlyArray3<'py, u8>),
    Owned(Vec<u8>),
}

impl Pixels<'_> {
    fn slice(&self) -> &[u8] {
        match &self.data {
            PixelData::Borrowed(a) => a.as_slice().expect("checked contiguous"),
            PixelData::Owned(v) => v,
        }
    }
    fn image(&self) -> Result<RgbImage<'_>, CdchError> {
        RgbImage::new(self.width, self.height, self.slice())
    }
}

fn check_uint8(arr: &Bound<'_, PyUntypedArray>, what: &str) -> PyResult<()> {
    let dtype = arr.dtype();
    let ok = dtype.is_equiv_to(&numpy::dtype::<u8>(arr.py()));
    if !ok {
        return Err(PyTypeError::new_err(format!(
            "{what} must have dtype uint8 (decoded 8-bit RGB), got {}",
            dtype.str()?
        )));
    }
    Ok(())
}

fn as_array<'py>(obj: &Bound<'py, PyAny>, what: &str) -> PyResult<Bound<'py, PyUntypedArray>> {
    obj.cast::<PyUntypedArray>()
        .map(|a| a.clone())
        .map_err(|_| PyTypeError::new_err(format!("{what} must be a numpy.ndarray, got {}", obj.get_type().name().map(|n| n.to_string()).unwrap_or_default())))
}

/// Extract a single `(H, W, 3)` uint8 image.
fn image_from<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Pixels<'py>> {
    let arr = as_array(obj, "rgb")?;
    check_uint8(&arr, "rgb")?;
    let shape = arr.shape().to_vec();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(PyValueError::new_err(format!(
            "rgb must have shape (height, width, 3) in RGB channel order, got {shape:?}"
        )));
    }
    let typed = arr.cast::<PyArray3<u8>>()?.readonly();
    let data = if typed.as_slice().is_ok() {
        PixelData::Borrowed(typed)
    } else {
        PixelData::Owned(typed.as_array().iter().copied().collect())
    };
    Ok(Pixels { width: shape[1], height: shape[0], data })
}

fn params_for(width: usize, height: usize, center: Option<(f64, f64)>, angle_step: u32) -> PyResult<Params> {
    let params = Params {
        center: center.map_or(Center::archived(width, height), |(x, y)| Center::new(x, y)),
        angle_step,
    };
    params.validate().map_err(err)?;
    Ok(params)
}

fn representation(kind: &str) -> PyResult<Representation> {
    Representation::parse(kind)
        .ok_or_else(|| PyValueError::new_err(format!("unknown CDCH representation {kind:?}; expected 'A', 'B', 'C' or 'D'")))
}

/// Canonical CDCH matrix, shape `(n_rays, 256)`, float64.
///
/// `center` defaults to the archived protocol `(width // 2, height // 2)`;
/// pass `(width / 2, height / 2)` to reproduce the oracle's own default.
#[pyfunction]
#[pyo3(signature = (rgb, center=None, angle_step=1))]
fn cdch<'py>(py: Python<'py>, rgb: &Bound<'py, PyAny>, center: Option<(f64, f64)>, angle_step: u32) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let px = image_from(rgb)?;
    let params = params_for(px.width, px.height, center, angle_step)?;
    let rows = params.n_rays();
    let img = px.image().map_err(err)?;
    let mut out = vec![0.0; rows * BINS];
    EXTRACTOR.with(|fx| fx.borrow_mut().matrix_into(&img, &params, &mut out)).map_err(err)?;
    Ok(Array2::from_shape_vec((rows, BINS), out).expect("shape").into_pyarray(py))
}

/// Canonical CDCH via the literal reference port (slow; for verification).
#[pyfunction]
#[pyo3(signature = (rgb, center=None, angle_step=1))]
fn cdch_reference<'py>(py: Python<'py>, rgb: &Bound<'py, PyAny>, center: Option<(f64, f64)>, angle_step: u32) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let px = image_from(rgb)?;
    let params = params_for(px.width, px.height, center, angle_step)?;
    let m = reference::cdch(&px.image().map_err(err)?, &params).map_err(err)?;
    let rows = m.rows();
    Ok(Array2::from_shape_vec((rows, BINS), m.into_vec()).expect("shape").into_pyarray(py))
}

/// Integer ray histograms, shape `(n_rays, 256)`, uint32.
#[pyfunction]
#[pyo3(signature = (rgb, center=None, angle_step=1))]
fn histograms<'py>(py: Python<'py>, rgb: &Bound<'py, PyAny>, center: Option<(f64, f64)>, angle_step: u32) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let px = image_from(rgb)?;
    let params = params_for(px.width, px.height, center, angle_step)?;
    let rows = params.n_rays();
    let plan = engine().plan(px.width, px.height, params).map_err(err)?;
    let mut ws = cdch_core::Workspace::new();
    let data = plan.histograms(&px.image().map_err(err)?, &mut ws).map_err(err)?;
    Ok(Array2::from_shape_vec((rows, BINS), data).expect("shape").into_pyarray(py))
}

/// Sampled coordinates: `(ray_lengths, coords)` with `coords` of shape `(N, 2)` holding
/// `(px, py)` in emission order (multiplicities preserved).
#[pyfunction]
#[pyo3(signature = (width, height, center=None, angle_step=1))]
fn sample_coordinates<'py>(
    py: Python<'py>,
    width: usize,
    height: usize,
    center: Option<(f64, f64)>,
    angle_step: u32,
) -> PyResult<(Bound<'py, PyArrayDyn<i64>>, Bound<'py, PyArray2<i64>>)> {
    let params = params_for(width, height, center, angle_step)?;
    let plan = engine().plan(width, height, params).map_err(err)?;
    let lengths: Vec<i64> = (0..plan.n_rays()).map(|i| plan.ray_len(i) as i64).collect();
    let mut coords = Vec::with_capacity(plan.total_samples() * 2);
    for i in 0..plan.n_rays() {
        for (x, y) in plan.ray_coords(i) {
            coords.push(x as i64);
            coords.push(y as i64);
        }
    }
    let n = coords.len() / 2;
    Ok((
        numpy::ndarray::ArrayD::from_shape_vec(vec![lengths.len()], lengths).expect("shape").into_pyarray(py),
        Array2::from_shape_vec((n, 2), coords).expect("shape").into_pyarray(py),
    ))
}

fn shaped<'py>(py: Python<'py>, rep: Representation, v: Vec<f64>) -> Bound<'py, PyArrayDyn<f64>> {
    let shape = if rep == Representation::D { vec![24, 32] } else { vec![v.len()] };
    numpy::ndarray::ArrayD::from_shape_vec(shape, v).expect("shape").into_pyarray(py)
}

/// One representation computed directly from pixels (fused where possible).
/// A: (56,), B: (98,), C: (768,), D: (24, 32).
#[pyfunction]
#[pyo3(signature = (rgb, kind="A", center=None, fused=true))]
fn features<'py>(py: Python<'py>, rgb: &Bound<'py, PyAny>, kind: &str, center: Option<(f64, f64)>, fused: bool) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let rep = representation(kind)?;
    let px = image_from(rgb)?;
    let params = params_for(px.width, px.height, center, 1)?;
    let img = px.image().map_err(err)?;
    let mut out = vec![0.0; rep.dim()];
    EXTRACTOR
        .with(|fx| {
            let mut fx = fx.borrow_mut();
            if fused {
                fx.extract(&img, &params, rep, &mut out)
            } else {
                fx.extract_via_matrix(&img, &params, rep, &mut out)
            }
        })
        .map_err(err)?;
    Ok(shaped(py, rep, out))
}

/// Several representations from one image in a single call: returns `{kind: array}`.
#[pyfunction]
#[pyo3(signature = (rgb, kinds, center=None))]
fn features_multi<'py>(py: Python<'py>, rgb: &Bound<'py, PyAny>, kinds: Vec<String>, center: Option<(f64, f64)>) -> PyResult<Bound<'py, PyDict>> {
    let reps = kinds.iter().map(|k| representation(k)).collect::<PyResult<Vec<_>>>()?;
    let px = image_from(rgb)?;
    let params = params_for(px.width, px.height, center, 1)?;
    let img = px.image().map_err(err)?;
    let outs = EXTRACTOR
        .with(|fx| {
            let mut fx = fx.borrow_mut();
            reps.iter()
                .map(|&rep| {
                    let mut out = vec![0.0; rep.dim()];
                    fx.extract(&img, &params, rep, &mut out).map(|_| out)
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(err)?;
    let d = PyDict::new(py);
    for ((k, rep), v) in kinds.iter().zip(&reps).zip(outs) {
        d.set_item(k, shaped(py, *rep, v))?;
    }
    Ok(d)
}

/// A representation from an existing canonical matrix of shape `(n_rays, 256)`.
#[pyfunction]
#[pyo3(signature = (cdf, kind="A"))]
fn features_from_matrix<'py>(py: Python<'py>, cdf: &Bound<'py, PyAny>, kind: &str) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let rep = representation(kind)?;
    let arr = as_array(cdf, "cdf")?;
    if !arr.dtype().is_equiv_to(&numpy::dtype::<f64>(py)) {
        return Err(PyTypeError::new_err(format!("cdf must have dtype float64, got {}", arr.dtype().str()?)));
    }
    let shape = arr.shape().to_vec();
    if shape.len() != 2 || shape[1] != BINS {
        return Err(PyValueError::new_err(format!("cdf must have shape (n_rays, 256), got {shape:?}")));
    }
    let typed = arr.cast::<PyArray2<f64>>()?.readonly();
    let owned: Vec<f64>;
    let data: &[f64] = match typed.as_slice() {
        Ok(s) => s,
        Err(_) => {
            owned = typed.as_array().iter().copied().collect();
            &owned
        }
    };
    let mut out = vec![0.0; rep.dim()];
    cdch_features::from_matrix(&MatrixView::new(shape[0], data).map_err(err)?, rep, &mut out).map_err(err)?;
    Ok(shaped(py, rep, out))
}

fn stack_from(obj: &Bound<'_, PyAny>) -> PyResult<(usize, usize, usize, Vec<u8>)> {
    let arr = as_array(obj, "images")?;
    check_uint8(&arr, "images")?;
    let shape = arr.shape().to_vec();
    if shape.len() != 4 || shape[3] != 3 {
        return Err(PyValueError::new_err(format!("images must have shape (n, height, width, 3), got {shape:?}")));
    }
    let typed = arr.cast::<PyArray4<u8>>()?.readonly();
    let data = match typed.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => typed.as_array().iter().copied().collect(),
    };
    Ok((shape[0], shape[1], shape[2], data))
}

fn pool(threads: Option<usize>) -> PyResult<Option<rayon::ThreadPool>> {
    match threads {
        None | Some(0) => Ok(None),
        Some(n) => rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map(Some)
            .map_err(|e| PyValueError::new_err(e.to_string())),
    }
}

/// Batch feature extraction over a `(n, H, W, 3)` uint8 stack → `(n, dim)` float64.
/// `threads=1` runs serially on the calling thread; `None` uses all cores (rayon).
#[pyfunction]
#[pyo3(signature = (images, kind="A", center=None, threads=None))]
fn batch_features<'py>(py: Python<'py>, images: &Bound<'py, PyAny>, kind: &str, center: Option<(f64, f64)>, threads: Option<usize>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rep = representation(kind)?;
    let (n, h, w, data) = stack_from(images)?;
    let params = params_for(w, h, center, 1)?;
    let dim = rep.dim();
    let pool = pool(threads)?;
    let out = py
        .detach(|| {
            let mut out = vec![0.0; n * dim];
            let stride = h * w * 3;
            let work = |(i, dst): (usize, &mut [f64])| -> Result<(), CdchError> {
                let img = RgbImage::new(w, h, &data[i * stride..(i + 1) * stride])?;
                EXTRACTOR.with(|fx| fx.borrow_mut().extract(&img, &params, rep, dst))
            };
            if threads == Some(1) || n <= 1 {
                out.chunks_mut(dim.max(1)).enumerate().try_for_each(work)?;
            } else if let Some(p) = &pool {
                p.install(|| out.par_chunks_mut(dim).enumerate().try_for_each(work))?;
            } else {
                out.par_chunks_mut(dim).enumerate().try_for_each(work)?;
            }
            Ok::<_, CdchError>(out)
        })
        .map_err(err)?;
    Ok(Array2::from_shape_vec((n, dim), out).expect("shape").into_pyarray(py))
}

/// Batch canonical matrices: `(n, H, W, 3)` uint8 → `(n, n_rays, 256)` float64.
#[pyfunction]
#[pyo3(signature = (images, center=None, angle_step=1, threads=None))]
fn batch_cdch<'py>(py: Python<'py>, images: &Bound<'py, PyAny>, center: Option<(f64, f64)>, angle_step: u32, threads: Option<usize>) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let (n, h, w, data) = stack_from(images)?;
    let params = params_for(w, h, center, angle_step)?;
    let rows = params.n_rays();
    let per = rows * BINS;
    let pool = pool(threads)?;
    let out = py
        .detach(|| {
            let plan = engine().plan(w, h, params)?;
            let mut out = vec![0.0; n * per];
            let stride = h * w * 3;
            let work = |(i, dst): (usize, &mut [f64])| -> Result<(), CdchError> {
                let img = RgbImage::new(w, h, &data[i * stride..(i + 1) * stride])?;
                let mut ws = cdch_core::Workspace::new();
                plan.matrix_into(&img, &mut ws, dst)
            };
            if threads == Some(1) || n <= 1 {
                out.chunks_mut(per.max(1)).enumerate().try_for_each(work)?;
            } else if let Some(p) = &pool {
                p.install(|| out.par_chunks_mut(per).enumerate().try_for_each(work))?;
            } else {
                out.par_chunks_mut(per).enumerate().try_for_each(work)?;
            }
            Ok::<_, CdchError>(out)
        })
        .map_err(err)?;
    Ok(Array3::from_shape_vec((n, rows, BINS), out).expect("shape").into_pyarray(py))
}

/// Build metadata (for benchmark provenance).
#[pyfunction]
fn build_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("version", env!("CARGO_PKG_VERSION"))?;
    d.set_item("target_arch", std::env::consts::ARCH)?;
    d.set_item("target_os", std::env::consts::OS)?;
    d.set_item("debug_assertions", cfg!(debug_assertions))?;
    d.set_item("canonical_sha256", cdch_core::CANONICAL_SHA256)?;
    d.set_item("rayon_threads", rayon::current_num_threads())?;
    d.set_item("cached_plans", engine().cached_plans())?;
    Ok(d)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("CANONICAL_SHA256", cdch_core::CANONICAL_SHA256)?;
    m.add("A_DIM", cdch_features::A_DIM)?;
    m.add("B_DIM", cdch_features::B_DIM)?;
    m.add("C_DIM", cdch_features::C_DIM)?;
    m.add_function(wrap_pyfunction!(cdch, m)?)?;
    m.add_function(wrap_pyfunction!(cdch_reference, m)?)?;
    m.add_function(wrap_pyfunction!(histograms, m)?)?;
    m.add_function(wrap_pyfunction!(sample_coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(features, m)?)?;
    m.add_function(wrap_pyfunction!(features_multi, m)?)?;
    m.add_function(wrap_pyfunction!(features_from_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(batch_features, m)?)?;
    m.add_function(wrap_pyfunction!(batch_cdch, m)?)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    Ok(())
}
