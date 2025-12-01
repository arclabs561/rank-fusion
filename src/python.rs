//! Python bindings for rank-fusion using PyO3.
//!
//! Provides a Python API that mirrors the Rust API, enabling seamless
//! integration with Python RAG/search stacks.
//!
//! # Usage
//!
//! ```python
//! import rank_fusion
//!
//! bm25 = [("d1", 12.5), ("d2", 11.0)]
//! dense = [("d2", 0.9), ("d3", 0.8)]
//!
//! fused = rank_fusion.rrf(bm25, dense, k=60)
//! # [("d2", 0.033), ("d1", 0.016), ("d3", 0.016)]
//! ```

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::{PyList, PyTuple};

#[cfg(feature = "pyo3")]
use crate::{rrf, rrf_multi, RrfConfig};

/// Python module for rank fusion.
#[cfg(feature = "pyo3")]
#[pymodule]
fn rank_fusion(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rrf_py, m)?)?;
    m.add_function(wrap_pyfunction!(rrf_multi_py, m)?)?;
    m.add_class::<RrfConfigPy>()?;
    Ok(())
}

/// RRF fusion for two ranked lists.
///
/// # Arguments
/// * `results_a`: List of (id, score) tuples from first retriever
/// * `results_b`: List of (id, score) tuples from second retriever
/// * `k`: Smoothing constant (default: 60)
///
/// # Returns
/// List of (id, score) tuples sorted by fused score (descending)
#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (results_a, results_b, k = 60))]
fn rrf_py(
    py: Python<'_>,
    results_a: &Bound<'_, PyList>,
    results_b: &Bound<'_, PyList>,
    k: u32,
) -> PyResult<Bound<'_, PyList>> {
    // Convert Python lists to Rust Vec<(String, f32)>
    let a: Vec<(String, f32)> = py_list_to_ranked(results_a)?;
    let b: Vec<(String, f32)> = py_list_to_ranked(results_b)?;

    // Call Rust function
    let fused = rrf(&a, &b);

    // Convert back to Python list
    let result = PyList::empty_bound(py);
    for (id, score) in fused {
        let tuple = PyTuple::new_bound(py, &[id.into_py(py), score.into_py(py)]);
        result.append(tuple)?;
    }
    Ok(result)
}

/// RRF fusion for multiple ranked lists.
///
/// # Arguments
/// * `lists`: List of lists, each containing (id, score) tuples
/// * `k`: Smoothing constant (default: 60)
///
/// # Returns
/// List of (id, score) tuples sorted by fused score (descending)
#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (lists, k = 60))]
fn rrf_multi_py(py: Python<'_>, lists: &Bound<'_, PyList>, k: u32) -> PyResult<Bound<'_, PyList>> {
    // Convert Python lists to Rust Vec<Vec<(String, f32)>>
    let mut rust_lists = Vec::new();
    for list in lists.iter() {
        let py_list = list.downcast::<PyList>()?;
        rust_lists.push(py_list_to_ranked(py_list)?);
    }

    // Convert to slice of slices for Rust API
    let slices: Vec<&[(String, f32)]> = rust_lists.iter().map(|v| v.as_slice()).collect();

    // Call Rust function
    let config = RrfConfig::new(k);
    let fused = rrf_multi(&slices, config);

    // Convert back to Python list
    let result = PyList::empty_bound(py);
    for (id, score) in fused {
        let tuple = PyTuple::new_bound(py, &[id.into_py(py), score.into_py(py)]);
        result.append(tuple)?;
    }
    Ok(result)
}

/// Helper to convert Python list of (id, score) tuples to Rust Vec.
#[cfg(feature = "pyo3")]
fn py_list_to_ranked(py_list: &Bound<'_, PyList>) -> PyResult<Vec<(String, f32)>> {
    let mut result = Vec::new();
    for item in py_list.iter() {
        let tuple = item.downcast::<PyTuple>()?;
        if tuple.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Each item must be a (id, score) tuple",
            ));
        }
        let id = tuple.get_item(0)?.extract::<String>()?;
        let score = tuple.get_item(1)?.extract::<f32>()?;
        result.push((id, score));
    }
    Ok(result)
}

/// Python wrapper for RrfConfig.
#[cfg(feature = "pyo3")]
#[pyclass]
struct RrfConfigPy {
    inner: RrfConfig,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl RrfConfigPy {
    #[new]
    fn new(k: u32) -> Self {
        Self {
            inner: RrfConfig::new(k),
        }
    }

    #[getter]
    fn k(&self) -> u32 {
        self.inner.k
    }

    fn with_k(&self, k: u32) -> Self {
        Self {
            inner: self.inner.with_k(k),
        }
    }
}
