mod multivar;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "single_es_multivar_predict")]
pub fn single_predict_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    a: PyReadonlyArray2<f64>,
    l_0: PyReadonlyArray2<f64>,
) -> &'py PyArray1<f64> {
    return multivar::single_predict(&data.as_array(), &a.as_array(), &l_0.as_array())
        .into_pyarray(py);
}
