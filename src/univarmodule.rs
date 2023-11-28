mod univarmodule;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "single_es_univar_predict")]
pub fn predict_py(alpha: f64, data: PyReadonlyArray1<f64>) -> f64 {
    univarmodule::single_predict(
        alpha,
        data.as_slice().expect("Could not get numpy array as slice"),
    )
}
