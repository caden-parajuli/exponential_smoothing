use pyo3::prelude::*;

mod multivarmodule;
// mod univarmodule;
pub mod univar;

#[pymodule]
fn _exponential_smoothing(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(univarmodule::predict_py, m)?)?;
    // m.add_function(wrap_pyfunction!(univar::predict_py, m)?)?;
    m.add_function(wrap_pyfunction!(multivarmodule::single_predict_py, m)?)?;
    Ok(())
}
