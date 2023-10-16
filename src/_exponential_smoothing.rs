use pyo3::prelude::*;

mod multivarmodule;
mod univarmodule;

#[pymodule]
fn _exponential_smoothing(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(univarmodule::predict_py, m)?)?;
    m.add_function(wrap_pyfunction!(multivarmodule::single_predict_py, m)?)?;
    Ok(())
}
