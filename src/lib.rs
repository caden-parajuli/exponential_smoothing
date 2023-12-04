use pyo3::prelude::*;

// mod multivarmodule;
// mod univarmodule;
pub mod univar;

#[pymodule()]
#[pyo3(name = "_exp_smooth")]
fn exp_smooth(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(univarmodule::predict_py, m)?)?;
    // m.add_function(wrap_pyfunction!(univar::predict_py, m)?)?;
    // m.add_function(wrap_pyfunction!(multivarmodule::single_predict_py, m)?)?;
    m.add_class::<univar::UnivarSingleModel>()?;
    Ok(())
}
