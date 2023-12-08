use pyo3::prelude::*;

// mod multivarmodule;
// mod univarmodule;
pub mod univar;

#[pymodule()]
#[pyo3(name = "_exp_smooth")]
fn exp_smooth(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<univar::UnivarSingleModel>()?;
    Ok(())
}
