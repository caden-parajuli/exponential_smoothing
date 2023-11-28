use cobyla::{minimize, Func, RhoBeg};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

pub fn predict(alpha: f64, data: &[f64]) -> f64 {
    let mut weight: f64 = alpha;
    let mut result: f64 = 0f64;
    for x in data.iter().rev() {
        result += weight * x;
        weight *= 1f64 - alpha;
    }
    result
}

// pub fn mse_loss(alpha: f64, data: &[f64]) -> f64 {
//     0.0
// }

fn mse_loss(alpha: f64, l: f64, data: &[f64]) -> f64 {
    // Use prefix sum to predict all data
    data.iter()
        .scan(l, |state, &x| {
            *state = alpha * x + (1.0 - alpha) * *state;
            Some(*state)
        })
        .zip(data.iter().next()) // Not sure if this .next() works. I think it realigns the prediction to the data (ignoring l as a prediction)
        .fold(0.0, |acc, (x, &y)| acc + (x - y) * (x - y))
        / (data.len() as f64 - 1.0) // - 1.0 because of the .next() earlier
}

/// Problem cost function
fn mse_loss_mut(hyperparams: &[f64], data: &mut &[f64]) -> f64 {
    mse_loss(hyperparams[0], hyperparams[1], data)
}

fn train(initial_alpha: f64, initial_l: f64, data: &[f64]) -> (f64, f64) {
    let hyperparams = &[initial_alpha, initial_l];
    let cstr1 = |hyperparams: &[f64], _user_data: &mut &[f64]| -hyperparams[0];
    let cstr2 = |hyperparams: &[f64], _user_data: &mut &[f64]| hyperparams[0] - 1.0;
    let cons: Vec<&dyn Func<&[f64]>> = vec![&cstr1, &cstr2];

    match minimize(
        mse_loss_mut,
        hyperparams,
        &[(0.0, 1.0), (-10., 10.)],
        &cons,
        data,
        200,
        RhoBeg::All(0.5),
        None,
    ) {
        Ok((status, x_opt, _y_opt)) => {
            println!("Training status: {:?}", status);
            (x_opt[0], x_opt[1])
        }
        Err((e, _, _)) => panic!("Optimization error: {:?}", e),
    }
}

#[pyclass]
#[derive(Default)]
struct UnivarSingleModel {
    alpha: f64,
    l: f64,
}

#[pymethods]
impl UnivarSingleModel {
    #[new]
    pub fn new(alpha: f64, l: f64) -> Self {
        Self { alpha, l }
    }

    pub fn predict(&self, data: PyReadonlyArray1<f64>) -> f64 {
        predict(
            self.alpha,
            data.as_slice().expect("Could not get numpy array as slice"),
        )
    }

    pub fn fit(&mut self, data: PyReadonlyArray1<f64>) {
        (self.alpha, self.l) = train(
            self.alpha,
            self.l,
            data.as_slice().expect("Could not get numpy array as slice"),
        );
    }
}
