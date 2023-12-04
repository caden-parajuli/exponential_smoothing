use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

/// Macro for getting a Rust slice from a numpy array
macro_rules! np_to_slice {
    ($data:expr) => {{
        $data
            .as_slice()
            .expect("Could not get numpy array as slice. Consider using numpy.ascontiguousarray.")
    }};
}

pub fn predict(alpha: f64, data: &[f64]) -> f64 {
    let mut weight: f64 = alpha;
    let mut result: f64 = 0f64;
    for x in data.iter().rev() {
        result += weight * x;
        weight *= 1f64 - alpha;
    }
    result
}

/// Returns a vector where the element at index i is the prediction for the
/// element at index i in the data array, given the data previous to this.
fn predict_all(alpha: f64, l: f64, data: &[f64]) -> Vec<f64> {
    let mut arr = vec![0.0; data.len() + 1];
    data.iter()
        .scan(l, |state, &x| {
            // println!("{alpha} * {x} + (1.0 - {alpha}) * {state}");
            *state = alpha * x + (1.0 - alpha) * *state;
            Some(*state)
        })
        .zip(1..(data.len() + 1))
        .for_each(|(pred, index)| arr[index] = pred);
    arr[0] = l;
    arr
}

fn mse_loss(alpha: f64, l: f64, data: &[f64]) -> f64 {
    let mut pred = l;
    let mut loss = 0.0;
    for i in 0..(data.len() - 1) {
        pred = alpha * data[i] + (1.0 - alpha) * pred;
        loss += (pred - data[i + 1]) * (pred - data[i + 1])
    }
    loss / (data.len() as f64 - 1.0)
}

/// Normalized RMS value. Given by dividing RMS by the mean of the test data
fn nrmse(alpha: f64, l: f64, data: &[f64], test_cutoff: usize) -> f64 {
    let mut pred = l;
    let mut loss = 0.0;
    let mut mean = 0.0;
    for i in 0..(data.len() - 1) {
        pred = alpha * data[i] + (1.0 - alpha) * pred;
        if i > test_cutoff {
            loss += (pred - data[i + 1]) * (pred - data[i + 1]);
            mean += data[i + 1];
        }
    }
    mean /= (data.len() - test_cutoff - 1) as f64;
    loss /= (data.len() - test_cutoff - 1) as f64;
    let rmse = loss.sqrt();
    rmse / mean
}

/// Nash-Sutcliffe Efficiency
fn nse(alpha: f64, l: f64, data: &[f64], test_cutoff: usize) -> f64 {
    let mut pred: f64 = l;
    let mut loss: f64 = 0.0;
    let mut variance: f64 = 0.0;

    let mean =
        data[(test_cutoff + 1)..].into_par_iter().sum::<f64>() / (data.len() - test_cutoff) as f64;

    for i in 0..(data.len() - 1) {
        pred = alpha * data[i] + (1.0 - alpha) * pred;
        if i > test_cutoff {
            loss += (pred - data[i + 1]) * (pred - data[i + 1]);
            variance += (mean - data[i + 1]) * (mean - data[i + 1]);
        }
    }
    1.0 - loss / variance
}

fn grid_search(
    alpha_bounds: (f64, f64),
    l_bounds: (f64, f64),
    d_alpha: f64,
    d_l: f64,
    data: &[f64],
) -> (f64, f64, f64) {
    let mut alpha = alpha_bounds.0;
    let mut l = l_bounds.0;
    let mut loss = f64::INFINITY;
    let mut min = (alpha, l, loss);

    while alpha <= alpha_bounds.1 {
        l = l_bounds.0;
        while l <= l_bounds.1 {
            loss = mse_loss(alpha, l, data);
            if loss < min.2 {
                min = (alpha, l, loss);
            }
            l += d_l;
        }
        alpha += d_alpha;
    }
    min
}

#[pyclass(get_all, set_all)]
#[derive(Default)]
pub struct UnivarSingleModel {
    alpha: f64,
    l: f64,
}

#[pymethods]
impl UnivarSingleModel {
    #[new]
    #[pyo3(signature = (alpha=0.0, l=0.0))]
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
        let data_slice = np_to_slice!(data);
        let dl = data_slice[0] / 100.0;

        (self.alpha, self.l, _) = grid_search(
            (0.0, 1.0),
            (-5.0 * data_slice[0], 5.0 * data_slice[0]),
            0.01,
            dl,
            data_slice,
        );
    }

    pub fn grid_search(
        &mut self,
        alpha_bounds: (f64, f64),
        l_bounds: (f64, f64),
        d_alpha: f64,
        d_l: f64,
        data: PyReadonlyArray1<f64>,
    ) {
        (self.alpha, self.l, _) =
            grid_search(alpha_bounds, l_bounds, d_alpha, d_l, np_to_slice!(data));
    }

    pub fn mse_loss(&self, data: PyReadonlyArray1<f64>) -> f64 {
        mse_loss(self.alpha, self.l, np_to_slice!(data))
    }

    pub fn nrmse(&self, data: PyReadonlyArray1<f64>, test_cutoff: usize) -> f64 {
        nrmse(self.alpha, self.l, np_to_slice!(data), test_cutoff)
    }

    pub fn nse(&self, data: PyReadonlyArray1<f64>, test_cutoff: usize) -> f64 {
        nse(self.alpha, self.l, np_to_slice!(data), test_cutoff)
    }

    pub fn predict_all(&self, data: PyReadonlyArray1<f64>) -> Vec<f64> {
        predict_all(self.alpha, self.l, np_to_slice!(data))
    }

    pub fn __repr__(&self) -> String {
        format!("UnivarSingleModel({}, {})", self.alpha, self.l)
    }
}

impl std::fmt::Display for UnivarSingleModel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Single variable exponential smoothing model \n\t alpha: {} \n\t l: {}",
            self.alpha, self.l
        )
    }
}
