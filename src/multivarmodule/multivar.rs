use ndarray::prelude::*;
use ndarray::Array;

/// Single multivariate exponential smoothing (i.e. no trend or seasonality).
/// The weight matrices are given by $ A_i = R G^i $ and the estimate is given by $ X_t(1) = X_{t-1}(1) + (I - a)(X_t - X_{t-1}(1)) $
/// where the initial prediction is l_0
pub fn single_predict(
    data: &ArrayView<f64, Ix2>,
    a: &ArrayView<f64, Ix2>,
    l_0: &ArrayView<f64, Ix2>,
) -> Array<f64, Ix1> {
    assert_eq!(
        a.shape()[0],
        a.shape()[1],
        "Matrix arguments a and l_0 must be square"
    );
    assert_eq!(
        a.shape(),
        l_0.shape(),
        "Matrix arguments a and l_0 must have the same shape"
    );
    assert_eq!(
        a.shape()[1],
        data.shape()[1],
        "Matrix argument a must have the same number of rows as the data matrix"
    );
    let i_minus_a = a - Array::<f64, _>::eye(a.shape()[0]);
    let mut x_pred = l_0.dot(&data.row(0)).to_owned();
    for r in data.rows() {
        x_pred += &i_minus_a.dot(&(&r - &x_pred));
    }
    return x_pred;
}
