pub fn single_predict(alpha: f64, data: &[f64]) -> f64 {
    let mut weight: f64 = alpha;
    let mut result: f64 = 0f64;
    for x in data.iter().rev() {
        result += weight * x;
        weight *= 1f64 - alpha;
    }
    result
}
