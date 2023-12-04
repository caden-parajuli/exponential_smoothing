// use cobyla::{minimize, Func, RhoBeg};

// /// Problem cost function
// fn mse_loss_mut(hyperparams: &[f64], data: &mut &[f64]) -> f64 {
//     mse_loss(hyperparams[0], hyperparams[1], data)
// }

// fn train(initial_alpha: f64, initial_l: f64, data: &[f64]) -> (f64, f64) {
//     let hyperparams = &[initial_alpha, initial_l];
//     let cstr1 = |hyperparams: &[f64], _user_data: &mut &[f64]| -hyperparams[0];
//     let cstr2 = |hyperparams: &[f64], _user_data: &mut &[f64]| hyperparams[0] - 1.0;
//     let cons: Vec<&dyn Func<&[f64]>> = vec![&cstr1, &cstr2];

//     match minimize(
//         mse_loss_mut,
//         hyperparams,
//         &[(0.0, 1.0), (-10., 10.)],
//         &cons,
//         data,
//         1000,
//         RhoBeg::All(0.5),
//         None,
//     ) {
//         Ok((status, x_opt, _y_opt)) => {
//             println!("Training status: {:?}", status);
//             (x_opt[0], x_opt[1])
//         }
//         Err((e, _, _)) => panic!("Optimization error: {:?}", e),
//     }
// }
