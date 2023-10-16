# exponential_smoothing

An implementation of exponential smoothing in Rust, with Python bindings. As far as I'm aware, this contains the **only** publicly available implementation of multivariate exponential smoothing. 

## Features

Unfortunately fitting algorithms have not yet been implemented for these models.

- [x] Univariate simple ES
- [x] Multivariate simple ES
- [ ] Univariate double (Holt linear) ES
- [ ] Multivariate double (Holt linear) ES
- [ ] Univariate triple (Holt-Winters) ES
- [ ] Multivariate triple (Holt-Winters) ES

## Other Tasks

- Fix project structure to isolate Python bindings from Rust code.
- Implement fitting algorithms

## Why Rust?

- **Performance**: As a systems programming language, Rust is *much* faster than an interpreted language like Python, and still faster than a garbage-collected language like Go.
- **Ecosystem**: Rust has a surprisingly vibrant ecosystem with packages like `ndarray` and `argmin`, which are especially useful for this library.
- **Reliability**: Rust is best known for its ownership model, which guarantees memory-safety. This helps eliminate bugs, making software more reliable.
- **Python integration**: Although the standard Python implementation, CPython, is written in C, I've found interfacing Python with C code to be an unmaintainable mess. Rust's `maturin` is much more pleasant.