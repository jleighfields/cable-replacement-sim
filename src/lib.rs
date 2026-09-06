//! Rust compute kernel for the cable replacement simulation.
//!
//! The Python package `cablesim` wraps this module. Everything here is
//! reached through PyO3, and the Python reference in `python/cablesim/` mirrors
//! it deliberately — the parity tests between the two are what validate this
//! side, so neither is redundant with the other.

use pyo3::prelude::*;

/// Adds two integers.
///
/// This exists to prove the build pipeline end to end: that maturin compiles
/// the crate, that the resulting extension module imports under the project's
/// Python, and that a value survives the round trip. The simulation kernel
/// replaces it; until then it is what the smoke test asserts on.
///
/// # Arguments
///
/// * `a` - left operand.
/// * `b` - right operand.
///
/// # Returns
///
/// The sum of `a` and `b`.
#[pyfunction]
fn add(a: i64, b: i64) -> i64 {
    a + b
}

/// Registers the extension module's contents under the name `_cablesim`.
#[pymodule]
fn _cablesim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_sums_its_arguments() {
        assert_eq!(add(2, 3), 5);
        assert_eq!(add(-1, 1), 0);
    }
}
