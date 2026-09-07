//! Rust compute kernel for the cable replacement simulation.
//!
//! The Python package `cablesim` wraps this module. Everything here is reached
//! through PyO3, and the Python reference in `python/cablesim/` mirrors it
//! deliberately — the parity tests between the two are what validate this
//! side, so neither is redundant with the other. A mirrored pair carries the
//! same module name on both sides: `weibull`, `policies`, `simulate`.
//!
//! This file is the boundary and nothing else. It checks what arrives, calls
//! `simulate::run_chunk`, and hands back seven arrays; every decision the model
//! makes is in the three modules below it.
//!
//! # Reading this beside the Python
//!
//! Everything below is either an attribute telling PyO3 to generate glue code,
//! or a check on what arrived. No part of the model is decided here, so the
//! unfamiliar syntax carries none of it.
//!
//! * **`#[pyfunction]` and `#[pymodule]` are code generators.** They expand
//!   into the C-level functions CPython actually calls, converting each
//!   argument from a Python object into the Rust type named in the signature
//!   and converting the return value back. A `TypeError` for a wrong dtype
//!   comes from that generated code, not from anything written here.
//! * **`PyReadonlyArray1<'py, f64>` borrows NumPy's own buffer.** Nothing is
//!   copied and nothing is converted: the `f64` values Rust reads are the
//!   bytes NumPy already holds. That is what makes the draw array free to pass
//!   — it is the largest object in a run — and it is why the dtype has to
//!   match exactly rather than being coerced.
//! * **`Bound<'py, T>` is a reference to a Python object that holds the
//!   interpreter lock.** It is PyO3's smart pointer, and the `'py` lifetime
//!   is what stops a Python object being used after the lock is released.
//!   Older PyO3 examples use `&PyArray1` and `Py<...>` instead and will not
//!   compile against this version.
//! * **`Python<'py>` is a token proving the caller holds the lock.** Functions
//!   that need it take it as their first argument. Nothing in Python
//!   corresponds, because there the lock is always held.
//! * **The generic bounds `<T: Element, D: Dimension>` are duck typing checked
//!   at compile time.** `contiguous` works for any element type NumPy
//!   supports and any number of axes, and the compiler generates a separate
//!   specialised copy for each combination actually used.

mod policies;
mod simulate;
mod weibull;

use numpy::ndarray::{Array3, Dimension};
use numpy::PyUntypedArrayMethods;
use numpy::{
    Element, IntoPyArray, PyArray3, PyReadonlyArray, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Borrows a NumPy array as a flat slice, rejecting a non-contiguous one.
///
/// Every array crosses this boundary as the same bytes NumPy holds, rather
/// than as a copy, which is what removes cross-language divergence in the
/// draws entirely as opposed to testing for it. What can still go wrong is
/// layout: a transposed view or a strided slice arrives non-contiguous, and
/// reading it as a flat slice would take the wrong elements in the wrong
/// order. The caller that gets this wrong is a notebook passing `arr.T`, and
/// the symptom without this check is a plausible wrong number.
///
/// **C order is checked explicitly rather than left to `as_slice`.** That call
/// accepts a Fortran-ordered array too, since it is contiguous — just
/// column-major. A one-dimensional array is both, so the eleven per-segment
/// arrays cannot tell the difference; the draw arrays can, and a transposed
/// three-dimensional view of the right shape would be read with its axes
/// exchanged and return a run that completes. A strided slice is rejected
/// either way, which is what makes the weaker check look like it works.
///
/// # Arguments
///
/// * `name` - the argument's name, so the error says which one it was.
/// * `array` - the borrowed array.
///
/// # Returns
///
/// The array's elements in C order.
fn contiguous<'a, T: Element, D: Dimension>(
    name: &str,
    array: &'a PyReadonlyArray<'_, T, D>,
) -> PyResult<&'a [T]> {
    if !array.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} is not C-contiguous; pass the array itself rather than a \
             transposed view or a strided slice of it"
        )));
    }
    // `PyResult<T>` is `Result<T, PyErr>`: either the slice or a Python
    // exception, returned rather than raised. `map_err` replaces whatever
    // error the call produced with one carrying a message that names the
    // argument, since the original says only that the array was not
    // contiguous. Unreachable once the check above has passed.
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} could not be read as a flat slice")))
}

/// Checks that a per-segment array has one entry per segment.
///
/// # Arguments
///
/// * `name` - the argument's name.
/// * `actual` - the length it has.
/// * `n_segments` - the length every per-segment array must have.
///
/// # Errors
///
/// `ValueError` naming the argument and both lengths, if they differ. A
/// segment's array position is its `segment_id`, so an array of the wrong
/// length is not merely short — every entry after the first missing one names
/// a different segment than the caller meant.
fn one_entry_per_segment(name: &str, actual: usize, n_segments: usize) -> PyResult<()> {
    if actual == n_segments {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} has {actual} entries against {n_segments} segments; every \
             per-segment array is one entry per segment, ordered by segment_id"
        )))
    }
}

/// Reshapes one result buffer into the `(replications, years, classes)` array
/// the caller reads back.
///
/// # Arguments
///
/// * `py` - the Python token.
/// * `values` - the buffer, already `replications * years * classes` long.
/// * `shape` - that triple.
///
/// # Returns
///
/// The buffer as a three-dimensional NumPy array, without copying it.
fn as_result_array(
    py: Python<'_>,
    values: Vec<f64>,
    shape: (usize, usize, usize),
) -> Bound<'_, PyArray3<f64>> {
    // `into_pyarray` consumes the `Vec` and hands its memory to NumPy rather
    // than copying it — `into_` is the Rust naming convention for a conversion
    // that takes ownership. The `expect` cannot fire: the buffer was allocated
    // at exactly this size, so a mismatch would mean this file and
    // `simulate.rs` disagree about the shape.
    Array3::from_shape_vec(shape, values)
        .expect("every result buffer is allocated as replications * years * classes")
        .into_pyarray(py)
}

/// Runs one chunk of replications under one policy.
///
/// The arguments are `cablesim.simulate.run_chunk`'s, under the same names and
/// in the same order, so the reference and this kernel are interchangeable
/// behind a keyword call. `cablesim.kernel.run_chunk` is the Python wrapper
/// that calls this and rebuilds the reference's `Results` from what it returns.
///
/// # Arguments
///
/// See `simulate::run_chunk`, which these are passed straight through to.
/// `lifetime_uniforms` is `(replications, segments, n_years + 1)` and
/// `policy_uniforms` is `(replications, segments)`; every other array is one
/// entry per segment, ordered by `segment_id`, except `budget` and
/// `cost_escalation`, which are one entry per year.
///
/// # Returns
///
/// Seven `(replications, years, classes)` arrays, in the field order of
/// `cablesim.simulate.Results`: failures, customers interrupted, customer
/// minutes, planned customer minutes, planned replacements, planned spend and
/// emergency spend.
///
/// # Errors
///
/// `ValueError` if an array is not C-contiguous, if a per-segment or per-year
/// array is the wrong length, if the draw array is not the shape the horizon
/// implies, or if a candidate scores a rank key that is not a number.
#[pyfunction]
#[pyo3(signature = (
    length_ft, customers, customer_minutes_per_failure,
    customer_minutes_per_planned, outage_cost_per_failure, class_index,
    age0, shape, scale, replacement_shape, replacement_scale, cost_per_ft,
    lifetime_uniforms, policy_uniforms, budget, cost_escalation, policy,
    emergency_multiplier, mobilization_per_segment,
    emergency_charged_to_budget, n_classes, n_years,
))]
#[allow(clippy::too_many_arguments)]
fn run_chunk<'py>(
    py: Python<'py>,
    length_ft: PyReadonlyArray1<'py, f64>,
    customers: PyReadonlyArray1<'py, f64>,
    customer_minutes_per_failure: PyReadonlyArray1<'py, f64>,
    customer_minutes_per_planned: PyReadonlyArray1<'py, f64>,
    outage_cost_per_failure: PyReadonlyArray1<'py, f64>,
    class_index: PyReadonlyArray1<'py, u8>,
    age0: PyReadonlyArray1<'py, f64>,
    shape: PyReadonlyArray1<'py, f64>,
    scale: PyReadonlyArray1<'py, f64>,
    replacement_shape: PyReadonlyArray1<'py, f64>,
    replacement_scale: PyReadonlyArray1<'py, f64>,
    cost_per_ft: PyReadonlyArray1<'py, f64>,
    lifetime_uniforms: PyReadonlyArray3<'py, f64>,
    policy_uniforms: PyReadonlyArray2<'py, f64>,
    budget: PyReadonlyArray1<'py, f64>,
    cost_escalation: PyReadonlyArray1<'py, f64>,
    policy: policies::Resolved,
    emergency_multiplier: f64,
    mobilization_per_segment: f64,
    emergency_charged_to_budget: bool,
    n_classes: usize,
    n_years: usize,
) -> PyResult<Bound<'py, PyTuple>> {
    // An unrecognized tag would otherwise score every candidate zero and fund
    // them in segment_id order, which is a plausible-looking run rather than an
    // error. `policies.resolve` is the only thing that builds this struct, so
    // reaching here means the two sides disagree about the mapping.
    if !(policies::kind::RUN_TO_FAILURE..=policies::kind::RANDOM).contains(&policy.kind) {
        return Err(PyValueError::new_err(format!(
            "policy.kind is {}, which names no policy; the tags are authored \
             in cablesim.policies.KIND and run from {} to {}",
            policy.kind,
            policies::kind::RUN_TO_FAILURE,
            policies::kind::RANDOM
        )));
    }

    // Destructuring a slice into named parts, with the `else` branch required
    // because the compiler cannot know the length from the type alone. The
    // two-dimensional array type guarantees it, so the branch is unreachable.
    let [n_reps, n_segments] = *policy_uniforms.shape() else {
        unreachable!("a PyReadonlyArray2 has exactly two axes")
    };

    // The replication count is recovered by dividing by the segment count, so
    // an empty population divides by zero. Rust does not raise on that: it
    // panics, and PyO3 surfaces a panic as `PanicException`, which does not
    // inherit from `Exception` and so passes straight through a driver
    // script's error handling.
    if n_segments == 0 {
        return Err(PyValueError::new_err(
            "the population is empty; there is nothing to simulate",
        ));
    }
    if n_classes == 0 {
        return Err(PyValueError::new_err(
            "n_classes is 0, so the results have no class axis to accumulate into",
        ));
    }

    // The year axis is why this check exists: a short draw array raises on its
    // own only when a replacement happens to fall in the final year, so a run
    // can complete against the wrong array and be wrong nowhere visible. The
    // other two axes are checked with it because they cost nothing to compare.
    if lifetime_uniforms.shape() != [n_reps, n_segments, n_years + 1] {
        return Err(PyValueError::new_err(format!(
            "lifetime_uniforms is {:?}, expected {:?}: one draw per segment \
             per year, plus the left-truncated draw at index 0",
            lifetime_uniforms.shape(),
            [n_reps, n_segments, n_years + 1]
        )));
    }

    // A fixed-size array of name-and-length pairs, checked in one loop so that
    // adding a per-segment argument without checking it is a visible omission
    // rather than an invisible one. `[(&str, usize); 12]` is the type: twelve
    // tuples, a length the compiler enforces.
    let per_segment: [(&str, usize); 12] = [
        ("length_ft", length_ft.len()),
        ("customers", customers.len()),
        (
            "customer_minutes_per_failure",
            customer_minutes_per_failure.len(),
        ),
        (
            "customer_minutes_per_planned",
            customer_minutes_per_planned.len(),
        ),
        ("outage_cost_per_failure", outage_cost_per_failure.len()),
        ("class_index", class_index.len()),
        ("age0", age0.len()),
        ("shape", shape.len()),
        ("scale", scale.len()),
        ("replacement_shape", replacement_shape.len()),
        ("replacement_scale", replacement_scale.len()),
        ("cost_per_ft", cost_per_ft.len()),
    ];
    for (name, actual) in per_segment {
        one_entry_per_segment(name, actual, n_segments)?;
    }
    for (name, actual) in [
        ("budget", budget.len()),
        ("cost_escalation", cost_escalation.len()),
    ] {
        if actual != n_years {
            return Err(PyValueError::new_err(format!(
                "{name} has {actual} entries against a {n_years}-year horizon"
            )));
        }
    }

    // A class index past the end of the result axis indexes out of bounds and
    // panics, and it does so only once a segment of that class is selected in
    // some year — so the same mismatched arguments complete on one seed and
    // panic on another. The reference raises here instead, because NumPy
    // refuses the out-of-range bin.
    let classes = contiguous("class_index", &class_index)?;
    if let Some(&past_the_axis) = classes.iter().find(|&&c| usize::from(c) >= n_classes) {
        return Err(PyValueError::new_err(format!(
            "class_index holds {past_the_axis}, which is past the {n_classes} \
             classes the results have an axis for; the index is a position in \
             the class list, not a name"
        )));
    }

    let results = simulate::run_chunk(
        contiguous("length_ft", &length_ft)?,
        contiguous("customers", &customers)?,
        contiguous(
            "customer_minutes_per_failure",
            &customer_minutes_per_failure,
        )?,
        contiguous(
            "customer_minutes_per_planned",
            &customer_minutes_per_planned,
        )?,
        contiguous("outage_cost_per_failure", &outage_cost_per_failure)?,
        classes,
        contiguous("age0", &age0)?,
        contiguous("shape", &shape)?,
        contiguous("scale", &scale)?,
        contiguous("replacement_shape", &replacement_shape)?,
        contiguous("replacement_scale", &replacement_scale)?,
        contiguous("cost_per_ft", &cost_per_ft)?,
        contiguous("lifetime_uniforms", &lifetime_uniforms)?,
        contiguous("policy_uniforms", &policy_uniforms)?,
        contiguous("budget", &budget)?,
        contiguous("cost_escalation", &cost_escalation)?,
        policy,
        emergency_multiplier,
        mobilization_per_segment,
        emergency_charged_to_budget,
        n_classes,
        n_years,
    )
    .map_err(|unranked| PyValueError::new_err(unranked.to_string()))?;

    let dimensions = (n_reps, n_years, n_classes);
    PyTuple::new(
        py,
        [
            as_result_array(py, results.failures, dimensions).into_any(),
            as_result_array(py, results.customers_interrupted, dimensions).into_any(),
            as_result_array(py, results.customer_minutes, dimensions).into_any(),
            as_result_array(py, results.planned_customer_minutes, dimensions).into_any(),
            as_result_array(py, results.planned_replacements, dimensions).into_any(),
            as_result_array(py, results.planned_spend, dimensions).into_any(),
            as_result_array(py, results.emergency_spend, dimensions).into_any(),
        ],
    )
}

/// Registers the extension module's contents under the name `_cablesim`.
#[pymodule]
fn _cablesim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_chunk, m)?)?;
    // Which profile this was compiled with, so a run records what actually ran
    // rather than what the person starting it believed. `debug_assertions` is
    // on in a debug build and off in a release one, and it is resolved at
    // compile time, so this cannot disagree with the binary it ships in.
    // Measured on one thread at 2,000 segments, 20 replications and a 30-year
    // horizon under `risk_ranked`, a debug build takes 0.75 s against 0.12 s
    // for release — about six times — which is what makes an unlabelled
    // timing meaningless rather than merely imprecise.
    m.add(
        "BUILD_PROFILE",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    )?;
    Ok(())
}
