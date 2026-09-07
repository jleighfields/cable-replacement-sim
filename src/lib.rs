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
    array.as_slice().map_err(|_| {
        PyValueError::new_err(format!(
            "{name} is not C-contiguous; pass the array itself rather than a \
             transposed view or a strided slice of it"
        ))
    })
}

/// Checks that a per-segment array has one entry per segment.
///
/// # Arguments
///
/// * `name` - the argument's name.
/// * `actual` - the length it has.
/// * `n_segments` - the length every per-segment array must have.
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

    let [n_reps, n_segments] = *policy_uniforms.shape() else {
        unreachable!("a PyReadonlyArray2 has exactly two axes")
    };

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
        contiguous("class_index", &class_index)?,
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
    // compile time, so this cannot disagree with the binary it ships in. A
    // debug build is some tens of times slower here, which makes it the one
    // way a timing is quietly meaningless.
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
