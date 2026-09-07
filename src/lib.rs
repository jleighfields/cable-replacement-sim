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
//!   bytes NumPy already holds. That is why the dtype has to match exactly
//!   rather than being coerced — a coercion would allocate a converted copy of
//!   every per-segment array on every call.
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

mod draws;
mod policies;
mod simulate;
mod weibull;

use numpy::ndarray::{Array3, Dimension};
use numpy::PyUntypedArrayMethods;
use numpy::{Element, IntoPyArray, PyArray1, PyArray3, PyReadonlyArray, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Borrows a NumPy array as a flat slice, rejecting a non-contiguous one.
///
/// Every array crosses this boundary as the same bytes NumPy holds, rather
/// than as a copy. What can still go wrong is layout: a strided slice arrives
/// non-contiguous, and reading it as a flat slice would take the wrong
/// elements in the wrong order. The caller that gets this wrong passes a step
/// slice such as `age[::2]`, and the symptom without this check is a plausible
/// wrong number rather than an error.
///
/// **C order is checked explicitly rather than left to `as_slice`.** That call
/// accepts a Fortran-ordered array too, since it is contiguous — just
/// column-major. Every array reaching this today is one-dimensional and so is
/// both at once, which makes the two checks indistinguishable on current
/// callers; the stricter one is kept because it is what an argument of two or
/// more axes would need, and adding one must not silently relax the check.
/// `contiguous` is generic over the number of axes for the same reason.
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
/// Every array is one entry per segment, ordered by `segment_id`, except
/// `budget` and `cost_escalation`, which are one entry per year.
///
/// **No draws are passed in.** `draw_key` is the two key words every uniform is
/// computed under, and `first_replication` says where this chunk sits in the
/// run, because a draw is addressed by its position rather than read from a
/// stream. `n_reps` is explicit for the same reason: there is no longer an
/// array whose shape it could be recovered from.
///
/// `threads` spreads the replications over that many workers and changes no
/// number: each replication reads its own slice of the draws and writes its own
/// block of the results. One thread runs them in sequence with no pool built.
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
/// `ValueError` if the policy tag names no policy, if the population is empty,
/// if `n_classes` is 0, if a class index is past the end of the class axis, if
/// an array is not C-contiguous, if a per-segment or per-year array is the
/// wrong length, if a replication, segment or year this chunk would draw at is
/// past what a draw index can carry, if `threads` is 0, or if a candidate
/// scores a rank key that is not a number.
///
/// `RuntimeError` if a thread pool of the requested size could not be built,
/// which is the operating system refusing to start the threads rather than
/// anything about the arguments.
#[pyfunction]
#[pyo3(signature = (
    length_ft, customers, customer_minutes_per_failure,
    customer_minutes_per_planned, outage_cost_per_failure, class_index,
    age0, shape, scale, replacement_shape, replacement_scale, cost_per_ft,
    draw_key, first_replication, n_reps, budget, cost_escalation, policy,
    emergency_multiplier, mobilization_per_segment,
    emergency_charged_to_budget, n_classes, n_years, threads=1,
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
    draw_key: (u64, u64),
    first_replication: u64,
    n_reps: usize,
    budget: PyReadonlyArray1<'py, f64>,
    cost_escalation: PyReadonlyArray1<'py, f64>,
    policy: policies::Resolved,
    emergency_multiplier: f64,
    mobilization_per_segment: f64,
    emergency_charged_to_budget: bool,
    n_classes: usize,
    n_years: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyTuple>> {
    // An unrecognized tag would otherwise score every candidate zero and fund
    // them in segment_id order, which is a plausible-looking run rather than an
    // error. `policies.resolve` is the only thing that builds this struct, so
    // reaching here means the two sides disagree about the mapping.
    //
    // The five tags are listed rather than checked against a range, because a
    // range's upper bound is whichever policy currently happens to be last: a
    // sixth policy added to the Python mapping would fall inside it and be
    // accepted here before `rank_key` had a branch for it. `policies.RANKABLE`
    // enumerates the same five on the other side for the same reason, so a
    // policy added to one language and not the other is refused by both until
    // both are edited.
    if !matches!(
        policy.kind,
        policies::kind::RUN_TO_FAILURE
            | policies::kind::AGE_THRESHOLD
            | policies::kind::RISK_RANKED
            | policies::kind::WORST_FIRST
            | policies::kind::RANDOM
    ) {
        return Err(PyValueError::new_err(format!(
            "policy.kind is {}, which no ranking branch covers; the tags are \
             authored in cablesim.policies.KIND and the ones that can be \
             scored are {:?}",
            policy.kind,
            [
                policies::kind::RUN_TO_FAILURE,
                policies::kind::AGE_THRESHOLD,
                policies::kind::RISK_RANKED,
                policies::kind::WORST_FIRST,
                policies::kind::RANDOM
            ]
        )));
    }

    if n_reps == 0 {
        return Err(PyValueError::new_err(
            "n_reps is 0, so no replication would run; a chunk covering none of \
             them is a caller's arithmetic gone wrong rather than an empty \
             result",
        ));
    }

    let n_segments = age0.len();

    // Reaching the loop with no segments divides by zero or indexes past the
    // end of a buffer. Rust does not raise on that: it panics, and PyO3
    // surfaces a panic as `PanicException`, which does not inherit from
    // `Exception` and so passes straight through a driver script's error
    // handling.
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

    // Every position this chunk will draw at has to be one the index can carry.
    // Past a field's width two positions would share a draw, which is a
    // correlation nothing downstream could detect. The year passed is `n_years`
    // rather than `n_years - 1`: a segment replaced in the final year draws its
    // next lifetime from the year it would have entered service on.
    within_the_index(
        first_replication + n_reps as u64 - 1,
        n_segments as u64 - 1,
        n_years as u64,
    )?;

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
    // panic on another. Checking every entry here refuses them before any year
    // runs, which is what makes the failure independent of the draws.
    //
    // The reference is not equivalent on this input. `numpy.bincount` grows
    // its output to fit the largest index it is given rather than refusing it,
    // so the reference raises only in a year where a segment of that class is
    // selected — the mismatched totals will not broadcast onto the class axis
    // — and it completes silently in every other year.
    let classes = contiguous("class_index", &class_index)?;
    if let Some(&past_the_axis) = classes.iter().find(|&&c| usize::from(c) >= n_classes) {
        return Err(PyValueError::new_err(format!(
            "class_index holds {past_the_axis}, which is past the {n_classes} \
             classes the results have an axis for; the index is a position in \
             the class list, not a name"
        )));
    }

    if threads == 0 {
        return Err(PyValueError::new_err(
            "threads is 0, so no replication would run; 1 is the sequential \
             path and the baseline a parallel run is measured against",
        ));
    }
    // Every borrow is taken here, while the interpreter lock is still held,
    // because each one reads the NumPy object's own metadata. What crosses into
    // the released region below is plain slices of `f64` and `u8`.
    let length_ft = contiguous("length_ft", &length_ft)?;
    let customers = contiguous("customers", &customers)?;
    let customer_minutes_per_failure = contiguous(
        "customer_minutes_per_failure",
        &customer_minutes_per_failure,
    )?;
    let customer_minutes_per_planned = contiguous(
        "customer_minutes_per_planned",
        &customer_minutes_per_planned,
    )?;
    let outage_cost_per_failure = contiguous("outage_cost_per_failure", &outage_cost_per_failure)?;
    let age0 = contiguous("age0", &age0)?;
    let shape = contiguous("shape", &shape)?;
    let scale = contiguous("scale", &scale)?;
    let replacement_shape = contiguous("replacement_shape", &replacement_shape)?;
    let replacement_scale = contiguous("replacement_scale", &replacement_scale)?;
    let cost_per_ft = contiguous("cost_per_ft", &cost_per_ft)?;
    let budget = contiguous("budget", &budget)?;
    let cost_escalation = contiguous("cost_escalation", &cost_escalation)?;

    // `detach` releases the interpreter lock for the whole computation and
    // takes it back when the closure returns. Two things make that safe, and
    // both are checked when this compiles rather than trusted: nothing inside
    // touches a Python object — the arrays became plain slices above — and
    // `Python<'py>` is not available in there, so code that needed the lock
    // could not be written without the compiler rejecting it.
    //
    // Without this, rayon's workers below would each wait for the lock and the
    // extra threads would buy nothing. It also lets an unrelated Python thread
    // run while a chunk is in flight, which is what keeps an application
    // responsive while a run is going.
    //
    // This method was called `allow_threads` until PyO3 renamed it; examples
    // found elsewhere still use that name, and it now compiles with a
    // deprecation warning rather than failing outright.
    let results = py
        .detach(|| {
            {
                simulate::run_chunk(
                    length_ft,
                    customers,
                    customer_minutes_per_failure,
                    customer_minutes_per_planned,
                    outage_cost_per_failure,
                    classes,
                    age0,
                    shape,
                    scale,
                    replacement_shape,
                    replacement_scale,
                    cost_per_ft,
                    [draw_key.0, draw_key.1],
                    first_replication,
                    n_reps,
                    budget,
                    cost_escalation,
                    policy,
                    emergency_multiplier,
                    mobilization_per_segment,
                    emergency_charged_to_budget,
                    n_classes,
                    n_years,
                    threads,
                )
            }
        })
        .map_err(|failure| match failure {
            simulate::ChunkError::NanScore(_) => PyValueError::new_err(failure.to_string()),
            // The operating system refusing to start threads, rather than
            // anything wrong with the arguments, so not a `ValueError`.
            simulate::ChunkError::ThreadPool(_) => PyRuntimeError::new_err(failure.to_string()),
        })?;

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

/// How many threads this machine can run at once.
///
/// Read here rather than in Python so that the number a run records and the
/// number rayon would pick come from one place. `available_parallelism` honours
/// a CPU affinity mask and a container's CPU quota, which a bare processor
/// count does not, and it is the same call rayon's own default is built on.
///
/// # Returns
///
/// The available parallelism, or 1 where the platform will not say.
#[pyfunction]
fn available_threads() -> usize {
    std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
}

/// Uniforms from the counter-based generator, one per position asked for.
///
/// Exposed so that the Python side can be held to producing the identical
/// numbers. The reference implementation and the batched loops still take their
/// draws as arrays, so both languages have to agree on every bit of every draw;
/// checking that against NumPy's own Philox is what establishes it, and this is
/// what the check calls.
///
/// # Arguments
///
/// * `key_low` - low word of the key, derived from the run's seed.
/// * `key_high` - high word of the key.
/// * `start` - the first position in the stream to produce.
/// * `count` - how many consecutive positions to produce.
///
/// # Returns
///
/// `count` doubles in `[0, 1)`.
#[pyfunction]
fn philox_uniforms(
    py: Python<'_>,
    key_low: u64,
    key_high: u64,
    start: u64,
    count: usize,
) -> Bound<'_, PyArray1<f64>> {
    // Released for the same reason the simulation releases it: nothing in here
    // touches a Python object, and a caller asking for millions of draws should
    // not hold the interpreter while they are computed.
    let values: Vec<f64> = py.detach(|| {
        (0..count as u64)
            .map(|offset| draws::uniform_at(start + offset, [key_low, key_high]))
            .collect()
    });
    values.into_pyarray(py)
}

/// Uniforms for a block of the simulation, every replication and segment.
///
/// The dense case: the left-truncated draw every segment takes at the start of
/// a run, and the fixed per-segment priority the random policy ranks on. Both
/// are read for every segment of every replication, so there is nothing to
/// select and the positions are known from the shape alone.
///
/// # Arguments
///
/// * `key_low` - low word of the key, derived from the run's seed.
/// * `key_high` - high word of the key.
/// * `purpose` - which stream, keeping unrelated draws independent.
/// * `first_replication` - the replication this chunk starts at, so a chunk
///   draws the same numbers wherever it sits in a run.
/// * `n_reps` - replications in this chunk.
/// * `n_segments` - segments in the population.
/// * `year` - the year these draws belong to.
///
/// # Returns
///
/// `n_reps * n_segments` doubles, replication-major.
///
/// # Errors
///
/// `ValueError` if any position would not fit the index, which is a run larger
/// than the packing allows rather than anything about this call.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn uniforms_dense<'py>(
    py: Python<'py>,
    key_low: u64,
    key_high: u64,
    purpose: u64,
    first_replication: u64,
    n_reps: usize,
    n_segments: usize,
    year: u64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    within_the_index(first_replication + n_reps as u64, n_segments as u64, year)?;
    let key = [key_low, key_high];
    let values: Vec<f64> = py.detach(|| {
        (0..n_reps as u64)
            .flat_map(|offset| {
                let replication = first_replication + offset;
                (0..n_segments as u64).map(move |segment| {
                    draws::uniform_at(draws::index(purpose, replication, segment, year), key)
                })
            })
            .collect()
    });
    Ok(values.into_pyarray(py))
}

/// Uniforms at named positions, for the segments that actually need one.
///
/// The sparse case, and the reason the generator is indexed at all: a year's
/// replacement draw is consumed only by a segment replaced that year, which is
/// a few percent of them. A stream would have to produce the rest anyway to
/// keep its position; this produces what is asked for and nothing else.
///
/// # Arguments
///
/// * `key_low` - low word of the key, derived from the run's seed.
/// * `key_high` - high word of the key.
/// * `purpose` - which stream.
/// * `replications` - one entry per wanted draw.
/// * `segments` - the matching segment of each.
/// * `year` - the year these draws belong to.
///
/// # Returns
///
/// One double per position, in the order given.
///
/// # Errors
///
/// `ValueError` if the two position arrays are different lengths, or if a
/// position would not fit the index.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn uniforms_at<'py>(
    py: Python<'py>,
    key_low: u64,
    key_high: u64,
    purpose: u64,
    replications: PyReadonlyArray1<'py, u32>,
    segments: PyReadonlyArray1<'py, u32>,
    year: u64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let replications = contiguous("replications", &replications)?;
    let segments = contiguous("segments", &segments)?;
    if replications.len() != segments.len() {
        return Err(PyValueError::new_err(format!(
            "replications has {} entries against {} segments; a draw is named \
             by both, so they pair up one for one",
            replications.len(),
            segments.len()
        )));
    }
    within_the_index(
        u64::from(replications.iter().copied().max().unwrap_or(0)),
        u64::from(segments.iter().copied().max().unwrap_or(0)),
        year,
    )?;
    let key = [key_low, key_high];
    let values: Vec<f64> = py.detach(|| {
        (0..replications.len())
            .map(|row| {
                draws::uniform_at(
                    draws::index(
                        purpose,
                        u64::from(replications[row]),
                        u64::from(segments[row]),
                        year,
                    ),
                    key,
                )
            })
            .collect()
    });
    Ok(values.into_pyarray(py))
}

/// Refuses a position the index cannot represent.
///
/// The packing gives each field a fixed width, so a run past one of them would
/// silently alias two positions onto the same draw — a correlation nothing
/// downstream could detect. The shipped run is nowhere near any of the bounds.
///
/// # Arguments
///
/// * `replication` - the largest replication in this call.
/// * `segment` - the largest segment.
/// * `year` - the year.
fn within_the_index(replication: u64, segment: u64, year: u64) -> PyResult<()> {
    for (name, value, limit) in [
        ("replication", replication, draws::MAX_REPLICATION),
        ("segment", segment, draws::MAX_SEGMENT),
        ("year", year, draws::MAX_YEAR),
    ] {
        if value > limit {
            return Err(PyValueError::new_err(format!(
                "{name} {value} is past the {limit} a draw index can carry; \
                 beyond it two positions would share one draw"
            )));
        }
    }
    Ok(())
}

/// Registers the extension module's contents under the name `_cablesim`.
#[pymodule]
fn _cablesim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_chunk, m)?)?;
    m.add_function(wrap_pyfunction!(available_threads, m)?)?;
    m.add_function(wrap_pyfunction!(philox_uniforms, m)?)?;
    m.add_function(wrap_pyfunction!(uniforms_dense, m)?)?;
    m.add_function(wrap_pyfunction!(uniforms_at, m)?)?;
    // The engine names are authored in this crate and read on the Python side,
    // so the two cannot drift into disagreeing about what a name means.
    // The draw index's field widths, so the Python side packs a position the
    // same way this crate does rather than repeating the numbers. A packing
    // written twice would diverge silently: two positions would share a draw,
    // and nothing downstream could tell.
    m.add(
        "DRAW_INDEX_LIMITS",
        (draws::MAX_REPLICATION, draws::MAX_SEGMENT, draws::MAX_YEAR),
    )?;
    m.add(
        "DRAW_INDEX_SHIFTS",
        (
            draws::PURPOSE_SHIFT,
            draws::REPLICATION_SHIFT,
            draws::YEAR_SHIFT,
            draws::SEGMENT_SHIFT,
        ),
    )?;
    m.add(
        "DRAW_PURPOSES",
        (
            ("lifetimes", draws::purpose::LIFETIMES),
            ("policies", draws::purpose::POLICIES),
            ("population", draws::purpose::POPULATION),
            ("records", draws::purpose::RECORDS),
        ),
    )?;
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
