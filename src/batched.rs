//! The annual loop over a polars frame, in Rust.
//!
//! The Rust side of a second mirror. `python/cablesim/batched.py` carries the
//! same loop written against the Python polars package, and this file exists to
//! separate two things that implementation cannot separate on its own.
//!
//! # Why this exists
//!
//! The Python polars package is not a reimplementation of anything: it is a
//! Python expression layer over the same compiled Rust query engine this file
//! calls directly. So when the frame implementation is slower than the array
//! one, the Python measurement alone cannot say whether the engine is the wrong
//! shape for this work or whether the cost is in reaching it — building
//! expression trees in Python, crossing into the engine, and returning. Running
//! the same algorithm against the same engine from Rust removes the second
//! term, and the difference between the two is what that term costs.
//!
//! # What is and is not a polars operation here
//!
//! The split is the same one `batched.py` makes, deliberately, so the two are
//! comparable:
//!
//! * **Expressions** carry the frame-shaped work — the year's failure test, the
//!   ranking sort, the cumulative sum within a replication that spends the
//!   budget, and the grouped totals.
//! * **Plain loops over extracted columns** carry the scalar mathematics: the
//!   Weibull draws and the policy score. `batched.py` reaches those through
//!   NumPy's ufunc protocol, which runs the identical elementwise formula over
//!   a column. Writing them as polars expressions instead would be a second
//!   copy of arithmetic that already exists in `weibull.rs` and `policies.rs`,
//!   and it would stop being the same split the Python side makes.
//!
//! # Reading this beside the Python
//!
//! * **`.over()` returns a `Result`.** Python's `.over("rep")` yields an
//!   expression; here it can fail, so the call needs `?` and cannot sit inside
//!   an expression array without being bound to a name first.
//! * **`cum_sum` takes its `reverse` argument positionally.** Python defaults
//!   it; Rust has no default arguments, so `cum_sum(false)` appears everywhere
//!   Python writes `cum_sum()`.
//! * **Sorting takes an options struct** rather than keyword arguments:
//!   `SortMultipleOptions` carries the per-column descending flags.
//! * **A `LazyFrame` is a plan, not data.** Nothing runs until `.collect()`,
//!   which is also true of Python's `pl.LazyFrame` — but here the eager
//!   `DataFrame` and the lazy plan are different types, so the conversion is
//!   visible at each step.

use polars::prelude::*;

use crate::policies::{self, Resolved};
use crate::simulate::{ChunkError, Results};
use crate::weibull;

/// Column names the year loop reads and writes, so a typo is a compile error
/// in one place rather than a runtime lookup failure in several.
mod column {
    /// Replication index.
    pub const REP: &str = "rep";
    /// Segment index within a replication, which is its identifier.
    pub const SEGMENT_ID: &str = "segment_id";
    /// Segment class, indexing the results' third axis.
    pub const CLASS: &str = "class_index";
    /// Customers served.
    pub const CUSTOMERS: &str = "customers";
    /// Customer-minutes lost when this segment fails.
    pub const MINUTES_FAILURE: &str = "customer_minutes_per_failure";
    /// Customer-minutes lost to planned work on this segment.
    pub const MINUTES_PLANNED: &str = "customer_minutes_per_planned";
    /// Value of lost load at year-0 prices.
    pub const OUTAGE_COST: &str = "outage_cost_per_failure";
    /// Planned replacement cost at year-0 prices.
    pub const PLANNED_AT_PAR: &str = "planned_at_par";
    /// Weibull shape a replacement would take.
    pub const REPLACEMENT_SHAPE: &str = "replacement_shape";
    /// Weibull scale a replacement would take.
    pub const REPLACEMENT_SCALE: &str = "replacement_scale";
    /// Current age, in years.
    pub const AGE: &str = "age";
    /// Weibull shape currently in the ground.
    pub const CURRENT_SHAPE: &str = "current_shape";
    /// Weibull scale currently in the ground.
    pub const CURRENT_SCALE: &str = "current_scale";
    /// Simulation time at which this segment next fails.
    pub const FAILURE_TIME: &str = "failure_time";
    /// The fixed per-segment priority the random policy ranks on.
    pub const PRIORITY: &str = "priority";
    /// Planned replacement cost at this year's prices.
    pub const PLANNED_NOW: &str = "planned_now";
    /// Whether this segment failed this year.
    pub const FAILED: &str = "failed";
    /// Whether this segment was funded as planned work this year.
    pub const FUNDED: &str = "funded";
    /// Whether this policy may fund this segment this year.
    pub const ELIGIBLE: &str = "eligible";
    /// The rank key, meaningful only where eligible.
    pub const RANK: &str = "rank";
    /// The rank key with ineligible segments pushed below every real one.
    pub const DEMOTED: &str = "demoted";
    /// What this segment adds to the running cost, infinite where ineligible.
    pub const STEP: &str = "step";
    /// Cumulative planned cost down this replication's ranked order.
    pub const RUNNING: &str = "running";
    /// What this replication has left to spend this year.
    pub const AVAILABLE: &str = "available";
}

/// Whether a segment was replaced this year, by failure or as planned work.
///
/// Outside the `column` module because it lives only inside one step: it is
/// added, read and dropped within the same year.
const REPLACED: &str = "replaced";

/// The seven reported quantities, in the order `Results` holds them.
const QUANTITIES: [&str; 7] = [
    "failures",
    "customers_interrupted",
    "customer_minutes",
    "planned_customer_minutes",
    "planned_replacements",
    "planned_spend",
    "emergency_spend",
];

/// Repeats a per-segment column once per replication.
///
/// The frame holds one row per replication and segment, ordered by replication
/// and then by segment, which is the order the draw arrays are laid out in.
///
/// # Arguments
///
/// * `values` - one entry per segment.
/// * `times` - how many replications to repeat them for.
fn tile(values: &[f64], times: usize) -> Vec<f64> {
    let mut repeated = Vec::with_capacity(values.len() * times);
    for _ in 0..times {
        repeated.extend_from_slice(values);
    }
    repeated
}

/// Copies a float column out as a plain vector.
///
/// The scalar mathematics runs over slices rather than over expressions, for
/// the reason the module comment gives, and this is the boundary between the
/// two. It copies: a column can be split across chunks, so there is not always
/// one contiguous slice to borrow.
///
/// # Arguments
///
/// * `frame` - the frame to read.
/// * `name` - the column to read.
fn column_values(frame: &DataFrame, name: &str) -> PolarsResult<Vec<f64>> {
    Ok(frame.column(name)?.f64()?.into_no_null_iter().collect())
}

/// Copies a boolean column out as a plain vector.
///
/// # Arguments
///
/// * `frame` - the frame to read.
/// * `name` - the column to read.
fn column_flags(frame: &DataFrame, name: &str) -> PolarsResult<Vec<bool>> {
    // `into_no_null_iter` is only for numeric columns; a boolean one yields
    // `Option<bool>` and the nulls are ruled out by construction here, since
    // every boolean column is built by a comparison over non-null values.
    Ok(frame
        .column(name)?
        .bool()?
        .iter()
        .map(|flag| flag.unwrap_or(false))
        .collect())
}

/// A quantity's contribution, zero where the row does not contribute.
///
/// # Arguments
///
/// * `flag` - the boolean column deciding whether the row contributes.
/// * `value` - what it contributes when it does.
fn contribution(flag: &str, value: Expr) -> Expr {
    when(col(flag)).then(value).otherwise(lit(0.0))
}

/// Totals each quantity per replication and per segment class.
///
/// **A cumulative sum's last element rather than a grouped sum.** A grouped sum
/// adds in whatever order the engine finds convenient, which differs from a
/// running total in the last bits and therefore differs from what the reference
/// reports. A cumulative sum is defined by its order, so it keeps one. This is
/// the same choice `batched.py` makes and for the same reason.
///
/// # Arguments
///
/// * `frame` - rows in the order the totals should accumulate: segment order
///   for what failures contribute, rank order for what funded work contributes.
/// * `quantities` - the reported quantity names, paired with the expression
///   giving each row's contribution.
fn totals_per_class(frame: LazyFrame, quantities: Vec<(&str, Expr)>) -> LazyFrame {
    let aggregated: Vec<Expr> = quantities
        .into_iter()
        .map(|(name, value)| value.cum_sum(false).last().alias(name))
        .collect();
    frame
        .group_by([col(column::REP), col(column::CLASS)])
        .agg(aggregated)
}

/// Runs one chunk of replications under one policy, over a polars frame.
///
/// The arguments are `simulate::run_chunk`'s and mean the same things. The
/// three numbered steps below are its three steps in the same order, with the
/// replication axis carried by the frame rather than by an outer loop.
///
/// # Arguments
///
/// See `simulate::run_chunk`. Every slice is C-contiguous, which the binding
/// checks before calling this.
///
/// # Returns
///
/// The seven per-year, per-class buffers for this chunk, or a `ChunkError` if a
/// candidate scored a key that was not a number or the engine refused a step.
#[allow(clippy::too_many_arguments)]
pub fn run_chunk(
    length_ft: &[f64],
    customers: &[f64],
    customer_minutes_per_failure: &[f64],
    customer_minutes_per_planned: &[f64],
    outage_cost_per_failure: &[f64],
    class_index: &[u8],
    age0: &[f64],
    shape: &[f64],
    scale: &[f64],
    replacement_shape: &[f64],
    replacement_scale: &[f64],
    cost_per_ft: &[f64],
    lifetime_uniforms: &[f64],
    policy_uniforms: &[f64],
    budget: &[f64],
    cost_escalation: &[f64],
    policy: Resolved,
    emergency_multiplier: f64,
    mobilization_per_segment: f64,
    emergency_charged_to_budget: bool,
    n_classes: usize,
    n_years: usize,
) -> Result<Results, ChunkError> {
    let n_segments = age0.len();
    let n_reps = policy_uniforms.len() / n_segments;
    let draws_per_segment = n_years + 1;

    let planned_at_par: Vec<f64> = (0..n_segments)
        .map(|segment| {
            policies::planned_cost(
                length_ft[segment],
                cost_per_ft[segment],
                mobilization_per_segment,
            )
        })
        .collect();
    // The uniforms for the left-truncated draw are the first of each segment's
    // row, so they are gathered rather than sliced.
    let first_uniforms: Vec<f64> = (0..n_reps * n_segments)
        .map(|row| lifetime_uniforms[row * draws_per_segment])
        .collect();

    let ages = tile(age0, n_reps);
    let shapes = tile(shape, n_reps);
    let scales = tile(scale, n_reps);
    // Conditional on survival to the starting age, drawn by the same function
    // the reference draws it with.
    let failure_time: Vec<f64> = (0..n_reps * n_segments)
        .map(|row| {
            weibull::draw_remaining_life(first_uniforms[row], ages[row], shapes[row], scales[row])
        })
        .collect();

    let classes: Vec<i32> = class_index.iter().map(|&class| i32::from(class)).collect();
    let mut state = df![
        column::REP => (0..n_reps)
            .flat_map(|replication| std::iter::repeat_n(replication as i32, n_segments))
            .collect::<Vec<i32>>(),
        column::SEGMENT_ID => (0..n_reps)
            .flat_map(|_| 0..n_segments as i32)
            .collect::<Vec<i32>>(),
        column::CLASS => (0..n_reps)
            .flat_map(|_| classes.iter().copied())
            .collect::<Vec<i32>>(),
        column::CUSTOMERS => tile(customers, n_reps),
        column::MINUTES_FAILURE => tile(customer_minutes_per_failure, n_reps),
        column::MINUTES_PLANNED => tile(customer_minutes_per_planned, n_reps),
        column::OUTAGE_COST => tile(outage_cost_per_failure, n_reps),
        column::PLANNED_AT_PAR => tile(&planned_at_par, n_reps),
        column::REPLACEMENT_SHAPE => tile(replacement_shape, n_reps),
        column::REPLACEMENT_SCALE => tile(replacement_scale, n_reps),
        column::AGE => ages,
        column::CURRENT_SHAPE => shapes,
        column::CURRENT_SCALE => scales,
        column::PRIORITY => policy_uniforms.to_vec(),
        column::FAILURE_TIME => failure_time,
    ]
    .map_err(ChunkError::Polars)?;

    let mut yearly: Vec<DataFrame> = Vec::with_capacity(n_years);
    for year in 0..n_years {
        let escalation = cost_escalation[year];
        let year_start = year as f64;

        // 1. Failures, resolved before planned work so that a segment failing
        //    this year is not also a candidate this year.
        state = state
            .clone()
            .lazy()
            .with_column((col(column::PLANNED_AT_PAR) * lit(escalation)).alias(column::PLANNED_NOW))
            .with_column(
                col(column::FAILURE_TIME)
                    .gt_eq(lit(year_start))
                    .and(col(column::FAILURE_TIME).lt(lit(year_start + 1.0)))
                    .alias(column::FAILED),
            )
            .collect()
            .map_err(ChunkError::Polars)?;

        let emergency_now = contribution(
            column::FAILED,
            col(column::PLANNED_NOW) * lit(emergency_multiplier),
        );
        let failures = totals_per_class(
            state.clone().lazy(),
            vec![
                (QUANTITIES[0], contribution(column::FAILED, lit(1.0))),
                (
                    QUANTITIES[1],
                    contribution(column::FAILED, col(column::CUSTOMERS)),
                ),
                (
                    QUANTITIES[2],
                    contribution(column::FAILED, col(column::MINUTES_FAILURE)),
                ),
                (QUANTITIES[6], emergency_now.clone()),
            ],
        );

        // 2. Planned replacement, funded greedily down the ranked order.
        let mut available = df![
            column::REP => (0..n_reps as i32).collect::<Vec<i32>>(),
            column::AVAILABLE => vec![budget[year]; n_reps],
        ]
        .map_err(ChunkError::Polars)?;
        if emergency_charged_to_budget {
            // Accumulated one row at a time within a replication, in segment
            // order, which is the order the reference adds it in. This total is
            // subtracted from the budget the greedy fill then compares a
            // cumulative cost against, so a last-bit difference here decides
            // which segment is funded last — a discrete outcome rather than a
            // rounding difference.
            //
            // A year whose failures cost more than the budget leaves this
            // negative, and it is left negative: every planned cost is
            // positive, so nothing is funded at or below zero and nothing
            // carries into the next year.
            let charged = emergency_now
                .cum_sum(false)
                .over([col(column::REP)])
                .map_err(ChunkError::Polars)?
                .last()
                .alias("charge");
            available = state
                .clone()
                .lazy()
                .group_by([col(column::REP)])
                .agg([charged])
                .join(
                    available.lazy(),
                    [col(column::REP)],
                    [col(column::REP)],
                    JoinArgs::new(JoinType::Inner),
                )
                .select([
                    col(column::REP),
                    (col(column::AVAILABLE) - col("charge")).alias(column::AVAILABLE),
                ])
                .collect()
                .map_err(ChunkError::Polars)?;
        }

        let age = column_values(&state, column::AGE).map_err(ChunkError::Polars)?;
        let failed = column_flags(&state, column::FAILED).map_err(ChunkError::Polars)?;
        let eligible: Vec<bool> = (0..age.len())
            .map(|row| policies::eligible(policy, age[row], failed[row]))
            .collect();

        let planned = if eligible.iter().any(|&candidate| candidate) {
            let current_shape =
                column_values(&state, column::CURRENT_SHAPE).map_err(ChunkError::Polars)?;
            let current_scale =
                column_values(&state, column::CURRENT_SCALE).map_err(ChunkError::Polars)?;
            let outage = column_values(&state, column::OUTAGE_COST).map_err(ChunkError::Polars)?;
            let planned_now =
                column_values(&state, column::PLANNED_NOW).map_err(ChunkError::Polars)?;
            let priority = column_values(&state, column::PRIORITY).map_err(ChunkError::Polars)?;
            let rank: Vec<f64> = (0..age.len())
                .map(|row| {
                    policies::rank_key(
                        policy,
                        age[row],
                        weibull::conditional_failure_probability(
                            age[row],
                            current_shape[row],
                            current_scale[row],
                        ),
                        outage[row] * escalation,
                        planned_now[row],
                        emergency_multiplier,
                        priority[row],
                    )
                })
                .collect();
            if let Some(row) = (0..rank.len()).find(|&row| eligible[row] && rank[row].is_nan()) {
                // Reported the way the reference reports it, since a caller
                // running both must not get a different diagnosis from each.
                let unranked = (0..rank.len())
                    .filter(|&other| eligible[other] && rank[other].is_nan())
                    .count();
                return Err(ChunkError::NanScore(policies::NanScore {
                    count: unranked,
                    first_segment_id: row % n_segments,
                }));
            }

            // Ineligible segments are demoted below every real rank key and
            // given an infinite cost, so the sort puts them behind the
            // candidates and the cumulative cost cannot run past the last real
            // one however much budget is left.
            let ranked = state
                .clone()
                .lazy()
                .with_columns([
                    lit(Series::new(column::ELIGIBLE.into(), eligible.clone())),
                    lit(Series::new(column::RANK.into(), rank)),
                ])
                .with_columns([
                    when(col(column::ELIGIBLE))
                        .then(col(column::RANK))
                        .otherwise(lit(f64::NEG_INFINITY))
                        .alias(column::DEMOTED),
                    when(col(column::ELIGIBLE))
                        .then(col(column::PLANNED_NOW))
                        .otherwise(lit(f64::INFINITY))
                        .alias(column::STEP),
                ])
                .sort(
                    [column::REP, column::DEMOTED, column::SEGMENT_ID],
                    SortMultipleOptions::default()
                        .with_order_descending_multi([false, true, false]),
                );
            let running = col(column::STEP)
                .cum_sum(false)
                .over([col(column::REP)])
                .map_err(ChunkError::Polars)?
                .alias(column::RUNNING);
            let ranked = ranked
                .with_column(running)
                .join(
                    available.lazy(),
                    [col(column::REP)],
                    [col(column::REP)],
                    JoinArgs::new(JoinType::Inner),
                )
                .with_column(
                    col(column::RUNNING)
                        .lt_eq(col(column::AVAILABLE))
                        .alias(column::FUNDED),
                )
                .collect()
                .map_err(ChunkError::Polars)?;

            // Totalled here, while the rows are still in rank order, because
            // that is the order the reference adds the funded segments in.
            let planned = totals_per_class(
                ranked.clone().lazy(),
                vec![
                    (
                        QUANTITIES[3],
                        contribution(column::FUNDED, col(column::MINUTES_PLANNED)),
                    ),
                    (QUANTITIES[4], contribution(column::FUNDED, lit(1.0))),
                    (
                        QUANTITIES[5],
                        contribution(column::FUNDED, col(column::PLANNED_NOW)),
                    ),
                ],
            );
            // Back to the canonical order, which every later year's uniforms
            // are attached by position.
            state = ranked
                .lazy()
                .sort(
                    [column::REP, column::SEGMENT_ID],
                    SortMultipleOptions::default(),
                )
                .drop(by_name(
                    [
                        column::ELIGIBLE,
                        column::RANK,
                        column::DEMOTED,
                        column::STEP,
                        column::RUNNING,
                        column::AVAILABLE,
                    ],
                    true,
                    false,
                ))
                .collect()
                .map_err(ChunkError::Polars)?;
            planned
        } else {
            state = state
                .clone()
                .lazy()
                .with_column(lit(false).alias(column::FUNDED))
                .collect()
                .map_err(ChunkError::Polars)?;
            totals_per_class(
                state.clone().lazy().filter(lit(false)),
                vec![
                    (QUANTITIES[3], lit(0.0)),
                    (QUANTITIES[4], lit(0.0)),
                    (QUANTITIES[5], lit(0.0)),
                ],
            )
        };

        yearly.push(
            failures
                .join(
                    planned,
                    [col(column::REP), col(column::CLASS)],
                    [col(column::REP), col(column::CLASS)],
                    JoinArgs::new(JoinType::Left),
                )
                .with_columns([
                    col(QUANTITIES[3]).fill_null(lit(0.0)),
                    col(QUANTITIES[4]).fill_null(lit(0.0)),
                    col(QUANTITIES[5]).fill_null(lit(0.0)),
                ])
                .collect()
                .map_err(ChunkError::Polars)?,
        );

        // 3. Everything replaced this year enters service next year, as new
        //    cable of the replacement technology.
        let funded = column_flags(&state, column::FUNDED).map_err(ChunkError::Polars)?;
        let failed = column_flags(&state, column::FAILED).map_err(ChunkError::Polars)?;
        let replaced: Vec<bool> = (0..funded.len())
            .map(|row| funded[row] || failed[row])
            .collect();
        let replacement_shapes =
            column_values(&state, column::REPLACEMENT_SHAPE).map_err(ChunkError::Polars)?;
        let replacement_scales =
            column_values(&state, column::REPLACEMENT_SCALE).map_err(ChunkError::Polars)?;
        let renewed: Vec<f64> = (0..replaced.len())
            .map(|row| {
                let segment = row % n_segments;
                let replication = row / n_segments;
                let draw = lifetime_uniforms
                    [(replication * n_segments + segment) * draws_per_segment + year + 1];
                (year + 1) as f64
                    + weibull::draw_lifetime(draw, replacement_shapes[row], replacement_scales[row])
            })
            .collect();

        state = state
            .clone()
            .lazy()
            .with_column(lit(Series::new(REPLACED.into(), replaced)))
            .with_columns([
                when(col(REPLACED))
                    .then(col(column::REPLACEMENT_SHAPE))
                    .otherwise(col(column::CURRENT_SHAPE))
                    .alias(column::CURRENT_SHAPE),
                when(col(REPLACED))
                    .then(col(column::REPLACEMENT_SCALE))
                    .otherwise(col(column::CURRENT_SCALE))
                    .alias(column::CURRENT_SCALE),
                when(col(REPLACED))
                    .then(lit(Series::new(column::FAILURE_TIME.into(), renewed)))
                    .otherwise(col(column::FAILURE_TIME))
                    .alias(column::FAILURE_TIME),
                when(col(REPLACED))
                    .then(lit(0.0))
                    .otherwise(col(column::AGE) + lit(1.0))
                    .alias(column::AGE),
            ])
            .drop(by_name(
                [
                    column::FAILED,
                    column::FUNDED,
                    REPLACED,
                    column::PLANNED_NOW,
                ],
                true,
                false,
            ))
            .collect()
            .map_err(ChunkError::Polars)?;
    }

    scatter(yearly, n_reps, n_years, n_classes).map_err(ChunkError::Polars)
}

/// Scatters the per-year totals into the seven reported buffers.
///
/// Done once at the end rather than per year, so the year loop holds only frame
/// work. A replication and class that contributed nothing in a year has no row
/// and reads back as the zero the reference reports.
///
/// # Arguments
///
/// * `yearly` - one frame per year, each with a replication column, a class
///   column and a column per quantity.
/// * `n_reps` - replications in this chunk.
/// * `n_years` - horizon, in years.
/// * `n_classes` - number of segment classes.
fn scatter(
    yearly: Vec<DataFrame>,
    n_reps: usize,
    n_years: usize,
    n_classes: usize,
) -> PolarsResult<Results> {
    let mut results = Results::zeros(n_reps, n_years, n_classes);
    let buffers: [&mut Vec<f64>; 7] = [
        &mut results.failures,
        &mut results.customers_interrupted,
        &mut results.customer_minutes,
        &mut results.planned_customer_minutes,
        &mut results.planned_replacements,
        &mut results.planned_spend,
        &mut results.emergency_spend,
    ];
    for (year, frame) in yearly.into_iter().enumerate() {
        let reps: Vec<i32> = frame
            .column(column::REP)?
            .i32()?
            .into_no_null_iter()
            .collect();
        let classes: Vec<i32> = frame
            .column(column::CLASS)?
            .i32()?
            .into_no_null_iter()
            .collect();
        for (index, name) in QUANTITIES.iter().enumerate() {
            let values: Vec<f64> = frame.column(name)?.f64()?.into_no_null_iter().collect();
            for row in 0..values.len() {
                let cell =
                    (reps[row] as usize * n_years + year) * n_classes + classes[row] as usize;
                buffers[index][cell] = values[row];
            }
        }
    }
    Ok(results)
}
