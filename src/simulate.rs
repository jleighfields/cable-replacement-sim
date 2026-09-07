//! The annual simulation loop, one replication at a time.
//!
//! The Rust side of the mirror: `python/cablesim/simulate.py` is the
//! correctness reference, written to be checkable by reading, and this module
//! is the same loop with the same three numbered steps in the same order. When
//! the two disagree, the reference arbitrates.
//!
//! Two conventions are load-bearing, and getting either wrong leaves a run that
//! completes with plausible-looking curves:
//!
//! **Failure times are simulation time measured from year 0, never ages.** A
//! segment starting at `age0` with a drawn age-at-failure `T` fails at
//! `T - age0`. The loop compares against year boundaries, so holding an age
//! instead would be wrong by `age0`, which ranges over decades here.
//!
//! **A replacement enters service at the start of the following year.** A
//! segment replaced in year `y` is age 0 when year `y + 1` is scored. The age
//! it carries for the rest of year `y` is never read: it cannot fail again,
//! because its next failure time is at least `y + 1`, and it cannot be planned
//! work, because it has already been replaced this year. That is what keeps the
//! year loop free of any inner iteration, and what makes the deterministic
//! parity test terminate when lifetimes are forced to zero.

use crate::policies::{self, Resolved};
use crate::weibull;

/// One chunk of replications, per year and per segment class.
///
/// Every buffer holds `replications * years * classes` values, indexed
/// `(replication, year, class)` with class varying fastest, which is C order —
/// the layout NumPy reads them back in. The class axis is kept rather than
/// summed away because failures are read by class and a system total cannot be
/// decomposed afterwards.
///
/// The field names and their order match `simulate.Results`, the Python
/// NamedTuple the binding's caller rebuilds from these buffers.
pub struct Results {
    /// Segments that failed.
    pub failures: Vec<f64>,
    /// Customers out, counted per failure.
    pub customers_interrupted: Vec<f64>,
    /// Customer-minutes lost to failures.
    pub customer_minutes: Vec<f64>,
    /// Customer-minutes lost to planned work, which enters no reliability
    /// index and is reported on its own.
    pub planned_customer_minutes: Vec<f64>,
    /// Segments replaced as planned work.
    pub planned_replacements: Vec<f64>,
    /// Dollars spent on planned work, nominal.
    pub planned_spend: Vec<f64>,
    /// Dollars spent replacing failures, nominal.
    pub emergency_spend: Vec<f64>,
}

impl Results {
    /// Seven zeroed buffers of `replications * years * classes`.
    fn zeros(n_reps: usize, n_years: usize, n_classes: usize) -> Self {
        let cells = n_reps * n_years * n_classes;
        Self {
            failures: vec![0.0; cells],
            customers_interrupted: vec![0.0; cells],
            customer_minutes: vec![0.0; cells],
            planned_customer_minutes: vec![0.0; cells],
            planned_replacements: vec![0.0; cells],
            planned_spend: vec![0.0; cells],
            emergency_spend: vec![0.0; cells],
        }
    }
}

/// Runs one chunk of replications under one policy.
///
/// The arguments are the ones `simulate.run_chunk` takes, in that order and
/// under those names, so the two implementations are interchangeable behind a
/// keyword call. Every slice is C-contiguous, which the binding checks before
/// calling this.
///
/// # Arguments
///
/// * `length_ft` - segment length, in feet.
/// * `customers` - customers served, counted equally for the frequency index.
/// * `customer_minutes_per_failure` - customer-minutes lost when this segment
///   fails, already carrying its class's restoration time.
/// * `customer_minutes_per_planned` - the same for planned work, zero for a
///   class that is switched out without interrupting anyone.
/// * `outage_cost_per_failure` - value of lost load if this segment fails, in
///   dollars at year-0 prices.
/// * `class_index` - which segment class each segment belongs to, indexing the
///   third axis of the results.
/// * `age0` - age at the start of the run, in years.
/// * `shape` - Weibull shape for the cable in the ground, effective.
/// * `scale` - Weibull scale for the cable in the ground, effective.
/// * `replacement_shape` - Weibull shape a replacement would take, effective
///   for this segment's own geometry.
/// * `replacement_scale` - the same for scale.
/// * `cost_per_ft` - installed cost per foot.
/// * `lifetime_uniforms` - `replications * segments * (n_years + 1)` uniforms
///   in C order. Index 0 of a segment's row is the left-truncated draw made at
///   the start of the run, and a replacement made in year `y` reads `y + 1`.
/// * `policy_uniforms` - `replications * segments` in C order, one fixed
///   priority per segment, which only the random policy ranks on.
/// * `budget` - planned capital per year, already escalated.
/// * `cost_escalation` - per-year multiplier applied to every dollar quantity.
/// * `policy` - the resolved replacement policy.
/// * `emergency_multiplier` - what replacing a failure costs relative to the
///   same work planned.
/// * `mobilization_per_segment` - fixed cost of turning up at all.
/// * `emergency_charged_to_budget` - charge the year's emergency spend against
///   the planned budget before scoring planned work.
/// * `n_classes` - number of segment classes, sizing the third result axis.
/// * `n_years` - horizon, in years.
///
/// # Returns
///
/// The seven per-year, per-class buffers for this chunk, or `NanScore` if a
/// candidate scored a key that was not a number.
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
) -> Result<Results, policies::NanScore> {
    let n_segments = age0.len();
    let n_reps = policy_uniforms.len() / n_segments;
    let draws_per_segment = n_years + 1;

    let mut results = Results::zeros(n_reps, n_years, n_classes);
    // Costs at year-0 prices; the year's escalation is applied inside the loop.
    let planned_at_par: Vec<f64> = (0..n_segments)
        .map(|segment| {
            policies::planned_cost(
                length_ft[segment],
                cost_per_ft[segment],
                mobilization_per_segment,
            )
        })
        .collect();

    // Allocated once and reused across years and replications: at the shipped
    // population these are twelve thousand elements apiece, and the loop runs
    // thirty times per replication.
    let mut age = vec![0.0; n_segments];
    let mut current_shape = vec![0.0; n_segments];
    let mut current_scale = vec![0.0; n_segments];
    let mut failure_time = vec![0.0; n_segments];
    let mut planned_now = vec![0.0; n_segments];
    let mut rank = vec![0.0; n_segments];
    let mut replaced = vec![false; n_segments];
    let mut candidates: Vec<usize> = Vec::with_capacity(n_segments);

    for replication in 0..n_reps {
        age.copy_from_slice(age0);
        current_shape.copy_from_slice(shape);
        current_scale.copy_from_slice(scale);

        let priority = &policy_uniforms[replication * n_segments..][..n_segments];
        let lifetimes = &lifetime_uniforms[replication * n_segments * draws_per_segment..]
            [..n_segments * draws_per_segment];

        // Conditional on survival to age0: a population that starts partway
        // through its life must not behave as though it were new.
        for segment in 0..n_segments {
            failure_time[segment] = weibull::draw_remaining_life(
                lifetimes[segment * draws_per_segment],
                age[segment],
                current_shape[segment],
                current_scale[segment],
            );
        }

        for year in 0..n_years {
            let escalation = cost_escalation[year];
            for segment in 0..n_segments {
                planned_now[segment] = planned_at_par[segment] * escalation;
            }
            let cell = (replication * n_years + year) * n_classes;

            // 1. Failures, which are resolved before planned work so that a
            //    segment failing this year is not also a candidate this year.
            replaced.fill(false);
            let mut emergency_total = 0.0;
            for segment in 0..n_segments {
                let year_start = year as f64;
                if failure_time[segment] >= year_start && failure_time[segment] < year_start + 1.0 {
                    let emergency_now = planned_now[segment] * emergency_multiplier;
                    let class = cell + usize::from(class_index[segment]);
                    results.failures[class] += 1.0;
                    results.customers_interrupted[class] += customers[segment];
                    results.customer_minutes[class] += customer_minutes_per_failure[segment];
                    results.emergency_spend[class] += emergency_now;
                    emergency_total += emergency_now;
                    replaced[segment] = true;
                }
            }

            // 2. Planned replacement, funded greedily down the ranked order.
            let mut available = budget[year];
            if emergency_charged_to_budget {
                // Charged before this year's planned pass is scored, which is
                // what produces the loop where failures crowd out prevention.
                // The floor changes no funding decision — every planned cost is
                // positive, so the greedy fill funds nothing at zero or below —
                // and it stops a year whose failures cost more than the budget
                // from carrying a negative that reads as a debt.
                available = (available - emergency_total).max(0.0);
            }

            candidates.clear();
            for segment in 0..n_segments {
                if policies::eligible(policy, age[segment], replaced[segment]) {
                    candidates.push(segment);
                }
            }
            if !candidates.is_empty() {
                // Only the candidates are scored. The reference scores every
                // segment because that is what vectorizes, and then reads the
                // candidates' entries; the values it computes for the rest are
                // never read, so the two agree on everything either one uses.
                for &segment in &candidates {
                    rank[segment] = policies::rank_key(
                        policy,
                        age[segment],
                        weibull::conditional_failure_probability(
                            age[segment],
                            current_shape[segment],
                            current_scale[segment],
                        ),
                        outage_cost_per_failure[segment] * escalation,
                        planned_now[segment],
                        emergency_multiplier,
                        priority[segment],
                    );
                }
                let ordered = policies::order_by_rank(&rank, &candidates)?;
                for &segment in policies::fund(&ordered, &planned_now, available) {
                    let class = cell + usize::from(class_index[segment]);
                    results.planned_replacements[class] += 1.0;
                    results.planned_customer_minutes[class] +=
                        customer_minutes_per_planned[segment];
                    results.planned_spend[class] += planned_now[segment];
                    replaced[segment] = true;
                }
            }

            // 3. Everything replaced this year enters service next year, as new
            //    cable of the replacement technology, with a lifetime drawn
            //    from that segment's cell for the following year.
            for segment in 0..n_segments {
                if replaced[segment] {
                    current_shape[segment] = replacement_shape[segment];
                    current_scale[segment] = replacement_scale[segment];
                    failure_time[segment] = (year + 1) as f64
                        + weibull::draw_lifetime(
                            lifetimes[segment * draws_per_segment + year + 1],
                            replacement_shape[segment],
                            replacement_scale[segment],
                        );
                    age[segment] = 0.0;
                } else {
                    age[segment] += 1.0;
                }
            }
        }
    }

    Ok(results)
}
