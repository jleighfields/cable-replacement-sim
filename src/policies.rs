//! Replacement-policy eligibility and ranking.
//!
//! Each year the annual loop asks three questions of every segment: may it be
//! replaced, if several may then which first, and how far down that order the
//! budget reaches. This module answers all three, one segment at a time, for
//! the five policies `kind` names.
//!
//! Two policies exist as controls rather than as proposals. `WORST_FIRST`
//! ranks on failure probability alone, so the gap between it and `RISK_RANKED`
//! is the value of weighting by consequence; `RANDOM` ranks on a fixed
//! per-segment priority, so the gap between it and everything else is the
//! value of ranking at all rather than merely of spending.
//!
//! The Rust side of the mirror: `python/cablesim/policies.py` carries the same
//! functions under the same names, taking the same arguments in the same
//! order, over whole arrays rather than one segment at a time.

use std::fmt;

use pyo3::prelude::*;

/// The integer tag each policy is compared by.
///
/// The mapping is authored in `python/cablesim/policies.py` as `KIND`, and
/// these constants are the Rust reading of it. Nothing derives one from the
/// other, so what catches a divergence is the deterministic parity test: it
/// runs all five policies through both implementations, and any two tags
/// exchanged funds a different set of segments.
pub mod kind {
    /// Replaces nothing; failures are still repaired.
    pub const RUN_TO_FAILURE: u8 = 0;
    /// Every segment at or over an age, oldest first.
    pub const AGE_THRESHOLD: u8 = 1;
    /// Failure probability weighted by what a failure would cost.
    pub const RISK_RANKED: u8 = 2;
    /// Failure probability alone, ignoring consequence.
    pub const WORST_FIRST: u8 = 3;
    /// A fixed per-segment priority, ignoring everything else.
    pub const RANDOM: u8 = 4;
}

/// A validated policy reduced to what the annual loop needs.
///
/// Extracted from `policies.Resolved`, the Python NamedTuple of the same name
/// and the same three fields. The unused fields carry explicit neutral values
/// rather than an `Option`, so eligibility is one comparison for every policy
/// and the loop keeps a single code path. **This struct carries no defaults of
/// its own**: choosing the neutral values happens once, in Python, beside the
/// schema they come from, because a default written on both sides of the
/// boundary diverges silently.
#[derive(FromPyObject, Clone, Copy, Debug)]
pub struct Resolved {
    /// The tag from `kind`.
    pub kind: u8,
    /// Eligibility age, compared as `age >= threshold_years`. Negative
    /// infinity makes every in-service segment eligible; positive infinity
    /// makes none, which is how `RUN_TO_FAILURE` funds nothing without a
    /// branch of its own.
    pub threshold_years: f64,
    /// Divide the score by planned replacement cost. True only for
    /// `RISK_RANKED` with `rank_by: score_per_dollar`.
    pub rank_by_cost: bool,
}

/// A candidate whose rank key was not a number.
///
/// Every input to the score is finite by construction — planned cost carries
/// the mobilization term so it is strictly positive, and the annual failure
/// probability is never NaN at any age the model reaches — so this is a defect
/// upstream of ranking. Unreported it would surface as a segment that quietly
/// never gets funded, because a NaN sorts last under the total comparator.
#[derive(Debug)]
pub struct NanScore {
    /// How many candidates scored NaN.
    pub count: usize,
    /// The lowest-numbered of them, which is its array position.
    pub first_segment_id: usize,
}

impl fmt::Display for NanScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} candidate segments scored NaN, first at segment_id {}; every \
             input to the score is finite by construction, so this is a defect \
             upstream of ranking",
            self.count, self.first_segment_id
        )
    }
}

/// Cost of replacing one segment as planned work.
///
/// The mobilization term is what makes short segments expensive per foot, and
/// it is also the only reason ranking by score per dollar differs from ranking
/// by score: without it, cost is exactly proportional to length and dividing by
/// it rescales every candidate equally.
///
/// # Arguments
///
/// * `length_ft` - segment length, in feet.
/// * `cost_per_ft` - installed cost per foot, which keys on segment class.
/// * `mobilization_per_segment` - fixed cost of turning up at all.
///
/// # Returns
///
/// Planned replacement cost, in dollars.
pub fn planned_cost(length_ft: f64, cost_per_ft: f64, mobilization_per_segment: f64) -> f64 {
    length_ft * cost_per_ft + mobilization_per_segment
}

/// Scores one segment on what its policy ranks by, highest funded first.
///
/// `RISK_RANKED` carries two terms and needs both. The first is the customer
/// value at risk this year. The second is what the utility avoids by doing the
/// work planned rather than after a failure, and without it a lateral serving
/// three customers is never replaced at any age, however far past the point
/// where planned replacement is the cheaper of the two.
///
/// # Arguments
///
/// * `policy` - the resolved policy.
/// * `age` - current age of the segment, in years.
/// * `failure_probability` - probability of failing within the year, given
///   survival to `age`.
/// * `outage_cost_per_failure` - value of lost load if this segment fails, in
///   dollars, already carrying the year's escalation.
/// * `planned` - planned replacement cost, in dollars, likewise escalated.
/// * `emergency_multiplier` - what an emergency replacement costs relative to
///   the same work planned.
/// * `priority` - one fixed uniform for this segment, which `RANDOM` ranks on.
///
/// # Returns
///
/// The rank key. The value for a segment this policy cannot fund is never
/// read, because eligibility is applied before ordering.
pub fn rank_key(
    policy: Resolved,
    age: f64,
    failure_probability: f64,
    outage_cost_per_failure: f64,
    planned: f64,
    emergency_multiplier: f64,
    priority: f64,
) -> f64 {
    let key = match policy.kind {
        kind::AGE_THRESHOLD => age,
        kind::RISK_RANKED => {
            // The cost avoided by acting first, which is the emergency premium
            // rather than the whole emergency cost: the planned work is paid
            // either way.
            let avoided = planned * (emergency_multiplier - 1.0);
            failure_probability * (outage_cost_per_failure + avoided)
        }
        kind::WORST_FIRST => failure_probability,
        kind::RANDOM => priority,
        // run_to_failure, whose eligible set is empty, so nothing is ranked.
        _ => 0.0,
    };

    if policy.rank_by_cost {
        // Planned cost, because that is what the budget is charged, so the
        // ratio is value per budget dollar. Dividing by emergency cost would
        // rank on value per dollar of exposure, which is not what the
        // constraint is denominated in.
        key / planned
    } else {
        key
    }
}

/// Whether this policy may fund this segment this year.
///
/// Nothing is ever out of service — a failure is replaced the same year — so
/// the only exclusion beyond the age threshold is having already been replaced
/// this year, which is what stops a segment that failed in year `y` also being
/// planned work in year `y`.
///
/// # Arguments
///
/// * `policy` - the resolved policy.
/// * `age` - current age of the segment, in years.
/// * `replaced_this_year` - whether it has already been replaced.
///
/// # Returns
///
/// True if the segment is a candidate.
pub fn eligible(policy: Resolved, age: f64, replaced_this_year: bool) -> bool {
    age >= policy.threshold_years && !replaced_this_year
}

/// Orders eligible segments by rank, breaking ties on segment identifier.
///
/// The key is `(rank descending, segment_id ascending)` and is **total**, so
/// ties cannot decide the answer. They arise constantly — `AGE_THRESHOLD`
/// ranks on an integer age shared by thousands of segments — and because the
/// greedy fill stops at the first candidate that does not fit, which tied
/// segment lands last is what decides whether it is funded. A total key makes
/// the result independent of whether an implementation's sort is stable, which
/// is what lets this side and the Python side agree.
///
/// `f64` implements `PartialOrd` and not `Ord`, so the comparator is written
/// out rather than reached for: it places NaN last and the caller is told a
/// NaN was there, instead of a defect upstream turning into a segment that is
/// silently never funded.
///
/// A segment's position in these arrays is its identifier, which is why the
/// tie-break needs nothing passed to it.
///
/// # Arguments
///
/// * `rank` - the rank key per segment, as `rank_key` filled it. Only the
///   candidates' entries are read.
/// * `candidates` - positions of the eligible segments, ascending.
///
/// # Returns
///
/// The eligible positions in the order the budget should be spent on them, or
/// `NanScore` if any candidate's key was not a number.
pub fn order_by_rank(rank: &[f64], candidates: &[usize]) -> Result<Vec<usize>, NanScore> {
    let unranked: Vec<usize> = candidates
        .iter()
        .copied()
        .filter(|&index| rank[index].is_nan())
        .collect();
    if let Some(&first_segment_id) = unranked.first() {
        return Err(NanScore {
            count: unranked.len(),
            first_segment_id,
        });
    }

    let mut ordered = candidates.to_vec();
    ordered.sort_by(|&left, &right| {
        rank[right]
            .partial_cmp(&rank[left])
            .expect("NaN keys are rejected above, so every comparison here is ordered")
            .then(left.cmp(&right))
    });
    Ok(ordered)
}

/// Spends the budget down the ranked order, stopping at the first misfit.
///
/// Funding down a ranked list until the money runs out is what a capital plan
/// does operationally. The alternative — passing over an unaffordable candidate
/// and continuing to fund cheaper ones below it — spends more of the budget,
/// but it is a knapsack heuristic with nothing behind it. Ranking per dollar is
/// what compensates for a cheap candidate being passed over, and it does so in
/// the ranking rather than in the fill.
///
/// A candidate costing exactly what remains is funded: the rule is that
/// spending may not exceed the budget, not that it must fall short.
///
/// The running total is accumulated one candidate at a time, which is what
/// `numpy.cumsum` does on the Python side. `numpy.sum` is free to add pairwise
/// and would disagree in the last bits, and here that is not a rounding
/// difference to tolerate: it decides which segment is the last one funded,
/// which is a discrete outcome the parity tests compare exactly.
///
/// # Arguments
///
/// * `ranked` - eligible positions, best first, as `order_by_rank` returns them.
/// * `planned` - planned replacement cost per segment, in dollars, already
///   carrying the year's cost escalation.
/// * `budget` - what this year has to spend, in dollars.
///
/// # Returns
///
/// The positions to replace, a prefix of `ranked`.
pub fn fund<'a>(ranked: &'a [usize], planned: &[f64], budget: f64) -> &'a [usize] {
    let mut running = 0.0;
    let mut funded = 0;
    for &index in ranked {
        running += planned[index];
        if running > budget {
            break;
        }
        funded += 1;
    }
    &ranked[..funded]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(kind: u8) -> Resolved {
        Resolved {
            kind,
            threshold_years: f64::NEG_INFINITY,
            rank_by_cost: false,
        }
    }

    /// Equal scores are ordered by segment identifier, ascending.
    #[test]
    fn ties_break_on_the_segment_identifier() {
        let rank = [5.0, 5.0, 5.0, 9.0];

        assert_eq!(
            order_by_rank(&rank, &[0, 1, 2, 3]).unwrap(),
            vec![3, 0, 1, 2]
        );
    }

    /// A NaN key is reported rather than sorted to the end and forgotten.
    #[test]
    fn a_nan_key_is_reported_with_its_segment() {
        let rank = [1.0, f64::NAN, 3.0, f64::NAN];

        let error = order_by_rank(&rank, &[0, 1, 2, 3]).unwrap_err();

        assert_eq!(error.count, 2);
        assert_eq!(error.first_segment_id, 1);
    }

    /// The fill stops at the first candidate that does not fit, rather than
    /// skipping it to fund something cheaper below.
    #[test]
    fn the_fill_stops_at_the_first_candidate_that_does_not_fit() {
        let planned = [10.0, 100.0, 10.0];

        assert_eq!(fund(&[0, 1, 2], &planned, 50.0), &[0]);
    }

    /// A candidate costing exactly what remains is funded.
    #[test]
    fn a_candidate_costing_exactly_the_remainder_is_funded() {
        let planned = [10.0, 40.0, 10.0];

        assert_eq!(fund(&[0, 1, 2], &planned, 50.0), &[0, 1]);
    }

    /// `run_to_failure` scores every segment the same, so nothing about its
    /// ranking can fund one segment over another; its eligible set is empty.
    #[test]
    fn run_to_failure_ranks_nothing() {
        let scored = rank_key(
            policy(kind::RUN_TO_FAILURE),
            40.0,
            0.5,
            1000.0,
            100.0,
            2.5,
            0.7,
        );

        assert_eq!(scored, 0.0);
    }

    /// Ranking per dollar divides by planned cost, not by emergency cost.
    #[test]
    fn ranking_per_dollar_divides_by_the_planned_cost() {
        let mut risk_ranked = policy(kind::RISK_RANKED);
        risk_ranked.rank_by_cost = true;

        let scored = rank_key(risk_ranked, 40.0, 0.5, 1000.0, 100.0, 2.5, 0.7);

        // 0.5 * (1000 + 100 * 1.5) = 575, over a planned cost of 100.
        assert!((scored - 5.75).abs() < 1e-12);
    }
}
