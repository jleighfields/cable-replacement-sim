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
//!
//! # Reading this beside the Python
//!
//! Four constructs here have no direct Python equivalent, and each is doing a
//! job Python does some other way.
//!
//! * **`&[f64]` is a slice: a borrowed view of somebody else's numbers.** It
//!   is closest to a NumPy view — a pointer and a length, no copy — except
//!   that the compiler tracks how long the view may live and refuses to
//!   compile code that outlives what it points at. Where a Python function
//!   takes `rank` and `planned` as arrays it might in principle mutate, these
//!   take read-only views and cannot.
//! * **Errors are returned, not raised.** `order_by_rank` returns
//!   `Result<Vec<usize>, NanScore>`, which is either the ordering or the
//!   failure. Rust has no exceptions, so a caller cannot forget the failure
//!   case: it has to unwrap the `Result`, and the `?` operator at the call
//!   site is the shorthand that says "hand this failure to my own caller",
//!   which is what a bare `raise` propagating up a call stack does in Python.
//! * **`match` is `if`/`elif`/`else` with a compiler check.** The final `_`
//!   arm is the `else`; without some arm covering every possible value the
//!   code does not compile, so a policy tag with no branch is caught at build
//!   time rather than falling through to whatever the last branch happened to
//!   be.
//! * **Sorting floats needs a comparator written out.** Python's `sorted`
//!   works on floats because it only ever asks whether one is less than
//!   another. Rust separates `PartialOrd`, which floats have, from `Ord`,
//!   which they do not, precisely because NaN is not ordered against anything
//!   — `NaN < 1.0`, `NaN > 1.0` and `NaN == 1.0` are all false, so a sort that
//!   assumes a total order can produce a garbage permutation rather than an
//!   error. `order_by_rank` therefore rejects NaN first and then sorts.

use std::fmt;

use pyo3::prelude::*;

use crate::weibull::Real;

/// The integer tag each policy is compared by.
///
/// The mapping is authored in `python/cablesim/policies.py` as `KIND`, and
/// these constants are the Rust reading of it. Nothing derives one from the
/// other, so what catches a divergence is the deterministic parity test: it
/// runs all five policies through both implementations, and any two tags
/// exchanged funds a different set of segments.
pub mod kind {
    // `i64` rather than `u8`, which is wide enough for five tags but is not
    // what decides the type here. PyO3 refuses to extract a value outside a
    // narrow integer's range *before* any check of ours runs, and it refuses
    // with a `TypeError` naming neither the tag nor the branch it lacks — so a
    // tag of -1 or 256 would be reported one way by this side and another by
    // the reference. Extracting the whole integer range and rejecting it
    // ourselves is what lets both sides refuse the same input with the same
    // error and the same words.
    // A module of `const` values, which is how Rust spells the namespace that
    // `policies.KIND` gets from being a dict: `kind::RISK_RANKED` here reads
    // as `KIND["risk_ranked"]` does there. A `const` is substituted at every
    // use site at compile time, so this costs nothing at run time and, unlike
    // the dict, cannot be looked up with a key that does not exist.
    /// Replaces nothing; failures are still repaired.
    pub const RUN_TO_FAILURE: i64 = 0;
    /// Every segment at or over an age, oldest first.
    pub const AGE_THRESHOLD: i64 = 1;
    /// Failure probability weighted by what a failure would cost.
    pub const RISK_RANKED: i64 = 2;
    /// Failure probability alone, ignoring consequence.
    pub const WORST_FIRST: i64 = 3;
    /// A fixed per-segment priority, ignoring everything else.
    pub const RANDOM: i64 = 4;
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
// `#[derive(...)]` asks the compiler to write these implementations. Ordinary
// Rust structs get none of them by default, which is unlike Python, where every
// object can be copied, printed and passed around from the moment it exists.
// `FromPyObject` is PyO3's: it generates the code that reads a Python object's
// `kind`, `threshold_years` and `rank_by_cost` attributes into these fields,
// which is what lets `policies.Resolved` — a NamedTuple — arrive here directly.
// `Copy` makes the struct pass by value like an integer rather than by
// reference, which is why callers below write `policy` and not `&policy`; it is
// three small fields, so copying it is cheaper than pointing at it.
#[derive(FromPyObject, Clone, Copy, Debug)]
pub struct Resolved {
    /// The tag from `kind`.
    pub kind: i64,
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

// `impl Trait for Type` attaches behaviour to a type from outside the type's
// own definition — there is no equivalent of writing methods inside a `class`
// body, and any trait can be implemented for any type the crate owns.
// `Display` is what `{}` in a format string calls, so this is the job
// `__str__` does in Python.
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
pub fn planned_cost<T: Real>(length_ft: T, cost_per_ft: T, mobilization_per_segment: T) -> T {
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
pub fn rank_key<T: Real>(
    policy: Resolved,
    age: T,
    failure_probability: T,
    outage_cost_per_failure: T,
    planned: T,
    emergency_multiplier: T,
    priority: T,
) -> T {
    let key = match policy.kind {
        kind::AGE_THRESHOLD => age,
        kind::RISK_RANKED => {
            // The cost avoided by acting first, which is the emergency premium
            // rather than the whole emergency cost: the planned work is paid
            // either way.
            let avoided = planned * (emergency_multiplier - T::one());
            failure_probability * (outage_cost_per_failure + avoided)
        }
        kind::WORST_FIRST => failure_probability,
        kind::RANDOM => priority,
        // run_to_failure, whose eligible set is empty, so nothing is ranked.
        // The binding refuses any tag outside the five before the loop starts,
        // so this arm is that policy; it is written as a catch-all only
        // because a `match` on `u8` has to cover every value.
        _ => T::zero(),
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
pub fn eligible<T: Real>(policy: Resolved, age: T, replaced_this_year: bool) -> bool {
    // Narrowed rather than the age widened, because that is the comparison
    // NumPy makes when a Python float meets an array of this width.
    age >= T::from_double(policy.threshold_years) && !replaced_this_year
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
pub fn order_by_rank<T: Real>(rank: &[T], candidates: &[usize]) -> Result<Vec<usize>, NanScore> {
    // **The key travels with the identifier rather than being looked up.**
    // Sorting the candidate positions directly would make every comparison
    // read `rank[left]` and `rank[right]`, two scattered accesses into an
    // array as long as the population. Pairing each key with its position
    // first costs one pass and one allocation, and the comparator then reads
    // two adjacent machine words. Measured on one thread at 12,000 segments, 5
    // replications and a 30-year horizon under `risk_ranked`, in a release
    // build: 0.197 s paired against 0.216 s looked up, so about nine percent.
    //
    // Scoring, sorting and filling is where that run's time goes, which is why
    // nine percent of it is worth one allocation: `run_to_failure` over the
    // same inputs does none of the three and takes 0.009 s, against 0.216 s
    // for `risk_ranked`.
    //
    // `.iter().map(...).collect()` is a list comprehension read left to right:
    // iterate, build a pair from each position, and collect into a `Vec` —
    // Rust's growable list. Nothing runs until `collect` asks for it, the same
    // way a Python generator does nothing until something consumes it.
    let mut keyed: Vec<(T, usize)> = candidates
        .iter()
        .map(|&segment| (rank[segment], segment))
        .collect();

    // Scanned here rather than inside the comparator, where it would run once
    // per comparison instead of once per candidate.
    let unranked = keyed.iter().filter(|(key, _)| key.is_nan());
    // `.next()` on an iterator is Python's `next(...)`, and the `Option` it
    // returns is either `Some(value)` or `None`. Rust has no `null`, so "there
    // might not be one here" is in the type and the compiler makes the caller
    // handle both; `if let` runs a block only in the `Some` case.
    if let Some(&(_, first_segment_id)) = unranked.clone().next() {
        return Err(NanScore {
            count: unranked.count(),
            first_segment_id,
        });
    }

    // `sort_unstable_by` takes a comparator returning `Ordering::Less`,
    // `Equal` or `Greater`, which is Python 2's `cmp` argument rather than
    // Python 3's `key`. Comparing `right.0` against `left.0` — right before
    // left — is what makes the primary key descending, and `.then(...)` uses
    // the second comparison only when the first came out `Equal`, so the pair
    // reads as Python's `key=lambda i: (-rank[i], i)`.
    //
    // The unstable sort is the faster one and does not preserve the order of
    // equal elements. That is safe **only because this key is total**: no two
    // candidates share an identifier, so no two pairs ever compare `Equal` and
    // there is no order left for stability to preserve. Against the plain
    // `sort_by` used on the key alone it would decide ties arbitrarily, and
    // the reference and the kernel would fund different segments.
    keyed.sort_unstable_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            // `partial_cmp` returns `None` for a comparison involving NaN.
            // `expect` turns that into a panic carrying this message — the
            // blunt tool, used here only because the NaN scan above has
            // already ruled the case out, so reaching it would mean this
            // function is broken rather than that its input was.
            .expect("NaN keys are rejected above, so every comparison here is ordered")
            .then(left.1.cmp(&right.1))
    });
    Ok(keyed.into_iter().map(|(_, segment)| segment).collect())
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
// `<'a>` names a lifetime, and tying it to `ranked` and to the return type
// says the returned slice borrows from `ranked` and not from `planned`. It is
// not a run-time cost or a check the caller performs; it is what lets the
// compiler prove nobody holds this prefix after the list it points into is
// gone. Python leaves that to the garbage collector, so the annotation has no
// counterpart there — it is the price of returning a view rather than a copy.
pub fn fund<'a, T: Real>(ranked: &'a [usize], planned: &[T], budget: T) -> &'a [usize] {
    let mut running = T::zero();
    let mut funded = 0;
    for &index in ranked {
        // `Float` does not require `AddAssign`, so this is written long.
        running = running + planned[index];
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

    /// A policy with the neutral threshold, so every segment is eligible.
    fn policy(kind: i64) -> Resolved {
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

        let scored = rank_key::<f64>(risk_ranked, 40.0, 0.5, 1000.0, 100.0, 2.5, 0.7);

        // 0.5 * (1000 + 100 * 1.5) = 575, over a planned cost of 100.
        assert!((scored - 5.75).abs() < 1e-12);
    }
}
