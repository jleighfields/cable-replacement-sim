"""Two annual loops that hold every replication in flight at once.

These exist for the benchmark and for nothing else. `simulate.py` is the
correctness reference, and a speedup measured against it would be measured
against a program written to be read rather than to be fast. **The batched
NumPy loop here is the baseline a speedup is honestly claimed against**, and
the batched polars loop asks a second question: whether a frame engine is
competitive for this shape of work.

Nothing outside the benchmark and the parity tests imports this module.

## What "batched" changes

The reference holds one replication's state and loops replications outside
years, so the year body runs `replications * years` times. These hold state as
`(replications, segments)` arrays, so the year body runs `years` times and
every operation inside it spans the whole chunk. At a thousand replications
that is thirty passes rather than thirty thousand.

Nothing about the model changes, and nothing about the arithmetic changes
either. **Every quantity here agrees with the reference bit for bit**, which
takes two deliberate choices rather than luck:

* **A cumulative sum with zeros interleaved equals the cumulative sum of the
  compacted array.** Adding `0.0` to a float returns that float unchanged, so
  totalling `where(failed, cost, 0.0)` across every segment gives exactly what
  the reference gets by totalling the failed segments' costs alone. That is
  what lets the emergency bill be computed without compacting each replication
  separately, and the bill is subtracted from the budget the greedy fill then
  compares against, so a last-bit difference there would change which segment
  is funded last.
* **`bincount` accumulates in the order its index array is given.** Offsetting
  each replication's class bins by `replication * n_classes` and calling it
  once therefore adds each class's segments in the same ascending segment order
  the reference adds them in, rather than in some order a grouping engine chose.

Scoring, eligibility and the Weibull draws are not reimplemented here: the
functions in `policies.py` and `weibull.py` are elementwise, so they take a
`(replications, segments)` array wherever the reference hands them a
`(segments,)` one and compute the identical values. Only the two steps that
are inherently per-replication — ordering candidates and spending the budget
down that order — need a form of their own, and they are the two functions
below.
"""

import numpy as np

from cablesim import policies, simulate, weibull


def order_all_by_rank(
    policy: policies.Resolved,
    rank: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    """Orders every segment of every replication, ineligible ones last.

    The reference compacts each replication's candidates and sorts those. A
    batched form cannot, because different replications have different numbers
    of candidates and the result would be ragged. It sorts whole rows instead
    and pushes the ineligible to the end by scoring them negative infinity,
    which is below every real rank key: the scores are ages, probabilities,
    dollar values and uniforms, all finite and none negative.

    The key is ``(rank descending, segment_id ascending)`` and is total, so the
    eligible segments come back in exactly the order the reference puts them
    in, whatever the sort does with the ineligible tail behind them.

    Args:
        policy: The resolved policy, whose ranking branch decides nothing here
            but whose presence keeps this callable the same way the reference's
            ordering is.
        rank: ``(replications, segments)`` rank keys.
        eligible: ``(replications, segments)`` mask of what may be funded.

    Returns:
        ``(replications, segments)`` of segment indices, best first per row.

    Raises:
        ValueError: If any eligible segment's rank is not a number. Every input
            to the score is finite by construction, so a NaN here is a defect
            upstream; without this it would surface as a segment that quietly
            never gets funded.
    """
    del policy
    if np.isnan(rank[eligible]).any():
        # Reported the way the reference reports it, since a caller running
        # both must not get a different diagnosis from each.
        replication, segment = np.nonzero(eligible & np.isnan(rank))
        raise ValueError(
            f"{replication.size} candidate segments scored NaN, first at "
            f"segment_id {int(segment[0])}; every input to the score is "
            f"finite by construction, so this is a defect upstream of ranking"
        )
    demoted = np.where(eligible, rank, -np.inf)
    segment_ids = np.broadcast_to(np.arange(rank.shape[1]), rank.shape)
    # `lexsort` takes its keys least significant first, and sorts ascending, so
    # the negated rank is the primary key and the segment id breaks its ties.
    return np.lexsort((segment_ids, -demoted), axis=1)


def fund_all(
    ordered: np.ndarray,
    planned: np.ndarray,
    eligible: np.ndarray,
    available: np.ndarray,
) -> np.ndarray:
    """Spends each replication's budget down its own ranked order.

    The stopping rule is the reference's: take the cumulative cost in rank
    order and cut at the first candidate that would exceed what is available,
    funding one that costs exactly the remainder. Counting how many cumulative
    costs fall at or below the budget is the same cut expressed without a
    search, which is what makes it work on every row at once.

    Ineligible segments are given infinite cost, so the cut cannot run past the
    end of a row's real candidates however much budget is left.

    Args:
        ordered: ``(replications, segments)`` segment indices, best first.
        planned: ``(segments,)`` planned replacement cost at this year's prices.
        eligible: ``(replications, segments)`` mask of what may be funded.
        available: ``(replications,)`` dollars this year has left to spend.

    Returns:
        ``(replications, segments)`` mask **in rank order**, true for a
        position that is funded. Returned in that order rather than as a mask
        over segments because the totals taken from it have to be accumulated
        in rank order to match the reference in the last bits; ``in_rank_order``
        puts anything else into the same order, and ``np.put_along_axis`` with
        ``ordered`` converts it back to a mask over segments.
    """
    in_order = np.where(
        np.take_along_axis(eligible, ordered, axis=1), planned[ordered], np.inf
    )
    # Accumulated one partial sum at a time, which is what the reference's own
    # running total does. `sum` is free to add pairwise and would disagree in
    # the last bits, and that difference decides which segment is the last one
    # funded — a discrete outcome rather than a rounding difference.
    running = np.cumsum(in_order, axis=1)
    funded_count = (running <= available[:, None]).sum(axis=1)
    return np.arange(ordered.shape[1]) < funded_count[:, None]


def in_rank_order(values: np.ndarray, ordered: np.ndarray) -> np.ndarray:
    """Rearranges a per-segment quantity into each replication's rank order.

    The reference funds a list of segment indices in rank order and totals them
    in that order, so this is what makes a batched total add the same numbers in
    the same sequence. Ordinary floating-point addition is not associative, and
    the difference reaches the reported spend.

    Args:
        values: ``(segments,)`` or ``(replications, segments)`` quantity.
        ordered: ``(replications, segments)`` segment indices, best first.

    Returns:
        ``(replications, segments)`` with each row in that row's rank order.
    """
    return np.take_along_axis(np.broadcast_to(values, ordered.shape), ordered, axis=1)


def totals_by_class(
    selected: np.ndarray,
    bins: np.ndarray,
    n_classes: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Totals a quantity per replication and per class.

    One `bincount` over every replication at once, with each replication's
    class bins offset by its own position. Called per replication instead, this
    would add the same values in the same order; called as a grouping it might
    not, and the order is what keeps this equal to the reference in the last
    bits.

    Args:
        selected: ``(replications, segments)`` mask of what to total.
        bins: ``(replications, segments)`` class index offset by replication.
        n_classes: Number of segment classes.
        weights: ``(replications, segments)`` value to add per segment, or None
            to count them.

    Returns:
        ``(replications, n_classes)`` totals.
    """
    n_reps = selected.shape[0]
    chosen = bins[selected]
    weighted = (
        None if weights is None else np.broadcast_to(weights, bins.shape)[selected]
    )
    return np.bincount(chosen, weights=weighted, minlength=n_reps * n_classes).reshape(
        n_reps, n_classes
    )


def run_chunk_numpy(
    length_ft: np.ndarray,
    customers: np.ndarray,
    customer_minutes_per_failure: np.ndarray,
    customer_minutes_per_planned: np.ndarray,
    outage_cost_per_failure: np.ndarray,
    class_index: np.ndarray,
    age0: np.ndarray,
    shape: np.ndarray,
    scale: np.ndarray,
    replacement_shape: np.ndarray,
    replacement_scale: np.ndarray,
    cost_per_ft: np.ndarray,
    lifetime_uniforms: np.ndarray,
    policy_uniforms: np.ndarray,
    budget: np.ndarray,
    cost_escalation: np.ndarray,
    policy: policies.Resolved,
    emergency_multiplier: float,
    mobilization_per_segment: float,
    emergency_charged_to_budget: bool,
    n_classes: int,
    n_years: int,
    threads: int = 1,
) -> simulate.Results:
    """Runs one chunk of replications under one policy, every replication at once.

    The arguments are `simulate.run_chunk`'s and mean the same things; see that
    function for what each one is. The three numbered steps below are its three
    steps, in the same order, with the replication axis carried through every
    operation rather than looped over.

    Args:
        length_ft: Segment length, in feet.
        customers: Customers served, counted equally for the frequency index.
        customer_minutes_per_failure: Customer-minutes lost when this segment
            fails, already carrying its class's restoration time.
        customer_minutes_per_planned: The same for planned work, zero for a
            class that is switched out without interrupting anyone.
        outage_cost_per_failure: Value of lost load if this segment fails, in
            dollars at year-0 prices.
        class_index: Which segment class each segment belongs to.
        age0: Age at the start of the run, in years.
        shape: Weibull shape for the cable in the ground, effective.
        scale: Weibull scale for the cable in the ground, effective.
        replacement_shape: Weibull shape a replacement would take.
        replacement_scale: The same for scale.
        cost_per_ft: Installed cost per foot.
        lifetime_uniforms: ``(replications, segments, n_years + 1)`` uniforms.
        policy_uniforms: ``(replications, segments)`` fixed priorities.
        budget: Planned capital per year, already escalated.
        cost_escalation: Per-year multiplier applied to every dollar quantity.
        policy: The resolved replacement policy.
        emergency_multiplier: What replacing a failure costs relative to the
            same work planned.
        mobilization_per_segment: Fixed cost of turning up at all.
        emergency_charged_to_budget: Charge the year's emergency spend against
            the planned budget before scoring planned work.
        n_classes: Number of segment classes.
        n_years: Horizon, in years.
        threads: Workers to spread the replications over. This implementation
            has one and refuses any other value.

    Returns:
        The seven per-year, per-class arrays for this chunk.

    Raises:
        ValueError: Everything `simulate.check_arguments` refuses, which is
            what makes this interchangeable with the other implementations, and
            a rank key that is not a number.
        TypeError: If ``policy.kind`` is not an integer.
    """
    n_reps, n_segments = simulate.check_arguments(locals())

    results = simulate.Results(
        *(np.zeros((n_reps, n_years, n_classes)) for _ in simulate.Results._fields)
    )
    planned_at_par = policies.planned_cost(
        length_ft, cost_per_ft, mobilization_per_segment
    )
    # Each replication's class bins offset into its own block, so one
    # `bincount` covers the chunk and still adds in segment order within a
    # class.
    bins = class_index.astype(np.intp) + n_classes * np.arange(n_reps)[:, None]

    # `(replications, segments)` state, which is the whole difference from the
    # reference. Broadcast rather than tiled where the starting value is the
    # same for every replication, then copied so the copies can diverge.
    age = np.broadcast_to(age0.astype(float), (n_reps, n_segments)).copy()
    current_shape = np.broadcast_to(shape, (n_reps, n_segments)).copy()
    current_scale = np.broadcast_to(scale, (n_reps, n_segments)).copy()
    failure_time = weibull.draw_remaining_life(
        lifetime_uniforms[:, :, 0], age, current_shape, current_scale
    )

    for year in range(n_years):
        escalation = cost_escalation[year]
        planned_now = planned_at_par * escalation

        # 1. Failures, resolved before planned work so that a segment failing
        #    this year is not also a candidate this year.
        failed = (failure_time >= year) & (failure_time < year + 1)
        emergency_now = np.where(failed, planned_now * emergency_multiplier, 0.0)
        results.failures[:, year] += totals_by_class(failed, bins, n_classes)
        results.customers_interrupted[:, year] += totals_by_class(
            failed, bins, n_classes, customers
        )
        results.customer_minutes[:, year] += totals_by_class(
            failed, bins, n_classes, customer_minutes_per_failure
        )
        results.emergency_spend[:, year] += totals_by_class(
            failed, bins, n_classes, emergency_now
        )

        replaced = failed.copy()

        # 2. Planned replacement, funded greedily down the ranked order.
        available = np.full(n_reps, budget[year])
        if emergency_charged_to_budget:
            # Left negative where a year's failures cost more than the budget,
            # for the reason the reference leaves it negative: every planned
            # cost is positive, so nothing is funded at or below zero and
            # nothing carries into the next year.
            #
            # The cumulative sum runs over every segment with the survivors
            # contributing zero, which totals the failures in the same order
            # and to the same last bit as compacting them first would.
            available -= np.cumsum(emergency_now, axis=1)[:, -1]

        eligible = policies.eligible(policy, age, replaced)
        if eligible.any():
            rank = policies.rank_key(
                policy,
                age=age,
                failure_probability=weibull.conditional_failure_probability(
                    age, current_shape, current_scale
                ),
                outage_cost_per_failure=outage_cost_per_failure * escalation,
                planned=planned_now,
                emergency_multiplier=emergency_multiplier,
                priority=policy_uniforms,
            )
            ordered = order_all_by_rank(policy, rank, eligible)
            funded_positions = fund_all(ordered, planned_now, eligible, available)
            # Totalled in rank order, because that is the order the reference
            # adds them in and floating-point addition is not associative. The
            # failure totals above are taken in segment order for the same
            # reason: that is the order the reference takes those in.
            ranked_bins = in_rank_order(bins, ordered)
            results.planned_replacements[:, year] += totals_by_class(
                funded_positions, ranked_bins, n_classes
            )
            results.planned_customer_minutes[:, year] += totals_by_class(
                funded_positions,
                ranked_bins,
                n_classes,
                in_rank_order(customer_minutes_per_planned, ordered),
            )
            results.planned_spend[:, year] += totals_by_class(
                funded_positions,
                ranked_bins,
                n_classes,
                in_rank_order(planned_now, ordered),
            )
            funded = np.zeros(ordered.shape, dtype=bool)
            np.put_along_axis(funded, ordered, funded_positions, axis=1)
            replaced |= funded

        # 3. Everything replaced this year enters service next year, as new
        #    cable of the replacement technology.
        if replaced.any():
            current_shape = np.where(replaced, replacement_shape, current_shape)
            current_scale = np.where(replaced, replacement_scale, current_scale)
            failure_time = np.where(
                replaced,
                (year + 1)
                + weibull.draw_lifetime(
                    lifetime_uniforms[:, :, year + 1],
                    replacement_shape,
                    replacement_scale,
                ),
                failure_time,
            )
        age = np.where(replaced, 0.0, age + 1.0)

    return results
