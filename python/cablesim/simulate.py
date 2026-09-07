"""The annual simulation loop, one replication at a time.

This is the correctness reference: it is written to be checkable by reading, so
that when it and another implementation disagree, this one arbitrates. It takes
the same arguments as the Rust kernel and returns the same seven arrays, so
whatever runs a simulation can call either without knowing which it has.

Failure times are continuous and drawn once per installation; the budget cycle
is annual, because utilities budget annually, and the loop resolves the two
against each other. There is no event queue: one failure time per segment plus
a scan per year is enough, and a priority queue would maintain an ordering
nothing consumes.

Two conventions are load-bearing, and getting either wrong leaves a run that
completes with plausible-looking curves:

**Failure times are simulation time measured from year 0, never ages.** A
segment starting at ``age0`` with a drawn age-at-failure ``T`` fails at
``T - age0``. The loop compares against year boundaries, so holding an age
instead would be wrong by ``age0``, which ranges over decades here.

**A replacement enters service at the start of the following year.** A segment
replaced in year ``y`` is age 0 when year ``y + 1`` is scored. The age it
carries for the rest of year ``y`` is never read: it cannot fail again, because
its next failure time is at least ``y + 1``, and it cannot be planned work,
because it has already been replaced this year. That is what keeps the year
loop free of any inner iteration, and what makes the deterministic parity test
terminate when lifetimes are forced to zero.
"""

from typing import NamedTuple

import numpy as np

from cablesim import policies, weibull


class Results(NamedTuple):
    """One chunk of replications, per year and per segment class.

    Every array is ``(replications, years, classes)``. The class axis is
    returned rather than summed away because failures are read by class and a
    system total cannot be decomposed afterwards.

    Both a count of customers and a duration-weighted sum of customer-minutes
    are returned, because one cannot be recovered from the other once
    restoration time varies by class: the interruption frequency index needs
    the count and the duration index needs the minutes.

    Attributes:
        failures: Segments that failed.
        customers_interrupted: Customers out, counted per failure.
        customer_minutes: Customer-minutes lost to failures.
        planned_customer_minutes: Customer-minutes lost to planned work, which
            enters no reliability index and is reported on its own.
        planned_replacements: Segments replaced as planned work.
        planned_spend: Dollars spent on planned work, nominal.
        emergency_spend: Dollars spent replacing failures, nominal.
    """

    failures: np.ndarray
    customers_interrupted: np.ndarray
    customer_minutes: np.ndarray
    planned_customer_minutes: np.ndarray
    planned_replacements: np.ndarray
    planned_spend: np.ndarray
    emergency_spend: np.ndarray


def running_total(values: np.ndarray) -> float:
    """Adds an array one element at a time, left to right.

    ``sum`` is free to add pairwise, which is faster and gives a different
    answer in the last bits. That difference is tolerable wherever the result
    is only reported, and is not tolerable where it feeds a comparison that
    decides a discrete outcome: both places this is used reach the greedy
    budget fill, where a last-bit difference changes which segment is the last
    one funded. Accumulating left to right is also what a running total in
    another language does, so it is what makes the two implementations agree
    exactly rather than approximately.

    Args:
        values: What to add. May be empty.

    Returns:
        The total, or 0.0 for an empty array.
    """
    if values.size == 0:
        return 0.0
    return float(np.cumsum(values)[-1])


def run_chunk(
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
) -> Results:
    """Runs one chunk of replications under one policy.

    Args:
        length_ft: Segment length, in feet.
        customers: Customers served, counted equally for the frequency index.
        customer_minutes_per_failure: Customer-minutes lost when this segment
            fails, already carrying its class's restoration time.
        customer_minutes_per_planned: The same for planned work, zero for a
            class that is switched out without interrupting anyone.
        outage_cost_per_failure: Value of lost load if this segment fails, in
            dollars at year-0 prices.
        class_index: Which segment class each segment belongs to, indexing the
            third axis of the returned arrays.
        age0: Age at the start of the run, in years.
        shape: Weibull shape for the cable in the ground, effective.
        scale: Weibull scale for the cable in the ground, effective.
        replacement_shape: Weibull shape a replacement would take, effective
            for this segment's own geometry.
        replacement_scale: The same for scale.
        cost_per_ft: Installed cost per foot.
        lifetime_uniforms: ``(replications, segments, n_years + 1)`` uniforms.
            Index 0 is the left-truncated draw made at the start of the run,
            and a replacement made in year ``y`` reads index ``y + 1``.
        policy_uniforms: ``(replications, segments)``, one fixed priority per
            segment, which only the random policy ranks on.
        budget: Planned capital per year, already escalated.
        cost_escalation: Per-year multiplier applied to every dollar quantity.
        policy: The resolved replacement policy.
        emergency_multiplier: What replacing a failure costs relative to the
            same work planned.
        mobilization_per_segment: Fixed cost of turning up at all.
        emergency_charged_to_budget: Charge the year's emergency spend against
            the planned budget before scoring planned work.
        n_classes: Number of segment classes, sizing the third result axis.
        n_years: Horizon, in years.

    Returns:
        The seven per-year, per-class arrays for this chunk.

    Raises:
        ValueError: If the draw array's shape is not ``(replications,
            segments, n_years + 1)``. The year axis is why the check exists: a
            short one raises on its own only when a replacement happens to fall
            in the final year, so a run can complete against a wrong array and
            be wrong nowhere visible. The other two axes are checked with it
            because they cost nothing to compare.
    """
    n_reps, n_segments = policy_uniforms.shape
    if lifetime_uniforms.shape != (n_reps, n_segments, n_years + 1):
        raise ValueError(
            f"lifetime_uniforms is {lifetime_uniforms.shape}, expected "
            f"{(n_reps, n_segments, n_years + 1)}: one draw per segment per "
            f"year, plus the left-truncated draw at index 0"
        )

    results = Results(
        *(np.zeros((n_reps, n_years, n_classes)) for _ in Results._fields)
    )
    # Costs at year-0 prices; the year's escalation is applied inside the loop.
    planned_at_par = policies.planned_cost(
        length_ft, cost_per_ft, mobilization_per_segment
    )
    # `bincount` wants a plain integer index, and the class axis arrives as the
    # narrowest type that holds it.
    bins = class_index.astype(np.intp)

    def by_class(selected: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        """Totals a quantity over the selected segments, per class.

        Args:
            selected: Indices or a mask selecting the segments to total.
            weights: What to add per segment, or None to count them.

        Returns:
            One total per class.
        """
        return np.bincount(bins[selected], weights=weights, minlength=n_classes)

    for replication in range(n_reps):
        age = age0.astype(float, copy=True)
        current_shape = shape.astype(float, copy=True)
        current_scale = scale.astype(float, copy=True)
        priority = policy_uniforms[replication]
        # Conditional on survival to age0: a population that starts partway
        # through its life must not behave as though it were new.
        failure_time = weibull.draw_remaining_life(
            lifetime_uniforms[replication, :, 0], age, current_shape, current_scale
        )

        for year in range(n_years):
            escalation = cost_escalation[year]
            planned_now = planned_at_par * escalation

            # 1. Failures, which are resolved before planned work so that a
            #    segment failing this year is not also a candidate this year.
            failed = (failure_time >= year) & (failure_time < year + 1)
            emergency_now = planned_now[failed] * emergency_multiplier
            results.failures[replication, year] += by_class(failed)
            results.customers_interrupted[replication, year] += by_class(
                failed, customers[failed]
            )
            results.customer_minutes[replication, year] += by_class(
                failed, customer_minutes_per_failure[failed]
            )
            results.emergency_spend[replication, year] += by_class(
                failed, emergency_now
            )

            replaced = failed.copy()

            # 2. Planned replacement, funded greedily down the ranked order.
            available = budget[year]
            if emergency_charged_to_budget:
                # Charged before this year's planned pass is scored, which is
                # what produces the loop where failures crowd out prevention.
                # The floor changes no funding decision — every planned cost is
                # positive, so the greedy fill funds nothing at zero or below —
                # and it stops a year whose failures cost more than the budget
                # from carrying a negative that reads as a debt.
                #
                # `cumsum` rather than `sum`, for the reason the greedy fill
                # uses it: this total is subtracted from the budget that the
                # fill then compares a cumulative cost against, so a last-bit
                # difference here decides which segment is funded last. `sum`
                # is free to add pairwise, and on a few hundred uneven costs it
                # disagrees with a running total often enough to change that
                # decision — measured, the two orders differ in the last bits
                # for most years with more than a handful of failures.
                available = max(0.0, available - running_total(emergency_now))

            candidates = np.flatnonzero(policies.eligible(policy, age, replaced))
            if candidates.size > 0:
                rank = policies.rank_key(
                    policy,
                    age=age,
                    failure_probability=weibull.conditional_failure_probability(
                        age, current_shape, current_scale
                    ),
                    outage_cost_per_failure=outage_cost_per_failure * escalation,
                    planned=planned_now,
                    emergency_multiplier=emergency_multiplier,
                    priority=priority,
                )
                funded = policies.fund(
                    policies.order_by_rank(rank, candidates), planned_now, available
                )
                results.planned_replacements[replication, year] += by_class(funded)
                results.planned_customer_minutes[replication, year] += by_class(
                    funded, customer_minutes_per_planned[funded]
                )
                results.planned_spend[replication, year] += by_class(
                    funded, planned_now[funded]
                )
                replaced[funded] = True

            # 3. Everything replaced this year enters service next year, as new
            #    cable of the replacement technology, with a lifetime drawn
            #    from that segment's cell for the following year.
            if replaced.any():
                current_shape[replaced] = replacement_shape[replaced]
                current_scale[replaced] = replacement_scale[replaced]
                failure_time[replaced] = (year + 1) + weibull.draw_lifetime(
                    lifetime_uniforms[replication, :, year + 1][replaced],
                    replacement_shape[replaced],
                    replacement_scale[replaced],
                )
            age += 1.0
            age[replaced] = 0.0

    return results
