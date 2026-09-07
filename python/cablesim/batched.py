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
import polars as pl

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
        over segments because the totals taken from it have to accumulate in
        rank order to match the reference in the last bits, and because reading
        the funded positions out of it gives that order directly.
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


def totals_by_class(
    bins: np.ndarray,
    n_reps: int,
    n_classes: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Totals a quantity per replication and per class.

    One `bincount` over every replication at once, with each replication's
    class bins already offset by its own position. Called per replication
    instead, this would add the same values in the same order; called as a
    grouping it might not, and the order is what keeps this equal to the
    reference in the last bits.

    Args:
        bins: The offset class bin of each contributing cell, **in the order
            the totals should accumulate**. The caller decides that order:
            segment order for what failures contribute, rank order for what
            funded work contributes.
        n_reps: Replications in this chunk.
        n_classes: Number of segment classes.
        weights: What each contributor adds, in the same order, or None to
            count them.

    Returns:
        ``(replications, n_classes)`` totals.
    """
    return np.bincount(bins, weights=weights, minlength=n_reps * n_classes).reshape(
        n_reps, n_classes
    )


def gathered(values: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Reads a per-segment or per-cell quantity at the contributing cells only.

    A few percent of segments fail or are funded in a year, so reading the whole
    array and masking afterwards does twenty times the work for the same
    numbers. This implementation is the baseline a speedup is claimed against,
    so anything gratuitously slow in it flatters the claim.

    Args:
        values: ``(segments,)`` or ``(replications, segments)``.
        rows: Replication index of each contributing cell.
        columns: Segment index of each contributing cell.

    Returns:
        One value per contributing cell, in the order they were given.
    """
    if values.ndim == 1:
        return values[columns]
    return values[rows, columns]


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
        # `nonzero` returns the cells row by row, so these are in replication
        # order and then segment order — which is the order the reference visits
        # them in, and therefore the order its totals accumulate in.
        failed_rows, failed_columns = np.nonzero(failed)
        failed_bins = bins[failed_rows, failed_columns]
        failed_cost = (
            gathered(planned_now, failed_rows, failed_columns) * emergency_multiplier
        )
        results.failures[:, year] += totals_by_class(failed_bins, n_reps, n_classes)
        results.customers_interrupted[:, year] += totals_by_class(
            failed_bins,
            n_reps,
            n_classes,
            gathered(customers, failed_rows, failed_columns),
        )
        results.customer_minutes[:, year] += totals_by_class(
            failed_bins,
            n_reps,
            n_classes,
            gathered(customer_minutes_per_failure, failed_rows, failed_columns),
        )
        results.emergency_spend[:, year] += totals_by_class(
            failed_bins, n_reps, n_classes, failed_cost
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
            # Scattered back to full width first, so the cumulative sum runs
            # over every segment with the survivors contributing zero. Adding
            # `0.0` returns a float unchanged, so that totals the failures in
            # the same order and to the same last bit as a running total over
            # the failed ones alone would — and unlike a difference of two
            # cumulative sums, which is not the same arithmetic. The full-width
            # pass is paid only when the budget is charged, which the shipped
            # configuration does not do.
            charged = np.zeros((n_reps, n_segments))
            charged[failed_rows, failed_columns] = failed_cost
            available -= np.cumsum(charged, axis=1)[:, -1]

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
            #
            # `nonzero` walks the funded positions row by row and, within a row,
            # from the best-ranked position downwards — replication order and
            # then rank order, which is exactly the sequence the reference funds
            # in. Only the funded cells are read, rather than the whole array
            # rearranged and then masked.
            funded_rows, positions = np.nonzero(funded_positions)
            funded_columns = ordered[funded_rows, positions]
            funded_bins = bins[funded_rows, funded_columns]
            results.planned_replacements[:, year] += totals_by_class(
                funded_bins, n_reps, n_classes
            )
            results.planned_customer_minutes[:, year] += totals_by_class(
                funded_bins,
                n_reps,
                n_classes,
                gathered(customer_minutes_per_planned, funded_rows, funded_columns),
            )
            results.planned_spend[:, year] += totals_by_class(
                funded_bins,
                n_reps,
                n_classes,
                gathered(planned_now, funded_rows, funded_columns),
            )
            replaced[funded_rows, funded_columns] = True

        # 3. Everything replaced this year enters service next year, as new
        #    cable of the replacement technology.
        #
        # Written to the replaced cells only, rather than computed for every
        # cell and selected between. A few percent of segments are replaced in
        # a year, so drawing a lifetime for all of them costs the transcendental
        # in `draw_lifetime` about fifty times over for one time it is used. It
        # is also what the reference does, which is the point: this
        # implementation is the baseline a speedup is claimed against, so where
        # it is gratuitously slower than the reference the claim is flattered.
        if replaced.any():
            rows, columns = np.nonzero(replaced)
            current_shape[rows, columns] = replacement_shape[columns]
            current_scale[rows, columns] = replacement_scale[columns]
            failure_time[rows, columns] = (year + 1) + weibull.draw_lifetime(
                lifetime_uniforms[rows, columns, year + 1],
                replacement_shape[columns],
                replacement_scale[columns],
            )
        # Two passes rather than a `where`, in place: the reference does the
        # same, and allocating a fresh array per year is the cost this
        # implementation exists to avoid.
        age += 1.0
        age[replaced] = 0.0

    return results


QUANTITIES: tuple[str, ...] = simulate.Results._fields
"""The seven reported quantities, in the order a result holds them."""

FAILURE_QUANTITIES: tuple[str, ...] = (
    "failures",
    "customers_interrupted",
    "customer_minutes",
    "emergency_spend",
)
"""What a year's failures contribute, totalled over segments in segment order."""

PLANNED_QUANTITIES: tuple[str, ...] = (
    "planned_replacements",
    "planned_customer_minutes",
    "planned_spend",
)
"""What a year's funded work contributes, totalled in rank order.

Split from the failure quantities because the two are accumulated over
differently ordered rows, which is what keeps both equal to the reference in
their last bits rather than only close to it.
"""


def state_frame(
    n_reps: int,
    segments: dict[str, np.ndarray],
    policy_uniforms: np.ndarray,
    first_uniforms: np.ndarray,
) -> pl.DataFrame:
    """Builds the long frame the polars loop carries, one row per replication
    and segment.

    Rows are ordered by replication and then by segment, which is the order the
    draw arrays are laid out in and the order the reference visits segments in.
    Every later step either preserves that order or sorts back to it, because
    the uniforms for each year are attached positionally.

    Args:
        n_reps: Replications in this chunk.
        segments: The per-segment arrays, keyed by column name.
        policy_uniforms: ``(replications, segments)`` fixed priorities.
        first_uniforms: ``(replications, segments)`` uniforms for the
            left-truncated draw.

    Returns:
        The starting state, with the first failure time already drawn.
    """
    n_segments = segments["age0"].size
    repeated = {
        name: np.tile(column, n_reps)
        for name, column in segments.items()
        if name != "age0"
    }
    frame = pl.DataFrame(
        {
            "rep": np.repeat(np.arange(n_reps, dtype=np.int32), n_segments),
            "segment_id": np.tile(np.arange(n_segments, dtype=np.int32), n_reps),
            **repeated,
            "age": np.tile(segments["age0"].astype(float), n_reps),
            "current_shape": np.tile(segments["shape"], n_reps),
            "current_scale": np.tile(segments["scale"], n_reps),
            "priority": policy_uniforms.ravel(),
            "first_uniform": first_uniforms.ravel(),
        }
    )
    # Conditional on survival to the starting age, exactly as the reference
    # draws it: the same function, on a column rather than on an array.
    return frame.with_columns(
        failure_time=weibull.draw_remaining_life(
            frame["first_uniform"],
            frame["age"],
            frame["current_shape"],
            frame["current_scale"],
        )
    ).drop("first_uniform")


def totals_per_class(
    frame: pl.DataFrame, quantities: dict[str, pl.Expr]
) -> pl.DataFrame:
    """Totals each quantity per replication and per segment class.

    The rows are consumed in the order the frame holds them, which is the whole
    reason ordering is managed by the caller: the failure quantities are
    totalled over a frame in segment order and the funded ones over a frame in
    rank order, matching what the reference adds and in what sequence.

    **A cumulative sum's last element rather than a grouped sum**, because a
    grouped sum is free to add in whatever order suits the engine and does:
    with `sum`, the reported planned and emergency spend differ from the
    reference in their last bits — around 1e-16 relative — on every policy,
    while the counts stay exact because a sum of integers is exact in any
    order.

    Three things that look like they would fix it do not, and the measurements
    are worth keeping because each is a plausible guess:

    * **`maintain_order` is about the groups, not the sum.** It fixes the order
      the groups come back in. What accumulates inside one is unaffected, and
      the two are independent settings of the same call.
    * **Sorting the rows first makes it worse.** A sum over a frame that has
      been sorted differs from a running total over the same rows at *any*
      thread count, and `rechunk` does not restore it. The rows are already in
      the wanted order here, so this was never going to help; it is worth
      recording that it actively hurts.
    * **One thread is a fix that cannot be used.** Totalling a group in a
      thread pool of one does match a running total exactly, and with polars
      held to a single thread the `run_to_failure` results here come back
      exact. Every other policy still differs, because their funded totals are
      taken over the rank-sorted frame and hit the case above. Even where it
      worked, a single-threaded polars is not the thing this implementation
      exists to measure.

    A cumulative sum is the operation that is *defined* by its order, so it is
    the one that keeps it — exactly, at every thread count, sorted or not.

    Forcing the order costs 1.7% of this implementation's runtime (2,000
    segments, 20 replications, ranking by risk), which is small enough that
    holding every implementation to exact agreement is cheaper than maintaining
    a tolerance and arguing about what could hide beneath it.

    Args:
        frame: Rows in the order the totals should accumulate.
        quantities: One expression per reported quantity, each already zero
            where the row does not contribute.

    Returns:
        One row per replication and class, with a column per quantity.
    """
    return frame.group_by(["rep", "class_index"], maintain_order=True).agg(
        [
            expression.cum_sum().last().alias(name)
            for name, expression in quantities.items()
        ]
    )


def as_result_arrays(
    yearly: list[pl.DataFrame], n_reps: int, n_years: int, n_classes: int
) -> simulate.Results:
    """Scatters the per-year totals into the seven reported arrays.

    Done once at the end rather than per year, so the year loop holds only
    frame operations. A replication and class that contributed nothing in a
    year has no row, and reads back as the zero the reference reports.

    Args:
        yearly: One frame per year, each carrying `rep`, `class_index`, `year`
            and a column per quantity.
        n_reps: Replications in this chunk.
        n_years: Horizon, in years.
        n_classes: Number of segment classes.

    Returns:
        The seven ``(replications, years, classes)`` arrays.
    """
    gathered = pl.concat(yearly)
    results = simulate.Results(
        *(np.zeros((n_reps, n_years, n_classes)) for _ in QUANTITIES)
    )
    index = (
        gathered["rep"].to_numpy(),
        gathered["year"].to_numpy(),
        gathered["class_index"].to_numpy(),
    )
    for name, array in zip(QUANTITIES, results, strict=True):
        array[index] = gathered[name].to_numpy()
    return results


def run_chunk_polars(
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
    """Runs one chunk of replications under one policy, over a polars frame.

    State is one long frame of `replications * segments` rows rather than a
    two-dimensional array, and the two steps that are per-replication become
    frame operations: ranking is a sort, and the greedy fill is a cumulative
    sum within a replication compared against that replication's budget.

    **This is not purely polars, and that is not a shortcut.** The uniforms
    arrive from NumPy because polars has no addressable per-element generator,
    and every implementation here must read the identical draws for a
    comparison between them to be paired. The Weibull transforms and the policy
    scoring are the same functions the reference calls: they are elementwise,
    and NumPy's ufunc protocol lets a polars column through them unchanged, so
    the formulas are written once rather than once per implementation.

    What is genuinely measured is therefore the frame work — the sort, the
    windowed cumulative sum, and the grouped totals — against the same work
    expressed as array indexing.

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
            has one and refuses any other value; polars parallelizes inside an
            operation rather than over replications, and that parallelism is
            its own and not something a caller sizes here.

    Returns:
        The seven per-year, per-class arrays for this chunk.

    Raises:
        ValueError: Everything `simulate.check_arguments` refuses, which is
            what makes this interchangeable with the other implementations.
        TypeError: If ``policy.kind`` is not an integer.
    """
    n_reps, n_segments = simulate.check_arguments(locals())

    state = state_frame(
        n_reps,
        {
            "class_index": class_index.astype(np.int32),
            "customers": customers,
            "customer_minutes_per_failure": customer_minutes_per_failure,
            "customer_minutes_per_planned": customer_minutes_per_planned,
            "outage_cost_per_failure": outage_cost_per_failure,
            "planned_at_par": policies.planned_cost(
                length_ft, cost_per_ft, mobilization_per_segment
            ),
            "replacement_shape": replacement_shape,
            "replacement_scale": replacement_scale,
            "age0": age0,
            "shape": shape,
            "scale": scale,
        },
        policy_uniforms,
        lifetime_uniforms[:, :, 0],
    )

    yearly: list[pl.DataFrame] = []
    for year in range(n_years):
        escalation = cost_escalation[year]

        # 1. Failures, resolved before planned work so that a segment failing
        #    this year is not also a candidate this year.
        state = state.with_columns(
            planned_now=pl.col("planned_at_par") * escalation,
        ).with_columns(
            failed=(pl.col("failure_time") >= year)
            & (pl.col("failure_time") < year + 1),
        )
        contribution = pl.when("failed")
        failures = totals_per_class(
            state,
            {
                "failures": contribution.then(1.0).otherwise(0.0),
                "customers_interrupted": contribution.then("customers").otherwise(0.0),
                "customer_minutes": contribution.then(
                    "customer_minutes_per_failure"
                ).otherwise(0.0),
                "emergency_spend": contribution.then(
                    pl.col("planned_now") * emergency_multiplier
                ).otherwise(0.0),
            },
        )

        # 2. Planned replacement, funded greedily down the ranked order.
        available = pl.DataFrame(
            {"rep": np.arange(n_reps, dtype=np.int32), "available": budget[year]}
        )
        if emergency_charged_to_budget:
            # Accumulated one row at a time within a replication, in segment
            # order, which is the order the reference adds it in. A grouped sum
            # is free to add in whatever order suits the engine, and this total
            # is subtracted from the budget the greedy fill then compares a
            # cumulative cost against — so a last-bit difference here decides
            # which segment is funded last, a discrete outcome rather than a
            # rounding difference.
            charged = (
                state.with_columns(
                    running_emergency=pl.when("failed")
                    .then(pl.col("planned_now") * emergency_multiplier)
                    .otherwise(0.0)
                    .cum_sum()
                    .over("rep")
                )
                .group_by("rep", maintain_order=True)
                .agg(charge=pl.col("running_emergency").last())
            )
            available = available.join(charged, on="rep").select(
                "rep",
                # Left negative where a year's failures cost more than the
                # budget, for the reason the reference leaves it negative:
                # every planned cost is positive, so nothing is funded at or
                # below zero and nothing carries into the next year.
                available=pl.col("available") - pl.col("charge"),
            )

        eligible = policies.eligible(policy, state["age"], state["failed"])
        planned = totals_per_class(
            state.clear(), dict.fromkeys(PLANNED_QUANTITIES, pl.lit(0.0))
        )
        if eligible.any():
            rank = policies.rank_key(
                policy,
                age=state["age"],
                failure_probability=weibull.conditional_failure_probability(
                    state["age"], state["current_shape"], state["current_scale"]
                ),
                outage_cost_per_failure=state["outage_cost_per_failure"] * escalation,
                planned=state["planned_now"],
                emergency_multiplier=emergency_multiplier,
                priority=state["priority"],
            )
            # Ineligible segments are demoted below every real rank key and
            # given an infinite cost, so the sort puts them behind the
            # candidates and the cumulative cost cannot run past the last real
            # one however much budget is left.
            ranked = (
                state.with_columns(
                    eligible=pl.Series(eligible),
                    rank=pl.Series(rank),
                )
                .with_columns(
                    demoted=pl.when("eligible").then(pl.col("rank")).otherwise(-np.inf),
                    step=pl.when("eligible")
                    .then(pl.col("planned_now"))
                    .otherwise(np.inf),
                )
                .sort(["rep", "demoted", "segment_id"], descending=[False, True, False])
                .with_columns(running=pl.col("step").cum_sum().over("rep"))
                .join(available, on="rep")
                .with_columns(funded=pl.col("running") <= pl.col("available"))
            )
            # Totalled here, while the rows are still in rank order, because
            # that is the order the reference adds the funded segments in.
            contribution = pl.when("funded")
            planned = totals_per_class(
                ranked,
                {
                    "planned_replacements": contribution.then(1.0).otherwise(0.0),
                    "planned_customer_minutes": contribution.then(
                        "customer_minutes_per_planned"
                    ).otherwise(0.0),
                    "planned_spend": contribution.then("planned_now").otherwise(0.0),
                },
            )
            # Back to the canonical order, which every later year's uniforms
            # are attached by position.
            state = ranked.sort(["rep", "segment_id"]).drop(
                "eligible", "rank", "demoted", "step", "running", "available"
            )
        else:
            state = state.with_columns(funded=pl.lit(False))

        yearly.append(
            failures.join(planned, on=["rep", "class_index"], how="left")
            # Only the funded columns: a class that funded nothing this year
            # has no row on that side, and reads back as the zero the
            # reference reports. Filling the whole frame would widen the two
            # integer key columns to floats along with it.
            .with_columns(pl.col(PLANNED_QUANTITIES).fill_null(0.0))
            .with_columns(year=pl.lit(year, dtype=pl.Int32))
        )

        # 3. Everything replaced this year enters service next year, as new
        #    cable of the replacement technology.
        state = state.with_columns(replaced=pl.col("failed") | pl.col("funded"))
        renewed = (year + 1) + weibull.draw_lifetime(
            pl.Series(lifetime_uniforms[:, :, year + 1].ravel()),
            state["replacement_shape"],
            state["replacement_scale"],
        )
        state = state.with_columns(
            current_shape=pl.when("replaced")
            .then("replacement_shape")
            .otherwise(pl.col("current_shape")),
            current_scale=pl.when("replaced")
            .then("replacement_scale")
            .otherwise(pl.col("current_scale")),
            failure_time=pl.when("replaced")
            .then(pl.lit(renewed))
            .otherwise(pl.col("failure_time")),
            age=pl.when("replaced").then(0.0).otherwise(pl.col("age") + 1.0),
        ).drop("failed", "funded", "replaced", "planned_now")

    return as_result_arrays(yearly, n_reps, n_years, n_classes)
