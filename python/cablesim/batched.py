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


# Column names the polars year loop reads and writes. Named once here rather
# than spelled at every use, because a typo in a string is a lookup failure at
# run time where a typo in a name is caught before anything runs. `batched.rs`
# keeps the same list for the same reason, and the two must agree: they are the
# same loop over the same frame in two languages.
REP = "rep"
SEGMENT_ID = "segment_id"
CLASS = "class_index"
CUSTOMERS = "customers"
MINUTES_FAILURE = "customer_minutes_per_failure"
MINUTES_PLANNED = "customer_minutes_per_planned"
OUTAGE_COST = "outage_cost_per_failure"
PLANNED_AT_PAR = "planned_at_par"
REPLACEMENT_SHAPE = "replacement_shape"
REPLACEMENT_SCALE = "replacement_scale"
AGE = "age"
CURRENT_SHAPE = "current_shape"
CURRENT_SCALE = "current_scale"
FAILURE_TIME = "failure_time"
PRIORITY = "priority"
PLANNED_NOW = "planned_now"
FAILED = "failed"

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


def summed(column: str) -> pl.Expr:
    """Totals a column within a group, one row at a time in the frame's order.

    Args:
        column: The column to total.

    Returns:
        The expression giving that group's total.
    """
    return pl.col(column).cum_sum().last()


def counted() -> pl.Expr:
    """Counts the rows in a group.

    A count needs no ordering argument: a sum of ones is exact whatever order it
    is taken in, which is why this is the group's length rather than a running
    total of a literal. Writing it as one would also be wrong rather than merely
    unnecessary — a bare literal is a scalar, and a cumulative sum of a scalar
    is that scalar, so every group would count as one.

    Returns:
        The expression giving that group's row count as a float.
    """
    return pl.len().cast(pl.Float64)


YEAR = "year"
"""The year a row's contribution belongs to, carried on the retained rows."""


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

    ``rep`` is marked sorted, which it is: it is built by repeating each
    replication index and the frame is never permuted afterwards. The flag is
    what lets the ranking sort and the windowed cumulative sum below take their
    contiguous-group paths, and neither is derived by the engine on its own.

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
    age = np.tile(segments["age0"].astype(float), n_reps)
    current_shape = np.tile(segments["shape"], n_reps)
    current_scale = np.tile(segments["scale"], n_reps)
    return pl.DataFrame(
        {
            REP: np.repeat(np.arange(n_reps, dtype=np.int32), n_segments),
            SEGMENT_ID: np.tile(np.arange(n_segments, dtype=np.int32), n_reps),
            **repeated,
            AGE: age,
            CURRENT_SHAPE: current_shape,
            CURRENT_SCALE: current_scale,
            PRIORITY: policy_uniforms.ravel(),
            # Conditional on survival to the starting age, exactly as the
            # reference draws it: the same function, on the arrays that became
            # the columns rather than on the columns, so the frame is built once
            # instead of built, extended and then trimmed.
            FAILURE_TIME: weibull.draw_remaining_life(
                first_uniforms.ravel(), age, current_shape, current_scale
            ),
        }
    ).with_columns(pl.col(REP).set_sorted())


def accumulate(
    results: simulate.Results,
    contributions: list[pl.DataFrame],
    quantities: dict[str, pl.Expr],
) -> None:
    """Totals every year's contributing rows at once and writes them in.

    The rows are consumed in the order they are held, which is the whole reason
    ordering is managed by the caller: the failure rows arrive in segment order
    and the funded rows in rank order, matching what the reference adds and in
    what sequence. Grouping on ``(rep, year, class)`` puts each year's rows in
    their own groups, so one grouped pass over every year accumulates exactly
    what a pass per year accumulated — the same rows, in the same order, within
    each group.

    **One grouped pass rather than thirty.** A grouped total costs about 1.3 ms
    of engine dispatch however few rows it covers (measured: 8,700 rows,
    48 threads, four aggregations), and a year contributes a few thousand rows
    out of 600,000. Deferring costs the retained rows — five columns over
    roughly 260,000 rows for the failures and 170,000 for the funded work,
    about 16 MB together at 50 replications — and saves about 1.1 ms per
    replication.

    **A cumulative sum's last element rather than a grouped sum**, because a
    grouped sum is free to add in whatever order suits the engine and does:
    with `sum`, the reported planned and emergency spend differ from the
    reference in their last bits — around 1e-16 relative — on every policy,
    while the counts stay exact because a sum of integers is exact in any
    order. `maintain_order` does not fix it, because it fixes the order the
    groups come back in rather than what accumulates inside one.

    Args:
        results: The arrays to write into, in place.
        contributions: One frame per year, each carrying `rep`, `year`,
            `class_index` and the columns the expressions read, with rows in the
            order the totals must accumulate. May be empty, which is what a
            policy funding nothing produces.
        quantities: One expression per reported quantity, keyed by the name of
            the result array it fills.
    """
    if not contributions:
        return
    totals = (
        pl.concat(contributions)
        .group_by([REP, YEAR, CLASS], maintain_order=True)
        .agg([expression.alias(name) for name, expression in quantities.items()])
    )
    index = (
        totals[REP].to_numpy(),
        totals[YEAR].to_numpy(),
        totals[CLASS].to_numpy(),
    )
    for name in quantities:
        getattr(results, name)[index] = totals[name].to_numpy()


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

    **Only the rows that contribute are ever carried.** A year's failures are a
    percent or two of the population and a year's candidates can be as few,
    which is where a long frame beats a rectangular one rather than merely
    matching it: the batched NumPy loop scores and sorts every segment of every
    replication because its candidate rows differ between replications and its
    array cannot be ragged, and a frame answers that by having fewer rows. The
    year's escalated costs and the policy score are therefore computed after
    the filter that selects those rows, not before it, which is the same values
    over a twentieth of the elements when a policy is selective.

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
            its own and not something a caller sizes here. **Its pool is sized
            from available parallelism and that is not its best setting on a
            frame this small** — measured on this workload, everything but the
            ranking sort runs about a fifth faster on four to sixteen threads
            than on forty-eight, while the sort itself wants all of them. The
            pool is fixed when polars is imported and a library must not
            reconfigure its host's, so a driver that wants a different one sets
            ``POLARS_MAX_THREADS`` before importing, and
            ``benchmarks.provenance`` records what the pool actually was.

    Returns:
        The seven per-year, per-class arrays for this chunk.

    Raises:
        ValueError: Everything `simulate.check_arguments` refuses, which is
            what makes this interchangeable with the other implementations, and
            a rank key that is not a number.
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
    # One row per frame row, one column per year. The draw array is contiguous
    # and the frame's row order is replication-major, so this is a reshaped view
    # rather than a copy, and a year's draws for the replaced rows are then one
    # gather at `[renewing, year + 1]`. Slicing the year out first —
    # `lifetime_uniforms[:, :, year + 1]` — copies 4.8 MB per year at 50
    # replications because that slice is not contiguous, and measured on this
    # shape the copy-then-gather costs 3.5 ms against this gather's 0.03 ms.
    draws = lifetime_uniforms.reshape(n_reps * n_segments, n_years + 1)
    # The replacement parameters as flat arrays in the frame's own row order,
    # for the renewal draw. That draw runs over the replaced rows alone, and a
    # gather wants an array rather than a column.
    replacement_shape_all = np.tile(replacement_shape, n_reps)
    replacement_scale_all = np.tile(replacement_scale, n_reps)

    failing_years: list[pl.DataFrame] = []
    funded_years: list[pl.DataFrame] = []
    for year in range(n_years):
        escalation = cost_escalation[year]

        # 1. Failures, resolved before planned work so that a segment failing
        #    this year is not also a candidate this year.
        #
        # One pass: the year's test, the filter it selects with, and this
        # year's prices on the rows that survive it. `planned_at_par *
        # escalation` after the filter is the same product on the same rows as
        # before it, and a percent or two of the rows.
        failing = (
            state.lazy()
            .select(
                REP,
                SEGMENT_ID,
                CLASS,
                CUSTOMERS,
                MINUTES_FAILURE,
                PLANNED_AT_PAR,
                FAILURE_TIME,
            )
            .filter((pl.col(FAILURE_TIME) >= year) & (pl.col(FAILURE_TIME) < year + 1))
            .select(
                REP,
                SEGMENT_ID,
                CLASS,
                CUSTOMERS,
                MINUTES_FAILURE,
                (pl.col(PLANNED_AT_PAR) * escalation).alias(PLANNED_NOW),
            )
            .collect()
        )
        failing_years.append(
            failing.drop(SEGMENT_ID).with_columns(year=pl.lit(year, dtype=pl.Int32))
        )

        # The state frame is never permuted, so a row's position is still
        # `replication * segments + segment_id`, and that is what marks the
        # failed and the funded rows without a join back.
        replaced = np.zeros(n_reps * n_segments, dtype=bool)
        replaced[
            failing[REP].to_numpy() * n_segments + failing[SEGMENT_ID].to_numpy()
        ] = True

        # 2. Planned replacement, funded greedily down the ranked order.
        available = np.full(n_reps, budget[year])
        if emergency_charged_to_budget:
            # Accumulated one row at a time within a replication, in segment
            # order, which is the order the reference adds it in. A grouped sum
            # would add in whatever order suits the engine, and this total is
            # subtracted from the budget the greedy fill then compares a
            # cumulative cost against, so a last-bit difference here decides
            # which segment is funded last.
            #
            # A year whose failures cost more than the budget leaves this
            # negative, and it is left negative: every planned cost is
            # positive, so nothing is funded at or below zero and nothing
            # carries into the next year.
            charged = (
                failing.lazy()
                .with_columns(pl.col(REP).set_sorted())
                .with_columns(
                    running=(pl.col(PLANNED_NOW) * emergency_multiplier)
                    .cum_sum()
                    .over(REP)
                )
                .group_by(REP)
                .agg(charge=pl.col("running").last())
                .collect()
            )
            available[charged[REP].to_numpy()] -= charged["charge"].to_numpy()

        eligible = policies.eligible(policy, state[AGE].to_numpy(), replaced)
        if eligible.any():
            # The candidate rows and the columns the score reads, and nothing
            # else. Scoring runs over these rows rather than over the frame:
            # `rank_key` and `conditional_failure_probability` are elementwise,
            # so a row's score does not depend on which other rows are present,
            # and under `age_threshold` two percent of rows are candidates.
            candidates = (
                state.lazy()
                .select(
                    REP,
                    SEGMENT_ID,
                    CLASS,
                    MINUTES_PLANNED,
                    PLANNED_AT_PAR,
                    AGE,
                    CURRENT_SHAPE,
                    CURRENT_SCALE,
                    OUTAGE_COST,
                    PRIORITY,
                )
                .filter(pl.Series(eligible))
                .with_columns(planned_now=pl.col(PLANNED_AT_PAR) * escalation)
                .collect()
            )
            rank = policies.rank_key(
                policy,
                age=candidates[AGE],
                failure_probability=weibull.conditional_failure_probability(
                    candidates[AGE],
                    candidates[CURRENT_SHAPE],
                    candidates[CURRENT_SCALE],
                ),
                outage_cost_per_failure=candidates[OUTAGE_COST] * escalation,
                planned=candidates[PLANNED_NOW],
                emergency_multiplier=emergency_multiplier,
                priority=candidates[PRIORITY],
            )
            # `rank_key` hands back a polars Series for every policy that
            # reads a column and a NumPy array for the one that scores zero, and
            # `isnan` does not dispatch on the former. Asking for an array is
            # free where the column is contiguous, which it is here.
            unranked = np.flatnonzero(np.isnan(np.asarray(rank)))
            if unranked.size > 0:
                # Reported the way the reference reports it, since a caller
                # running both must not get a different diagnosis from each.
                #
                # Left to itself this implementation would give an answer:
                # polars sorts NaN *first* under a descending key, so a
                # candidate that scored one would be funded ahead of every real
                # candidate rather than reported. Every input to the score is
                # finite by construction, so that is a defect upstream showing
                # up as a segment inexplicably funded.
                first = int(candidates[SEGMENT_ID][int(unranked[0])])
                raise ValueError(
                    f"{unranked.size} candidate segments scored NaN, first at "
                    f"segment_id {first}; every input to the score is "
                    f"finite by construction, so this is a defect upstream of "
                    f"ranking"
                )
            candidates = (
                candidates.lazy()
                .select(REP, SEGMENT_ID, CLASS, MINUTES_PLANNED, PLANNED_NOW)
                .with_columns(rank=pl.Series(rank))
                .sort([REP, "rank", SEGMENT_ID], descending=[False, True, False])
                # True by construction — `rep` is the sort's ascending first
                # key — and asserted because the engine does not carry the flag
                # through a sort. The windowed cumulative sum below takes its
                # contiguous-group path only with it: measured on 594,000 rows,
                # 11.6 ms without against 3.1 ms with, to the same bits.
                .with_columns(pl.col(REP).set_sorted())
                # The cumulative cost down each replication's ranked order,
                # which is the greedy fill: one partial sum at a time, in the
                # order the reference spends the money.
                .with_columns(running=pl.col(PLANNED_NOW).cum_sum().over(REP))
                .collect()
            )
            # A candidate costing exactly what remains is funded: the rule is
            # that spending may not exceed the budget, not that it must fall
            # short. Every planned cost is positive, so the running total only
            # rises and this cut is a prefix of the ranked order.
            funded = candidates.filter(
                pl.col("running") <= available[candidates[REP].to_numpy()]
            )
            if funded.height > 0:
                # Still in rank order, which is the order the reference adds the
                # funded segments in and therefore the order these have to
                # accumulate in.
                funded_years.append(
                    funded.select(
                        REP, CLASS, MINUTES_PLANNED, PLANNED_NOW
                    ).with_columns(year=pl.lit(year, dtype=pl.Int32))
                )
                replaced[
                    funded[REP].to_numpy() * n_segments + funded[SEGMENT_ID].to_numpy()
                ] = True

        # 3. Everything replaced this year enters service next year, as new
        #    cable of the replacement technology.
        renewing = np.flatnonzero(replaced)
        failure_time = state[FAILURE_TIME].to_numpy().copy()
        if renewing.size > 0:
            # Drawn at the replaced rows alone. Computing it for every row and
            # selecting afterwards draws a lifetime for the ninety-seven percent
            # that keep the one they have.
            failure_time[renewing] = (year + 1) + weibull.draw_lifetime(
                draws[renewing, year + 1],
                replacement_shape_all[renewing],
                replacement_scale_all[renewing],
            )
        renewed = pl.Series(replaced)
        state = state.with_columns(
            failure_time=pl.Series(failure_time),
            current_shape=pl.when(renewed)
            .then(pl.col(REPLACEMENT_SHAPE))
            .otherwise(pl.col(CURRENT_SHAPE)),
            current_scale=pl.when(renewed)
            .then(pl.col(REPLACEMENT_SCALE))
            .otherwise(pl.col(CURRENT_SCALE)),
            age=pl.when(renewed).then(0.0).otherwise(pl.col(AGE) + 1.0),
        )

    results = simulate.Results(
        *(np.zeros((n_reps, n_years, n_classes)) for _ in simulate.Results._fields)
    )
    accumulate(
        results,
        failing_years,
        {
            "failures": counted(),
            "customers_interrupted": summed(CUSTOMERS),
            "customer_minutes": summed(MINUTES_FAILURE),
            "emergency_spend": (pl.col(PLANNED_NOW) * emergency_multiplier)
            .cum_sum()
            .last(),
        },
    )
    accumulate(
        results,
        funded_years,
        {
            "planned_customer_minutes": summed(MINUTES_PLANNED),
            "planned_replacements": counted(),
            "planned_spend": summed(PLANNED_NOW),
        },
    )
    return results
