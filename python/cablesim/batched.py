"""The annual loop with every replication in flight at once.

This exists for the benchmark and for nothing else. `simulate.py` is the
correctness reference, and a speedup measured against it would be measured
against a program written to be read rather than to be fast. **This is the
baseline a speedup is honestly claimed against.**

Nothing outside the benchmark and the parity tests imports this module.

A polars implementation of the same loop was here and is now in `deprecated/`,
with the measurement that retired it and the reason a column store is the wrong
shape for a simulation.

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

import concurrent.futures

import numpy as np

from cablesim import policies, random_draws, simulate, weibull


def order_all_by_rank(rank: np.ndarray, eligible: np.ndarray) -> np.ndarray:
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


def gathered(values: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Reads a per-segment quantity at the contributing cells only.

    A few percent of segments fail or are funded in a year, so reading the whole
    array and masking afterwards does twenty times the work for the same
    numbers. This implementation is the baseline a speedup is claimed against,
    so anything gratuitously slow in it flatters the claim.

    Every quantity gathered here is per-segment and shared across replications,
    so only the segment of each cell is needed — the replication decides which
    result bin the value lands in, not which value it is.

    Args:
        values: ``(segments,)``.
        columns: Segment index of each contributing cell.

    Returns:
        One value per contributing cell, in the order they were given.
    """
    return values[columns]


def spread_over_threads(
    arguments: dict[str, object], threads: int
) -> simulate.Results:
    """Runs contiguous blocks of the replications on a pool, and joins them.

    Worth doing in an implementation whose arrays NumPy operates on one at a
    time: of the operations this loop uses, ``lexsort`` costs more than all the
    others together and releases the interpreter lock while it runs, as do
    ``argsort``, ``cumsum``, ``where`` and the elementwise arithmetic. The three
    that hold it — ``nonzero``, ``bincount``, ``isnan`` — are together about a
    fiftieth of the work.

    Splitting by replication is safe because a draw is computed from the run key
    and the position being read: a block produces its own draws with no
    coordination, and a replication reads identical values whichever worker runs
    it. The blocks are contiguous and joined in order, so the answer does not
    depend on the thread count or on which block finished first.

    Args:
        arguments: Every argument of ``run_chunk_numpy``, keyed by name, as
            ``locals()`` gives them on its first line.
        threads: Blocks to split into. More than there are replications would
            leave workers with nothing, so the count is capped at the
            replication count.

    Returns:
        The seven per-year, per-class arrays for the whole chunk.
    """
    n_reps = arguments["n_reps"]
    first_replication = arguments["first_replication"]
    blocks = min(threads, n_reps)
    edges = np.linspace(0, n_reps, blocks + 1).astype(int)

    def block(start: int, stop: int) -> simulate.Results:
        """Runs the replications in ``[start, stop)`` of this chunk.

        Args:
            start: First replication of the block, counted within the chunk.
            stop: One past its last.

        Returns:
            That block's seven arrays.
        """
        return run_chunk_numpy(
            **{
                **arguments,
                "first_replication": first_replication + start,
                "n_reps": stop - start,
                "threads": 1,
            }
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=blocks) as pool:
        produced = list(
            pool.map(block, edges[:-1], edges[1:])
        )
    return simulate.Results(
        *(np.concatenate(arrays) for arrays in zip(*produced, strict=True))
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
    draw_key: tuple[int, int],
    first_replication: int,
    n_reps: int,
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
        draw_key: The two key words every uniform is computed under. A draw
            is a function of where it sits rather than of how far a stream has
            been read.
        first_replication: Where this chunk starts in the run, so splitting a
            run into chunks changes no number.
        n_reps: Replications this chunk covers.
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
        threads: Workers to spread the replications over. Above one, the
            chunk is split into that many contiguous blocks of replications and
            each is run by ``spread_over_threads``.

    Returns:
        The seven per-year, per-class arrays for this chunk.

    Raises:
        ValueError: Everything `simulate.check_arguments` refuses, which is
            what makes this interchangeable with the other implementations, and
            a rank key that is not a number.
        TypeError: If ``policy.kind`` is not an integer.
    """
    arguments = locals()
    n_segments = simulate.check_arguments(arguments, concurrent=True)
    if threads > 1:
        return spread_over_threads(arguments, threads)

    results = simulate.Results(
        *(np.zeros((n_reps, n_years, n_classes)) for _ in simulate.Results._fields)
    )
    planned_at_par = policies.planned_cost(
        length_ft, cost_per_ft, mobilization_per_segment
    )
    # The working precision, read off the arrays this was handed rather than
    # passed down. Draws are produced in double whatever it is, and narrowed
    # where they meet the state.
    floating = age0.dtype

    # The fixed per-segment priority the random policy ranks on, read for every
    # segment of every replication, so there is nothing to select.
    priorities = random_draws.uniforms_dense(
        draw_key,
        random_draws.PURPOSE["policies"],
        first_replication,
        n_reps,
        n_segments,
        0,
    ).astype(floating)
    # Each replication's class bins offset into its own block, so one
    # `bincount` covers the chunk and still adds in segment order within a
    # class.
    bins = class_index.astype(np.intp) + n_classes * np.arange(n_reps)[:, None]

    # `(replications, segments)` state, which is the whole difference from the
    # reference. Broadcast rather than tiled where the starting value is the
    # same for every replication, then copied so the copies can diverge.
    age = np.broadcast_to(age0, (n_reps, n_segments)).copy()
    current_shape = np.broadcast_to(shape, (n_reps, n_segments)).copy()
    current_scale = np.broadcast_to(scale, (n_reps, n_segments)).copy()
    failure_time = weibull.draw_remaining_life(
        random_draws.uniforms_dense(
            draw_key,
            random_draws.PURPOSE["lifetimes"],
            first_replication,
            n_reps,
            n_segments,
            0,
        ).astype(floating),
        age,
        current_shape,
        current_scale,
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
            gathered(planned_now, failed_columns) * emergency_multiplier
        )
        results.failures[:, year] += totals_by_class(failed_bins, n_reps, n_classes)
        results.customers_interrupted[:, year] += totals_by_class(
            failed_bins,
            n_reps,
            n_classes,
            gathered(customers, failed_columns),
        )
        results.customer_minutes[:, year] += totals_by_class(
            failed_bins,
            n_reps,
            n_classes,
            gathered(customer_minutes_per_failure, failed_columns),
        )
        results.emergency_spend[:, year] += totals_by_class(
            failed_bins, n_reps, n_classes, failed_cost
        )

        replaced = failed.copy()

        # 2. Planned replacement, funded greedily down the ranked order.
        available = np.full(n_reps, budget[year], dtype=planned_at_par.dtype)
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
            charged = np.zeros((n_reps, n_segments), dtype=planned_at_par.dtype)
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
                priority=priorities,
            )
            ordered = order_all_by_rank(rank, eligible)
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
                gathered(customer_minutes_per_planned, funded_columns),
            )
            results.planned_spend[:, year] += totals_by_class(
                funded_bins,
                n_reps,
                n_classes,
                gathered(planned_now, funded_columns),
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
                random_draws.uniforms_at(
                    draw_key,
                    random_draws.PURPOSE["lifetimes"],
                    rows + first_replication,
                    columns,
                    year + 1,
                ).astype(floating),
                replacement_shape[columns],
                replacement_scale[columns],
            )
        # Two passes rather than a `where`, in place: the reference does the
        # same, and allocating a fresh array per year is the cost this
        # implementation exists to avoid.
        age += 1.0
        age[replaced] = 0.0

    return results
