"""The annual loop over a polars frame — deprecated, kept as evidence.

**This is not part of the package.** It is retained so the measurement that
retired it can be reproduced, and for no other purpose: nothing imports it,
no test runs it, and it is not on any path a run takes.

## What it was for

Two questions. Whether a frame engine is competitive for this shape of work,
and — with the matching implementation in `batched.rs` — whether the cost of a
frame engine is the engine itself or the trip into it from Python.

## What it found

Both were answered, and neither answer favours keeping it. At 12,000 segments
over 50 replications, seconds per replication, against the batched NumPy loop
that remains the baseline:

| policy | batched NumPy | this | Rust kernel, 48 threads |
|---|---|---|---|
| `risk_ranked` | 0.0453 | 0.0475 | 0.0023 |
| `age_threshold` | 0.0253 | 0.0119 | 0.00031 |
| `run_to_failure` | 0.0054 | 0.0075 | 0.00025 |

**A long frame does have one real advantage**, and it is worth recording
because it is not obvious: it can filter to the eligible candidates *before*
scoring them, where the rectangular NumPy form cannot — candidate sets differ
between replications, so compacting them would leave a ragged array. Under
`age_threshold`, where 2% of segments are eligible, that makes it **2.1 times
faster** than the array form. Its ranking sort is faster too: 5.9 ms a year
against `numpy.lexsort`'s 9.1.

**It loses everywhere else, and by far more than it wins.** The Rust kernel is
20 to 80 times faster than either Python form once replications run in
parallel, which is the comparison that decides how this project is built. A
frame implementation that is sometimes twice as fast as one Python baseline is
not a rival to that.

**And the interop question came back empty.** The Rust implementation of this
same loop was *slower* than this one, not faster, so there is no cost of
driving polars from Python to be recovered by writing the loop in Rust.

## What it cost to keep

The Rust half pulled the polars crate into the compute crate, which took a cold
release build from about 9 seconds to 191, and a rebuild after any Rust edit to
142. That is the price that ended it: an implementation answering a question
already answered, charging two minutes of every edit.

## What survives

polars remains the package's frame library for results, metrics, the sweep
reader and the figures, which is what it is good at here. What is deprecated is
using it to *evolve simulation state* — a column store has no in-place update,
so every year rewrites whole columns to change a few percent of them.
"""

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
    priorities: np.ndarray,
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
        priorities: ``(replications, segments)`` fixed priorities.
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
            PRIORITY: priorities.ravel(),
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
    n_segments = simulate.check_arguments(locals())

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
        random_draws.uniforms_dense(
            draw_key,
            random_draws.PURPOSE["policies"],
            first_replication,
            n_reps,
            n_segments,
            0,
        ),
        random_draws.uniforms_dense(
            draw_key,
            random_draws.PURPOSE["lifetimes"],
            first_replication,
            n_reps,
            n_segments,
            0,
        ),
    )
    # A frame row's position is `replication * segments + segment_id`, which is
    # what turns a row into the pair of indices a draw is addressed by. There is
    # no draw array to slice: the year's uniforms are produced for the replaced
    # rows and for nothing else.
    row_replication = np.repeat(
        np.arange(first_replication, first_replication + n_reps), n_segments
    )
    row_segment = np.tile(np.arange(n_segments), n_reps)
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
                random_draws.uniforms_at(
                    draw_key,
                    random_draws.PURPOSE["lifetimes"],
                    row_replication[renewing],
                    row_segment[renewing],
                    year + 1,
                ),
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
