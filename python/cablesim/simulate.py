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

import collections
from typing import NamedTuple

import numpy as np

from cablesim import constants, policies, random_draws, weibull

SEGMENT_ARGUMENTS: tuple[str, ...] = (
    "length_ft",
    "customers",
    "customer_minutes_per_failure",
    "customer_minutes_per_planned",
    "outage_cost_per_failure",
    "class_index",
    "age0",
    "shape",
    "scale",
    "replacement_shape",
    "replacement_scale",
    "cost_per_ft",
)
"""``run_chunk``'s arguments that carry one entry per segment, in its order.

Authored here, beside the signature that defines them, because three places
need the list and copies of it drift: the length guard below, the test helper
that takes a prefix of a population, and — differing only in ``age`` against
``age0`` — the population columns ``run.SEGMENT_COLUMNS`` names.

Named rather than derived from the signature, because a rule of the form
"every array as long as the segment count" would also catch ``budget`` and
``cost_escalation`` on any run whose horizon happened to equal its segment
count, and check them against the wrong axis. The Rust binding has a
compile-time equivalent — its checked list is typed as twelve pairs, so a
thirteenth per-segment argument is a build error there; this tuple is what a
thirteenth would have to be added to here.
"""


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

    ``sum`` is free to add pairwise, which gives a different answer in the last
    bits. That difference is tolerable wherever the result is only reported,
    and is not tolerable where it feeds a comparison that decides a discrete
    outcome. Its one caller subtracts this total from the year's budget, which
    the greedy fill then compares a cumulative cost against, so a last-bit
    difference changes which segment is the last one funded. Accumulating left
    to right is also what a running total in another language does, so it is
    what makes the two implementations agree exactly rather than
    approximately.

    Args:
        values: What to add. May be empty.

    Returns:
        The total, or 0.0 for an empty array.
    """
    if values.size == 0:
        return 0.0
    return float(np.cumsum(values)[-1])


def check_arguments(arguments: dict[str, object], concurrent: bool = False) -> int:
    """Refuses an argument set no implementation of this loop should accept.

    Authored once and called by every Python implementation of the loop — this
    module's and ``batched.run_chunk_numpy`` — so that the two cannot drift
    into refusing different things. The Rust binding makes
    the same checks in the same order and with the same wording; that copy is
    the deliberate one, because a rule written in two languages is the only way
    a caller gets the same answer from either side of the boundary.

    Left to themselves the implementations would give an answer to most of
    these. An unrecognized policy tag falls to the catch-all in ``rank_key``,
    scores every candidate zero and funds them in segment order; a per-segment
    array of the wrong length raises ``IndexError`` from NumPy rather than
    ``ValueError``; a per-year series longer than the horizon has its extra
    entries read by nothing; and the rest complete and return zeros, or
    silently widen a ``bincount``.

    The kernel cannot give an answer to most of them: reaching its loop divides
    by zero or indexes past the end of a buffer, which is a Rust panic, and a
    panic crosses into Python as ``PanicException``, which does not inherit
    from ``Exception``.

    The configuration schema forbids every one of these, which makes a direct
    caller the only way to arrive here: every parity test and every driver
    script is one.

    Args:
        arguments: Every argument of ``run_chunk``, keyed by name. Callers pass
            ``locals()`` from the first line of their own ``run_chunk``, which
            is that mapping exactly. Read by name rather than as a positional
            tuple: a second list of the twelve per-segment names would fix
            their order in a second place, and a pair whose order drifted would
            check the wrong array against the wrong name while every length
            still matched.
        concurrent: Whether the calling implementation can spread replications
            over workers. False refuses any count but 1; True refuses only a
            count below 1, which is what the Rust binding refuses. The
            distinction is a property of the caller, so it is passed rather
            than guessed.

    Returns:
        The segment count, read off the population.

    Raises:
        ValueError: If the policy tag names no policy, if the population is
            empty, if there is no class axis to accumulate into, if a
            replication, segment or year this chunk would draw at is past what
            a draw index can carry, if a per-segment or per-year array is the
            wrong length, if a class index is past the end of the class axis,
            or if ``threads`` is a count the calling implementation cannot
            run — 1 is the only count an implementation that runs one
            replication at a time accepts, and anything below 1 is refused by
            every implementation.
        TypeError: If ``policy.kind`` is not an integer, if the call carries
            more than one floating width, or if it carries one width that no
            run computes at — an all-``float16`` call names a single width and
            is still one the binding cannot borrow. The width check is a type
            error rather than a value error because the width is not a value
            any argument holds — it is the dtype the arrays are stored at,
            and the binding refuses the same call by failing to borrow the
            array as the element type its signature names.

            **The binding makes that refusal first, before any check here.**
            PyO3 extracts every argument before the function body runs, so a
            call that both mixes widths and, say, names an unknown policy is
            refused for the width by the kernel and for the policy here. Where
            the width is the only thing wrong, both raise ``TypeError``.
    """
    policy = arguments["policy"]

    # One width per call. The precision a run computes in travels as the dtype
    # of these arrays, so a call carrying two of them is a call at neither: an
    # array left at the wider one silently widens everything it touches, and
    # because every implementation widens the same way they all go on agreeing
    # with each other in every cell. That is a defect with no symptom, and it
    # has been shipped here twice.
    #
    # First, and that is the one place this function's order departs from the
    # binding's for a reason rather than by accident: PyO3 extracts every
    # argument before the body runs, so on the Rust side a width it cannot
    # borrow is refused ahead of everything, including the policy tag. Checking
    # it anywhere later would have the two sides report different problems for
    # a call that has both. The class matches too — a dtype that cannot be
    # borrowed is a type error rather than a bad value — and the wording is
    # PyO3's on that side by design.
    widths = {
        name: arguments[name].dtype
        for name in (*SEGMENT_ARGUMENTS, "budget", "cost_escalation")
        if getattr(arguments[name], "dtype", None) is not None
        and arguments[name].dtype.kind == "f"
    }
    known = {np.dtype(floating) for floating in constants.PRECISIONS.values()}
    unknown = {
        name: str(kind) for name, kind in widths.items() if kind not in known
    }
    if unknown:
        # Uniform is not the same as known: an all-`float16` call is one width
        # and the binding still refuses it at extraction, which is the
        # asymmetry this function exists to prevent.
        raise TypeError(
            f"this call carries a floating width no run computes at: {unknown}. "
            f"The widths are {sorted(constants.PRECISIONS)}"
        )
    if len(set(widths.values())) > 1:
        # The most common width is named only to make the message readable;
        # on an exact tie it is an arbitrary half, so the wording says "most of
        # them" rather than "everywhere else", which a tie would make untrue.
        counted = collections.Counter(widths.values())
        common, _ = counted.most_common(1)[0]
        odd = {name: str(kind) for name, kind in widths.items() if kind != common}
        raise TypeError(
            f"this call carries more than one floating width: {odd} against "
            f"{common} for most of them. The precision a run computes in is "
            f"the dtype of these arrays, so a mixed call is a call at neither "
            f"width"
        )

    n_classes = arguments["n_classes"]
    n_years = arguments["n_years"]
    n_reps = arguments["n_reps"]
    n_segments = arguments["age0"].size

    # Every check below is in the order `src/lib.rs` makes it and carries the
    # same message, so a caller gets the same answer whichever side of the
    # boundary it asked.
    #
    # `isinstance` before membership, because `2.0 in {0, 1, 2, 3, 4}` is true
    # by hash equality: a float tag would pass the membership test and run the
    # branch it happens to equal. The binding refuses a non-integer with a
    # `TypeError` of PyO3's own wording, which is the class mirrored here.
    if isinstance(policy.kind, bool) or not isinstance(policy.kind, int):
        raise TypeError(
            f"policy.kind is {policy.kind!r}, which is not an integer tag; "
            f"the tags are authored in cablesim.policies.KIND"
        )
    if policy.kind not in policies.RANKABLE:
        raise ValueError(
            f"policy.kind is {policy.kind}, which no ranking branch covers; "
            f"the tags are authored in cablesim.policies.KIND and the ones "
            f"that can be scored are {sorted(policies.RANKABLE)}"
        )
    if n_reps < 1:
        raise ValueError(
            f"n_reps is {n_reps}, so no replication would run; a chunk covering "
            f"none of them is a caller's arithmetic gone wrong rather than an "
            f"empty result"
        )
    if n_segments == 0:
        raise ValueError("the population is empty; there is nothing to simulate")
    if n_classes == 0:
        raise ValueError(
            "n_classes is 0, so the results have no class axis to accumulate into"
        )
    # Every position this chunk will draw at has to be one the index can carry.
    # Past a field's width two positions would share a draw, which is a
    # correlation nothing downstream could detect. Checked once here rather than
    # per draw: the largest position is known from the shape.
    # No purpose is passed: this loop draws only at the purposes the crate
    # names, and those are read from it rather than written here, so none of
    # them can be past the field it sits in. Only a caller naming a purpose of
    # its own can get that wrong, and `uniforms_at` and `uniforms_dense` bound
    # it there.
    random_draws.check_positions(
        arguments["first_replication"] + n_reps - 1, n_segments - 1, n_years
    )

    for name in SEGMENT_ARGUMENTS:
        column = arguments[name]
        if column.ndim != 1:
            # The binding takes a one-dimensional array and rejects anything
            # else with a `TypeError` of PyO3's own wording, so this case is
            # one the two implementations report differently by nature. What
            # it must not do is pass: a two-dimensional array of the right
            # element count would otherwise fail later inside NumPy as a
            # broadcast error naming neither the argument nor the reason.
            raise ValueError(
                f"{name} has shape {column.shape}; every per-segment array is "
                f"one-dimensional, one entry per segment, ordered by segment_id"
            )
        if column.size != n_segments:
            raise ValueError(
                f"{name} has {column.size} entries against {n_segments} "
                f"segments; every per-segment array is one entry per segment, "
                f"ordered by segment_id"
            )
    for name in ("budget", "cost_escalation"):
        series = arguments[name]
        if series.size != n_years:
            raise ValueError(
                f"{name} has {series.size} entries against a {n_years}-year horizon"
            )
    class_index = arguments["class_index"]
    past_the_axis = class_index[class_index >= n_classes]
    if past_the_axis.size > 0:
        raise ValueError(
            f"class_index holds {int(past_the_axis[0])}, which is past the "
            f"{n_classes} classes the results have an axis for; the index is a "
            f"position in the class list, not a name"
        )
    # Last, which is where the binding checks its own thread count, so the two
    # sides ask their questions in the same order. What they ask differs, and
    # that difference is what `concurrent` carries: an implementation that runs
    # one replication at a time takes only 1, and one that spreads them takes
    # any count above 0, which is what the binding refuses.
    threads = arguments["threads"]
    if concurrent and threads < 1:
        raise ValueError(
            f"threads is {threads}; a chunk cannot be spread over fewer than "
            f"one worker"
        )
    elif not concurrent and threads != 1:
        raise ValueError(
            f"threads is {threads}; this implementation runs one replication "
            f"at a time and cannot use more than 1. cablesim.kernel and "
            f"cablesim.batched are the implementations that can"
        )

    return n_segments


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
        draw_key: The two key words every uniform is computed under, from
            ``random_draws.draw_key``. **Not a seed and not a generator**: a
            draw here is a function of where it sits rather than of how far a
            stream has been read, so there is no position to carry and nothing
            to hand across a thread boundary.
        first_replication: Where this chunk starts in the run. The replication
            is part of a draw's address, so a chunk has to know where it sits
            or splitting a run into chunks would change its numbers — and the
            split is provenance rather than a parameter of the model.
        n_reps: Replications this chunk covers. Passed explicitly because
            nothing else in the signature carries the replication axis: every
            array here is per segment or per year.
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
        threads: Workers to spread the replications over. This implementation
            has one and refuses any other value. The argument exists so that
            every annual loop takes the same one and a caller choosing between
            them does not have to know which it holds; refusing rather than
            ignoring is what stops a benchmark row recording a thread count the
            run did not use.

    Returns:
        The seven per-year, per-class arrays for this chunk.

    Raises:
        ValueError: If the policy tag names no policy, if the population is
            empty, if there is no class axis to accumulate into, if a
            replication, segment or year this chunk would draw at is past what
            a draw index can carry, if a per-segment or per-year array is the
            wrong length, if a class index is past the end of the class axis,
            if ``threads`` is not 1, or if a candidate scores a rank key that
            is not a number. The position check is the one worth knowing
            about: each field of a draw index has a fixed width, so a position
            past one of them would wrap onto another position's draw, and two
            replications reading one number is a correlation nothing
            downstream could detect.

            These are the checks ``cablesim.kernel.run_chunk`` makes, in the
            order it makes them and word for word, because the two are
            documented as interchangeable behind one call: a caller must not
            get an answer from one and an error from the other. The mixed-width
            refusal is the exception to the ordering: the binding makes it at
            extraction, before every check in this list, where this
            implementation makes it after the draw-position check.

            What the two report differently is whatever the binding's
            argument types refuse before any check of ours runs: an array
            that is not C-contiguous, one whose dtype is not the width the
            call is being made at — ``uint8`` for ``class_index``, and
            ``float64`` or ``float32`` for the rest, the same one for all of
            them — one with the wrong number of axes, and a ``policy.kind``
            that is not an integer. PyO3 owns those messages.
            This implementation needs none of them to be true, and raises the
            same class for the last two: a non-integer tag would otherwise
            match a branch by hash equality, and a two-dimensional array of the
            right element count would fail later inside NumPy as a broadcast
            error naming neither the argument nor the reason. So an argument
            set this accepts is not guaranteed to cross the boundary.

        TypeError: If ``policy.kind`` is not an integer, or if the call carries
            more than one floating width — the precision a run computes in is
            the dtype of these arrays, so a call carrying two of them is a call
            at neither, and widening one silently would give an answer at a
            width nobody asked for.

            A per-segment array that is not one-dimensional raises
            ``ValueError`` here and ``TypeError`` from the binding, which is
            one of the reported-differently cases above rather than a class the
            two share.
    """
    n_segments = check_arguments(locals())
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

    # The working precision, read off the arrays this was handed rather than
    # passed down. Draws are produced in double whatever it is — the generator
    # is validated against NumPy's own Philox and narrowing it would break that
    # for no gain — and are narrowed where they meet the state, below.
    floating = age0.dtype

    # The two draws every segment takes whatever happens to it: the fixed
    # priority the random policy ranks on, and the left-truncated lifetime at
    # the start of the run. Both are read for every segment of every
    # replication, so there is nothing to select and they are taken up front.
    priorities = random_draws.uniforms_dense(
        draw_key,
        random_draws.PURPOSE["policies"],
        first_replication,
        n_reps,
        n_segments,
        0,
    ).astype(floating)
    initial = random_draws.uniforms_dense(
        draw_key,
        random_draws.PURPOSE["lifetimes"],
        first_replication,
        n_reps,
        n_segments,
        0,
    ).astype(floating)

    for replication in range(n_reps):
        # Copied rather than cast: the dtype of the arrays this was handed is
        # the working precision, and casting to `float` here would silently
        # widen a single-precision run back to double.
        age = age0.copy()
        current_shape = shape.copy()
        current_scale = scale.copy()
        priority = priorities[replication]
        # Conditional on survival to age0: a population that starts partway
        # through its life must not behave as though it were new.
        failure_time = weibull.draw_remaining_life(
            initial[replication], age, current_shape, current_scale
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
                #
                # A year whose failures cost more than the budget leaves this
                # negative, and that is left alone rather than floored at zero.
                # Every planned cost is positive, so the greedy fill funds
                # nothing at any value at or below zero, and nothing carries to
                # the next year — each year takes the amount in the budget
                # series and no more. A floor here would be a line no result
                # could distinguish from its absence, which mutation testing
                # confirms: removing one left the whole suite green.
                #
                # `cumsum` rather than `sum`, for the reason the greedy fill
                # uses it: this total is subtracted from the budget that the
                # fill then compares a cumulative cost against, so a last-bit
                # difference here decides which segment is funded last. `sum`
                # is free to add pairwise, and on a few hundred uneven costs it
                # disagrees with a running total often enough to change that
                # decision — measured, the two orders differ in the last bits
                # for most years with more than a handful of failures.
                available -= running_total(emergency_now)

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
                # Drawn at the replaced segments alone, which is what an
                # indexed generator is for: the draw a segment takes in a year
                # is the same number whether or not anything else was replaced,
                # so there is no stream to keep aligned by producing the rest.
                renewing = np.flatnonzero(replaced)
                failure_time[replaced] = (year + 1) + weibull.draw_lifetime(
                    random_draws.uniforms_at(
                        draw_key,
                        random_draws.PURPOSE["lifetimes"],
                        np.full(renewing.size, first_replication + replication),
                        renewing,
                        year + 1,
                    ).astype(floating),
                    replacement_shape[replaced],
                    replacement_scale[replaced],
                )
            age += 1.0
            age[replaced] = 0.0

    return results
