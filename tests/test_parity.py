"""Every other implementation of the annual loop, against the Python reference.

Several programs implement the same model, and this is what that duplication
buys. When any of them disagrees with the reference, `simulate.py` arbitrates:
it is written to be checkable by reading, so a difference is a defect in the
other implementation until shown otherwise.

Which implementations run here is read from the runnable registry rather than
listed, so one added there is held to these tests from the moment it exists.
Each is compared against the reference and none against another: a defect two
of them share would otherwise pass by agreeing with itself.

**The draws are identical by test rather than by construction, and that is
the one guarantee this design traded away.** No array of uniforms is handed
round: each implementation computes every draw from the run's key and the
position it is reading, so the Python and Rust generators have to agree bit for
bit. `test_draws.py` is what establishes that, against NumPy's own Philox.
Nothing here re-checks it, so a failure in that file makes every comparison
below unsafe to interpret.

Most of these tests remove randomness entirely by forcing the Weibull scale
through the ordinary `scale` array, and this is where allocation defects
actually surface: a scale near zero makes every segment fail in its first year,
one far past the horizon makes none fail at all, and the two mixed together
makes failures crowd out prevention in the same year. Because every
implementation then consumes the same draws and takes the same decisions, these
compare **exactly** — every array, every cell, with no tolerance anywhere. One
test runs a real population instead and compares replication against
replication, which is what the shared draws make possible.

What an implementation does with input it cannot run is a separate question,
asked in `test_kernel.py`: a boundary check that never fires is invisible here,
because they all agree on every input these tests build.

All five policies run through the exact tests, which is also what would catch
two implementations disagreeing about a policy's integer tag: the tags are
authored in `policies.py` and read again as constants in `policies.rs`, and any
two exchanged funds a different set of segments.
"""

import pathlib

import numpy as np
import polars as pl
import pytest
from cablesim import (
    config,
    constants,
    kernel,
    policies,
    population,
    random_draws,
    results,
    run,
    simulate,
    weibull,
)

from tests import conftest, helpers

POLICIES: tuple[policies.Resolved, ...] = (
    helpers.resolved("run_to_failure"),
    helpers.resolved("age_threshold", threshold_years=45),
    helpers.resolved("risk_ranked", rank_by="score_per_dollar"),
    helpers.resolved("risk_ranked"),
    helpers.resolved("worst_first"),
    helpers.resolved("random"),
)
"""Every policy the configuration ships, plus both of `risk_ranked`'s rankings.

Ranking on the raw score and ranking per dollar are the two code paths through
the score, and only the second divides by planned cost, so running one of them
would leave the other unexercised on this side of the boundary.
"""


UNDER_TEST: tuple[str, ...] = tuple(
    name for name in run.RUNNABLE if name != "reference"
)
"""Every implementation that is not the reference, by the name a run records.

Read from the runnable set rather than listed, so an implementation added there
is held to these tests from the moment it exists rather than from the moment
someone remembers to add it here.
"""


@pytest.fixture(params=UNDER_TEST, ids=str)
def implementation(request: pytest.FixtureRequest) -> run.Implementation:
    """One annual loop to compare against the reference.

    Every test taking this runs once per implementation. They are all held to
    the same bar — every cell of every array, exactly — because they all read
    the same draws and take the same decisions from them; nothing here is a
    tolerance.

    Args:
        request: Supplies the implementation name this run is parametrized on.

    Returns:
        The callable, taking `simulate.run_chunk`'s arguments by keyword.
    """
    return run.RUNNABLE[request.param]


def assert_identical(reference: simulate.Results, produced: simulate.Results) -> None:
    """Asserts two results agree in every cell of every array.

    Args:
        reference: What the Python reference returned.
        produced: What the implementation under test returned.

    Raises:
        AssertionError: On the first array that differs, named.
    """
    for name, expected, actual in zip(
        simulate.Results._fields, reference, produced, strict=True
    ):
        np.testing.assert_array_equal(actual, expected, err_msg=f"{name} differs")


@pytest.mark.parametrize("policy", POLICIES, ids=lambda spec: str(spec.kind))
def test_every_implementation_matches_the_reference_with_no_failures(
    deterministic_arguments: dict[str, object],
    policy: policies.Resolved,
    implementation: run.Implementation,
) -> None:
    """With no failures at all, the results agree cell for cell.

    This is the greedy fill on its own: the budget is the configured one, which
    funds a small fraction of the population, so every year runs off the end of
    the ranked order with candidates behind it and the last funded segment is
    decided by the cumulative cost. It is exact rather than tolerant because
    every implementation accumulates that total one candidate at a time —
    `numpy.cumsum` in Python and a running sum in Rust — which is a discrete
    outcome rather than a rounding difference.
    """
    arguments = helpers.forced_lifetimes(deterministic_arguments, helpers.NEVER_FAILS)

    assert_identical(
        simulate.run_chunk(**arguments, policy=policy),
        implementation(**arguments, policy=policy),
    )


@pytest.mark.parametrize("policy", POLICIES, ids=lambda spec: str(spec.kind))
def test_every_implementation_matches_when_everything_fails_at_once(
    deterministic_arguments: dict[str, object],
    policy: policies.Resolved,
    implementation: run.Implementation,
) -> None:
    """With every segment failing every year, the results agree cell for cell.

    Nothing is ever a planned candidate here, because a segment that failed
    this year has already been replaced, so what this pins is the emergency
    path: the failure accumulation, the emergency spend, and the rule that a
    replacement enters service the following year — without which this case
    would not terminate at all.
    """
    arguments = helpers.forced_lifetimes(
        deterministic_arguments, helpers.FAILS_AT_ONCE, from_new=True
    )

    assert_identical(
        simulate.run_chunk(**arguments, policy=policy),
        implementation(**arguments, policy=policy),
    )


def test_the_forced_scale_fails_every_segment_at_both_widths(
    deterministic_arguments: dict[str, object],
) -> None:
    """`helpers.FAILS_AT_ONCE` has to force the case the tests above name.

    Those tests compare implementations against each other, so they pass on any
    scenario the fixture happens to produce — including one where half the
    population never fails at all. What the scenario *is* has to be asserted
    separately, and this is that assertion: at the scale documented as putting
    every segment's remaining life inside year one, every segment fails in year
    zero.

    The remaining life is `scale * ((age / scale) ** shape - ln1p(-u)) ** (1 /
    shape) - age`, and at a scale this far below the ages the bracket is
    dominated by the first term, so the whole expression is a difference of two
    numbers that agree to within a rounding step of `age`. In double that
    leaves a positive value around 1e-14; in single the rounding step at an age
    of 57 is about 4e-6, which is larger than the value being recovered, so the
    result lands either side of zero. A negative remaining life is a failure
    time no year's `[year, year + 1)` window contains, so that segment never
    fails in any year of the horizon.
    """
    arguments = helpers.forced_lifetimes(
        deterministic_arguments, helpers.FAILS_AT_ONCE, from_new=True
    )
    every_segment = np.size(arguments["age0"]) * arguments["n_reps"]

    failures = simulate.run_chunk(
        **arguments, policy=helpers.resolved("run_to_failure")
    ).failures

    assert failures[:, 0].sum() == every_segment, (
        f"{failures[:, 0].sum():.0f} of {every_segment} segment-replications "
        f"failed in year zero at {arguments['age0'].dtype}; the fixture "
        f"documents itself as failing all of them"
    )


def test_the_alternating_scale_fails_the_half_it_names_at_both_widths(
    deterministic_arguments: dict[str, object],
) -> None:
    """The half-and-half scenario has to be half and half at both widths.

    Two tests below are built on an alternating scale and describe the case
    they reach as half the population failing immediately and the other half
    never failing. Both compare implementations against each other, so both
    pass on whatever scenario the fixture happened to produce; what the
    scenario *is* has to be asserted separately, and this is that assertion.

    `helpers.FAILS_AT_ONCE` forces a failure inside year one only for a segment
    starting from new. On an aged segment the remaining life is drawn
    conditional on survival, so the accumulated hazard dominates the bracket and
    adding the draw to it changes nothing a single-precision float holds — the
    result lands either side of zero, and a negative remaining life is a
    failure time no year's `[year, year + 1)` window contains. That is why
    `helpers.alternating_lifetimes` zeroes the age of the half it forces to
    fail and leaves the other half aged: applied to the fixture's aged
    population without that, only about a quarter of it fails and the two
    tests below no longer reach the scenario they describe.
    """
    arguments = helpers.alternating_lifetimes(deterministic_arguments)
    # Counted off the fixture the helper returned rather than by restating its
    # rule here: a second copy of "even segments fail" would have to be changed
    # with it, and dividing by two is wrong on an odd population besides.
    intended = int(
        np.count_nonzero(np.isclose(arguments["scale"], helpers.FAILS_AT_ONCE))
    ) * arguments["n_reps"]

    failures = simulate.run_chunk(
        **arguments, policy=helpers.resolved("run_to_failure")
    ).failures

    assert failures[:, 0].sum() == intended, (
        f"{failures[:, 0].sum():.0f} of {intended} segment-replications failed "
        f"in year zero at {arguments['age0'].dtype}; the tests built on this "
        f"scenario describe half the population failing immediately"
    )


@pytest.mark.parametrize("policy", POLICIES, ids=lambda spec: str(spec.kind))
def test_every_implementation_matches_when_failures_crowd_out_prevention(
    deterministic_arguments: dict[str, object],
    policy: policies.Resolved,
    implementation: run.Implementation,
) -> None:
    """Both paths run in the same year, with the year's failures charged first.

    This is the case the other two cannot reach: half the population fails
    immediately and the other half never does, so a year has both an emergency
    bill and a candidate list, and `emergency_charged_to_budget` makes the
    first decide how far down the second the budget reaches.

    Costs are forced round — every segment 100 feet at 10 dollars plus 500 of
    mobilization, and no escalation — so that the year's emergency bill is
    exactly representable whatever order it is added in. That keeps this test
    on the ordering: it compares which segments each implementation funds, not
    whether the reductions that produce the budget agree in their last
    bits. `test_both_implementations_charge_the_same_emergency_total` leaves
    the costs uneven and pins that arithmetic instead.
    """
    n_segments = np.size(deterministic_arguments["age0"])
    n_years = deterministic_arguments["n_years"]
    arguments = helpers.at_call_width(
        {
            **helpers.alternating_lifetimes(deterministic_arguments),
            "length_ft": np.full(n_segments, 100.0),
            "cost_per_ft": np.full(n_segments, 10.0),
            "mobilization_per_segment": 500.0,
            "cost_escalation": np.ones(n_years),
            "emergency_charged_to_budget": True,
        }
    )

    assert_identical(
        simulate.run_chunk(**arguments, policy=policy),
        implementation(**arguments, policy=policy),
    )


def test_every_implementation_funds_a_candidate_costing_the_remainder(
    deterministic_arguments: dict[str, object],
    implementation: run.Implementation,
) -> None:
    """The greedy fill's boundary: spending may equal the budget, not exceed it.

    Every other test here compares an implementation against the reference,
    which cannot see a boundary they are both wrong about in the same
    direction. This one knows the answer without simulating anything: nothing
    fails, every segment costs exactly 1,500 to replace, and the year's budget
    is exactly ten of them, so exactly ten are funded. A fill that stopped at
    the first candidate reaching the budget rather than exceeding it would fund
    nine and still agree with a reference that did the same.
    """
    n_segments = np.size(deterministic_arguments["age0"])
    n_years = deterministic_arguments["n_years"]
    funded_exactly = 10
    planned = 100.0 * 10.0 + 500.0
    arguments = helpers.at_call_width(
        {
            **helpers.forced_lifetimes(deterministic_arguments, helpers.NEVER_FAILS),
            "length_ft": np.full(n_segments, 100.0),
            "cost_per_ft": np.full(n_segments, 10.0),
            "mobilization_per_segment": 500.0,
            "cost_escalation": np.ones(n_years),
            "budget": np.full(n_years, funded_exactly * planned),
        }
    )
    policy = helpers.resolved("worst_first")

    reference = simulate.run_chunk(**arguments, policy=policy)
    produced = implementation(**arguments, policy=policy)

    assert reference.planned_replacements[0, 0].sum() == funded_exactly
    assert reference.planned_spend[0, 0].sum() == funded_exactly * planned
    assert_identical(reference, produced)


def one_year_at_single_precision(
    multiplier: float, budget: float, charged: bool, n_segments: int | None = None
) -> dict[str, object]:
    """One year of the shipped population at single precision, on a boundary.

    **The multiplier and budget each caller passes were found by search against
    the population the shipped seed, size and cost parameters generate**, so
    this is a statement about that exact population and not about its size.
    Measured: at 12,001 segments, at 11,000, and at one step of the seed, the
    constants stop straddling their boundary and the callers pin nothing. Each
    asserts the funded count it was built on, so a change to any of those
    reports rather than passing quietly, and the fix is to search again.

    A fixture's few hundred segments will not do: what these pin is a last-bit
    difference deciding which candidate the budget reaches last, and a small
    population does not put enough candidates near the cut for one to fall the
    other side. One replication over one year keeps the shipped size
    affordable.

    Args:
        multiplier: What an emergency replacement costs, relative to planned.
        budget: The year's budget, chosen to sit between the two cut points.
        charged: Whether the year's emergency bill is taken off the budget
            before the planned pass is scored.
        n_segments: A population size to build instead of the shipped one.
            The callers below leave it None and get the population their
            constants were searched against; the test that checks those
            constants are still on their boundary passes one of the sizes
            named above, which is the perturbation that moves it.

    Returns:
        Every argument of ``simulate.run_chunk`` except ``policy``.
    """
    base = config.load_config(constants.DEFAULT_CONFIG_PATH)
    if n_segments is not None:
        base = config.with_overrides(base, {"population.n_segments": n_segments})
    settings = base.model_copy(
        update={
            "simulation": base.simulation.model_copy(
                update={"precision": "f32", "n_reps": 1, "n_years": 1}
            )
        }
    )
    # Read once, from the configuration, so the width is named in one place.
    precision = settings.simulation.precision
    floating = constants.PRECISIONS[precision]
    built = {
        **run.segment_arrays(population.generate(settings), precision),
        "draw_key": random_draws.draw_key(settings.simulation.seed),
        "first_replication": 0,
        "n_reps": 1,
        "budget": np.full(1, budget, dtype=floating),
        "cost_escalation": np.ones(1, dtype=floating),
        "emergency_multiplier": multiplier,
        "mobilization_per_segment": settings.costs.mobilization_per_segment,
        "emergency_charged_to_budget": charged,
        "n_classes": len(settings.population.classes),
        "n_years": 1,
    }
    # This builder assembles its own arguments rather than going through
    # `conftest.simulation_arguments`, so the width check that fixture makes
    # does not reach it. Building at the wrong width would leave both callers
    # passing and pinning nothing.
    helpers.assert_at_width(built, precision)
    return built


# Every segment a candidate and the raw score, so the ranking is the premium
# rather than the premium divided by cost — which is where both of the
# divergences below live.
UNFILTERED_RISK = policies.Resolved(
    kind=policies.KIND["risk_ranked"],
    threshold_years=float("-inf"),
    rank_by_cost=False,
)


def test_the_emergency_premium_is_narrowed_where_the_reference_narrows_it(
    implementation: run.Implementation,
) -> None:
    """Where a configured double meets the working width decides who is funded.

    The risk-ranked score reads ``planned * (emergency_multiplier - 1)``. The
    reference does that subtraction in Python's double arithmetic and lets
    NumPy narrow the result; narrowing the multiplier first and subtracting at
    the working width gives a different premium for any multiplier the width
    cannot hold, which is about two fifths of them at single precision.

    The multiplier and budget here are not arbitrary. The shipped 2.5 is
    exactly representable and cannot show this at all, and on a few hundred
    segments the ordering moves without the funded set changing. These put the
    budget between the two cut points on the shipped population, where the two
    orders fund 181 candidates and 180.
    """
    arguments = one_year_at_single_precision(
        multiplier=2.942645377983026, budget=41938160.0, charged=False
    )
    reference = simulate.run_chunk(**arguments, policy=UNFILTERED_RISK)

    assert reference.planned_replacements.sum() == 181, (
        "the budget no longer sits between the two cut points, so this test "
        "pins nothing whatever it asserts next; the multiplier and budget were "
        "searched against a particular population and need searching again, "
        "unless the reference's own premium arithmetic changed"
    )
    assert_identical(
        reference, implementation(**arguments, policy=UNFILTERED_RISK)
    )


def test_the_emergency_bill_totals_at_the_width_the_budget_compares_at(
    implementation: run.Implementation,
) -> None:
    """The year's emergency total is a running sum at the working width.

    It is subtracted from the budget that the greedy fill then compares a
    cumulative cost against, so it has to round the way that comparison rounds.
    NumPy's ``cumsum`` preserves the width, which is what the reference uses;
    totalling in double and narrowing once at the subtraction lands on a
    different remaining budget.

    The budget here sits between the two: the single-precision running total is
    79,962,480 against 79,962,488 for the double-then-narrow one, and one
    candidate falls either side.
    """
    arguments = one_year_at_single_precision(
        multiplier=2.5, budget=80962232.0, charged=True
    )
    reference = simulate.run_chunk(**arguments, policy=UNFILTERED_RISK)

    # The funded count alone cannot guard this one: at 12,001 segments it still
    # reads 1 while the boundary has stopped straddling, so the test would pass
    # having pinned nothing. The emergency bill is the quantity the boundary is
    # cut from, and it separates every perturbation that breaks this.
    assert reference.planned_replacements.sum() == 1, (
        "the funded count moved, so the budget no longer sits between the two "
        "totals; this was searched against a particular population and needs "
        "searching again, unless the reference's own arithmetic changed"
    )
    assert reference.emergency_spend.sum() == pytest.approx(
        79962487.52734375, abs=1.0
    ), (
        "the emergency bill moved, so the budget is no longer between what the "
        "two accumulator widths total to and this test pins nothing; re-search "
        "it against this population"
    )
    assert_identical(
        reference, implementation(**arguments, policy=UNFILTERED_RISK)
    )


BOUNDARY_CASES = {
    "premium": (
        {"multiplier": 2.942645377983026, "budget": 41938160.0, "charged": False},
        lambda produced: (produced.planned_replacements.sum(),),
    ),
    "emergency_bill": (
        {"multiplier": 2.5, "budget": 80962232.0, "charged": True},
        # Both quantities its guard reads. The funded count alone does not move
        # at the drifted population, which is why that guard needed the bill.
        lambda produced: (
            produced.planned_replacements.sum(),
            produced.emergency_spend.sum(),
        ),
    ),
}
"""The two boundary cases above, with the quantity each one's guard asserts.

Keyed by the boundary rather than by the test name so a reader can see which
constants belong to which. The second element reads back exactly what that
test's guard reads, so the check below is a check on the guard rather than on
something adjacent to it.
"""

DRIFTED_POPULATION = 12_001
"""One segment more than the shipped population.

``one_year_at_single_precision`` names this among the perturbations that stop
its constants straddling their boundary, so it is the smallest change that
must be visible to a guard placed there.
"""


@pytest.mark.parametrize("boundary", sorted(BOUNDARY_CASES), ids=str)
def test_each_boundary_guard_notices_that_its_boundary_has_moved(
    boundary: str,
) -> None:
    """A guard that reads the same number either side of a drift disarms silently.

    The two tests above are searched against one exact population, and each
    opens with a guard whose message says the search has to be redone if the
    constants stop straddling. That message is only reached if the guarded
    quantity moves when the population does — a guard reading a number that is
    the same on both sides passes on a population its test pins nothing about,
    which is the failure the guard was added to prevent.

    One segment added to the population is the perturbation the builder's own
    docstring names. Both boundaries are checked, so a guard that does move is
    the control showing this comparison discriminates.

    Args:
        boundary: Which of the two searched boundaries to check.
    """
    constants_for, guarded = BOUNDARY_CASES[boundary]

    searched = guarded(
        simulate.run_chunk(
            **one_year_at_single_precision(**constants_for), policy=UNFILTERED_RISK
        )
    )
    drifted = guarded(
        simulate.run_chunk(
            **one_year_at_single_precision(
                **constants_for, n_segments=DRIFTED_POPULATION
            ),
            policy=UNFILTERED_RISK,
        )
    )

    assert searched != drifted, (
        f"the {boundary} guard reads {searched} on the population it was "
        f"searched against and {searched} again at {DRIFTED_POPULATION} "
        f"segments, so it cannot tell that the boundary has moved; it needs "
        f"to assert a quantity the drift changes as well"
    )


def test_every_implementation_charges_the_same_emergency_total(
    deterministic_arguments: dict[str, object],
    implementation: run.Implementation,
) -> None:
    """The year's emergency bill must be reduced the same way by everything.

    Where `emergency_charged_to_budget` is true, the year's emergency spend is
    subtracted from the budget before the planned pass is scored, so it decides
    how far down the ranked order the money reaches. All of them total it one
    failure at a time — `simulate.running_total` on the reference, a running
    sum in the kernel — and this is what holds them to that. A reference that
    reduced with `numpy.sum` instead would add pairwise, which disagrees in the
    last bits for as few as eight failures, and that difference lands on a
    budget the greedy fill then compares against a cumulative cost, turning a
    rounding difference into a different set of funded segments.

    `test_the_kernel_matches_the_reference_when_failures_crowd_out_prevention`
    cannot see this: it forces every cost round, so both reductions are exact.
    This one leaves the costs uneven and puts the budget exactly on the funding
    boundary the two totals straddle, which is where the discrete outcome
    flips. Both are needed — that test pins the ordering, this one pins the
    arithmetic underneath it.
    """
    n_segments = 60
    fails = np.arange(n_segments) % 2 == 0
    # Uneven lengths, so the emergency total is not exactly representable and
    # the two reduction orders land on different doubles.
    length_ft = np.random.default_rng(13).uniform(300.0, 4000.0, n_segments)
    planned = length_ft * 10.0 + 500.0
    emergency = planned[fails] * 2.5

    pairwise = float(emergency.sum())
    sequential = 0.0
    for cost in emergency:
        sequential += float(cost)
    assert pairwise != sequential, (
        "the two reduction orders agree on these costs, so this test can no "
        "longer reach the boundary it exists to pin"
    )

    scales = np.where(fails, helpers.FAILS_AT_ONCE, helpers.NEVER_FAILS).astype(float)
    arguments = helpers.at_call_width(
        {
            # From new, so the half meant to fail actually does at both
            # widths: a conditional draw on an aged segment loses the draw
            # entirely in single precision, and the boundary this test
            # constructs assumes every one of them fails.
            **helpers.forced_lifetimes(
                helpers.first_segments(deterministic_arguments, n_segments),
                helpers.NEVER_FAILS,
                from_new=True,
            ),
            "length_ft": length_ft,
            "cost_per_ft": np.full(n_segments, 10.0),
            "mobilization_per_segment": 500.0,
            "scale": scales,
            "replacement_scale": scales,
            "cost_escalation": np.ones(deterministic_arguments["n_years"]),
            "emergency_charged_to_budget": True,
        }
    )
    policy = helpers.resolved("worst_first")

    # The budget is put exactly on a funding boundary that the two totals
    # straddle: adding the sequential total leaves it recoverable to the cent,
    # and subtracting the pairwise one lands just below, so the reference funds
    # one candidate fewer than the kernel.
    survivors = np.flatnonzero(~fails)
    probability = weibull.conditional_failure_probability(
        arguments["age0"], arguments["shape"], scales
    )
    running = np.cumsum(planned[policies.order_by_rank(probability, survivors)])
    boundary = next(
        total
        for total in running
        if (total + sequential) - sequential == total
        and (total + sequential) - pairwise < total
    )
    arguments["budget"] = np.full(
        deterministic_arguments["n_years"],
        boundary + sequential,
        dtype=np.asarray(arguments["age0"]).dtype,
    )

    assert_identical(
        simulate.run_chunk(**arguments, policy=policy),
        implementation(**arguments, policy=policy),
    )


@pytest.mark.parametrize("charged", [False, True], ids=["uncharged", "charged"])
@pytest.mark.parametrize("policy", POLICIES, ids=lambda spec: str(spec.kind))
def test_every_implementation_matches_over_a_real_population(
    statistical_arguments: dict[str, object],
    policy: policies.Resolved,
    charged: bool,
    implementation: run.Implementation,
) -> None:
    """Paired, replication against replication, on drawn lifetimes.

    Run with the year's emergency bill charged against the budget and without.
    The shipped configuration leaves it off, so without this parameter the
    charged path is reached only by the deterministic tests, and those force
    costs round precisely so that every implementation's reduction is exact.
    Drawn lifetimes are what make the emergency bill uneven, so this is where a
    difference in how an implementation reduces it would reach the budget the
    candidates are then scored against.

    Every implementation consumes the same draws, so replication `r` sees
    identical lifetimes in each, and the results should differ only where a last-place
    difference in a score flipped a sort and changed which candidate was funded
    last. The comparison is paired for that reason: treating the two runs as
    independent samples would throw the pairing away and could only see a
    difference large enough to move a whole distribution.

    Two things are asserted, and the second has two acceptable forms. Fewer
    than one percent of replications may differ by more than `1e-9` relative on
    any reported quantity. And the mean paired difference must sit within three
    standard errors of zero **or** be identically zero — the second clause
    because exact agreement is the expected case here, same draws and same
    arithmetic, and a standard error of zero would otherwise make the criterion
    a division by zero rather than a pass.
    """
    arguments = {**statistical_arguments, "emergency_charged_to_budget": charged}
    reference = simulate.run_chunk(**arguments, policy=policy)
    produced = implementation(**arguments, policy=policy)

    for name, expected, actual in zip(
        simulate.Results._fields, reference, produced, strict=True
    ):
        # Per replication, over the year and class axes, which is the unit the
        # pairing is defined on.
        per_replication = tuple(range(1, expected.ndim))
        expected_totals = expected.sum(axis=per_replication)
        actual_totals = actual.sum(axis=per_replication)

        scale = np.maximum(np.abs(expected_totals), 1.0)
        differing = np.abs(actual_totals - expected_totals) / scale > 1e-9
        assert differing.mean() < 0.01, (
            f"{name}: {differing.sum()} of {differing.size} replications "
            f"differ by more than 1e-9 relative"
        )

        paired = actual_totals - expected_totals
        if np.any(paired != 0.0):
            standard_error = paired.std(ddof=1) / np.sqrt(paired.size)
            assert abs(paired.mean()) <= 3.0 * standard_error, (
                f"{name}: mean paired difference {paired.mean():.6g} is more "
                f"than three standard errors from zero"
            )


def test_two_saved_runs_agree_row_for_row(tmp_path: pathlib.Path) -> None:
    """Every implementation agrees on disk, not only in memory.

    An in-memory comparison ends when the test does. Runs written out can
    be diffed afterwards, which is what makes a parity failure diagnosable
    rather than merely red, and it is also the only check that covers what sits
    between the annual loop and the saved file: the chunking, the per-policy
    loop, the concatenation and the row builder are all outside `run_chunk` and
    are driven identically for all of them.

    The runs are written through the same entry point a real sweep uses, so
    what is compared is the shipped path. Only the manifest may differ — it
    records which implementation ran, and a wall-clock time that is not a
    result.
    """
    settings = config.resize_population(
        config.load_config(constants.DEFAULT_CONFIG_PATH), 300, n_reps=4
    )

    saved = {
        name: run.run(settings, tmp_path / name, implementation=name, batch_size=3)
        for name in run.RUNNABLE
    }

    frames = {
        name: pl.read_parquet(directory / results.RESULTS_NAME).sort(
            ["policy", "replication", "year", "class"]
        )
        for name, directory in saved.items()
    }

    assert frames["reference"].height > 0
    # Every implementation against the reference, rather than the kernel by
    # name: a third one added to the registry is run by the loop above, and
    # naming its comparison here is what stops it being run and never checked.
    for name, frame in frames.items():
        assert frames["reference"].equals(frame), f"{name} differs on disk"
    assert len(frames) > 1, (
        "the reference compared against itself is not a parity check"
    )


def test_a_saved_kernel_run_records_the_profile_it_was_built_with(
    tmp_path: pathlib.Path,
) -> None:
    """A run through the kernel records which Cargo profile produced it.

    The reference has no build profile and records none rather than inventing
    one. The kernel's comes from the compiled binary itself, so it cannot
    disagree with what ran; a debug build is slower by enough that a timing
    recorded without it says nothing.
    """
    settings = config.resize_population(
        config.load_config(constants.DEFAULT_CONFIG_PATH), 100, n_reps=2
    )

    directory = run.run(settings, tmp_path / "kernel", implementation="kernel")

    manifest = results.Manifest.model_validate_json(
        (directory / results.MANIFEST_NAME).read_text(encoding="utf-8")
    )

    assert manifest.implementation == "kernel"
    assert manifest.build_profile in {"debug", "release"}

    # And the reference records none rather than inventing one. Without this,
    # a default of "release" on the lookup would have every pure-Python run
    # claiming a Rust build, and nothing would report it.
    pure_python = run.run(settings, tmp_path / "reference", implementation="reference")
    assert (
        results.Manifest.model_validate_json(
            (pure_python / results.MANIFEST_NAME).read_text(encoding="utf-8")
        ).build_profile
        is None
    )


def test_charging_an_empty_year_of_failures_leaves_the_budget_whole(
    deterministic_arguments: dict[str, object],
    implementation: run.Implementation,
) -> None:
    """A year with nothing to charge is the empty-reduction case.

    The reference adds the year's emergency bill left to right so that it and
    every other implementation reach the greedy fill with the same budget. Left
    to itself that reduction raises on an empty array — a cumulative sum of
    nothing has no last element — which is a year in which nothing failed, and
    the shipped
    configuration reaches one as soon as anything is charged to the budget.

    Nothing fails here at all, so every year takes that path, and the budget
    must arrive at the fill unreduced: the implementations agree, and both
    fund what an uncharged run would.
    """
    arguments = {
        **helpers.forced_lifetimes(deterministic_arguments, helpers.NEVER_FAILS),
        "emergency_charged_to_budget": True,
    }
    # An age threshold rather than a whole-population policy, because the fill
    # stops at the first candidate that does not fit and `risk_ranked` puts the
    # largest feeder first: at this reduced size that one segment costs more
    # than the year's whole budget, so nothing is funded and the test would
    # assert on a run in which the budget was never reached.
    policy = helpers.resolved("age_threshold", threshold_years=45)

    charged = simulate.run_chunk(**arguments, policy=policy)

    assert charged.failures.sum() == 0.0
    assert charged.planned_replacements.sum() > 0.0
    assert_identical(charged, implementation(**arguments, policy=policy))
    uncharged = simulate.run_chunk(
        **{**arguments, "emergency_charged_to_budget": False}, policy=policy
    )
    assert_identical(charged, uncharged)


def test_a_year_that_overruns_its_budget_funds_nothing_and_carries_no_debt(
    deterministic_arguments: dict[str, object],
    implementation: run.Implementation,
) -> None:
    """The limiting case of failures crowding out prevention.

    Half the population fails every year and the other half never does, so a
    year has both an emergency bill and a candidate list. The bill is far
    larger than a budget set to ten dollars, which leaves the planned pass with
    a negative amount to spend.

    **The surviving half is what makes this a test of the greedy fill.** Force
    every segment to fail and there are no candidates at all, because a segment
    replaced after failure is out of the running for planned work in the same
    year — the budget is then computed and never used, and the run funds
    nothing whatever the fill does with a negative number.

    Neither implementation floors that negative at zero. Nothing needs to:
    every planned cost is positive, so the fill stops at its first candidate,
    and no year's remainder carries to the next. Both must fund nothing, spend
    nothing, and agree on all seven arrays.
    """
    n_years = deterministic_arguments["n_years"]
    arguments = helpers.at_call_width(
        {
            **helpers.alternating_lifetimes(deterministic_arguments),
            "emergency_charged_to_budget": True,
            "budget": np.full(n_years, 10.0),
        }
    )
    policy = helpers.resolved("age_threshold", threshold_years=45)

    overrun = simulate.run_chunk(**arguments, policy=policy)

    # The surviving half leaves candidates for the fill to refuse.
    eligible = simulate.run_chunk(
        **helpers.at_call_width({**arguments, "budget": np.full(n_years, 1e9)}),
        policy=policy,
    )
    assert eligible.planned_replacements.sum() > 0.0, (
        "the budget must be what stops this funding, not an empty candidate list"
    )

    assert overrun.emergency_spend.sum() > 10.0 * n_years
    assert overrun.planned_replacements.sum() == 0.0
    assert overrun.planned_spend.sum() == 0.0
    assert_identical(overrun, implementation(**arguments, policy=policy))


def test_the_parity_fixture_refuses_a_single_replication() -> None:
    """The builder behind both fixtures rejects a chunk it cannot cover.

    One replication removes what the parity tests exist to check without
    failing anything: the reference takes a NumPy row per replication and the
    kernel offsets into a flat slice, so an offset defect needs a second row to
    be visible at all. Mutating either offset away reddens seventeen tests at
    three replications and none at one.

    Driven here because the fixtures themselves always pass a good value, so
    nothing else would ever reach the check.
    """
    settings = config.resize_population(
        config.load_config(constants.DEFAULT_CONFIG_PATH), 50, n_reps=1
    )

    with pytest.raises(ValueError, match="at least 2 replications"):
        conftest.simulation_arguments(settings)


@pytest.mark.parametrize("policy", POLICIES, ids=lambda spec: str(spec.kind))
def test_a_chunk_gives_the_same_rows_wherever_it_starts(
    deterministic_arguments: dict[str, object],
    policy: policies.Resolved,
) -> None:
    """Splitting a run into chunks changes no number in it.

    A chunk is told where it starts, and every draw it takes is computed from
    that position, so running a chunk whole and running it as two consecutive
    pieces have to give the same rows in the same order. This is what lets a long run be
    divided across calls, and it is the only assertion here that varies
    ``first_replication`` — every fixture otherwise starts a run at zero, which
    leaves an implementation free to ignore the offset entirely and still agree
    with the reference everywhere else.
    """
    whole = simulate.run_chunk(**deterministic_arguments, policy=policy)
    n_reps = deterministic_arguments["n_reps"]
    split = n_reps // 2

    for implementation in sorted(run.RUNNABLE):
        loop = run.RUNNABLE[implementation]
        pieces = [
            loop(
                **{
                    **deterministic_arguments,
                    "first_replication": deterministic_arguments["first_replication"]
                    + start,
                    "n_reps": stop - start,
                },
                policy=policy,
            )
            for start, stop in ((0, split), (split, n_reps))
        ]
        joined = simulate.Results(
            *(np.concatenate(arrays) for arrays in zip(*pieces, strict=True))
        )
        for name, wanted, actual in zip(
            simulate.Results._fields, whole, joined, strict=True
        ):
            assert np.array_equal(wanted, actual), (
                f"{implementation}, {name}: two chunks disagree with one run"
            )


@pytest.mark.parametrize("policy", POLICIES, ids=lambda spec: str(spec.kind))
def test_a_threaded_chunk_that_does_not_start_at_zero_matches_the_reference(
    deterministic_arguments: dict[str, object],
    policy: policies.Resolved,
) -> None:
    """Chunk offset and threading are correct together, not only apart.

    Two assertions here cover one of these each: one varies the starting
    replication and never threads, the other threads and always starts at zero.
    Their combination is what a run actually does — ``run.execute`` chunks a
    long sweep and hands each chunk to a threaded implementation — and an
    implementation that passed its block's own offset instead of the run's would
    satisfy both while getting every chunk after the first wrong.
    """
    offset = {**deterministic_arguments, "first_replication": 50}
    expected = simulate.run_chunk(**offset, policy=policy)

    for implementation in sorted(run.CONCURRENT):
        produced = run.RUNNABLE[implementation](**offset, policy=policy, threads=3)
        for name, wanted, actual in zip(
            simulate.Results._fields, expected, produced, strict=True
        ):
            assert np.array_equal(wanted, actual), (
                f"{implementation}, {name}: a threaded chunk starting at 50 "
                f"disagrees with the reference"
            )


@pytest.mark.parametrize("policy", POLICIES, ids=lambda spec: str(spec.kind))
def test_every_thread_count_gives_the_reference_answer(
    deterministic_arguments: dict[str, object],
    policy: policies.Resolved,
) -> None:
    """Spreading replications over workers changes nothing, not even a last bit.

    Order independence is a property of how the work is split rather than of
    how well the arithmetic behaves, so this asserts exact equality rather than
    a tolerance. Each replication computes its own draws from their positions
    and writes its own block of the results, sharing nothing with any other, and
    the blocks are concatenated in replication order rather than in the order
    they finished. A kernel that accumulated into one shared buffer instead
    would still pass a tolerance-based check most of the time and would give a
    different answer run to run.

    Two counts are load-bearing beyond the plain many-threads case. Two is the
    smallest that splits at all. And this fixture has three replications, so
    the machine's full count is almost always more workers than there is work,
    which is where an implementation that assumed each worker gets at least one
    replication comes apart.
    """
    expected = simulate.run_chunk(**deterministic_arguments, policy=policy)
    for implementation in sorted(run.CONCURRENT):
        loop = run.RUNNABLE[implementation]
        for threads in (1, 2, 3, kernel.AVAILABLE_THREADS):
            produced = loop(
                **deterministic_arguments, policy=policy, threads=threads
            )
            for name, wanted, actual in zip(
                simulate.Results._fields, expected, produced, strict=True
            ):
                assert np.array_equal(wanted, actual), (
                    f"{implementation}, {name}: {threads} threads disagrees "
                    f"with the reference"
                )


def test_threading_a_real_population_changes_no_value(
    statistical_arguments: dict[str, object],
) -> None:
    """The same, on drawn lifetimes rather than forced ones.

    The deterministic tests give every replication the same work to do, so a
    scheduler has nothing to decide. Here the replications differ, workers
    finish out of order, and the greedy fill's running total is where a shared
    accumulator would show up first — it decides which candidate is the last
    one funded, which is a discrete outcome rather than a rounding difference.

    Run under the policy that scores every segment in the population, since a
    policy funding nothing exercises none of the allocation this is about.
    """
    ranked = next(
        spec for spec in POLICIES if spec.kind == policies.KIND["risk_ranked"]
    )
    one = kernel.run_chunk(**statistical_arguments, policy=ranked, threads=1)
    many = kernel.run_chunk(
        **statistical_arguments, policy=ranked, threads=kernel.AVAILABLE_THREADS
    )
    for name, wanted, actual in zip(simulate.Results._fields, one, many, strict=True):
        assert np.array_equal(wanted, actual), (
            f"{name}: {kernel.AVAILABLE_THREADS} threads disagrees with 1"
        )
