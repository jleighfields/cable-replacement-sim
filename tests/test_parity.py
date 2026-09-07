"""The Rust kernel against the Python reference.

The two implement the same model twice, and this is what that duplication buys.
When they disagree, `simulate.py` arbitrates: it is written to be checkable by
reading, so a difference is a defect in the kernel until shown otherwise.

**The draws are identical by construction rather than by test.** Both
implementations read the same uniforms, generated once in NumPy by the fixture
in `conftest.py`, so there is no random-number stream to reconcile across the
two languages and nothing here verifies that there is not.

Most of these tests remove randomness entirely by forcing the Weibull scale
through the ordinary `scale` array, and this is where allocation defects
actually surface: a scale near zero makes every segment fail in its first year,
one far past the horizon makes none fail at all, and the two mixed together
makes failures crowd out prevention in the same year. Because both sides then
consume the same draws and take the same decisions, these compare **exactly** —
every array, every cell. One test runs a real population instead and compares
replication against replication, which is what the shared draws make possible.

What the kernel does with input it cannot run is a separate question, asked in
`test_kernel.py`: a boundary check that never fires is invisible here, because
both implementations agree on every input these tests build.

All five policies run through the exact tests, which is also what would catch
the two sides disagreeing about a policy's integer tag: the tags are authored
in `policies.py` and read again as constants in `policies.rs`, and any two
exchanged funds a different set of segments.
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
    results,
    run,
    simulate,
    weibull,
)

from tests import helpers

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


def assert_identical(reference: simulate.Results, produced: simulate.Results) -> None:
    """Asserts two results agree in every cell of every array.

    Args:
        reference: What the Python reference returned.
        produced: What the kernel returned.

    Raises:
        AssertionError: On the first array that differs, named.
    """
    for name, expected, actual in zip(
        simulate.Results._fields, reference, produced, strict=True
    ):
        np.testing.assert_array_equal(actual, expected, err_msg=f"{name} differs")


@pytest.mark.parametrize("policy", POLICIES, ids=lambda spec: str(spec.kind))
def test_the_kernel_matches_the_reference_when_nothing_ever_fails(
    deterministic_arguments: dict[str, object], policy: policies.Resolved
) -> None:
    """With no failures at all, the two agree cell for cell.

    This is the greedy fill on its own: the budget is the configured one, which
    funds a small fraction of the population, so every year runs off the end of
    the ranked order with candidates behind it and the last funded segment is
    decided by the cumulative cost. It is exact rather than tolerant because
    both sides accumulate that total one candidate at a time — `numpy.cumsum`
    on one side and a running sum on the other — which is a discrete outcome
    rather than a rounding difference.
    """
    arguments = helpers.forced_lifetimes(deterministic_arguments, helpers.NEVER_FAILS)

    assert_identical(
        simulate.run_chunk(**arguments, policy=policy),
        kernel.run_chunk(**arguments, policy=policy),
    )


@pytest.mark.parametrize("policy", POLICIES, ids=lambda spec: str(spec.kind))
def test_the_kernel_matches_the_reference_when_everything_fails_at_once(
    deterministic_arguments: dict[str, object], policy: policies.Resolved
) -> None:
    """With every segment failing every year, the two agree cell for cell.

    Nothing is ever a planned candidate here, because a segment that failed
    this year has already been replaced, so what this pins is the emergency
    path: the failure accumulation, the emergency spend, and the rule that a
    replacement enters service the following year — without which this case
    would not terminate at all.
    """
    arguments = helpers.forced_lifetimes(deterministic_arguments, helpers.FAILS_AT_ONCE)

    assert_identical(
        simulate.run_chunk(**arguments, policy=policy),
        kernel.run_chunk(**arguments, policy=policy),
    )


@pytest.mark.parametrize("policy", POLICIES, ids=lambda spec: str(spec.kind))
def test_the_kernel_matches_the_reference_when_failures_crowd_out_prevention(
    deterministic_arguments: dict[str, object], policy: policies.Resolved
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
    whether the two reductions that produce the budget agree in their last
    bits. `test_both_implementations_charge_the_same_emergency_total` leaves
    the costs uneven and pins that arithmetic instead.
    """
    n_segments = np.size(deterministic_arguments["age0"])
    n_years = deterministic_arguments["n_years"]
    scales = np.where(
        np.arange(n_segments) % 2 == 0, helpers.FAILS_AT_ONCE, helpers.NEVER_FAILS
    ).astype(float)
    arguments = {
        **deterministic_arguments,
        "scale": scales,
        "replacement_scale": scales,
        "length_ft": np.full(n_segments, 100.0),
        "cost_per_ft": np.full(n_segments, 10.0),
        "mobilization_per_segment": 500.0,
        "cost_escalation": np.ones(n_years),
        "emergency_charged_to_budget": True,
    }

    assert_identical(
        simulate.run_chunk(**arguments, policy=policy),
        kernel.run_chunk(**arguments, policy=policy),
    )


def test_both_implementations_fund_a_candidate_costing_exactly_the_remainder(
    deterministic_arguments: dict[str, object],
) -> None:
    """The greedy fill's boundary: spending may equal the budget, not exceed it.

    Every other test here compares the two implementations against each other,
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
    arguments = {
        **helpers.forced_lifetimes(deterministic_arguments, helpers.NEVER_FAILS),
        "length_ft": np.full(n_segments, 100.0),
        "cost_per_ft": np.full(n_segments, 10.0),
        "mobilization_per_segment": 500.0,
        "cost_escalation": np.ones(n_years),
        "budget": np.full(n_years, funded_exactly * planned),
    }
    policy = helpers.resolved("worst_first")

    reference = simulate.run_chunk(**arguments, policy=policy)
    produced = kernel.run_chunk(**arguments, policy=policy)

    assert reference.planned_replacements[0, 0].sum() == funded_exactly
    assert reference.planned_spend[0, 0].sum() == funded_exactly * planned
    assert_identical(reference, produced)


def test_both_implementations_charge_the_same_emergency_total(
    deterministic_arguments: dict[str, object],
) -> None:
    """The year's emergency bill must be reduced the same way on both sides.

    Where `emergency_charged_to_budget` is true, the year's emergency spend is
    subtracted from the budget before the planned pass is scored, so it decides
    how far down the ranked order the money reaches. Both sides total it one
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
    arguments = {
        **helpers.first_segments(deterministic_arguments, n_segments),
        "length_ft": length_ft,
        "cost_per_ft": np.full(n_segments, 10.0),
        "mobilization_per_segment": 500.0,
        "scale": scales,
        "replacement_scale": scales,
        "cost_escalation": np.ones(deterministic_arguments["n_years"]),
        "emergency_charged_to_budget": True,
    }
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
        deterministic_arguments["n_years"], boundary + sequential
    )

    assert_identical(
        simulate.run_chunk(**arguments, policy=policy),
        kernel.run_chunk(**arguments, policy=policy),
    )


@pytest.mark.parametrize("charged", [False, True], ids=["uncharged", "charged"])
@pytest.mark.parametrize("policy", POLICIES, ids=lambda spec: str(spec.kind))
def test_the_kernel_matches_the_reference_over_a_real_population(
    statistical_arguments: dict[str, object],
    policy: policies.Resolved,
    charged: bool,
) -> None:
    """Paired, replication against replication, on drawn lifetimes.

    Run with the year's emergency bill charged against the budget and without.
    The shipped configuration leaves it off, so without this parameter the
    charged path is reached only by the deterministic tests, and those force
    costs round precisely so that both implementations' reductions are exact.
    Drawn lifetimes are what make the emergency bill uneven, so this is where a
    difference in how the two implementations reduce it would reach the budget
    the candidates are then scored against.

    Both sides consume the same draws, so replication `r` sees identical
    lifetimes in each, and the results should differ only where a last-place
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
    produced = kernel.run_chunk(**arguments, policy=policy)

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
    """The two implementations agree on disk, not only in memory.

    An in-memory comparison ends when the test does. Two runs written out can
    be diffed afterwards, which is what makes a parity failure diagnosable
    rather than merely red, and it is also the only check that covers what sits
    between the annual loop and the saved file: the chunking, the per-policy
    loop, the concatenation and the row builder are all outside `run_chunk` and
    are driven identically for both.

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
    assert frames["reference"].equals(frames["kernel"])


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
    assert results.Manifest.model_validate_json(
        (pure_python / results.MANIFEST_NAME).read_text(encoding="utf-8")
    ).build_profile is None


def test_charging_an_empty_year_of_failures_leaves_the_budget_whole(
    deterministic_arguments: dict[str, object],
) -> None:
    """A year with nothing to charge is the empty-reduction case.

    The reference adds the year's emergency bill left to right so that it and
    the kernel reach the greedy fill with the same budget. Left to itself that
    reduction raises on an empty array — a cumulative sum of nothing has no
    last element — which is a year in which nothing failed, and the shipped
    configuration reaches one as soon as anything is charged to the budget.

    Nothing fails here at all, so every year takes that path, and the budget
    must arrive at the fill unreduced: the two implementations agree, and both
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
    assert_identical(charged, kernel.run_chunk(**arguments, policy=policy))
    uncharged = simulate.run_chunk(
        **{**arguments, "emergency_charged_to_budget": False}, policy=policy
    )
    assert_identical(charged, uncharged)


def test_a_year_that_overruns_its_budget_funds_nothing_and_carries_no_debt(
    deterministic_arguments: dict[str, object],
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
    n_segments = np.size(deterministic_arguments["age0"])
    n_years = deterministic_arguments["n_years"]
    alternating = np.where(
        np.arange(n_segments) % 2 == 0, helpers.FAILS_AT_ONCE, helpers.NEVER_FAILS
    ).astype(float)
    arguments = {
        **deterministic_arguments,
        "scale": alternating,
        "replacement_scale": alternating,
        "emergency_charged_to_budget": True,
        "budget": np.full(n_years, 10.0),
    }
    policy = helpers.resolved("age_threshold", threshold_years=45)

    overrun = simulate.run_chunk(**arguments, policy=policy)

    # The surviving half leaves candidates for the fill to refuse.
    eligible = simulate.run_chunk(
        **{**arguments, "budget": np.full(n_years, 1e9)}, policy=policy
    )
    assert eligible.planned_replacements.sum() > 0.0, (
        "the budget must be what stops this funding, not an empty candidate list"
    )

    assert overrun.emergency_spend.sum() > 10.0 * n_years
    assert overrun.planned_replacements.sum() == 0.0
    assert overrun.planned_spend.sum() == 0.0
    assert_identical(overrun, kernel.run_chunk(**arguments, policy=policy))
