"""The Rust kernel against the Python reference.

The two implement the same model twice, and this is what that duplication buys.
When they disagree, `simulate.py` arbitrates: it is written to be checkable by
reading, so a difference is a defect in the kernel until shown otherwise.

**The draws are identical by construction rather than by test.** Both
implementations read the same uniforms, generated once in NumPy by the fixture
in `conftest.py`, so there is no random-number stream to reconcile across the
two languages and nothing here verifies that there is not.

Three tests remove randomness entirely by forcing the Weibull scale through the
ordinary `scale` array, and this is where allocation defects actually surface:
a scale near zero makes every segment fail in its first year, one far past the
horizon makes none fail at all, and the two mixed together makes failures crowd
out prevention in the same year. Because both sides then consume the same
draws and take the same decisions, these compare **exactly** — every array,
every cell. The fourth test runs a real population and compares replication
against replication, which is what the shared draws make possible.

All five policies run through the exact tests, which is also what would catch
the two sides disagreeing about a policy's integer tag: the tags are authored
in `policies.py` and read again as constants in `policies.rs`, and any two
exchanged funds a different set of segments.
"""

import numpy as np
import pytest
from cablesim import kernel, policies, simulate

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

FAILS_AT_ONCE = 1e-6
"""A scale that puts every segment's remaining life inside its first year."""

NEVER_FAILS = 1e6
"""A scale that puts the first failure hundreds of thousands of years out."""


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
    arguments = conftest.forced_lifetimes(deterministic_arguments, NEVER_FAILS)

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
    arguments = conftest.forced_lifetimes(deterministic_arguments, FAILS_AT_ONCE)

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
    mobilization, and no escalation — for a reason the assertion depends on.
    The reference sums the year's emergency spend with `numpy.sum`, which is
    free to add pairwise, while the kernel accumulates it one failure at a
    time; on arbitrary costs the two agree only to the last bits, and that
    difference reaches the budget the candidates are then scored against, where
    it could flip which segment is funded last. At 3,750 dollars a failure both
    orders are exact, so the comparison stays exact and tests the ordering
    rather than the arithmetic.
    """
    n_segments = np.size(deterministic_arguments["age0"])
    n_years = deterministic_arguments["n_years"]
    scales = np.where(
        np.arange(n_segments) % 2 == 0, FAILS_AT_ONCE, NEVER_FAILS
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
        **conftest.forced_lifetimes(deterministic_arguments, NEVER_FAILS),
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


@pytest.mark.parametrize("policy", POLICIES, ids=lambda spec: str(spec.kind))
def test_the_kernel_matches_the_reference_over_a_real_population(
    statistical_arguments: dict[str, object], policy: policies.Resolved
) -> None:
    """Paired, replication against replication, on drawn lifetimes.

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
    reference = simulate.run_chunk(**statistical_arguments, policy=policy)
    produced = kernel.run_chunk(**statistical_arguments, policy=policy)

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
