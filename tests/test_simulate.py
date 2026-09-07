"""The annual loop, checked against cases whose answers are known by hand.

Randomness is removed wherever it is not the thing under test, by forcing the
Weibull scale through the ordinary ``scale`` array rather than through a
test-only argument. A scale near zero makes every segment fail in its first
year; one far past the horizon makes none fail at all. Both arrive the way
every run's scales arrive, so these exercise the shipped path rather than a
branch only tests reach.
"""

import numpy as np
import pytest
from cablesim import policies, random_draws, run, simulate, weibull

from tests import helpers

N_SEGMENTS = 4
N_YEARS = 3
N_CLASSES = 2


def inputs(**overrides: object) -> dict[str, object]:
    """Builds one call's arguments, with everything varied deliberately.

    Costs are round so that a budget can be set to fund an exact number of
    segments: every segment is 100 feet at 10 dollars a foot plus 500 of
    mobilization, so planned replacement is 1,500 and emergency is 3,750.

    Args:
        **overrides: Arguments to replace.

    Returns:
        Keyword arguments for the annual loop.
    """
    arguments: dict[str, object] = {
        "length_ft": np.full(N_SEGMENTS, 100.0),
        "customers": np.array([10.0, 20.0, 30.0, 40.0]),
        "customer_minutes_per_failure": np.array([100.0, 200.0, 300.0, 400.0]),
        "customer_minutes_per_planned": np.array([1.0, 2.0, 3.0, 4.0]),
        "outage_cost_per_failure": np.array([1_000.0, 2_000.0, 3_000.0, 4_000.0]),
        "class_index": np.array([0, 0, 1, 1], dtype=np.uint8),
        "age0": np.array([10.0, 20.0, 30.0, 40.0]),
        "shape": np.full(N_SEGMENTS, 6.2),
        "scale": np.full(N_SEGMENTS, 50.0),
        "replacement_shape": np.full(N_SEGMENTS, 6.2),
        "replacement_scale": np.full(N_SEGMENTS, 65.0),
        "cost_per_ft": np.full(N_SEGMENTS, 10.0),
        "draw_key": random_draws.draw_key(20260907),
        "first_replication": 0,
        "n_reps": 1,
        "budget": np.full(N_YEARS, 1e9),
        "cost_escalation": np.ones(N_YEARS),
        "policy": helpers.resolved("run_to_failure"),
        "emergency_multiplier": 2.5,
        "mobilization_per_segment": 500.0,
        "emergency_charged_to_budget": False,
        "n_classes": N_CLASSES,
        "n_years": N_YEARS,
    }
    arguments.update(overrides)
    return arguments


def test_forcing_the_scale_to_zero_fails_every_segment_every_year() -> None:
    """The deterministic case, and the one that pins the replacement rule.

    A replacement enters service at the start of the following year, so nothing
    can fail twice in one year and the year loop needs no inner iteration.
    Without that rule this case does not terminate.
    """
    results = simulate.run_chunk(
        **inputs(
            scale=np.full(N_SEGMENTS, helpers.FAILS_AT_ONCE),
            replacement_scale=np.full(N_SEGMENTS, helpers.FAILS_AT_ONCE),
        )
    )

    for year in range(N_YEARS):
        assert results.failures[0, year].tolist() == [2.0, 2.0], year
        assert results.customers_interrupted[0, year].tolist() == [30.0, 70.0]
        assert results.customer_minutes[0, year].tolist() == [300.0, 700.0]
        # Two segments a class, at 1,500 planned times the 2.5 multiplier.
        assert results.emergency_spend[0, year].tolist() == [7_500.0, 7_500.0]


def test_forcing_the_scale_past_the_horizon_fails_nothing() -> None:
    """The other end of the deterministic case."""
    results = simulate.run_chunk(
        **inputs(
            scale=np.full(N_SEGMENTS, helpers.NEVER_FAILS),
            replacement_scale=np.full(N_SEGMENTS, helpers.NEVER_FAILS),
        )
    )

    assert not results.failures.any()
    assert not results.emergency_spend.any()
    assert not results.customer_minutes.any()


def test_the_initial_draw_is_conditional_on_the_age_already_survived() -> None:
    """Drawing unconditionally makes an old population behave as though new.

    An 80-year-old segment at a scale of 50 is far past its median life and has
    months left, not decades. Drawn unconditionally it would get the lifetime
    of new cable — about 47 years at this uniform — and fail nowhere inside the
    horizon, which is a run that completes with plausible-looking curves.
    """
    arguments = inputs(
        age0=np.full(N_SEGMENTS, 80.0),
        replacement_scale=np.full(N_SEGMENTS, helpers.NEVER_FAILS),
    )
    results = simulate.run_chunk(**arguments)

    # The draws are no longer handed in, so the expectation is computed from
    # the position they are read at. That makes this a stronger check than
    # forcing a uniform would: it pins the address and the formula together.
    starting = random_draws.uniforms_at(
        arguments["draw_key"],
        random_draws.PURPOSE["lifetimes"],
        np.zeros(N_SEGMENTS, dtype=np.uint32),
        np.arange(N_SEGMENTS, dtype=np.uint32),
        0,
    )
    conditional = weibull.draw_remaining_life(
        starting, np.full(N_SEGMENTS, 80.0), arguments["shape"], arguments["scale"]
    )
    unconditional = weibull.draw_lifetime(
        starting, arguments["shape"], arguments["scale"]
    )
    assert (conditional < N_YEARS).all(), "every segment should fail in the horizon"
    assert (unconditional > N_YEARS).all(), (
        "drawn as new cable they would all outlive the horizon, which is what "
        "makes the two draws tell apart here"
    )

    # The replacements never fail, so every reported failure is a starting one
    # and lands in the year its remaining life implies.
    for year in range(N_YEARS):
        assert results.failures[0, year].sum() == float(
            (np.floor(conditional) == year).sum()
        ), f"year {year} does not match the conditional draw"


def test_a_segment_that_failed_this_year_is_not_also_planned_work() -> None:
    """Eligibility means not already replaced this year, and nothing else.

    Everything fails every year here, so a policy that funds from the whole
    population has an empty candidate set every year despite an unlimited
    budget.
    """
    results = simulate.run_chunk(
        **inputs(
            policy=helpers.resolved("risk_ranked"),
            scale=np.full(N_SEGMENTS, helpers.FAILS_AT_ONCE),
            replacement_scale=np.full(N_SEGMENTS, helpers.FAILS_AT_ONCE),
        )
    )

    assert results.failures.sum() == N_SEGMENTS * N_YEARS
    assert not results.planned_replacements.any()
    assert not results.planned_spend.any()


def test_unspent_budget_does_not_carry_into_the_next_year() -> None:
    """Each year gets the amount in the series and no more.

    A budget of 1,400 a year never funds a 1,500 segment. Carried forward it
    would reach 2,800 by the second year and fund one, which is what this
    refuses.
    """
    results = simulate.run_chunk(
        **inputs(
            policy=helpers.resolved("risk_ranked"),
            budget=np.full(N_YEARS, 1_400.0),
            scale=np.full(N_SEGMENTS, helpers.NEVER_FAILS),
            replacement_scale=np.full(N_SEGMENTS, helpers.NEVER_FAILS),
        )
    )

    assert not results.planned_replacements.any()


def test_the_budget_funds_candidates_down_the_ranked_order() -> None:
    """Three segments at 1,500 fit inside 5,000; the fourth does not."""
    results = simulate.run_chunk(
        **inputs(
            policy=helpers.resolved("risk_ranked"),
            budget=np.array([5_000.0, 0.0, 0.0]),
            scale=np.full(N_SEGMENTS, helpers.NEVER_FAILS),
            replacement_scale=np.full(N_SEGMENTS, helpers.NEVER_FAILS),
        )
    )

    assert results.planned_replacements[0, 0].sum() == 3.0
    assert results.planned_spend[0, 0].sum() == pytest.approx(4_500.0)
    assert not results.planned_replacements[0, 1:].any(), "no budget after year 0"


def test_planned_work_reports_its_customer_minutes_outside_the_indices() -> None:
    """Planned minutes are a real cost and enter no reliability index.

    A customer out for four hours does not care that the work was scheduled,
    but the indices are defined over unplanned interruptions, so the two totals
    stay apart.
    """
    results = simulate.run_chunk(
        **inputs(
            policy=helpers.resolved("risk_ranked"),
            budget=np.array([1e9, 0.0, 0.0]),
            scale=np.full(N_SEGMENTS, helpers.NEVER_FAILS),
            replacement_scale=np.full(N_SEGMENTS, helpers.NEVER_FAILS),
        )
    )

    assert results.planned_customer_minutes[0, 0].tolist() == [3.0, 7.0]
    assert not results.customer_minutes.any(), "no failures, so no index minutes"
    assert not results.customers_interrupted.any()


def test_charging_emergency_spend_to_the_budget_crowds_out_planned_work() -> None:
    """The ordering is part of the contract, not an implementation detail.

    One segment fails every year and the other three never do. Its emergency
    replacement costs 3,750 against a 5,000 budget. Left in its own operations
    bucket, the remaining three segments are all affordable; charged first,
    what is left will not cover even one.
    """
    scale = np.array(
        [helpers.FAILS_AT_ONCE, *[helpers.NEVER_FAILS] * (N_SEGMENTS - 1)]
    )
    arguments = inputs(
        policy=helpers.resolved("risk_ranked"),
        budget=np.full(N_YEARS, 5_000.0),
        scale=scale,
        replacement_scale=scale,
    )

    separate = simulate.run_chunk(**{**arguments, "emergency_charged_to_budget": False})
    charged = simulate.run_chunk(**{**arguments, "emergency_charged_to_budget": True})

    assert separate.planned_replacements[0, 0].sum() == 3.0
    assert charged.planned_replacements[0, 0].sum() == 0.0
    # The failure itself is unaffected either way; only what follows it moves.
    assert separate.failures[0, 0].sum() == charged.failures[0, 0].sum() == 1.0


def test_a_tie_is_broken_on_segment_identifier_when_only_one_fits() -> None:
    """Equal ages, a budget for exactly one, and the lower identifier wins."""
    results = simulate.run_chunk(
        **inputs(
            policy=helpers.resolved("age_threshold", threshold_years=5),
            age0=np.full(N_SEGMENTS, 40.0),
            budget=np.array([1_500.0, 0.0, 0.0]),
            scale=np.full(N_SEGMENTS, helpers.NEVER_FAILS),
            replacement_scale=np.full(N_SEGMENTS, helpers.NEVER_FAILS),
        )
    )

    # Segments 0 and 1 are class 0; 2 and 3 are class 1.
    assert results.planned_replacements[0, 0].tolist() == [1.0, 0.0]


def test_every_policy_at_zero_budget_matches_run_to_failure() -> None:
    """A free end-to-end check on the whole loop.

    Nothing funded means nothing differs, whatever the policy would have
    ranked, so any divergence here is the loop leaking policy state into
    something other than the funding decision.
    """
    # Old enough at the shipped scale that failures land inside the horizon;
    # at the default ages nothing fails in three years and the comparison
    # below would hold for the wrong reason.
    aged = np.array([60.0, 70.0, 80.0, 85.0])
    baseline = simulate.run_chunk(
        **inputs(age0=aged, budget=np.full(N_YEARS, 1e9))
    )
    assert baseline.failures.any(), "the comparison is vacuous without failures"

    for name, params in (
        ("age_threshold", {"threshold_years": 5}),
        ("risk_ranked", {}),
        ("risk_ranked", {"rank_by": "score_per_dollar"}),
        ("worst_first", {}),
        ("random", {}),
    ):
        starved = simulate.run_chunk(
            **inputs(
                policy=helpers.resolved(name, **params),
                age0=aged,
                budget=np.zeros(N_YEARS),
            )
        )
        for field, left, right in zip(
            simulate.Results._fields, baseline, starved, strict=True
        ):
            assert np.array_equal(left, right), f"{name} {params} differs on {field}"


def test_cost_escalation_lifts_every_dollar_in_the_year_together() -> None:
    """Escalating construction while holding lost load fixed reweights scoring.

    Over a long horizon that quietly shrinks the consequence term against the
    cost term, so the multiplier applies to both or to neither.
    """
    escalation = np.array([1.0, 2.0, 4.0])
    results = simulate.run_chunk(
        **inputs(
            cost_escalation=escalation,
            scale=np.full(N_SEGMENTS, helpers.FAILS_AT_ONCE),
            replacement_scale=np.full(N_SEGMENTS, helpers.FAILS_AT_ONCE),
        )
    )

    spend = results.emergency_spend[0].sum(axis=1)
    assert spend.tolist() == pytest.approx([15_000.0, 30_000.0, 60_000.0])


def test_a_replaced_segment_is_age_zero_when_the_next_year_is_scored() -> None:
    """Otherwise the same segment wins the same tie-break every year.

    Four segments of equal age, a budget for exactly one, and a threshold they
    all meet. The tie goes to the lowest identifier, so year 0 funds segment 0.
    In year 1 that segment is new cable and below the threshold, so the tie is
    now between the other three and segment 1 wins it; in year 2, segment 2.
    Without the reset, segment 0 stays oldest-equal and is funded three times.
    """
    results = simulate.run_chunk(
        **inputs(
            policy=helpers.resolved("age_threshold", threshold_years=40),
            age0=np.full(N_SEGMENTS, 40.0),
            budget=np.full(N_YEARS, 1_500.0),
            scale=np.full(N_SEGMENTS, helpers.NEVER_FAILS),
            replacement_scale=np.full(N_SEGMENTS, helpers.NEVER_FAILS),
        )
    )

    # Segments 0 and 1 are class 0; 2 and 3 are class 1.
    funded = [results.planned_replacements[0, year].tolist() for year in range(N_YEARS)]
    assert funded == [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]


def test_the_value_of_lost_load_escalates_with_construction_cost() -> None:
    """Escalating one and not the other reweights the score year by year.

    Two candidates with the same failure probability: one where most of the
    score is customer value at risk, one where most of it is the emergency
    premium avoided. Escalating both terms leaves their order alone, which is
    the point of applying the multiplier to every dollar in the year. Escalate
    only construction and the second overtakes the first, and it is dear enough
    that the budget then funds nothing at all.
    """
    results = simulate.run_chunk(
        **inputs(
            policy=helpers.resolved("risk_ranked"),
            # Planned cost is 1,000 and 6,000 at par, so 4,000 and 24,000 once
            # the year's escalation is applied; the last two are inert.
            cost_per_ft=np.array([10.0, 60.0, 100.0, 100.0]),
            mobilization_per_segment=0.0,
            outage_cost_per_failure=np.array([10_000.0, 1_000.0, 0.0, 0.0]),
            age0=np.full(N_SEGMENTS, 40.0),
            # The last two never fail and score zero, so they rank below both.
            scale=np.array([50.0, 50.0, helpers.NEVER_FAILS, helpers.NEVER_FAILS]),
            replacement_scale=np.full(N_SEGMENTS, helpers.NEVER_FAILS),
            class_index=np.array([0, 1, 1, 1], dtype=np.uint8),
            cost_escalation=np.array([4.0, 1.0, 1.0]),
            # Covers the first candidate at escalated prices and not the second.
            budget=np.array([4_000.0, 0.0, 0.0]),
        )
    )

    assert results.planned_replacements[0, 0].tolist() == [1.0, 0.0]
    assert results.planned_spend[0, 0].tolist() == pytest.approx([4_000.0, 0.0])


def test_a_replacement_reads_the_draw_for_the_year_it_enters_service() -> None:
    """Index ``y + 1`` for a replacement made in year ``y``, not index ``y``.

    The year is a field of a draw's address, and it is what two
    implementations have to agree on. Nothing else pins it: comparing two runs
    that both read the wrong position is comparing a thing against itself.

    Here every segment fails in year 0 by construction, so every replacement
    enters service in year 1 and there is exactly one position its lifetime
    could have come from.
    """
    # Every segment fails in year 0 by construction rather than by a draw, so
    # every replacement enters service in year 1 and there is one position its
    # lifetime could have come from.
    arguments = inputs(
        scale=np.full(N_SEGMENTS, helpers.FAILS_AT_ONCE),
        replacement_scale=np.full(N_SEGMENTS, 20.0),
        replacement_shape=np.full(N_SEGMENTS, 6.2),
    )
    results = simulate.run_chunk(**arguments)
    assert results.failures[0, 0].sum() == N_SEGMENTS, "everything fails in year 0"

    def failure_year_if_it_reads(year: int) -> np.ndarray:
        """Which year each replacement fails in, had it read that year's draw."""
        drawn = random_draws.uniforms_at(
            arguments["draw_key"],
            random_draws.PURPOSE["lifetimes"],
            np.zeros(N_SEGMENTS, dtype=np.uint32),
            np.arange(N_SEGMENTS, dtype=np.uint32),
            year,
        )
        life = weibull.draw_lifetime(
            drawn, arguments["replacement_shape"], arguments["replacement_scale"]
        )
        return np.floor(1.0 + life).astype(int)

    correct, wrong = failure_year_if_it_reads(1), failure_year_if_it_reads(0)
    assert (correct != wrong).any(), (
        "the two positions imply the same failure years here, so this cannot "
        "tell them apart and the fixture needs a different seed"
    )

    # Count the second failures the year-1 position implies, and assert the run
    # reported those rather than the ones the year-0 position would imply.
    for year in range(1, N_YEARS):
        assert results.failures[0, year].sum() == float((correct == year).sum()), (
            f"year {year} does not match the draw at position year 1, which is "
            f"the one a replacement made in year 0 enters service on"
        )


def test_a_replaced_segment_is_scored_at_age_zero_not_age_one() -> None:
    """The rule is age 0 in ``y + 1`` and age 1 in ``y + 2``.

    Off by one, a segment replaced last year is already old enough for a
    one-year threshold and is funded again immediately.

    The budget deliberately covers every segment, so the difference shows in
    how many are funded rather than in which. With a budget for only one, a
    just-replaced segment is the youngest either way and ranks last either way,
    so being wrongly eligible costs it nothing and the error is invisible.
    """
    results = simulate.run_chunk(
        **inputs(
            policy=helpers.resolved("age_threshold", threshold_years=1),
            age0=np.full(N_SEGMENTS, 40.0),
            budget=np.full(N_YEARS, 1e9),
            scale=np.full(N_SEGMENTS, helpers.NEVER_FAILS),
            replacement_scale=np.full(N_SEGMENTS, helpers.NEVER_FAILS),
        )
    )

    # Year 0 replaces all four. In year 1 every one of them is age 0, below the
    # threshold, so nothing is eligible; by year 2 they are age 1 and all four
    # are funded again. At age 1 in year 1, all four would be funded then too.
    funded = [results.planned_replacements[0, year].sum() for year in range(N_YEARS)]
    assert funded == [4.0, 0.0, 4.0]


def test_the_rankable_tags_are_enumerated_rather_than_taken_from_the_mapping() -> None:
    """``RANKABLE`` must name the five, not derive itself from ``KIND``.

    Deriving it would accept a sixth policy the moment it is named in the
    mapping — before ``rank_key`` has a branch for it, and before the Rust side
    has been edited at all. The reference would then score that policy zero for
    every candidate and fund in segment order while the kernel refused it,
    which is the divergence the enumeration exists to prevent.

    No runtime mutation can tell the two spellings apart while the sets happen
    to be equal, so this asserts the membership directly: it turns that future
    divergence into a red test at the edit that would cause it.
    """
    assert {
        policies.KIND[name]
        for name in (
            "run_to_failure",
            "age_threshold",
            "risk_ranked",
            "worst_first",
            "random",
        )
    } == policies.RANKABLE


def test_the_per_segment_arguments_match_the_population_columns() -> None:
    """The two lists of the same twelve names, held in step.

    ``simulate.SEGMENT_ARGUMENTS`` is what the length guard iterates and
    ``run.SEGMENT_COLUMNS`` is what is taken from the population frame; they
    differ only in ``age``, which the loop receives as ``age0`` because it
    holds a current age that moves.

    Omitting a thirteenth name from the second fails loudly at the call.
    Omitting it from the first drops that array from the length guard silently,
    with no error and no other test failing — where the Rust binding's own list
    is typed as twelve pairs and a thirteenth argument is a build error.
    """
    assert set(simulate.SEGMENT_ARGUMENTS) == (set(run.SEGMENT_COLUMNS) - {"age"}) | {
        "age0"
    }
