"""Eligibility and ranking for the five replacement policies.

Every case here is hand-worked rather than compared against another
implementation: an analytical check can be wrong in only one way, where two
implementations agreeing can both be wrong in the same way.
"""

import math

import numpy as np
import pytest
from cablesim import batched, config, policies


def spec(name: str, **params: float | str) -> config.PolicySpec:
    """Builds one validated policy entry.

    Args:
        name: The policy name.
        **params: Policy parameters, validated by the schema.

    Returns:
        The validated entry.
    """
    return config.PolicySpec(name=name, params=params)


def test_run_to_failure_funds_nothing_at_any_age() -> None:
    """Its neutral threshold is positive infinity, not negative.

    Backwards, this policy replaces every segment every year rather than none,
    and three of the five policies would then be doing the opposite of what
    they are named.
    """
    policy = policies.resolve(spec("run_to_failure"))
    age = np.array([0.0, 45.0, 200.0, 1e308])

    assert policy.threshold_years == math.inf
    assert not policies.eligible(
        policy, age, np.zeros(age.size, dtype=bool)
    ).any()


def test_the_whole_population_policies_make_every_segment_eligible() -> None:
    """Their neutral threshold is negative infinity, so age never excludes."""
    age = np.array([0.0, 1.0, 90.0])
    never_replaced = np.zeros(age.size, dtype=bool)

    for name in ("risk_ranked", "worst_first", "random"):
        policy = policies.resolve(spec(name))
        assert policy.threshold_years == -math.inf
        assert policies.eligible(policy, age, never_replaced).all(), name


def test_an_age_threshold_is_inclusive_at_the_boundary() -> None:
    """`age >= threshold_years`, so a segment exactly at the threshold is in."""
    policy = policies.resolve(spec("age_threshold", threshold_years=45))
    age = np.array([44.0, 45.0, 46.0])

    eligible = policies.eligible(policy, age, np.zeros(3, dtype=bool))

    assert eligible.tolist() == [False, True, True]


def test_a_segment_replaced_this_year_is_not_also_planned_work() -> None:
    """The only exclusion besides age, since nothing is ever out of service."""
    policy = policies.resolve(spec("worst_first"))
    age = np.array([50.0, 50.0])

    eligible = policies.eligible(policy, age, np.array([True, False]))

    assert eligible.tolist() == [False, True]


def test_the_risk_score_needs_its_avoided_cost_term() -> None:
    """Without it a lateral serving nobody is never replaced at any age.

    A lateral with almost no customers has almost no value at risk, so the
    first term alone ranks it below a feeder that is nowhere near failing. The
    second term is what the utility actually saves by replacing it before it
    fails, and it reverses the order here.
    """
    policy = policies.resolve(spec("risk_ranked"))
    lateral, feeder = 0, 1
    failure_probability = np.array([0.9, 0.01])
    outage_cost = np.array([0.0, 1_000.0])
    planned = np.array([10_000.0, 100_000.0])

    key = policies.rank_key(
        policy,
        age=np.array([60.0, 20.0]),
        failure_probability=failure_probability,
        outage_cost_per_failure=outage_cost,
        planned=planned,
        emergency_multiplier=2.5,
        priority=np.zeros(2),
    )

    # 0.9 * (0 + 10_000 * 1.5) against 0.01 * (1_000 + 100_000 * 1.5).
    assert key[lateral] == pytest.approx(13_500.0)
    assert key[feeder] == pytest.approx(1_510.0)
    assert key[lateral] > key[feeder]
    # The value-at-risk term alone would order them the other way round.
    assert (failure_probability * outage_cost)[lateral] < (
        failure_probability * outage_cost
    )[feeder]


def test_worst_first_ignores_consequence_and_risk_ranked_does_not() -> None:
    """`worst_first` is the control that isolates weighting by consequence.

    The two policies must be able to disagree, or the gap between them measures
    nothing.
    """
    age = np.array([40.0, 40.0])
    failure_probability = np.array([0.5, 0.6])
    outage_cost = np.array([1_000_000.0, 0.0])
    planned = np.array([10_000.0, 10_000.0])
    arguments = {
        "age": age,
        "failure_probability": failure_probability,
        "outage_cost_per_failure": outage_cost,
        "planned": planned,
        "emergency_multiplier": 2.5,
        "priority": np.zeros(2),
    }
    candidates = np.array([0, 1])

    worst = policies.order_by_rank(
        policies.rank_key(policies.resolve(spec("worst_first")), **arguments),
        candidates,
    )
    risk = policies.order_by_rank(
        policies.rank_key(policies.resolve(spec("risk_ranked")), **arguments),
        candidates,
    )

    assert worst.tolist() == [1, 0], "worst_first ranks on p(t) alone"
    assert risk.tolist() == [0, 1], "risk_ranked weights by consequence"


def test_ranking_per_dollar_reorders_against_the_raw_score() -> None:
    """A cheap candidate below a dear one on raw score can outrank it per dollar."""
    arguments = {
        "age": np.array([40.0, 40.0]),
        "failure_probability": np.array([1.0, 1.0]),
        "outage_cost_per_failure": np.array([100.0, 150.0]),
        "planned": np.array([10_000.0, 100_000.0]),
        "emergency_multiplier": 1.0,  # no avoided term, so the score is the value
        "priority": np.zeros(2),
    }
    candidates = np.array([0, 1])

    raw = policies.order_by_rank(
        policies.rank_key(policies.resolve(spec("risk_ranked")), **arguments),
        candidates,
    )
    per_dollar = policies.order_by_rank(
        policies.rank_key(
            policies.resolve(spec("risk_ranked", rank_by="score_per_dollar")),
            **arguments,
        ),
        candidates,
    )

    assert raw.tolist() == [1, 0], "150 outranks 100"
    assert per_dollar.tolist() == [0, 1], "0.01 per dollar outranks 0.0015"


def test_the_per_dollar_divisor_is_planned_cost_and_not_emergency_cost() -> None:
    """Asserted on the value, because the order cannot tell the two apart.

    Emergency cost is planned cost times a scalar multiplier, so dividing by it
    rescales every candidate identically and produces the same ranking. The
    choice is still not free: the budget is charged planned cost, so that is
    what makes the ratio value per budget dollar, and the day the multiplier
    varies by segment the two divisors stop agreeing. Pinning the value is what
    keeps the formula honest while the ordering cannot.
    """
    policy = policies.resolve(spec("risk_ranked", rank_by="score_per_dollar"))
    planned = np.array([10_000.0, 100_000.0])
    arguments = {
        "age": np.array([40.0, 40.0]),
        "failure_probability": np.array([0.5, 0.25]),
        "outage_cost_per_failure": np.array([2_000.0, 8_000.0]),
        "planned": planned,
        "emergency_multiplier": 2.5,
        "priority": np.zeros(2),
    }

    key = policies.rank_key(policy, **arguments)

    # 0.5 * (2_000 + 10_000 * 1.5) = 8_500, over 10_000 planned.
    # 0.25 * (8_000 + 100_000 * 1.5) = 39_500, over 100_000 planned.
    assert key.tolist() == pytest.approx([0.85, 0.395])


def test_only_risk_ranked_ranks_per_dollar() -> None:
    """`rank_by` is a `risk_ranked` parameter, and the neutral value is False."""
    per_dollar = policies.resolve(spec("risk_ranked", rank_by="score_per_dollar"))

    assert per_dollar.rank_by_cost
    for name in ("run_to_failure", "worst_first", "random"):
        assert not policies.resolve(spec(name)).rank_by_cost, name
    assert not policies.resolve(
        spec("age_threshold", threshold_years=45)
    ).rank_by_cost


def test_ties_are_broken_on_segment_identifier_ascending() -> None:
    """The sort key is total, so a tie cannot decide what gets funded.

    Ties are the normal case rather than the edge case: `age_threshold` ranks
    on an integer age shared by thousands of segments, and the greedy fill
    stops at the first candidate that does not fit, so which tied segment lands
    last is what decides whether it is funded.
    """
    rank = np.array([7.0, 9.0, 7.0, 9.0])

    assert policies.order_by_rank(rank, np.array([0, 1, 2, 3])).tolist() == [1, 3, 0, 2]


def test_a_segment_left_out_of_the_candidates_is_not_ordered() -> None:
    """Ordering reads the mask, not just the scores."""
    rank = np.array([1.0, 99.0, 2.0])

    assert policies.order_by_rank(rank, np.array([0, 2])).tolist() == [2, 0]


def test_a_candidate_that_scores_nan_is_refused_rather_than_sorted_last() -> None:
    """NaN is unreachable, so its appearance is a defect upstream.

    Sorted rather than raised, it would surface as a segment that quietly never
    gets funded, which looks like a policy decision rather than a bug.
    """
    rank = np.array([1.0, np.nan, 3.0])

    with pytest.raises(ValueError, match="NaN"):
        policies.order_by_rank(rank, np.array([0, 1, 2]))


def test_the_batched_ordering_refuses_a_nan_the_same_way() -> None:
    """The reference's guard is pinned above; its batched twin was not.

    The two orderings are separate code — one compacts each replication's
    candidates and sorts those, the other sorts whole rows and pushes the
    ineligible behind them — so a guard removed from either is invisible to the
    other's test. And the batched form is where its absence hides best: it
    demotes an ineligible segment by scoring it negative infinity, and a NaN
    sorts to the tail alongside them, so the segment is simply never funded and
    the run completes.
    """
    rank = np.array([[1.0, np.nan, 3.0]])
    eligible = np.ones((1, 3), dtype=bool)

    with pytest.raises(ValueError, match="NaN"):
        batched.order_all_by_rank(rank, eligible)


def test_a_nan_outside_the_candidate_set_is_not_an_error() -> None:
    """Only what is ranked has to be rankable."""
    rank = np.array([1.0, np.nan, 3.0])

    assert policies.order_by_rank(rank, np.array([0, 2])).tolist() == [2, 0]


def test_planned_cost_carries_mobilization_so_short_segments_cost_more() -> None:
    """Without the fixed term, ranking per dollar would be a rescaling.

    Cost would be exactly proportional to length, so dividing by it would
    divide every candidate by the same thing up to a constant and change no
    order.
    """
    length = np.array([100.0, 1_000.0])
    cost_per_ft = np.array([95.0, 95.0])

    cost = policies.planned_cost(length, cost_per_ft, 3_500.0)

    assert cost.tolist() == [100 * 95 + 3_500, 1_000 * 95 + 3_500]
    assert cost[0] / length[0] > cost[1] / length[1]


def test_every_configured_policy_resolves_to_its_own_tag() -> None:
    """The tags are what cross the boundary, so they must be distinct."""
    names = list(policies.KIND)
    resolved = [
        policies.resolve(
            spec(name, **({"threshold_years": 45} if name == "age_threshold" else {}))
        )
        for name in names
    ]

    assert sorted(policy.kind for policy in resolved) == list(range(len(names)))


def test_funding_stops_at_the_first_candidate_that_does_not_fit() -> None:
    """Not at the cheapest thing that would still fit below it.

    Passing over an unaffordable candidate to fund cheaper ones underneath
    spends more of the budget, but it is inherently sequential and no
    vectorized implementation could reproduce it. Ranking per dollar is what
    compensates, in the ranking rather than in the fill.
    """
    ranked = np.array([0, 1, 2])
    planned = np.array([400.0, 900.0, 50.0])

    funded = policies.fund(ranked, planned, budget=1_000.0)

    assert funded.tolist() == [0], "the 50-dollar candidate below is not reached"


def test_a_candidate_costing_exactly_what_remains_is_funded() -> None:
    """Spending may not exceed the budget; it is not required to fall short."""
    ranked = np.array([0, 1])
    planned = np.array([600.0, 400.0])

    assert policies.fund(ranked, planned, budget=1_000.0).tolist() == [0, 1]
    assert policies.fund(ranked, planned, budget=999.99).tolist() == [0]


def test_a_budget_that_covers_nothing_funds_nothing() -> None:
    """Zero budget is a real configuration: it is the free end-to-end check.

    Every policy at zero budget must equal run-to-failure at any budget.
    """
    funded = policies.fund(np.array([0, 1]), np.array([10.0, 10.0]), budget=0.0)

    assert funded.tolist() == []


def test_funding_an_empty_candidate_set_is_not_an_error() -> None:
    """Run-to-failure reaches this every year of every run."""
    funded = policies.fund(
        np.array([], dtype=int), np.array([10.0, 10.0]), budget=1e9
    )

    assert funded.tolist() == []


def test_the_fill_follows_the_ranking_rather_than_the_segment_order() -> None:
    """It indexes cost through the ranked order, not through the raw array."""
    ranked = np.array([2, 0, 1])
    planned = np.array([1_000.0, 1_000.0, 10.0])

    funded = policies.fund(ranked, planned, budget=1_010.0)

    # Costs accumulate as 10 then 1,000, which exactly fills the budget. Read
    # in segment order instead they would accumulate 1,000 then 1,000, and the
    # second candidate would not fit.
    assert funded.tolist() == [2, 0]
