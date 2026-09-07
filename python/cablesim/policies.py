"""Replacement-policy eligibility and ranking.

Each year the annual loop asks three questions of every segment: may it be
replaced, if several may then which first, and how far down that order the
budget reaches. This module answers all three, over whole arrays, for the five
policies ``KIND`` names: ``run_to_failure``, ``age_threshold``,
``risk_ranked``, ``worst_first`` and ``random``. The year loop that calls it is
``simulate.py``.

Two policies exist as controls rather than as proposals. ``worst_first`` ranks
on failure probability alone, so the gap between it and ``risk_ranked`` is the
value of weighting by consequence; ``random`` ranks on a fixed per-segment
priority, so the gap between it and everything else is the value of ranking at
all rather than merely of spending.

Replacement cost lives here too, rather than in a module of its own, because
the ``risk_ranked`` score is where the difference between planned and emergency
cost is actually used, and the annual loop is the only other caller.
"""

import math
from typing import NamedTuple

import numpy as np

from cablesim import config as config_module

KIND: dict[str, int] = {
    "run_to_failure": 0,
    "age_threshold": 1,
    "risk_ranked": 2,
    "worst_first": 3,
    "random": 4,
}
"""Integer tag per policy name.

An integer rather than the name, because the kernel compares it once per
candidate per year, and because the same tags are read on the Rust side of the
boundary. The mapping is written here and nowhere else: a tag written in two
languages diverges silently, and only the deterministic parity test would
notice.
"""


class Resolved(NamedTuple):
    """A validated policy reduced to what the annual loop needs.

    The unused fields carry explicit neutral values rather than ``None``, so
    eligibility is one comparison for every policy and the loop keeps a single
    code path. Choosing those neutral values is done here, once, beside the
    schema they come from.

    Attributes:
        kind: The tag from ``KIND``.
        threshold_years: Eligibility age, compared as ``age >=
            threshold_years``. Negative infinity makes every in-service segment
            eligible; positive infinity makes none, which is how
            ``run_to_failure`` funds nothing without a branch of its own.
        rank_by_cost: Divide the score by planned replacement cost. True only
            for ``risk_ranked`` with ``rank_by: score_per_dollar``.
    """

    kind: int
    threshold_years: float
    rank_by_cost: bool


RANKABLE: frozenset[int] = frozenset(
    {
        KIND["run_to_failure"],
        KIND["age_threshold"],
        KIND["risk_ranked"],
        KIND["worst_first"],
        KIND["random"],
    }
)
"""The tags ``rank_key`` has a branch for, enumerated rather than derived.

``KIND`` is the mapping; this is the subset that can actually be scored, and
the annual loop refuses anything outside it. The two are the same set today and
must not be written as though that is guaranteed: deriving this from ``KIND``
would mean a sixth policy added to the mapping is accepted here the moment it
is named, before ``rank_key`` has a branch for it, and it would be scored zero
and funded in segment order. The Rust side enumerates the same five for the
same reason, and a policy added to one and not the other is refused by both
until both are edited.
"""


def resolve(spec: config_module.PolicySpec) -> Resolved:
    """Reduces one validated policy entry to the loop's three fields.

    This is the single place a policy becomes the numbers an implementation
    reads, including the Rust kernel: the struct it receives carries no
    defaults of its own, because a default written on both sides of the
    boundary diverges silently.

    Args:
        spec: One entry of the configuration's ``policies`` list.

    Returns:
        The policy as the annual loop needs it.
    """
    if spec.name == "run_to_failure":
        # Positive infinity, so no finite age is ever eligible. Backwards, this
        # policy would replace everything rather than nothing.
        threshold_years = math.inf
    elif "threshold_years" in spec.params:
        threshold_years = float(spec.params["threshold_years"])
    else:
        threshold_years = -math.inf

    return Resolved(
        kind=KIND[spec.name],
        threshold_years=threshold_years,
        # Ranking on the raw score is the meaningful default, so an absent
        # `rank_by` is not a missing value. The schema rejects the key outright
        # for every policy but `risk_ranked`.
        rank_by_cost=spec.params.get("rank_by") == "score_per_dollar",
    )


def planned_cost(
    length_ft: np.ndarray, cost_per_ft: np.ndarray, mobilization_per_segment: float
) -> np.ndarray:
    """Cost of replacing each segment as planned work.

    The mobilization term is what makes short segments expensive per foot, and
    it is also the only reason ranking by score per dollar differs from ranking
    by score: without it, cost is exactly proportional to length and dividing
    by it rescales every candidate equally.

    Args:
        length_ft: Segment length, in feet.
        cost_per_ft: Installed cost per foot, which keys on segment class.
        mobilization_per_segment: Fixed cost of turning up at all.

    Returns:
        Planned replacement cost per segment, in dollars.
    """
    return length_ft * cost_per_ft + mobilization_per_segment


def rank_key(
    policy: Resolved,
    age: np.ndarray,
    failure_probability: np.ndarray,
    outage_cost_per_failure: np.ndarray,
    planned: np.ndarray,
    emergency_multiplier: float,
    priority: np.ndarray,
) -> np.ndarray:
    """Scores every segment on what its policy ranks by, highest funded first.

    ``risk_ranked`` carries two terms and needs both. The first is the customer
    value at risk this year. The second is what the utility avoids by doing the
    work planned rather than after a failure, and without it a lateral serving
    three customers is never replaced at any age, however far past the point
    where planned replacement is the cheaper of the two.

    Args:
        policy: The resolved policy.
        age: Current age of each segment, in years.
        failure_probability: Probability of failing within the year, given
            survival to ``age``.
        outage_cost_per_failure: Value of lost load if this segment fails, in
            dollars.
        planned: Planned replacement cost per segment, in dollars.
        emergency_multiplier: What an emergency replacement costs relative to
            the same work planned.
        priority: One fixed uniform per segment, which ``random`` ranks on.

    Returns:
        The rank key per segment. Values for segments this policy cannot fund
        are never read, because eligibility is applied before ordering.
    """
    if policy.kind == KIND["age_threshold"]:
        key = age
    elif policy.kind == KIND["risk_ranked"]:
        # The cost avoided by acting first, which is the emergency premium
        # rather than the whole emergency cost: the planned work is paid either
        # way.
        avoided = planned * (emergency_multiplier - 1.0)
        key = failure_probability * (outage_cost_per_failure + avoided)
    elif policy.kind == KIND["worst_first"]:
        key = failure_probability
    elif policy.kind == KIND["random"]:
        key = priority
    else:
        # run_to_failure, whose eligible set is empty, so nothing is ranked.
        # The annual loop refuses any tag outside `RANKABLE` before reaching
        # here, so this branch is that policy and nothing else.
        key = np.zeros_like(age)

    if policy.rank_by_cost:
        # Planned cost, because that is what the budget is charged, so the
        # ratio is value per budget dollar. Dividing by emergency cost would
        # rank on value per dollar of exposure, which is not what the
        # constraint is denominated in.
        key = key / planned
    return key


def eligible(
    policy: Resolved, age: np.ndarray, replaced_this_year: np.ndarray
) -> np.ndarray:
    """Selects the segments this policy may fund this year.

    Nothing is ever out of service — a failure is replaced the same year — so
    the only exclusion beyond the age threshold is having already been replaced
    this year, which is what stops a segment that failed in year ``y`` also
    being planned work in year ``y``.

    Args:
        policy: The resolved policy.
        age: Current age of each segment, in years.
        replaced_this_year: True where the segment has already been replaced.

    Returns:
        A boolean mask over segments.
    """
    return (age >= policy.threshold_years) & ~replaced_this_year


def order_by_rank(rank: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Orders eligible segments by rank, breaking ties on segment identifier.

    The key is ``(rank descending, segment_id ascending)`` and is **total**, so
    ties cannot decide the answer. They arise constantly — ``age_threshold``
    ranks on an integer age shared by thousands of segments — and because the
    greedy fill stops at the first candidate that does not fit, which tied
    segment lands last is what decides whether it is funded. A total key makes
    the result independent of whether an implementation's sort is stable, which
    is what lets Python and Rust agree.

    A segment's position in these arrays is its identifier, which is why the
    tie-break needs nothing passed to it.

    Args:
        rank: The rank key for every segment, from ``rank_key``.
        candidates: Indices of the eligible segments, ascending.

    Returns:
        The eligible indices in the order the budget should be spent on them.

    Raises:
        ValueError: If any candidate's rank is not a number. Every input to
            the score is finite by construction, so a NaN here is a defect
            upstream; without this it would surface as a segment that quietly
            never gets funded.
    """
    scores = rank[candidates]
    if np.isnan(scores).any():
        unranked = candidates[np.isnan(scores)]
        raise ValueError(
            f"{unranked.size} candidate segments scored NaN, first at "
            f"segment_id {int(unranked[0])}; every input to the score is "
            f"finite by construction, so this is a defect upstream of ranking"
        )
    return candidates[np.lexsort((candidates, -scores))]


def fund(ranked: np.ndarray, planned: np.ndarray, budget: float) -> np.ndarray:
    """Spends the budget down the ranked order, stopping at the first misfit.

    Funding down a ranked list until the money runs out is what a capital plan
    does operationally, and it is also the rule that vectorizes: take the
    cumulative cost in rank order and cut at the first candidate that exceeds
    what is available. The alternative — passing over an unaffordable candidate
    and continuing to fund cheaper ones below it — spends more of the budget,
    but it is a knapsack heuristic with nothing behind it and it is inherently
    sequential, so no vectorized implementation could reproduce it. Ranking per
    dollar is what compensates for a cheap candidate being passed over, and it
    does so in the ranking rather than in the fill.

    A candidate costing exactly what remains is funded: the rule is that
    spending may not exceed the budget, not that it must fall short.

    Args:
        ranked: Eligible segment indices, best first, as ``order_by_rank``
            returns them.
        planned: Planned replacement cost per segment, in dollars, already
            carrying the year's cost escalation.
        budget: What this year has to spend, in dollars.

    Returns:
        The indices to replace, a prefix of ``ranked``.
    """
    # `cumsum` accumulates sequentially, one partial sum at a time, which is
    # what a running total in another language does. `sum` is free to add
    # pairwise and would disagree in the last bits, and here that is not a
    # rounding difference to tolerate: it decides which segment is the last one
    # funded, which is a discrete outcome the parity tests compare exactly.
    running = np.cumsum(planned[ranked])
    return ranked[: int(np.searchsorted(running, budget, side="right"))]
