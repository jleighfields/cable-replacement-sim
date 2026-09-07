"""The kernel's boundary: what it accepts, and what it must refuse.

`test_parity.py` asks whether the kernel computes the reference's numbers on
inputs both can run. This module asks the other question: what happens to input
the kernel cannot run. The two are separate because a boundary check that never
fires is invisible to a parity test — both implementations agree on every input
the tests hand them, and the guard is only reached by input no test builds.

Each case here must come back as a Python exception naming the argument. A Rust
panic reaches Python as `pyo3_runtime.PanicException`, which does not inherit
from `Exception`, so `except Exception` around a sweep does not catch it and the
panic message goes to stderr rather than into the traceback.
"""

import numpy as np
import pytest
from cablesim import kernel, policies, simulate

from tests import helpers


def minimal_arguments(
    n_segments: int, n_years: int = 1, n_reps: int = 1, n_classes: int = 1
) -> dict[str, object]:
    """Builds the smallest argument set the kernel accepts.

    The parity fixtures build a real population, which is what a parity test
    needs and what a boundary test cannot use: the cases below are degenerate
    by construction — no segments at all, or a class index past the result
    axis — and no configuration produces them.

    Every segment fails in its first year, so the accumulation paths that index
    the class axis are reached rather than skipped.

    Args:
        n_segments: Segments in the population.
        n_years: Horizon, in years.
        n_reps: Replications in the chunk.
        n_classes: Segment classes, sizing the third result axis.

    Returns:
        Every argument of ``kernel.run_chunk`` except ``policy``, keyed by
        name.
    """
    per_segment = np.zeros(n_segments)
    return {
        "length_ft": np.full(n_segments, 100.0),
        "customers": np.full(n_segments, 10.0),
        "customer_minutes_per_failure": np.full(n_segments, 60.0),
        "customer_minutes_per_planned": per_segment.copy(),
        "outage_cost_per_failure": per_segment.copy(),
        "class_index": np.zeros(n_segments, dtype=np.uint8),
        "age0": np.full(n_segments, 10.0),
        "shape": np.full(n_segments, 6.2),
        "scale": np.full(n_segments, helpers.FAILS_AT_ONCE),
        "replacement_shape": np.full(n_segments, 6.2),
        "replacement_scale": np.full(n_segments, helpers.FAILS_AT_ONCE),
        "cost_per_ft": np.full(n_segments, 10.0),
        "lifetime_uniforms": np.full((n_reps, n_segments, n_years + 1), 0.5),
        "policy_uniforms": np.full((n_reps, n_segments), 0.5),
        "budget": np.zeros(n_years),
        "cost_escalation": np.ones(n_years),
        "emergency_multiplier": 2.5,
        "mobilization_per_segment": 500.0,
        "emergency_charged_to_budget": False,
        "n_classes": n_classes,
        "n_years": n_years,
    }


def test_the_kernel_refuses_a_column_ordered_draw_array(
    deterministic_arguments: dict[str, object],
) -> None:
    """A Fortran-ordered array is not C-contiguous and must be rejected.

    The boundary reads every array as a flat slice in C order, so an array
    whose memory runs down the columns puts a different segment's draw at every
    position. Nothing about the values or the shape distinguishes it: the run
    completes and returns a plausible number that is not the one the caller's
    array describes. Measured on a six-segment chunk, the kernel returned
    failure counts of ``[0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1]`` against the
    reference's ``[0, 0, 1, 2, 0, 0, 1, 0, 0, 0, 0, 1]`` from the same array.

    A row-ordered array and a column-ordered one are the same values, which is
    why this cannot be caught by comparing results: it has to be refused at the
    boundary.
    """
    arguments = {
        **deterministic_arguments,
        "lifetime_uniforms": np.asfortranarray(
            deterministic_arguments["lifetime_uniforms"]
        ),
    }
    assert not arguments["lifetime_uniforms"].flags["C_CONTIGUOUS"]

    with pytest.raises(ValueError, match="lifetime_uniforms"):
        kernel.run_chunk(**arguments, policy=helpers.resolved("worst_first"))


def test_the_kernel_raises_rather_than_panicking_on_an_empty_population() -> None:
    """A chunk with no segments must report, not divide by zero.

    The replication count is recovered from the draw array's length divided by
    the segment count, so a population of none is an integer division by zero.
    That is a Rust panic rather than a raise, and a panic crosses into Python
    as something ``except Exception`` does not catch.
    """
    arguments = minimal_arguments(n_segments=0)

    with pytest.raises(ValueError):
        kernel.run_chunk(**arguments, policy=helpers.resolved("worst_first"))


def test_the_kernel_raises_rather_than_panicking_on_a_class_past_the_axis() -> None:
    """A class index the result axis has no room for must report, not panic.

    Without the boundary's range check, the first segment of that class to fail
    would index past the end of a result buffer, which is a Rust panic. That is
    data-dependent: the index is reached only when a segment of that class is
    selected in some year, so the same mismatched pair of arguments can
    complete on one seed and panic on another. The check reads every entry
    before the first year runs, so the refusal does not depend on the draws.

    The reference raises on this particular input too, because the year's
    totals no longer broadcast onto the class axis — but only because both
    segments fail here. A class index that is never selected passes through the
    reference silently.
    """
    arguments = minimal_arguments(n_segments=2, n_classes=1)
    arguments["class_index"] = np.array([0, 7], dtype=np.uint8)

    with pytest.raises(ValueError):
        kernel.run_chunk(**arguments, policy=helpers.resolved("run_to_failure"))


def test_the_kernel_refuses_a_per_segment_array_of_the_wrong_length() -> None:
    """A short per-segment array is refused, naming which one it was.

    Every array is indexed by segment position, so one entry short reads past
    the end of a buffer. This is the guard that keeps that a Python exception
    rather than a panic, and the argument's name is in the message because
    twelve arrays are checked by one loop.
    """
    arguments = {**minimal_arguments(4), "cost_per_ft": np.full(3, 10.0)}

    with pytest.raises(ValueError, match="cost_per_ft has 3 entries"):
        kernel.run_chunk(**arguments, policy=helpers.resolved("run_to_failure"))


def test_the_kernel_refuses_a_draw_array_that_is_short_a_year() -> None:
    """The draw array must carry one column per year plus the initial draw.

    A column short raises on its own only when a replacement happens to fall in
    the final year, so without this check a run completes against the wrong
    array and is wrong nowhere visible.
    """
    arguments = minimal_arguments(4, n_years=3)
    arguments["lifetime_uniforms"] = np.full((1, 4, 3), 0.5)

    with pytest.raises(ValueError, match="lifetime_uniforms is"):
        kernel.run_chunk(**arguments, policy=helpers.resolved("run_to_failure"))


def test_the_kernel_refuses_a_policy_tag_it_does_not_recognize() -> None:
    """A tag outside the mapping is refused rather than scored as zero.

    The tags are authored in the Python policy module and read again as
    constants in the Rust one. An unrecognized one would otherwise fall to the
    catch-all branch, score every candidate zero, and fund them in segment
    order — a run that completes and means nothing.
    """
    unmapped = helpers.resolved("worst_first")._replace(
        kind=max(policies.KIND.values()) + 1
    )

    with pytest.raises(ValueError, match="no ranking branch covers"):
        kernel.run_chunk(**minimal_arguments(4), policy=unmapped)


def test_a_rank_key_that_is_not_a_number_reaches_python_as_an_error() -> None:
    """The NaN check propagates out of the loop and across the boundary.

    The Rust unit tests cover the comparator itself, so what is untested there
    is the path: the error has to travel out of the year loop and be converted
    at the binding. Unconverted, a NaN sorts last under the total comparator
    and the segment is silently never funded.

    A NaN arrives the way one actually would, through a per-segment input the
    score reads, rather than through a rank array no caller can pass.
    """
    arguments = {
        **minimal_arguments(4),
        "scale": np.full(4, helpers.NEVER_FAILS),
        "replacement_scale": np.full(4, helpers.NEVER_FAILS),
        "outage_cost_per_failure": np.array([1.0, np.nan, 3.0, 4.0]),
        "budget": np.full(1, 1e9),
    }

    with pytest.raises(ValueError, match="scored NaN, first at segment_id 1"):
        kernel.run_chunk(**arguments, policy=helpers.resolved("risk_ranked"))


def a_class_index_past_the_axis() -> dict[str, object]:
    """A population whose out-of-range class never has a segment selected.

    The reference totals a year by class with ``numpy.bincount``, which grows
    its output to fit the largest index it is handed, so the mismatch surfaces
    only in a year where a segment of that class is selected. Segment 1 carries
    class 7 against a single-class axis and never fails, and the budget is zero
    so it is never funded either — which is the case that reaches neither
    implementation's accumulation and separates the two at the boundary.

    Returns:
        Every argument of ``run_chunk`` except ``policy``.
    """
    arguments = minimal_arguments(n_segments=2, n_classes=1)
    arguments["class_index"] = np.array([0, 7], dtype=np.uint8)
    scales = np.array([helpers.FAILS_AT_ONCE, helpers.NEVER_FAILS])
    arguments["scale"] = scales
    arguments["replacement_scale"] = scales.copy()
    return arguments


DEGENERATE_ARGUMENTS: dict[str, dict[str, object]] = {
    "empty_population": minimal_arguments(n_segments=0),
    "no_class_axis": minimal_arguments(n_segments=2, n_classes=0),
    "class_past_the_axis": a_class_index_past_the_axis(),
}
"""Argument sets no configuration produces, keyed by what is wrong with each.

The schema forbids all three — ``population.n_segments`` is at least 1, the
class list is non-empty, and ``class_index`` is built from that list — so these
reach an implementation only through a direct call, which is what both the
parity tests and a driver script make.
"""


EXPECTED_REFUSAL = {
    "empty_population": "the population is empty",
    "no_class_axis": "n_classes is 0",
    "class_past_the_axis": "class_index holds 7",
}
"""Which guard each degenerate case must be refused by, not merely that one was.

The three overlap, so a case can be refused by the wrong check and still raise:
without this, deleting the empty-class-axis guard leaves the class-index guard
catching the same input and the test stays green.
"""


@pytest.mark.parametrize("wrong", sorted(DEGENERATE_ARGUMENTS), ids=str)
def test_both_implementations_refuse_the_same_degenerate_population(
    wrong: str,
) -> None:
    """The boundary checks belong to the model, not to one implementation.

    The kernel has to refuse all three because each would otherwise reach a
    Rust panic, which crosses into Python as ``PanicException`` and is not
    caught by ``except Exception``. The reference has no panic to convert, so
    on its own it would complete and return a result: zeros of the shape the
    caller asked for, or — for the class index — totals that silently omit a
    segment class the results have no axis for.

    The two are meant to be interchangeable behind one call, so they must not
    answer differently to input neither should accept: a driver script that
    runs the reference and then the kernel over the same arguments must not
    have one complete and the other raise. The reference therefore carries the
    same three checks, and this asserts both sides refuse each case.

    **The two messages are compared to each other rather than to a pattern.**
    Asserting only that each raises cannot tell one guard from another, and
    these three overlap: an absent class axis is also a class index past the
    end of it, so removing the first check leaves the second refusing the same
    input under a different name. Comparing the messages pins which guard fired
    and, because the strings are authored separately in two languages, keeps
    them saying the same thing.
    """
    arguments = DEGENERATE_ARGUMENTS[wrong]
    policy = helpers.resolved("run_to_failure")

    with pytest.raises(ValueError) as from_kernel:
        kernel.run_chunk(**arguments, policy=policy)
    with pytest.raises(ValueError) as from_reference:
        simulate.run_chunk(**arguments, policy=policy)

    assert str(from_reference.value) == str(from_kernel.value)
    assert EXPECTED_REFUSAL[wrong] in str(from_kernel.value)


def test_both_implementations_refuse_a_policy_tag_that_names_no_policy() -> None:
    """An unrecognized tag must not produce a run on either side.

    The tags are authored in ``policies.KIND`` and read again as constants in
    the Rust module, so a tag outside that range means the two sides disagree
    about the mapping. The kernel refuses it. The reference falls to the
    catch-all branch of ``rank_key``, scores every candidate zero, and funds
    them in segment_id order, which is a run that completes and means nothing:
    on these arguments it funds two segments and reports 3,000 dollars of
    planned spend.

    The two are interchangeable behind one call, so a driver script running
    both over the same arguments must not have one complete and the other
    raise.
    """
    arguments = minimal_arguments(4, n_years=2)
    arguments["budget"] = np.array([3_000.0, 0.0])
    arguments["scale"] = np.full(4, helpers.NEVER_FAILS)
    arguments["replacement_scale"] = np.full(4, helpers.NEVER_FAILS)
    unmapped = helpers.resolved("risk_ranked")._replace(
        kind=max(policies.KIND.values()) + 1
    )

    with pytest.raises(ValueError, match="no ranking branch covers"):
        kernel.run_chunk(**arguments, policy=unmapped)
    with pytest.raises(ValueError, match="no ranking branch covers"):
        simulate.run_chunk(**arguments, policy=unmapped)


@pytest.mark.parametrize("series", ["budget", "cost_escalation"])
def test_both_implementations_refuse_a_per_year_series_longer_than_the_horizon(
    series: str,
) -> None:
    """A per-year array with spare entries is a horizon the caller got wrong.

    The kernel compares both lengths against ``n_years`` and refuses. The
    reference indexes ``budget[year]`` and ``cost_escalation[year]``, so it
    reads the first ``n_years`` entries and ignores the rest: a sweep that
    built a 40-year budget series and asked for 30 years completes, and the
    ten discarded years leave no trace in the result or the manifest.
    """
    arguments = minimal_arguments(4, n_years=3)
    arguments[series] = np.ones(5)

    with pytest.raises(ValueError, match=series):
        kernel.run_chunk(**arguments, policy=helpers.resolved("run_to_failure"))
    with pytest.raises(ValueError, match=series):
        simulate.run_chunk(**arguments, policy=helpers.resolved("run_to_failure"))


def test_a_per_segment_array_of_the_wrong_length_is_a_value_error_on_both_sides(
) -> None:
    """The same bad length must come back as the same kind of exception.

    ``simulate.run_chunk`` documents ``ValueError`` and nothing else, and the
    kernel raises one naming the argument. The reference reaches NumPy first
    and a boolean mask of the wrong length raises ``IndexError`` there, which
    a caller writing ``except ValueError`` around a sweep does not catch.
    """
    arguments = {**minimal_arguments(4), "customers": np.full(6, 10.0)}
    policy = helpers.resolved("run_to_failure")

    with pytest.raises(ValueError, match="customers"):
        kernel.run_chunk(**arguments, policy=policy)
    with pytest.raises(ValueError, match="customers"):
        simulate.run_chunk(**arguments, policy=policy)


def test_both_implementations_refuse_a_tag_no_ranking_branch_covers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tag inside the mapping but outside ``rank_key`` must be refused too.

    Both guards enumerate the five tags a ranking branch exists for —
    ``policies.RANKABLE`` on the reference, a ``matches!`` over the five
    constants in the kernel — rather than testing membership of
    ``policies.KIND`` or a range over it. Either of those would open for a
    sixth entry added to ``KIND`` the moment it was named, before either
    ``rank_key`` had a branch for it. This drives that: a tag inside the
    mapping and outside both guards, which both must refuse.

    What completes is also meaningless. Neither ``rank_key`` has a branch for
    the new tag, so both fall to the catch-all written for
    ``run_to_failure``, score every candidate zero, and fund them in
    ``segment_id`` order. On these arguments segment 3 is the only one with
    any value at risk and is the one left unfunded.
    """
    unbranched = max(policies.KIND.values()) + 1
    monkeypatch.setitem(policies.KIND, "condition_based", unbranched)
    arguments = minimal_arguments(4, n_years=1)
    arguments["budget"] = np.array([3_000.0])
    arguments["scale"] = np.full(4, helpers.NEVER_FAILS)
    arguments["replacement_scale"] = np.full(4, helpers.NEVER_FAILS)
    arguments["outage_cost_per_failure"] = np.array([1.0, 2.0, 3.0, 1e9])
    policy = helpers.resolved("risk_ranked")._replace(kind=unbranched)

    with pytest.raises(ValueError):
        simulate.run_chunk(**arguments, policy=policy)
    with pytest.raises(ValueError):
        kernel.run_chunk(**arguments, policy=policy)


MISMATCHED_ARGUMENTS: dict[str, dict[str, object]] = {
    "draw_array_short_a_year": {
        **minimal_arguments(4, n_years=3),
        "lifetime_uniforms": np.full((1, 4, 3), 0.5),
    },
    "per_segment_array_too_short": {
        **minimal_arguments(4),
        "cost_per_ft": np.full(3, 10.0),
    },
    "per_year_series_too_long": {
        **minimal_arguments(4, n_years=3),
        "budget": np.ones(5),
    },
}
"""The three length and shape guards the degenerate cases above do not reach."""


@pytest.mark.parametrize("wrong", sorted(MISMATCHED_ARGUMENTS), ids=str)
def test_both_implementations_word_a_mismatch_refusal_the_same_way(
    wrong: str,
) -> None:
    """Every guard says the same thing on both sides, not just three of them.

    ``test_both_implementations_refuse_the_same_degenerate_population``
    compares the messages for the empty population, the absent class axis and
    the class index past its end. The remaining three are otherwise checked
    only for raising *something* that names the argument, which cannot see two
    sides describing the same mismatch differently. The draw-array guard is
    where that mattered: the shapes reach the message through a Rust slice on
    one side and a Python tuple on the other, and printing either as it comes
    agrees on every word while differing on the brackets.
    """
    arguments = MISMATCHED_ARGUMENTS[wrong]
    policy = helpers.resolved("run_to_failure")

    with pytest.raises(ValueError) as from_kernel:
        kernel.run_chunk(**arguments, policy=policy)
    with pytest.raises(ValueError) as from_reference:
        simulate.run_chunk(**arguments, policy=policy)

    assert str(from_reference.value) == str(from_kernel.value)


UNRANKABLE_TAGS: dict[str, int] = {
    "just_past_the_mapping": max(policies.KIND.values()) + 1,
    "below_the_mapping": -1,
    "past_what_the_binding_reads": max(policies.KIND.values()) + 300,
}
"""Policy tags no ranking branch covers, keyed by where each sits.

Three rather than one because the binding reads ``kind`` as a ``u8``, so the
three land on different code: the first reaches both guards, and the other two
are refused by PyO3's own extraction before the kernel's guard runs at all.
``policies.resolve`` produces none of them — a direct caller building
``Resolved`` by hand is the only way to arrive here, which is what the guards
exist for and what every test in this module is.
"""


@pytest.mark.parametrize("tag", sorted(UNRANKABLE_TAGS), ids=str)
def test_both_implementations_refuse_an_unrankable_tag_the_same_way(tag: str) -> None:
    """A tag outside the five must be refused identically, not merely refused.

    ``simulate.run_chunk`` documents ``ValueError`` and nothing else, and its
    docstring names the checks as the kernel's, in the kernel's order and word
    for word. A caller writing ``except ValueError`` around a sweep that runs
    both implementations therefore catches whatever either one refuses — which
    holds for a tag just past the mapping and not for one outside the range the
    binding's ``u8`` can hold, where PyO3's extraction refuses first with a
    ``TypeError`` naming neither the tag nor the branch it lacks.
    """
    arguments = minimal_arguments(4)
    policy = helpers.resolved("risk_ranked")._replace(kind=UNRANKABLE_TAGS[tag])

    with pytest.raises(ValueError) as from_kernel:
        kernel.run_chunk(**arguments, policy=policy)
    with pytest.raises(ValueError) as from_reference:
        simulate.run_chunk(**arguments, policy=policy)

    assert str(from_reference.value) == str(from_kernel.value)


def test_the_reference_names_the_draw_array_it_cannot_read() -> None:
    """A ``policy_uniforms`` that is not two-dimensional must say so.

    The twelve per-segment arrays are checked for shape by name, because a
    two-dimensional one of the right element count would otherwise fail later
    inside NumPy as a broadcast error naming neither the argument nor the
    reason. ``policy_uniforms`` reaches no such check: the reference recovers
    the replication and segment counts by unpacking its shape on the first
    line, so a one-dimensional array raises an unpacking error that names
    nothing the caller passed. The kernel refuses it at the binding.
    """
    arguments = {**minimal_arguments(4), "policy_uniforms": np.full(4, 0.5)}

    with pytest.raises(ValueError, match="policy_uniforms"):
        simulate.run_chunk(**arguments, policy=helpers.resolved("run_to_failure"))


def test_both_implementations_refuse_a_tag_that_is_not_an_integer() -> None:
    """A float tag must be refused, not matched against a branch by value.

    ``2.0 in frozenset({0, 1, 2, 3, 4})`` is true — a float hashes equal to the
    integer it equals — and ``2.0 == KIND["risk_ranked"]`` is true with it, so
    a membership check alone lets a float run the branch it happens to equal
    while the binding refuses to extract it at all. That is one implementation
    returning an answer and the other an error, on the same arguments.

    The class is what matches here, not the wording: the binding refuses this
    before any check of ours runs, so the message is PyO3's.
    """
    arguments = minimal_arguments(4)
    policy = helpers.resolved("risk_ranked")._replace(kind=2.0)

    with pytest.raises(TypeError):
        kernel.run_chunk(**arguments, policy=policy)
    with pytest.raises(TypeError):
        simulate.run_chunk(**arguments, policy=policy)


def test_the_kernel_refuses_no_threads_at_all() -> None:
    """Zero workers would run no replication and return zeros.

    That is the shape the caller asked for, filled with the value a run of no
    replications legitimately produces, so nothing downstream could tell it
    from a real result. Rejecting it at the boundary is the only place the
    distinction still exists.
    """
    arguments = minimal_arguments(4)

    with pytest.raises(ValueError, match="threads is 0"):
        kernel.run_chunk(
            **arguments, policy=helpers.resolved("run_to_failure"), threads=0
        )


def test_the_reference_refuses_a_thread_count_it_cannot_honour() -> None:
    """Asking the scalar reference for two workers is refused, not ignored.

    Both implementations take a thread count so that a caller choosing between
    them passes the same arguments to either. Only one can act on it. Silently
    ignoring the request would let a benchmark row or a run manifest record a
    thread count that nothing ran with, which is provenance that reads as fact
    and is not.

    The kernel accepts the same value, and that difference is the point rather
    than a divergence: it is what the two implementations are for.
    """
    arguments = minimal_arguments(4)
    policy = helpers.resolved("run_to_failure")

    with pytest.raises(ValueError, match="threads is 2"):
        simulate.run_chunk(**arguments, policy=policy, threads=2)
    kernel.run_chunk(**arguments, policy=policy, threads=2)
