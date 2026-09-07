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
from cablesim import kernel, simulate

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
    unmapped = helpers.resolved("worst_first")._replace(kind=9)

    with pytest.raises(ValueError, match="policy.kind is 9"):
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


@pytest.mark.parametrize("wrong", sorted(DEGENERATE_ARGUMENTS), ids=str)
def test_both_implementations_refuse_the_same_degenerate_population(
    wrong: str,
) -> None:
    """The boundary checks belong to the model, not to one implementation.

    The kernel refuses all three because each would otherwise reach a Rust
    panic, which crosses into Python as ``PanicException`` and is not caught by
    ``except Exception``. The reference has no panic to convert, so it accepts
    all three and returns a result: zeros of the shape the caller asked for, or
    — for the class index — totals that silently omit a segment class the
    results have no axis for.

    That makes the two answer differently to input neither should accept, and
    the two are meant to be interchangeable behind one call. Whichever is
    right, they have to agree: a driver script that runs the reference and then
    the kernel over the same arguments must not have one complete and the other
    raise.
    """
    arguments = DEGENERATE_ARGUMENTS[wrong]
    policy = helpers.resolved("run_to_failure")

    with pytest.raises(ValueError):
        kernel.run_chunk(**arguments, policy=policy)
    with pytest.raises(ValueError):
        simulate.run_chunk(**arguments, policy=policy)
