"""The benchmark harness: what it times, and what it refuses to call a match.

A timing is only worth reading beside the claim that the thing timed still
computes the right answer, so the harness produces both in one pass. These
tests are about that pairing rather than about any duration: nothing here
asserts that something was fast, because a test that did would fail on a loaded
machine and teach nothing when it did.
"""

import pathlib

import numpy as np
import pytest
from cablesim import benchmarks, config, constants, policies, run, simulate

from tests import helpers


def small_settings(n_segments: int = 200, n_reps: int = 2) -> config.Config:
    """A configuration small enough to time several implementations against.

    Args:
        n_segments: Population size.
        n_reps: Replications per configuration.

    Returns:
        The resized configuration.
    """
    return config.resize_population(
        config.load_config(constants.DEFAULT_CONFIG_PATH), n_segments, n_reps=n_reps
    )


def test_every_runnable_implementation_is_timed_and_matches() -> None:
    """The table covers what exists, and each row reproduces the reference.

    Read from the runnable registry rather than from a list here, for the same
    reason the parity tests are: an implementation added there should be timed
    and checked from the moment it exists, not from the moment someone
    remembers to name it in a second place.
    """
    settings = small_settings()
    configurations = [
        benchmarks.Configuration(name, 1, settings.simulation.n_reps)
        for name in run.RUNNABLE
    ]

    table = benchmarks.compare(settings, "risk_ranked", configurations, repeats=1)

    assert table.height == len(run.RUNNABLE)
    assert table["matches_reference"].all(), table.filter(~table["matches_reference"])
    assert (table["seconds"] > 0.0).all()
    # Per-replication time is the column rows are compared on, so it has to be
    # the quotient rather than an independently measured number.
    assert np.allclose(
        table["seconds_per_replication"].to_numpy(),
        table["seconds"].to_numpy() / table["replications"].to_numpy(),
    )


def test_a_configuration_naming_no_implementation_is_refused() -> None:
    """A misspelled name fails before anything is timed.

    Left to the lookup inside the loop, it would fail after the reference had
    already been run for that replication count, which on a full-size table is
    minutes of work thrown away for a typo.
    """
    settings = small_settings()

    with pytest.raises(KeyError, match="are the ones that exist"):
        benchmarks.compare(
            settings,
            "risk_ranked",
            [benchmarks.Configuration("batched_pandas", 1, 2)],
            repeats=1,
        )


def test_a_result_that_differs_anywhere_is_not_a_match() -> None:
    """Agreement is every cell, not a summary that a difference could survive.

    Asserting that two identical results match cannot fail. What has to be
    shown is that a single altered cell is caught, because the harness reports
    agreement as one boolean and everything downstream trusts it.
    """
    settings = small_settings()
    arguments = benchmarks.chunk_arguments(settings, settings.simulation.n_reps)
    policy = policies.resolve(
        next(spec for spec in settings.policies if spec.name == "risk_ranked")
    )
    expected = simulate.run_chunk(**arguments, policy=policy)

    assert benchmarks.agrees(expected, expected)

    for index, name in enumerate(simulate.Results._fields):
        altered = list(expected)
        moved = altered[index].copy()
        # The smallest change representable in the last field it lands in,
        # rather than a visible one: a tolerance would pass this and the point
        # of the check is that there is no tolerance.
        moved.flat[0] = np.nextafter(moved.flat[0], np.inf)
        altered[index] = moved
        assert not benchmarks.agrees(expected, simulate.Results(*altered)), (
            f"a changed {name} was reported as a match"
        )


def test_the_speedup_columns_are_ratios_to_the_rows_they_name() -> None:
    """Both denominators are what they say, and the fastest Python is Python.

    Two ratio columns exist because the batched NumPy loop is not always the
    fastest Python implementation, and a speedup quoted against a baseline the
    reference beats is flattered. Whether that is still true on a given machine
    is not the point: what this pins is that each column divides by the row it
    is named after.
    """
    settings = small_settings()
    configurations = [
        benchmarks.Configuration(name, 1, settings.simulation.n_reps)
        for name in run.RUNNABLE
    ]

    table = benchmarks.compare(settings, "risk_ranked", configurations, repeats=1)

    baseline = table.filter(table["implementation"] == "batched_numpy")
    assert baseline["speedup_over_batched_numpy"][0] == pytest.approx(1.0)

    python_rows = table.filter(
        table["implementation"].is_in(benchmarks.PYTHON_IMPLEMENTATIONS)
    )
    fastest = python_rows["seconds_per_replication"].min()
    # The fastest Python row is the one whose ratio is exactly one; every other
    # row's is that row's time over its own, in whichever direction.
    assert python_rows.filter(python_rows["seconds_per_replication"] == fastest)[
        "speedup_over_fastest_python"
    ][0] == pytest.approx(1.0)
    assert np.allclose(
        table["speedup_over_fastest_python"].to_numpy(),
        fastest / table["seconds_per_replication"].to_numpy(),
    )


@pytest.mark.parametrize(
    "path", helpers.CALLERS_THAT_NAME_IMPLEMENTATIONS, ids=lambda p: p.name
)
def test_the_shipped_callers_only_name_implementations_that_exist(
    path: pathlib.Path,
) -> None:
    """A driver script or notebook naming a retired implementation is broken.

    ``compare`` refuses an unknown name before it times anything, so a caller
    that asks for one raises ``KeyError`` on every invocation rather than
    producing a smaller table. Neither of these files runs in a default test
    session — one is a script and the other is behind the ``notebooks`` marker,
    which nothing runs automatically — so retiring an implementation without
    editing them leaves the documented way to produce the benchmark table
    failing, with nothing reporting it.
    """
    requested = helpers.requested_implementations(path)
    runnable = set(run.RUNNABLE)

    assert requested, f"{path.name} names no implementation; has the call moved?"
    assert requested <= runnable, (
        f"{path.name} asks for {sorted(requested - runnable)}, which "
        f"{sorted(runnable)} does not carry"
    )
