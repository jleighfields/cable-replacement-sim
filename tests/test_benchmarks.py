"""The benchmark harness: what it times, and what it refuses to call a match.

A timing is only worth reading beside the claim that the thing timed still
computes the right answer, so the harness produces both in one pass. These
tests are about that pairing rather than about any duration: nothing here
asserts that something was fast, because a test that did would fail on a loaded
machine and teach nothing when it did.
"""

import importlib.util
import json
import pathlib
import types
from collections.abc import Callable

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


@pytest.mark.parametrize("precision", sorted(constants.PRECISIONS))
def test_every_float_an_implementation_is_handed_carries_the_precision(
    precision: str,
) -> None:
    """A run at one precision hands its implementations nothing at the other.

    **The parity tests cannot catch this.** The precision travels as the dtype
    of the arrays, so an argument left at double widens whatever it touches —
    and every implementation touches it the same way, so all of them widen
    together and go on agreeing with each other in every cell. Comparing
    implementations proves they compute the same thing, not that they compute
    it at the width the run asked for.

    This caught exactly that: the per-year budget and cost-escalation series
    were built in double while the per-segment arrays were narrowed, so the
    whole money path ran at double under a single-precision run.
    """
    settings = small_settings()
    settings = settings.model_copy(
        update={
            "simulation": settings.simulation.model_copy(
                update={"precision": precision}
            )
        }
    )
    arguments = benchmarks.chunk_arguments(settings, settings.simulation.n_reps)

    wanted = constants.PRECISIONS[precision]
    floats = {
        name: value.dtype
        for name, value in arguments.items()
        if hasattr(value, "dtype") and value.dtype.kind == "f"
    }
    assert floats, "no float arrays were found, so this checked nothing"
    wrong = {name: str(kind) for name, kind in floats.items() if kind != wanted}
    assert not wrong, f"at {precision} these arrive as something else: {wrong}"


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


def script(name: str) -> types.ModuleType:
    """Loads one of the driver scripts as a module.

    ``scripts/`` is not part of the importable package, so the file is loaded
    by path. Reading it this way is what lets a test drive a script's argument
    handling without starting a subprocess.

    Args:
        name: The file's stem, without the extension.

    Returns:
        The loaded module, whose ``main`` takes an argument list.

    Raises:
        ImportError: If the file could not be loaded as a module.
    """
    path = constants.PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} could not be loaded as a module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_caller_naming_no_configured_policy_is_told_which_exist() -> None:
    """A misspelled policy name is refused the way a misspelled implementation is.

    Both benchmark entry points take a policy by name and look it up in the
    configured list. The implementation name beside it is checked and refused
    with the set that exists; the policy name is not, so the lookup runs off
    the end of the list and the caller gets an exception carrying no message
    and naming neither the policy asked for nor the ones configured. The
    ``--policy`` flag is documented in ``README.md`` for both scripts, and a
    typo in it is the ordinary way to reach this.

    ``LookupError`` covers the ``KeyError`` the sibling check raises for an
    implementation name; ``ValueError`` covers reporting it as a bad value
    instead. Neither covers running off the end of the list, which is the
    behaviour this pins.
    """
    settings = small_settings()
    misspelled = "risk_rankd"
    configured = [spec.name for spec in settings.policies]
    assert misspelled not in configured, (
        f"{misspelled} is a configured policy, so this test no longer asks "
        f"for one that does not exist; the configured names are {configured}"
    )

    with pytest.raises((LookupError, ValueError)) as from_compare:
        benchmarks.compare(
            settings,
            misspelled,
            [benchmarks.Configuration("kernel", 1, 2)],
            repeats=1,
        )
    with pytest.raises((LookupError, ValueError)) as from_script:
        script("measure_memory").main(
            ["--policy", misspelled, "--segments", "50", "--reps", "2"]
        )

    for raised in (from_compare, from_script):
        assert misspelled in str(raised.value), (
            f"{raised.value!r} does not name the policy that was asked for"
        )
        assert any(name in str(raised.value) for name in configured), (
            f"{raised.value!r} does not name any policy that is configured, "
            f"so it does not say what to write instead"
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


def test_the_batched_baseline_does_not_depend_on_the_row_order() -> None:
    """The denominator is one named row, not whichever batched row came first.

    The batched NumPy loop can now appear twice in a table — once on one thread
    and once on many — and ``speedup_over_batched_numpy`` is meant to name a
    fixed baseline. Taking the first matching row makes the whole column a
    function of the order the caller listed its configurations in, so the same
    measurement yields two different published speedups. The row whose ratio is
    exactly one is the baseline by construction, so asserting on which row that
    is needs no timing and cannot flake.
    """
    settings = small_settings()
    reps = settings.simulation.n_reps
    one_thread = benchmarks.Configuration("batched_numpy", 1, reps)
    two_threads = benchmarks.Configuration("batched_numpy", 2, reps)

    for configurations in ([one_thread, two_threads], [two_threads, one_thread]):
        table = benchmarks.compare(settings, "risk_ranked", configurations, repeats=1)
        baseline = table.filter(table["speedup_over_batched_numpy"] == 1.0)
        assert baseline["threads"].to_list() == [1], (
            f"listed as {[c.threads for c in configurations]}, the baseline "
            f"became the {baseline['threads'].to_list()}-thread row"
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


def ran_at(
    module: types.ModuleType, out: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> str:
    """The width a script reports, read from wherever it records it.

    Args:
        module: The loaded script.
        out: The directory a table-writing script was pointed at.
        caplog: Captured log records, for a script that reports on one line.

    Returns:
        The precision the run recorded.
    """
    provenance = out / "provenance.json"
    if provenance.exists():
        return str(json.loads(provenance.read_text())["precision"])
    return str(json.loads(caplog.messages[-1])["precision"])


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        (
            "measure_memory",
            lambda out: [
                "--implementation", "reference", "--segments", "50", "--reps", "2",
            ],
        ),
        (
            "run_benchmarks",
            lambda out: [
                "--reduced", "--repeats", "1", "--policy", "run_to_failure",
                "--out", str(out),
            ],
        ),
    ],
    ids=["measure_memory", "run_benchmarks"],
)
def test_a_script_run_computes_at_the_width_its_configuration_names(
    name: str,
    arguments: Callable[[pathlib.Path], list[str]],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A flag left off must defer to the configuration, not overwrite it.

    Both benchmark scripts take a ``--precision`` and apply it to the loaded
    configuration as an override. The flag defaults to
    ``constants.DEFAULT_PRECISION`` rather than to nothing, so the override is
    applied on every run — including the run where nobody named a width. A
    configuration that names the other one is then discarded without a word,
    and the figure that comes back is labelled with the width the flag
    supplied rather than the width the configuration asked for.

    That is the whole shape of the defect the width checks elsewhere in this
    suite exist for, one layer up: the number is right for *some* run, and
    nothing says it is not the run that was asked for.

    Both driver scripts are checked, because both build the override the same
    way and the repair to one said nothing about the other — reverting it in
    ``run_benchmarks.py`` alone left the whole suite green.

    Args:
        name: The script's file stem.
        arguments: Builds its command line, given a directory to write into.
        tmp_path: Where a table-writing script puts its output.
        monkeypatch: Replaces the configuration the script loads.
        caplog: Captures the JSON line a reporting script writes.
    """
    asked_for = "f32"
    base = small_settings(n_segments=50, n_reps=2)
    settings = base.model_copy(
        update={
            "simulation": base.simulation.model_copy(update={"precision": asked_for})
        }
    )
    assert settings.simulation.precision != constants.DEFAULT_PRECISION, (
        f"this test needs a configuration naming the width the flag does not "
        f"default to; both are {asked_for}"
    )
    module = script(name)
    monkeypatch.setattr(config, "load_config", lambda *_args, **_kwargs: settings)
    with caplog.at_level("INFO"):
        module.main(arguments(tmp_path))

    assert ran_at(module, tmp_path, caplog) == asked_for, (
        f"{name} was given a configuration naming {asked_for} and ran at "
        f"something else; its --precision default overrode the configuration "
        f"instead of deferring to it"
    )
