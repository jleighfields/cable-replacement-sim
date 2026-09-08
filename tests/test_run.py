"""Turning a configuration into a saved run.

The claim this module has to earn is that chunking changes nothing. Every
replication reads its own children of the random streams, so replication 7 sees
the same draws whether the run was computed fifty at a time or all at once —
which is what lets the benchmark sweep the chunk size and still say it changed
no result.
"""

import json
import pathlib
import tempfile

import numpy as np
import polars as pl
import pytest
from cablesim import (
    config,
    constants,
    metrics,
    population,
    random_draws,
    results,
    run,
    simulate,
)


def small_config() -> config.Config:
    """A configuration small enough to run in a test, in seconds.

    Takes no overrides: the one case that needs one wraps this in
    ``config.with_overrides``, which keeps the dotted-path spelling readable — a
    dotted name is not a valid keyword argument, so an override parameter here
    could only be reached by unpacking a dictionary.

    Returns:
        The validated configuration.
    """
    settings = config.resize_population(
        config.load_config(constants.DEFAULT_CONFIG_PATH), 200, n_reps=6
    )
    return config.with_overrides(
        settings,
        {
            "simulation.n_years": 4,
            "policies": [{"name": "run_to_failure"}, {"name": "risk_ranked"}],
        },
    )


def test_a_run_reassembled_from_chunks_equals_one_computed_whole(
    tmp_path: pathlib.Path,
) -> None:
    """The claim behind holding the batch size out of the configuration.

    If this failed, every archived result would depend on how the machine that
    produced it happened to be batched, and the batch size would have to become
    a modelled parameter rather than provenance.
    """
    settings = small_config()

    whole = run.run(settings, tmp_path / "whole", batch_size=settings.simulation.n_reps)
    chunked = run.run(settings, tmp_path / "chunked", batch_size=2)
    odd = run.run(settings, tmp_path / "odd", batch_size=4)

    order = ["policy", "replication", "year", "class"]
    frames = [
        pl.read_parquet(directory / results.RESULTS_NAME).sort(order)
        for directory in (whole, chunked, odd)
    ]
    assert frames[0].equals(frames[1])
    assert frames[0].equals(frames[2]), "a batch size that does not divide evenly"


def test_the_replication_axis_is_continuous_across_chunks(
    tmp_path: pathlib.Path,
) -> None:
    """Chunks are labelled with where they sat in the run, not from zero."""
    settings = small_config()

    directory = run.run(settings, tmp_path, batch_size=2)

    frame = pl.read_parquet(directory / results.RESULTS_NAME)
    assert sorted(frame["replication"].unique().to_list()) == list(range(6))


def test_every_configured_policy_lands_in_one_file(tmp_path: pathlib.Path) -> None:
    """A run is one configuration across every policy, not one policy."""
    settings = small_config()

    directory = run.run(settings, tmp_path)

    frame = pl.read_parquet(directory / results.RESULTS_NAME)
    assert sorted(frame["policy"].unique().to_list()) == [
        "risk_ranked",
        "run_to_failure",
    ]


def test_swept_values_are_written_into_the_rows(tmp_path: pathlib.Path) -> None:
    """A frame carrying its own parameters is readable without its directory.

    A renamed directory, or a run copied elsewhere, still answers questions.
    """
    directory = run.run(small_config(), tmp_path, swept={"annual_budget": 1_000.0})

    frame = pl.read_parquet(directory / results.RESULTS_NAME)
    assert frame["annual_budget"].unique().to_list() == [1_000.0]


def test_chunks_cover_every_replication_exactly_once() -> None:
    """Including when the batch size does not divide the replication count."""
    for n_reps, batch_size in ((6, 2), (6, 4), (6, 6), (6, 10), (1, 50)):
        spans = run.replication_chunks(n_reps, batch_size)
        covered = [index for span in spans for index in span]
        assert covered == list(range(n_reps)), (n_reps, batch_size)


def test_a_batch_size_of_zero_is_refused() -> None:
    """It produces no chunks, and so a run with no results and no error."""
    with pytest.raises(ValueError, match="at least 1"):
        run.replication_chunks(10, 0)


def test_the_segment_arrays_are_ordered_by_identifier() -> None:
    """Array position is the identifier the tie-break compares.

    Both implementations must mean the same thing by it, and passing it
    implicitly as position is what stops them tie-breaking on different keys.
    """
    settings = small_config()
    frame = run.population.generate(settings).sample(fraction=1.0, shuffle=True, seed=1)

    arrays = run.segment_arrays(frame, constants.DEFAULT_PRECISION)

    ordered = frame.sort("segment_id")
    assert np.array_equal(arrays["length_ft"], ordered["length_ft"].to_numpy())
    assert np.array_equal(arrays["age0"], ordered["age"].to_numpy())


def test_a_population_missing_a_column_is_refused() -> None:
    """Rather than failing later inside the loop, or not at all."""
    settings = small_config()
    frame = run.population.generate(settings).drop("customers")

    with pytest.raises(KeyError, match="customers"):
        run.segment_arrays(frame, constants.DEFAULT_PRECISION)


def test_escalation_starts_at_one_and_compounds() -> None:
    """Year 0 is the year prices are quoted in, so it is not escalated."""
    series = run.escalation_series(0.03, 4, constants.DEFAULT_PRECISION)

    assert series[0] == 1.0
    assert series.tolist() == pytest.approx([1.0, 1.03, 1.03**2, 1.03**3])


def test_the_run_directory_records_how_it_was_produced(
    tmp_path: pathlib.Path,
) -> None:
    """Including the batch size, which is provenance rather than a parameter."""
    directory = run.run(small_config(), tmp_path, batch_size=3)

    manifest = results.Manifest.model_validate_json(
        (directory / results.MANIFEST_NAME).read_text()
    )
    assert manifest.implementation == "reference"
    assert manifest.batch_size == 3
    assert manifest.wall_seconds > 0.0


def test_the_named_implementation_is_the_one_that_runs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loop that ran and the name recorded cannot come apart.

    They were once two arguments, so a caller could run one implementation and
    record another, and a manifest that can be wrong is worse than no manifest.
    The name now selects the loop, and this drives that end to end: a counting
    stand-in registered under one name is what runs, and that name is what the
    saved manifest says.
    """
    calls = []

    def counting(**arguments: object) -> simulate.Results:
        calls.append(arguments["policy"])
        return simulate.run_chunk(**arguments)

    monkeypatch.setitem(run.RUNNABLE, "kernel", counting)

    directory = run.run(small_config(), tmp_path, implementation="kernel")

    assert len(calls) == 2, "one call per policy, at one chunk"
    manifest = results.Manifest.model_validate_json(
        (directory / results.MANIFEST_NAME).read_text()
    )
    assert manifest.implementation == "kernel"


def test_an_implementation_that_does_not_exist_is_refused_before_any_work(
    tmp_path: pathlib.Path,
) -> None:
    """A misspelled name fails at the call, not after the first run computed.

    The manifest validator would refuse it too, but only at the write, which on
    a sweep is one budget level's computation later.
    """
    # Matched on the guard's own wording, not on the name. The lookup a line
    # below raises `KeyError` carrying the same name, so a test matching only
    # that cannot tell the guard from its absence — and the guard exists for
    # the message, which names what would have worked.
    #
    # A misspelling rather than a name that exists but has nothing behind it:
    # every implementation a saved result may claim is now runnable, so that
    # second case has no example left to make.
    with pytest.raises(KeyError, match="are the ones that exist"):
        run.run(small_config(), tmp_path, implementation="kernal")


def test_a_name_no_result_may_claim_is_reported_as_unknown() -> None:
    """The check behind the import-time guard, driven with a bad name.

    Asserting the invariant directly — that what is runnable is a subset of
    what a result may claim — would be a test that cannot fail: breaking it
    raises at import and stops the suite at collection, so the assertion never
    runs. Calling the check is what can report.
    """
    assert run.unknown_implementations(["batched_pandas"]) == {"batched_pandas"}
    # And nothing else: asserting that `RUNNABLE` itself comes back empty would
    # be the invariant again, which breaking stops the suite at collection, and
    # asserting it of the closed set is a set minus itself.
    assert run.unknown_implementations([]) == set()


def test_the_escalation_series_compounds_in_double_and_narrows_once() -> None:
    """Compounding at the working width is not the same as narrowing after it.

    Raising `(1 + rate)` to each year at single precision accumulates that
    width's rounding once per year; computing the whole series in double and
    narrowing at the end does not. At the shipped rate over a thirty-year
    horizon the two disagree in 26 of 30 multipliers, first at year four, and
    every dollar in a single-precision run is scaled by one of them.

    Nothing else could see this. Every implementation reads the same series, so
    they all move together and go on agreeing with each other; the two width
    checks assert the dtype the series arrives at, not how it was computed.
    """
    single = run.escalation_series(0.03, 30, "f32")
    double = run.escalation_series(0.03, 30, constants.DEFAULT_PRECISION)

    assert single.dtype == np.float32
    assert np.array_equal(single, double.astype(np.float32)), (
        "the single-precision series is not the double one narrowed, so it "
        "compounded at the narrower width and carries thirty years of that "
        "width's rounding rather than one narrowing at the end"
    )


def test_a_precision_that_names_no_dtype_is_refused() -> None:
    """The two array builders refuse a width that does not exist.

    Reachable because ``precision`` is a required argument on both — it was
    defaulted until the same omission produced a double-precision run twice,
    and the configuration model's ``Literal`` only guards the callers that go
    through it. A direct caller is the way here, and every parity fixture and
    driver script is one.
    """
    frame = population.generate(small_config())
    for call in (
        lambda: run.segment_arrays(frame, "f16"),
        lambda: run.escalation_series(0.03, 4, "f16"),
    ):
        with pytest.raises(ValueError, match="names no dtype"):
            call()


def test_the_run_path_computes_at_the_configured_precision(
    tmp_path: pathlib.Path,
) -> None:
    """``run.run`` honours ``simulation.precision``, and its output shows it.

    The benchmark harness has its own check that every array it builds carries
    the configured width. This is the other entry point — the one that writes
    saved runs — and it builds its arguments separately, so nothing the harness
    asserts says anything about it. Dropping the precision from either of the
    two builders here left the whole suite green.

    Asserted on the result rather than on the dtypes, because results are
    written in double at both precisions so that a saved run has one schema.
    What differs is the arithmetic that produced them: at single precision the
    dollar columns move in the last bits while the counts do not, so comparing
    the two runs shows the width reached the loop without depending on how the
    rows are stored.
    """
    saved = {}
    for precision in ("f64", "f32"):
        settings = config.with_overrides(
            small_config(), {"simulation.precision": precision}
        )
        directory = run.run(settings, tmp_path / precision)
        saved[precision] = pl.read_parquet(directory / results.RESULTS_NAME).sort(
            ["policy", "replication", "year", "class"]
        )
        # The effective configuration is written beside the rows, so a reader a
        # year later can tell which width produced them.
        assert f"precision: {precision}" in (
            directory / results.CONFIG_NAME
        ).read_text()

    assert saved["f64"]["failures"].to_list() == saved["f32"]["failures"].to_list(), (
        "the counts should not move between widths on this configuration; if "
        "they do, this test is measuring something other than rounding"
    )
    assert saved["f64"]["planned_spend"].to_list() != (
        saved["f32"]["planned_spend"].to_list()
    ), (
        "the two widths produced identical spend, so the run path computed at "
        "one of them twice and simulation.precision reached nothing"
    )


def test_a_threading_claim_that_names_no_implementation_is_reported() -> None:
    """The check behind the threading registry, driven with a bad name.

    The same shape as the unknown-name check above and for the same reason: the
    invariant it guards cannot be asserted directly, because breaking it stops
    the suite at collection rather than reddening anything.
    """
    assert run.unthreadable_implementations(["batched_pandas"]) == {"batched_pandas"}
    assert run.unthreadable_implementations([]) == set()


def test_the_threading_registry_names_the_kernel_and_something_to_compare() -> None:
    """``CONCURRENT`` is not empty, and holds the kernel.

    The parity assertion that every thread count gives the reference answer
    loops over this set, so emptying it would leave that test collected, green,
    and checking nothing. Naming the kernel specifically is what stops the set
    shrinking to only the implementation whose threading is a negative result.
    """
    assert "kernel" in run.CONCURRENT
    # Both, not just the kernel. Narrowing this set to the kernel alone leaves
    # both threaded parity assertions green while they quietly stop checking
    # the batched loop, because each loops over the set inside its body rather
    # than parametrising on it, so the test count does not move either.
    assert "batched_numpy" in run.CONCURRENT
    assert set(run.RUNNABLE) > run.CONCURRENT, (
        "the reference cannot spread replications and must stay out of this set"
    )


def test_the_run_derives_the_draw_key_from_the_configured_seed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checked at the wiring, not only at the helper that derives it.

    The key is the whole of a run's randomness now: every uniform is a function
    of it and of a position. A run that derived it from anything but the
    configured seed — a constant, the clock, a chunk index — would still
    complete with plausible numbers, and no parity test would see it, because
    every implementation would compute from the same wrong key.

    What the chunk offset does is checked with it. A run splits replications
    into chunks to bound memory, and that split is provenance rather than part
    of the model, so chunk two has to start at the replication it actually
    covers or the same run at a different batch size would produce different
    numbers.
    """
    settings = small_config()
    seen: list[dict[str, object]] = []

    def capturing(**arguments: object) -> simulate.Results:
        seen.append(
            {
                "draw_key": arguments["draw_key"],
                "first_replication": arguments["first_replication"],
                "n_reps": arguments["n_reps"],
            }
        )
        return simulate.run_chunk(**arguments)

    monkeypatch.setitem(run.RUNNABLE, "reference", capturing)
    run.run(settings, tmp_path, batch_size=6)

    expected = random_draws.draw_key(settings.simulation.seed)
    assert {call["draw_key"] for call in seen} == {expected}
    # A different seed has to give a different key, or deriving it from the seed
    # would be indistinguishable from ignoring the seed.
    assert random_draws.draw_key(settings.simulation.seed + 1) != expected

    # Every replication of the run is covered exactly once, in order.
    covered: list[int] = []
    for call in seen[: len(seen) // len(settings.policies) or 1]:
        covered.extend(
            range(call["first_replication"], call["first_replication"] + call["n_reps"])
        )
    assert covered == list(range(settings.simulation.n_reps))


def test_a_thread_count_reaches_the_annual_loop_and_the_manifest(
    tmp_path: pathlib.Path,
) -> None:
    """Both, because recording one and running another is the failure to catch.

    Only the compute kernel spreads replications over workers, so this runs
    through it. What it pins is the wiring rather than the parallelism: a run
    that passed the count to the manifest and a literal 1 to the loop would
    report a number nothing acted on, and every result would still be correct,
    because the thread count changes no value.
    """
    settings = small_config()
    seen: list[int] = []
    annual_loop = run.RUNNABLE["kernel"]

    def capturing(**arguments: object) -> simulate.Results:
        seen.append(arguments["threads"])
        return annual_loop(**arguments)

    directory = run.run(
        settings,
        tmp_path,
        implementation="kernel",
        threads=2,
        batch_size=6,
    )
    manifest = json.loads((directory / results.MANIFEST_NAME).read_text())
    assert manifest["threads"] == 2

    # And again with the loop watched, so the manifest above is not the only
    # thing the number reached.
    run.RUNNABLE["kernel"] = capturing
    try:
        run.run(settings, tmp_path, implementation="kernel", threads=2, batch_size=6)
    finally:
        run.RUNNABLE["kernel"] = annual_loop
    assert seen and set(seen) == {2}

def test_the_budget_series_uses_the_budget_rate_and_not_the_cost_rate(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two rates that happen to be equal in the shipped file are not one rate.

    Both are 3% in the checked-in configuration and every test builds from it,
    so swapping them changes nothing anywhere. A sweep holding the budget flat
    in nominal terms — a natural thing to sweep — would then silently get 3%
    growth, and the axis of the deliverable figure would not be the axis it
    claims.
    """
    settings = config.with_overrides(
        small_config(), {"budget.escalation": 0.0, "costs.escalation_rate": 0.10}
    )
    captured: dict[str, np.ndarray] = {}

    def capturing(**arguments: object) -> simulate.Results:
        captured.update(
            budget=arguments["budget"], cost_escalation=arguments["cost_escalation"]
        )
        return simulate.run_chunk(**arguments)

    monkeypatch.setitem(run.RUNNABLE, "reference", capturing)
    run.run(settings, tmp_path)

    assert captured["budget"].tolist() == pytest.approx(
        [settings.budget.annual] * settings.simulation.n_years
    ), "a flat budget must stay flat"
    assert captured["cost_escalation"].tolist() == pytest.approx(
        [1.10**year for year in range(settings.simulation.n_years)]
    )


def test_the_swept_budget_grid_starts_at_zero() -> None:
    """The zero point is what makes the left-hand end an end-to-end check."""
    grid = run.budget_grid(1_000.0, levels=3)

    assert grid[0] == 0.0
    assert len(grid) == 4
    assert grid[1:] == pytest.approx([125.0, 500.0, 2_000.0])


def test_a_reliability_index_at_reduced_size_matches_one_at_full_size() -> None:
    """The behavioural check behind resizing, for a policy that spends.

    Both system figures have to scale together, and a policy that never spends
    cannot tell whether the budget did: run-to-failure agrees whatever happens
    to the capital. So this compares a policy whose whole behaviour is what the
    budget buys.

    A statistical comparison at a tolerance this wide is a floor rather than a
    proof — the bias it exists to catch was a factor of five, not a few
    percent.
    """
    full = config.with_overrides(
        config.load_config(constants.DEFAULT_CONFIG_PATH),
        {
            "simulation.n_reps": 8,
            "simulation.n_years": 10,
            "population.n_segments": 6_000,
            "policies": [{"name": "risk_ranked"}],
            "reporting": {"baseline_policy": "risk_ranked"},
        },
    )
    reduced = config.resize_population(full, 1_000)

    def mean_saidi(settings: config.Config) -> float:
        """Mean duration index across replications and years.

        Args:
            settings: The configuration to run.

        Returns:
            The mean index.
        """
        with tempfile.TemporaryDirectory() as root:
            directory = run.run(settings, pathlib.Path(root), batch_size=8)
            frame = pl.read_parquet(directory / results.RESULTS_NAME)
        per = metrics.indices_per_replication(
            frame.lazy(), settings.population.total_customers
        ).collect()
        return per["saidi"].mean()

    assert mean_saidi(reduced) == pytest.approx(mean_saidi(full), rel=0.35)
