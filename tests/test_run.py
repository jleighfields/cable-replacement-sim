"""Turning a configuration into a saved run.

The claim this module has to earn is that chunking changes nothing. Every
replication reads its own children of the random streams, so replication 7 sees
the same draws whether the run was computed fifty at a time or all at once —
which is what lets the benchmark sweep the chunk size and still say it changed
no result.
"""

import pathlib
import tempfile

import numpy as np
import polars as pl
import pytest
from cablesim import (
    config,
    constants,
    metrics,
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


def test_the_lifetime_and_policy_streams_stay_independent() -> None:
    """A control correlated with what it controls for has stopped being one.

    The random policy's priorities must not be a function of the same segments'
    lifetime draws: a larger uniform gives a shorter lifetime, so reusing the
    stream would have it ranking segments by imminence of failure, which is the
    thing it exists to be a control against. No parity test would notice, since
    every implementation reads the same arrays.
    """
    sources = random_draws.spawn_sources(20260902)
    replications = range(0, 3)

    lifetimes = random_draws.replication_uniforms(sources.lifetimes, replications, (5,))
    priorities = random_draws.replication_uniforms(sources.policies, replications, (5,))

    assert not np.array_equal(lifetimes, priorities)


def test_a_replication_reads_the_same_draws_at_any_chunk_size() -> None:
    """Directly, rather than only through the results a run happens to produce.

    ``SeedSequence.spawn`` counts the children it has handed out, so a run
    calling it once per chunk gives replication 7 different draws depending on
    how the run was batched. Deriving the child by index is what avoids that.
    """
    sources = random_draws.spawn_sources(20260902)
    shape = (4, 3)

    whole = random_draws.replication_uniforms(sources.lifetimes, range(0, 6), shape)
    later = random_draws.replication_uniforms(sources.lifetimes, range(4, 6), shape)
    again = random_draws.replication_uniforms(sources.lifetimes, range(0, 6), shape)

    assert np.array_equal(whole[4:], later)
    assert np.array_equal(whole, again), "asking twice must give the same answer"


def test_a_derived_child_matches_what_spawn_would_have_given() -> None:
    """The stateless derivation is the same stream, not merely a valid one."""
    source = random_draws.spawn_sources(20260902).lifetimes
    spawned = source.spawn(4)

    for index in range(4):
        assert np.array_equal(
            random_draws.uniforms(spawned[index], 8),
            random_draws.uniforms(random_draws.child_of(source, index), 8),
        ), index


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

    arrays = run.segment_arrays(frame)

    ordered = frame.sort("segment_id")
    assert np.array_equal(arrays["length_ft"], ordered["length_ft"].to_numpy())
    assert np.array_equal(arrays["age0"], ordered["age"].to_numpy())


def test_a_population_missing_a_column_is_refused() -> None:
    """Rather than failing later inside the loop, or not at all."""
    settings = small_config()
    frame = run.population.generate(settings).drop("customers")

    with pytest.raises(KeyError, match="customers"):
        run.segment_arrays(frame)


def test_escalation_starts_at_one_and_compounds() -> None:
    """Year 0 is the year prices are quoted in, so it is not escalated."""
    series = run.escalation_series(0.03, 4)

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
    with pytest.raises(KeyError, match="are the ones that exist"):
        run.run(small_config(), tmp_path, implementation="batched_numpy")


def test_a_name_no_result_may_claim_is_reported_as_unknown() -> None:
    """The check behind the import-time guard, driven with a bad name.

    Asserting the invariant directly — that what is runnable is a subset of
    what a result may claim — would be a test that cannot fail: breaking it
    raises at import and stops the suite at collection, so the assertion never
    runs. Calling the check is what can report.
    """
    assert run.unknown_implementations(["batched_pandas"]) == {"batched_pandas"}
    assert run.unknown_implementations(run.RUNNABLE) == set()
    assert run.unknown_implementations(results.IMPLEMENTATIONS) == set()


def test_the_run_takes_policy_priorities_from_the_policy_stream(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checked at the wiring, not only at the helper that builds the draws.

    Both arrays are uniforms of the same shape, so passing the wrong stream
    produces a run that completes with plausible numbers. What it costs is the
    random policy: a larger uniform gives a shorter lifetime, so priorities
    taken from the lifetime stream would rank by imminence of failure — the
    thing the policy exists to be a control against. No parity test would see
    it, because every implementation would read the same wrong array.
    """
    settings = small_config()
    captured: dict[str, np.ndarray] = {}

    def capturing(**arguments: object) -> simulate.Results:
        captured.update(
            lifetime_uniforms=arguments["lifetime_uniforms"],
            policy_uniforms=arguments["policy_uniforms"],
        )
        return simulate.run_chunk(**arguments)

    monkeypatch.setitem(run.RUNNABLE, "reference", capturing)
    run.run(settings, tmp_path, batch_size=6)

    sources = random_draws.spawn_sources(settings.simulation.seed)
    n_segments = settings.population.n_segments
    assert np.array_equal(
        captured["policy_uniforms"],
        random_draws.replication_uniforms(
            sources.policies, range(0, settings.simulation.n_reps), (n_segments,)
        ),
    )
    assert not np.array_equal(
        captured["policy_uniforms"], captured["lifetime_uniforms"][:, :, 0]
    )


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
