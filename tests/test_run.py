"""Turning a configuration into a saved run.

The claim this module has to earn is that chunking changes nothing. Every
replication reads its own children of the random streams, so replication 7 sees
the same draws whether the run was computed fifty at a time or all at once —
which is what lets the benchmark sweep the chunk size and still say it changed
no result.
"""

import pathlib

import numpy as np
import polars as pl
import pytest
from cablesim import config, constants, random_draws, results, run, simulate


def small(**overrides: object) -> config.Config:
    """A configuration small enough to run in a test, in seconds.

    Args:
        **overrides: Top-level configuration sections to replace.

    Returns:
        The validated configuration.
    """
    base = config.load_config(constants.DEFAULT_CONFIG_PATH).model_dump()
    base["simulation"] = {**base["simulation"], "n_reps": 6, "n_years": 4}
    base["population"] = {**base["population"], "n_segments": 200}
    base["policies"] = [{"name": "run_to_failure"}, {"name": "risk_ranked"}]
    base.update(overrides)
    return config.Config.model_validate(base)


def test_a_run_reassembled_from_chunks_equals_one_computed_whole(
    tmp_path: pathlib.Path,
) -> None:
    """The claim behind holding the batch size out of the configuration.

    If this failed, every archived result would depend on how the machine that
    produced it happened to be batched, and the batch size would have to become
    a modelled parameter rather than provenance.
    """
    settings = small()

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
    settings = small()

    directory = run.run(settings, tmp_path, batch_size=2)

    frame = pl.read_parquet(directory / results.RESULTS_NAME)
    assert sorted(frame["replication"].unique().to_list()) == list(range(6))


def test_every_configured_policy_lands_in_one_file(tmp_path: pathlib.Path) -> None:
    """A run is one configuration across every policy, not one policy."""
    settings = small()

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
    directory = run.run(small(), tmp_path, swept={"annual_budget": 1_000.0})

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
    how the run was batched. Deriving the child by index is what avoids that,
    and it is worth pinning at the source.
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
        covered = [index for span in run.chunks(n_reps, batch_size) for index in span]
        assert covered == list(range(n_reps)), (n_reps, batch_size)


def test_a_batch_size_of_zero_is_refused() -> None:
    """It produces no chunks, and so a run with no results and no error."""
    with pytest.raises(ValueError, match="at least 1"):
        run.chunks(10, 0)


def test_the_segment_arrays_are_ordered_by_identifier() -> None:
    """Array position is the identifier the tie-break compares.

    Both implementations must mean the same thing by it, and passing it
    implicitly as position is what stops them tie-breaking on different keys.
    """
    settings = small()
    frame = run.population.generate(settings).sample(fraction=1.0, shuffle=True, seed=1)

    arrays = run.segment_arrays(frame)

    ordered = frame.sort("segment_id")
    assert np.array_equal(arrays["length_ft"], ordered["length_ft"].to_numpy())
    assert np.array_equal(arrays["age0"], ordered["age"].to_numpy())


def test_a_population_missing_a_column_is_refused() -> None:
    """Rather than failing later inside the loop, or not at all."""
    settings = small()
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
    directory = run.run(small(), tmp_path, batch_size=3)

    manifest = results.Manifest.model_validate_json(
        (directory / results.MANIFEST_NAME).read_text()
    )
    assert manifest.implementation == "reference"
    assert manifest.batch_size == 3
    assert manifest.wall_seconds > 0.0


def test_the_reference_is_swappable_for_another_implementation(
    tmp_path: pathlib.Path,
) -> None:
    """A parity test drives both through this path, so it must take either."""
    calls = []

    def counting(**arguments: object) -> simulate.Results:
        calls.append(arguments["policy"])
        return simulate.simulate(**arguments)

    directory = run.run(
        small(), tmp_path, implementation=counting, implementation_name="kernel"
    )

    assert len(calls) == 2, "one call per policy, at one chunk"
    manifest = results.Manifest.model_validate_json(
        (directory / results.MANIFEST_NAME).read_text()
    )
    assert manifest.implementation == "kernel"


def test_the_run_takes_policy_priorities_from_the_policy_stream(
    tmp_path: pathlib.Path,
) -> None:
    """Checked at the wiring, not only at the helper that builds the draws.

    Both arrays are uniforms of the same shape, so passing the wrong stream
    produces a run that completes with plausible numbers. What it costs is the
    random policy: its priorities would be a function of the same segments'
    lifetime draws, and since a larger uniform gives a shorter lifetime it
    would rank by imminence of failure — the very thing it exists to be a
    control against. No parity test would see it, because every implementation
    would read the same wrong array.
    """
    settings = small()
    captured: dict[str, np.ndarray] = {}

    def capturing(**arguments: object) -> simulate.Results:
        captured.update(
            lifetime_uniforms=arguments["lifetime_uniforms"],
            policy_uniforms=arguments["policy_uniforms"],
        )
        return simulate.simulate(**arguments)

    run.run(settings, tmp_path, implementation=capturing, batch_size=6)

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
