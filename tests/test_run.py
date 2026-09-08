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
    kernel,
    metrics,
    population,
    random_draws,
    results,
    run,
    simulate,
)

from tests import helpers


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
    assert run.missing_names(["batched_pandas"], results.IMPLEMENTATIONS) == {
        "batched_pandas"
    }
    # And nothing else: asserting that `RUNNABLE` itself comes back empty would
    # be the invariant again, which breaking stops the suite at collection, and
    # asserting it of the closed set is a set minus itself.
    assert run.missing_names([], results.IMPLEMENTATIONS) == set()


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


AFFORDABLE_REPLICATIONS = 200
"""Replications this file will run a whole simulation for.

Not a property of the model — a ceiling on what a test costs, so that raising
``run.BOUNDED_BATCH_SIZE`` reports rather than hangs. Two tests here derive
their replication count from that constant, which is what makes them follow a
change to the value; without a ceiling they also follow a change that makes the
value absurd, and a suite whose runtime is linear in the constant under review
stops being able to report on it. ``replications_past_the_bound`` applies it.
"""

WORKING_ARRAY_BOUND_MB = 10.0
"""How large one `(batch, segments)` array may be at the shipped population.

The bound the batched loop's batch size exists to impose, stated as something
that can be checked. Measured, that loop carries twelve to twenty times one such
array, so this admits a peak of roughly 60 to 100 MB above the interpreter — at
the shipped 12,000 segments the batch of fifty makes one array 4.8 MB and peaks
at 225 MB, and a batch of 250 makes it 24 MB and peaks at 475. The value is a
ceiling with room under it, not a measured optimum; what it exists to refuse is
a batch size raised until it bounds nothing.
"""


def replications_past_the_bound() -> int:
    """More replications than the bounded batch, if a test can afford that many.

    Two tests need a run the bounded size would cut in two, and both derive the
    count from the constant so that they follow a change to it rather than
    breaking on one. Following it that faithfully also follows it upwards, and
    the cost of both tests is linear in the count — so a constant raised far
    enough stops reddening them and starts hanging the suite instead.

    Returns:
        Two more replications than the bounded batch size.

    Raises:
        AssertionError: If that is more than this file will run, which says the
            constant needs looking at rather than that either caller is wrong.
    """
    n_reps = run.BOUNDED_BATCH_SIZE + 2
    assert n_reps <= AFFORDABLE_REPLICATIONS, (
        f"BOUNDED_BATCH_SIZE is {run.BOUNDED_BATCH_SIZE}, and these tests run "
        f"whole simulations of two more replications than that; "
        f"test_the_bounded_batch_is_small_enough_to_bound_a_working_set is "
        f"where the constant itself is pinned"
    )
    return n_reps


def test_the_bounded_batch_is_small_enough_to_bound_a_working_set() -> None:
    """The bounded batch size still bounds something at the shipped population.

    ``BOUNDED_BATCH_SIZE`` is documented as untuned, and nothing here claims
    fifty is optimal. What is claimed is narrower and is the reason the constant
    exists: an implementation holding every replication of a chunk in flight has
    its memory bounded by the chunk, and a chunk large enough that one
    `(batch, segments)` array stops being small has given that up.

    Checked against the shipped population rather than a test one, because the
    memory this bounds is the memory of a real run.
    """
    shipped = config.load_config(constants.DEFAULT_CONFIG_PATH)
    one_array_mb = (
        run.BOUNDED_BATCH_SIZE * shipped.population.n_segments * 8 / 1e6
    )

    assert one_array_mb < WORKING_ARRAY_BOUND_MB, (
        f"a batch of {run.BOUNDED_BATCH_SIZE} makes one (batch, segments) "
        f"array {one_array_mb:.1f} MB at {shipped.population.n_segments} "
        f"segments, and the batched loop carries twelve to twenty of them"
    )


def test_each_implementation_is_batched_at_its_own_size_and_records_it(
    tmp_path: pathlib.Path,
) -> None:
    """A run that names no batch size gets the one its implementation wants.

    One number cannot serve all three. The batched loop holds every replication
    of a chunk in flight, so a chunk is what bounds its memory — measured, it
    carries twelve to twenty times one `(replications, segments)` array, which
    at a thousand replications over a hundred thousand segments is ten
    gigabytes. The reference and the kernel hold nothing shaped that way, and
    chunking costs the kernel the axis it parallelises over.

    **Asserted on the calls rather than on the manifest**, because the manifest
    records what was resolved and the chunking is done separately from it, so a
    run can record one size and perform another. The manifest is checked too,
    since it is what a saved run is read by later, but it is the weaker of the
    two claims.

    **The replication count has to exceed the bounded size or nothing here can
    differ**: at six replications every implementation makes one call of six
    whatever it resolved to, and the whole distinction this test exists for is
    invisible. It is taken from the constant rather than written as a number so
    that the two cannot drift apart.
    """
    n_reps = replications_past_the_bound()
    settings = config.with_overrides(small_config(), {"simulation.n_reps": n_reps})
    policy_count = len(settings.policies)
    resolved = {}
    calls = {}
    for name in sorted(run.RUNNABLE):
        with helpers.recorded_batches(name) as recorded:
            directory = run.run(settings, tmp_path / name, implementation=name)
        manifest = json.loads((directory / results.MANIFEST_NAME).read_text())
        resolved[name] = manifest["batch_size"]
        calls[name] = recorded

    assert calls["batched_numpy"] == [run.BOUNDED_BATCH_SIZE, 2] * policy_count, (
        "the batched loop was not chunked at the size that bounds its memory"
    )
    for unbounded in ("kernel", "reference"):
        assert calls[unbounded] == [n_reps] * policy_count, (
            f"{unbounded} was chunked; it holds nothing that grows with the "
            f"batch, and chunking costs the kernel the axis it spreads over"
        )
        assert resolved[unbounded] == n_reps

    assert resolved["batched_numpy"] == run.BOUNDED_BATCH_SIZE
    assert len(set(resolved.values())) > 1, (
        "every implementation resolved to the same size, so this default is "
        "the single number the split exists to replace"
    )


def test_watching_the_batch_sizes_does_not_change_what_the_run_records(
    tmp_path: pathlib.Path,
) -> None:
    """The recorder must leave the provenance of the run it watches alone.

    ``run.run`` reads the Rust build profile from the module the annual loop was
    defined in, so a wrapper defined anywhere else makes a kernel run record no
    profile at all — as ``None`` rather than as a refusal, which is the shape
    that goes unnoticed. Every batching assertion that also reads the manifest
    reads one written under that wrapper, and a timing read off such a manifest
    cannot say whether a release build produced it.
    """
    settings = small_config()

    with helpers.recorded_batches("kernel") as recorded:
        directory = run.run(settings, tmp_path, implementation="kernel")

    assert recorded, "no call was recorded, so the run did not go through it"
    manifest = json.loads((directory / results.MANIFEST_NAME).read_text())
    assert manifest["build_profile"] == kernel.BUILD_PROFILE, (
        "recording the batch sizes changed the provenance the run wrote down; "
        "the wrapper has to carry the wrapped loop's module for the build "
        "profile to be found"
    )


def test_the_sweep_script_defaults_to_no_batch_size_of_its_own(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command line defers to the implementation rather than fixing a number.

    The default is the whole point of the flag being optional: naming a size on
    the command line is still honoured, and naming none has to reach ``run.run``
    as ``None`` so that each implementation resolves its own.

    **Where a fixed default would cost something is a full-size sweep, not the
    reduced one the script runs by default.** ``--full`` is 1,000 replications
    per budget level, so a default of fifty would run, produce correct numbers,
    and chunk the kernel twenty times per level. At ``run.REDUCED_REPS`` it
    would change nothing at all, because 40 replications are a single chunk at
    either size.

    **The flag's default and what the sweep does with it are two claims**, and
    parsing the arguments checks only the first. Passing a fixed fifty at the
    call site leaves the default at ``None`` and chunks every sweep anyway, so
    the sweep is run here — into ``tmp_path``, since ``--out`` otherwise
    defaults inside the repository, over a single budget level and at more
    replications than the bound, because at ``run.REDUCED_REPS`` neither size
    would cut the run in two and there would be nothing to see.
    """
    sweep = helpers.script("budget_sweep")
    n_reps = replications_past_the_bound()
    monkeypatch.setattr(run, "REDUCED_REPS", n_reps)
    monkeypatch.setattr(run, "REDUCED_SEGMENTS", 200)
    monkeypatch.setattr(run, "budget_grid", lambda annual, **_: [annual])
    policy_count = len(config.load_config(constants.DEFAULT_CONFIG_PATH).policies)

    assert sweep.parse_arguments(["--out", "."]).batch_size is None, (
        "the sweep names a batch size, so every implementation gets that one "
        "rather than the one it declared"
    )

    with helpers.recorded_batches("kernel") as recorded:
        sweep.main(["--out", str(tmp_path), "--threads", "1"])

    assert recorded == [n_reps] * policy_count, (
        "the sweep chunked the kernel; its flag defaults to deferring, so the "
        "size has to reach the runner as None and resolve to the whole run"
    )


def test_an_explicit_batch_size_still_overrides_the_implementation(
    tmp_path: pathlib.Path,
) -> None:
    """Naming a size is what a memory-constrained caller does, and it is honoured.

    The per-implementation default is what happens when nobody chooses. A
    caller who does — because the machine is small, or because they are
    measuring the effect of the batch itself — has to be able to, and the
    manifest has to say what they chose.

    **Asserted on the calls first.** Resolving the size and doing the chunking
    are separate steps, so a run can record the size it was given and chunk at
    the implementation's own — which makes ``batch_size`` decorative while
    every equivalence test goes on passing, since chunking changes no number.
    The manifest is checked as well, because a caller who names a size needs
    the saved run to say so.
    """
    settings = small_config()

    with helpers.recorded_batches("kernel") as recorded:
        directory = run.run(settings, tmp_path, implementation="kernel", batch_size=2)

    chunks = [2] * (settings.simulation.n_reps // 2) * len(settings.policies)
    assert recorded == chunks, (
        "an explicitly named batch size did not reach the chunking; the run "
        "was cut at some other size while recording the one it was given"
    )
    manifest = json.loads((directory / results.MANIFEST_NAME).read_text())
    assert manifest["batch_size"] == 2, (
        "an explicitly named batch size was replaced by the implementation's"
    )


def test_an_implementation_with_no_declared_batch_size_is_reported() -> None:
    """The check behind the import-time guard, driven with a bad name.

    The same shape as the two registry checks beside it, and for the same
    reason: the invariant it guards cannot be asserted directly, because
    breaking it stops the suite at collection rather than reddening anything.
    """
    assert run.missing_names(["batched_pandas"], run.BATCH_SIZES) == {
        "batched_pandas"
    }
    assert run.missing_names([], run.BATCH_SIZES) == set()
    assert set(run.BATCH_SIZES) == set(run.RUNNABLE), (
        "every runnable implementation needs a declared batch size, because a "
        "run that names none has to resolve to something"
    )
    # The resolver's own refusal, which nothing reaches through ``run.run``:
    # that path checks the name against ``RUNNABLE`` first, and the guard above
    # is what makes the two sets agree. A direct caller is the only way here,
    # and leaving it unexercised would leave a raise nobody has seen fire.
    with pytest.raises(KeyError, match="batched_pandas"):
        run.batch_size_for("batched_pandas", 10)
    assert run.batch_size_for("kernel", 10) == 10
    assert run.batch_size_for("batched_numpy", 10) == run.BOUNDED_BATCH_SIZE


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
    assert run.missing_names(["batched_pandas"], run.RUNNABLE) == {
        "batched_pandas"
    }
    assert run.missing_names([], run.RUNNABLE) == set()


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
