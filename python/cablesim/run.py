"""Turning a configuration into a saved run.

This is the only module that writes anything. Everything else in the reporting
layer is a pure function of a frame; the loop that produces the frame is not,
because it owns the chunking, the ordering and the concatenation.

A **run** is one configuration evaluated for every policy in it, producing one
directory. The implementation is an argument, so the reference and the compute
kernel are interchangeable here — which is what lets a parity test drive both
through one path rather than through two that could differ in how they are
driven.

Replications are processed in chunks because the result arrays grow with the
replication count: seven of them at `(replications, years, classes)`, plus
whatever the implementation holds in flight, which for the batched loops is a
`(replications, segments)` working set. The chunk size changes no number, since
every draw is a function of the key and of a position that carries the
replication's index in the whole run, so it is an argument here and provenance
in the manifest rather than a configured parameter.
"""

import datetime
import logging
import pathlib
import sys
import tempfile
import time
from collections.abc import Callable, Iterable

import numpy as np
import polars as pl

from cablesim import (
    batched,
    constants,
    kernel,
    policies,
    population,
    random_draws,
    results,
    simulate,
)
from cablesim import config as config_module

log = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 50
"""Replications per call, which trades memory against time and nothing else."""

BUDGET_GRID_LOW = 0.125
"""The lowest swept budget, as a fraction of the configured one."""

BUDGET_GRID_HIGH = 2.0
"""The highest swept budget, as a multiple of the configured one."""

BUDGET_GRID_LEVELS = 7
"""How many non-zero levels the sweep places between those two bounds.

Spaced geometrically rather than linearly, so the region near a binding
constraint is sampled more densely than the flat region beyond it. All three
live in the package rather than in a driver script: a notebook and a script
that each spell the grid out are two designs that drift apart.
"""

SEGMENT_COLUMNS: tuple[str, ...] = (
    "length_ft",
    "customers",
    "customer_minutes_per_failure",
    "customer_minutes_per_planned",
    "outage_cost_per_failure",
    "class_index",
    "age",
    "shape",
    "scale",
    "replacement_shape",
    "replacement_scale",
    "cost_per_ft",
)
"""Population columns the annual loop reads, in the order it names them.

Everything that reduces to a per-segment number is reduced before this point —
conductor count and length into the effective scale, customer types into the
four derived columns — so what crosses into an implementation is arrays it
reads without interpreting.
"""


Implementation = Callable[..., simulate.Results]
"""An annual loop: the reference, or the compute kernel, called by keyword.

Deliberately not a Protocol spelling out the call. One that wrote
``__call__(**arguments: object)`` would accept any callable at all, which is
what a bare alias already says with less ceremony; one that pinned all
twenty-odd argument names would be the contract worth having, and that contract
is already written as ``simulate.run_chunk``'s signature — the argument names
the kernel's binding has to mirror, since every call here is by keyword.
"""


RUNNABLE: dict[str, Implementation] = {
    "reference": simulate.run_chunk,
    "batched_numpy": batched.run_chunk_numpy,
    "kernel": kernel.run_chunk,
}
"""The annual loops that exist, by the name a manifest records them under.

Named for what it holds rather than for the set it draws from: ``results``
carries the closed set of names a saved run may claim, and this is the subset
with something behind it.

The keys must all be names ``results`` accepts, which is checked below rather
than left to the manifest validator. Left there, a misspelling would be taken
on the command line, accepted through a whole budget level's computation, and
refused only at the write.
"""


CONCURRENT: frozenset[str] = frozenset({"batched_numpy", "kernel"})
"""The implementations that spread a chunk's replications over workers.

Declared rather than discovered, because the alternative is calling each one
with two threads and seeing which raises — a test of the refusal rather than of
the capability, which would quietly pass an implementation that accepted the
argument and ignored it.

The reference is the only one absent: it runs a replication at a time by
construction, being the version written to be checkable by reading. Every name
here must be one ``RUNNABLE`` holds, which
``unthreadable_implementations`` checks.
"""


def unthreadable_implementations(names: Iterable[str]) -> set[str]:
    """Names among these that no runnable implementation answers to.

    A function rather than a check run once at import, for the reason
    ``unknown_implementations`` below is one: an import-time check cannot be
    watched failing, because breaking it stops the whole suite at collection
    instead of reddening a test.

    Args:
        names: Implementation names claiming to spread replications.

    Returns:
        Those that name nothing in ``RUNNABLE``.
    """
    return set(names) - set(RUNNABLE)


UNTHREADABLE = unthreadable_implementations(CONCURRENT)
if UNTHREADABLE:
    raise ValueError(
        f"{sorted(UNTHREADABLE)} claim to spread replications but name no "
        f"implementation; the runnable set is {sorted(RUNNABLE)}"
    )


def unknown_implementations(names: Iterable[str]) -> set[str]:
    """Names among these that no saved result may claim.

    A function rather than an expression evaluated once at import, because an
    import-time check cannot be watched failing: breaking it stops the whole
    suite at collection rather than reddening a test. This can be called with
    a bad name and asserted on.

    Args:
        names: Implementation names to check.

    Returns:
        Those that are not in the closed set a manifest accepts.
    """
    return set(names) - set(results.IMPLEMENTATIONS)


UNRUNNABLE = unknown_implementations(RUNNABLE)
if UNRUNNABLE:
    raise ValueError(
        f"{sorted(UNRUNNABLE)} name no implementation a result may claim; "
        f"the closed set is {list(results.IMPLEMENTATIONS)}"
    )


REDUCED_SEGMENTS = 2_000
"""Population size for a run meant to finish while someone watches it.

Here rather than in a driver script because a notebook and a script that each
name their own reduced size are two sizes that drift apart, and a figure is
only readable against the size it was drawn at. Use it through
``config.resize_population``, which scales the customer count and the budget
with it.
"""

REDUCED_REPS = 40
"""Replication count to match ``REDUCED_SEGMENTS``."""


def budget_grid(annual: float, levels: int = BUDGET_GRID_LEVELS) -> list[float]:
    """Builds the swept budget levels for the deliverable figure.

    **Zero is in the grid deliberately.** No policy funds anything there, so
    every one of them must land on exactly the same point as run-to-failure —
    an end-to-end check on the population, the draws, the scoring, the fill,
    the loop and the reduction, for the cost of a point that was worth plotting
    anyway.

    Args:
        annual: The configured annual budget, which the grid brackets.
        levels: How many non-zero levels to place.

    Returns:
        Zero followed by geometrically spaced levels.
    """
    spaced = np.geomspace(annual * BUDGET_GRID_LOW, annual * BUDGET_GRID_HIGH, levels)
    return [0.0, *spaced.tolist()]


def escalation_series(rate: float, n_years: int) -> np.ndarray:
    """Compounds an annual rate into a per-year multiplier.

    Args:
        rate: Annual growth, as a fraction.
        n_years: Horizon.

    Returns:
        One multiplier per year, starting at 1.0 in year 0.
    """
    return (1.0 + rate) ** np.arange(n_years, dtype=float)


def segment_arrays(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Extracts the per-segment arrays an implementation reads.

    A segment's position in these arrays is its identifier, and the tie-break
    that decides which of two equally ranked candidates is funded reads that
    position, so the frame is sorted before anything is taken from it.

    Args:
        frame: The population, one row per segment.

    Returns:
        The arrays, keyed by the argument name each is passed as.

    Raises:
        KeyError: If the population is missing a column the loop reads.
    """
    ordered = frame.sort("segment_id")
    missing = [name for name in SEGMENT_COLUMNS if name not in ordered.columns]
    if missing:
        raise KeyError(f"the population has no {missing}; it cannot be simulated")
    arrays = {name: ordered[name].to_numpy() for name in SEGMENT_COLUMNS}
    # The loop names the starting age `age0`, because it holds a current age
    # that moves; the population column is the age at year 0.
    arrays["age0"] = arrays.pop("age").astype(float)
    arrays["class_index"] = arrays["class_index"].astype(np.uint8)
    return arrays


def replication_chunks(n_reps: int, batch_size: int) -> list[range]:
    """Splits the replication axis into chunks.

    Args:
        n_reps: Replications in the run.
        batch_size: Replications per chunk.

    Returns:
        One range per chunk, covering every replication exactly once.

    Raises:
        ValueError: If the batch size is not positive, which would otherwise
            produce no chunks and a run with no results.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    return [
        range(start, min(start + batch_size, n_reps))
        for start in range(0, n_reps, batch_size)
    ]


def simulate_policy(
    spec: config_module.PolicySpec,
    settings: config_module.Config,
    segments: dict[str, np.ndarray],
    class_names: list[str],
    implementation: Implementation,
    batch_size: int,
    threads: int,
    parts: pathlib.Path,
) -> list[pathlib.Path]:
    """Runs every replication under one policy, chunk by chunk.

    Args:
        spec: The policy to run.
        settings: The effective configuration.
        segments: The per-segment arrays.
        class_names: Segment class names in class-index order.
        implementation: The annual loop to call.
        batch_size: Replications per call.
        threads: Workers to spread each chunk's replications over. Passed to
            every implementation rather than only to the ones that can use it,
            so an implementation that cannot refuses the request instead of
            leaving the caller to believe it was honoured.
        parts: Where to write each chunk's rows.

    Returns:
        One file per chunk, in the order the chunks were run.

        **Written out rather than accumulated**, so that what a run holds at
        once is one chunk's rows rather than every chunk of every policy. Rows
        scale with the replication count — forty megabytes at a thousand
        replications and four gigabytes at a hundred thousand — and that count
        is exactly the knob someone turns to narrow a confidence interval. The
        files are concatenated lazily at the end, so the whole is never
        materialized either.
    """
    simulation = settings.simulation
    # One key for the whole run, from which every draw is computed by position.
    # A chunk needs no state of its own, which is what makes the chunking
    # provenance rather than a parameter: replication 7 meets the same draws
    # whichever chunk it landed in.
    key = random_draws.draw_key(simulation.seed)
    resolved = policies.resolve(spec)
    budget = settings.budget.annual * escalation_series(
        settings.budget.escalation, simulation.n_years
    )
    cost_escalation = escalation_series(
        settings.costs.escalation_rate, simulation.n_years
    )

    written = []
    for index, replications in enumerate(
        replication_chunks(simulation.n_reps, batch_size)
    ):
        block = implementation(
            **segments,
            draw_key=key,
            first_replication=replications.start,
            n_reps=len(replications),
            budget=budget,
            cost_escalation=cost_escalation,
            policy=resolved,
            emergency_multiplier=settings.costs.emergency_multiplier,
            mobilization_per_segment=settings.costs.mobilization_per_segment,
            emergency_charged_to_budget=settings.budget.emergency_charged_to_budget,
            n_classes=len(class_names),
            n_years=simulation.n_years,
            threads=threads,
        )
        part = parts / f"{spec.name}-{index:04d}.parquet"
        results.rows_from_chunk(
            block, spec.name, class_names, replications.start
        ).write_parquet(part)
        written.append(part)
    return written


def run(
    settings: config_module.Config,
    root: pathlib.Path,
    implementation: str = "reference",
    batch_size: int = DEFAULT_BATCH_SIZE,
    threads: int = 1,
    swept: dict[str, float] | None = None,
) -> pathlib.Path:
    """Runs one configuration for every policy and writes the result.

    Args:
        settings: The effective configuration, already validated.
        root: Where run directories are written.
        implementation: Which annual loop to run, keyed into ``RUNNABLE``.
            A name rather than the callable, because the manifest records this
            as provenance: passing the two separately let a caller run one
            implementation and record another, and a manifest that can be
            wrong is worse than no manifest. The Rust build profile follows
            from the same name — read from the compiled extension where the
            kernel ran, and absent for pure Python rather than invented, since
            a timing from the kernel without one means nothing.
        batch_size: Replications per call.
        threads: Workers to spread each chunk's replications over, recorded in
            the manifest beside the implementation and the build profile. The
            implementations named in ``CONCURRENT`` can use more than one; the
            scalar reference refuses, so a manifest cannot claim a thread count
            that nothing acted on. ``kernel.AVAILABLE_THREADS`` is this
            machine's count.
        swept: Values that vary between the runs of a sweep, written into the
            saved rows as columns. A frame carrying its own parameters is
            readable without the directory layout that produced it.

    Returns:
        The directory written.

    Raises:
        KeyError: If no implementation goes by that name.
    """
    if implementation not in RUNNABLE:
        raise KeyError(
            f"no implementation named {implementation!r}; "
            f"{sorted(RUNNABLE)} are the ones that exist"
        )
    annual_loop = RUNNABLE[implementation]
    # The profile belongs to whichever module the loop came from: a pure-Python
    # one defines none, and the kernel's wrapper reads it from the compiled
    # extension. Asking the module rather than testing the name for "kernel"
    # keeps that name out of a second place, so a later Rust-backed
    # implementation records its profile instead of silently recording none.
    build_profile = getattr(sys.modules[annual_loop.__module__], "BUILD_PROFILE", None)
    started = time.perf_counter()
    run_id = results.new_run_id()
    segments_frame = population.generate(settings)
    class_names = [segment_class.name for segment_class in settings.population.classes]
    segments = segment_arrays(segments_frame)
    log.info(
        "run %s: %d segments, %d policies, %d replications, %d thread(s)",
        run_id,
        segments["age0"].size,
        len(settings.policies),
        settings.simulation.n_reps,
        threads,
    )

    # Each chunk's rows go to a file of their own and are concatenated lazily at
    # the end, so neither a policy's rows nor the whole run is ever held at
    # once. The scratch directory is removed however this exits, including on a
    # failure part-way through a sweep, so a crashed run leaves no half-written
    # parts to be mistaken for a result.
    with tempfile.TemporaryDirectory(prefix=f"cablesim-{run_id}-") as scratch:
        parts = pathlib.Path(scratch)
        written: list[pathlib.Path] = []
        for spec in settings.policies:
            written.extend(
                simulate_policy(
                    spec,
                    settings,
                    segments,
                    class_names,
                    annual_loop,
                    batch_size,
                    threads,
                    parts,
                )
            )
        # Named in order rather than scanned by pattern: the rows are written in
        # policy order and then chunk order, and a directory listing would put
        # them in whatever order the filesystem returns.
        frame = pl.scan_parquet(written)
        for name, value in (swept or {}).items():
            frame = frame.with_columns(pl.lit(value).alias(name))

        commit, dirty = results.git_provenance(constants.PROJECT_ROOT)
        return results.write_run(
            root,
            frame,
            settings,
            results.Manifest(
                run_id=run_id,
                written_at=datetime.datetime.now(datetime.UTC),
                package_version=results.package_version(),
                git_commit=commit,
                git_dirty=dirty,
                implementation=implementation,
                build_profile=build_profile,
                threads=threads,
                batch_size=batch_size,
                wall_seconds=time.perf_counter() - started,
            ),
        )
