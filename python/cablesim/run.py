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
replication count: one per field of `simulate.Results`, each at
`(replications, years, classes)`, plus whatever the implementation holds in
flight, which for the batched loops is a `(replications, segments)` working
set. The chunk size changes no number, since
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

BOUNDED_BATCH_SIZE = 50
"""Replications per call for an implementation that holds them all at once.

The batched NumPy loop carries `(replications, segments)` arrays and enough of
them that its footprint is twelve to twenty times one such array: measured at
12,000 segments it grows from 99 MB above import at this batch to 350 MB at 250
and 1,250 MB at 1,000, where a single array accounts for 4.8 MB, 24 and 96. That
is what a batch bounds, and it is why this implementation has one at all. The
figures come from ``scripts/measure_memory.py --implementation batched_numpy
--segments 12000 --reps <batch>``, which runs one implementation per process
because a peak belongs to the process.

The value is not a tuned optimum. It is small enough that the memory stays a
detail at the sizes this project runs and large enough that per-call overhead is
not the cost, and no measurement asks it to be more precise than that.
"""

UNBOUNDED_BATCH: int | None = None
"""What an implementation that holds no such arrays is batched at: nothing.

The reference runs one replication at a time and the kernel gives each worker
scratch sized by *segments*, so neither holds a working set that grows with the
batch. What does grow is the results and the rows built from them, and chunking
those turns out to save nothing: measured, the chunked run peaks *higher* — 431
MB against 386 at 12,000 segments, 652 against 626 at 50,000 — because twenty
chunks hold twenty parquet parts where one holds one. So chunking these two buys
no memory and costs the axis the kernel parallelises over: fifty replications
across forty-eight workers is one each, repeated, with a pool built per chunk.

Measured through `run.run` on the kernel at 1,000 replications over five
policies, release build, 48 workers: 5.54 s chunked at fifty against 3.34 s
whole at 12,000 segments, over six runs each, and 24.85 s against 14.90 s at
50,000, over twelve. So **chunking costs the kernel 1.66x at 12,000 segments
and 1.67x at 50,000** — the same cost at both.

**Each figure needs its repeat count to mean anything, and the chunked one at
50,000 needs the most**: it spans 23.1 s to 27.9 s across those twelve runs,
while its unchunked denominator stays inside 14.7 to 16.0. A single run of that
cell lands anywhere in a 20% band, which is wide enough to make the two sizes
look as though they cost differently.

The reference is close to indifferent, as its one-replication-at-a-time loop
predicts, though less cleanly than a single measurement suggested. Two
independent runs of three at 2,000 segments and 200 replications on one thread:
8.26 s chunked against 8.20 s whole, and 8.23 s against 7.83 s. So the cost of
chunking it is somewhere between under 1% and about 5% — in the second, every
chunked run was slower than every whole one — against the 66% the kernel pays.
Memory does not move at all, 230.8 MB against 230.6 MB. Which of the two gaps is
right does not change what the number is used for: the reference needs no bound,
and it loses little by not having one.

**What a run holds grows with the replication count and nothing caps it.** The
result arrays of `(replications, years, classes)` are the visible part — one
per field of `simulate.Results`, 5.8 MB together at a thousand replications —
but they are not the figure to plan against: `results.rows_from_chunk` builds
a frame from them and holds it while they are still live, and the peak carries
both.

Measured at 1,000, 10,000, 20,000 and 40,000 replications: **327 MB, 938, 1,428
and 2,299**. That is about 51 MB per thousand over the whole range, some nine
times what the arrays alone account for. The growth decelerates — 68 MB per
thousand between the first two points and 44 between the last two — so
extrapolating from the bottom overstates, and the top-end rate puts a hundred
thousand replications near five gigabytes against the 576 MB the arrays
suggest. The count is what someone raises to narrow a confidence interval, so
this is the growth that would be met first.

**The polars thread pool is a condition of those numbers, not a detail.** It
sizes itself from the core count, and the frame the rows are built into carries
per-thread state, so the same run at 1,000 replications peaks at 166 MB with
one polars thread, 219 with four and 327 with forty-eight. A figure recorded
without that condition does not reproduce on a machine with a different core
count, and reads as a defect when it fails to.

The whole driver, so it can be run again: `run.run` on the kernel over the five
configured policies, at 200 segments, one worker thread, no batch, release
build, one process per point because a peak belongs to the process, above a
126 MB interpreter, with the polars pool left at this machine's forty-eight.
The population size moves these little — the result arrays and the rows frame
are shaped by replications, years and classes rather than by segments — which
is why 200 is enough to measure the growth that matters here, and why these
figures sit far below the 386 MB an unchunked kernel run reaches at 12,000
segments above.

It is stated rather than bounded because no run has been taken near the top of
that range, and a cap chosen without one would be a number with no measurement
behind it.
"""

BATCH_SIZES: dict[str, int | None] = {
    "reference": UNBOUNDED_BATCH,
    "batched_numpy": BOUNDED_BATCH_SIZE,
    "kernel": UNBOUNDED_BATCH,
}
"""What each implementation is batched at when a caller names no size.

One number cannot serve these: the batched loop needs a bound and the other two
are only slowed by one, so a default that suits either is wrong for the rest.
Keyed by implementation for the reason `CONCURRENT` is — it is a property of the
implementation, declared where the implementations are, and every name here has
to be one `RUNNABLE` holds.
"""

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
here must be one ``RUNNABLE`` holds; the import-time guard below calls
``missing_names`` and raises on any that is not.
"""


def batch_size_for(implementation: str, n_reps: int) -> int:
    """How many replications one call of this implementation should take.

    Args:
        implementation: The name it is keyed into ``RUNNABLE`` under.
        n_reps: Replications in the whole run, which is the answer for an
            implementation that wants no bound.

    Returns:
        The batch size to chunk with.

    Raises:
        KeyError: If no implementation goes by that name.
    """
    if implementation not in BATCH_SIZES:
        raise KeyError(
            f"no batch size is declared for {implementation!r}; "
            f"the implementations are {sorted(BATCH_SIZES)}"
        )
    bounded = BATCH_SIZES[implementation]
    return n_reps if bounded is None else bounded


def missing_names(names: Iterable[str], known: Iterable[str]) -> set[str]:
    """Names among these that ``known`` does not cover.

    A function rather than an expression inlined into the guards below, because
    an import-time check cannot be watched failing: breaking one stops the whole
    suite at collection rather than reddening a test. This can be called with a
    name that is deliberately wrong, and asserted on.

    Args:
        names: The names to check.
        known: The names that exist.

    Returns:
        Those that are not in ``known``.
    """
    return set(names) - set(known)


# The three registries have to agree, and each disagreement has its own
# consequence, so each is refused separately with the reason attached.
if unbatched := missing_names(RUNNABLE, BATCH_SIZES):
    raise ValueError(
        f"{sorted(unbatched)} are runnable but no batch size is declared for "
        f"them; every implementation needs one, because a run that names no "
        f"size has to resolve to something"
    )

if unthreadable := missing_names(CONCURRENT, RUNNABLE):
    raise ValueError(
        f"{sorted(unthreadable)} claim to spread replications but name no "
        f"implementation; the runnable set is {sorted(RUNNABLE)}"
    )

if unrunnable := missing_names(RUNNABLE, results.IMPLEMENTATIONS):
    raise ValueError(
        f"{sorted(unrunnable)} name no implementation a result may claim; "
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


def floating_for(precision: str) -> type[np.floating]:
    """The dtype a named precision builds its arrays at.

    Args:
        precision: A key of ``constants.PRECISIONS``.

    Returns:
        The NumPy floating type.

    Raises:
        ValueError: If the name is not one of the precisions that exist. The
            configuration model refuses one first, so only a direct caller
            arrives here.
    """
    floating = constants.PRECISIONS.get(precision)
    if floating is None:
        raise ValueError(
            f"precision is {precision!r}, which names no dtype; "
            f"the choices are {sorted(constants.PRECISIONS)}"
        )
    return floating


def escalation_series(rate: float, n_years: int, precision: str) -> np.ndarray:
    """Compounds an annual rate into a per-year multiplier.

    Args:
        rate: Annual growth, as a fraction.
        n_years: Horizon.
        precision: The dtype to return at, named as ``"f64"`` or ``"f32"``.
            Compounded in double whatever it is and narrowed once, so the
            series does not accumulate the rounding of the width it is
            returned at.

    Returns:
        One multiplier per year, starting at 1.0 in year 0.
    """
    return ((1.0 + rate) ** np.arange(n_years, dtype=np.float64)).astype(
        floating_for(precision)
    )


def segment_arrays(frame: pl.DataFrame, precision: str) -> dict[str, np.ndarray]:
    """Extracts the per-segment arrays an implementation reads.

    A segment's position in these arrays is its identifier, and the tie-break
    that decides which of two equally ranked candidates is funded reads that
    position, so the frame is sorted before anything is taken from it.

    **The dtype of these arrays is how the working precision reaches every
    implementation.** None of them takes a precision argument; each reads what
    it is handed, which is why a run at single precision needs nothing passed
    down a signature that already carries twenty-odd arguments.

    Args:
        frame: The population, one row per segment.
        precision: ``"f64"`` or ``"f32"``, naming the dtype every per-segment
            float array is built at. Required rather than defaulted: this
            argument has been forgotten twice, and both times it defaulted
            quietly to double and produced a run at a width nobody asked for.

    Returns:
        The arrays, keyed by the argument name each is passed as.

    Raises:
        KeyError: If the population is missing a column the loop reads.
        ValueError: If the precision names no dtype.
    """
    floating = floating_for(precision)
    ordered = frame.sort("segment_id")
    missing = [name for name in SEGMENT_COLUMNS if name not in ordered.columns]
    if missing:
        raise KeyError(f"the population has no {missing}; it cannot be simulated")
    arrays = {name: ordered[name].to_numpy() for name in SEGMENT_COLUMNS}
    # The loop names the starting age `age0`, because it holds a current age
    # that moves; the population column is the age at year 0.
    arrays["age0"] = arrays.pop("age")
    # The class index keys into the result axis and is not part of the
    # arithmetic, so it keeps its own width whatever the floats are doing.
    classes = arrays.pop("class_index").astype(np.uint8)
    return {
        **{name: array.astype(floating) for name, array in arrays.items()},
        "class_index": classes,
    }


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
    # At the run's precision, like the per-segment arrays. A double-precision
    # series multiplied into single-precision costs widens the whole money path
    # back to double, and because every implementation would widen the same way
    # the parity tests could not see it.
    budget = settings.budget.annual * escalation_series(
        settings.budget.escalation, simulation.n_years, simulation.precision
    )
    cost_escalation = escalation_series(
        settings.costs.escalation_rate, simulation.n_years, simulation.precision
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
    batch_size: int | None = None,
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
        batch_size: Replications per call, or None to let the implementation
            decide. It is a memory bound for the one implementation that holds
            every replication in flight and a cost to the two that do not, so
            the default is theirs rather than a single number — ``BATCH_SIZES``
            has what each resolves to and why. The manifest records the resolved
            size, because "whatever the default was" cannot be read later.
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
    # Resolved once, here, rather than defaulted in the signature: one number
    # cannot serve implementations whose constraints point opposite ways, and
    # what the manifest must record is the size that actually ran, not the
    # absence of a request.
    if batch_size is None:
        batch_size = batch_size_for(implementation, settings.simulation.n_reps)

    run_id = results.new_run_id()
    segments_frame = population.generate(settings)
    class_names = [segment_class.name for segment_class in settings.population.classes]
    segments = segment_arrays(segments_frame, settings.simulation.precision)
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
