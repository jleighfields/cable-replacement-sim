"""Turning a configuration into a saved run.

This is the only module that writes anything. Everything else in the reporting
layer is a pure function of a frame; the loop that produces the frame is not,
because it owns the chunking, the ordering and the concatenation.

A **run** is one configuration evaluated for every policy in it, producing one
directory. The implementation is an argument, so the reference and the compute
kernel are interchangeable here — which is what lets a parity test drive both
through one path rather than through two that could differ in how they are
driven.

Replications are processed in chunks because the draw array is the largest
thing in a run: at a thousand replications, twelve thousand segments and a
thirty-year horizon it is three gigabytes, and fifty replications at a time
makes it a hundred and fifty megabytes. The chunk size changes no number, since
every replication reads its own children of the stream whatever the chunking,
so it is an argument here and provenance in the manifest rather than a
configured parameter.
"""

import datetime
import logging
import pathlib
import time
from collections.abc import Callable

import numpy as np
import polars as pl

from cablesim import config as config_module
from cablesim import constants, policies, population, random_draws, results, simulate

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
) -> pl.DataFrame:
    """Runs every replication under one policy, chunk by chunk.

    Args:
        spec: The policy to run.
        settings: The effective configuration.
        segments: The per-segment arrays.
        class_names: Segment class names in class-index order.
        implementation: The annual loop to call.
        batch_size: Replications per call.

    Returns:
        Every replication's rows for this policy.
    """
    simulation = settings.simulation
    n_segments = segments["age0"].size
    sources = random_draws.spawn_sources(simulation.seed)
    resolved = policies.resolve(spec)
    budget = settings.budget.annual * escalation_series(
        settings.budget.escalation, simulation.n_years
    )
    cost_escalation = escalation_series(
        settings.costs.escalation_rate, simulation.n_years
    )

    blocks = []
    for replications in replication_chunks(simulation.n_reps, batch_size):
        block = implementation(
            **segments,
            lifetime_uniforms=random_draws.replication_uniforms(
                sources.lifetimes, replications, (n_segments, simulation.n_years + 1)
            ),
            policy_uniforms=random_draws.replication_uniforms(
                sources.policies, replications, (n_segments,)
            ),
            budget=budget,
            cost_escalation=cost_escalation,
            policy=resolved,
            emergency_multiplier=settings.costs.emergency_multiplier,
            mobilization_per_segment=settings.costs.mobilization_per_segment,
            emergency_charged_to_budget=settings.budget.emergency_charged_to_budget,
            n_classes=len(class_names),
            n_years=simulation.n_years,
        )
        blocks.append(
            results.rows_from_chunk(block, spec.name, class_names, replications.start)
        )
    return pl.concat(blocks)


def run(
    settings: config_module.Config,
    root: pathlib.Path,
    implementation: Implementation = simulate.run_chunk,
    implementation_name: str = "reference",
    build_profile: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    swept: dict[str, float] | None = None,
) -> pathlib.Path:
    """Runs one configuration for every policy and writes the result.

    Args:
        settings: The effective configuration, already validated.
        root: Where run directories are written.
        implementation: The annual loop to call.
        implementation_name: Which implementation that is, for the manifest.
        build_profile: The Rust build profile, where the kernel ran. Pure
            Python has none, so it stays absent rather than being invented; a
            timing from the kernel without one means nothing.
        batch_size: Replications per call.
        swept: Values that vary between the runs of a sweep, written into the
            saved rows as columns. A frame carrying its own parameters is
            readable without the directory layout that produced it.

    Returns:
        The directory written.
    """
    started = time.perf_counter()
    run_id = results.new_run_id()
    segments_frame = population.generate(settings)
    class_names = [segment_class.name for segment_class in settings.population.classes]
    segments = segment_arrays(segments_frame)
    log.info(
        "run %s: %d segments, %d policies, %d replications",
        run_id,
        segments["age0"].size,
        len(settings.policies),
        settings.simulation.n_reps,
    )

    frame = pl.concat(
        simulate_policy(
            spec, settings, segments, class_names, implementation, batch_size
        )
        for spec in settings.policies
    )
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
            implementation=implementation_name,
            build_profile=build_profile,
            threads=1,
            batch_size=batch_size,
            wall_seconds=time.perf_counter() - started,
        ),
    )
