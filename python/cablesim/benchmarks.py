"""Timing the implementations of the annual loop against each other.

Three programs compute the same numbers from the same draws — a scalar Python
reference, a batched NumPy loop and the Rust kernel — and this module times them
and checks that they still agree while doing it.

**A timing without its agreement is worth nothing**, so the two are produced
together rather than in separate passes. An implementation that has drifted is
fast at computing something else, and a table that reported only seconds would
show that as an improvement.

Three rules the numbers here follow, each of which the project would otherwise
be free to break:

* **The scalar reference is not the baseline.** It is written to be checkable
  by reading, and a speedup claimed against it would be flattered by that. It
  appears in the table for scale. The batched NumPy loop is what a speedup is
  honestly claimed against.
* **Each row states its own replication count**, because the reference is slow
  enough that pinning every row to a count it can finish would time the fast
  implementations on a workload too small to show them. Seconds per replication
  is the column to compare across rows; total seconds is not.
* **A timing carries its build profile and thread count.** A debug build is
  slower by a wide margin, and an unlabelled number cannot be checked against
  anything.

**One run happens before the clock starts, and is reported as its own column.**
An implementation Numba compiles builds machine code on its first call, which is
a cost paid once per process rather than once per run, so folding it into a mean
over three runs describes neither the first call nor the ones after it. Warming
first and reporting ``first_run_seconds`` beside the mean says both. The other
implementations pay something smaller on their first call — pages touched, a
buffer allocated — and warming removes that from every row alike rather than
from the one it would flatter.

**The measurement is the mean of several runs, and the count is in the table.**
The minimum is the other defensible choice and answers a different question:
noise on a shared machine is all in one direction, so the fastest run is the one
least interrupted, and a minimum therefore estimates the work rather than the
conditions. The mean is what someone actually waits for, and it is the more
conservative choice for a speedup claim, because interruption inflates the
numerator and the denominator alike rather than only the row one is pleased
with. Both are reported; neither means anything without the repeat count beside
it.
"""

import logging
import time
from typing import NamedTuple

import numpy as np
import polars as pl

from cablesim import config as config_module
from cablesim import kernel, policies, population, random_draws, run, simulate

log = logging.getLogger(__name__)

DEFAULT_REPEATS = 3
"""Runs per configuration, averaged into the reported time."""

PYTHON_IMPLEMENTATIONS: frozenset[str] = frozenset({"reference", "batched_numpy"})
"""The implementations that run in Python, whatever they call underneath.

Named here rather than derived from the runnable registry, because what makes a
row belong is that no compiled kernel of this project's own runs it. NumPy is
itself compiled, and that is the point of comparing against it: the claim being
tested is about this project's Rust, not about compiled code in general. The
fastest of these is the denominator any claim about the Rust kernel should use.
"""


class Configuration(NamedTuple):
    """One row of the table: an implementation, run a stated way.

    Attributes:
        implementation: The name in ``run.RUNNABLE``.
        threads: Workers to spread replications over. The implementations
            named in ``run.CONCURRENT`` accept more than one; every other one
            refuses, which is what keeps a row from claiming a thread count
            nothing acted on.
        n_reps: Replications to run. Stated per configuration because the
            scalar reference cannot afford what the others should be measured
            at.
    """

    implementation: str
    threads: int
    n_reps: int


def label(configuration: Configuration) -> str:
    """Names a configuration for a table.

    Args:
        configuration: The configuration to name.

    Returns:
        The implementation name, with the thread count appended where it is
        more than one — a name carrying "1 thread" on every row would say
        nothing, and one carrying it on none would hide the row that matters.
    """
    if configuration.threads > 1:
        return f"{configuration.implementation} × {configuration.threads} threads"
    return configuration.implementation


def population_arguments(settings: config_module.Config) -> dict[str, object]:
    """The arguments a run builds once, however many chunks it then runs.

    Separated from the draws because the two are paid at different rates and
    only one of them belongs inside a timing. A run generates its population
    once and its draws once per chunk, so a per-chunk measurement that included
    the population would charge every chunk for work the run did once.

    Args:
        settings: The configuration to build from.

    Returns:
        Every argument of ``simulate.run_chunk`` except the two uniform arrays,
        ``policy`` and ``threads``.
    """
    simulation = settings.simulation
    precision = simulation.precision
    segments = run.segment_arrays(population.generate(settings), precision)

    return {
        **segments,
        # The per-year series carry the precision too. Left in double they
        # would widen the money path back, and every implementation would widen
        # the same way, so nothing comparing them could see it.
        "budget": settings.budget.annual
        * run.escalation_series(
            settings.budget.escalation, simulation.n_years, precision
        ),
        "cost_escalation": run.escalation_series(
            settings.costs.escalation_rate, simulation.n_years, precision
        ),
        "emergency_multiplier": settings.costs.emergency_multiplier,
        "mobilization_per_segment": settings.costs.mobilization_per_segment,
        "emergency_charged_to_budget": settings.budget.emergency_charged_to_budget,
        "n_classes": len(settings.population.classes),
        "n_years": simulation.n_years,
    }


def draw_arguments(settings: config_module.Config, n_reps: int) -> dict[str, object]:
    """What a chunk needs to produce its own draws.

    A key and a position, rather than the uniforms themselves. **Every
    implementation now generates inside the timed region**, which is where the
    work belongs: an implementation that produces only the draws it reads
    deserves the credit, and one that produces eighteen million to read a few
    hundred thousand should be charged for it.

    Args:
        settings: The configuration to build from, which fixes the seed.
        n_reps: Replications this chunk covers.

    Returns:
        The key and the chunk's position, keyed by argument name.
    """
    return {
        "draw_key": random_draws.draw_key(settings.simulation.seed),
        "first_replication": 0,
        "n_reps": n_reps,
    }


def chunk_arguments(settings: config_module.Config, n_reps: int) -> dict[str, object]:
    """One call's arguments, population and draws together.

    Args:
        settings: The configuration to build from.
        n_reps: Replications this chunk covers.

    Returns:
        Every argument of ``simulate.run_chunk`` except ``policy`` and
        ``threads``.
    """
    return {**population_arguments(settings), **draw_arguments(settings, n_reps)}


def agrees(expected: simulate.Results, produced: simulate.Results) -> bool:
    """Whether two results are equal in every cell of every array.

    Exact rather than tolerant, because every implementation reads the same
    draws and takes the same decisions from them, and each is held to exact
    agreement by the parity tests. A tolerance here would report as a match
    something the suite would call a failure.

    Args:
        expected: What the scalar reference returned.
        produced: What the timed implementation returned.

    Returns:
        True if every cell matches.
    """
    return all(
        np.array_equal(wanted, actual)
        for wanted, actual in zip(expected, produced, strict=True)
    )


def time_once(
    configuration: Configuration,
    settings: config_module.Config,
    arguments: dict[str, object],
    policy: policies.Resolved,
    repeats: int,
) -> tuple[simulate.Results, float, float, float]:
    """Runs one configuration, returning its result and how long it took.

    Args:
        configuration: What to run and how.
        settings: The configuration, for rebuilding the draws each run.
        arguments: The population arguments, built once outside this.
        policy: The resolved policy to run under.
        repeats: How many times to run it.

    Returns:
        The result of the last run, the mean wall time, the shortest, and what
        the first call cost before the clock started. The mean and the minimum
        are both returned because the gap between them is worth seeing: a row
        whose mean sits well above its minimum was interrupted, and its mean is
        then measuring the machine rather than the implementation. The first
        call is returned because for an implementation compiled at runtime it
        is the compilation, which is real and is not part of what a run costs
        per chunk.

    Raises:
        ValueError: If asked for no runs at all, which would otherwise report a
            time for work that never happened.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    annual_loop = run.RUNNABLE[configuration.implementation]

    warming = time.perf_counter()
    annual_loop(
        **arguments,
        **draw_arguments(settings, configuration.n_reps),
        policy=policy,
        threads=configuration.threads,
    )
    first_run = time.perf_counter() - warming

    elapsed = []
    produced = None
    for _ in range(repeats):
        started = time.perf_counter()
        # The draws are built inside the clock because a run builds them once
        # per chunk, so they are part of what a chunk costs. The population is
        # not, because a run builds that once however many chunks follow.
        draws = draw_arguments(settings, configuration.n_reps)
        produced = annual_loop(
            **arguments, **draws, policy=policy, threads=configuration.threads
        )
        elapsed.append(time.perf_counter() - started)
    return produced, sum(elapsed) / len(elapsed), min(elapsed), first_run


def compare(
    settings: config_module.Config,
    policy_name: str,
    configurations: list[Configuration],
    repeats: int = DEFAULT_REPEATS,
) -> pl.DataFrame:
    """Times every configuration and checks each against the scalar reference.

    The reference is run once per distinct replication count and its result
    kept, so the agreement column compares like with like without paying for a
    reference run per row.

    **Draw generation is inside the timing and population generation is not**,
    because a run pays for one per chunk and the other once. That boundary is
    not a detail: producing eighteen million uniforms to read a few hundred
    thousand of them costs about a tenth of a second per chunk, and with it
    outside the clock a change that stopped doing so would read as a
    regression.

    Args:
        settings: The configuration to run, which fixes the population size,
            the horizon and the seed.
        policy_name: Which policy to run under. One rather than all, because a
            policy decides how much of the population is a candidate each year
            and therefore how much work there is — a table averaged over
            policies would hide the thing it is measuring.
        configurations: The rows to produce, in the order they should appear.
        repeats: Runs per configuration, averaged into the reported time.

    Returns:
        One row per configuration: what ran, the mean of its runs and the
        fastest of them, how many runs that was over, the mean per replication,
        whether it reproduced the reference exactly, and two speedup columns —
        against the batched NumPy baseline the design names, and against
        whichever Python implementation was actually fastest.

    Raises:
        KeyError: If a configuration names no implementation that exists.
    """
    unknown = sorted(
        {
            configuration.implementation
            for configuration in configurations
            if configuration.implementation not in run.RUNNABLE
        }
    )
    if unknown:
        raise KeyError(
            f"{unknown} name no implementation; {sorted(run.RUNNABLE)} are the "
            f"ones that exist"
        )

    policy = policies.resolve(
        next(spec for spec in settings.policies if spec.name == policy_name)
    )
    shared = population_arguments(settings)
    reference: dict[int, simulate.Results] = {}
    rows = []
    for configuration in configurations:
        n_reps = configuration.n_reps
        if n_reps not in reference:
            reference[n_reps] = simulate.run_chunk(
                **shared, **draw_arguments(settings, n_reps), policy=policy
            )
        produced, seconds, fastest, first_run = time_once(
            configuration, settings, shared, policy, repeats
        )
        log.info(
            "%s: %.3f s, mean of %d, over %d replications",
            label(configuration),
            seconds,
            repeats,
            n_reps,
        )
        rows.append(
            {
                "configuration": label(configuration),
                "implementation": configuration.implementation,
                "threads": configuration.threads,
                "replications": n_reps,
                "repeats": repeats,
                "seconds": seconds,
                "fastest_seconds": fastest,
                "first_run_seconds": first_run,
                "seconds_per_replication": seconds / n_reps,
                "matches_reference": agrees(reference[n_reps], produced),
            }
        )

    frame = pl.DataFrame(rows)
    # Two denominators, because one of them turned out not to be safe on its
    # own. The batched NumPy loop is the baseline the design named, and it is
    # the right one to quote for the array form. But it is not always the
    # fastest Python here — where a policy makes a small fraction of the
    # population eligible, the batched form sorts every segment while the
    # scalar reference sorts only the candidates, and the reference wins. A
    # speedup quoted against a baseline the reference beats is flattered, so
    # the second column is against whichever Python implementation was actually
    # fastest, and that is the one an outside claim should use.
    #
    # The single-threaded row specifically. That implementation can now be
    # listed at more than one thread count, and taking whichever row came first
    # would make the denominator depend on the order the caller wrote its
    # configurations in.
    baseline = frame.filter(
        (pl.col("implementation") == "batched_numpy") & (pl.col("threads") == 1)
    )
    if baseline.height > 0:
        per_replication = baseline["seconds_per_replication"][0]
        frame = frame.with_columns(
            speedup_over_batched_numpy=per_replication
            / pl.col("seconds_per_replication")
        )
    python_rows = frame.filter(pl.col("implementation").is_in(PYTHON_IMPLEMENTATIONS))
    if python_rows.height > 0:
        fastest_python = python_rows["seconds_per_replication"].min()
        frame = frame.with_columns(
            speedup_over_fastest_python=fastest_python
            / pl.col("seconds_per_replication")
        )
    return frame


def provenance() -> dict[str, object]:
    """What a timing table cannot be read without.

    Returns:
        The Rust build profile, the machine's thread count, the polars version
        and the size of polars' own thread pool.

        **No implementation timed here runs on polars.** The three are the
        scalar reference, the batched NumPy loop and the Rust kernel, so the
        last two entries describe the environment the table was assembled in
        rather than a property of any row in it. They are recorded because
        polars sizes its thread pool from available parallelism and nothing
        else in a result says what that pool was — which is what the retired
        frame implementations in ``deprecated/`` need stated if their
        measurements are ever reproduced against these rows.
    """
    return {
        "build_profile": kernel.BUILD_PROFILE,
        "available_threads": kernel.AVAILABLE_THREADS,
        "polars_version": pl.__version__,
        "polars_threads": pl.thread_pool_size(),
    }
