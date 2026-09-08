"""Times every implementation of the annual loop and writes the table.

A **configuration** is one implementation run a stated way — a thread count and
a replication count — and produces one row. This script does nothing but choose
the rows and call the package: the timing, the agreement check and the table
all live in ``cablesim.benchmarks``, so this script and the notebook that
renders the same table cannot disagree about what was measured.

The default size is the shipped population. The scalar reference runs a smaller
replication count than the rest, because it is in the table for scale rather
than as the baseline a speedup is claimed against, and pinning every row to a
count it can finish would time the fast implementations on a workload too small
to show them. Seconds per replication is the column that compares across rows.

Build with ``uv run maturin develop --release`` before running this. A debug
build is slower by a wide margin, and the profile is written into the output so
a table produced from one is not mistaken for a measurement.
"""

import argparse
import json
import logging
import pathlib

import polars as pl
from cablesim import benchmarks, config, constants, kernel, run

log = logging.getLogger("run_benchmarks")

REFERENCE_REPS = 10
"""Replications for the scalar reference.

Lower than the rest because it is the slowest by a wide margin and is shown for
scale rather than compared against. Its per-replication time is what the table
reports, so a smaller count costs nothing but precision.
"""

BENCHMARK_REPS = 96
"""Replications for every other row.

Two per thread on a 48-core machine, so that a threaded row is not measuring
how many workers sat idle. Replications are the axis being parallelised, so a
count below the thread count caps the speedup at the count rather than at the
threads — ten replications cap it at ten however many cores exist.

This is the count the table published in the README was taken at. A machine
with a different core count will want a different number here, and the table
carries the count it was measured with for that reason.
"""


def configurations(threads: int) -> list[benchmarks.Configuration]:
    """The rows of the table, in the order they should appear.

    Both threading implementations appear twice, at one thread and at the
    machine's full count, so the language and the parallelism are not conflated
    into one number — which matters here, because single-threaded the kernel is
    level with Python on the policy that scores every segment.

    The batched loop's threaded row is here despite being slower than its own
    single-threaded row wherever the sort is small. It is the denominator of the
    fastest-Python column wherever the sort is large enough for threading to pay,
    so leaving it out would quote a speedup against a baseline that is not the
    fastest Python available.

    Args:
        threads: The full thread count to run the threading implementations at.

    Returns:
        One configuration per row.
    """
    if threads > BENCHMARK_REPS:
        # The constant says why: replications are the axis being parallelised,
        # so a count below the thread count caps the speedup at the count. A
        # row capped that way reads as a result rather than as a starved run.
        log.warning(
            "%d workers over %d replications: the speedup is capped by the "
            "replication count, not by the threads. Raise BENCHMARK_REPS to at "
            "least the thread count before quoting this table.",
            threads,
            BENCHMARK_REPS,
        )
    return [
        benchmarks.Configuration("reference", 1, REFERENCE_REPS),
        benchmarks.Configuration("batched_numpy", 1, BENCHMARK_REPS),
        benchmarks.Configuration("batched_numpy", threads, BENCHMARK_REPS),
        benchmarks.Configuration("kernel", 1, BENCHMARK_REPS),
        benchmarks.Configuration("kernel", threads, BENCHMARK_REPS),
    ]


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Reads the command line.

    Args:
        argv: Arguments to parse, or None to read the real command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=constants.PROJECT_ROOT / "results" / "benchmarks",
        help="directory to write the table and its provenance into",
    )
    parser.add_argument(
        "--precision",
        default=constants.DEFAULT_PRECISION,
        choices=sorted(constants.PRECISIONS),
        help=(
            "the floating width every implementation computes in. It reaches "
            "them as the dtype of the arrays, and it changes the numbers as "
            "well as the timings, so it is recorded beside the table"
        ),
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=None,
        help=(
            "population size, resized from the shipped one; the customer "
            "denominator and the annual budget move with it. Without this the "
            "configured size is used"
        ),
    )
    parser.add_argument(
        "--reduced",
        action="store_true",
        help="run the smaller population, for a table while someone watches",
    )
    parser.add_argument(
        "--policy",
        default="risk_ranked",
        help=(
            "which policy to time; it decides how much of the population is a "
            "candidate each year and therefore how much work there is"
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=kernel.AVAILABLE_THREADS,
        help="the full thread count to run the threading implementations at",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=benchmarks.DEFAULT_REPEATS,
        help=(
            "runs per configuration; the table carries the mean of them and "
            "the fastest, and the count they were taken over"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Runs the benchmark and writes the table.

    Args:
        argv: Command-line arguments, or None to read the real ones.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    arguments = parse_arguments(argv)

    settings = config.with_overrides(
        config.load_config(constants.DEFAULT_CONFIG_PATH),
        {"simulation.precision": arguments.precision},
    )
    if arguments.reduced:
        settings = config.resize_population(
            settings, run.REDUCED_SEGMENTS, n_reps=run.REDUCED_REPS
        )
    if arguments.segments is not None:
        settings = config.resize_population(settings, arguments.segments)
    log.info(
        "timing %d segments over %d years under %s at %s",
        settings.population.n_segments,
        settings.simulation.n_years,
        arguments.policy,
        settings.simulation.precision,
    )

    table = benchmarks.compare(
        settings,
        arguments.policy,
        configurations(arguments.threads),
        repeats=arguments.repeats,
    )
    disagreed = table.filter(~pl.col("matches_reference"))
    if disagreed.height > 0:
        # Reported rather than raised: the table is still worth writing, and it
        # is the evidence for what went wrong. A timing of an implementation
        # that has drifted measures something else being computed.
        log.warning(
            "%d configuration(s) did not reproduce the reference: %s",
            disagreed.height,
            disagreed["configuration"].to_list(),
        )

    arguments.out.mkdir(parents=True, exist_ok=True)
    table.write_parquet(arguments.out / "benchmarks.parquet")
    provenance = {
        **benchmarks.provenance(),
        "n_segments": settings.population.n_segments,
        "n_years": settings.simulation.n_years,
        # The width changes the numbers as well as the timings, so two tables
        # taken at different precisions are otherwise indistinguishable on disk.
        "precision": settings.simulation.precision,
        "policy": arguments.policy,
        "repeats": arguments.repeats,
    }
    (arguments.out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    log.info("wrote %s", arguments.out)
    with pl.Config(tbl_rows=-1, tbl_width_chars=160):
        log.info("\n%s", table)


if __name__ == "__main__":
    main()
