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

BENCHMARK_REPS = 50
"""Replications for every other row."""


def configurations(threads: int) -> list[benchmarks.Configuration]:
    """The rows of the table, in the order they should appear.

    The Rust kernel appears twice, at one thread and at the machine's full
    count, so the language and the parallelism are not conflated into one
    number. The two frame implementations appear once each: neither spreads
    replications over workers, because polars sizes its own pool.

    Args:
        threads: The full thread count to run the kernel at.

    Returns:
        One configuration per row.
    """
    return [
        benchmarks.Configuration("reference", 1, REFERENCE_REPS),
        benchmarks.Configuration("batched_numpy", 1, BENCHMARK_REPS),
        benchmarks.Configuration("batched_polars", 1, BENCHMARK_REPS),
        benchmarks.Configuration("kernel", 1, BENCHMARK_REPS),
        benchmarks.Configuration("kernel", threads, BENCHMARK_REPS),
        benchmarks.Configuration("kernel_polars", 1, BENCHMARK_REPS),
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
        help="the full thread count to run the kernel at",
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

    settings = config.load_config(constants.DEFAULT_CONFIG_PATH)
    if arguments.reduced:
        settings = config.resize_population(
            settings, run.REDUCED_SEGMENTS, n_reps=run.REDUCED_REPS
        )
    log.info(
        "timing %d segments over %d years under %s",
        settings.population.n_segments,
        settings.simulation.n_years,
        arguments.policy,
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
        "policy": arguments.policy,
        "repeats": arguments.repeats,
    }
    (arguments.out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    log.info("wrote %s", arguments.out)
    with pl.Config(tbl_rows=-1, tbl_width_chars=160):
        log.info("\n%s", table)


if __name__ == "__main__":
    main()
