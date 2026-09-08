"""Runs the reliability-against-budget sweep and writes it to disk.

A **sweep point** is one budget level evaluated for every policy, and it
produces one run directory. This script does nothing but build configuration
overrides and call the runner in a loop: all the modelling lives in the
package, so that a notebook, an application and this script cannot disagree
about what a run is.

The grid is zero plus levels spaced geometrically from a fraction of the
configured budget to a multiple of it, so the region near a binding constraint
is sampled more densely than the flat region beyond it. **Zero is in the grid
deliberately**: no policy funds anything there, so every one of them must land
on the same point as run-to-failure, which makes the left-hand end of the
figure an end-to-end check on the whole stack.

The default size is reduced, because a full-size sweep takes tens of minutes.
Pass ``--full`` for the configured replication count and population.

The sweep runs through the Rust kernel by default, and
``--implementation reference`` runs the same sweep through the Python
reference, which computes the same numbers from the same draws. The reference
is what a disagreement is arbitrated against, so it stays one flag away rather
than being the thing everyone waits for.
"""

import argparse
import logging
import pathlib

from cablesim import config, constants, kernel, run

log = logging.getLogger("budget_sweep")


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
        default=constants.PROJECT_ROOT / "results" / "budget_sweep",
        help="directory to write run directories into",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="use the configured replication count and population size",
    )
    parser.add_argument(
        "--implementation",
        choices=sorted(run.RUNNABLE),
        default="kernel",
        help="which annual loop to run; the name is recorded in the manifest",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help=(
            "workers to spread each chunk's replications over; the kernel and "
            "the batched loop can use more than 1 and the scalar reference "
            "refuses it, though threading the batched loop is slower than one "
            f"thread wherever the sort is small. This machine offers "
            f"{kernel.AVAILABLE_THREADS}"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=run.DEFAULT_BATCH_SIZE,
        help="replications per call; changes timing and memory, not results",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Runs the sweep.

    Args:
        argv: Command-line arguments, or None to read the real ones.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    arguments = parse_arguments(argv)

    shipped = config.load_config(constants.DEFAULT_CONFIG_PATH)
    settings = shipped
    if not arguments.full:
        settings = config.resize_population(
            shipped, run.REDUCED_SEGMENTS, n_reps=run.REDUCED_REPS
        )
        log.info(
            "reduced size: %d replications, %d segments, %d customers. Pass "
            "--full for the configured %d, %d and %d.",
            settings.simulation.n_reps,
            settings.population.n_segments,
            settings.population.total_customers,
            shipped.simulation.n_reps,
            shipped.population.n_segments,
            shipped.population.total_customers,
        )

    grid = run.budget_grid(settings.budget.annual)
    log.info(
        "sweeping %d budget levels into %s, through the %s implementation on "
        "%d thread(s)",
        len(grid),
        arguments.out,
        arguments.implementation,
        arguments.threads,
    )
    for index, level in enumerate(grid, start=1):
        point = config.with_overrides(settings, {"budget.annual": level})
        directory = run.run(
            point,
            arguments.out,
            implementation=arguments.implementation,
            batch_size=arguments.batch_size,
            threads=arguments.threads,
            swept={"annual_budget": level},
        )
        log.info(
            "  %d/%d  budget %.0f -> %s", index, len(grid), level, directory.name
        )


if __name__ == "__main__":
    main()
