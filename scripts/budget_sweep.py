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

The default size is reduced, because a full-size sweep through the pure-Python
reference takes tens of minutes. Pass ``--full`` for the configured replication
count and population, and ``--implementation kernel`` to run the same sweep
through the Rust kernel, which computes the same numbers from the same draws.
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
        choices=sorted(run.IMPLEMENTATIONS),
        default="reference",
        help="which annual loop to run; the name is recorded in the manifest",
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

    # Read from the compiled extension rather than assumed, because a debug
    # build is the one way this sweep silently takes far longer than it should,
    # and a timing recorded without the profile it ran under says nothing.
    build_profile = (
        kernel.BUILD_PROFILE if arguments.implementation == "kernel" else None
    )

    grid = run.budget_grid(settings.budget.annual)
    log.info(
        "sweeping %d budget levels into %s, through the %s implementation",
        len(grid),
        arguments.out,
        arguments.implementation,
    )
    for index, level in enumerate(grid, start=1):
        point = config.with_overrides(settings, {"budget.annual": level})
        directory = run.run(
            point,
            arguments.out,
            implementation=run.IMPLEMENTATIONS[arguments.implementation],
            implementation_name=arguments.implementation,
            build_profile=build_profile,
            batch_size=arguments.batch_size,
            swept={"annual_budget": level},
        )
        log.info(
            "  %d/%d  budget %.0f -> %s", index, len(grid), level, directory.name
        )


if __name__ == "__main__":
    main()
