"""Runs the reliability-against-budget sweep and writes it to disk.

A **sweep point** is one budget level evaluated for every policy, and it
produces one run directory. This script does nothing but build configuration
overrides and call the runner in a loop: all the modelling lives in the
package, so that a notebook, an application and this script cannot disagree
about what a run is.

The grid is zero plus levels spaced geometrically from a fraction of the
configured budget to a multiple of it, so the region near a binding constraint
is sampled more densely than the flat region beyond it. **Zero is in the grid
deliberately**: every policy at zero budget must equal run-to-failure at any
budget, because none of them funds anything there, so the left-hand end of the
figure is a free end-to-end check on the whole stack.

The default size is reduced rather than the shipped one. In the pure-Python
reference a full-size sweep is tens of minutes, and a default that does not
finish while someone watches it is a default nobody runs. Pass ``--full`` for
the configured replication count and population.
"""

import argparse
import logging
import pathlib

import numpy as np
from cablesim import config, constants, run

log = logging.getLogger("budget_sweep")

REDUCED_REPS = 40
REDUCED_SEGMENTS = 2_000
GRID_LOW = 0.125
GRID_HIGH = 2.0
GRID_LEVELS = 7


def budget_grid(annual: float, levels: int = GRID_LEVELS) -> list[float]:
    """Builds the swept budget levels.

    Args:
        annual: The configured annual budget, which the grid brackets.
        levels: How many non-zero levels to place.

    Returns:
        Zero followed by geometrically spaced levels.
    """
    spaced = np.geomspace(annual * GRID_LOW, annual * GRID_HIGH, levels)
    return [0.0, *spaced.tolist()]


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

    settings = config.load_config(constants.DEFAULT_CONFIG_PATH)
    if not arguments.full:
        settings = config.Config.model_validate(
            {
                **settings.model_dump(),
                "simulation": {
                    **settings.simulation.model_dump(),
                    "n_reps": REDUCED_REPS,
                },
                "population": {
                    **settings.population.model_dump(),
                    "n_segments": REDUCED_SEGMENTS,
                },
            }
        )
        log.info(
            "reduced size: %d replications, %d segments. Pass --full for the "
            "configured %d and %d.",
            REDUCED_REPS,
            REDUCED_SEGMENTS,
            config.load_config(constants.DEFAULT_CONFIG_PATH).simulation.n_reps,
            config.load_config(constants.DEFAULT_CONFIG_PATH).population.n_segments,
        )

    grid = budget_grid(settings.budget.annual)
    log.info("sweeping %d budget levels into %s", len(grid), arguments.out)
    for index, level in enumerate(grid, start=1):
        point = config.Config.model_validate(
            {
                **settings.model_dump(),
                "budget": {**settings.budget.model_dump(), "annual": level},
            }
        )
        directory = run.run(
            point,
            arguments.out,
            batch_size=arguments.batch_size,
            swept={"annual_budget": level},
        )
        log.info(
            "  %d/%d  budget %.0f -> %s", index, len(grid), level, directory.name
        )


if __name__ == "__main__":
    main()
