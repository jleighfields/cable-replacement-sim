"""Measures peak resident memory for one implementation at one precision.

**One implementation per process, and that is the point of the script.** Peak
resident memory is a property of the process, so a program that ran three
implementations in turn reports the high-water mark of whichever needed most and
cannot attribute it. Running one per process is what makes the figure mean
something, and it is why this is a script that measures a single configuration
rather than a loop that builds a table.

The number reported is `ru_maxrss` for the whole process, which carries the
interpreter and every imported library as well as the model's arrays. The
figure printed alongside it is what that came to before any array existed, so
the difference between two runs is the model's and the ratio between them is
not.

Build with ``uv run maturin develop --release`` first: a debug build allocates
differently and the timing that usually accompanies these figures would be
meaningless besides.
"""

import argparse
import json
import logging
import resource
import sys

from cablesim import benchmarks, config, constants, policies, run

log = logging.getLogger("measure_memory")


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Reads the configuration this run measures.

    Args:
        argv: Command-line arguments, or None to read them from the process.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation",
        default="kernel",
        choices=sorted(run.RUNNABLE),
        help="which annual loop to measure",
    )
    parser.add_argument(
        "--precision",
        default=constants.DEFAULT_PRECISION,
        choices=sorted(constants.PRECISIONS),
        help="the floating width it computes in",
    )
    parser.add_argument(
        "--segments", type=int, default=12_000, help="population size"
    )
    parser.add_argument(
        "--reps", type=int, default=48, help="replications in the chunk"
    )
    parser.add_argument(
        "--policy", default="risk_ranked", help="which policy to run under"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "workers to spread the replications over; without this, every "
            "thread the machine has for an implementation that can use them "
            "and one for the reference, which cannot"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Runs one configuration and reports what the process peaked at.

    Args:
        argv: Command-line arguments, or None to read them from the process.

    Returns:
        A process exit status.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    arguments = parse_arguments(argv)

    after_import = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    settings = config.resize_population(
        config.with_overrides(
            config.load_config(constants.DEFAULT_CONFIG_PATH),
            {"simulation.precision": arguments.precision},
        ),
        arguments.segments,
        n_reps=arguments.reps,
    )
    chunk = benchmarks.chunk_arguments(settings, arguments.reps)
    spec = next(
        policy for policy in settings.policies if policy.name == arguments.policy
    )
    threads = arguments.threads
    if threads is None:
        threads = (
            kernel_threads()
            if arguments.implementation in run.CONCURRENT
            else 1
        )

    run.RUNNABLE[arguments.implementation](
        **chunk, policy=policies.resolve(spec), threads=threads
    )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    log.info(
        json.dumps(
            {
                "implementation": arguments.implementation,
                "precision": arguments.precision,
                "segments": arguments.segments,
                "replications": arguments.reps,
                "policy": arguments.policy,
                "threads": threads,
                "after_import_mb": round(after_import, 1),
                "peak_mb": round(peak, 1),
            }
        )
    )
    return 0


def kernel_threads() -> int:
    """The machine's thread count, read from the compiled extension.

    A function rather than a module-level constant so that importing this
    script does not require the extension to be built.

    Returns:
        How many workers the machine offers.
    """
    from cablesim import kernel

    return kernel.AVAILABLE_THREADS


if __name__ == "__main__":
    sys.exit(main())
