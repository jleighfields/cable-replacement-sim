"""The run directory: what a run writes, and how a sweep is read back.

One directory per run, named for the moment it started::

    results/<run_id>/
    ├── config.yaml      # the effective configuration, as validated
    ├── manifest.json    # what the numbers cannot be interpreted without
    └── results.parquet  # one row per policy, replication, year and class

A **run** is one configuration evaluated for every policy, so ``policy`` is a
key column inside a single file rather than a directory per policy. A **sweep**
is a set of runs varying something between them — a budget level, a seed — and
the values that vary between runs are written into the parquet as columns.
A frame carrying its own parameters can be read without the layout that
produced it, so a renamed directory or a run copied elsewhere still answers
questions, and reading a sweep stays a scan and a concatenation rather than a
configuration parser.

Parquet for storage and CSV only for export: parquet keeps dtypes and the
schema, compresses, and can be scanned lazily, where CSV round-trips floats
through text and loses both. Nothing in this project reads CSV back.
"""

import datetime
import importlib.metadata
import logging
import pathlib
import subprocess

import numpy as np
import polars as pl
import pydantic
import yaml

from cablesim import config as config_module
from cablesim import simulate

log = logging.getLogger(__name__)

RESULTS_NAME = "results.parquet"
CONFIG_NAME = "config.yaml"
MANIFEST_NAME = "manifest.json"

IMPLEMENTATIONS: tuple[str, ...] = (
    "reference",
    "batched_numpy",
    "batched_polars",
    "kernel",
    "kernel_polars",
)
"""The implementations of the annual loop a result can come from.

These name the *role* rather than the module: the reference is the one a parity
failure is arbitrated against, whatever file it lives in. The set is closed
because a misspelled implementation in a manifest is provenance that reads as
fact and is not.

Two pairs run the same algorithm in different languages, and the pairing is the
point. ``batched_numpy`` and ``kernel`` are the array form in Python and in
Rust; ``batched_polars`` and ``kernel_polars`` are the frame form in each. The
Python polars package is a binding over the same Rust query engine the crate
exposes, so timing one frame implementation against the other separates what it
costs to drive that engine from Python from what the engine itself costs — a
question neither one alone can answer.
"""

SCHEMA: dict[str, pl.DataType] = {
    "policy": pl.String,
    "replication": pl.Int32,
    "year": pl.Int32,
    "class": pl.String,
    "failures": pl.Float64,
    "customers_interrupted": pl.Float64,
    "customer_minutes": pl.Float64,
    "planned_customer_minutes": pl.Float64,
    "planned_replacements": pl.Float64,
    "planned_spend": pl.Float64,
    "emergency_spend": pl.Float64,
}
"""The columns every ``results.parquet`` carries, before swept parameters.

Every array an implementation returns is saved, without exception, because the
metrics are computed from this file and from nothing else — so a quantity
produced and not written is a metric that cannot be computed. The one most
likely to look redundant is the count of customers interrupted, which is not
recoverable from the customer-minutes beside it once restoration time varies by
class.

The counts are stored as floats rather than cast to integers: they are exactly
what the implementation returned, and a cast would be the place a non-integral
value went quiet.
"""


class Manifest(pydantic.BaseModel):
    """What a saved result cannot be interpreted without.

    Attributes:
        run_id: The directory name, and the run's identity.
        written_at: When the run finished, in UTC.
        package_version: The installed version of this package.
        git_commit: The commit the run was produced at, or None if it could not
            be determined.
        git_dirty: Whether the working tree had uncommitted changes, or None
            alongside an unknown commit.
        implementation: Which implementation of the annual loop ran.
        build_profile: The Rust build profile, where the kernel ran. A timing
            without one means nothing.
        threads: How many threads the run used.
        batch_size: Replications per call. It changes no number — every
            replication reads the same draws at any batch size — and it is
            recorded because it is what a timing has to be read against.
        wall_seconds: How long the run took.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    run_id: str
    written_at: datetime.datetime
    package_version: str
    git_commit: str | None
    git_dirty: bool | None
    implementation: str
    build_profile: str | None
    threads: int
    batch_size: int
    wall_seconds: float

    @pydantic.field_validator("implementation")
    @classmethod
    def implementation_is_known(cls, value: str) -> str:
        """Rejects an implementation name nothing produces.

        Args:
            value: The proposed name.

        Returns:
            The validated name.

        Raises:
            ValueError: If the name is not one of the known implementations.
        """
        if value not in IMPLEMENTATIONS:
            raise ValueError(
                f"implementation must be one of {list(IMPLEMENTATIONS)}, "
                f"got {value!r}"
            )
        return value


def package_version() -> str:
    """The installed version of this package, for the manifest.

    The distribution name lives here rather than at every call site, since a
    misspelling would surface as a missing-package error at the end of a run
    that has already done all its work.

    Returns:
        The version string.
    """
    return importlib.metadata.version("cablesim")


def new_run_id(moment: datetime.datetime | None = None) -> str:
    """Builds a run identifier from the time the run started.

    Ordering runs is what the identifier is for, so it is a timestamp rather
    than a random string. The microseconds are what keep two runs started in
    the same second from colliding.

    Args:
        moment: The instant to name the run after. Defaults to now, in UTC.

    Returns:
        A timestamp of the form ``YYYYMMDDTHHMMSSffffff``.
    """
    if moment is None:
        moment = datetime.datetime.now(datetime.UTC)
    return moment.strftime("%Y%m%dT%H%M%S%f")


def git_provenance(root: pathlib.Path) -> tuple[str | None, bool | None]:
    """Reads the commit a run was produced at, and whether the tree was dirty.

    A result produced from uncommitted work is not reproducible from the commit
    alone, so the flag matters as much as the hash. Where git cannot answer —
    no repository, no git — this warns and reports the state as unknown rather
    than omitting the fields, because a manifest silently missing its
    provenance reads exactly like one that never had any.

    Args:
        root: A directory inside the repository to ask about.

    Returns:
        The commit hash and dirty flag, or ``(None, None)`` if unavailable.
    """
    try:
        # A fixed argument list, no shell, and the arguments are not built from
        # anything a caller supplies.
        commit = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        changes = subprocess.run(  # noqa: S603
            ["git", "status", "--porcelain"],  # noqa: S607
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        log.warning("git provenance unavailable for %s: %s", root, error)
        return None, None
    return commit, bool(changes.strip())


def rows_from_chunk(
    results: simulate.Results,
    policy: str,
    class_names: list[str],
    first_replication: int = 0,
) -> pl.DataFrame:
    """Lays one chunk of results out as the rows that get saved.

    Results are kept per replication rather than summarized, because two things
    need that axis and cannot recover it from a mean: the bands drawn around a
    trajectory, and the paired difference between two policies reading the same
    draws. A saved mean discards the pairing, which is the whole reason the
    draws are held fixed.

    Args:
        results: One chunk, as the annual loop returns it.
        policy: The policy that produced it.
        class_names: Segment class names, in the order the class axis uses.
        first_replication: The index of this chunk's first replication within
            the run, so chunks concatenate into one continuous axis.

    Returns:
        One row per replication, year and class.

    Raises:
        ValueError: If the class axis and the supplied names disagree, which
            would otherwise label every row of a whole class wrongly.
    """
    n_reps, n_years, n_classes = results.failures.shape
    if len(class_names) != n_classes:
        raise ValueError(
            f"the class axis is {n_classes} wide but {len(class_names)} names "
            f"were given ({class_names}); every row of at least one class "
            f"would be mislabelled"
        )

    # C order throughout, so reshaping a result array to one column lines up
    # with these three keys built the same way.
    columns: dict[str, object] = {
        "policy": [policy] * (n_reps * n_years * n_classes),
        "replication": np.repeat(
            np.arange(first_replication, first_replication + n_reps),
            n_years * n_classes,
        ),
        "year": np.tile(np.repeat(np.arange(n_years), n_classes), n_reps),
        "class": class_names * (n_reps * n_years),
    }
    for field in simulate.Results._fields:
        columns[field] = getattr(results, field).reshape(-1)
    return pl.DataFrame(columns, schema={name: SCHEMA[name] for name in columns})


def write_run(
    root: pathlib.Path,
    frame: pl.DataFrame,
    config: config_module.Config,
    manifest: Manifest,
) -> pathlib.Path:
    """Writes one run's directory and returns it.

    The configuration is dumped from the validated object rather than copied
    from whatever file was loaded, because a driver script's overrides never
    reach that file — copying it would record settings the run did not use,
    which is the failure this convention exists to prevent.

    Args:
        root: Where run directories live.
        frame: The rows to save.
        config: The effective configuration, already validated.
        manifest: The run's provenance.

    Returns:
        The directory written.
    """
    directory = root / manifest.run_id
    directory.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(directory / RESULTS_NAME)
    (directory / CONFIG_NAME).write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    (directory / MANIFEST_NAME).write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return directory


def read_sweep(root: pathlib.Path) -> pl.LazyFrame:
    """Reads every run under a directory into one lazy frame.

    Scanning lazily is what keeps a question about one budget level from
    materializing a whole sweep.

    Args:
        root: A directory of run directories.

    Returns:
        Every run's rows, concatenated.

    Raises:
        FileNotFoundError: If the directory does not exist, holds no run, or
            holds a run directory with no results file. A sweep silently short
            one budget level draws a curve that looks fine and is wrong, so a
            gap is refused rather than skipped.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"no sweep directory at {root}")

    runs = sorted(child for child in root.iterdir() if child.is_dir())
    if not runs:
        raise FileNotFoundError(f"{root} holds no run directories")

    missing = [run.name for run in runs if not (run / RESULTS_NAME).is_file()]
    if missing:
        raise FileNotFoundError(
            f"these run directories under {root} have no {RESULTS_NAME}: "
            f"{missing}. A sweep short one run produces a curve that looks "
            f"right and is not, so this is refused rather than skipped"
        )

    return pl.concat(pl.scan_parquet(run / RESULTS_NAME) for run in runs)
