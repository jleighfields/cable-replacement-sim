"""The run directory format, and reading a sweep back out of it.

A round trip proves little on its own: a writer and a reader sharing a mistake
agree with each other perfectly. So the on-disk schema is asserted separately
from the round trip, against the column names and dtypes rather than against
whatever the writer happened to produce.

Every fixture is built in code and written into a temporary directory. No
parquet is committed; the whole population is synthetic and reproducible from a
seed, so a stored data blob has no reason to exist here.
"""

import datetime
import json
import pathlib

import numpy as np
import polars as pl
import pytest
import yaml
from cablesim import config, constants, results, simulate

CLASS_NAMES = ["main_feeder", "lateral_1ph"]


def chunk(n_reps: int = 3, n_years: int = 4, n_classes: int = 2) -> simulate.Results:
    """Builds a results chunk whose every cell is distinguishable.

    Each array is filled with a different decade of values, and each cell
    within an array differs, so a transposed axis or a column written into the
    wrong place shows up as a wrong number rather than as a coincidence.

    The default extents are deliberately all different. With as many
    replications as classes, building the year key by repeating and then
    tiling produces the identical array to tiling and then repeating, so a
    transposed layout passes every check that reads it.

    Args:
        n_reps: Replications.
        n_years: Years.
        n_classes: Segment classes.

    Returns:
        One chunk of results.
    """
    size = n_reps * n_years * n_classes
    return simulate.Results(
        *(
            (np.arange(size, dtype=float) + offset * 1_000).reshape(
                n_reps, n_years, n_classes
            )
            for offset in range(len(simulate.Results._fields))
        )
    )


def manifest(run_id: str) -> results.Manifest:
    """Builds a manifest for a run written by a test.

    Args:
        run_id: The run's identifier.

    Returns:
        A valid manifest.
    """
    return results.Manifest(
        run_id=run_id,
        written_at=datetime.datetime(2026, 9, 6, 12, 0, tzinfo=datetime.UTC),
        package_version=results.package_version(),
        git_commit="0" * 40,
        git_dirty=False,
        implementation="reference",
        build_profile=None,
        threads=1,
        batch_size=2,
        wall_seconds=1.5,
    )


def test_the_saved_schema_is_what_it_claims_to_be() -> None:
    """Written out here rather than compared against the module's constant.

    Comparing the frame to the same declaration that built it is a tautology:
    it passes for any pair of writer and declaration that agree, including a
    pair that agree on the wrong thing. Spelling the columns and dtypes out
    means a change to the saved format has to be made twice, deliberately,
    which is the point of an on-disk contract.
    """
    frame = results.to_frame(chunk(), "risk_ranked", CLASS_NAMES)

    assert dict(frame.schema) == {
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


def test_every_returned_array_reaches_the_file() -> None:
    """A quantity produced and not saved is a metric that cannot be computed.

    The one most likely to be dropped is the count of customers interrupted,
    which looks redundant beside customer-minutes and is not: a count cannot be
    recovered from a duration-weighted sum once duration varies by class.
    """
    frame = results.to_frame(chunk(), "risk_ranked", CLASS_NAMES)

    for field in simulate.Results._fields:
        assert field in frame.columns, field
    assert "customers_interrupted" in frame.columns


def test_the_row_keys_line_up_with_the_values_they_label() -> None:
    """Reshaping the result arrays must agree with how the keys are built.

    Both are laid out in C order. If one were built the other way round, every
    row would still be present and every value would belong to the wrong year
    or the wrong class, which no count of rows would reveal.
    """
    results_chunk = chunk()
    n_reps, n_years, _ = results_chunk.failures.shape
    frame = results.to_frame(results_chunk, "worst_first", CLASS_NAMES)

    for replication in range(n_reps):
        for year in range(n_years):
            for index, name in enumerate(CLASS_NAMES):
                row = frame.filter(
                    (pl.col("replication") == replication)
                    & (pl.col("year") == year)
                    & (pl.col("class") == name)
                )
                assert row.height == 1
                assert row["failures"].item() == (
                    results_chunk.failures[replication, year, index]
                )


def test_a_chunk_knows_where_its_replications_sit_in_the_run() -> None:
    """Chunks concatenate into one continuous replication axis."""
    frame = results.to_frame(chunk(n_reps=2), "random", CLASS_NAMES, 50)

    assert sorted(frame["replication"].unique().to_list()) == [50, 51]


def test_a_class_axis_that_does_not_match_its_names_is_refused() -> None:
    """Otherwise every row of at least one class is labelled wrongly."""
    with pytest.raises(ValueError, match="mislabelled"):
        results.to_frame(chunk(n_classes=2), "random", ["only_one_name"])


def test_a_run_round_trips_through_its_directory(tmp_path: pathlib.Path) -> None:
    """The three files land, and the rows come back unchanged."""
    settings = config.load_config(constants.DEFAULT_CONFIG_PATH)
    frame = results.to_frame(chunk(), "risk_ranked", CLASS_NAMES)
    run_id = results.new_run_id()

    directory = results.write_run(tmp_path, frame, settings, manifest(run_id))

    assert directory.name == run_id
    assert (directory / results.RESULTS_NAME).is_file()
    assert (directory / results.CONFIG_NAME).is_file()
    assert (directory / results.MANIFEST_NAME).is_file()
    assert results.read_sweep(tmp_path).collect().equals(frame)


def test_the_saved_config_is_the_one_that_ran(tmp_path: pathlib.Path) -> None:
    """Dumped from the validated object, never copied from the input file.

    A driver script's overrides never reach the file on disk, so copying it
    would record settings the run did not use — which is the whole reason for
    the convention.
    """
    settings = config.load_config(constants.DEFAULT_CONFIG_PATH)
    overridden = config.Config.model_validate(
        {**settings.model_dump(), "budget": {**settings.budget.model_dump(),
                                             "annual": 1_234.0}}
    )
    frame = results.to_frame(chunk(), "risk_ranked", CLASS_NAMES)

    directory = results.write_run(
        tmp_path, frame, overridden, manifest(results.new_run_id())
    )

    saved = yaml.safe_load((directory / results.CONFIG_NAME).read_text())
    assert saved["budget"]["annual"] == 1_234.0
    assert (
        yaml.safe_load(constants.DEFAULT_CONFIG_PATH.read_text())["budget"]["annual"]
        != 1_234.0
    ), "the override has to differ from the file, or this proves nothing"


def test_the_manifest_records_what_a_result_cannot_be_read_without(
    tmp_path: pathlib.Path,
) -> None:
    """A timing without its build profile and thread count means nothing."""
    settings = config.load_config(constants.DEFAULT_CONFIG_PATH)
    frame = results.to_frame(chunk(), "risk_ranked", CLASS_NAMES)
    run_id = results.new_run_id()

    directory = results.write_run(tmp_path, frame, settings, manifest(run_id))

    saved = json.loads((directory / results.MANIFEST_NAME).read_text())
    assert saved["run_id"] == run_id
    assert saved["implementation"] == "reference"
    assert saved["package_version"] == results.package_version()
    for field in ("git_commit", "git_dirty", "threads", "batch_size",
                  "wall_seconds", "build_profile"):
        assert field in saved, field


def test_an_unknown_implementation_is_refused() -> None:
    """Provenance that reads as fact and is not is worse than none."""
    with pytest.raises(ValueError, match="implementation must be one of"):
        results.Manifest(
            **{**manifest("20260906T120000000000").model_dump(),
               "implementation": "kernal"}
        )


def test_run_identifiers_order_and_do_not_collide_within_a_second() -> None:
    """Ordering runs is what the identifier is for, hence a timestamp.

    The microseconds are what keep two runs started in the same second apart.
    """
    earlier = datetime.datetime(2026, 9, 6, 12, 0, 0, 10, tzinfo=datetime.UTC)
    later = earlier + datetime.timedelta(microseconds=1)

    assert results.new_run_id(earlier) < results.new_run_id(later)
    assert results.new_run_id(earlier) != results.new_run_id(later)


def test_a_sweep_reads_every_run_in_it(tmp_path: pathlib.Path) -> None:
    """A sweep is a directory of runs, concatenated into one frame."""
    settings = config.load_config(constants.DEFAULT_CONFIG_PATH)
    for index in range(3):
        results.write_run(
            tmp_path,
            results.to_frame(chunk(), "risk_ranked", CLASS_NAMES),
            settings,
            manifest(f"2026090{index}T120000000000"),
        )

    collected = results.read_sweep(tmp_path).collect()

    assert collected.height == 3 * chunk().failures.size


def test_a_sweep_missing_a_run_raises_rather_than_returning_a_short_frame(
    tmp_path: pathlib.Path,
) -> None:
    """A curve short one budget level looks fine and is wrong.

    This is the failure the "never silently skip a missing file" convention
    exists for: the reader cannot know a level is missing, and neither can the
    person reading the figure.
    """
    settings = config.load_config(constants.DEFAULT_CONFIG_PATH)
    for index in range(2):
        results.write_run(
            tmp_path,
            results.to_frame(chunk(), "risk_ranked", CLASS_NAMES),
            settings,
            manifest(f"2026090{index}T120000000000"),
        )
    (tmp_path / "20260901T120000000000" / results.RESULTS_NAME).unlink()

    with pytest.raises(FileNotFoundError, match="20260901T120000000000"):
        results.read_sweep(tmp_path)


def test_an_empty_or_absent_sweep_directory_raises(tmp_path: pathlib.Path) -> None:
    """Returning an empty frame would read as a sweep that found nothing."""
    with pytest.raises(FileNotFoundError, match="no sweep directory"):
        results.read_sweep(tmp_path / "not_here")

    with pytest.raises(FileNotFoundError, match="no run directories"):
        results.read_sweep(tmp_path)


def test_git_provenance_reports_unknown_rather_than_inventing_it(
    tmp_path: pathlib.Path,
) -> None:
    """Outside a repository there is no commit, and saying so is the answer."""
    commit, dirty = results.git_provenance(tmp_path)

    assert commit is None
    assert dirty is None


def test_git_provenance_reads_this_repository() -> None:
    """Inside one, it answers with a real commit."""
    commit, dirty = results.git_provenance(constants.PROJECT_ROOT)

    assert commit is not None
    assert len(commit) == 40
    assert isinstance(dirty, bool)
