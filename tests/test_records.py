"""Checks on the synthetic failure history the MLE fits.

The table this builds is what every rung of the recovery ladder is fitted to,
so an error here would be attributed to the likelihood rather than to the data.
Each check below is a property the likelihood depends on.
"""

import numpy as np
import polars as pl
import pytest
from cablesim import config, records


@pytest.fixture(scope="module")
def settings() -> config.Config:
    """The checked-in configuration.

    Returns:
        The validated run configuration.
    """
    return config.load_config()


@pytest.fixture(scope="module")
def table(settings: config.Config) -> pl.DataFrame:
    """A modest record table, big enough for the structural checks.

    Returns:
        One row per installation episode.
    """
    return records.episode_table(settings, n_segments=3000)


def test_a_segment_that_fails_contributes_more_than_one_row(
    table: pl.DataFrame,
) -> None:
    """Replacement chains, so the table is longer than the segment count.

    One row per segment would either discard the failure or measure the second
    installation's life from the first one's install date, and both bias the
    fit toward longer life.
    """
    per_segment = table.group_by("segment_id").len()

    assert table.height > per_segment.height
    assert per_segment["len"].max() > 1


def test_chained_episodes_start_where_the_previous_one_ended(
    table: pl.DataFrame,
) -> None:
    """A replacement is installed at the moment the cable it replaces failed.

    A gap or an overlap would make the second episode's age wrong by that
    amount, silently, since the row would still look well formed.
    """
    chained = (
        table.sort("segment_id", "install_year")
        .with_columns(
            pl.col("failure_year").shift(1).over("segment_id").alias("previous_end")
        )
        .filter(pl.col("previous_end").is_not_null())
    )

    assert chained.height > 0, "no segment chained; this test proves nothing"
    assert chained["install_year"].to_numpy() == pytest.approx(
        chained["previous_end"].to_numpy()
    )


def test_lifetimes_are_continuous(table: pl.DataFrame) -> None:
    """Years are fractional, because the likelihood is continuous-time.

    Rounding to whole years would make this interval-censored data fitted with
    a continuous likelihood, and the first rung of the recovery ladder would
    fail for a reason that is not a bug in the code.
    """
    observed = table.filter(pl.col("failure_year").is_not_null())

    fractional = observed["failure_year"].to_numpy() % 1.0
    assert (fractional > 0).mean() > 0.9


def test_censored_episodes_carry_no_failure_year(table: pl.DataFrame) -> None:
    """Still-in-service episodes are null, not a sentinel.

    A NaN would read as an observed failure to anything asking whether the
    column is null, which is how the censoring indicator gets built.
    """
    censored = table.filter(pl.col("failure_year").is_null())

    assert censored.height > 0
    assert not table["failure_year"].is_nan().any()


def test_the_observation_window_is_applied(
    table: pl.DataFrame, settings: config.Config
) -> None:
    """Entry is the later of install and monitoring start, and nothing precedes it.

    An episode that ended before monitoring began was never observable.
    Keeping it would add a row with no exposure after entry, which contributes
    nothing to the likelihood while inflating the apparent sample size.
    """
    start = settings.records.monitoring_start

    assert (table["entry_year"] >= start).all()
    assert table["entry_year"].to_numpy() == pytest.approx(
        np.maximum(table["install_year"].to_numpy(), float(start))
    )

    observed = table.filter(pl.col("failure_year").is_not_null())
    assert (observed["failure_year"] > start).all()
    assert (observed["failure_year"] < settings.records.study_end).all()
    assert (observed["failure_year"] > observed["entry_year"]).all()


def test_a_single_technology_removes_the_other_sources_of_variation(
    settings: config.Config,
) -> None:
    """The overrides give the early rungs data with one thing varying.

    Rungs one and two isolate the censored likelihood and the truncation
    correction, so length and conductor count have to be held fixed or a
    failure there could be either.
    """
    one = settings.population.technologies[0]
    table = records.episode_table(
        settings,
        technologies=[one],
        length_ft=500.0,
        n_conductors=[1],
        n_segments=800,
    )

    assert set(table["technology"].unique()) == {one.name}
    assert set(table["n_conductors"].unique()) == {1}
    assert table["length_ft"].n_unique() == 1


def test_generation_is_reproducible_and_seed_dependent(
    settings: config.Config,
) -> None:
    """The same configuration gives the same table; an offset gives another."""
    again = records.episode_table(settings, n_segments=500)
    shifted = records.episode_table(settings, n_segments=500, seed_offset=1)

    assert records.episode_table(settings, n_segments=500).equals(again)
    assert not shifted.equals(again)


def test_implausibly_short_lifetimes_raise_rather_than_truncate(
    settings: config.Config,
) -> None:
    """Running out of episodes is reported, not silently cut short.

    A history truncated at the cap would look like data and fit like data, so
    the guard raises instead. Reaching it means the drawn lifetimes are short
    against the window, not that the history is genuinely long.
    """
    absurd = settings.model_copy(deep=True)
    for technology in absurd.population.technologies:
        technology.weibull.scale = 0.5

    with pytest.raises(ValueError, match="still failing after"):
        records.episode_table(absurd, n_segments=200)


def test_a_technology_with_no_observed_failure_is_refused(
    table: pl.DataFrame,
) -> None:
    """An all-censored technology has no estimable scale, so coding it raises.

    Its rows enter the likelihood only through accumulated hazard, which falls
    as the scale grows without ever turning back, so the maximum sits at
    infinity and a solver stops wherever its tolerance runs out. That returns a
    finite, ordinary-looking number with nothing behind it, which is worse than
    a refusal because nothing about it invites a second look.
    """
    censored = table.with_columns(
        pl.when(pl.col("technology") == "xlpe")
        .then(None)
        .otherwise(pl.col("failure_year"))
        .alias("failure_year")
    )
    with pytest.raises(ValueError, match="no observed failure.*xlpe"):
        records.technology_indicators(censored, reference="hmwpe")


def test_technologies_that_do_fail_are_coded_against_the_reference(
    table: pl.DataFrame,
) -> None:
    """The reference is left without a column and the rest get one each.

    An intercept alongside an indicator for every level would be rank
    deficient, so the reference carries no column of its own and each
    coefficient reads as a log ratio against it.
    """
    design, names = records.technology_indicators(table, reference="hmwpe")

    assert names == ["technology_tr_xlpe", "technology_xlpe"]
    assert design.shape == (table.height, 2)
    # Every row belongs to exactly one technology, so a row is either the
    # reference, coded as zeros, or carries a single one.
    assert set(design.sum(axis=1)) <= {0.0, 1.0}
    assert (design.sum(axis=1) == 0.0).sum() == (table["technology"] == "hmwpe").sum()
