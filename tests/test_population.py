"""Checks on the synthetic segment table.

The population is reproducible from a seed, so every fixture here is generated
in code and nothing is committed.
"""

import numpy as np
import polars as pl
import pytest
from cablesim import config, constants, population, weibull


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    """The population generated from the checked-in configuration.

    Returns:
        One row per segment.
    """
    return population.generate(config.load_config())


def test_class_shares_match_the_configuration(frame: pl.DataFrame) -> None:
    """Class assignment reproduces the configured shares.

    The draw is multinomial, so the tolerance is three standard errors of the
    sample share rather than an exact match — an exact assertion would fail on
    a correct implementation.
    """
    settings = config.load_config()
    n = settings.population.n_segments
    observed = frame.group_by("class").len().to_dict(as_series=False)
    counts = dict(zip(observed["class"], observed["len"], strict=True))

    for segment_class in settings.population.classes:
        share = segment_class.share
        standard_error = np.sqrt(share * (1 - share) / n)
        assert abs(counts[segment_class.name] / n - share) < 3 * standard_error


def test_customer_counts_are_ordered_by_class_and_by_type(
    frame: pl.DataFrame,
) -> None:
    """Feeders serve more customers than laterals, and residential dominates.

    Failures on main feeders dominate the reliability metrics precisely because
    of this ordering, so it is the property worth pinning rather than any
    particular mean.
    """
    by_class = dict(
        frame.group_by("class")
        .agg(pl.col("customers").mean())
        .iter_rows()
    )
    assert (
        by_class["main_feeder"]
        > by_class["distribution_3ph"]
        > by_class["lateral_1ph"]
    )

    totals = frame.select(["residential", "commercial", "industrial"]).sum().row(0)
    assert totals[0] > totals[1] > totals[2]


def test_technology_covers_the_whole_install_year_range(
    frame: pl.DataFrame,
) -> None:
    """Every segment resolves to exactly one technology, with none left over.

    A vintage gap would leave segments carrying whichever technology index the
    array happened to be initialised with, which is a silent wrong answer.
    """
    settings = config.load_config()
    first, last = settings.population.initial_age.install_year_range

    assert frame["install_year"].min() >= first
    assert frame["install_year"].max() <= last
    assert set(frame["technology"].unique()) == {
        t.name for t in settings.population.technologies
    }

    for technology in settings.population.technologies:
        rows = frame.filter(pl.col("technology") == technology.name)
        low, high = technology.vintage
        assert rows["install_year"].min() >= low
        assert rows["install_year"].max() <= high


def test_generation_is_reproducible_and_seed_dependent() -> None:
    """The same seed gives the same table; a different one does not."""
    settings = config.load_config()
    again = population.generate(settings)

    changed = settings.model_copy(deep=True)
    changed.simulation.seed += 1

    assert population.generate(settings).equals(again)
    assert not population.generate(changed).equals(again)


def test_three_phase_segments_have_a_shorter_effective_scale(
    frame: pl.DataFrame,
) -> None:
    """A three-phase segment is out when any conductor fails, so it lives less.

    Compared within one technology, because the technology carries its own
    scale and would otherwise confound the comparison.
    """
    # Compare at one technology, one class shape and one length, so the only
    # thing differing is the conductor count. Comparing classes directly would
    # confound it with their lengths and with the feeder shape override.
    settings = config.load_config()
    reference = frame.filter(pl.col("technology") == "xlpe").row(0, named=True)
    geometry = (
        np.array(reference["length_ft"]),
        settings.population.length_ref_ft,
        settings.population.length_exponent,
    )
    shape = np.array(reference["shape"])
    scale = np.array(settings.population.technologies[1].weibull.scale)

    single = weibull.effective_scale(shape, scale, np.array(1), *geometry)
    three = weibull.effective_scale(shape, scale, np.array(3), *geometry)

    assert three < single
    assert float(three / single) == pytest.approx(3.0 ** (-1.0 / float(shape)))


def test_derived_columns_follow_from_the_counts(frame: pl.DataFrame) -> None:
    """The four derived columns equal the closed forms the design writes down.

    These are what the kernel reads, so an error here reaches every result
    while leaving the inputs looking correct. Asserting only that they are
    non-negative would pass with the wrong restoration time, or with the
    hours-to-minutes conversion dropped entirely — a sixtyfold error in SAIDI.
    """
    settings = config.load_config()
    types = settings.population.customer_types
    reliability = settings.reliability

    emergency_hours = frame["class"].replace_strict(
        reliability.outage_hours_emergency, return_dtype=pl.Float64
    )
    planned_hours = frame["class"].replace_strict(
        reliability.outage_hours_planned, return_dtype=pl.Float64
    )
    customers = frame.select(types).sum_horizontal()
    value_per_hour = sum(
        frame[name] * reliability.voll_per_customer_hour[name] for name in types
    )

    assert frame["customers"].to_numpy() == pytest.approx(customers.to_numpy())
    assert frame["customer_minutes_per_failure"].to_numpy() == pytest.approx(
        (customers * constants.MINUTES_PER_HOUR * emergency_hours).to_numpy()
    )
    assert frame["customer_minutes_per_planned"].to_numpy() == pytest.approx(
        (customers * constants.MINUTES_PER_HOUR * planned_hours).to_numpy()
    )
    assert frame["outage_cost_per_failure"].to_numpy() == pytest.approx(
        (value_per_hour * emergency_hours).to_numpy()
    )

    # Planned work on a looped feeder interrupts nobody; a radial lateral's
    # customers are out for the whole job.
    by_class = dict(
        frame.group_by("class")
        .agg(pl.col("customer_minutes_per_planned").mean())
        .iter_rows()
    )
    assert by_class["main_feeder"] == 0.0
    assert by_class["lateral_1ph"] > 0.0


def test_attributes_drawn_from_different_uniform_rows_are_independent(
    frame: pl.DataFrame,
) -> None:
    """Length and customer counts do not move together.

    Each attribute reads its own row of the shared uniform block. Reading the
    same row twice would make two attributes exactly comonotonic — every long
    segment also the one with most customers — which changes what a
    consequence-weighted policy is ranking and looks entirely plausible in a
    summary table.
    """
    laterals = frame.filter(pl.col("class") == "lateral_1ph")

    for name in ("residential", "commercial", "industrial"):
        correlation = laterals.select(
            pl.corr("length_ft", name, method="spearman")
        ).item()
        assert abs(correlation) < 0.1, f"length is rank-correlated with {name}"


def test_a_class_shape_override_reaches_the_segments(frame: pl.DataFrame) -> None:
    """A class that names its own shape gets it; the others take technology's.

    Conductor size is a second axis technology does not capture, so feeder
    cable carries its own shape. Dropping the override leaves every class on
    the technology default and changes every feeder's hazard.
    """
    settings = config.load_config()
    overridden = {
        c.name: c.weibull_shape
        for c in settings.population.classes
        if c.weibull_shape is not None
    }
    assert overridden, "no class overrides its shape; this test proves nothing"

    for name, shape in overridden.items():
        rows = frame.filter(pl.col("class") == name)
        assert (rows["shape"] == shape).all()
        assert (rows["replacement_shape"] == shape).all()

    technology_shapes = {
        t.name: t.weibull.shape for t in settings.population.technologies
    }
    plain = frame.filter(~pl.col("class").is_in(list(overridden)))
    expected = plain["technology"].replace_strict(
        technology_shapes, return_dtype=pl.Float64
    )
    assert (plain["shape"] == expected).all()
