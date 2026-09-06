"""The recovery ladder: simulate from known parameters, censor, refit, check.

Each rung adds exactly one thing that can be wrong. A single test of the whole
model says something is broken; the ladder says what. Rungs are asserted by
requiring the truth to fall inside the fitted 95% interval at a pinned seed,
and every parameter in a rung must contain its truth simultaneously — stricter
than each separately, and safe only because the seed is fixed, which turns a
coverage question into a check that either passes or reveals a real error.
"""

import numpy as np
import pytest
from cablesim import config, records, weibull

SEGMENTS: int = 6000
"""Sized so each rung's fit has a few hundred observed failures.

Heavy censoring is the point rather than a nuisance — most cable never fails
inside the window — so the sample is set by how many failures come out, not by
how many segments go in.
"""


def one_technology_config() -> config.Config:
    """A configuration with nothing truncated and one technology in play.

    Monitoring starts at the first install year, so every episode is watched
    from installation and the truncation term contributes nothing. That is what
    isolates the censored likelihood in the first rung.

    Returns:
        A copy of the checked-in configuration, adjusted.
    """
    settings = config.load_config()
    settings.records.monitoring_start = (
        settings.population.initial_age.install_year_range[0]
    )
    return settings


def contains(interval: tuple[float, float], truth: float) -> bool:
    """Whether a fitted interval covers the value the data was generated from.

    Args:
        interval: Lower and upper bound.
        truth: The generating value.

    Returns:
        True if the truth lies inside.
    """
    return interval[0] < truth < interval[1]


def test_rung_1_recovers_from_right_censored_data() -> None:
    """The censored likelihood itself, with nothing else varying.

    One technology, every segment at the reference length with one conductor,
    so the effective-scale reduction is the identity and the fitted scale is
    the configured one. Nothing is truncated. A failure here is the likelihood
    or the optimizer, and cannot be anything else.
    """
    settings = one_technology_config()
    technology = settings.population.technologies[0]

    table = records.episode_table(
        settings,
        technologies=[technology],
        length_ft=settings.population.length_ref_ft,
        n_conductors=[1],
        n_segments=SEGMENTS,
    )
    end, entry, observed = records.lifetimes(table, settings.records.study_end)
    assert observed.sum() > 200, "too few failures to identify the shape"
    assert (entry == 0).all(), "this rung must not be truncated"

    fit = weibull.fit_censored(end, entry, observed)

    assert contains(fit.shape_interval, technology.weibull.shape)
    assert contains(fit.scale_interval, technology.weibull.scale)


@pytest.mark.parametrize("monitoring_start", [1985, 1998, 2010])
def test_rung_2_recovers_through_left_truncation(monitoring_start: int) -> None:
    """The truncation correction, at three entry ages spanning the range.

    Cable that failed before monitoring began is absent from the table, so the
    population that survived to be recorded is not the population installed.
    Ignoring that biases the fit toward longer life, and the bias grows with
    entry age — a single early monitoring date passes with the correction
    missing altogether, which is why this runs at three.
    """
    settings = one_technology_config()
    settings.records.monitoring_start = monitoring_start
    technology = settings.population.technologies[0]

    table = records.episode_table(
        settings,
        technologies=[technology],
        length_ft=settings.population.length_ref_ft,
        n_conductors=[1],
        n_segments=SEGMENTS,
    )
    end, entry, observed = records.lifetimes(table, settings.records.study_end)
    assert (entry > 0).any(), f"nothing truncated at {monitoring_start}"

    fit = weibull.fit_censored(end, entry, observed)

    assert contains(fit.shape_interval, technology.weibull.shape)
    assert contains(fit.scale_interval, technology.weibull.scale)


def test_the_truncation_correction_matters_when_entry_ages_are_late() -> None:
    """Dropping the correction moves the answer — but only when entry is late.

    Both halves are asserted because both are true and the second is easy to
    assume away. `H(a)` is `(a/scale)**shape`, so at a sharp shape an episode
    that entered observation early carries almost no accumulated hazard and the
    correction is worth a fraction of a percent. It bites when entry ages
    approach the scale.

    The design used to claim the correction was required outright. It is
    required for a record system that begins well after the installs, and
    nearly free otherwise, which is worth pinning so nobody removes it after
    measuring only the configured case.
    """
    settings = one_technology_config()
    technology = settings.population.technologies[0]
    # Read before the helper below mutates it, or the second call repeats the
    # first with the same window.
    configured = config.load_config().records.monitoring_start

    def fitted(monitoring_start: int) -> tuple[weibull.Fit, weibull.Fit]:
        """Fits the same data with and without the truncation term.

        Args:
            monitoring_start: The year failure records begin.

        Returns:
            The corrected fit and the one that ignores entry ages.
        """
        settings.records.monitoring_start = monitoring_start
        table = records.episode_table(
            settings,
            technologies=[technology],
            length_ft=settings.population.length_ref_ft,
            n_conductors=[1],
            n_segments=SEGMENTS,
        )
        end, entry, observed = records.lifetimes(table, settings.records.study_end)
        return (
            weibull.fit_censored(end, entry, observed),
            weibull.fit_censored(end, np.zeros_like(entry), observed),
        )

    # Late record start: the correction is load-bearing.
    corrected, ignored = fitted(2018)
    assert contains(corrected.scale_interval, technology.weibull.scale)
    assert not contains(ignored.scale_interval, technology.weibull.scale)
    assert ignored.scale > corrected.scale, "ignoring it must inflate the life"

    # Configured record start: the same term is worth well under one percent.
    corrected, ignored = fitted(configured)
    assert abs(ignored.scale / corrected.scale - 1.0) < 0.01


def test_a_table_with_no_failures_is_rejected() -> None:
    """Without a failure the shape is unidentified, and the fit says so."""
    ages = np.full(50, 10.0)

    with pytest.raises(ValueError, match="not identified"):
        weibull.fit_censored(ages, np.zeros_like(ages), np.zeros_like(ages))
