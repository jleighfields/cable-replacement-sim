"""The recovery ladder: simulate from known parameters, censor, refit, check.

Each rung adds exactly one thing that can be wrong. A single test of the whole
model says something is broken; the ladder says what. Rungs are asserted by
requiring the truth to fall inside the fitted 95% interval at a pinned seed,
and every parameter in a rung must contain its truth simultaneously — stricter
than each separately, and safe only because the seed is fixed, which turns a
coverage question into a check that either passes or reveals a real error.
"""

import numpy as np
import polars as pl
import pytest
from cablesim import config, records, weibull
from scipy import optimize, stats

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


def equal_exposure_config(shapes: list[float], scales: list[float]) -> config.Config:
    """A configuration where every technology has been in the ground as long.

    Technology follows install year, so in the shipped population the newest
    one is the youngest cable and produces almost no failures — ten in a table
    of 180,000 segments. That is a property of the population rather than of
    the estimator, and raising the sample does not fix it, because the binding
    constraint is exposure time and not sample size.

    Rungs four and five isolate the technology terms, so they need every
    technology to have comparable exposure. A narrow, old install range split
    three ways gives that: every episode is at least fifty years old by the
    study end, whichever technology it carries.

    Args:
        shapes: Weibull shape per technology, in configured order.
        scales: Weibull scale per technology, in configured order.

    Returns:
        A configuration with nothing truncated and the vintages rearranged.
    """
    raw = config.load_config().model_dump()
    raw["population"]["initial_age"]["install_year_range"] = (1965, 1976)
    raw["population"]["initial_age"]["install_volume"] = {1965: 1.0, 1976: 1.0}
    vintages = [(1965, 1968), (1969, 1972), (1973, 1976)]
    for technology, vintage, shape, scale in zip(
        raw["population"]["technologies"], vintages, shapes, scales, strict=True
    ):
        technology["vintage"] = vintage
        technology["weibull"]["shape"] = shape
        technology["weibull"]["scale"] = scale
    raw["records"]["monitoring_start"] = 1965
    return config.Config.model_validate(raw)


def fitted_table(settings: config.Config, n_segments: int):
    """Generates a record table and the arrays a fit needs from it.

    Args:
        settings: The configuration to generate against.
        n_segments: Segments to draw.

    Returns:
        The table, then age at end, entry age and the failure indicator.
    """
    table = records.episode_table(
        settings,
        length_ft=settings.population.length_ref_ft,
        n_conductors=[1],
        n_segments=n_segments,
    )
    return (table, *records.lifetimes(table, settings.records.study_end))


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


def test_rung_3_recovers_the_geometry_coefficients() -> None:
    """The weakest-link law, tested by fitting rather than assumed.

    This is the rung the phase exists for. The effective-scale reduction is
    derived, not fitted, so both geometry coefficients have predicted values:
    `log(n)` must come back at `-1/shape`, which follows from the minimum of n
    conductor lifetimes and cannot be tuned, and `log(L/L_ref)` at
    `-length_exponent/shape`, which checks that the generator and the estimator
    agree about the exponent the configuration sets.

    Failing here means the reduction or the regression specification is wrong,
    and every rung above inherits it — so this is where to stop rather than
    press on.
    """
    settings = one_technology_config()
    technology = settings.population.technologies[0]
    shape, scale = technology.weibull.shape, technology.weibull.scale
    exponent = settings.population.length_exponent

    table = records.episode_table(
        settings,
        technologies=[technology],
        n_conductors=[1, 3],
        n_segments=20_000,
    )
    end, entry, observed = records.lifetimes(table, settings.records.study_end)
    design, names = records.geometry_covariates(
        table, settings.population.length_ref_ft
    )
    assert observed.sum() > 1000, "too few failures to separate two coefficients"

    fit = weibull.fit_regression(end, entry, observed, design, names)

    assert contains(fit.shape_interval, shape)
    assert contains(fit.reference_scale_interval, scale)
    assert contains(fit.coefficient_intervals["log_n_conductors"], -1.0 / shape)
    assert contains(fit.coefficient_intervals["log_length_ratio"], -exponent / shape)


def test_one_composite_covariate_recovers_neither_effect() -> None:
    """Why the design matrix carries two columns rather than one.

    Conductor count enters the reduction exactly and length raised to the
    configured exponent, so a single column combining them forces one
    coefficient onto two effects at different strengths. The fitted value lands
    between them and matches neither, which is a plausible-looking number and
    the reason this is pinned rather than left as a comment.
    """
    settings = one_technology_config()
    technology = settings.population.technologies[0]
    shape = technology.weibull.shape
    exponent = settings.population.length_exponent

    table = records.episode_table(
        settings,
        technologies=[technology],
        n_conductors=[1, 3],
        n_segments=20_000,
    )
    end, entry, observed = records.lifetimes(table, settings.records.study_end)
    design, _ = records.geometry_covariates(table, settings.population.length_ref_ft)

    composite = (design[:, 0] + design[:, 1]).reshape(-1, 1)
    fit = weibull.fit_regression(end, entry, observed, composite, ["log_composite"])
    interval = fit.coefficient_intervals["log_composite"]

    assert not contains(interval, -1.0 / shape)
    assert not contains(interval, -exponent / shape)


def test_a_table_with_no_failures_is_rejected() -> None:
    """Without a failure the shape is unidentified, and the fit says so."""
    ages = np.full(50, 10.0)

    with pytest.raises(ValueError, match="not identified"):
        weibull.fit_censored(ages, np.zeros_like(ages), np.zeros_like(ages))


def test_rung_4_recovers_a_scale_per_technology_at_a_common_shape() -> None:
    """The technology indicators, with the shape held common.

    Reference coding: one technology carries no column and the fitted intercept
    is its scale, with each remaining coefficient a log ratio against it. An
    intercept plus an indicator for every level would be rank-deficient, and a
    solver would either fail or return one of infinitely many equally good
    answers.
    """
    shape = 5.5
    scales = [45.0, 55.0, 68.0]
    settings = equal_exposure_config([shape] * 3, scales)
    table, end, entry, observed = fitted_table(settings, 12_000)

    design, names = records.technology_indicators(table, reference="hmwpe")
    per_technology = dict(
        table.filter(pl.col("failure_year").is_not_null())
        .group_by("technology")
        .len()
        .iter_rows()
    )
    assert min(per_technology.values()) > 200, per_technology

    fit = weibull.fit_regression(end, entry, observed, design, names)

    assert contains(fit.shape_interval, shape)
    assert contains(fit.reference_scale_interval, scales[0])
    for name, scale in zip(names, [scales[2], scales[1]], strict=True):
        low, high = fit.coefficient_intervals[name]
        assert contains((np.exp(low), np.exp(high)), scale / scales[0])


def test_rung_5_recovers_a_shape_per_technology() -> None:
    """Shape as an ancillary term, which rung 4 cannot reach.

    A common-shape fit assumes every technology's hazard rises at the same
    rate. Where it does not, that fit recovers a compromise and biases every
    scale with it — and the failure presents as a tolerance problem rather than
    as the misspecification it is, which is why this rung is separate.
    """
    shapes = [4.0, 5.5, 7.0]
    scales = [45.0, 55.0, 68.0]
    settings = equal_exposure_config(shapes, scales)
    table, end, entry, observed = fitted_table(settings, 12_000)
    design, names = records.technology_indicators(table, reference="hmwpe")

    fit = weibull.fit_regression(
        end, entry, observed, design, names, ancillary=design, ancillary_names=names
    )

    assert contains(fit.shape_interval, shapes[0])
    assert contains(fit.reference_scale_interval, scales[0])
    # Columns come back alphabetically: tr_xlpe then xlpe.
    for name, shape in zip(names, [shapes[2], shapes[1]], strict=True):
        low, high = fit.ancillary_intervals[name]
        assert contains((np.exp(low), np.exp(high)), shape / shapes[0])


def test_a_common_shape_fit_is_biased_when_the_shape_varies() -> None:
    """Why rungs 4 and 5 are separate rather than one test with an option.

    Fitting the common-shape model to data whose shapes differ returns a value
    between them that matches none, and drags the scales with it. Nothing about
    the output says the model was wrong.
    """
    shapes = [4.0, 5.5, 7.0]
    scales = [45.0, 55.0, 68.0]
    settings = equal_exposure_config(shapes, scales)
    table, end, entry, observed = fitted_table(settings, 12_000)
    design, names = records.technology_indicators(table, reference="hmwpe")

    common = weibull.fit_regression(end, entry, observed, design, names)
    varying = weibull.fit_regression(
        end, entry, observed, design, names, ancillary=design, ancillary_names=names
    )

    assert not contains(common.shape_interval, shapes[0])
    assert contains(varying.shape_interval, shapes[0])
    assert varying.log_likelihood > common.log_likelihood


WELL_DETERMINED_SEGMENTS: int = 15_000
"""Sized so every parameter comes back tightly enough to reason about.

An estimate that is barely identified agrees with almost anything, so a check
made against a loose fit passes whatever it is compared with. Larger than the
ladder's own sample for that reason alone.
"""


def well_determined_data(
    settings: config.Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build one lifetime table with every parameter tightly determined.

    One technology and both conductor counts, so a fit of it has a shape, a
    reference scale and two geometry coefficients to recover at once.

    Args:
        settings: Supplies the technology, the reference length and the study
            window. Whether episodes are truncated follows from its
            `records.monitoring_start`.

    Returns:
        Age at the end of observation, age at entry, whether that end was a
        failure, the covariate matrix, and the covariate names.
    """
    table = records.episode_table(
        settings,
        technologies=[settings.population.technologies[0]],
        n_conductors=[1, 3],
        n_segments=WELL_DETERMINED_SEGMENTS,
    )
    end, entry, observed = records.lifetimes(table, settings.records.study_end)
    design, names = records.geometry_covariates(
        table, settings.population.length_ref_ft
    )
    return end, entry, observed, design, names


def test_the_fit_does_not_depend_on_where_the_search_starts() -> None:
    """The optimum is reached from starting values far from it.

    `fit_censored` starts at an exponential -- shape one, scale the mean age at
    the end of observation -- and that choice is only safe if the surface has a
    single optimum to find. It is not a detail that can be left untested: a
    start derived instead from a least-squares fit that treats censored rows as
    failures walks away from the optimum on data this heavily censored, where
    around nine rows in ten are censored, and settles at roughly twice the true
    shape.

    Nelder-Mead is used rather than the fit's own BFGS, so nothing about how
    the optimizer steps -- its finite differences, its curvature estimate --
    can be what carries every start to the same place. What is left is the
    surface.
    """
    settings = one_technology_config()
    end, entry, observed, _, _ = well_determined_data(settings)
    reference = weibull.fit_censored(end, entry, observed)

    def negative(logged: np.ndarray) -> float:
        shape, scale = np.exp(logged)
        return -weibull.log_likelihood(shape, scale, end, entry, observed)

    for shape in (0.2, 3.0, 40.0):
        for scale in (1.0, 200.0, 5000.0):
            found = optimize.minimize(
                negative,
                np.log([shape, scale]),
                method="Nelder-Mead",
                options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20_000},
            )
            assert found.success
            assert np.allclose(np.exp(found.x), [reference.shape, reference.scale])


def test_a_late_record_start_is_fitted_rather_than_refused() -> None:
    """A pinned draw the fit refuses while a finite optimum exists.

    The objective is `nan` wherever the log-likelihood is, which happens as
    soon as the search steps somewhere the accumulated hazard overflows -- a
    tiny scale against a large shape. `nan` compares false against every
    bound, so the line search cannot reject the step; it walks out to an
    infinite shape and a zero scale, and `fit_censored` raises `the fit did
    not converge`. The message points at the data, and the data is fine.

    Returning positive infinity instead of `nan` for a non-finite objective
    makes the step rejectable, and the same search then lands on the optimum
    below. It changes nothing where the fit already converges, because the
    substitution only applies outside the region the likelihood is defined on.

    A late record start is what makes the region reachable: entry ages
    approach the scale, so the surface near the start is flat enough for the
    first steps to be long. Measured over sixty consecutive seeds at this
    size, two fail at a 2018 record start and five at 2020, against none at
    the configured 1998 -- so the pinned seed here is one of a class, not a
    curiosity.
    """
    settings = config.load_config()
    settings.simulation.seed = 20260908
    settings.records.monitoring_start = 2018
    table = records.episode_table(
        settings,
        technologies=[settings.population.technologies[0]],
        length_ft=settings.population.length_ref_ft,
        n_conductors=[1],
        n_segments=6000,
    )
    end, entry, observed = records.lifetimes(table, settings.records.study_end)
    assert observed.sum() > 100, "too few failures for this draw to mean anything"

    # The answer the fit should reach, found without derivatives so that no
    # property of the line search under test can be what produced it.
    def negative(logged: np.ndarray) -> float:
        shape, scale = np.exp(logged)
        return -weibull.log_likelihood(shape, scale, end, entry, observed)

    reference = optimize.minimize(
        negative,
        np.log([1.0, float(np.mean(end))]),
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20_000},
    )
    assert reference.success

    fitted = weibull.fit_censored(end, entry, observed)

    # Loose against the two searches' own stopping rules and tight against the
    # question: the fitted shape's 95% interval is about 0.4 wide here, so a
    # thousandth is well inside the noise the fit is entitled to and nowhere
    # near the infinite shape the unguarded search walks out to.
    assert fitted.shape == pytest.approx(float(np.exp(reference.x[0])), rel=1e-3)
    assert fitted.scale == pytest.approx(float(np.exp(reference.x[1])), rel=1e-3)


def test_the_reported_likelihood_is_the_summed_one() -> None:
    """Averaging is an optimizer detail and must not reach the result.

    The objective is divided by the episode count so that convergence means the
    same thing at every sample size, which would silently divide the reported
    log-likelihood too. It is compared against models fitted elsewhere, so it
    has to stay the sum.
    """
    settings = one_technology_config()
    end, entry, observed, _, _ = well_determined_data(settings)
    fitted = weibull.fit_censored(end, entry, observed)

    assert fitted.log_likelihood == pytest.approx(
        weibull.log_likelihood(fitted.shape, fitted.scale, end, entry, observed)
    )


def test_a_fit_carries_no_shared_mutable_default() -> None:
    """`RegressionFit` must give no field a mutable default.

    A NamedTuple evaluates a field default once, at class creation, so a `{}`
    default is a single dictionary shared by every instance built without it:
    writing through one instance is then visible through all the others. Ruff's
    B006 catches that on a function argument and does not reach a class
    attribute, so nothing but this notices.

    Asserting the absence of defaults rather than constructing two instances
    and writing through one, because with the defaults gone the class cannot be
    built without them and that construction no longer compiles. This states
    the property that keeps it safe.
    """
    assert not weibull.RegressionFit._field_defaults


def test_the_regression_likelihood_is_the_summed_one() -> None:
    """The mirror of the check above, for the fit that carries covariates.

    `fit_regression` averages its objective over episodes for the same reason
    `fit_censored` does, and multiplies the reported value back by the same
    count. Nothing watched that: dividing the reported likelihood by the
    episode count leaves every recovery test green, because the one test
    comparing two regression likelihoods compares two values scaled
    identically, and the difference survives. The error would be a factor of
    the episode count, which is thousands here.

    Both parameterisations are checked. Where the shape is common the model is
    one shape and a scale per episode; where an ancillary design is given the
    shape varies too, and that path is exercised by less than the other.
    """
    settings = one_technology_config()
    end, entry, observed, design, names = well_determined_data(settings)

    common = weibull.fit_regression(end, entry, observed, design, names)
    scales = common.reference_scale * np.exp(
        design @ np.array([common.coefficients[name] for name in names])
    )
    assert common.log_likelihood == pytest.approx(
        weibull.log_likelihood(common.shape, scales, end, entry, observed)
    )

    varying = weibull.fit_regression(
        end, entry, observed, design, names, ancillary=design, ancillary_names=names
    )
    # The ancillary coefficients are on the log scale, so the shape where a
    # covariate is one is the reference shape times the exponent of its
    # coefficient.
    shapes = varying.shape * np.exp(
        design @ np.array([varying.ancillary_coefficients[name] for name in names])
    )
    scales = varying.reference_scale * np.exp(
        design @ np.array([varying.coefficients[name] for name in names])
    )
    assert varying.log_likelihood == pytest.approx(
        weibull.log_likelihood(shapes, scales, end, entry, observed)
    )


def test_the_reported_intervals_match_the_curvature_they_claim_to_measure() -> None:
    """A confidence interval must be the width its level implies.

    Every rung asserts that the truth falls inside a fitted 95% interval, and
    widening an interval only makes that easier: a fit reporting intervals
    twice as wide as they should be passes all of them more comfortably than
    the honest one. So nothing in the suite that gates a merge notices if the
    widths are wrong, and the intervals now come from the inverse Hessian a
    derivative-free BFGS accumulates, which is an approximation rather than a
    derivation.

    Checked against the curvature directly, with no Monte Carlo: the observed
    information is the second derivative of the summed log-likelihood at the
    optimum, built here by central differences on the parameters the fit works
    in, and its inverse gives the standard errors an interval of a stated level
    must be built from. Five percent of tolerance covers what the optimizer's
    approximation costs, measured at one to three percent.

    The level itself is checked by refitting at another one, because a fit that
    ignored `level` and always reported 95% would satisfy every assertion
    above.
    """
    settings = one_technology_config()
    end, entry, observed, _, _ = well_determined_data(settings)
    fitted = weibull.fit_censored(end, entry, observed)

    optimum = np.log([fitted.shape, fitted.scale])

    def summed(logged: np.ndarray) -> float:
        """The log-likelihood the intervals describe, in the fitted parameters."""
        return weibull.log_likelihood(*np.exp(logged), end, entry, observed)

    step = 1e-5
    information = np.empty((2, 2))
    for row in range(2):
        for column in range(2):
            forward, back = np.zeros(2), np.zeros(2)
            forward[row] = back[column] = step
            information[row, column] = -(
                summed(optimum + forward + back)
                - summed(optimum + forward - back)
                - summed(optimum - forward + back)
                + summed(optimum - forward - back)
            ) / (4.0 * step * step)
    errors = np.sqrt(np.diag(np.linalg.inv(information)))

    # The intervals are exponentiated from the log scale, so their half-widths
    # are recovered there rather than on the parameter itself.
    for index, interval in enumerate((fitted.shape_interval, fitted.scale_interval)):
        low, high = np.log(interval)
        expected = stats.norm.ppf(0.975) * errors[index]
        assert (high - low) / 2.0 == pytest.approx(expected, rel=0.05)

    # A fit ignoring `level` would report the same width whatever it was asked
    # for, so the ratio of two levels' half-widths is what pins it.
    narrower = weibull.fit_censored(end, entry, observed, level=0.5)
    for wide, narrow in (
        (fitted.shape_interval, narrower.shape_interval),
        (fitted.scale_interval, narrower.scale_interval),
    ):
        wide_low, wide_high = np.log(wide)
        narrow_low, narrow_high = np.log(narrow)
        assert (wide_high - wide_low) / (narrow_high - narrow_low) == pytest.approx(
            stats.norm.ppf(0.975) / stats.norm.ppf(0.75), rel=0.01
        )
