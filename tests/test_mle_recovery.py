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


CROSS_CHECK_SEGMENTS: int = 15_000
"""Sized so the cross-checks compare well-determined parameters.

An estimate that is barely identified agrees with anything, so a cross-check on
a loose fit passes whatever the other implementation computes. This is larger
than the ladder's own sample for that reason alone.
"""


def cross_check_data(
    settings: config.Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build one lifetime table for an outside implementation to refit.

    One technology and both conductor counts, so the fit has a shape, a
    reference scale and two geometry coefficients to recover at once.

    Args:
        settings: Supplies the technology, the reference length and the study
            window. Whether episodes are truncated is decided by its
            `records.monitoring_start`, which is what separates the two
            cross-checks that call this.

    Returns:
        Age at the end of observation, age at entry, whether that end was a
        failure, the covariate matrix, and the covariate names.
    """
    table = records.episode_table(
        settings,
        technologies=[settings.population.technologies[0]],
        n_conductors=[1, 3],
        n_segments=CROSS_CHECK_SEGMENTS,
    )
    end, entry, observed = records.lifetimes(table, settings.records.study_end)
    design, names = records.geometry_covariates(
        table, settings.population.length_ref_ft
    )
    return end, entry, observed, design, names


def truncating_config() -> config.Config:
    """The checked-in configuration, whose monitoring window truncates.

    The companion to `one_technology_config`, which moves monitoring back to
    the first install year so nothing is truncated. Here monitoring starts long
    after installation began, so a share of episodes enter partway through life
    and the truncation term of the likelihood carries weight.

    Returns:
        The configuration as checked in.
    """
    return config.load_config()


def standard_error(interval: tuple[float, float]) -> float:
    """Recover the standard error a 95% confidence interval was built from.

    The cross-checks compare against a fraction of this rather than a fixed
    epsilon, so the tolerance tracks how well the data determines each
    parameter instead of being tuned to one dataset.

    Args:
        interval: Lower and upper bound of a 95% interval.

    Returns:
        The half-width divided by the normal quantile it was scaled by.
    """
    low, high = interval
    return (high - low) / (2.0 * stats.norm.ppf(0.975))


def assert_agrees(
    mine: weibull.RegressionFit, theirs: dict[str, float], names: list[str]
) -> None:
    """Require an outside fit to match every parameter of `mine`.

    To a tenth of a standard error, on point estimates. Comparing intervals
    would test the two implementations' standard-error machinery rather than
    the likelihood they both claim to maximise, and comparing every parameter
    at once is stricter than comparing them one at a time.

    Args:
        mine: The fit from this package.
        theirs: Parameter name to estimate, using the names `cross_check.R`
            prints -- `shape`, `log_scale`, and one per covariate.
        names: The covariate names, in the order they were fitted.

    Raises:
        AssertionError: If any parameter differs by more than the tolerance.
    """

    def close(
        mine_value: float, theirs_value: float, interval: tuple[float, float]
    ) -> bool:
        """Whether two estimates agree to within a tenth of a standard error."""
        return abs(mine_value - theirs_value) < 0.1 * standard_error(interval)

    assert close(mine.shape, theirs["shape"], mine.shape_interval)
    assert close(
        np.log(mine.reference_scale),
        theirs["log_scale"],
        tuple(np.log(mine.reference_scale_interval)),
    )
    for name in names:
        assert close(
            mine.coefficients[name], theirs[name], mine.coefficient_intervals[name]
        )


def lifelines_estimates(
    end: np.ndarray,
    entry: np.ndarray,
    observed: np.ndarray,
    design: np.ndarray,
    names: list[str],
) -> dict[str, float]:
    """Fit the same table with `lifelines` and return comparable estimates.

    The parameterisation bridge is applied here rather than recalled.
    `lifelines` names the scale `lambda_` and the shape `rho_`, regresses the
    log of the first on the covariates and holds the second constant, which is
    this model exactly -- but it reports both as logs, so the shape is
    exponentiated and the scale is not.

    Args:
        end: Age when observation of each episode ended.
        entry: Age when each episode came under observation. `lifelines` takes
            this as `entry_col`, which is how it is told about truncation.
        observed: Whether each end was a failure rather than the study closing.
        design: Covariate matrix, one row per episode.
        names: Covariate names, in column order.

    Returns:
        Estimates keyed `shape`, `log_scale`, and one entry per covariate --
        the naming `assert_agrees` compares against.
    """
    pandas = pytest.importorskip("pandas")
    fitters = pytest.importorskip("lifelines")

    frame = pandas.DataFrame(
        {
            "T": end,
            "E": observed,
            "entry": entry,
            **dict(zip(names, design.T, strict=True)),
        }
    )
    fitted = fitters.WeibullAFTFitter().fit(
        frame, duration_col="T", event_col="E", entry_col="entry"
    )
    return {
        "shape": float(np.exp(fitted.params_["rho_"]["Intercept"])),
        "log_scale": float(fitted.params_["lambda_"]["Intercept"]),
        **{name: float(fitted.params_["lambda_"][name]) for name in names},
    }


def test_lifelines_agrees_without_truncation() -> None:
    """`lifelines` fits the same data and must reach the same answer.

    Every rung above uses one likelihood on both sides, so a parameterisation
    transposed between the accelerated-failure-time form and the hazard form
    would pass all of them: the generator and the fit would agree with each
    other while both differed from what the configuration means. A second
    implementation is the only thing that catches it, and the cost of that
    correspondence being false is every number in the project being quietly
    wrong.
    """
    settings = one_technology_config()
    end, entry, observed, design, names = cross_check_data(settings)
    assert not entry.any(), "this cross-check is the untruncated one"

    mine = weibull.fit_regression(end, entry, observed, design, names)
    assert_agrees(mine, lifelines_estimates(end, entry, observed, design, names), names)


def test_lifelines_agrees_under_truncation() -> None:
    """The same agreement where a quarter of the episodes enter partway through.

    The other cross-checks all run on data watched from installation, which
    leaves the entry term of the likelihood multiplied by nothing -- an error in
    it would not show. Here monitoring starts long after installation began, so
    the term carries weight and an outside implementation has to reproduce its
    effect.

    Not every survival implementation can be used for this: some express
    right censoring but have no way to state a delayed entry age, and given
    this table they would quietly fit the untruncated model instead. The
    measured cost of that mistake on this data is a shape biased upward by
    about one percent.
    """
    settings = truncating_config()
    end, entry, observed, design, names = cross_check_data(settings)
    assert entry.any(), "this cross-check requires truncated data"

    mine = weibull.fit_regression(end, entry, observed, design, names)
    assert_agrees(mine, lifelines_estimates(end, entry, observed, design, names), names)


def test_the_fit_does_not_depend_on_where_the_search_starts() -> None:
    """The optimum is reached from starting values far from it.

    `fit_censored` starts at an exponential -- shape one, scale the mean age at
    the end of observation -- and that choice is only safe if the surface has a
    single optimum to find. It is not a detail that can be left untested: a
    start derived instead from a least-squares fit that treats censored rows as
    failures walks away from the optimum on data this heavily censored, where
    around nine rows in ten are censored, and settles at roughly twice the true
    shape.

    The search here takes no derivatives, so a correct gradient cannot be what
    rescues it.
    """
    settings = one_technology_config()
    end, entry, observed, _, _ = cross_check_data(settings)
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
