"""Analytical checks on the Weibull forms.

These compare against the closed form rather than against another
implementation, so they build their own draws at their own sizes and seeds.
An analytical check is worth more than a parity check because it can be wrong
in only one way.
"""

import re

import numpy as np
import pytest
from cablesim import config, constants, population, random_draws, run, weibull
from numpy.random import SeedSequence
from scipy import stats

from tests import helpers

KS_DRAWS: int = 100_000
KS_ALPHA: float = 0.001
"""Sample size and significance chosen together.

The size gives power to detect an exponent wrong in its second decimal; the
small alpha keeps a correct implementation from failing by chance. The upper
bound on size is runtime, not sensitivity — floating-point differences perturb
the distribution far below the rejection threshold at any size used here.
"""


def weibull_draws(
    source: SeedSequence, shape: float, scale: float, n: int
) -> np.ndarray:
    """Draws lifetimes from a Weibull by inverting its survivor function.

    Args:
        source: Seed sequence to draw from.
        shape: Weibull shape parameter.
        scale: Weibull scale parameter.
        n: How many lifetimes to draw.

    Returns:
        Ages at failure, in years.
    """
    return weibull.draw_lifetime(random_draws.uniforms(source, n), shape, scale)


def test_min_of_n_matches_the_reduced_scale() -> None:
    """The minimum of three conductor lifetimes is Weibull at the reduced scale.

    This is the closed form the whole three-phase treatment rests on, and it is
    checked against sampled minima rather than assumed.
    """
    shape, scale, n = 2.4, 55.0, KS_DRAWS
    source = random_draws.spawn_sources(11).population
    draws = weibull.draw_lifetime(
        random_draws.uniforms(source, 3 * n).reshape(3, n), shape, scale
    )

    reduced = weibull.effective_scale(
        np.array(shape), np.array(scale), np.array(3.0), np.array(500.0), 500.0, 1.0
    )
    result = stats.kstest(
        draws.min(axis=0), "weibull_min", args=(shape, 0.0, float(reduced))
    )

    assert result.pvalue > KS_ALPHA, f"min of 3 is not Weibull({shape}, {reduced})"


def test_length_enters_through_the_exponent() -> None:
    """A longer segment fails sooner, by the configured power of its length.

    With an exponent of 1 this is the spatial-Poisson case; the point of the
    check is that whatever exponent is configured is the one applied.
    """
    shape, scale, n = 2.4, 55.0, KS_DRAWS
    length, reference, exponent = 2000.0, 500.0, 0.5
    reduced = weibull.effective_scale(
        np.array(shape),
        np.array(scale),
        np.array(1.0),
        np.array(length),
        reference,
        exponent,
    )
    expected = scale * ((length / reference) ** exponent) ** (-1.0 / shape)
    assert reduced == np.float64(expected)

    source = random_draws.spawn_sources(12).population
    draws = weibull_draws(source, shape, float(reduced), n)
    result = stats.kstest(draws, "weibull_min", args=(shape, 0.0, float(reduced)))
    assert result.pvalue > KS_ALPHA


def test_annual_probability_telescopes_to_the_survival_curve() -> None:
    """Surviving every year in turn equals surviving the horizon.

    Deterministic, so it is asserted exactly rather than statistically: the
    product of ``1 - p(t)`` over the horizon is ``S(30)``.
    """
    shape, scale, horizon = 2.2, 50.0, 30
    ages = np.arange(horizon, dtype=float)

    survived = np.prod(
        1.0 - weibull.conditional_failure_probability(ages, shape, scale)
    )

    closed_form = np.exp(-((horizon / scale) ** shape))
    assert (
        survived == np.float64(closed_form) or abs(survived / closed_form - 1.0) < 1e-12
    )


def test_left_truncated_draw_matches_conditional_survival() -> None:
    """Remaining life given survival to an age follows ``S(t)/S(a)``.

    Checked at three entry ages spanning the range, because the correction's
    error grows with entry age and a single young cohort passes even with the
    correction missing altogether.
    """
    shape, scale, n = 2.2, 50.0, KS_DRAWS
    for index, entry_age in enumerate((5.0, 30.0, 70.0)):
        source = random_draws.spawn_sources(20 + index).population
        remaining = weibull.draw_remaining_life(
            random_draws.uniforms(source, n), entry_age, shape, scale
        )
        total = entry_age + remaining

        # Conditioning on survival to `entry_age` is a truncation of the same
        # Weibull, so the truncated distribution is the reference.
        reference = stats.weibull_min(shape, scale=scale)
        expected = (reference.sf(total) / reference.sf(entry_age)).clip(0.0, 1.0)
        result = stats.kstest(1.0 - expected, "uniform")

        assert result.pvalue > KS_ALPHA, f"entry age {entry_age} is not conditional"
        assert (remaining > 0).all()


def test_a_censored_episode_of_zero_age_does_not_poison_the_likelihood() -> None:
    """Zero exposure contributes nothing rather than making the sum undefined.

    The failure term is multiplied by whether the episode failed, so a censored
    row should contribute none of it. The logarithm inside is still evaluated,
    though, and at zero age it is negative infinity: multiplied by zero that is
    `nan`, not zero, and one such row makes the whole log-likelihood undefined.
    The fit downstream then fails naming nothing that points here.

    Not reachable through `records.lifetimes`, where a censored episode is
    measured to the study end and so has positive age. `log_likelihood` is
    public and takes arrays directly, which is what makes it worth pinning.
    """
    age = np.array([0.0, 10.0, 20.0])
    entry = np.zeros(3)

    censored = weibull.log_likelihood(6.0, 50.0, age, entry, np.array([0.0, 1.0, 1.0]))
    assert np.isfinite(censored)
    # The zero-age row is the only difference, and it contributes nothing, so
    # dropping it must leave the same value.
    assert censored == pytest.approx(
        weibull.log_likelihood(6.0, 50.0, age[1:], entry[1:], np.array([1.0, 1.0]))
    )

    # An observed failure at age zero is a degenerate lifetime rather than a
    # rounding artefact, and must still be refused rather than smoothed over.
    with np.errstate(divide="ignore"):
        degenerate = weibull.log_likelihood(
            6.0, 50.0, age, entry, np.array([1.0, 1.0, 1.0])
        )
    assert not np.isfinite(degenerate)


FORCED_SCALE_BOUND = re.compile(
    r"that term is ``\((?P<age>[0-9.]+) / scale\) \*\* (?P<shape>[0-9.]+)``\."
    r".*?a scale under about (?P<bound>[0-9.e-]+) overflows it"
    r".*?clears that bound by a factor of (?P<margin>[0-9.]+)"
    r".*?the same term peaks at (?P<peak>[0-9.]+)",
    re.DOTALL,
)
"""The arithmetic ``helpers.FAILS_AT_ONCE`` quotes for its own lower bound.

Read out of the docstring rather than restated here, so this fails when the
prose and the fleet disagree rather than when someone forgets a second copy —
the same arrangement ``test_parity.py`` uses for the premium comment in
``src/policies.rs``.
"""


def test_the_forced_scale_stays_inside_single_precision() -> None:
    """The margin ``helpers.FAILS_AT_ONCE`` quotes is the one the fleet gives.

    That constant is a Weibull scale small enough to force a failure inside
    year one, and how small it may go is bounded by single precision rather
    than by the outcome: the hazard is a difference of two ``(age / scale) **
    shape`` terms, and past the ceiling both terms are infinite and their
    difference is a NaN the ranking refuses. The docstring states that bound by
    naming an age and a shape, then quotes how far the constant sits above it.

    Both inputs are facts about the shipped configuration, so both can go stale
    without anything reporting it — a vintage added to the class table moves
    the largest shape, and a longer horizon moves the oldest age. A margin
    quoted in prose is what a later edit reads before deciding how far it may
    lower the constant, and this is the only thing that reads it back.

    The age is the oldest the hazard is evaluated at: a segment never replaced
    reaches ``max(age0) + n_years``, since the year's hazard reads ``age + 1``.
    The shape is the largest any segment carries, in the ground or after
    replacement. Pairing the two overstates the bound, because no one segment
    has both — which is the direction a lower bound should err in.
    """
    settings = config.load_config(constants.DEFAULT_CONFIG_PATH)
    segments = run.segment_arrays(
        population.generate(settings), constants.DEFAULT_PRECISION
    )
    oldest = float(segments["age0"].max()) + settings.simulation.n_years
    largest_shape = float(
        max(segments["shape"].max(), segments["replacement_shape"].max())
    )
    # The scale at which ``(oldest / scale) ** largest_shape`` reaches the
    # single-precision ceiling; anything smaller overflows it.
    bound = oldest / float(np.finfo(np.float32).max) ** (1.0 / largest_shape)

    source = (constants.PROJECT_ROOT / "tests" / "helpers.py").read_text(
        encoding="utf-8"
    )
    quoted = FORCED_SCALE_BOUND.search(" ".join(source.split()).replace("` `", "``"))
    assert quoted is not None, (
        "tests/helpers.py no longer states FAILS_AT_ONCE's lower bound in the "
        "form this reads; either the docstring was reworded, in which case "
        "update this pattern, or the bound was dropped"
    )

    assert (float(quoted["age"]), float(quoted["shape"])) == (
        oldest,
        largest_shape,
    ), (
        f"FAILS_AT_ONCE's docstring bounds itself with "
        f"({quoted['age']} / scale) ** {quoted['shape']}, but the shipped "
        f"fleet's oldest evaluated age is {oldest:g} and its largest shape is "
        f"{largest_shape:g}; the bound it derives is not this fleet's"
    )
    assert float(quoted["bound"]) == pytest.approx(bound, rel=0.05), (
        f"the docstring says a scale under {quoted['bound']} overflows single "
        f"precision; from this fleet the bound is {bound:.3g}"
    )
    assert float(quoted["margin"]) == pytest.approx(
        helpers.FAILS_AT_ONCE / bound, rel=0.05
    ), (
        f"the docstring says {helpers.FAILS_AT_ONCE:g} clears that bound by "
        f"{quoted['margin']}x; it clears {bound:.3g} by "
        f"{helpers.FAILS_AT_ONCE / bound:.3g}x"
    )


    # Per segment, each against the oldest age it reaches, which is the term
    # the model computes — not every segment paired with the fleet's oldest.
    reached = segments["age0"] + settings.simulation.n_years
    peak = float(((reached / segments["scale"]) ** segments["shape"]).max())
    assert float(quoted["peak"]) == pytest.approx(peak, rel=0.05), (
        f"the docstring says the hazard term peaks at {quoted['peak']} at the "
        f"shipped scales; over this fleet it peaks at {peak:.3g}"
    )

    # The bound is a claim about arithmetic, so it is run as well as compared.
    # At the constant the hazard is finite everywhere; two orders of magnitude
    # below it, it is not, which is the refusal the docstring asks for. One
    # order is not enough, because the bound pairs an age and a shape no single
    # segment has — the oldest segments carry neither of the largest shapes.
    narrow = np.float32
    at_own_age = segments["age0"].astype(narrow)
    shapes = segments["shape"].astype(narrow)
    with np.errstate(over="ignore", invalid="ignore"):
        finite = weibull.conditional_failure_probability(
            at_own_age, shapes, narrow(helpers.FAILS_AT_ONCE)
        )
        overflowed = weibull.conditional_failure_probability(
            at_own_age, shapes, narrow(helpers.FAILS_AT_ONCE / 100.0)
        )
    assert np.isfinite(finite).all(), (
        f"{helpers.FAILS_AT_ONCE:g} already overflows single precision "
        f"somewhere in this fleet, so it sits below its own documented bound"
    )
    assert np.isnan(overflowed).mean() > 0.5, (
        f"two orders of magnitude below {helpers.FAILS_AT_ONCE:g} leaves "
        f"{np.isnan(overflowed).mean():.0%} of the fleet at NaN; the docstring "
        f"says most of it, which is the reason it gives for not lowering this"
    )


def test_the_survivor_is_one_over_e_at_the_scale_whatever_the_shape() -> None:
    """The definition of the Weibull scale, and it pins the parameterization.

    ``S(scale) = exp(-1)`` for every shape is what makes the scale the
    characteristic life rather than the mean or the median. A transposed
    parameterization — the other convention's ``(mu, sigma)``, or scale and
    shape swapped — does not have this property, which is the transposition
    the recovery ladder exists to catch and this catches in one line.
    """
    scale = np.float64(50.0)
    for shape in (np.float64(1.0), np.float64(2.4), np.float64(6.2)):
        assert weibull.survival(scale, shape, scale) == pytest.approx(np.exp(-1.0))

    assert weibull.survival(np.float64(0.0), np.float64(6.2), scale) == 1.0


def test_the_survivor_and_the_annual_probability_are_the_same_model() -> None:
    """Two functions over one hazard, and where they stop agreeing is the point.

    ``conditional_failure_probability`` is ``1 - S(t+1)/S(t)`` by definition,
    and it is computed a different way — a difference of cumulative hazards
    through ``expm1`` rather than a ratio of two survivors. Agreeing across the
    ages the model reaches is what says they are one model and not two, and it
    would catch a shape and scale swapped in either.

    **They agree absolutely and not relatively, and that is the reason for the
    ``expm1`` form.** Forming ``1 - S(t+1)/S(t)`` subtracts two numbers that
    are both within an ulp of 1 at young ages, so the result keeps almost no
    significant figures: at age 0 the annual probability is 2.9e-11 and the two
    routes differ by 1.6e-6 of it, while differing by 1.2e-16 in absolute
    terms. By age 5, where the probability has passed 1e-6, the relative
    difference is 3.8e-11 and falls from there.

    So the assertion is absolute across the whole range and relative only where
    cancellation is not doing the damage. A test written the other way round
    fails, and what it would be reporting is the naive formula's arithmetic
    rather than a disagreement about the model.
    """
    ages = np.linspace(0.0, 90.0, 91)
    shape, scale = np.float64(6.2), np.float64(50.0)

    annual = weibull.conditional_failure_probability(ages, shape, scale)
    through_survivor = 1.0 - (
        weibull.survival(ages + 1.0, shape, scale)
        / weibull.survival(ages, shape, scale)
    )

    np.testing.assert_allclose(annual, through_survivor, rtol=0.0, atol=1e-15)

    worth_reading = annual > 1e-6
    assert worth_reading.any(), "no age carries a probability big enough to compare"
    np.testing.assert_allclose(
        annual[worth_reading], through_survivor[worth_reading], rtol=1e-9
    )


def test_the_annual_probability_keeps_its_figures_at_a_young_age() -> None:
    """What the ``expm1`` form buys, pinned rather than asserted in prose.

    At a young age the year's hazard is tiny — 2.93e-11 for the shipped
    technology at age zero — and there the annual probability is that hazard to
    every figure a double carries, because the next term of the series is
    4e-22. Computing it as ``1 - exp(-h)`` subtracts two numbers within an ulp
    of each other and keeps about five significant figures; ``-expm1(-h)``
    keeps all of them.

    The sibling test above cannot catch this. It compares the annual
    probability against a ratio of survivors, and that ratio suffers the same
    cancellation — so replacing ``expm1`` with the naive form moves both sides
    together and they agree *better*. Measured: the naive form lands 1.6e-6
    away in relative terms, the ``expm1`` form 1.5e-11.
    """
    shape, scale = np.float64(6.2), np.float64(50.0)
    hazard = (1.0 / scale) ** shape

    annual = weibull.conditional_failure_probability(
        np.float64(0.0), shape, scale
    )

    # `abs=0` is load-bearing. `pytest.approx` passes when *either* tolerance
    # is met and its default absolute one is 1e-12 — a hundredfold larger than
    # the quantity being compared here, so a relative tolerance alone would
    # accept any answer at all and this test would pin nothing.
    assert annual == pytest.approx(hazard, rel=1e-9, abs=0.0), (
        "the annual probability at age zero has lost significant figures to "
        "cancellation; at a hazard this small it is the hazard itself, and a "
        "subtraction of two near-equal numbers is what throws that away"
    )
