"""Analytical checks on the Weibull forms.

These compare against the closed form rather than against another
implementation, so they build their own draws at their own sizes and seeds.
An analytical check is worth more than a parity check because it can be wrong
in only one way.
"""

import numpy as np
from cablesim import random_draws, weibull
from numpy.random import SeedSequence
from scipy import stats

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
    assert survived == np.float64(closed_form) or abs(
        survived / closed_form - 1.0
    ) < 1e-12


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
