"""Weibull lifetimes: the effective-scale reduction, hazard, and draws.

Two derived forms of the Weibull are used, for two different jobs. Sampling a
lifetime inverts the survivor function; scoring a replacement candidate uses
the discrete annual failure probability. A policy ranks on the second because
that is what a planner knows — it never sees the sampled failure time.

The Python side of the mirror: ``src/weibull.rs`` implements the same
functions, and the parity tests between them are what validate the kernel.
"""

from typing import NamedTuple

import numpy as np
from scipy import optimize, stats


def effective_scale(
    shape: np.ndarray,
    scale: np.ndarray,
    n_conductors: np.ndarray,
    length_ft: np.ndarray,
    length_ref_ft: float,
    length_exponent: float,
) -> np.ndarray:
    """Reduces a conductor-level scale to the segment it belongs to.

    A three-phase segment is out when any one of its conductors fails, so its
    lifetime is the minimum of three, and the minimum of ``n`` independent
    Weibulls sharing a shape is Weibull with the same shape and a reduced
    scale. Length enters the same way but raised to an exponent::

        scale_eff = scale * (n * (L / L_ref) ** beta) ** (-1 / shape)

    Shape is unchanged, which is what lets the terms compose and what keeps one
    ``(shape, scale)`` pair enough per segment.

    **The conductor term is exact; the length term is not.** A pure
    spatial-Poisson argument would set ``beta = 1``, treating a segment as
    unit-length pieces in series. That is rejected on both physics and
    arithmetic. Physically, a large share of underground faults occur at
    splices, terminations and elbows, which scale with the count of accessories
    rather than with feet of run, so failure rate is sub-linear in length.
    Quantitatively, ``beta = 1`` fixes the ratio between a long three-phase
    feeder and a short lateral at about ``10 ** (1 / shape)``, which needs a
    shape near 9 to bring the population into one plausible band; a sub-linear
    exponent separates the two three-phase classes from each other without
    that.

    What the conductor term implies cannot be tuned away: a three-phase segment
    lives ``3 ** (-1 / shape)`` as long as an otherwise identical single-phase
    one, so on conductor count alone a lateral outlasts a feeder by
    ``3 ** (1 / shape)`` — 1.22 at a shape of 5.5 and 1.18 at 6.8, the range
    ``configs/base.yaml`` configures. No choice of scale moves that factor;
    only shape does, and it shrinks toward 1 as shape rises.

    Args:
        shape: Weibull shape parameter. First, matching every other function
            here — both parameters are positive floats, so a swapped call
            would type-check, run, and return a plausible number.
        scale: Conductor-level Weibull scale at the reference length, in years.
        n_conductors: Conductors per segment.
        length_ft: Segment length, in feet.
        length_ref_ft: The reference length the configured scale describes.
        length_exponent: How strongly length drives failure. 1 is the
            spatial-Poisson case, 0 makes failures purely per-segment.

    Returns:
        The segment's effective Weibull scale, in years.
    """
    # Multiplies the cumulative hazard, which is equivalent to dividing the
    # scale by its `shape`-th root. Under a linear length term this would be a
    # count of unit-length pieces in series; the sub-linear exponent is what
    # stops it being a count of anything.
    hazard_multiplier = n_conductors * (length_ft / length_ref_ft) ** length_exponent
    return scale * hazard_multiplier ** (-1.0 / shape)


def conditional_failure_probability(
    age: np.ndarray, shape: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Probability of failing within a year, given survival to ``age``.

    This is what a replacement policy ranks on::

        p(t) = 1 - S(t+1)/S(t) = 1 - exp(-[((t+1)/scale)**k - (t/scale)**k])

    Args:
        age: Current age, in years.
        shape: Weibull shape parameter.
        scale: Effective Weibull scale, in years.

    Returns:
        The annual failure probability, on [0, 1]. It reaches exactly 1 once
        the accumulated hazard passes about 745, where ``exp`` underflows —
        an age far beyond anything this model simulates, but the bound is
        closed rather than half-open.
    """
    accumulated = ((age + 1.0) / scale) ** shape - (age / scale) ** shape
    return -np.expm1(-accumulated)


def draw_lifetime(u: np.ndarray, shape: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Inverts the survivor function to sample a lifetime for new cable.

    Args:
        u: Uniforms on [0, 1).
        shape: Weibull shape parameter.
        scale: Effective Weibull scale, in years.

    Returns:
        Age at failure, in years.
    """
    return scale * (-np.log1p(-u)) ** (1.0 / shape)


def draw_remaining_life(
    u: np.ndarray, age: np.ndarray, shape: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Samples remaining life for cable that has already survived to ``age``.

    The draw is **conditional** on that survival, which adds the hazard already
    accumulated back in::

        T = scale * ((age/scale)**k - ln u) ** (1/k),   remaining = T - age

    Drawing unconditionally here makes a population that starts partway through
    its life behave as though it were new, which inflates every policy's
    apparent performance, and it does so silently — the run completes and the
    curves look plausible.

    Args:
        u: Uniforms on [0, 1).
        age: Current age, in years.
        shape: Weibull shape parameter.
        scale: Effective Weibull scale, in years.

    Returns:
        Remaining life from ``age``, in years.
    """
    accumulated = (age / scale) ** shape
    total = scale * (accumulated - np.log1p(-u)) ** (1.0 / shape)
    return total - age


class Fit(NamedTuple):
    """A fitted Weibull, with intervals wide enough to judge recovery by.

    Attributes:
        shape: Fitted shape.
        scale: Fitted scale, in years.
        shape_interval: Confidence interval for the shape.
        scale_interval: Confidence interval for the scale.
        log_likelihood: Value at the optimum, for comparing nested fits.
    """

    shape: float
    scale: float
    shape_interval: tuple[float, float]
    scale_interval: tuple[float, float]
    log_likelihood: float


def log_likelihood(
    shape: float,
    scale: float,
    age_at_end: np.ndarray,
    entry_age: np.ndarray,
    observed: np.ndarray,
) -> float:
    """Right-censored, left-truncated Weibull log-likelihood.

    Each episode contributes its hazard at the moment it failed, if it did, and
    the hazard it accumulated over the window it was actually watched::

        loglik = sum_i [ observed_i * log h(t_i) ] - sum_i [ H(t_i) - H(a_i) ]

    **Censoring and truncation enter in different places, and the difference
    is not cosmetic.** Censoring is handled by ``observed``: an episode that
    reached the study end without failing contributes only accumulated hazard,
    not a failure. Truncation is handled by ``entry_age``: an episode is in the
    sample partly because it survived long enough to be recorded, so the
    ``+ H(a_i)`` divides its contribution by ``S(a_i)`` and conditions on that.

    Put the other way round: a censored episode is one you have but cannot see
    the end of; a truncated one is an episode you never had. The likelihood can
    only correct for the second because the asset register records install
    dates, which is what makes an entry age computable at all.

    Charging each episode for hazard accumulated before anyone was watching
    counts exposure that could never have produced an observation, and biases
    the fit toward longer life. How much depends on how far into its life an
    episode was when observation began: ``H(a)`` is ``(a/scale)**shape``, so at
    a sharp shape an early entry contributes almost nothing and the correction
    is worth a fraction of a percent. It grows as entry ages approach the
    scale.

    Args:
        shape: Weibull shape parameter.
        scale: Weibull scale parameter, in years.
        age_at_end: Age at failure, or at the study end for a censored episode.
        entry_age: Age when observation began — zero where the episode was
            watched from installation, in which case ``H(0)`` is zero and the
            truncation term vanishes. The same expression therefore covers a
            complete history and a partial one, with no branch.
        observed: 1 where the episode ended in a failure, 0 where censored.

    Returns:
        The log-likelihood.
    """
    hazard_at_end = (age_at_end / scale) ** shape
    hazard_at_entry = (entry_age / scale) ** shape
    log_hazard = np.log(shape / scale) + (shape - 1.0) * np.log(age_at_end / scale)
    return float(
        np.sum(observed * log_hazard) - np.sum(hazard_at_end - hazard_at_entry)
    )


def fit_censored(
    age_at_end: np.ndarray,
    entry_age: np.ndarray,
    observed: np.ndarray,
    level: float = 0.95,
) -> Fit:
    """Fits a Weibull to censored, left-truncated lifetimes.

    Optimizes over ``(log shape, log scale)`` so that both stay positive
    without the solver needing bounds, and reports intervals by exponentiating
    the endpoints on that scale — which keeps them positive and asymmetric,
    as an interval for a positive quantity should be.

    Starting values are the shape of an exponential and the mean age at end.
    A shape of 1 is the neutral choice, assuming nothing about whether the
    hazard rises or falls; the mean age is the right order of magnitude for the
    scale whatever the censoring.

    Args:
        age_at_end: Age at failure, or at the study end for a censored episode.
        entry_age: Age when observation began, zero where none.
        observed: 1 where the episode ended in a failure, 0 where censored.
        level: Confidence level for the intervals.

    Returns:
        The fitted parameters and their intervals.

    Raises:
        ValueError: If no episode ended in a failure, which leaves the shape
            unidentified, or if the optimizer does not converge.
    """
    if not np.any(observed):
        raise ValueError("no observed failures: the shape is not identified")

    def negative(log_parameters: np.ndarray) -> float:
        shape, scale = np.exp(log_parameters)
        return -log_likelihood(shape, scale, age_at_end, entry_age, observed)

    def gradient(log_parameters: np.ndarray) -> np.ndarray:
        """Analytic gradient of the negative log-likelihood in log-parameters.

        Supplied rather than left to finite differences: near the optimum the
        likelihood is flat enough that a numerical gradient loses precision and
        the optimizer reports failure at the right answer, which is worse than
        either succeeding or failing honestly.
        """
        shape, scale = np.exp(log_parameters)
        end_ratio = age_at_end / scale
        entry_ratio = entry_age / scale
        end_hazard = end_ratio**shape
        entry_hazard = entry_ratio**shape

        # An episode watched from installation has zero entry age, where the
        # hazard is zero and its logarithm is not finite. The product is zero,
        # so the term is dropped rather than evaluated.
        safe_ratio = np.where(entry_age > 0.0, entry_ratio, 1.0)
        entry_term = np.where(
            entry_age > 0.0, entry_hazard * np.log(safe_ratio), 0.0
        )
        d_shape = np.sum(observed * (1.0 / shape + np.log(end_ratio))) - np.sum(
            end_hazard * np.log(end_ratio) - entry_term
        )
        d_scale = (shape / scale) * (
            np.sum(end_hazard - entry_hazard) - np.sum(observed)
        )
        # Chain to the log-parameters the optimizer works in.
        return -np.array([shape * d_shape, scale * d_scale])

    start = np.log([1.0, float(np.mean(age_at_end))])
    result = optimize.minimize(negative, start, jac=gradient, method="BFGS")
    if not result.success:
        raise ValueError(f"the fit did not converge: {result.message}")

    shape, scale = np.exp(result.x)
    # hess_inv is the covariance of the log-parameters, so the interval is
    # symmetric there and exponentiates to an asymmetric one for the parameter.
    errors = np.sqrt(np.diag(result.hess_inv))
    width = stats.norm.ppf(0.5 + level / 2.0)
    low, high = np.exp(result.x - width * errors), np.exp(result.x + width * errors)

    return Fit(
        shape=float(shape),
        scale=float(scale),
        shape_interval=(float(low[0]), float(high[0])),
        scale_interval=(float(low[1]), float(high[1])),
        log_likelihood=float(-result.fun),
    )
