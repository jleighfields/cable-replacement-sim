"""Weibull lifetimes: the effective-scale reduction, hazard, and draws.

Two derived forms of the Weibull are used, for two different jobs. Sampling a
lifetime inverts the survivor function; scoring a replacement candidate uses
the discrete annual failure probability. A policy ranks on the second because
that is what a planner knows — it never sees the sampled failure time.

The Python side of the mirror: ``src/weibull.rs`` implements the same
functions, and the parity tests between them are what validate the kernel.
"""

import numpy as np


def effective_scale(
    scale: np.ndarray,
    shape: np.ndarray,
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
    one, so laterals outlast feeders by a factor no scale choice reaches below
    about 1.25 even at implausibly sharp shapes.

    Args:
        scale: Conductor-level Weibull scale at the reference length, in years.
        shape: Weibull shape parameter.
        n_conductors: Conductors per segment.
        length_ft: Segment length, in feet.
        length_ref_ft: The reference length the configured scale describes.
        length_exponent: How strongly length drives failure. 1 is the
            spatial-Poisson case, 0 makes failures purely per-segment.

    Returns:
        The segment's effective Weibull scale, in years.
    """
    pieces = n_conductors * (length_ft / length_ref_ft) ** length_exponent
    return scale * pieces ** (-1.0 / shape)


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
        The annual failure probability, on [0, 1).
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

    Drawing unconditionally here is the most likely correctness bug in this
    model: it makes a population that starts partway through its life behave as
    though it were new, which inflates every policy's apparent performance, and
    it does so silently — the run completes and the curves look plausible.

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
