"""Generates the synthetic segment table the simulation runs on.

Three tables describe the population — segments, customers served, and the
per-class replacement cost — and they are joined here into the one frame the
rest of the project uses. The split is a modeling convenience; a join never
appears inside the annual loop.

Four columns are derived before any run starts, so that the compute kernel
receives flat per-segment arrays and never learns that customer types or
restoration times exist. Adding a customer type changes this module and the
configuration, and leaves the Rust side untouched.
"""

import numpy as np
import polars as pl
from scipy.special import ndtri

from cablesim import config as config_module
from cablesim import constants, streams, weibull

# ndtri(0) is negative infinity, which would give a segment zero length and
# then an infinite effective scale. The raw stream produces exactly zero with
# probability 2**-53; clipping costs nothing and removes the special case.
UNIT_EPS: float = 1e-15


def lognormal_from_uniforms(
    u: np.ndarray, spec: config_module.LogNormalSpec
) -> np.ndarray:
    """Maps uniforms to a lognormal by inverting its distribution function.

    Inverse transform rather than a generator method, so the values rest on the
    bit-stream guarantee described in :mod:`cablesim.streams`.

    Args:
        u: Uniforms on [0, 1).
        spec: The median and log-scale sigma to draw against.

    Returns:
        Lognormal draws, in the units of ``spec.median``.
    """
    z = ndtri(np.clip(u, UNIT_EPS, 1.0 - UNIT_EPS))
    return spec.median * np.exp(spec.sigma * z)


def install_year_distribution(
    spec: config_module.InitialAgeSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds the probability mass function over install years.

    The configured breakpoints are interpolated linearly at every integer year
    in the range and then normalized, which is what turns a curve of relative
    install volume into something that can be drawn from.

    Args:
        spec: The install-year range and its volume breakpoints.

    Returns:
        The years, and the probability of each.
    """
    first, last = spec.install_year_range
    years = np.arange(first, last + 1)
    breakpoints = np.array(sorted(spec.install_volume))
    weights = np.array([spec.install_volume[year] for year in breakpoints], dtype=float)
    interpolated = np.interp(years, breakpoints, weights)
    return years, interpolated / interpolated.sum()


def draw_categories(u: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Draws category indices from a probability mass function.

    Args:
        u: Uniforms on [0, 1), one per draw.
        probabilities: The mass function, summing to 1.

    Returns:
        Index into ``probabilities`` for each uniform.
    """
    return np.searchsorted(np.cumsum(probabilities), u, side="right").clip(
        0, len(probabilities) - 1
    )


def generate(config: config_module.Config) -> pl.DataFrame:
    """Builds the segment population from a validated configuration.

    Every draw comes from the population stream, which is spawned by purpose so
    it cannot correlate with the lifetime or policy streams.

    Args:
        config: The validated run configuration.

    Returns:
        One row per segment, ordered by ``segment_id``, carrying the identity
        columns, the effective Weibull pair for the cable in the ground, the
        pair a replacement would take, and the four derived columns the kernel
        consumes.
    """
    population = config.population
    n = population.n_segments
    types = population.customer_types

    source = streams.spawn_roots(config.simulation.seed).population
    draws = streams.uniforms(source, n * (3 + len(types))).reshape(3 + len(types), n)

    class_index = draw_categories(
        draws[0], np.array([c.share for c in population.classes])
    )
    years, year_probabilities = install_year_distribution(population.initial_age)
    install_year = years[draw_categories(draws[1], year_probabilities)]

    # Technology follows install year, so the correlation between age and
    # hazard is physical rather than assumed.
    technology_index = np.zeros(n, dtype=np.int64)
    for index, technology in enumerate(population.technologies):
        first, last = technology.vintage
        technology_index[(install_year >= first) & (install_year <= last)] = index

    length_ft = np.zeros(n)
    cost_per_ft = np.zeros(n)
    n_conductors = np.zeros(n, dtype=np.int64)
    shape_override = np.full(n, np.nan)
    outage_hours_emergency = np.zeros(n)
    outage_hours_planned = np.zeros(n)
    counts = {name: np.zeros(n) for name in types}
    for index, segment_class in enumerate(population.classes):
        rows = class_index == index
        drawn_length = lognormal_from_uniforms(draws[2], segment_class.length_ft)
        length_ft[rows] = drawn_length[rows]
        cost_per_ft[rows] = segment_class.cost_per_ft
        n_conductors[rows] = segment_class.n_conductors
        if segment_class.weibull_shape is not None:
            shape_override[rows] = segment_class.weibull_shape
        outage_hours_emergency[rows] = config.reliability.outage_hours_emergency[
            segment_class.name
        ]
        outage_hours_planned[rows] = config.reliability.outage_hours_planned[
            segment_class.name
        ]
        for offset, name in enumerate(types):
            drawn = lognormal_from_uniforms(
                draws[3 + offset], segment_class.customer_mix[name]
            )
            counts[name][rows] = np.rint(drawn[rows])

    shapes = np.array([t.weibull.shape for t in population.technologies])
    scales = np.array([t.weibull.scale for t in population.technologies])
    # Technology sets the shape unless the class names its own, which is how
    # a larger-conductor feeder cable differs from a lateral of the same
    # vintage.
    shape = np.where(np.isnan(shape_override), shapes[technology_index], shape_override)
    scale = scales[technology_index]
    replacement = next(
        t
        for t in population.technologies
        if t.name == population.replacement_technology
    )

    geometry = (n_conductors, length_ft, population.length_ref_ft,
                population.length_exponent)
    effective = weibull.effective_scale(scale, shape, *geometry)
    replacement_shape = np.where(
        np.isnan(shape_override), replacement.weibull.shape, shape_override
    )
    replacement_effective = weibull.effective_scale(
        np.full(n, replacement.weibull.scale), replacement_shape, *geometry
    )

    customers = sum(counts.values())
    value_per_hour = sum(
        counts[name] * config.reliability.voll_per_customer_hour[name] for name in types
    )
    frame = pl.DataFrame(
        {
            "segment_id": np.arange(n, dtype=np.int64),
            "class": [population.classes[i].name for i in class_index],
            "class_index": class_index,
            "technology": [population.technologies[i].name for i in technology_index],
            "n_conductors": n_conductors,
            "length_ft": length_ft,
            "install_year": install_year,
            "age": (config.simulation.start_year - install_year).astype(float),
            "cost_per_ft": cost_per_ft,
            **{name: counts[name] for name in types},
            # SAIFI counts customers equally; the value-weighted figure drives
            # scoring, and the minute figures drive SAIDI and CMI.
            "customers": customers,
            "customer_minutes_per_failure": customers
            * constants.MINUTES_PER_HOUR
            * outage_hours_emergency,
            "customer_minutes_per_planned": customers
            * constants.MINUTES_PER_HOUR
            * outage_hours_planned,
            "outage_cost_per_failure": value_per_hour * outage_hours_emergency,
            "shape": shape,
            "scale": effective,
            "replacement_shape": replacement_shape,
            "replacement_scale": replacement_effective,
        }
    )
    return frame
