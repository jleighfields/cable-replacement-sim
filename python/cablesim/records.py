"""Generates the synthetic censored failure history the MLE fits.

`population.py` emits one row per segment, sized by how large a system is being
modeled. This emits one row per **installation episode**, sized by how many
observed failures the recovery test needs to have power.

Technologies, length and sample size are arguments rather than config lookups,
because the early rungs of the recovery ladder need one technology and a fixed
length while the configured record block describes the full population mix.
"""

import numpy as np
import polars as pl

from cablesim import config as config_module
from cablesim import population, random_draws, weibull

MAX_EPISODES: int = 12
"""Episodes one segment may accumulate before the generator gives up.

A guard rather than a modeling limit. Reaching it means the drawn lifetimes are
short against the study window — a scale near zero, or a window of centuries —
and the honest response is to raise rather than to emit a truncated history
that looks like data.
"""


def episode_table(
    config: config_module.Config,
    *,
    technologies: list[config_module.Technology] | None = None,
    length_ft: config_module.LogNormalSpec | float | None = None,
    n_conductors: list[int] | None = None,
    n_segments: int | None = None,
    seed_offset: int = 0,
) -> pl.DataFrame:
    """Builds the failure history, one row per installation episode.

    Each segment is installed once, and is replaced by new cable of the
    configured replacement technology every time it fails inside the study
    window. Every installation is one row: a segment installed in 1972, failed
    in 2006 and still in service at a 2026 study end contributes an uncensored
    lifetime of 34 years and a censored one of 20. Collapsing that to one row
    per segment would discard the failure or mismeasure the age at which it
    happened, and both bias the fit toward longer life.

    Args:
        config: The validated run configuration. Supplies the study window, the
            install-year range, the reference length and exponent, and the
            replacement technology.
        technologies: Technologies to draw from, defaulting to the configured
            list. A single-element list gives the one-technology data the first
            three rungs of the recovery ladder need.
        length_ft: Length distribution, or a fixed length in feet. Defaults to
            the configured mix. A fixed value removes length as a source of
            variation, which is what isolates the earlier rungs.
        n_conductors: Conductor counts to draw from, uniformly. Defaults to the
            counts the configured classes use.
        n_segments: Segments to draw, defaulting to the configured record size.
        seed_offset: Added to the configured seed, so a rung can draw a fresh
            table without disturbing any other.

    Returns:
        One row per installation episode, with `failure_year` null where the
        episode was still in service at the study end. Lifetimes are
        **continuous**: the year columns hold fractional years, because the
        likelihood is continuous-time and rounding would make this
        interval-censored data fitted with the wrong likelihood.

    Raises:
        ValueError: If any segment needs more than `MAX_EPISODES` installations,
            which means the lifetimes are implausibly short for the window
            rather than that the history is long.
    """
    settings = config.population
    records = config.records
    technologies = list(
        technologies if technologies is not None else settings.technologies
    )
    total = n_segments if n_segments is not None else records.n_segments

    source = random_draws.spawn_sources(config.simulation.seed + seed_offset).records
    setup = random_draws.uniforms(source, 4 * total).reshape(4, total)

    # By default the record table mirrors the population: a class is drawn from
    # its share and supplies both the length distribution and the conductor
    # count, so the mix being fitted is the mix that exists. Either can be
    # overridden to hold it fixed, which is what isolates the early rungs.
    class_index = population.draw_categories(
        setup[3], np.array([c.share for c in settings.classes])
    )

    # Install year, then the technology that follows from it. With one
    # technology configured its vintage need not cover the range, so the draw
    # falls back to that technology for every segment.
    years, year_probabilities = population.install_year_distribution(
        settings.initial_age
    )
    drawn_year = population.draw_categories(setup[0], year_probabilities)
    install_year = years[drawn_year].astype(float)
    technology_index = np.zeros(total, dtype=np.int64)
    for index, technology in enumerate(technologies):
        first, last = technology.vintage
        technology_index[(install_year >= first) & (install_year <= last)] = index

    length = np.zeros(total)
    conductors = np.zeros(total, dtype=np.int64)
    if isinstance(length_ft, (int, float)):
        length[:] = float(length_ft)
    if n_conductors is not None:
        counts = np.array(n_conductors)
        conductors[:] = counts[
            population.draw_categories(
                setup[2], np.full(len(counts), 1.0 / len(counts))
            )
        ]
    for index, segment_class in enumerate(settings.classes):
        rows = class_index == index
        if not isinstance(length_ft, (int, float)):
            spec = length_ft if length_ft is not None else segment_class.length_ft
            length[rows] = population.lognormal_from_uniforms(setup[1][rows], spec)
        if n_conductors is None:
            conductors[rows] = segment_class.n_conductors

    shapes = np.array([t.weibull.shape for t in technologies])
    scales = np.array([t.weibull.scale for t in technologies])
    replacement = next(
        (t for t in technologies if t.name == settings.replacement_technology),
        technologies[-1],
    )

    # One episode per round: every segment still failing inside the window
    # draws again, so the loop runs as many times as the busiest segment needs
    # rather than once per segment.
    episode_source = random_draws.spawn_sources(
        config.simulation.seed + seed_offset
    ).records.spawn(MAX_EPISODES)
    segment_id = np.arange(total, dtype=np.int64)
    alive = np.arange(total)
    start = install_year.copy()
    current = technology_index.copy()
    rows: list[dict[str, np.ndarray]] = []

    for episode in range(MAX_EPISODES):
        if alive.size == 0:
            break
        effective = weibull.effective_scale(
            shapes[current[alive]],
            scales[current[alive]],
            conductors[alive],
            length[alive],
            settings.length_ref_ft,
            settings.length_exponent,
        )
        drawn = weibull.draw_lifetime(
            random_draws.uniforms(episode_source[episode], alive.size),
            shapes[current[alive]],
            effective,
        )
        ends = start[alive] + drawn
        failed = ends < records.study_end

        rows.append(
            {
                "segment_id": segment_id[alive],
                "technology": np.array([technologies[i].name for i in current[alive]]),
                "n_conductors": conductors[alive],
                "length_ft": length[alive],
                "install_year": start[alive].copy(),
                # None rather than NaN: a censored episode has no failure
                # year, and a NaN would read as an observed one to anything
                # asking whether the column is null.
                "failure_year": np.where(failed, ends, np.nan),
            }
        )
        start[alive[failed]] = ends[failed]
        current[alive[failed]] = technologies.index(replacement)
        alive = alive[failed]
    else:
        if alive.size:
            raise ValueError(
                f"{alive.size} segments still failing after {MAX_EPISODES} episodes; "
                f"the drawn lifetimes are short against a "
                f"{records.study_end - min(years)}-year window"
            )

    frame = pl.DataFrame(
        {key: np.concatenate([r[key] for r in rows]) for key in rows[0]}
    ).with_columns(
        pl.when(pl.col("failure_year").is_nan())
        .then(None)
        .otherwise(pl.col("failure_year"))
        .alias("failure_year")
    )
    # The observation window. An episode that ended at or before monitoring
    # began was never observable, and dropping it is what the truncation
    # correction compensates for; the comparison is inclusive so nothing is
    # kept with zero exposure after entry.
    return (
        frame.with_columns(
            pl.max_horizontal(
                pl.col("install_year"), pl.lit(float(records.monitoring_start))
            ).alias("entry_year")
        )
        .filter(
            pl.col("failure_year").is_null()
            | (pl.col("failure_year") > records.monitoring_start)
        )
        .filter(pl.col("install_year") < records.study_end)
        .select(
            "segment_id", "technology", "n_conductors", "length_ft",
            "install_year", "entry_year", "failure_year",
        )
        .sort("segment_id", "install_year")
    )
