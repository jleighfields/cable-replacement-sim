"""Generates the synthetic censored failure history the MLE fits.

`population.py` emits one row per segment, sized by how large a system is being
modeled. This emits one row per **installation episode**, sized by how many
observed failures the recovery test needs to have power.

Technologies, length and sample size are arguments rather than config lookups,
because the early rungs of the recovery ladder need one technology and a fixed
length while the configured record block describes the full population mix.

The study design this assumes
-----------------------------

A **prospective follow-up**. The utility inventories what is in the ground in
``monitoring_start`` — install dates come from the asset register, which keeps
them whether or not failures were being tracked — and records failures from
that date until ``study_end``.

That is what makes an episode's history knowable. A cable installed in 1992 and
inventoried in 1998 is known to exist and known to be six years old. A cable
installed in 1990 that failed in 1995 is not in the inventory at all; its
replacement is there instead, and nothing in the data says the earlier one ever
existed.

Two other designs would give different data, and neither is what this builds:
an asset register carrying every install and replacement back to the first
install year has no truncation, because nothing is missing; a register listing
only current inventory with no failure log has no failures, and the shape is
then unidentified.

Censoring and truncation
------------------------

Two different things happen at the edges of the study window, and the words are
not interchangeable.

**Left truncation is a missing row. Right censoring is a row with a missing
end.** That is the whole distinction, and one episode can meet both.

Worked from the generator: a segment has cable installed in 1967, which fails
in 1996 at age 29 and is replaced the same moment. The 1998 study sees one row
— the replacement, installed 1996, still running at the 2026 study end.

The 1967 episode is the **missing row**. Not a row with an unknown age: no row
at all. Nothing in the data says the 1996 cable is a replacement rather than an
original install, and no term in the likelihood corresponds to it.

The surviving row is **right-censored** at the exit: in the table, still
running, end unknown. It is also **left-truncated** at the entry, because it is
in the table only by virtue of having lasted until the inventory — cable
installed in 1996 that failed before 1998 would not be here, its own
replacement would be. So the row is asked a conditional question: given it
reached age 2, what did it then do? Answering that one instead of the
unconditional one is the entire correction. The missing row is never added
back; the rows present are asked something slightly different.

The generator makes them in different places. It simulates each segment's
complete history first, so every lifetime is drawn whether or not a record
system would have caught it. Applying the record window then deletes the
episodes that ended before records began.

**Those deleted episodes are gone, and the correction does not bring them
back.** Nothing could: they left no trace, and a row cannot be reconstructed
from its own absence. What the correction fixes is the bias in what *remains*.
An episode is in the table partly because it lasted long enough to still be
running when records began, so the surviving sample over-represents long lives.
Conditioning each retained episode on having reached its entry age removes
exactly that much and no more.

The entry age is computable because the asset register records install dates
even where failures are not recorded. That is the whole of what the correction
needs — not knowledge of the missing episodes, only the age each surviving one
had already reached when someone started watching.
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
    # The record window, which creates the truncation. Dropping an episode
    # that ended before monitoring began is not the same as censoring one: a
    # censored episode stays in the table with an unknown end, while this one
    # leaves no trace at all. The comparison is inclusive so nothing is kept
    # with zero exposure after entry, which would contribute nothing to the
    # likelihood while inflating the apparent sample size.
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
            "segment_id",
            "technology",
            "n_conductors",
            "length_ft",
            "install_year",
            "entry_year",
            "failure_year",
        )
        .sort("segment_id", "install_year")
    )


def lifetimes(
    table: pl.DataFrame, study_end: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts the episode table into the ages the likelihood is written in.

    The table records calendar years; the likelihood is a function of age. An
    episode that ran from its install year to its failure year contributes the
    difference, and a censored one is measured to the study end instead.

    Args:
        table: The episode table.
        study_end: The year observation stopped.

    Returns:
        Age at end, age at entry, and 1 where the episode ended in a failure.
    """
    install = table["install_year"].to_numpy()
    failure = table["failure_year"].to_numpy()
    observed = np.where(np.isnan(failure), 0.0, 1.0)
    end = np.where(np.isnan(failure), float(study_end), failure)
    return end - install, table["entry_year"].to_numpy() - install, observed


def geometry_covariates(
    table: pl.DataFrame, length_ref_ft: float
) -> tuple[np.ndarray, list[str]]:
    """Builds the design matrix the effective-scale reduction predicts.

    Two columns, never one composite. Conductor count and length enter the
    reduction at different strengths — the first exactly, the second raised to
    the configured exponent — so a single column would force one coefficient
    onto two effects and recover neither.

    Length enters as a ratio to the reference rather than as raw feet, which
    makes the fitted intercept the scale of a single conductor at that
    reference, directly comparable to what the configuration sets.

    Args:
        table: The episode table.
        length_ref_ft: The reference length the configured scales describe.

    Returns:
        The design matrix and its column names.
    """
    return (
        np.column_stack(
            [
                np.log(table["n_conductors"].to_numpy().astype(float)),
                np.log(table["length_ft"].to_numpy() / length_ref_ft),
            ]
        ),
        ["log_n_conductors", "log_length_ratio"],
    )


def technology_indicators(
    table: pl.DataFrame, reference: str
) -> tuple[np.ndarray, list[str]]:
    """Builds reference-coded indicator columns for technology.

    One technology carries no column of its own. An intercept plus an indicator
    for every level is rank-deficient — the intercept and the coefficients are
    not separately identifiable, and a solver either fails or returns one of
    infinitely many answers that all fit equally well.

    Under this coding the fitted intercept is the reference technology's value
    and each coefficient is a log ratio against it.

    **A technology with no observed failure gets no coefficient, because there
    is none to get.** Its rows are all censored, so its scale enters the
    likelihood only through accumulated hazard, which shrinks without limit as
    the scale grows: the derivative stays strictly positive, no interior
    optimum exists, and the estimate runs to infinity. What a solver reports
    then is wherever it gave up. Refusing the fit is the only honest answer, so
    this raises rather than returning a column that cannot be estimated.

    Few failures is the same problem short of the boundary rather than a
    different one, and it is not caught here because no threshold separates the
    two. It is worth knowing what it looks like: at the shipped parameters the
    newest technology draws 25 failures from 77,151 rows, and its coefficient
    then varies across seeds by far more than its own value -- while any single
    run returns something that looks reasonable. Check the failure count per
    technology before believing a coefficient.

    Args:
        table: The episode table.
        reference: The technology to leave without a column.

    Returns:
        The indicator matrix and its column names.

    Raises:
        ValueError: If the reference technology does not appear in the table,
            or if any technology present has no observed failure.
    """
    present = sorted(set(table["technology"].to_list()))
    if reference not in present:
        raise ValueError(f"reference {reference!r} is not in the table: {present}")

    failures = (
        table.group_by("technology")
        .agg(pl.col("failure_year").is_not_null().sum().alias("failures"))
        .filter(pl.col("failures") == 0)["technology"]
        .to_list()
    )
    if failures:
        raise ValueError(
            f"no observed failure for {sorted(failures)}: their scale is not "
            f"identified and the fit would report wherever the solver stopped. "
            f"Drop these technologies, or widen the record window until they fail."
        )

    others = [name for name in present if name != reference]
    values = table["technology"].to_numpy()
    return (
        np.column_stack([(values == name).astype(float) for name in others]),
        [f"technology_{name}" for name in others],
    )
