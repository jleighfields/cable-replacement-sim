import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # 01 — The synthetic population

    Every segment this project simulates is generated from a configuration and
    a seed; no observed utility data enters the repository. This notebook walks
    the interface that does it, one layer at a time, and ends by calling the
    top-level function and requiring it to agree with the composed steps.

    Walking the layers rather than making one call is deliberate. A notebook
    that calls `generate` and plots the answer teaches nothing about how the
    package is built, and it never reveals whether anything *but* the top-level
    function can be called at all.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## 10 · Configuration""")
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    from cablesim import config, population, streams, weibull

    settings = config.load_config()
    settings.population.n_segments, settings.simulation.seed
    return config, np, pl, population, settings, streams, weibull


@app.cell
def _(mo, settings):
    n_segments = mo.ui.slider(
        2_000,
        40_000,
        step=2_000,
        value=settings.population.n_segments,
        label="segments",
    )
    seed = mo.ui.slider(1, 50, value=1, label="seed offset")
    mo.hstack([n_segments, seed])
    return n_segments, seed


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 20 · Generate

    The population is a pure function of the validated configuration, so
    overriding a field and regenerating is the whole of a sweep.
    """
    )
    return


@app.cell
def _(config, n_segments, population, seed, settings):
    scenario = settings.model_copy(deep=True)
    scenario.population.n_segments = n_segments.value
    scenario.simulation.seed = settings.simulation.seed + seed.value

    segments = population.generate(scenario)
    segments.head(6)
    return scenario, segments


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 30 · Class shares

    Class is a multinomial draw, so the observed shares sit near the configured
    ones rather than on them. The assertion below is three standard errors of
    the sample share; an exact one would fail on a correct implementation.
    """
    )
    return


@app.cell
def _(np, pl, scenario, segments):
    observed = (
        segments.group_by("class")
        .len()
        .with_columns((pl.col("len") / segments.height).alias("share"))
    )

    for _cls in scenario.population.classes:
        _seen = observed.filter(pl.col("class") == _cls.name)["share"][0]
        _se = np.sqrt(_cls.share * (1 - _cls.share) / segments.height)
        assert abs(_seen - _cls.share) < 3 * _se, _cls.name

    observed.sort("share", descending=True)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 40 · Install year, and the technology that follows from it

    Install year is drawn from the configured volume curve, interpolated at
    every integer year and normalized. Technology is then a lookup on that
    year, which is what makes the link between age and hazard physical rather
    than assumed.
    """
    )
    return


@app.cell
def _(population, scenario):
    years, year_probabilities = population.install_year_distribution(
        scenario.population.initial_age
    )
    years[:5], year_probabilities[:5].round(5)
    return year_probabilities, years


@app.cell
def _(pl, segments, year_probabilities, years):
    import matplotlib.pyplot as plt

    _figure, _axes = plt.subplots(1, 2, figsize=(11, 3.2))
    _axes[0].plot(years, year_probabilities)
    _axes[0].set(xlabel="install year", ylabel="probability", title="configured curve")

    for _name in segments["technology"].unique().sort():
        _rows = segments.filter(pl.col("technology") == _name)
        _axes[1].hist(_rows["install_year"], bins=28, alpha=0.7, label=_name)
    _axes[1].set(xlabel="install year", title="drawn, by technology")
    _axes[1].legend()
    _figure.tight_layout()
    _figure
    return (plt,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 50 · Length and customers

    Lengths and customer counts are lognormal per class. The ordering that
    matters is by customers: main feeders serve far more than laterals, which
    is why failures on them dominate the reliability metrics and why a policy
    that weights consequence can beat one that does not.
    """
    )
    return


@app.cell
def _(pl, plt, segments):
    _order = ["lateral_1ph", "distribution_3ph", "main_feeder"]
    _figure, _axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for _name in _order:
        _rows = segments.filter(pl.col("class") == _name)
        _axes[0].hist(_rows["length_ft"], bins=40, alpha=0.6, label=_name)
        _axes[1].hist(_rows["customers"], bins=40, alpha=0.6, label=_name)
    _axes[0].set(xlabel="length (ft)", title="segment length")
    _axes[1].set(xlabel="customers", title="customers served", yscale="log")
    _axes[1].legend()
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(pl, segments):
    by_class = segments.group_by("class").agg(
        pl.len().alias("segments"),
        pl.col("length_ft").median().round(0).alias("length_median"),
        pl.col("customers").mean().round(1).alias("customers_mean"),
        pl.col("age").mean().round(1).alias("age_mean"),
    )

    _mean = dict(zip(by_class["class"], by_class["customers_mean"], strict=True))
    assert _mean["main_feeder"] > _mean["distribution_3ph"] > _mean["lateral_1ph"]

    by_class.sort("customers_mean", descending=True)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 60 · The effective-scale reduction

    A three-phase segment is out when any one of its conductors fails, so its
    lifetime is the minimum of three, and length adds to that sub-linearly. The
    reduction turns the technology's conductor-level pair into the one pair per
    segment that everything downstream reads.

    Laterals come out longest-lived. That is the minimum-of-three result rather
    than an assumption about lateral cable: both terms in the reduction favour
    short single-phase segments, and no choice of scale reorders them.
    """
    )
    return


@app.cell
def _(np, pl, segments):
    lives = segments.with_columns(
        (pl.col("scale") * np.log(2) ** (1 / pl.col("shape"))).alias("median_life")
    )

    lives.group_by(["class", "technology"]).agg(
        pl.col("median_life").median().round(0)
    ).pivot(on="technology", index="class", values="median_life").select(
        ["class", "hmwpe", "xlpe", "tr_xlpe"]
    )
    return


@app.cell
def _(np, scenario, segments, weibull):
    _single, _three = (
        segments.filter(
            (segments["technology"] == "xlpe") & (segments["n_conductors"] == _n)
        )["scale"].median()
        for _n in (1, 3)
    )
    assert _three < _single, "three-phase segments must not outlive single-phase"

    # The reduction against its closed form, at one segment.
    _row = segments.row(0, named=True)
    _expected = weibull.effective_scale(
        np.array(_row["scale"])
        * (
            _row["n_conductors"]
            * (_row["length_ft"] / scenario.population.length_ref_ft)
            ** scenario.population.length_exponent
        )
        ** (1 / _row["shape"]),
        np.array(_row["shape"]),
        np.array(_row["n_conductors"]),
        np.array(_row["length_ft"]),
        scenario.population.length_ref_ft,
        scenario.population.length_exponent,
    )
    float(_expected), _row["scale"]
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 70 · The four derived columns

    Customer types and restoration times are collapsed here, before anything
    runs, so the compute kernel receives flat per-segment arrays and never
    learns either exists. Adding a customer type changes the generator and the
    configuration and leaves the Rust side untouched.

    Note the planned-outage column: zero for a looped feeder, which is switched
    out without interrupting anyone, and not zero for a radial lateral, whose
    customers are out for the whole job.
    """
    )
    return


@app.cell
def _(pl, segments):
    derived = segments.group_by("class").agg(
        pl.col("customers").mean().round(1),
        pl.col("customer_minutes_per_failure").mean().round(0),
        pl.col("customer_minutes_per_planned").mean().round(0),
        pl.col("outage_cost_per_failure").mean().round(0),
    )

    _planned = dict(
        zip(derived["class"], derived["customer_minutes_per_planned"], strict=True)
    )
    assert _planned["main_feeder"] == 0.0
    assert _planned["lateral_1ph"] > 0.0

    derived.sort("outage_cost_per_failure", descending=True)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 80 · The steps against the whole

    Everything above called one layer at a time. This rebuilds four columns
    from those same layers and requires the top-level function to agree, which
    pins the convenience wrapper against the pieces it composes — a check
    nobody writes otherwise.
    """
    )
    return


@app.cell
def _(np, population, scenario, segments, streams, weibull):
    _n = scenario.population.n_segments
    _types = scenario.population.customer_types

    _source = streams.spawn_roots(scenario.simulation.seed).population
    _wide = 3 + len(_types)
    _draws = streams.uniforms(_source, _n * _wide).reshape(_wide, _n)

    _class_index = population.draw_categories(
        _draws[0], np.array([c.share for c in scenario.population.classes])
    )
    _years, _probabilities = population.install_year_distribution(
        scenario.population.initial_age
    )
    _install_year = _years[population.draw_categories(_draws[1], _probabilities)]

    _length = np.zeros(_n)
    for _index, _cls in enumerate(scenario.population.classes):
        _length[_class_index == _index] = population.lognormal_from_uniforms(
            _draws[2], _cls.length_ft
        )[_class_index == _index]

    assert (_class_index == segments["class_index"].to_numpy()).all()
    assert (_install_year == segments["install_year"].to_numpy()).all()
    assert np.allclose(_length, segments["length_ft"].to_numpy())

    _shape = segments["shape"].to_numpy()
    _reduced = weibull.effective_scale(
        segments["scale"].to_numpy()
        * (
            segments["n_conductors"].to_numpy()
            * (_length / scenario.population.length_ref_ft)
            ** scenario.population.length_exponent
        )
        ** (1 / _shape),
        _shape,
        segments["n_conductors"].to_numpy(),
        _length,
        scenario.population.length_ref_ft,
        scenario.population.length_exponent,
    )
    assert np.allclose(_reduced, segments["scale"].to_numpy())

    "the composed steps reproduce generate() on class, install year, length and scale"
    return


if __name__ == "__main__":
    app.run()
