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
    # 03 — Fitting censored, truncated lifetimes

    Cable that has not failed yet still tells you something, and cable that
    failed before anyone was keeping records tells you nothing at all. Handling
    those two facts correctly is most of what separates a usable estimate from
    a badly optimistic one.

    This notebook walks the likelihood a term at a time, then shows what each
    term is worth by removing it.
    """
    )
    return


@app.cell
def _():
    import numpy as np
    from cablesim import config, records, weibull

    settings = config.load_config()
    technology = settings.population.technologies[0]
    truth_shape = technology.weibull.shape
    truth_scale = technology.weibull.scale
    truth_shape, truth_scale
    return (
        config,
        np,
        records,
        settings,
        technology,
        truth_scale,
        truth_shape,
        weibull,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 10 · The study, and what it can and cannot see

    A **prospective follow-up**. The utility inventories what is in the ground
    at the monitoring date, taking install dates from the asset register, and
    records failures from then until the study end.

    Three things follow, and the third is the one that causes trouble.
    """
    )
    return


@app.cell
def _(config, records, settings):
    complete = records.episode_table(
        config.Config.model_validate(
            {
                **settings.model_dump(),
                "records": {
                    **settings.records.model_dump(),
                    "monitoring_start": (
                        settings.population.initial_age.install_year_range[0]
                    ),
                },
            }
        ),
        technologies=[settings.population.technologies[0]],
        length_ft=settings.population.length_ref_ft,
        n_conductors=[1],
        n_segments=4000,
    )
    visible = records.episode_table(
        settings,
        technologies=[settings.population.technologies[0]],
        length_ft=settings.population.length_ref_ft,
        n_conductors=[1],
        n_segments=4000,
    )
    (
        f"everything that happened: {complete.height} episodes"
        f" | the study sees: {visible.height}"
    )
    return complete, visible


@app.cell
def _(complete, mo, settings, visible):
    import polars as pl

    _lost = complete.join(
        visible, on=["segment_id", "install_year"], how="anti"
    ).sort("install_year")
    _segment = _lost["segment_id"][0]
    _kept = visible.filter(pl.col("segment_id") == _segment)["install_year"].to_list()

    def _mark(year: float) -> str:
        return "yes" if year in _kept else "**no row at all**"
    mo.md(
        f"""
    ### One segment, both views

    Segment `{_segment}` — what actually happened, and what the study records.

    | | install | failed | in the study? |
    |---|---|---|---|
    {chr(10).join(
        f"| episode {i+1} | {r['install_year']:.0f} | "
        f"{'—' if r['failure_year'] is None else format(r['failure_year'], '.0f')} | "
        f"{_mark(r['install_year'])} |"
        for i, r in enumerate(
            complete.filter(pl.col("segment_id") == _segment).iter_rows(named=True)
        )
    )}

    The first episode is the **missing row**. Not a row with an unknown age —
    no row. Nothing in the study's data even says the second cable is a
    replacement rather than an original install.

    The second is in the table because it was still running at
    {settings.records.monitoring_start}. It is **right-censored** at its exit
    and **left-truncated** at its entry, both at once.
    """
    )
    return (pl,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 20 · What each term of the likelihood is

    One episode contributes

    $$
    \underbrace{\delta \log h(t)}_{\text{it failed, here}}
    \;-\;
    \underbrace{\big[H(t) - H(a)\big]}_{\text{hazard while watched}}
    $$

    $H(t)$ is cumulative hazard from age $0$ to $t$, so $H(t) - H(a)$ is the
    hazard accumulated **between entry and exit** — the stretch the study was
    actually looking.

    Drop the $-H(a)$ and you charge the episode for hazard back to age zero,
    including years before it entered. A failure in those years would have
    **removed it from the sample** rather than appearing in it, so that is
    exposure with no possible failure attached. The model sees a lot of
    exposure and few failures, reads the hazard as low, and reports a long
    life.
    """
    )
    return


@app.cell
def _(np, truth_scale, truth_shape):
    import matplotlib.pyplot as plt

    _entry, _exit = 20.0, 45.0
    _age = np.linspace(0.1, 60, 500)
    _hazard = (truth_shape / truth_scale) * (_age / truth_scale) ** (truth_shape - 1)

    _figure, _axes = plt.subplots(1, 2, figsize=(11, 3.5))
    _axes[0].plot(_age, _hazard, color="black", lw=1)
    _axes[0].fill_between(
        _age, _hazard, where=(_age <= _entry), alpha=0.35,
        label="H(a): before entry — charged only if the term is dropped",
    )
    _axes[0].fill_between(
        _age, _hazard, where=(_age >= _entry) & (_age <= _exit), alpha=0.65,
        label="H(t) − H(a): hazard while watched",
    )
    _axes[0].set(xlabel="age (years)", ylabel="hazard h(age)",
                 title="the likelihood counts the darker area only")
    _axes[0].legend(fontsize=7, loc="upper left")

    _survival = np.exp(-((_age / truth_scale) ** truth_shape))
    _conditional = np.clip(
        _survival / np.exp(-((_entry / truth_scale) ** truth_shape)), 0, 1
    )
    _axes[1].plot(_age, _survival, label="S(t) — as installed")
    _axes[1].plot(_age[_age >= _entry], _conditional[_age >= _entry],
                  label=f"S(t)/S(a) — given it reached {_entry:.0f}")
    _axes[1].axvline(_entry, color="grey", lw=0.8, ls=":")
    _axes[1].set(xlabel="age (years)", ylabel="survival",
                 title="dividing by S(a) renormalises to the survivors")
    _axes[1].legend(fontsize=8)
    _figure.tight_layout()
    _figure
    return (plt,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 25 · Why knowing the install date is not enough

    The obvious objection: the register gives the install date, so the cable's
    age is known from the day it went in — why not start the clock at zero and
    call it right-censored?

    Because **knowing the age is not the same as having observed those years.**
    Exposure counts only if a failure during it would have been recorded, and
    a failure before the inventory would have deleted the cable from it. The
    cohort below makes the size of that gap visible.
    """
    )
    return


@app.cell
def _(config, np, pl, records, settings, truth_scale, truth_shape):
    _cohort = (1965, 1975)
    _late = 2018

    def _built(monitoring_start: int) -> pl.DataFrame:
        """Episodes visible to a study beginning in the given year."""
        _cfg = config.Config.model_validate(
            {**settings.model_dump(),
             "records": {**settings.records.model_dump(),
                         "monitoring_start": monitoring_start}}
        )
        return records.episode_table(
            _cfg, technologies=[settings.population.technologies[0]],
            length_ft=settings.population.length_ref_ft, n_conductors=[1],
            n_segments=20_000,
        )

    _in_cohort = pl.col("install_year").is_between(*_cohort)
    _installed = _built(_cohort[0]).filter(_in_cohort)
    _inventoried = _built(_late).filter(_in_cohort)

    _n_installed, _n_left = _installed.height, _inventoried.height
    _mean_age = float(_late - _inventoried["install_year"].mean())
    _expected = float(
        np.mean(
            np.exp(
                -(((_late - _installed["install_year"].to_numpy()) / truth_scale)
                  ** truth_shape)
            )
        )
    )

    # The survivors are a biased sample of the cohort, by construction.
    assert _n_left < _n_installed
    assert abs(_n_left / _n_installed - _expected) < 0.1

    print(f"cable installed {_cohort[0]}-{_cohort[1]}, study starting {_late}")
    print(f"  actually installed           : {_n_installed}")
    print(f"  present in the inventory     : {_n_left}")
    print(f"  missing, failed beforehand   : {_n_installed - _n_left}")
    print(f"  survival to entry: observed {_n_left / _n_installed:.3f}, "
          f"theory {_expected:.3f}\n")
    print("  starting the survivors at age 0 claims:")
    print(f"    {_n_left} cables x {_mean_age:.0f} yr = "
          f"{_n_left * _mean_age:,.0f} cable-years, 0 failures")
    print("  what happened over those same years:")
    print(f"    {_n_installed} cables x {_mean_age:.0f} yr = "
          f"{_n_installed * _mean_age:,.0f} cable-years, "
          f"{_n_installed - _n_left} failures")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The survivors cannot speak for the cohort they came from. Crediting their
    early years as failure-free exposure asserts a fact — *none of these failed
    young* — that was never observed and is not true of the cohort.

    So the clock starts when the cable came under observation, not when it went
    in the ground. That is the whole of what $-H(a)$ enforces, and the install
    date is what makes the starting age computable rather than what makes it
    zero.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 30 · A missing row is never a term

    Worth stating plainly, because it is the question the algebra invites: the
    correction does **not** put the missing episode back. It cannot — the row
    left no trace, and nothing can reconstruct a row from its own absence.

    The sum has one term per row **present**, and never more. What the $+H(a)$
    changes is the question asked of the rows that are there:

    - unconditional — *what is the chance this cable did what it did?*
    - conditional — *given it reached age $a$, what is the chance it did what
      it did?*

    Episodes that failed before $a$ drop out of that conditional probability by
    construction. That is why nothing here has to count them, or know anything
    about them at all.
    """
    )
    return


@app.cell
def _(np, records, settings, visible, weibull):
    end, entry, observed = records.lifetimes(visible, settings.records.study_end)

    assert len(end) == visible.height, "one term per visible row, never more"
    assert (entry[entry > 0] > 0).all()

    (
        f"{visible.height} rows in, {len(end)} terms in the sum, "
        f"{int((entry > 0).sum())} of them truncated"
    )
    return end, entry, observed


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 40 · What each term is worth, by removing it

    Two terms, two ways to be wrong, and they pull in opposite directions.
    """
    )
    return


@app.cell
def _(end, entry, np, observed, truth_scale, truth_shape, weibull):
    correct = weibull.fit_censored(end, entry, observed)
    no_truncation = weibull.fit_censored(end, np.zeros_like(entry), observed)
    # Treating every episode as a failure: the censored ones did not fail, so
    # this invents failures at the study end and reads the hazard as high.
    no_censoring = weibull.fit_censored(end, entry, np.ones_like(observed))

    _rows = [
        ("the likelihood as written", correct),
        ("dropping the truncation term", no_truncation),
        ("ignoring censoring", no_censoring),
    ]
    for _label, _fit in _rows:
        print(f"{_label:32} shape {_fit.shape:6.3f}   scale {_fit.scale:7.2f}")
    print(f"{'truth':32} shape {truth_shape:6.3f}   scale {truth_scale:7.2f}")

    assert (
        correct.scale_interval[0] < truth_scale < correct.scale_interval[1]
    ), "the correct likelihood must recover the scale"
    assert no_censoring.scale < correct.scale, "inventing failures shortens life"
    return correct, no_censoring, no_truncation


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 50 · How much the truncation term is worth, and when

    $H(a) = (a/\lambda)^k$, so at a sharp shape an episode that entered young
    carries almost no accumulated hazard and the correction is nearly free. It
    bites when entry ages approach the scale — which is to say, when the record
    system starts long after the cable went in.

    The term is kept because a record system beginning after the installs is
    the ordinary case, not because it rescues this particular configuration.
    """
    )
    return


@app.cell
def _(config, np, plt, records, settings, truth_scale, weibull):
    _starts = [1965, 1980, 1990, 1998, 2006, 2014, 2018]
    _corrected, _ignored = [], []
    for _start in _starts:
        _cfg = config.Config.model_validate(
            {**settings.model_dump(),
             "records": {**settings.records.model_dump(), "monitoring_start": _start}}
        )
        _table = records.episode_table(
            _cfg, technologies=[settings.population.technologies[0]],
            length_ft=settings.population.length_ref_ft, n_conductors=[1],
            n_segments=6000,
        )
        _e, _a, _o = records.lifetimes(_table, _cfg.records.study_end)
        _corrected.append(weibull.fit_censored(_e, _a, _o).scale)
        _ignored.append(weibull.fit_censored(_e, np.zeros_like(_a), _o).scale)

    _figure, _axis = plt.subplots(figsize=(6.5, 3.4))
    _axis.axhline(truth_scale, color="grey", ls=":", lw=0.9, label="truth")
    _axis.plot(_starts, _corrected, "o-", label="with the truncation term")
    _axis.plot(_starts, _ignored, "s--", label="without it")
    _axis.set(xlabel="year the record system starts", ylabel="fitted scale (years)",
              title="the correction only matters once entry ages are late")
    _axis.legend(fontsize=8)
    _figure.tight_layout()
    _figure
    return


if __name__ == "__main__":
    app.run()
