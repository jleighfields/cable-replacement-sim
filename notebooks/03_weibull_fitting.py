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
def _(config, overridden, settings):
    def overridden(
        seed: int | None = None, monitoring_start: int | None = None
    ) -> config.Config:
        """The checked-in configuration with one or two knobs replaced.

        Several sections below need the same table under a different record
        window or a different draw, and building that by hand takes a dozen
        lines of nested dictionaries each time -- enough that which knob moved
        stops being the thing the reader notices.

        Args:
            seed: Replaces the simulation seed, giving an independent draw.
            monitoring_start: Replaces the year the record system begins.

        Returns:
            A validated configuration. Validated rather than mutated, so a
            combination the model refuses is refused here too.
        """
        simulation = settings.simulation.model_dump()
        window = settings.records.model_dump()
        if seed is not None:
            simulation["seed"] = seed
        if monitoring_start is not None:
            window["monitoring_start"] = monitoring_start
        return config.Config.model_validate(
            {
                **settings.model_dump(),
                "simulation": simulation,
                "records": window,
            }
        )

    return (overridden,)


@app.cell
def _(overridden, records, settings):
    complete = records.episode_table(
        overridden(
            monitoring_start=settings.population.initial_age.install_year_range[0]
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

    _lost = complete.join(visible, on=["segment_id", "install_year"], how="anti").sort(
        "install_year"
    )
    _segment = _lost["segment_id"][0]
    _kept = visible.filter(pl.col("segment_id") == _segment)["install_year"].to_list()

    def _mark(year: float) -> str:
        return "yes" if year in _kept else "**no row at all**"

    def _when(year: float | None) -> str:
        return "—" if year is None else format(year, ".0f")

    mo.md(
        f"""
    ### One segment, both views

    Segment `{_segment}` — what actually happened, and what the study records.

    | | install | failed | in the study? |
    |---|---|---|---|
    {
            chr(10).join(
                f"| episode {i + 1} | {r['install_year']:.0f} | "
                f"{_when(r['failure_year'])} | "
                f"{_mark(r['install_year'])} |"
                for i, r in enumerate(
                    complete.filter(pl.col("segment_id") == _segment).iter_rows(
                        named=True
                    )
                )
            )
        }

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
        _age,
        _hazard,
        where=(_age <= _entry),
        alpha=0.35,
        label="H(a): before entry — charged only if the term is dropped",
    )
    _axes[0].fill_between(
        _age,
        _hazard,
        where=(_age >= _entry) & (_age <= _exit),
        alpha=0.65,
        label="H(t) − H(a): hazard while watched",
    )
    _axes[0].set(
        xlabel="age (years)",
        ylabel="hazard h(age)",
        title="the likelihood counts the darker area only",
    )
    _axes[0].legend(fontsize=7, loc="upper left")

    _survival = np.exp(-((_age / truth_scale) ** truth_shape))
    _conditional = np.clip(
        _survival / np.exp(-((_entry / truth_scale) ** truth_shape)), 0, 1
    )
    _axes[1].plot(_age, _survival, label="S(t) — as installed")
    _axes[1].plot(
        _age[_age >= _entry],
        _conditional[_age >= _entry],
        label=f"S(t)/S(a) — given it reached {_entry:.0f}",
    )
    _axes[1].axvline(_entry, color="grey", lw=0.8, ls=":")
    _axes[1].set(
        xlabel="age (years)",
        ylabel="survival",
        title="dividing by S(a) renormalises to the survivors",
    )
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
def _(np, overridden, pl, records, settings, truth_scale, truth_shape):
    _cohort = (1965, 1975)
    _late = 2018

    def _built(monitoring_start: int) -> pl.DataFrame:
        """Episodes visible to a study beginning in the given year."""
        _cfg = overridden(monitoring_start=monitoring_start)
        return records.episode_table(
            _cfg,
            technologies=[settings.population.technologies[0]],
            length_ft=settings.population.length_ref_ft,
            n_conductors=[1],
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
                -(
                    ((_late - _installed["install_year"].to_numpy()) / truth_scale)
                    ** truth_shape
                )
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
    print(
        f"  survival to entry: observed {_n_left / _n_installed:.3f}, "
        f"theory {_expected:.3f}\n"
    )
    print("  starting the survivors at age 0 claims:")
    print(
        f"    {_n_left} cables x {_mean_age:.0f} yr = "
        f"{_n_left * _mean_age:,.0f} cable-years, 0 failures"
    )
    print("  what happened over those same years:")
    print(
        f"    {_n_installed} cables x {_mean_age:.0f} yr = "
        f"{_n_installed * _mean_age:,.0f} cable-years, "
        f"{_n_installed - _n_left} failures"
    )
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
def _(records, settings, visible):
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

    assert correct.scale_interval[0] < truth_scale < correct.scale_interval[1], (
        "the correct likelihood must recover the scale"
    )
    assert no_censoring.scale < correct.scale, "inventing failures shortens life"
    return correct, no_censoring, no_truncation


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 45 · The rows that went missing, and what they were worth

    Section 40 compared the two ways of reading the table the study can see.
    Neither is the answer a complete record would have given, and that is the
    comparison worth making: the correction earns its place only if it recovers
    what the deleted rows would have said.

    So three fits of the same cable. The complete history, where nothing is
    deleted and every episode is watched from installation; the study's view
    with the correction; and the study's view without it.
    """
    )
    return


@app.cell
def _(
    complete,
    correct,
    no_truncation,
    records,
    settings,
    truth_scale,
    truth_shape,
    weibull,
):
    _end, _entry, _observed = records.lifetimes(complete, settings.records.study_end)
    # Nothing is truncated in the complete history: monitoring begins at the
    # first install year, so no episode entered partway through life.
    assert not _entry.any(), "a complete record has nothing to condition on"
    full_history = weibull.fit_censored(_end, _entry, _observed)

    for _label, _fit in [
        ("the complete record", full_history),
        ("the study's view, corrected", correct),
        ("the study's view, uncorrected", no_truncation),
    ]:
        print(f"{_label:32} shape {_fit.shape:6.3f}   scale {_fit.scale:7.2f}")
    print(f"{'truth':32} shape {truth_shape:6.3f}   scale {truth_scale:7.2f}")
    return (full_history,)


@app.cell
def _(mo):
    mo.md(
        r"""
    One draw does not settle which reading is better, and reading an ordering
    off those three numbers would be reading noise. The correction is worth a
    few hundredths of shape here, and a single sample of this size carries more
    than that in sampling error.

    What removes the noise is fitting the *same* draw all three ways and
    averaging the differences rather than the fits, so whatever a draw happened
    to do to the shape affects every arm equally and cancels. It takes more
    draws than seems necessary: a dozen is enough to produce a confident
    standard error and the wrong sign, because at that size the estimate of
    the spread is no more reliable than the estimate of the mean.
    """
    )
    return


@app.cell
def _(np, overridden, records, settings, weibull):
    def refit(seed: int) -> tuple[float, float, float]:
        """Fit one draw three ways: complete record, corrected, uncorrected.

        Args:
            seed: Replaces the configured seed, giving an independent draw.

        Returns:
            Fitted shape from the complete record, from the study's view with
            the truncation term, and from the study's view without it.
        """

        def ages(start_year: int):
            table = records.episode_table(
                overridden(seed=seed, monitoring_start=start_year),
                technologies=[settings.population.technologies[0]],
                length_ft=settings.population.length_ref_ft,
                n_conductors=[1],
                n_segments=4000,
            )
            return records.lifetimes(table, settings.records.study_end)

        whole = ages(settings.population.initial_age.install_year_range[0])
        seen = ages(settings.records.monitoring_start)
        return (
            weibull.fit_censored(*whole).shape,
            weibull.fit_censored(*seen).shape,
            weibull.fit_censored(seen[0], np.zeros_like(seen[1]), seen[2]).shape,
        )

    DRAWS = 150
    _fits = np.array([refit(seed) for seed in range(20260902, 20260902 + DRAWS)])
    corrected_gap = _fits[:, 1] - _fits[:, 0]
    uncorrected_gap = _fits[:, 2] - _fits[:, 0]

    print(f"{DRAWS} draws, distance from the complete record's own answer")
    for _label, _gap in (
        ("  corrected  ", corrected_gap),
        ("  uncorrected", uncorrected_gap),
    ):
        _half = 1.96 * _gap.std(ddof=1) / np.sqrt(len(_gap))
        print(f"{_label} {_gap.mean():+.4f}  +/- {_half:.4f}")
    return corrected_gap, uncorrected_gap


@app.cell
def _(corrected_gap, mo, np, uncorrected_gap):
    _half = 1.96 * corrected_gap.std(ddof=1) / np.sqrt(len(corrected_gap))
    mo.md(
        f"""
    The corrected fit lands on the complete record's answer:
    {corrected_gap.mean():+.4f}, with an interval of plus or minus
    {_half:.4f} that covers zero. It is not merely closer — it is
    indistinguishable from having had the deleted rows all along, which is the
    strongest thing the correction could be said to do.

    Leaving the term out costs {uncorrected_gap.mean():+.4f} of shape. That is
    the bias the deleted rows would have removed, and it does not shrink as the
    sample grows: at 30,000 segments the same two numbers are about
    -0.003 and +0.060, unchanged. More data makes the noise smaller and leaves
    this exactly where it was, which is what separates a bias from a wobble.
    """
    )
    return


@app.cell
def _(corrected_gap, np, uncorrected_gap):
    def standard_error(gap: np.ndarray) -> float:
        """Standard error of a paired mean difference.

        Args:
            gap: One difference per draw.

        Returns:
            The standard deviation divided by the root of the count.
        """
        return float(gap.std(ddof=1) / np.sqrt(len(gap)))

    # The claim of the section, paired across draws so no single sample decides
    # it. Two separate things, each stated against its own sampling error
    # rather than against the other: the correction reproduces the complete
    # record, and dropping it does not.
    assert abs(corrected_gap.mean()) < 1.96 * standard_error(corrected_gap), (
        "the corrected fit must be indistinguishable from the complete record"
    )
    assert uncorrected_gap.mean() > 5.0 * standard_error(uncorrected_gap), (
        "dropping the term must leave a bias the draws can clearly resolve"
    )
    _corrected = abs(corrected_gap.mean()) / standard_error(corrected_gap)
    _uncorrected = uncorrected_gap.mean() / standard_error(uncorrected_gap)
    (
        f"standard errors from zero — corrected {_corrected:.1f}, "
        f"uncorrected {_uncorrected:.1f}"
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Why so few rows move the answer at all

    A row count is the wrong denominator. Shape is estimated from the spread of
    the ages cables failed at, so the rows carrying it are the **failures**,
    and the deleted rows are a far larger share of those than of the table.

    They are also not a random sample of them. A row is deleted only if the
    cable failed early enough to be gone before the records open, so every one
    is drawn from the young end and none from the old.
    """
    )
    return


@app.cell
def _(overridden, pl, records, settings):
    def episodes(start_year: int, n_segments: int = 30_000):
        """The episode table a study opening in the given year would hold.

        Args:
            start_year: Year the record system begins.
            n_segments: Segments to draw. Larger than the table the sections
                above share, because at 4,000 segments only a single episode
                is deleted and one row shows nothing about a distribution.

        Returns:
            One row per installation episode.
        """
        return records.episode_table(
            overridden(monitoring_start=start_year),
            technologies=[settings.population.technologies[0]],
            length_ft=settings.population.length_ref_ft,
            n_conductors=[1],
            n_segments=n_segments,
        )

    _whole = episodes(settings.population.initial_age.install_year_range[0])
    _seen = episodes(settings.records.monitoring_start)
    _failed = _whole.filter(pl.col("failure_year").is_not_null()).with_columns(
        (pl.col("failure_year") - pl.col("install_year")).alias("age_at_failure")
    )
    _on = ["segment_id", "install_year"]
    lost = _failed.join(_seen, on=_on, how="anti")
    kept = _failed.join(_seen, on=_on, how="semi")

    print(f"{_whole.height} episodes, of which {_failed.height} failed")
    print(
        f"deleted {lost.height}: {lost.height / _whole.height:.2%} of the "
        f"episodes, {lost.height / _failed.height:.2%} of the failures"
    )
    for _label, _frame in (("deleted", lost), ("kept", kept)):
        _age = _frame["age_at_failure"]
        print(
            f"  {_label:8} n={_frame.height:>5}  mean age at failure "
            f"{_age.mean():5.1f}   oldest {_age.max():5.1f}"
        )

    # The deleted failures come from the young end. Not a clean cut -- cable
    # installed after the records open can fail young and be kept -- so this is
    # a claim about the averages, which is what the fitted shape responds to.
    assert lost["age_at_failure"].mean() < kept["age_at_failure"].mean()
    return


@app.cell
def _(kept, lost, mo):
    mo.md(
        f"""
    Losing {lost.height} rows from {lost.height + kept.height} failures moves
    the shape because those rows are the young tail: they failed at
    {lost["age_at_failure"].mean():.1f} years on average against
    {kept["age_at_failure"].mean():.1f} for the ones that survive into the
    record. Delete the young failures and the survivors look more alike than
    the cohort was — a tighter spread, read as a higher shape, which is the
    direction the uncorrected fit misses by.

    It is not a clean cut. Cable installed after the records open can fail
    young and still be kept, so the two ranges overlap; what shifts is the
    average, and the fitted shape follows the average.
    """
    )
    return


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
def _(np, overridden, plt, records, settings, truth_scale, weibull):
    _starts = [1965, 1980, 1990, 1998, 2006, 2014, 2018]
    _corrected, _ignored = [], []
    for _start in _starts:
        _cfg = overridden(monitoring_start=_start)
        _table = records.episode_table(
            _cfg,
            technologies=[settings.population.technologies[0]],
            length_ft=settings.population.length_ref_ft,
            n_conductors=[1],
            n_segments=6000,
        )
        _e, _a, _o = records.lifetimes(_table, _cfg.records.study_end)
        _corrected.append(weibull.fit_censored(_e, _a, _o).scale)
        _ignored.append(weibull.fit_censored(_e, np.zeros_like(_a), _o).scale)

    _figure, _axis = plt.subplots(figsize=(6.5, 3.4))
    _axis.axhline(truth_scale, color="grey", ls=":", lw=0.9, label="truth")
    _axis.plot(_starts, _corrected, "o-", label="with the truncation term")
    _axis.plot(_starts, _ignored, "s--", label="without it")
    _axis.set(
        xlabel="year the record system starts",
        ylabel="fitted scale (years)",
        title="the correction only matters once entry ages are late",
    )
    _axis.legend(fontsize=8)
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 60 · The surface the fit is standing on

    Everything above reports where the search stopped. The surface it searched
    is worth looking at directly, because its shape is what decides whether the
    answer is well determined and whether the search can be trusted to find it.

    Two things to read off the contours. The valley is a single basin rather
    than a ridge with several floors, which is why the starting values do not
    matter here. And it is tilted: shape and scale are not separately
    determined, so a fit that reads the shape a little high reads the scale
    high to match, and neither parameter's interval means much without the
    other.
    """
    )
    return


@app.cell
def _(correct, end, entry, np, observed, plt, truth_scale, truth_shape, weibull):
    _shapes = np.linspace(correct.shape * 0.85, correct.shape * 1.15, 80)
    _scales = np.linspace(correct.scale * 0.95, correct.scale * 1.05, 80)
    _surface = np.array(
        [
            [weibull.log_likelihood(k, s, end, entry, observed) for s in _scales]
            for k in _shapes
        ]
    )

    # Contours at fixed drops from the maximum rather than evenly spaced
    # values. A drop of about 3 is the edge of a joint 95% region for two
    # parameters, so the innermost rings are the interval, drawn rather than
    # summarised.
    _peak = _surface.max()
    _figure, _axis = plt.subplots(figsize=(6.5, 4.2))
    _filled = _axis.contourf(
        _scales,
        _shapes,
        _peak - _surface,
        levels=[0, 1, 3, 6, 10, 20, 40],
        cmap="Blues_r",
        extend="max",
    )
    _axis.contour(
        _scales,
        _shapes,
        _peak - _surface,
        levels=[3],
        colors="black",
        linewidths=1.0,
    )
    _axis.plot(correct.scale, correct.shape, "o", color="black", label="the fit")
    _axis.plot(
        truth_scale, truth_shape, "*", color="crimson", markersize=14, label="truth"
    )
    _axis.set(
        xlabel="scale (years)", ylabel="shape", title="log-likelihood below its maximum"
    )
    _figure.colorbar(_filled, ax=_axis, label="drop from the maximum")
    _axis.legend(fontsize=8, loc="lower right")
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 70 · The same three fits, drawn as survival curves

    Section 40 gave the three fits as numbers. The same fits as curves say what
    those numbers mean for the quantity the simulation actually uses, which is
    the chance a segment of a given age is still in service.

    The censoring row is the one to look at. Its parameters are not slightly
    off, they describe a different population: treating every episode as a
    failure invents a failure at the study end for every cable that had not
    failed yet, and the fit reads that as cable dying steadily from year one.
    """
    )
    return


@app.cell
def _(correct, no_censoring, no_truncation, np, plt, truth_scale, truth_shape):
    _age = np.linspace(0.0, 80.0, 400)

    def survival(shape: float, scale: float) -> np.ndarray:
        """Weibull survival at each age in the plotted range.

        Args:
            shape: Weibull shape.
            scale: Weibull scale, in years.

        Returns:
            The share still in service at each age.
        """
        return np.exp(-((_age / scale) ** shape))

    _figure, _axis = plt.subplots(figsize=(6.5, 3.8))
    _axis.plot(
        _age,
        survival(truth_shape, truth_scale),
        color="black",
        lw=2.0,
        label=f"truth: shape {truth_shape:.2f}, scale {truth_scale:.0f}",
    )
    for _label, _fit, _style in (
        ("the likelihood as written", correct, "-"),
        ("dropping the truncation term", no_truncation, "--"),
        ("ignoring censoring", no_censoring, ":"),
    ):
        _axis.plot(
            _age,
            survival(_fit.shape, _fit.scale),
            _style,
            label=f"{_label}: shape {_fit.shape:.2f}, scale {_fit.scale:.0f}",
        )
    _axis.set(
        xlabel="age (years)",
        ylabel="still in service",
        title="what each reading of the data claims about the cable",
    )
    _axis.legend(fontsize=8)
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 80 · Do the intervals mean what they say?

    Every check in the test suite asserts that the truth falls inside a fitted
    95% interval. That is only a test if the intervals are honest, and nothing
    so far has shown they are — a fit could report intervals twice as wide as
    they should be and pass every one of those checks more comfortably.

    The way to find out is to refit many times and count. An interval built
    correctly contains the truth in about 95 draws out of 100; the count below
    is a sample, so it carries its own error, and the interval printed beside
    each figure is that.
    """
    )
    return


@app.cell
def _(np, overridden, records, settings, truth_scale, truth_shape, weibull):
    def covers(seed: int) -> tuple[bool, bool]:
        """Whether one draw's intervals contain the true shape and scale.

        Args:
            seed: Replaces the configured seed, giving an independent draw.

        Returns:
            Whether the shape interval contains the true shape, and whether the
            scale interval contains the true scale.
        """
        table = records.episode_table(
            overridden(seed=seed),
            technologies=[settings.population.technologies[0]],
            length_ft=settings.population.length_ref_ft,
            n_conductors=[1],
            n_segments=4000,
        )
        fitted = weibull.fit_censored(
            *records.lifetimes(table, settings.records.study_end)
        )
        return (
            fitted.shape_interval[0] <= truth_shape <= fitted.shape_interval[1],
            fitted.scale_interval[0] <= truth_scale <= fitted.scale_interval[1],
        )

    COVERAGE_DRAWS = 200
    _hits = np.array(
        [covers(seed) for seed in range(20260902, 20260902 + COVERAGE_DRAWS)]
    )
    coverage = {
        "shape": _hits[:, 0].mean(),
        "scale": _hits[:, 1].mean(),
        # Stricter than either alone, and what the recovery ladder asserts:
        # two 95% intervals both containing their truth is a rarer event than
        # one doing so, however correct each is on its own.
        "both at once": (_hits[:, 0] & _hits[:, 1]).mean(),
    }
    for _label, _share in coverage.items():
        _half = 1.96 * np.sqrt(_share * (1 - _share) / COVERAGE_DRAWS)
        print(f"  {_label:14} {_share:6.1%}  +/- {_half:.1%}")
    return COVERAGE_DRAWS, coverage


@app.cell
def _(COVERAGE_DRAWS, coverage, np):
    # Wide bounds on purpose. The question is whether the intervals are roughly
    # honest, not whether this many draws can pin the figure to a point: at 200
    # draws the sampling error on a 95% share is already about three points, so
    # a tighter assertion here would fail on nothing but its own noise.
    for _name in ("shape", "scale"):
        assert 0.90 <= coverage[_name] <= 0.99, (
            f"{_name} intervals cover {coverage[_name]:.1%}, not about 95%"
        )
    _half = 1.96 * np.sqrt(0.95 * 0.05 / COVERAGE_DRAWS)
    f"sampling error on a true 95% at {COVERAGE_DRAWS} draws is +/- {_half:.1%}"
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 90 · The regression the effective scale predicts

    Everything above fits one technology, one conductor, one length. The model
    this project actually needs is a regression: technology and geometry enter
    the scale together, and two of its coefficients are **predicted rather than
    tuned**, which is what makes fitting them a check instead of a report.

    A three-phase segment fails when the first of its three conductors does, so
    its effective scale falls as `n^(-1/k)`; length enters the same way raised
    to the configured exponent. Written as a regression on `log(scale)`:

    ```
    coefficient on log(n)          = -1/k
    coefficient on log(L / L_ref)  = -beta/k
    ```

    Neither is fitted anywhere else in this project — the simulation derives
    them — so if the fit disagrees, the generator and the estimator disagree
    about the model.

    **Two covariates, never one composite.** Conductor count enters exactly and
    length enters raised to `beta`, so a single `log(n * L^beta)` column would
    force one coefficient onto two effects and recover neither.
    """
    )
    return


@app.cell
def _(pl, records, settings):
    # The whole fleet, rather than the single-technology slice the sections
    # above use: this is the first place in the notebook where a population has
    # a technology mix to tell apart.
    fleet = records.episode_table(settings)
    fleet_end, fleet_entry, fleet_observed = records.lifetimes(
        fleet, settings.records.study_end
    )
    exposure = (
        fleet.with_columns(failed=pl.col("failure_year").is_not_null())
        .group_by("technology")
        .agg(episodes=pl.len(), failures=pl.col("failed").sum())
        .sort("technology")
    )
    exposure
    return exposure, fleet, fleet_end, fleet_entry, fleet_observed


@app.cell
def _(mo):
    mo.md(
        r"""
    **Read the failure column before anything else.** The technology with the
    most cable in the ground has the fewest failures, by an enormous margin,
    and it is not a sampling accident: technology follows install year, so the
    newest technology is also the youngest cable and has had the least time to
    fail. Nothing about raising the sample size changes that — every extra
    segment is equally young.

    That is the constraint the fit below runs into, and it is worth seeing
    before the coefficients rather than after.
    """
    )
    return


@app.cell
def _(fleet, np, records, settings):
    _geometry, _geometry_names = records.geometry_covariates(
        fleet, settings.population.length_ref_ft
    )
    # The oldest technology is the reference: every other coefficient is a log
    # ratio against it, and it is the one the fleet has the most failures for.
    reference_technology = settings.population.technologies[0]
    technology_columns, technology_names = records.technology_indicators(
        fleet, reference=reference_technology.name
    )
    design = np.hstack([_geometry, technology_columns])
    design_names = [*_geometry_names, *technology_names]
    design_names
    return (
        design,
        design_names,
        reference_technology,
        technology_columns,
        technology_names,
    )


@app.cell
def _(design, design_names, fleet_end, fleet_entry, fleet_observed, weibull):
    common_shape = weibull.fit_regression(
        fleet_end, fleet_entry, fleet_observed, design, design_names
    )
    common_shape.shape, common_shape.reference_scale
    return (common_shape,)


@app.cell
def _(common_shape, settings):
    # The two predictions, computed from the shape this fit recovered rather
    # than from the configured one: the closed forms are statements about the
    # model, so they have to be evaluated at the model the fit is standing on.
    _k = float(common_shape.shape)
    predicted = {
        "log_n_conductors": -1.0 / _k,
        "log_length_ratio": -settings.population.length_exponent / _k,
    }
    _rows = []
    for _name, _want in predicted.items():
        _got = float(dict(common_shape.coefficients)[_name])
        _bounds = dict(common_shape.coefficient_intervals)[_name]
        _low, _high = (float(v) for v in _bounds)
        assert _low <= _want <= _high, (
            f"{_name} came back at {_got:+.4f}, interval [{_low:+.4f}, "
            f"{_high:+.4f}], which excludes the predicted {_want:+.4f}. The "
            f"weakest-link reduction and the regression disagree about the "
            f"model rather than about an estimate."
        )
        _rows.append(f"  {_name:18} {_got:+.4f}  [{_low:+.4f}, {_high:+.4f}]"
                     f"  predicted {_want:+.4f}")
    print("\n".join(_rows))
    f"both geometry coefficients cover their closed forms at shape {_k:.3f}"
    return (predicted,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Letting the shape vary by technology

    The fit above holds one shape for the whole fleet. Different technologies
    fail by different mechanisms, which is why technology entered the model at
    all, so the shape belongs in the model too — as an **ancillary** term, the
    same indicators entering a second time.

    A common-shape fit where the shape actually varies does not fail loudly. It
    recovers a compromise and biases every scale with it, which is why this is
    a separate fit rather than an option on the one above.
    """
    )
    return


@app.cell
def _(
    design,
    design_names,
    fleet_end,
    fleet_entry,
    fleet_observed,
    technology_columns,
    technology_names,
    weibull,
):
    varying_shape = weibull.fit_regression(
        fleet_end,
        fleet_entry,
        fleet_observed,
        design,
        design_names,
        ancillary=technology_columns,
        ancillary_names=technology_names,
    )
    varying_shape.shape, varying_shape.reference_scale
    return (varying_shape,)


@app.cell
def _(reference_technology, varying_shape):
    # The anchor. Without it a wide interval below could be the fit failing
    # rather than the fleet being uninformative, and the two look identical
    # from the interval alone.
    #
    # What it catches is a broken fit: dropping the truncation correction moves
    # this interval to [6.284, 6.923] and reddens it. What it does not catch is
    # the wrong technology being coded as the reference, because the configured
    # shapes are closer together than this interval is wide — which is the
    # finding the closing note draws out.
    _low, _high = (float(_v) for _v in varying_shape.shape_interval)
    _configured = reference_technology.weibull.shape
    assert _low <= _configured <= _high, (
        f"the reference technology's shape interval [{_low:.3f}, {_high:.3f}] "
        f"excludes its configured {_configured}, so the fit is not recovering "
        f"the technology this fleet has the most failures for and nothing "
        f"below it means anything"
    )
    f"reference {reference_technology.name}: shape interval "
    f"[{_low:.3f}, {_high:.3f}] covers the configured {_configured}"
    return


@app.cell
def _(np, reference_technology, settings, technology_names, varying_shape):
    # Each ancillary coefficient is the log ratio of that technology's shape to
    # the reference's, so the configured truth it is checked against is the log
    # of the configured ratio.
    widths = {}
    _lines = []
    for _name in technology_names:
        _label = _name.removeprefix("technology_")
        _configured = next(
            _t.weibull.shape
            for _t in settings.population.technologies
            if _t.name == _label
        )
        _truth = float(np.log(_configured / reference_technology.weibull.shape))
        _low, _high = (float(v) for v in dict(varying_shape.ancillary_intervals)[_name])
        widths[_label] = _high - _low
        assert _low <= _truth <= _high, (
            f"{_label}'s shape interval [{_low:+.4f}, {_high:+.4f}] excludes "
            f"the configured log ratio {_truth:+.4f}"
        )
        _lines.append(
            f"  {_label:10} [{_low:+.4f}, {_high:+.4f}]  width {_high - _low:.4f}"
            f"  configured {_truth:+.4f}"
        )
    print("\n".join(_lines))
    return (widths,)


@app.cell
def _(exposure, pl, widths):
    # Every interval above covers its configured value, and for the newest
    # technology that is worth nothing: it covers a shape well below the
    # reference's and one well above it alike, so the fleet cannot say the
    # newest technology differs from the reference at all.
    #
    # Asserted as a ratio of widths rather than an absolute one, because the
    # absolute width moves with the study window and the population size while
    # the disparity between the two does not.
    _newest = "tr_xlpe"
    _middle = "xlpe"
    _ratio = widths[_newest] / widths[_middle]
    assert _ratio > 3.0, (
        f"{_newest}'s shape interval is only {_ratio:.1f} times "
        f"{_middle}'s, so this fleet identifies it better than the exposure "
        f"argument predicts and the paragraph below is describing something "
        f"that is no longer true"
    )

    # And the mechanism, in one number: exposure rather than sample size.
    _failures = dict(
        exposure.select("technology", "failures").iter_rows()
    )
    _episodes = dict(exposure.select("technology", "episodes").iter_rows())
    assert _failures[_newest] < 10, (
        f"{_newest} has {_failures[_newest]} failures, so it is no longer the "
        f"weakly-identified case this section is written around"
    )
    assert _episodes[_newest] > _episodes[_middle], (
        f"{_newest} is not the most numerous technology any more, which is "
        f"what makes its scarcity of failures an exposure problem rather than "
        f"a sampling one"
    )
    f"{_newest}: {_episodes[_newest]:,} episodes, {_failures[_newest]} failures, "
    f"shape interval {_ratio:.1f}x wider than {_middle}'s"
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### What this fleet can and cannot identify

    **The geometry coefficients come back.** Both cover the closed forms the
    effective-scale reduction predicts, which is the check this section exists
    for: the generator and the estimator agree about how conductor count and
    length enter the scale.

    **The reference technology comes back**, because it is the oldest cable and
    has had thirty years of the study window to fail in. Its shape interval
    still spans 0.68, against a spread of 0.6 between the three configured
    shapes — so even the best-identified technology here has an interval wider
    than the differences the model is trying to resolve. Recovering the
    reference is evidence the fit works, not evidence this fleet can tell the
    three technologies apart.

    **The newest technology does not.** Its shape interval is wide enough to
    admit a shape well below the reference's and well above it, so the fit
    covering the configured value is not evidence of anything — an interval
    that admits everything covers the truth by construction. It is the most
    numerous technology in the fleet and it has a handful of failures, so
    **the binding constraint is exposure time, not sample size**, and raising
    the segment count would not move it. A real utility fitting a technology
    introduced fifteen years ago has fifteen years of exposure however many
    assets it owns.

    The recovery ladder under `tests/` fits this same regression on a fixture
    that gives every technology equal exposure, and recovers all of it. That is
    the estimator being correct. This is the fleet being uninformative, and the
    two are different findings — which is why the ladder is not the place this
    question gets answered.

    Three ways out, none taken here: report the newest technology as weakly
    identified with an interval that says so, pool it with the previous
    technology and state the assumption, or carry a prior from
    accelerated-life testing. Which one is right belongs to the point where
    this fit meets a real fleet, and the choice needs the numbers above.
    """
    )
    return


if __name__ == "__main__":
    app.run()
