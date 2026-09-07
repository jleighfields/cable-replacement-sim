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
    # 04 — What a replacement budget buys

    Cable fails; replacing it before it fails is cheaper and less disruptive
    than replacing it after. A utility cannot replace everything, so the
    question is not *whether* to replace but **which segments, and what does
    each extra dollar of that budget actually buy in reliability**.

    This notebook builds that answer from the bottom: score the candidates for
    one year, spend one year's budget, run a whole horizon, then sweep the
    budget and draw the curve. Each step calls the package rather than
    reimplementing it, so a step that reads badly here is a package interface
    to change rather than a notebook to rewrite.

    **Everything is synthetic.** The population is generated in-process from a
    seed. No observed utility data is used anywhere in this project.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 10 · A population to replace

    Segments differ in age, length, conductor count, what technology they are,
    and how many customers sit downstream. All of that has already collapsed
    into the handful of per-segment numbers the simulation reads — an effective
    Weibull pair, a customer count, customer-minutes per failure, and the value
    of lost load. That reduction is the reason adding a customer type or a
    technology touches no simulation code.

    A reduced size is used throughout, so this notebook runs in seconds. The
    replication count is stated with every figure, because a band whose
    replication count is unknown cannot be read.
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    from cablesim import config, metrics, plots, policies, population, results, run

    # The reduced size lives in the package, so this notebook, the sweep script
    # and anything else showing a figure are all drawn at the same one.
    N_REPS = run.REDUCED_REPS
    N_SEGMENTS = run.REDUCED_SEGMENTS

    # `resize_population` scales the customer denominator with the population. Left at
    # the system total, every reliability index below would be understated by
    # the population ratio -- a systematic bias rather than sampling noise, and
    # one that leaves the curves looking entirely plausible.
    settings = config.resize_population(config.load_config(), N_SEGMENTS, n_reps=N_REPS)
    segments = population.generate(settings)
    segments.select(
        "segment_id", "class", "technology", "age", "length_ft", "customers", "scale"
    ).head()
    return (
        N_REPS,
        N_SEGMENTS,
        config,
        metrics,
        np,
        pl,
        plots,
        policies,
        population,
        results,
        run,
        segments,
        settings,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 20 · Scoring one year's candidates

    A policy answers two questions about every segment: may it be replaced, and
    if several may, which first.

    The risk score has two terms and needs both. The first is customer value at
    risk this year. The second is the emergency premium the utility avoids by
    doing the work planned rather than after a failure — and without it, a
    lateral serving three customers is never replaced at any age, however far
    past the point where planned replacement is the cheaper of the two.
    """
    )
    return


@app.cell
def _(config, np, policies, segments, settings):
    risk = policies.resolve(config.PolicySpec(name="risk_ranked", params={}))
    planned_cost = policies.planned_cost(
        segments["length_ft"].to_numpy(),
        segments["cost_per_ft"].to_numpy(),
        settings.costs.mobilization_per_segment,
    )

    from cablesim import weibull

    failure_probability = weibull.conditional_failure_probability(
        segments["age"].to_numpy().astype(float),
        segments["shape"].to_numpy(),
        segments["scale"].to_numpy(),
    )
    score = policies.rank_key(
        risk,
        age=segments["age"].to_numpy().astype(float),
        failure_probability=failure_probability,
        outage_cost_per_failure=segments["outage_cost_per_failure"].to_numpy(),
        planned=planned_cost,
        emergency_multiplier=settings.costs.emergency_multiplier,
        priority=np.zeros(len(segments)),
    )
    value_at_risk_only = failure_probability * segments[
        "outage_cost_per_failure"
    ].to_numpy()
    risk, planned_cost, failure_probability, score, value_at_risk_only
    return (
        failure_probability,
        planned_cost,
        risk,
        score,
        value_at_risk_only,
        weibull,
    )


@app.cell
def _(np, score, segments, value_at_risk_only):
    # The claim above, measured on this population rather than asserted: the
    # avoided-cost term changes which segments the ranking puts first, and it
    # does so by promoting cheap segments serving few customers.
    _full_top = set(np.argsort(-score)[:200].tolist())
    _partial_top = set(np.argsort(-value_at_risk_only)[:200].tolist())
    _promoted = sorted(_full_top - _partial_top)
    _customers = segments["customers"].to_numpy()

    assert _promoted, (
        "the avoided-cost term changed nothing, so it is not doing its job"
    )
    assert _customers[_promoted].mean() < _customers[list(_partial_top)].mean(), (
        "the term should promote segments serving fewer customers, not more"
    )
    (
        f"{len(_promoted)} of the top 200 are there only because of the "
        f"avoided-cost term; they serve {_customers[_promoted].mean():.0f} "
        f"customers on average against "
        f"{_customers[list(_partial_top)].mean():.0f} for value at risk alone"
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 30 · Spending one year's budget

    Candidates are ranked, then funded down the list until the money runs out.
    The fill **stops at the first candidate that does not fit** rather than
    skipping it to fund cheaper ones below. That is what a capital plan does
    operationally, and it is also the only rule a vectorized implementation can
    reproduce — the skip-and-continue alternative is inherently sequential.
    Ranking per dollar is what compensates for a cheap candidate being passed
    over, and it does so in the ranking rather than in the fill.
    """
    )
    return


@app.cell
def _(np, planned_cost, policies, risk, score, segments, settings):
    eligible = policies.eligible(
        risk,
        segments["age"].to_numpy().astype(float),
        np.zeros(len(segments), dtype=bool),
    )
    ranked = policies.order_by_rank(score, np.flatnonzero(eligible))
    funded = policies.fund(ranked, planned_cost, settings.budget.annual)

    assert planned_cost[funded].sum() <= settings.budget.annual, "overspent"
    assert len(funded) < len(ranked), "the budget must bind, or the sweep shows nothing"
    (
        f"{len(funded)} of {len(ranked)} candidates funded, spending "
        f"{planned_cost[funded].sum():,.0f} of {settings.budget.annual:,.0f}"
    )
    return eligible, funded, ranked


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 40 · A whole horizon, every policy

    One run is one configuration evaluated for every policy, over the whole
    horizon, for every replication. Every policy reads the **same** random
    draws, so a comparison between two of them is a paired difference and
    carries far less noise than either side alone. That is the entire reason
    the draws are held fixed, and it is why the saved results keep their
    replication axis rather than being averaged as they are written.
    """
    )
    return


@app.cell
def _(pl, results, run, settings):
    import pathlib
    import tempfile

    # Read back into memory and then discarded. A notebook that runs headless
    # in a test as well as by hand should not leave a directory behind for
    # every execution.
    with tempfile.TemporaryDirectory(prefix="cablesim_04_") as _root:
        _directory = run.run(settings, pathlib.Path(_root))
        saved = pl.read_parquet(_directory / results.RESULTS_NAME)
    saved.head()
    return pathlib, saved, tempfile


@app.cell
def _(metrics, saved, settings):
    per_replication = metrics.indices_per_replication(
        saved.lazy(), settings.population.total_customers
    )
    banded = metrics.summarize_replications(
        per_replication, ["saidi", "saifi", "planned_spend", "emergency_spend"]
    ).collect()
    banded.head()
    return banded, per_replication


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 50 · What the policies do over thirty years

    The duration index is customer-minutes lost to **unplanned** interruptions
    per customer served. Planned work costs customer-minutes too, and they are
    reported separately rather than folded in: the indices are defined over
    unplanned interruptions, and mixing the two would leave their ratio
    dividing two different populations of outage.
    """
    )
    return


@app.cell
def _(N_REPS, banded, plots):
    plots.trajectory(
        banded,
        "saidi",
        f"Duration index over the horizon ({N_REPS} replications, mean and "
        f"10-90% band)",
        "Customer-minutes per customer",
    )
    return


@app.cell
def _(N_REPS, banded, plots):
    plots.trajectory(
        banded,
        "saifi",
        f"Interruption frequency ({N_REPS} replications, mean and 10-90% band)",
        "Interruptions per customer",
    )
    return


@app.cell
def _(plots, saved):
    plots.failures_by_class(saved, "risk_ranked")
    return


@app.cell
def _(banded, plots, run, settings):
    _budget = settings.budget.annual * run.escalation_series(
        settings.budget.escalation, settings.simulation.n_years
    )
    plots.spend_against_budget(banded, _budget.tolist(), "risk_ranked")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 60 · Sweeping the budget

    The deliverable. Each point is one budget level evaluated for every policy,
    and the grid is spaced geometrically so the region near a binding
    constraint is sampled more densely than the flat region beyond it.

    **Zero is in the grid deliberately.** At zero budget no policy funds
    anything, so every one of them must land on exactly the same point as
    run-to-failure. That is a free end-to-end check on the whole stack — the
    population, the draws, the scoring, the fill, the loop and the reduction —
    and it is asserted below rather than eyeballed.
    """
    )
    return


@app.cell
def _(config, metrics, pathlib, results, run, settings, tempfile):
    # Every level writes into one directory, which is what a sweep is, and the
    # package reads it back. Reading each run by hand instead would skip the
    # check that refuses a sweep missing a level, and a curve short one point
    # still draws.
    # Kept for the life of the cell only, so a headless run leaves nothing
    # behind.
    _sweep_dir = tempfile.TemporaryDirectory(prefix="cablesim_04_sweep_")
    _root = pathlib.Path(_sweep_dir.name)
    for _level in run.budget_grid(settings.budget.annual):
        run.run(
            config.with_overrides(settings, {"budget.annual": _level}),
            _root,
            swept={"annual_budget": _level},
        )

    # The swept column has to be named as a grouping key. Without it the levels
    # are summed together: nothing raises, and the frame keeps the shape of a
    # legitimate result.
    _keys = ("annual_budget",)
    sweep = metrics.horizon_totals(
        metrics.discount(
            metrics.indices_per_replication(
                results.read_sweep(_root),
                settings.population.total_customers,
                by=_keys,
            ),
            settings.costs.discount_rate,
        ),
        by=_keys,
    ).collect()
    _sweep_dir.cleanup()
    sweep.select("annual_budget", "policy", "customer_minutes", "failures").head(10)
    return (sweep,)


@app.cell
def _(pl, sweep):
    # Every policy at zero budget must equal run-to-failure at any budget,
    # because none of them funds anything there. Exact equality, not
    # approximate: they read the identical draws and take the identical path.
    _at_zero = sweep.filter(pl.col("annual_budget") == 0.0)
    assert _at_zero["customer_minutes"].n_unique() == 1, (
        "policies differ at zero budget, so something other than funding "
        f"differs between them: {_at_zero.select('policy', 'customer_minutes')}"
    )
    assert _at_zero["planned_replacements"].sum() == 0.0, "funded work at zero budget"

    # Run-to-failure never spends, so its curve must be flat across the sweep.
    _baseline = sweep.filter(pl.col("policy") == "run_to_failure")
    assert _baseline["customer_minutes"].n_unique() == 1, (
        "run_to_failure responded to a budget it never spends"
    )
    _lost = _at_zero["customer_minutes"][0]
    f"zero-budget check passed at {_lost:,.0f} customer-minutes"
    return


@app.cell
def _(plots, sweep):
    plots.reliability_against_budget(
        sweep.rename({"customer_minutes": "customer_minutes_lost"}),
        "annual_budget",
        "customer_minutes_lost",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 70 · Reading the curve

    Three things the figure should show, and each is asserted below rather than
    left to the eye.

    **Ranking by consequence beats ranking by probability alone.** The gap
    between the risk-ranked curve and the worst-first curve is what weighting
    by customers and cost is worth; worst-first is the control that isolates
    it.

    **Ranking at all beats spending at random.** The random policy spends every
    dollar the others do and buys far less, which is what makes it the control
    for the value of the ranking rather than of the spending.

    **An age threshold flattens.** The segments it may fund are a fixed subset,
    and it works through them, so its return falls away while the
    whole-population policies keep improving. Over the last step of this grid
    it buys under a fifth of what ranking by risk buys.

    And one the figure does not show, which falls out of the cost comparison:
    **preventive replacement pays for itself across this whole budget range.**
    Replacing a segment before it fails costs the planned price; letting it
    fail costs the emergency price, which is a multiple of it. Over every level
    from nothing to twice the configured budget, the premium avoided is larger
    than the planned work bought — so the policy is both cheaper and more
    reliable than run-to-failure, and the cost per customer-minute avoided is
    *negative* throughout.

    That does not make it free forever. The figure deepens and then turns back
    toward zero as the cheap opportunities are used up, so the break-even point
    exists; it simply sits above the range a budget of this size brackets.
    Beyond it the question stops being "should we do this at all" and starts
    being "how much is a customer-minute worth" — which is the question the
    placeholder value-of-lost-load figures below cannot yet answer.
    """
    )
    return


@app.cell
def _(metrics, settings, sweep):
    # What each policy avoids against the configured baseline, at each budget
    # level. Each level is measured against its own baseline row: comparing a
    # policy at one budget against run-to-failure at another would charge the
    # difference between two budgets to the difference between two policies.
    avoided = metrics.against_baseline(
        sweep.lazy(), settings.reporting.baseline_policy, by=("annual_budget",)
    ).collect()
    avoided.filter(avoided["customer_minutes_avoided"] > 0).select(
        "annual_budget",
        "policy",
        "customer_minutes_avoided",
        "additional_spend_discounted",
        "cost_per_customer_minute_avoided",
    ).sort("annual_budget", "policy").head(12)
    return (avoided,)


@app.cell
def _(avoided, pl):
    # Cost per customer-minute avoided is negative at every level in this grid:
    # across the whole range the configured budget brackets, preventive
    # replacement costs less in total than running to failure, because the
    # emergency premium it avoids exceeds the planned work it pays for. The
    # figure deepens and then turns back toward zero, so the diminishing return
    # is visible even though the break-even point sits above the top of the
    # grid.
    _priced = avoided.filter(
        (pl.col("policy") == "risk_ranked")
        & pl.col("cost_per_customer_minute_avoided").is_not_null()
    ).sort("annual_budget")
    _cost = _priced["cost_per_customer_minute_avoided"].to_list()

    assert len(_cost) >= 4, "too few priced levels to read a trend from"
    assert all(value < 0 for value in _cost), (
        "prevention should pay for itself at every level this grid covers, "
        f"against the emergency spend it avoids: {_cost}"
    )
    assert _cost[-1] > min(_cost), (
        "the return should be diminishing by the top of the grid, so the "
        f"cheapest level is not the last one: {_cost}"
    )
    assert (_priced["customer_minutes_avoided"].diff().drop_nulls() > 0).all(), (
        "more budget should avoid more customer-minutes"
    )
    _priced.select(
        "annual_budget",
        "customer_minutes_avoided",
        "additional_spend_discounted",
        "cost_per_customer_minute_avoided",
    )
    return


@app.cell
def _(pl, sweep):
    _wide = sweep.pivot(
        values="customer_minutes", index="annual_budget", on="policy"
    ).sort("annual_budget")
    _top = _wide.tail(1)

    assert _top["risk_ranked"].item() < _top["worst_first"].item(), (
        "weighting by consequence should beat ranking on probability alone"
    )
    assert _top["worst_first"].item() < _top["random"].item(), (
        "any ranking should beat spending at random"
    )
    assert _top["random"].item() < _top["run_to_failure"].item(), (
        "even random replacement should beat replacing nothing"
    )

    # The age threshold's return falls away long before the whole-population
    # policies' do, because the segments it may fund are a fixed subset and it
    # works through them. It has not run out entirely at the top of this grid,
    # so the claim asserted is the flattening rather than a flat line.
    _last_step = {
        _name: _wide[_name].to_list()[-2] - _wide[_name].to_list()[-1]
        for _name in ("age_threshold", "risk_ranked")
    }
    assert _last_step["age_threshold"] < _last_step["risk_ranked"] / 5, (
        "the age threshold's last budget step should buy far less than the "
        f"risk-ranked policy's: {_last_step}"
    )
    _wide
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 75 · The same answer, computed in Rust

    Everything above ran through the Python reference: the annual loop written
    to be checkable by reading, so that when two implementations disagree it is
    the one that arbitrates. The Rust kernel computes the same model from the
    same draws, and the two are interchangeable behind the same call — same
    argument names, same result type — so the run below differs from the run in
    section 40 only in which implementation it named.

    Every saved row must match. The draws are not regenerated and not compared:
    both implementations read the identical uniforms, so there is no
    random-number stream to reconcile across the two languages.

    **No timing is claimed here.** The reference is the correctness reference
    and not the benchmark baseline — it is written for readability, so timing
    the kernel against it would overstate what the kernel is worth. The
    benchmark notebook measures against a batched NumPy implementation instead,
    with the build profile and thread count stated.
    """
    )
    return


@app.cell
def _(pathlib, pl, results, run, saved, settings, tempfile):
    from cablesim import kernel

    with tempfile.TemporaryDirectory(prefix="cablesim_04_kernel_") as _root:
        _directory = run.run(settings, pathlib.Path(_root), implementation="kernel")
        from_kernel = pl.read_parquet(_directory / results.RESULTS_NAME)

    _order = ["policy", "replication", "year", "class"]
    assert saved.sort(_order).equals(from_kernel.sort(_order)), (
        "the kernel and the reference disagree on a saved run; the reference "
        "arbitrates, so this is a defect in the kernel until shown otherwise"
    )
    (
        f"{from_kernel.height:,} rows, identical in both implementations, "
        f"from a kernel compiled in {kernel.BUILD_PROFILE} mode on one thread"
    )
    return from_kernel, kernel


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 80 · What this is and is not

    The value-of-lost-load figures and the Weibull parameters in the shipped
    configuration are **placeholders**. The machinery is what is being
    demonstrated here, not a finding about any real fleet: change those numbers
    and the curves move, which is the point of having them in one validated
    configuration rather than scattered through the code.

    The figures above are drawn at a reduced size so this notebook runs in
    seconds, and the customer denominator is scaled with the population so the
    indices stay comparable with a full-size run. The driver script
    `scripts/budget_sweep.py` writes a sweep to disk, and takes `--full` for
    the shipped replication count and population.
    """
    )
    return


if __name__ == "__main__":
    app.run()
