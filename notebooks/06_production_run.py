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
    # 06 — A production run, end to end

    The other notebooks take the package apart. This one runs it the way a
    study would: **the shipped configuration, unchanged, on the Rust kernel,
    across every worker the machine has**, and then reads back what it wrote.

    At its shipped size that is 12,000 segments, 1,000 replications and a
    30-year horizon for each of five policies — 30,000 simulated years per
    policy, 150,000 in total. The population size and replication count are
    parameters, so the same demonstration runs at a larger fleet unchanged.

    Every stage is timed, and the total is reported at the end, because the
    claim this project makes is that a study of this size is something you wait
    seconds for rather than schedule. A timing is only worth reading with its
    conditions attached, so the build profile, the thread count and the
    replication count are asserted and printed rather than assumed.
    """
    )
    return


@app.cell
def _():
    import json
    import logging
    import pathlib
    import tempfile
    import time

    import polars as pl
    from cablesim import config, constants, kernel, metrics, plots, results, run

    # A notebook is an entry point, so it may configure logging. A library
    # module may not, which is why no import above did.
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    started = time.perf_counter()
    return (
        config,
        constants,
        json,
        kernel,
        metrics,
        pathlib,
        pl,
        plots,
        results,
        run,
        started,
        tempfile,
        time,
    )


@app.cell
def _(time):
    def elapsed_since(mark: float) -> float:
        """Seconds since a `perf_counter` mark.

        Args:
            mark: The earlier `time.perf_counter()` reading.

        Returns:
            Seconds elapsed.
        """
        return time.perf_counter() - mark

    return (elapsed_since,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 10 · The configuration

    Loaded from `configs/base.yaml`, which is the documented default and the run
    everything else is a variation on. **The population size and the replication
    count are the two parameters this notebook exposes**, so the same end-to-end
    demonstration can be run at the shipped fleet or at a larger one without
    editing the base file — a study varies configuration from a driver rather
    than by editing the default.

    `resize_population` moves two system figures with the segment count, and
    missing either leaves a bias that looks like a result: `total_customers` is
    the denominator of both reliability indices, and `budget.annual` is the
    capital the whole system has to spend. Simulating four times the fleet
    against the original customer count and the original budget would overstate
    the indices and starve every policy that spends.
    """
    )
    return


@app.cell
def _(config, constants, kernel):
    # The two parameters. Change these to run the same demonstration at another
    # size; everything below reads them rather than the file.
    N_SEGMENTS = 12_000
    N_REPS = 1_000

    settings = config.resize_population(
        config.load_config(constants.DEFAULT_CONFIG_PATH), N_SEGMENTS, n_reps=N_REPS
    )
    threads = kernel.AVAILABLE_THREADS

    simulated_years = (
        settings.simulation.n_reps
        * settings.simulation.n_years
        * len(settings.policies)
    )
    print(f"segments      {settings.population.n_segments:>12,}")
    print(f"replications  {settings.simulation.n_reps:>12,}")
    print(f"horizon       {settings.simulation.n_years:>12,} years")
    print(f"policies      {len(settings.policies):>12,}")
    print(f"seed          {settings.simulation.seed:>12,}")
    print(f"threads       {threads:>12,}")
    print(f"total work    {simulated_years:>12,} simulated policy-years")
    return N_REPS, N_SEGMENTS, settings, simulated_years, threads


@app.cell
def _(kernel, threads):
    # A timing off a debug build means nothing, and a debug build is the most
    # likely reason a number here looks wrong. Asserted rather than printed and
    # left to the reader, because the whole notebook is a timing claim.
    assert kernel.BUILD_PROFILE == "release", (
        f"the kernel was built in {kernel.BUILD_PROFILE} mode; rebuild with "
        f"`uv run maturin develop --release` before reading any time below"
    )
    assert threads >= 1
    print(f"kernel built in {kernel.BUILD_PROFILE} mode, {threads} workers available")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 20 · The run

    `run.run` is the whole production path: it generates the population, runs
    every policy, and writes the rows, the effective configuration and a
    manifest into a directory named for the run.

    **The implementation is named rather than passed.** The manifest records
    that name as provenance, and handing over the callable separately would let
    a caller run one implementation and record another — a manifest that can be
    wrong is worse than none.

    Written into a temporary directory and read straight back, so running this
    notebook by hand or under the test suite leaves nothing behind.

    **The whole run goes to the kernel in one chunk, and nothing here asks for
    that.** `batch_size` splits the replications into separate calls and exists
    to bound memory: the batched NumPy loop holds `(replications, segments)`
    arrays and enough of them that its footprint is twelve to twenty times one
    of them — ten gigabytes at a thousand replications over a hundred thousand
    segments. The kernel holds nothing shaped that way; its per-worker scratch
    is sized by *segments*, and only the results array grows with replications,
    at 5 MB for this whole run.

    So each implementation carries its own default: a bound for the one that
    needs it, and none for the two that only lose the axis they parallelise
    over. `run.BATCH_SIZES` has them.
    """
    )
    return


@app.cell
def _(
    N_REPS,
    elapsed_since,
    json,
    pathlib,
    pl,
    results,
    run,
    settings,
    tempfile,
    threads,
    time,
):
    _mark = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="cablesim_06_") as _root:
        _directory = run.run(
            settings,
            pathlib.Path(_root),
            implementation="kernel",
            threads=threads,
        )
        saved = pl.read_parquet(_directory / results.RESULTS_NAME)
        manifest = json.loads(
            (_directory / results.MANIFEST_NAME).read_text(encoding="utf-8")
        )
    run_seconds = elapsed_since(_mark)

    print(f"{saved.height:,} rows in {run_seconds:.2f} s")
    saved.head()
    return manifest, run_seconds, saved


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 30 · What it wrote

    The manifest is the reason a saved run can be read a year later and still
    mean something: it carries which implementation produced the rows, the
    build profile it was compiled at, the thread count it spread over, and how
    long it took. The seed is not here — it is a configured value, and the
    effective configuration is written beside the rows for exactly that reason.

    **The run times itself**, so `wall_seconds` below is the figure a manifest
    carries into any later comparison, rather than something this notebook
    measured about it.
    """
    )
    return


@app.cell
def _(N_REPS, kernel, manifest, threads):
    for _field in (
        "implementation",
        "build_profile",
        "threads",
        "batch_size",
        "wall_seconds",
        "package_version",
        "git_commit",
        "git_dirty",
    ):
        print(f"{_field:<16} {manifest[_field]}")

    # The manifest is provenance, so what it claims has to be what ran. A
    # manifest that can be wrong is worse than no manifest.
    assert manifest["implementation"] == "kernel", manifest["implementation"]
    assert manifest["threads"] == threads, manifest["threads"]
    assert manifest["build_profile"] == kernel.BUILD_PROFILE, manifest["build_profile"]
    # The replication count itself, which is what "the kernel takes the whole
    # run in one chunk" means. Nothing above asks for it, so this is the only
    # thing here that says the resolved default is the one the prose describes.
    assert manifest["batch_size"] == N_REPS, manifest["batch_size"]
    return


@app.cell
def _(saved, settings):
    _classes = len(settings.population.classes)
    _expected = (
        len(settings.policies)
        * settings.simulation.n_reps
        * settings.simulation.n_years
        * _classes
    )
    # A saved row is one policy, replication, year and segment class — the class
    # axis is carried through to disk, because a system total cannot be
    # decomposed afterwards and the failure-by-class figures are drawn from it.
    # A row count that does not factor this way means a chunk was dropped or
    # counted twice, which no check on the values would notice.
    assert saved.height == _expected, f"{saved.height:,} rows, expected {_expected:,}"
    print(f"{saved.height:,} rows = {len(settings.policies)} policies x "
          f"{settings.simulation.n_reps:,} replications x "
          f"{settings.simulation.n_years} years x {_classes} classes")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 40 · Reliability, with its uncertainty

    The saved rows are per-class counts. `metrics.indices_per_replication` sums
    the class axis away and forms SAIFI and SAIDI against the configured
    system-wide customer count — a configured figure rather than a sum over
    segments, because the indices are defined against customers served.

    `summarize_replications` then reduces 1,000 replications to a mean and the
    quantile band the figure draws. **The band is the point of running 1,000 of
    anything**: a single replication is one draw from a distribution wide enough
    that two policies can trade places in it.
    """
    )
    return


@app.cell
def _(elapsed_since, metrics, saved, settings, time):
    _mark = time.perf_counter()
    per_replication = metrics.indices_per_replication(
        saved.lazy(), settings.population.total_customers
    )
    banded = metrics.summarize_replications(
        per_replication, ("saifi", "saidi")
    ).collect()
    metrics_seconds = elapsed_since(_mark)

    print(f"summarised {settings.simulation.n_reps:,} replications in "
          f"{metrics_seconds:.2f} s")
    banded.head()
    return banded, metrics_seconds, per_replication


@app.cell
def _(banded, plots):
    plots.trajectory(
        banded,
        "saifi",
        "Interruptions per customer, by policy",
        "SAIFI (interruptions per customer per year)",
    )
    return


@app.cell
def _(banded, plots):
    plots.trajectory(
        banded,
        "saidi",
        "Interruption duration per customer, by policy",
        "SAIDI (minutes per customer per year)",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 50 · What it cost

    The run figure is the one worth quoting, and it is what someone waits for:
    every policy, every replication, the draws each one reads, and the write to
    disk.

    The notebook total is larger, and most of the difference is the population
    generator and the censored fit that runs before any policy does — work a
    study pays once however many policies follow it.
    """
    )
    return


@app.cell
def _(
    elapsed_since,
    manifest,
    metrics_seconds,
    run_seconds,
    simulated_years,
    started,
    threads,
):
    total_seconds = elapsed_since(started)

    print(f"the run             {run_seconds:>8.2f} s  (manifest says "
          f"{manifest['wall_seconds']:.2f})")
    print(f"metrics             {metrics_seconds:>8.2f} s")
    print(f"notebook total      {total_seconds:>8.2f} s")
    print()
    print(f"{simulated_years / run_seconds:>,.0f} simulated policy-years per "
          f"second, on {threads} workers")
    print("generating the population and writing the rows are both inside that "
          "figure")

    # A guard rather than a target: this notebook exists to show the run is
    # something you wait seconds for, and if it ever takes minutes the claim in
    # its opening paragraph has stopped being true and should be rewritten
    # rather than quietly left standing.
    assert run_seconds < 120, (
        f"the production run took {run_seconds:.1f} s; this notebook claims a "
        f"study of this size is a seconds-long wait, so either the machine is "
        f"loaded or that claim needs retaking"
    )
    return (total_seconds,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 60 · What this demonstrates

    - **The shipped configuration runs end to end on the kernel**, and the
      manifest records what produced it rather than what was intended.
    - **1,000 replications is affordable**, which is what makes the uncertainty
      band above a band rather than a guess.
    - **Nothing here defines any modelling.** Every number came from the
      package; this notebook chose a configuration and called it, which is
      exactly what a driver script does.

    Notebook 05 answers the question this one does not ask — whether the kernel
    and the two Python implementations agree, and what each costs.
    """
    )
    return


if __name__ == "__main__":
    app.run()
