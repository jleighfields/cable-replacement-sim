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
    # 05 — Three programs, one model

    The same simulation is implemented three times: a scalar Python reference,
    a batched NumPy loop, and the Rust kernel. That is not redundancy — it is
    what validates the fast ones. When any two disagree, the scalar reference
    arbitrates, because it is the one written to be checkable by reading.

    This notebook asks the two questions that duplication exists to answer:

    1. **Do they agree?** Not approximately — in every cell of every array.
    2. **What does each cost?** With the population size, the replication
       count, the build profile and the thread count stated beside every
       number, because a timing without those cannot be checked against
       anything.

    The pairing matters as much as the list. `batched_numpy` and `kernel` are
    the same algorithm in the two languages, which is the comparison this
    project exists to make. The reference is shown for scale and is explicitly
    **not** the baseline a speedup is claimed against.

    Two further implementations over a polars frame, one in each language, were
    here and are now in `deprecated/`, with the measurements that retired them
    and the reason a column store is the wrong shape for a simulation.

    **Everything is synthetic.** The population is generated in-process from a
    seed.
    """
    )
    return


@app.cell
def _():
    import logging

    import numpy as np
    import polars as pl
    from cablesim import (
        benchmarks,
        constants,
        kernel,
        policies,
        run,
        simulate,
    )
    from cablesim import (
        config as config_module,
    )

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    return (
        benchmarks,
        config_module,
        constants,
        kernel,
        np,
        pl,
        policies,
        run,
        simulate,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 10 · What is being compared, and on what

    A reduced population, so this notebook runs while someone watches it. The
    ratios move with size — a fixed per-call cost matters more at 2,000
    segments than at 12,000 — so the size is stated with every figure and the
    full-size numbers are quoted separately at the end rather than implied by
    these.

    **The population is a control below**, defaulting to the reduced size so
    this opens in seconds. Raising it to 100,000 is worth doing once — it is
    eight times the shipped population, and nothing else in the test suite runs
    there. The scalar reference is what makes a large size expensive, so lower
    the replication count with it; the reference is here for scale rather than
    as a baseline, and a smaller count costs only precision.

    **Read the threading ratio against the replication count, not the
    population.** Replications are the axis being parallelised, so the speedup
    cannot exceed how many there are however many threads the machine has. At
    100,000 segments and 10 replications the kernel on forty-eight threads comes
    out 8.8 times faster than the same kernel on one — which is 88% of the
    ceiling of 10, not a sign that parallelism degrades with population. Against
    the fastest Python rather than against itself, the same run gives 9.5.
    Lowering replications to afford a larger population lowers that ceiling
    with it.
    """
    )
    return


@app.cell
def _(kernel, mo, run):
    # Defaults to the reduced size so the notebook opens in seconds and so the
    # headless run that gates nothing still finishes quickly — a slider takes
    # its default when nobody is there to move it.
    #
    # The large sizes are reachable at all because draws are computed from their
    # position rather than materialized: at 100,000 segments the old design
    # needed about 1.2 GB of uniforms per chunk, where the population itself is
    # under 10 MB.
    n_segments = mo.ui.dropdown(
        options={
            "2,000 — reduced, opens in seconds": run.REDUCED_SEGMENTS,
            "12,000 — the shipped population": 12_000,
            "100,000 — eight times the shipped size": 100_000,
        },
        value="2,000 — reduced, opens in seconds",
        label="population",
    )
    # The scalar reference is the binding cost at every size, and it is in the
    # table for scale rather than as a baseline, so a smaller count costs only
    # precision.
    #
    # It reaches twice the machine's thread count deliberately. Replications are
    # the axis being parallelised, so this number is the ceiling on the speedup
    # — below the thread count, threads sit idle and the ratio understates what
    # the kernel does; at two per worker there is also something left to steal
    # when one replication finishes before another.
    n_reps = mo.ui.slider(
        2,
        max(2 * kernel.AVAILABLE_THREADS, 50),
        step=2,
        value=run.REDUCED_REPS,
        label="replications",
    )
    mo.hstack([n_segments, n_reps])
    return n_reps, n_segments


@app.cell
def _(config_module, constants, kernel, n_reps, n_segments, run):
    settings = config_module.resize_population(
        config_module.load_config(constants.DEFAULT_CONFIG_PATH),
        n_segments.value,
        n_reps=n_reps.value,
    )
    provenance = kernel.BUILD_PROFILE, kernel.AVAILABLE_THREADS
    print(
        f"{settings.population.n_segments:,} segments, "
        f"{settings.simulation.n_years} years, "
        f"{settings.simulation.n_reps} replications"
    )
    print(f"Rust build profile: {provenance[0]}, threads available: {provenance[1]}")
    return provenance, settings


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 20 · One chunk, five ways

    Every implementation takes the same arguments and returns the same seven
    arrays. Build one chunk's arguments and hand that one set of objects to
    each, so "the same draws" is true by construction rather than by
    coincidence — there is no random-number stream to reconcile across the two
    languages, because there is only one.
    """
    )
    return


@app.cell
def _(benchmarks, policies, settings):
    arguments = benchmarks.chunk_arguments(settings, settings.simulation.n_reps)
    ranked = policies.resolve(
        next(spec for spec in settings.policies if spec.name == "risk_ranked")
    )
    print(f"{len(arguments)} arguments, shared by every implementation")
    print("policy: risk_ranked, which scores every segment every year")
    return arguments, ranked


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 30 · Do they agree?

    Cell for cell, with no tolerance anywhere. Two things make that achievable
    rather than lucky, and both are choices rather than accidents:

    * **Where a reduction feeds a decision, its order is pinned.** The year's
      emergency bill is subtracted from the budget, and the greedy fill then
      compares a cumulative cost against that budget — so a last-bit difference
      there decides which segment is funded last, which is a whole segment's
      worth of spend and reliability. Every implementation accumulates that
      total one element at a time.
    * **The reported totals are accumulated in the same order too.** Each
      implementation adds a class's contributions in the order the reference
      adds them — segment order for failures, rank order for funded work — so
      the totals match to the last bit and the check needs no threshold. A
      tolerance is what lets a real divergence pass unnoticed.
    """
    )
    return


@app.cell
def _(arguments, np, pl, ranked, run, simulate):
    def agreement_table():
        """One row per implementation: does it reproduce the reference exactly?"""
        expected = simulate.run_chunk(**arguments, policy=ranked)
        rows = []
        for name, annual_loop in run.RUNNABLE.items():
            produced = annual_loop(**arguments, policy=ranked)
            rows.append(
                {
                    "implementation": name,
                    "arrays_matching": sum(
                        np.array_equal(wanted, actual)
                        for wanted, actual in zip(expected, produced, strict=True)
                    ),
                    "arrays": len(simulate.Results._fields),
                    "exact": all(
                        np.array_equal(wanted, actual)
                        for wanted, actual in zip(expected, produced, strict=True)
                    ),
                }
            )
        return pl.DataFrame(rows)

    agreement = agreement_table()
    # The notebook's own claim, asserted where it is made rather than rebuilt
    # under `tests/`: every implementation reproduces the reference exactly.
    assert agreement["exact"].all(), agreement.filter(~agreement["exact"])
    agreement
    return (agreement,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 40 · What does each cost?

    The table below is the deliverable of this notebook. Each row is one
    implementation run a stated way; the columns are how long it took and
    whether it still reproduced the scalar Python reference.

    Three things about how to read it:

    * **The scalar reference is not the baseline.** It is in the table for
      scale. Claiming a speedup against a program written to be read rather
      than to be fast would flatter whatever it was compared with, so the ratio
      column is against the batched NumPy loop.
    * **Seconds per replication is the comparable column**, not total seconds,
      because rows may run different replication counts.
    * **Both the mean of several runs and the fastest are reported**, with the
      count beside them. Noise on a shared machine only ever adds, so the
      fastest run is the closest to what the work costs without interference;
      the mean is what someone actually waits for. Neither means anything
      without the repeat count.
    """
    )
    return


@app.cell
def _(benchmarks, kernel, settings):
    reps = settings.simulation.n_reps
    threads = kernel.AVAILABLE_THREADS
    configurations = [
        benchmarks.Configuration("reference", 1, reps),
        benchmarks.Configuration("batched_numpy", 1, reps),
        benchmarks.Configuration("batched_numpy", threads, reps),
        benchmarks.Configuration("kernel", 1, reps),
        benchmarks.Configuration("kernel", threads, reps),
    ]
    timings = benchmarks.compare(settings, "risk_ranked", configurations)
    # Timing an implementation that has drifted measures something else being
    # computed, so this is asserted here rather than read off the table.
    assert timings["matches_reference"].all(), timings.filter(
        ~timings["matches_reference"]
    )
    timings.select(
        "configuration",
        "replications",
        "seconds",
        "seconds_per_replication",
        "first_run_seconds",
        "matches_reference",
        "speedup_over_batched_numpy",
    )
    return configurations, timings


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 50 · What the table separates

    The table above has more in it than a ranking. Each row differs from
    another in exactly one respect, and reading those pairs off is what the
    duplication was built for.

    Three questions it answers, none of which a single ranking would:

    * **Does the array form pay off?** Compare the batched loop against the
      scalar reference it vectorises, on one thread. It wins where a policy
      funds nothing or everything and loses where few segments are eligible,
      because vectorising costs the ability to skip.
    * **Does the language matter?** Compare the kernel against the batched loop,
      both on one thread.
    * **Do threads matter, and can Python have them?** Compare each
      implementation against itself at one thread and at many. The batched loop
      is the interesting case: its arrays are operated on one at a time, and
      only some of those operations release the interpreter lock — which turns
      out not to be enough. A compiled Python implementation was measured here
      and reached the kernel; `deprecated/README.md` has that measurement and
      why it is not in this table.
    """
    )
    return


@app.cell
def _(pl, timings):
    def per_replication(name: str, threads: int = 1) -> float:
        """Seconds per replication for one row of the table.

        Args:
            name: The implementation, as ``run.RUNNABLE`` names it.
            threads: The thread count the wanted row was run at.

        Returns:
            That row's mean seconds per replication.
        """
        row = timings.filter(
            (pl.col("implementation") == name) & (pl.col("threads") == threads)
        )
        return row["seconds_per_replication"][0]

    widest = timings["threads"].max()

    array_form = per_replication("reference") / per_replication("batched_numpy")
    language = per_replication("batched_numpy") / per_replication("kernel")
    threads_kernel = per_replication("kernel") / per_replication("kernel", widest)
    threads_numpy = per_replication("batched_numpy") / per_replication(
        "batched_numpy", widest
    )

    print(f"the array form against the reference: {array_form:>7.2f}x")
    print(f"Rust against Python, both one thread: {language:>7.2f}x")
    print(f"threads, within the kernel:           {threads_kernel:>7.2f}x")
    print(f"threads, within the batched loop:     {threads_numpy:>7.2f}x")
    return (
        array_form,
        language,
        per_replication,
        threads_kernel,
        threads_numpy,
        widest,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    **The threading ratio is where the kernel's win actually is.** Single
    threaded, the kernel and the scalar reference are close for a policy that
    scores every segment, because the reference's NumPy expressions are already
    compiled loops over the same arrays. What Python cannot follow is the
    replication axis: it is embarrassingly parallel, and the interpreter lock is
    released for the whole computation.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 60 · At the full population

    The ratios above are measured at the reduced size so this notebook runs
    quickly. They move with size, because a fixed per-call cost is a larger
    share of a smaller run. `scripts/run_benchmarks.py` produces the same table
    at the shipped population, and its numbers are the ones any claim outside
    this notebook should quote.

    Run it with `uv run python scripts/run_benchmarks.py`, after
    `uv run maturin develop --release`.
    """
    )
    return


@app.cell
def _(mo, provenance):
    mo.md(
        f"""
    Everything above was measured on a **{provenance[0]}** build with
    **{provenance[1]}** threads available. A debug build is slower by a wide
    enough margin to make an unlabelled timing meaningless, which is why the
    profile is read from the compiled extension rather than assumed.
    """
    )
    return


if __name__ == "__main__":
    app.run()
