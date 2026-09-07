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
    # 05 — Five programs, one model

    The same simulation is implemented five times: a scalar Python reference, a
    batched NumPy loop, a batched polars loop, a Rust kernel over slices, and a
    Rust loop over a polars frame. That is not redundancy — it is what
    validates the fast ones. When any two disagree, the scalar reference
    arbitrates, because it is the one written to be checkable by reading.

    This notebook asks the two questions that duplication exists to answer:

    1. **Do they agree?** Not approximately — in every cell of every array.
    2. **What does each cost?** With the population size, the replication
       count, the build profile and the thread count stated beside every
       number, because a timing without those cannot be checked against
       anything.

    The pairing matters as much as the list. `batched_numpy` and `kernel` are
    the array form of the loop in Python and in Rust; `batched_polars` and
    `kernel_polars` are the frame form in each. The Python polars package is an
    expression layer over the same compiled Rust engine the crate calls
    directly, so the second pair separates what it costs to *drive* that engine
    from Python from what the engine itself costs — which neither one alone can
    say.

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
    """
    )
    return


@app.cell
def _(config_module, constants, kernel, run):
    settings = config_module.resize_population(
        config_module.load_config(constants.DEFAULT_CONFIG_PATH),
        run.REDUCED_SEGMENTS,
        n_reps=run.REDUCED_REPS,
    )
    provenance = kernel.BUILD_PROFILE, kernel.AVAILABLE_THREADS
    print(
        f"{settings.population.n_segments} segments, "
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
    * **The reported totals are accumulated in the same order too**, which
      costs the polars implementations about 2% and buys a test with no
      threshold in it. A tolerance is what lets a real divergence pass
      unnoticed.
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
    configurations = [
        benchmarks.Configuration("reference", 1, reps),
        benchmarks.Configuration("batched_numpy", 1, reps),
        benchmarks.Configuration("batched_polars", 1, reps),
        benchmarks.Configuration("kernel", 1, reps),
        benchmarks.Configuration("kernel", kernel.AVAILABLE_THREADS, reps),
        benchmarks.Configuration("kernel_polars", 1, reps),
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
        "matches_reference",
        "speedup_over_batched_numpy",
    )
    return configurations, timings


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 50 · The two pairs

    The table above has more in it than a ranking. Reading it as two pairs is
    what answers the questions the duplication was built to answer.
    """
    )
    return


@app.cell
def _(pl, timings):
    def per_replication(name, threads=1):
        """Seconds per replication for one row of the table."""
        row = timings.filter(
            (pl.col("implementation") == name) & (pl.col("threads") == threads)
        )
        return row["seconds_per_replication"][0]

    array_form = per_replication("batched_numpy") / per_replication("kernel")
    frame_form = per_replication("batched_polars") / per_replication("kernel_polars")
    parallel = per_replication("kernel") / per_replication(
        "kernel", threads=timings["threads"].max()
    )
    frame_against_array = per_replication("batched_polars") / per_replication(
        "batched_numpy"
    )

    print(f"array form, Rust against Python:  {array_form:.2f}x")
    print(f"frame form, Rust against Python:  {frame_form:.2f}x")
    print(f"threads, within the Rust kernel:  {parallel:.2f}x")
    print(f"frame against array, in Python:   {frame_against_array:.2f}x")
    return (
        array_form,
        frame_against_array,
        frame_form,
        parallel,
        per_replication,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    **The frame-form ratio is the one worth dwelling on.** If driving the
    engine from Python were expensive, the Rust frame implementation would be
    far ahead of the Python one; they run the same query engine, so anything
    between them is the cost of getting there. A ratio near one says that cost
    is small and that whatever the frame form loses, it loses inside the
    engine rather than on the way in.

    That is a question the Python implementation could not answer on its own,
    and it is the reason the sixth implementation was written. A cheap answer
    was available — time the same expressions from both languages — and it
    would have told us about those expressions rather than about this
    workload.

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
