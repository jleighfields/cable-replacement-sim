# cable-replacement-sim

Simulation of underground cable failure and replacement policy for an electric
distribution utility, with a Rust compute kernel exposed to Python via
PyO3/maturin.

Censored-MLE Weibull failure model, three-phase segments governed by the
minimum of three conductor lifetimes, and budget-constrained replacement
policies evaluated against customer reliability (SAIFI / SAIDI / CMI) over a
30-year horizon. All inputs are parameterized via config; the cable population
is fully synthetic.

**Status: the simulation runs end to end in Python and in Rust.**
The configuration schema and its validators, the counter-based generator every
uniform is drawn from, the Weibull forms and the synthetic population generator
exist, along with the synthetic failure history and the censored, left-truncated
maximum-likelihood fit that recovers the parameters it was generated from.
Four marimo notebooks walk them. Every rung of the recovery ladder fits data
generated at parameters the configuration states, and checks the estimates come
back at them rather than checking that the generator and the estimator agree
with each other; the confidence intervals have been checked for coverage rather
than assumed.

The annual simulation loop, the five replacement policies, the budget-constrained
allocation, the reliability metrics and the shared figures all exist, and a
budget sweep draws the reliability-against-budget curve — at zero budget every
policy lands on the same point as run-to-failure, which is the end-to-end check
that costs nothing to run.

**The annual loop is implemented three times**, and that is the validation
strategy rather than duplication. A scalar Python reference written to be
checkable by reading; a batched NumPy loop that holds every replication in
flight at once; and a Rust kernel that spreads replications over worker
threads. All three take the same arguments, compute the same draws and return
the same result type, so any of them can be named at the call. No array of
uniforms is passed: each computes every draw from the run's key and the
position it is reading, and the two generators are held to agreeing bit for bit
against NumPy's own Philox.

Two further implementations over a polars frame, one in each language, were
built and retired; [deprecated/README.md](deprecated/README.md) has the
measurements and what they say about running a simulation on a column store.

They are compared by tests that force the lifetimes and check every cell, by a
paired test over drawn lifetimes, and by runs written to disk and diffed row for
row. **All three agree in every cell of every array**, with no tolerance
anywhere — which takes deliberate care rather than luck, because floating-point
addition is not associative and the year's emergency bill decides which segment
the budget reaches last.

What the timings say, at 12,000 segments over 30 years, on a release build with
48 cores available. Seconds per replication, mean of 3 runs, and **including the
cost of producing each run's random draws**, which is work a run actually does:

| Implementation | `run_to_failure` | `age_threshold` | `risk_ranked` |
|---|---|---|---|
| Scalar Python reference | 0.00747 | **0.02039** | 0.04425 |
| Batched NumPy | **0.00445** | 0.02426 | **0.04400** |
| Rust kernel, 1 thread | 0.00201 | 0.00262 | 0.04057 |
| **Rust kernel, 48 threads** | **0.00024** | **0.00026** | **0.00195** |
| **Fastest Python, beaten by** | **18.5x** | **78.4x** | **22.6x** |

The three policies differ in how much of the population they make eligible each
year — none, between 76 and 868 of 12,000, and all of it — and that matters more
than anything else here. Bold marks the fastest Python in each column, and it is
not always the same implementation: the batched loop wins where a policy funds
nothing or everything, and the scalar reference wins where few segments are
eligible, because the batched form still sorts every one of them.

**On one thread the kernel is not reliably faster than Python.** It is level
under `risk_ranked`, where every segment is a candidate and the array
expressions it competes with are already compiled loops over the same data. It
pulls ahead where the candidate set is small — 7.8x under `age_threshold` —
because it scores only the candidates. All three produce only the draws they
read, so that part of the work is the same in every row.

**The win is the replication axis.** They are independent, the interpreter lock
is released for the whole computation, and no Python implementation follows
without multiprocessing. That is a fact about the axis rather than about Rust;
what Rust contributes is that the compiler refused the first version, which
shared its working buffers between workers.

Run the table yourself with `uv run python scripts/run_benchmarks.py`, after
building with `--release`. See [PLAN.md](PLAN.md) for the model, the decisions
behind it, and the phased roadmap.

## Layout

| Path | What it holds |
|---|---|
| `src/` | the Rust crate: the compute kernel and the counter-based generator it draws from, built as one Python extension module |
| `python/cablesim/` | the Python package: configuration, the sources of randomness, the Weibull forms, the population generator, the synthetic failure history and its censored maximum-likelihood fit, the replacement policies, the annual loop that is the correctness reference for every other implementation, the batched NumPy loop and the wrapper that puts the kernel behind that same call, the benchmark harness, and the run, metrics and figure layers above it |
| `scripts/` | driver scripts that build configuration overrides and call the package in a loop; they hold no modelling logic |
| `configs/base.yaml` | the documented default run configuration |
| `notebooks/` | marimo notebooks that walk the package interface layer by layer |
| `tests/` | the test suite |
| `deprecated/` | retired implementations, kept with the measurements that retired them; nothing imports them and no test runs them |
| [PLAN.md](PLAN.md) | the model, the configuration schema, the kernel contract, and the roadmap |

The Python reference and everything else implement the same model repeatedly.
That is deliberate: the parity tests between them are what validate the fast
ones, so no side is redundant. When any two disagree, the reference arbitrates,
because it is the one written to be checkable by reading.

## Getting started

Requires [rustup](https://rustup.rs) and [uv](https://docs.astral.sh/uv/).
Both versions are pinned in the repository — Python in `.python-version`, Rust
in `rust-toolchain.toml` — and each tool installs what its file names on first
use, so neither needs choosing.

```bash
uv sync --extra plots              # create the environment
uv run maturin develop --release   # build and install the extension module
uv run pytest                      # run the suite
```

Plotting is an extra rather than a dependency, so installing this package for
the compute kernel alone does not pull a plotting stack. `cablesim.plots`
imports plotly at module scope and fails at the import without it, saying what
is missing. Without `--extra plots` the default suite loses
`tests/test_plots.py` to a collection error, and the notebook suite loses
notebook 04, which imports the module. **`uv sync` installs exactly what it is
asked for and removes the rest**, so every later sync has to name the extra
again — `uv sync --extra plots --group notebooks`, not `uv sync --group
notebooks`, which uninstalls plotly.

`--release` is not optional for anything timed: the parity tests run many
replications, and a debug build is slow enough to dominate the run. Rebuild
after any change under `src/`, or the suite reports on the extension module
that is currently installed rather than the one you just edited. A run through
the kernel records which profile it was compiled with, read from the binary
rather than assumed, so a debug build shows up in the saved manifest instead of
being discovered later.

The sweep script runs any of the implementations, and takes a thread count:

```bash
uv run python scripts/budget_sweep.py                              # the Rust kernel
uv run python scripts/budget_sweep.py --threads 8                  # over 8 workers
uv run python scripts/budget_sweep.py --implementation reference   # the Python reference
```

Only the Rust kernel uses more than one thread. Every other implementation
refuses a larger count rather than ignoring it, so a saved run cannot record a
thread count that nothing acted on.

The benchmark script times all three — with the kernel appearing twice, at one
thread and at the machine's full count — and checks each against the reference
in the same pass, because timing an implementation that has drifted measures
something else being computed:

```bash
uv run python scripts/run_benchmarks.py            # the shipped population
uv run python scripts/run_benchmarks.py --reduced  # while someone watches
```

## Testing

```bash
uv run pytest -n auto              # the default set, in parallel — what CI runs
uv run ruff check .                # Python lint
cargo fmt --check                  # Rust format
cargo clippy --all-targets --no-default-features -- -D warnings
cargo test --no-default-features
```

Two suites are excluded from a default run because each costs minutes:

```bash
uv sync --extra plots --group notebooks && uv run pytest -m notebooks  # every notebook
uv sync --extra plots --group app && uv run pytest -m app              # the app, in a browser
```

Both name `--extra plots` as well as their group, because `uv sync` removes
whatever it is not asked for. The notebook suite needs it because notebook 04
imports `cablesim.plots`; the app command needs it so that a later default
`pytest` still collects `tests/test_plots.py`, which imports the same module.

Neither gates a merge. The app suite runs weekly and on pushes to `main`, so a
break in it surfaces within a week. **Nothing runs the notebooks for you** —
they execute only when someone runs the command above, so run it after changing
any package API a notebook imports.

`cargo test` needs `--no-default-features` because the default build enables
PyO3's `extension-module`, and a test binary linked against it cannot resolve
the CPython symbols the interpreter supplies at import time.

## Reading the plan

`PLAN.md` is long and heavily cross-referenced. It renders to HTML with a
table of contents:

```bash
quarto render PLAN.md      # writes PLAN.html; both it and PLAN_files/ are ignored
```

The table of contents comes from the front matter, so there is none to
maintain in the file itself. A test checks that every section citation in the
document resolves to a section that exists.
