# cable-replacement-sim

Simulation of underground cable failure and replacement policy for an electric
distribution utility, with a Rust compute kernel exposed to Python via
PyO3/maturin.

Censored-MLE Weibull failure model, three-phase segments governed by the
minimum of three conductor lifetimes, and budget-constrained replacement
policies evaluated against customer reliability (SAIFI / SAIDI / CMI) over a
30-year horizon. All inputs are parameterized via config; the cable population
is fully synthetic.

**Status: the simulation runs end to end in Python.**
The configuration schema and its validators, the purpose-spawned sources of
randomness, the Weibull forms and the synthetic population generator exist,
along with the synthetic failure history and the censored, left-truncated
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
that costs nothing to run. The continuous integration workflows run and the
extension module builds and imports under Python; the Rust kernel that will
replace the reference in the inner loop is not written yet. See [PLAN.md](PLAN.md) for the model, the decisions behind it, and the
phased roadmap.

## Layout

| Path | What it holds |
|---|---|
| `src/` | the Rust crate: the compute kernel, built as a Python extension module |
| `python/cablesim/` | the Python package: configuration, the sources of randomness, the Weibull forms, the population generator, the synthetic failure history and its censored maximum-likelihood fit, the replacement policies, the annual loop that is the correctness reference for the kernel, and the run, metrics and figure layers above it |
| `scripts/` | driver scripts that build configuration overrides and call the package in a loop; they hold no modelling logic |
| `configs/base.yaml` | the documented default run configuration |
| `notebooks/` | marimo notebooks that walk the package interface layer by layer |
| `tests/` | the test suite |
| [PLAN.md](PLAN.md) | the model, the configuration schema, the kernel contract, and the roadmap |

The Python reference and the Rust kernel implement the same model twice. That is
deliberate: the parity tests between them are what validate the kernel, so
neither side is redundant.

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
is missing; leave `--extra plots` off and its tests are the only ones that
cannot run.

`--release` is not optional for anything timed: the parity tests run many
replications, and a debug build is slow enough to dominate the run. Rebuild
after any change under `src/`, or the suite reports on the extension module
that is currently installed rather than the one you just edited.

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
uv sync --group notebooks && uv run pytest -m notebooks   # executes every marimo notebook
uv sync --group app && uv run pytest -m app               # drives the Shiny app in a browser
```

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
