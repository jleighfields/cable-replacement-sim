# cable-replacement-sim

Simulation of underground cable failure and replacement policy for an electric
distribution utility, with a Rust compute kernel exposed to Python via
PyO3/maturin.

Censored-MLE Weibull failure model, three-phase segments governed by the
minimum of three conductor lifetimes, and budget-constrained replacement
policies evaluated against customer reliability (SAIFI / SAIDI / CMI) over a
30-year horizon. All inputs are parameterized via config; the cable population
is fully synthetic.

**Status: scaffolded.** The build pipeline works end to end — the Rust crate
compiles, the extension module imports under Python, and the configuration
schema validates the checked-in defaults. The simulation itself is not written
yet. See [PLAN.md](PLAN.md) for the model and the phased roadmap.

## Layout

| Path | What it holds |
|---|---|
| `src/` | the Rust crate: the compute kernel, built as a Python extension module |
| `python/cablesim/` | the Python package: configuration, and the pure-Python reference implementation that mirrors the kernel |
| `configs/base.yaml` | the documented default run configuration |
| `tests/` | the test suite |

The Python reference and the Rust kernel implement the same model twice. That is
deliberate: the parity tests between them are what validate the kernel, so
neither side is redundant.

## Getting started

Requires the Rust toolchain ([rustup](https://rustup.rs)) and
[uv](https://docs.astral.sh/uv/). The Python version is pinned in
`.python-version`.

```bash
uv sync                            # create the environment
uv run maturin develop --release   # build and install the extension module
uv run pytest                      # run the suite
```

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

Neither gates a merge — they run on pushes to `main` and weekly — so a break
in either surfaces on the next push rather than on the pull request that
caused it.

`cargo test` needs `--no-default-features` because the default build enables
PyO3's `extension-module`, and a test binary linked against it cannot resolve
the CPython symbols the interpreter supplies at import time.
