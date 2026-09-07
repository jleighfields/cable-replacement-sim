# Test suite

Run it with the commands in the root [README](../README.md#testing). This file
covers how tests are laid out here and why.

## Layout

- **`tests/` mirrors the package.** A test module sits at the path its subject
  sits at under `python/cablesim/`.
- **Every test directory needs an `__init__.py`**, subdirectories included.
  Under pytest's `prepend` import mode this is what makes
  `from tests.helpers import ...` resolve, and it is mutually exclusive with
  the rootdir-relative style. Without the `__init__.py` files, two modules
  sharing a basename in different subdirectories fail collection outright.
- **Pytest machinery in `conftest.py`, plain functions in `helpers.py`.** A
  helper should be readable without knowing pytest.
- **Fixtures are generated in code.** The population is synthetic and
  reproducible from a seed, so a committed data blob has no reason to exist.
  The suite reaches no network.

| Module | Subject |
|---|---|
| `test_smoke.py` | the extension module imports and the checked-in config loads |
| `test_config.py` | each validator rejects what its message claims |
| `test_random_draws.py` | the four sources are independent, and the stream is pinned |
| `test_weibull.py` | the closed forms, against the analytical results |
| `test_population.py` | the generated table, and the columns derived from it |
| `test_records.py` | the synthetic failure history, and the technology coding |
| `test_mle_recovery.py` | the recovery ladder, and the fit's reported likelihood, interval widths and starting values |
| `test_policies.py` | eligibility, the rank key per policy, the tie-break, and the greedy fill |
| `test_simulate.py` | the annual loop, against cases whose answers are known by hand |
| `test_parity.py` | the Rust kernel against the Python reference, exactly where the lifetimes are forced and paired where they are drawn |
| `test_results.py` | the run directory format, the saved schema, and the sweep reader |
| `test_run.py` | the chunk and policy loops, and that chunk size changes no number |
| `test_metrics.py` | the reliability indices, discounting, and the baseline comparison |
| `test_plots.py` | the figures, asserted on their data rather than their pixels |
| `test_plan_document.py` | `PLAN.md` structure: citations resolve, and it quotes the config verbatim |
| `test_workflows.py` | the workflow files name no Rust toolchain of their own |
| `test_notebooks.py` | every notebook runs headless (marker: `notebooks`) |

## Two things that are easy to get wrong here

**A test that cannot fail is worse than a missing one**, because it reports
coverage it does not have. Two have already shipped in this repository: a
validator test that passed on an unrelated validation error from a stale
schema, and a derived-column test asserting values were non-negative, which no
valid configuration can violate. When adding a test, break the thing it names
and watch it go red before trusting it.

**Marker groups are excluded from a default run.** `notebooks` and `app` each
cost minutes, so `addopts` deselects them. The `app` group runs on pushes to
the default branch and weekly, so a break in it surfaces after a merge rather
than before it. **Nothing runs the `notebooks` group at all** — it executes
only when someone runs `uv run pytest -m notebooks`, so run it after changing
any package interface a notebook imports. Both need `uv sync --extra plots`
alongside their dependency group, because notebook 04 imports
`cablesim.plots`.
