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
| `test_plan_document.py` | `PLAN.md` structure: citations resolve, and it quotes the config verbatim |
| `test_notebooks.py` | every notebook runs headless (marker: `notebooks`) |

## Two things that are easy to get wrong here

**A test that cannot fail is worse than a missing one**, because it reports
coverage it does not have. Two have already shipped in this repository: a
validator test that passed on an unrelated validation error from a stale
schema, and a derived-column test asserting values were non-negative, which no
valid configuration can violate. When adding a test, break the thing it names
and watch it go red before trusting it.

**Marker groups are excluded from a default run.** `notebooks` and `app` each
cost minutes, so `addopts` deselects them and they run on pushes to the default
branch and weekly. A break in either surfaces after a merge rather than before
it, which is the price of keeping the merge gate fast.
