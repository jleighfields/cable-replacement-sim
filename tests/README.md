# Test suite

Run it with the commands in the root [README](../README.md#testing). This file
covers how tests are laid out here and why.

## Layout

- **`tests/` mirrors the package.** A test module sits at the path its subject
  sits at under `python/cablesim/`.
- **Every test directory needs an `__init__.py`**, subdirectories included.
  Under pytest's `prepend` import mode this is what makes
  `from tests.helpers import ...` resolve, and it is mutually exclusive with
  the rootdir-relative style — pick one, and this suite has picked packages.
  Without the `__init__.py` files, two modules sharing a basename in different
  subdirectories fail collection outright with "use a unique basename".
- **Pytest machinery in `conftest.py`, plain functions in `helpers.py`.** A
  helper should be readable without knowing pytest.

## Fixtures

**Generate them in code.** The whole population is synthetic and reproducible
from a seed, so a committed data blob has nothing to offer that a builder
function does not. The suite reaches no network, and no original utility data
belongs in this repo at all.

## The three validation layers

In order of authority — `PLAN.md` §6, Validation strategy, argues each one:

1. **Analytical checks** against the closed form. Worth more than a parity
   check, because there is only one way for them to be wrong.
2. **Oracle parity** between the Python reference implementation and the Rust
   kernel.
3. **Benchmarks.**

Two consequences worth stating where the tests are written:

- **Compare the two implementations statistically**, with a tolerance derived
  from replication standard error. Bit-exact RNG parity across the two
  languages proves little and costs a great deal, so no test should chase it —
  but a statistical comparison passes on a real divergence smaller than its
  tolerance, so pin policy and budget logic with a deterministic case (hazard
  forced to 0 or 1) as well.
- **Rebuild before running** whenever `src/` changed:
  `uv run maturin develop --release`. pytest imports whichever extension
  module is installed, so a stale one reports on code that is not in the diff.

## The two opt-in marker groups

`addopts` excludes both from a default run because each costs minutes; the
root README has the commands. `notebooks` executes every marimo notebook
headless so they cannot silently rot, and `app` drives the Shiny app through
Playwright.

- **The app suite tests wiring, not numbers.** Its one assertion worth a
  browser drives the UI, then calls the package directly with the same
  overrides and seed and requires the two to agree. That is what catches a
  control bound to the wrong config field — something every UI-free test
  passes. Run it at interactive-scale defaults with a pinned seed.
- **Never sleep in a browser test.** The app runs its simulation through
  `ExtendedTask`, so results arrive asynchronously. Use Playwright's
  auto-waiting assertions with a generous timeout, and select on stable
  element `id`s rather than on rendered text or DOM position.
- **A notebook may carry its own assertions.** Where a notebook has already
  computed something whose value is known, `assert` on it in the cell that
  computed it rather than rebuilding the setup here. The boundary: an
  assertion about the *package* belongs in `tests/`, where it runs in seconds;
  one about the *notebook* belongs in the notebook.
