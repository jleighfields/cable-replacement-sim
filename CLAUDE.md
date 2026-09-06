# Project instructions

Read `PLAN.md` before writing code. It carries the domain model, the config
schema, the repo layout, the validation strategy and the phased roadmap; this
file carries the conventions that apply whatever phase you are in.

Where the two disagree, `PLAN.md` wins on *what the project is* and this file
wins on *how work is done here*.

## Orientation

- **What this is:** a simulation of underground cable failure and replacement
  policy for an electric distribution utility, with a Rust compute kernel
  exposed to Python via PyO3/maturin. Two goals — learn PyO3/maturin on a
  workload that justifies Rust, and produce a publishable portfolio artifact.
- **The population is fully synthetic.** No original utility data is used, and
  none should enter this repo. A file that appears to hold observed failure
  records is a problem, not an input.
- **The package implements, the notebooks demonstrate, the reference validates.**
  Simulation logic lives in `python/cablesim/` and the Rust crate in `src/`.
  Notebooks and the Shiny app import from the package and define no modeling
  logic of their own.
- **How to verify a change:** the commands are in `README.md`, under Testing,
  which is also what `.github/workflows/test.yml` runs. Two things about them
  that are easy to skip and expensive to skip: anything touching `src/` needs
  `maturin develop --release` first, or the suite reports on the extension
  module currently installed rather than the one in the diff; and `--release`
  is not optional for anything timed, because a debug build makes every timing
  number meaningless.
- **Where each kind of thing is written down.** `README.md` is how a person
  runs this. `PLAN.md` is what the project is and why — the domain model, the
  config schema, the kernel contract, the validation strategy, the roadmap.
  This file is how work is done here. Rules belong in exactly one of the three,
  and a rule restated in a second one drifts.

## Core principles

- **Simplicity first.** Make every change as simple as possible, and touch
  only what the task requires — no features, abstractions, or error handling
  beyond it.
- **Root causes, not workarounds.** If a change only stops the symptom, ask
  "knowing everything I know now, what removes the cause?" before shipping.
  A fix that leaves the cause in place has to be made again.
- **Ask, don't assume.** When requirements are ambiguous or several
  approaches exist, ask. Don't guess at intent or decide design alone.
  `PLAN.md` §13, Decisions and what is still open, records the questions that
  are settled with their reasoning, and lists the ones that are not; answer
  those with the user rather than picking a default and moving on.
- **Verify by running, not by reading.** Any claim you can run is unverified
  until you have run it — yours and everyone else's. Never mark work complete
  without running the tests, building the extension, executing the command.
  When a comment, docstring, README or gate asserts a behavior, run it before
  believing it.
- **A check you have not watched fail is not known to work.** Break what it
  guards and confirm it reports. Some operations fail by doing nothing — a
  find-and-replace that matched nothing, a scan with an unreadable baseline,
  a loop over an empty list — and each returns success, letting every later
  step pass for the wrong reason.
- **A statistical test is the easiest kind to fool yourself with.** The parity
  suite compares Rust against the Python reference within Monte Carlo error, so a
  real divergence smaller than the tolerance passes. Every claim of agreement
  needs the deterministic parity test — hazard forced to 0 or 1 — behind it,
  because that is the one that pins policy and budget logic exactly.
- **Plan non-trivial work.** Plan mode for anything spanning 3+ steps or an
  architectural decision. If work goes sideways, stop and re-plan.
- **Use subagents to keep the main context clean.** Offload research,
  exploration, and heavy read-only passes.
- **Review in two tiers**, both by the `code-reviewer` agent: `commit` per
  commit over what is being committed; the full pass per branch before its
  pull request opens, adding the suite over the whole change and a failing
  test pinning any defect. The commit tier does not run the suite — you run
  the tests that commit can reach. Resolve or waive every Must Fix and
  Should Fix before opening the pull request.

## The Python/Rust mirror

The Python reference and the Rust kernel implement the same model twice, and
that is the validation strategy rather than an accident. **`PLAN.md` §5.1, What
is implemented where, has the table** — which modules mirror each other, what
lives only in Python, and the rule that decides the split. It is not repeated
here.

- **Never collapse the two sides.** Deleting the reference deletes the only thing
  that validates the kernel. `simplify-audit` is told this explicitly so it
  does not report either side as dead code.
- **A change to one side is incomplete until the other is read.** Report and
  fix divergence; the parity test may not be sensitive enough to catch it.
- **A numeric literal appearing in both is a defect.** The copies drift, and
  drift here is silent — see *Configuration and parameters* below.
- **The FFI boundary is crossed once per policy per chunk of replications.**
  Arrays in, arrays out, `py.allow_threads` around the compute. Never call back
  into Python inside the loop; that erases the speedup this project exists to
  demonstrate. Random draws arrive as an array — the kernel neither seeds nor
  generates.
- **The reference implementation is not the benchmark baseline.** They are
  different programs with different jobs, and comparing the kernel against the
  scalar reference overstates the speedup. `PLAN.md` §6.C, Benchmarks, has the
  four implementations and which one the claim is made against.

## Code style

- Prioritize simplicity and readability.
- **Docstrings**: Google style — summary line, then `Args`, `Returns`,
  `Yields`, `Raises` as appropriate. Every public function and class.
- **Type hints on every signature.** Use `X | None`, not `Optional[X]`.
- Import modules directly for type hints; no quoted forward references.
- **No `from __future__ import annotations` or `TYPE_CHECKING` guards** as
  circular-import workarounds. Fix the cycle structurally — move one side's
  import into the function that uses it.
- **No `_` prefix on function names.** Internal helpers get real names. In
  marimo notebooks use local functions where names would otherwise collide.
- Preserve inline comments when refactoring or extracting helpers.
- Avoid duplication. Two near-identical blocks is a smell; three is a refactor.
  The Python/Rust mirror above is the one exception, and it is deliberate.
- Prefer `if/elif/else` over `continue` — it makes the structure explicit.
- Comments explain *why*, not *what*. Magic numbers need a justification.
- **Never silently skip a missing file or directory.** Raise or warn, so the
  problem surfaces at its source rather than three steps downstream.
- **Prefer polars over pandas** for result tables and analysis. NumPy is what
  crosses the FFI boundary — `PyReadonlyArray1` in, `PyArray` out, zero copy —
  so per-segment arrays stay NumPy on the Python side of that call.

### Rust

- **`///` doc comments on every public item**, saying what it computes and
  what its arguments mean. A `#[pyfunction]` is a contract two languages read:
  it gets a doc comment on the Rust side and a docstring on the Python
  wrapper, and they must not say different things.
- **Pin the PyO3 version and read that version's guide.** PyO3 migrated to the
  `Bound<'py, T>` smart-pointer API; older tutorials and answers found online
  will not compile against a current pin.
- **Seed per replication** (`seed + rep_index`) via `StdRng::seed_from_u64`, so
  results are reproducible and order-independent under rayon. Never share one
  RNG across threads.
- **`unsafe` belongs in PyO3 bindings and nowhere else.** One outside a
  binding needs a comment naming the invariant it upholds.
- `cargo clippy` is the mechanical pass for Rust — ruff does not read `.rs`,
  so without it half this repo gets no linting at all.

## Configuration and parameters

- **One config model is the single source of truth for run knobs.** Every
  tunable lives on the pydantic model in `cablesim/config.py`, loaded from
  `configs/base.yaml`. Construct it with overrides at the entry point and pass
  the instance down — never read module-level globals for values that belong
  on it.
- **Sweeps override config values from a driver script or notebook**, never by
  editing `configs/base.yaml`. The base file is the documented default.
- **The same validated object feeds the reference, the kernel, and the app.** One
  code path, three front ends. A knob the Shiny UI sets and the notebooks
  cannot is a knob that has escaped the model.
- **Fixed constants go in the package's constants module.** The test: a value
  belongs there if a caller overriding it would be a *bug*, and on the config
  model if a caller legitimately overrides it for one run.
- **Never write a value in two places.** A constant that also appears as a
  literal elsewhere is a defect; the copies will drift. Across the FFI
  boundary this is worse than usual — a default written in Python and again in
  Rust diverges silently, and only the parity test would notice.
- **Record the config with the run.** Anything producing artifacts writes the
  config alongside them — reconstructing settings from memory is guesswork
  once the defaults have moved.

## Logging, paths, and secrets

- **Module-scoped logger, never `basicConfig`.** `log =
  logging.getLogger(__name__)` at module scope; a library calling
  `basicConfig()` silently reconfigures its host application. Configure
  handlers only in an entry point — a notebook, the app, or a driver script.
- **Anchor paths to `__file__`, never the working directory.** Derive from a
  project-root constant, so a notebook run from `notebooks/` and the app run
  from the repo root resolve the same file.
- **Nothing here authenticates to anything, and no `.env` is expected.** The
  population is generated in-process from a seed. `.gitignore` covers the
  pattern defensively, so a tracked `.env` appearing is a real finding rather
  than the usual false positive.
- **RNG seeds are not secrets.** `simulation.seed` and everything derived from
  it exist so a run reproduces; they are meant to be committed and read.

## Tests

- **Run unit tests whenever code changes**, and add tests when behavior
  changes. Rebuild with `maturin develop --release` first when `src/` changed.
- **`tests/` mirrors the package.** Every test directory needs an
  `__init__.py`, subdirectories included, and helpers import as
  `from tests.helpers import ...` — under pytest's `prepend` import mode the
  two styles are mutually exclusive, and duplicate basenames across
  subdirectories collide outright.
- **Pytest machinery in `conftest.py`, plain functions in `helpers.py`** — a
  helper should be readable without knowing pytest.
- **Generate fixtures in code.** The whole population is synthetic and
  reproducible from a seed, so a committed data blob has no reason to exist
  here. The suite reaches no network.
- **The three validation layers, in order of authority** (`PLAN.md` §6,
  Validation strategy, which has the assertions, tolerances and sample sizes):
  analytical checks against the closed form, parity between implementations,
  then benchmarks. An analytical check is worth more than a parity check,
  because it can be wrong in only one way.
  - **MLE recovery is what validates the fitting code.** Simulate lifetimes from
    known `(k, lambda)`, censor, refit, confirm the truth falls inside the
    fitted CI. It is built as a ladder (`PLAN.md` §6, Validation strategy) so
    each rung adds exactly one thing that can be wrong — censoring, then left
    truncation, then length, then technology indicators, then per-technology
    shape. A single test of the whole model says something is broken; the
    ladder says what.
  - **Know which comparisons are exact and which are statistical** (`PLAN.md`
    §6, Validation strategy). Draws are addressable, so the uniform generator
    is a pure function of four integers and is compared bit-exactly across
    languages. The three Python implementations are compared exactly too, since
    they share the draws and the arithmetic. Only Python against Rust is
    statistical, because a last-place floating-point difference in a score
    flips a sort and changes which candidate is funded last — use a tolerance
    derived from replication standard error, never a fixed epsilon.
- **Two opt-in marker groups, both excluded from a default run** because each
  costs minutes: `notebooks` (`uv run pytest -m notebooks`) executes every
  marimo notebook headless so they cannot silently rot, and `app`
  (`uv run pytest -m app`) drives the Shiny app through Playwright. Neither
  is required — nor is `test` yet, see *Git and tooling* below — so a break
  in either surfaces on the next push to `main`.
- **A notebook may carry its own assertions.** Where a notebook has already
  computed something whose value is known, `assert` on it in the cell that
  computed it rather than rebuilding the setup under `tests/`. The boundary:
  an assertion about the *package* belongs in `tests/` where it runs in
  seconds; one about the *notebook* belongs in the notebook.
- **The app suite tests wiring, not numbers.** Its one assertion worth the
  browser drives the UI and then calls the package directly with the same
  overrides and seed, requiring the two to agree — that is what catches a
  control bound to the wrong config field, which every UI-free test passes.
  Run it at interactive-scale defaults with a pinned seed.
- **Never sleep in a browser test.** The app runs its simulation through
  `ExtendedTask`, so results arrive asynchronously; use Playwright's
  auto-waiting assertions with a generous timeout. Select on stable element
  `id`s, never on rendered text or DOM position.

## Notebooks and the app

- **marimo, not Jupyter.** Notebooks are plain `.py` files, and both run modes
  — headless and interactive — must keep working. Both need the notebooks
  dependency group installed first; a plain `uv sync` leaves marimo out, and
  the headless run then fails on `import marimo`.
- **Notebooks import from `cablesim` and define no modeling logic.** If a
  notebook needs a function, it belongs in the package. The same rule governs
  `app/`.
- **Number sparsely** so a step can be inserted without renumbering
  everything downstream.
- Notebooks and the app are entry points, so they may configure logging and
  construct the config. Library modules may not.
- **The app does not run on every input change.** A 30-year Monte Carlo behind
  reactive inputs is unusable — an explicit Run button, `ExtendedTask` so the
  run is async, and a progress indicator.
- **Interactive defaults are smaller than batch defaults**, and the UI says
  so. Full-size runs belong in batch scripts and notebooks.

## Git and tooling

- **Use `uv`, not `pip`.** `uv add <pkg>` for new deps, never `uv pip install`.
  Rust dependencies go through `cargo add`.
- **Never reference Claude, Anthropic, or AI in commit messages.** No
  `Co-Authored-By`, no "generated by", no attribution of any kind.
- **Feature branches** for anything non-trivial, cut from latest `main`:
  ```
  git checkout main && git pull origin main
  git checkout -b feature/<short-kebab-name>
  ```
  Features **squash-merge** into `main` with a message summarizing the whole
  feature, not just the last commit on the branch.
- **`main` is protected and takes no direct pushes.** Every change arrives
  through a squash-merged pull request. The ruleset is checked in at
  `.github/rulesets/protect-main.json`, and `PLAN.md` §10.4, Branch protection
  on `main`, has the setup and the ways branch protection goes wrong.
- **The `test` check is not yet *required*, and that is temporary.** The
  applied ruleset is `.github/rulesets/protect-main.json`, which carries
  everything but the required-status-check rule — a check that has never
  reported green blocks every merge including the one that would fix it. The
  rule waits in `.github/rulesets/protect-main-required-check.json` and is
  applied at the end of Phase 0. Until then a red check does not physically
  stop a merge — treat it as though it did.
- **A red check is a finding, not a flake.** Read the failure before re-running
  it. `uv sync --locked` failing means the lockfile does not match
  `pyproject.toml`, and no number of re-runs fixes that.
- **Use the GitHub CLI (`gh`)** for pull requests and repository settings. An
  agent may cut a branch, push, open the PR with `gh pr create`, and confirm
  the `test` check is green — then it stops. Never run `gh pr merge`; merging
  is a human action.
- Review all commits for quality and style before pushing.
- **Check for README updates** after changes affecting usage or the public API.
- **Don't assume.** Verify against the codebase and ask rather than guessing.
- **Each phase in `PLAN.md` §11, Phased roadmap, ends in a working,
  committed state.** A phase that does not build and does not pass its own
  tests is not finished.
- **Versioning and publishing arrive at Phase 7**, not before. Until then
  there is no wheel on an index, so nothing pins this project and a version
  number is free to move. When the first wheel publishes, a published version
  becomes immutable — replacing a file on the index leaves every existing
  lock unresolvable — and the release rules get written then.

Tooling configuration lives in the file that configures it, not here —
restating it in prose creates drift: **`pyproject.toml`** (maturin build
backend, ruff rules and per-file ignores, pytest settings, dependency groups),
**`Cargo.toml`** (crate dependencies and the release profile), and
**`.gitignore`**.

## Security checks

- **Never baseline a real secret.** Remove it and **rotate** it — working-tree
  removal is not enough if it was ever committed.
- **Prefer removing the secret over baselining it.** Drop the secret-shaped
  value, or mark that line `# pragma: allowlist secret` with the reason
  beside it — the same rule as `# noqa: S106` for ruff's equivalent.
- **`.claude/skills/security-scan/SKILL.md` owns the rest**: which tools run
  and with what, why the `-hook` entry point is required, the required
  baseline format, and this repo's caveats — including the fact that seeds and
  simulation parameters are meant to be committed.

## Plans and lessons

- **Write a plan first** for non-trivial work, to
  `tasks/<description>-<YYYY>-<MM>-<DD>.md` — short, kebab-case, meaningful
  (not "todo"). Check it in with the user before implementing, mark items
  complete as you go, and add a review section when finished. `PLAN.md` is the
  standing project plan and is not one of these; it is updated, not replaced.
- **Move finished plans to `tasks/completed/`.** Several can coexist.
- **After any correction from the user, add the pattern to
  `docs/lessons.md`** — the rule that prevents the same mistake next time.
  Review it when starting related work.

## Minimalism (write less)

Run this *before* writing code; stop at the first step that solves the problem.

1. **Does this need to exist?** Prefer deletion, then refactoring, then
   addition.
2. **Use the standard library** before hand-rolling.
3. **Use a library already imported here** — reach for its built-in first.
4. **Use an installed dependency** before adding a new one.
5. **Prefer the smallest correct form.** No helper, class, config knob, or
   generalization until there are 2–3 real call sites.
6. **Only then** write minimal custom code.

`/simplify-audit` finds existing bloat (report-only delete-list).

## Prose is professional and factual

**Everything written here — comments, docstrings, READMEs, commit messages,
pull-request bodies, skills, agents and plans — states what is true and how the
reader can check it.** A sentence that rates something without evidence
describes the author's opinion, not the code's behavior. When the code
changes, unsupported ratings do not update with it.

Common categories to avoid: unmeasured rankings, personified programs where
the verb stands in for a mechanism, unmeasured cost or effort claims,
aesthetic verdicts like "elegant" or "hacky", aphorisms, and filler run-ups.
See `comment-docstring` for rewrites, greps, and the categories that need
manual review.

**A speedup claim is a measurement or it is nothing.** State the baseline it
was measured against, the build profile, and the thread count. "Rust is
faster" is the exact sentence this section exists to prevent.

**Argument is not editorializing.** State each claim with its reason, in the
same sentence or the next one — for example, "two copies of the same value
drift apart over time." Give the reader something to check; keep the
reasoning and drop unsupported ratings.

Judge sentences in context — some individual words that look like offenders
are fine. See `comment-docstring` for details.

## Comments & docstrings are self-contained

**Every comment, docstring, marimo cell, and doc must stand on its own for
a reader who has the repo and nothing else**, and must describe the code as
it is now. References that only make sense outside the repo, or only to
people involved in the original conversation, break for future readers.

Common categories to avoid: references to plan files under `tasks/`, commits,
tickets, "as discussed", earlier versions of the code, shortened domain terms
that collapse to common English words, and bare dates. See `comment-docstring`
for examples and greps.

Describe the thing directly — what it does, what the constraint is, why
this way rather than the obvious alternative. Test: **delete every ticket and
commit message; would this sentence still teach a new reader anything?**

Point to durable references freely: a README section, another module, an
external spec, and `PLAN.md` by section — it is checked in and is where the
modeling decisions are argued. **Cite the number and the title**, as in
"`PLAN.md` §2.3, Effective scale": inserting a section renumbers every
one after it, and a bare number then points confidently at the wrong place,
while a number plus a title shows the mismatch on sight.

- **Pull-request and issue bodies, at a stricter bar.** Their reader has
  the diff and little else, so even a pointer into this repo fails when
  the diff omits the file it points at. Name the thing, not its number —
  "the greedy budget allocation", not "`policy.rs` step 4".
  `.github/PULL_REQUEST_TEMPLATE.md` carries this reminder at the point of
  writing, along with the checklist for a change touching one side of the
  Python/Rust mirror.
- **Directory READMEs point, never restate.** Each says what belongs in its
  directory and links to whatever owns the detail. Duplicated descriptions
  go stale when the code moves.

## Skills and agents

- **Where they live.** Slash commands in `.claude/skills/`, subagents in
  `.claude/agents/`. All are committed, and their names and descriptions are
  injected at session start — so none is listed here, and a new agent needs a
  session restart to be seen.
- **Agents read their skill's `SKILL.md` at runtime** rather than copying the
  checklist, so it stays in step as skills evolve. Given no path,
  `code-reviewer commit` defaults to the changed files (`git diff --name-only
  HEAD` plus untracked) and its full pass to the branch diff against
  `origin/main`; `simplify-auditor` defaults to the whole repo.
- **A rule belongs to whichever file owns the activity.** What to look for,
  what counts as a finding, and what else to update when a given file changes
  belong to the skill running that phase, where they also apply when someone
  invokes the skill directly. How a pass is ordered and what it escalates
  belongs to the agent. Only what everyone needs before starting work belongs
  here — this file loads on every session.
