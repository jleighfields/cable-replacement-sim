# Project instructions

Read `PLAN.md` before writing code. It carries the domain model, the config
schema, the repo layout, the validation strategy and the phased roadmap; this
file carries the conventions that apply whatever phase you are in.

Where the two disagree, `PLAN.md` wins on *what the project is* and this file
wins on *how work is done here*.

## Orientation

- **What this is:** a simulation of underground cable failure and replacement
  policy for an electric distribution utility, with a Rust compute kernel
  exposed to Python via PyO3/maturin. `README.md` summarizes it, and `PLAN.md`
  opens with the goals.
- **The population is fully synthetic.** No original utility data is used, and
  none should enter this repo. A file that appears to hold observed failure
  records is a problem, not an input.
- **The package implements, the notebooks demonstrate, the reference
  validates.** Simulation logic lives in `python/cablesim/` and the Rust crate
  in `src/`. Notebooks and the Shiny app import from the package and define no
  modeling logic of their own.
- **Two things about the verification commands**, which are in `README.md`
  under Getting started and Testing, that are easy to skip and expensive to
  skip: anything touching `src/` needs `uv run maturin develop --release` first, or
  the suite reports on the extension module already installed rather than the
  one in the diff; and `--release` is not optional for anything timed.
- **Where each kind of rule is written down.** `README.md` is how a person runs
  this. `PLAN.md` is what the project is and why. This file is how work is done
  here. A rule belongs in exactly one of the three; restated in a second, the
  copies drift.

## Core principles

- **Simplicity first.** Touch only what the task requires — no features,
  abstractions, or error handling beyond it.
- **Root causes, not workarounds.** A fix that leaves the cause in place has to
  be made again.
- **Ask, don't assume.** When requirements are ambiguous or several approaches
  exist, ask. `PLAN.md` ends with a list of decisions and what is still open;
  answer the open ones with the user rather than picking a default.
- **Verify by running, not by reading.** Any claim you can run is unverified
  until you have run it — yours and everyone else's. When a comment, docstring,
  README or gate asserts a behavior, run it before believing it.
- **A check you have not watched fail is not known to work.** Break what it
  guards and confirm it reports. Some operations fail by doing nothing — a
  find-and-replace that matched nothing, a scan with an unreadable baseline, a
  loop over an empty list — and each returns success, letting every later step
  pass for the wrong reason.
- **A statistical test is the easiest kind to fool yourself with.** A real
  divergence smaller than the tolerance passes. Every claim that two
  implementations agree needs the deterministic test behind it. `PLAN.md`, on
  implementation parity, has which comparisons are exact, which are
  statistical, and why.
- **Plan non-trivial work.** Plan mode for anything spanning 3+ steps or an
  architectural decision. If work goes sideways, stop and re-plan.
- **Use subagents to keep the main context clean.** Offload research,
  exploration, and heavy read-only passes.
- **Review in two tiers**, both by the `code-reviewer` agent: `commit` per
  commit over what is being committed, and the full pass per branch before its
  pull request opens. The commit tier does not run the suite — you run the
  tests that commit can reach. Resolve or waive every Must Fix and Should Fix
  before opening the pull request.

## The Python/Rust mirror

The Python reference and the Rust kernel implement the same model twice, and
that is the validation strategy rather than an accident. **`PLAN.md` has the
table of what is implemented where** — which modules mirror each other, what
lives only in Python, and the rule that decides the split. It is not repeated
here. A mirrored pair carries the same module name on both sides:
`weibull.py`/`weibull.rs`, `policies.py`/`policies.rs`,
`simulate.py`/`simulate.rs`.

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
  scalar reference overstates the speedup. The baseline is the batched NumPy
  implementation in `batched.py`; `PLAN.md`, on benchmarks, has all four
  implementations and how each is reported.

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
- **The kernel neither seeds nor generates.** Every uniform arrives as an array
  from NumPy, and every policy reads the identical draws so that a comparison
  between two of them is a paired difference. So no random-number crate belongs
  in `Cargo.toml`, and results are order-independent under rayon because each
  replication reads its own slice rather than drawing from a shared stream.
- **`f64` has no `Ord`.** The candidate sort needs a hand-written total
  comparator with a stated position for NaN, not `partial_cmp` and a hope:
  sort NaN last **and assert none is produced**. Every input to the score is
  finite by construction, so a NaN is a defect upstream, and unasserted it
  surfaces as a segment that quietly never gets funded.
- **`unsafe` belongs in PyO3 bindings and nowhere else.** One outside a
  binding needs a comment naming the invariant it upholds.
- `cargo clippy` is the mechanical pass for Rust — ruff does not read `.rs`,
  so without it half this repo gets no linting at all.

## Configuration and parameters

- **One config model is the single source of truth for run knobs.** Every
  tunable lives on the pydantic model in `cablesim/config.py`, loaded from
  `configs/base.yaml`. Construct it with overrides at the entry point and pass
  the instance down — never read module-level globals for values that belong on
  it. The same validated object feeds the reference, the kernel and the app;
  a knob the Shiny UI sets and the notebooks cannot is a knob that has escaped
  the model.
- **Sweeps override config values from a driver script or notebook**, never by
  editing `configs/base.yaml`. The base file is the documented default.
- **Fixed constants go in the package's constants module.** The test: a value
  belongs there if a caller overriding it would be a *bug*, and on the config
  model if a caller legitimately overrides it for one run.
- **Never write a value in two places.** A constant that also appears as a
  literal elsewhere is a defect; the copies will drift. Across the FFI boundary
  this is worse than usual — a default written in Python and again in Rust
  diverges silently, and only the parity test would notice.
- **Record the config with the run.** Anything producing artifacts writes the
  effective config alongside them, dumped from the validated model rather than
  copied from the input file, since a driver script's overrides never reach
  that file.

## Logging, paths, and secrets

- **Module-scoped logger, never `basicConfig`.** `log =
  logging.getLogger(__name__)` at module scope; a library calling
  `basicConfig()` silently reconfigures its host application. Configure
  handlers only in an entry point — a notebook, the app, or a driver script.
- **Anchor paths to `__file__`, never the working directory.** Derive from a
  project-root constant, so a notebook run from `notebooks/` and the app run
  from the repo root resolve the same file.
- **Nothing here authenticates to anything, and RNG seeds are not secrets.**
  The population is generated in-process from a seed, and seeds exist so a run
  reproduces — they are meant to be committed and read. `.gitignore` covers
  `.env` defensively, so a tracked one appearing is a real finding rather than
  the usual false positive.

## Tests

- **Run unit tests whenever code changes**, and add tests when behavior
  changes. Rebuild with `uv run maturin develop --release` first when `src/` changed.
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
- **One fixture builds the inputs every parity test shares**, and no test
  constructs a generator of its own — two fixtures that drift apart make a
  parity failure a question about the fixtures first.
- **An analytical check is worth more than a parity check**, because it can be
  wrong in only one way. `PLAN.md`, on the validation strategy, has the three
  layers in order of authority, with the assertions, tolerances and sample
  sizes; do not restate them here.
- **A notebook may carry its own assertions.** Where a notebook has already
  computed something whose value is known, `assert` on it in the cell that
  computed it rather than rebuilding the setup under `tests/`. The boundary:
  an assertion about the *package* belongs in `tests/` where it runs in
  seconds; one about the *notebook* belongs in the notebook.
- **The `notebooks` and `app` marker groups are excluded from a default run**
  because each costs minutes and neither gates a merge. The app suite runs
  weekly and on pushes to `main`; **the notebook suite runs nowhere
  automatically**, so run `pytest -m notebooks` yourself after changing any
  package API a notebook imports, and before opening a pull request that
  touches one. `README.md` has the commands, and `PLAN.md`, on the Shiny app
  integration tests, has what that suite is for.

## Notebooks and the app

- **marimo, not Jupyter.** Notebooks are plain `.py` files, and both run modes
  — headless and interactive — must keep working. Both need the notebooks
  dependency group installed first; a plain `uv sync` leaves marimo out.
- **Notebooks and `app/` import from `cablesim` and define no modeling logic.**
  If a notebook needs a function, it belongs in the package. They walk the API
  layer by layer rather than making one top-level call: a notebook that cannot
  show a step without reaching inside the package is a package that has no seam
  there, so this is a constraint on the package rather than a notebook style.
- **Number sparsely** so a step can be inserted without renumbering everything
  downstream.
- Notebooks and the app are entry points, so they may configure logging and
  construct the config. Library modules may not.
- **The app's design is in `PLAN.md`, under the Shiny application** — the Run
  button and `ExtendedTask`, the smaller interactive defaults, and the
  denominator that must scale with them or every reliability index is wrong by
  the ratio.

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
  through a squash-merged pull request. `PLAN.md`, on branch protection, has
  the rulesets, the setup, and the ways branch protection goes wrong —
  including why the `test` check is not required yet and why a red check should
  be treated as blocking anyway.
- **A red check is a finding, not a flake.** Read the failure before re-running
  it. `uv sync --locked` failing means the lockfile does not match
  `pyproject.toml`, and no number of re-runs fixes that.
- **Use the GitHub CLI (`gh`)** for pull requests and repository settings. An
  agent may cut a branch, push, open the PR with `gh pr create`, and confirm
  the `test` check is green — then it stops. Never run `gh pr merge`; merging
  is a human action.
- Review all commits for quality and style before pushing.
- **Check for README updates** after changes affecting usage or the public API.
- **A phase ends in a working, committed state** — the phases are the roadmap
  in `PLAN.md`. A phase that does not build and does not pass its own tests is
  not finished.

Tooling configuration lives in the file that configures it, not here —
restating it in prose creates drift: **`pyproject.toml`** (maturin build
backend, ruff rules and per-file ignores, pytest settings, dependency groups),
**`Cargo.toml`** (crate dependencies and the release profile), and
**`.gitignore`**.

## Security checks

- **Never baseline a real secret.** Remove it and **rotate** it — working-tree
  removal is not enough if it was ever committed. Prefer removing the value
  over baselining it; failing that, mark the line `# pragma: allowlist secret`
  with the reason beside it, the same rule as `# noqa: S106` for ruff.
- **`.claude/skills/security-scan/SKILL.md` owns the rest**: which tools run
  and with what, why the `-hook` entry point is required, the baseline format,
  and this repo's caveats.

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
describes the author's opinion, not the code's behavior, and does not update
when the code changes.

Categories to avoid: unmeasured rankings, personified programs where the verb
stands in for a mechanism, unmeasured cost or effort claims, aesthetic verdicts
like "elegant" or "hacky", aphorisms, and filler run-ups. **Argument is not
editorializing** — state each claim with its reason, in the same sentence or
the next one, and judge sentences in context. `comment-docstring` owns the
rewrites, the greps, and the cases that need manual review.

**A speedup claim is a measurement or it is nothing.** State the baseline it
was measured against, the build profile, and the thread count. "Rust is faster"
is the exact sentence this section exists to prevent.

## Comments & docstrings are self-contained

**Every comment, docstring, marimo cell, and doc must stand on its own for a
reader who has the repo and nothing else**, and must describe the code as it is
now. Avoid references to plan files under `tasks/`, commits, tickets, "as
discussed", earlier versions of the code, shortened domain terms that collapse
to common English words, and bare dates. Test: **delete every ticket and commit
message; would this sentence still teach a new reader anything?**
`comment-docstring` owns the examples and greps.

Point to durable references freely — a README section, another module, an
external spec. Where one has numbered parts, **cite the number and the title**
together: inserting a part renumbers every one after it, and a bare number then
points confidently at the wrong place, while a number plus a title shows the
mismatch on sight.

**`PLAN.md` is not a durable reference, and nothing outside it cites a section
of it.** It is a working document: sections are inserted, renumbered and
rewritten as the project moves, and a citation into it rots without anything
reporting that it has. Its own internal cross-references are a different case
and stay — a test checks that every one of them resolves, which is exactly the
check an outward citation cannot have. Where a comment, docstring or rule needs a fact the plan argues, **state
the fact where it is needed** — that is what a reader with this file and
nothing else can act on. Where the plan is genuinely the place to go and
restating it would take a paragraph, name the topic rather than the number, so
a stale pointer degrades into a search rather than into a confident pointer at
the wrong section.

- **Pull-request and issue bodies, at a stricter bar.** Their reader has the
  diff and little else, so even a pointer into this repo fails when the diff
  omits the file it points at. Name the thing, not its number — "the greedy
  budget allocation", not "`policy.rs` step 4".
  `.github/PULL_REQUEST_TEMPLATE.md` carries this at the point of writing.
- **Directory READMEs point, never restate.** Each says what belongs in its
  directory and links to whatever owns the detail.

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
