---
name: simplify-audit
description: Repo-wide bloat audit. Finds code that should not exist or is not in its minimal form — dead code, unused deps, single-use abstractions, premature generalization — and reports a delete-list. Report-only; makes no edits.
disable-model-invocation: false
allowed-tools: Read, Glob, Grep, Bash
argument-hint: "[file-or-directory]"
---
# Simplify Audit

Find excess and report a **delete-list** — code that should not exist, or
that exists but is not in its minimal form. Report issues but do NOT make
edits. Removal is a separate, user-driven step.

This skill checks **minimalism**, while `code-quality-review` checks
readability and documentation:

| | `code-quality-review` | `simplify-audit` (this skill) |
|---|---|---|
| Asks | "Is this clear and documented?" | "Should this exist, and is it minimal?" |
| Default scope | changed files (diff) | whole repo |
| Output | readability findings | a delete-list (LOC removable) |

For **duplication** specifically, defer to `code-quality-review`'s
**Code Duplication & Helper Functions** section — do not restate that
checklist here. Point the user at it when you spot repeated patterns.
Likewise, repeated *parameter values* belong to that skill's **Single
Source of Truth for Parameter Values** section.

## Arguments

- **file-or-directory** (optional): Path to audit. If omitted, audit the
  whole repo.

## What counts as in scope

- **In scope:** every tracked Python file and every tracked Rust file in
  `src/`, except `tests/`, which is in scope only for unused test-helper
  bloat. `notebooks/` and `app/` are in scope for one thing specifically —
  modeling logic defined there that belongs in `cablesim` instead.
- **Out of scope:** any gitignored or untracked path. Run `git ls-files`
  if you are unsure whether a path is tracked — do not infer scope from
  directory names.
- **Runtime dependencies deserve special attention, on both sides.** An
  unused entry in `[project] dependencies` is installed by everything that
  runs this project, including the deployed Shiny app; an unused crate in
  `[dependencies]` is compiled into the extension module. Check both against
  what is actually imported and `use`d.
- **The Python reference and the Rust kernel duplicate each other on purpose,
  and neither side is a finding.** `python/cablesim/simulate.py` reimplements
  what `src/simulate.rs` computes, and `python/cablesim/policies.py` mirrors the
  scoring in `src/policies.rs`. The parity tests are the only thing that
  validates the kernel, so deleting either side deletes the validation.
  Report a *divergence* between the two — the parity test is what catches it
  — but never propose collapsing them.
- **`benches/` holds three implementations of one calculation, also on
  purpose.** A naive Python loop, a vectorized NumPy reference, and the Rust
  kernel exist so the benchmark reports an honest baseline rather than a
  speedup measured against bad Python. Do not report the slow ones as dead.
- **An unreferenced `cablesim` helper is still a finding here.** This repo is
  an application and a portfolio artifact, not a library other repos install,
  so nothing outside this tree calls in. Grep `notebooks/`, `app/`, `benches/`
  and `tests/` before reporting one — those are the call sites that are easy
  to miss.

## Audit checklist (minimalism)

- **Existence / YAGNI** — functions, classes, branches, or config fields
  never referenced in live code. **Grep-confirm zero references outside
  the definition** before reporting (see Steps). Prefer deletion, then
  refactoring, then addition.
- **Reinvention** — hand-rolled logic that duplicates a built-in from the
  stdlib or from a library this project already imports. Read the
  project's actual imports rather than assuming a fixed library set, and
  cite the specific built-in that replaces the hand-rolled code.
- **Dependency justification** — cross-check each runtime dependency in
  `pyproject.toml` against live `import` usage. Flag any dep with zero or
  near-zero live imports as a removal candidate.
- **Single-use abstraction** — wrapper functions, one-method classes, or
  indirection layers used exactly once. Recommend inlining at the single
  call site.
- **Premature generalization** — parameters, branches, config knobs, or
  "flexibility" that handle cases which never occur in practice. Flag the
  unused case and the code that exists only to serve it.
- **Repo-wide dead exports** — public symbols (functions, classes,
  constants) with no references anywhere in live code. This is broader
  than `code-quality-review`, which only sees the diff.
- **Size signals** — files > ~800 lines or functions > ~80 lines,
  reported as bloat candidates. Cite them; do NOT prescribe the split here
  (defer the "how" to `code-quality-review`'s function-length guidance).
- **Dead scaffolding** — commented-out code blocks and stale TODO stubs
  that were never finished.

## Output format

Start with a one-block **summary**:

- Total live LOC.
- Estimated removable LOC.
- A **top 10 cleanups by LOC removed** table: `rank | file:line | action |
  est. LOC | one-line rationale`.

Then the full **delete-list**, grouped by action:

### Delete
Confirmed dead — grep-proven zero references. Each: `file:line`, est. LOC,
one-line rationale.

### Simplify
Exists but over-built — inline the single-use wrapper, use the library
built-in, drop the unused knob. Each: `file:line`, est. LOC, what to do.

### Verify
Looks removable but needs a human check before deleting (e.g. referenced
only via dynamic dispatch, a public entry point, or an external caller).
Each: `file:line`, what to verify.

## Steps

1. **Determine scope.** If a path argument is given, audit that path.
   Otherwise audit the whole repo (the in-scope set above).
2. **Mechanical passes first** — reuse existing tooling, do not reinvent it:
   - `uv run ruff check --select F401,F811,SIM .` — unused imports (F401),
     redefinitions (F811), and simplifiable code (SIM). These are mechanical
     bloat; cite the rule code in each finding.
   - **Dependency cross-check:** for each runtime dep in `pyproject.toml`
     (`[project].dependencies`), grep its import name across live code. A
     dep with no live `import` is a removal candidate. Two traps: the
     **import name often differs from the package name** (e.g.
     `python-dotenv` → `dotenv`), and some dependencies are **never
     imported directly at all** — an engine or backend that another
     library loads under the hood is required despite having no `import`
     anywhere. Verify these cases before recommending removal.
3. **Grep-confirm dead symbols.** For every candidate from the checklist,
   grep the symbol name across the repo and confirm it has **no references
   outside its own definition** before listing it under **Delete**. If
   there is any ambiguity (dynamic dispatch, entry point, re-export),
   downgrade it to **Verify**.
4. **Grep the non-obvious call sites before deleting.** `notebooks/`, `app/`,
   `benches/` and `tests/` import from `cablesim` and are easy to miss, and a
   `#[pyfunction]` is called by name from Python rather than from Rust. For
   anything reached across the FFI boundary or from those directories,
   downgrade to **Verify** and say where you looked.
5. **Read the suspicious files** to confirm context before listing — do not
   report from grep counts alone.
6. **Emit the report** (summary + delete-list). Make **no edits** — this
   skill is report-only.

## Note on marimo notebooks

`ruff` is configured to ignore `F401`, `F811`, `F821`, `B018`, `E501`, `I001`
and `S101` under `notebooks/**/*.py` because marimo's reactive graph relies on
bare expressions and cross-cell imports. Do not flag those as bloat in
notebook files — they are required by the framework.
