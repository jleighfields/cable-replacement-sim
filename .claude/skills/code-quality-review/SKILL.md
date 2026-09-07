---
name: code-quality-review
description: Review code for correctness bugs plus readability, documentation quality, onboarding ease, and minimal form (simplification per the project Minimalism rules). Report-only — presents findings, makes no edits.
disable-model-invocation: false
allowed-tools: Read, Glob, Grep, Bash
argument-hint: "[file-or-directory]"
---
# Code Quality Review

Review code for **correctness bugs** plus readability, documentation
quality, ease of on-boarding, and whether the code is in its minimal
form (simplification per the project Minimalism rules). Report issues
but do NOT make edits — present findings for the user to approve.

## Arguments

- **file-or-directory** (optional): Path to review. If omitted, review
  all staged and unstaged changed files (`git diff --name-only HEAD`)
  unioned with the untracked ones (`git ls-files --others
  --exclude-standard`) — union, not fallback.

## Verify by running

**A claim you can run is not reviewed until you have run it.** This
governs the whole pass, not one checklist item. It applies to every claim
you meet and every claim you make:

| The claim | How you settle it |
|---|---|
| A docstring, comment or README describes behavior | Call the thing and compare |
| A doc names a command, flag or path | Run it; open the path |
| A script or workflow step is said to work | Execute it |
| A gate/validator rejects bad input | Feed it bad input and watch it reject |
| A generated or committed artifact has some content | Open the artifact |
| You suspect a bug | Construct the input and trigger it |
| A dependency, version or env var is assumed present | Resolve it; import it |

Reading tells you what the author believed. Running tells you what the code
does, and defects survive when those differ. Prose can go stale without any
syntax or test failure, so plausibility is not evidence.

Where running is genuinely infeasible — it needs prod credentials, a
cluster, a paid API, hardware you don't have — say so in the finding
rather than presenting a read-only judgement as a verified one.

You have `Bash`. Use it throughout the review, not only at the end.

### Look for a harness before building one

Read `tests/` before hand-rolling a fixture. A repo that has needed one of
these before usually has a builder for it already — something that assembles a
throwaway project, a repo, a config — and reusing it keeps your fixture
identical to the suite's. A hand-rolled one differs in ways you did not
choose, so a finding proved against it may not reproduce against the tests.

This is the Minimalism hierarchy applied to verification rather than to
shipped code: use what is there, then extend it, and only then write your own.
Extending is often the right answer — a builder that takes one more argument
serves the next review too.

Say which you did. "Reused the `segment_table` fixture from
`tests/conftest.py`" and "the suite had no builder for a repo in this state,
so I built one" are both acceptable; building a ninth bespoke fixture beside
eight existing ones is not, because the next reviewer then has nine to choose
from and no reason to prefer any.

**If you did build one, say whether it should stay.** A builder is not the
test — it is reusable setup for the *next* test, and it is the part most likely
to be needed again, because the states worth constructing are few and keep
recurring. When the state you had to construct is one a reasonable next change
would also need, propose the builder as an addition to the suite's helpers
rather than as lines inside a single test: which existing function it extends,
or what it should be called if it is genuinely new.

Do not propose one for a state that took two lines to reach, and do not
propose one you only used once and cannot name a second use for. Propose it
when the same state is likely to recur; otherwise keep it local.

### Keep useful hand-built verification

Settling a claim often means building a harness: a fixture repo, a crafted
input, a few lines that drive a function and print what came back. When that
harness confirms a real defect, **say so in the finding and give the harness
alongside it**, phrased as the test it wants to become — which function, what
input, what assertion.

When a harness is thrown away, the next review rebuilds it from scratch, and
nothing prevents the defect from returning in between. A harness turned into a
test costs one more paragraph now and runs on every commit. Prioritize silent
wrong behavior, a guard that does not guard, and an error path nobody
exercises.

Apply it where a test would be small and durable, not to everything. A
one-off `grep` proving a doc names a real path is not a test. Skip it too when
the repo has nowhere for the test to live and creating that home is a bigger
change than the fix — say that, rather than omitting the harness.

**Name the mutation that must make it fail.** Every proposed test includes one
line saying what to break to see it go red — "revert the range to `,$p`",
"return the first span instead of all of them", "drop the `strict=`". Without
it a proposal is only a suggestion: the implementer can substitute something
weaker that passes either way. A stated mutation makes the proposal checkable
in one command — apply it, watch red, revert, watch green — and a weaker
substitute fails that check immediately rather than at the next review.

This is the repo's own rule about watching a check fail. The extra line proves
the test rejects the defect rather than only confirming today's code.

**The same question is asked of the tests already in the diff**, and settling
it is not this skill's job. A test the change set adds or edits is a test
nobody has watched fail either, and it stops being watched the moment it is
written — file-by-file review does not recheck it.
`.claude/skills/test-review/SKILL.md` is where that is settled, by breaking the
code in a throwaway worktree and confirming the test reddens. **Its caller is
`code-reviewer`'s Phase 4**; a direct invocation of this skill has no such
phase, and wherever no mutation follows, name it among what was not run rather
than leaving a reader to assume it happened.

**A finding's rationale is a claim like any other.** Where it asserts something
about behaviour and was written from reading, *Verify by running* covers it as
it covers everything else. If the conclusion was proved by running but the
explanation was not, say which part remains unverified.

This skill stays report-only: propose the test, do not write it.

## Read for what the linter cannot see

`ruff` has already run by the time you read anything, with the rule selection
in `pyproject.toml`, and the build fails on what it finds. Several checklist
items below are in that set — they are written down because they are project
rules, not because a reading pass is how they get caught:

| Already mechanical | Rule |
|---|---|
| Unused imports, unused locals, undefined names | `F401`, `F841`, `F821` |
| Import order | `I001` |
| Line length | `E501` |
| Simplifiable constructs, likely bugs | `SIM`, `B` |
| Insecure patterns backing the security phase | `S` |

If `ruff` passed, those are settled — do not spend a reading pass confirming
them, and do not report one as a finding without a rule code, because the
linter disagreeing with you is the more likely explanation.

**Check the selection in `pyproject.toml` rather than trusting this table.**
It is a summary of a configured list, and a summary is wrong the moment the
list changes. `UP045` was in this table until it turned out the selection
carries no `UP` rules at all, so `Optional[X]` — which this project forbids in
favour of `X | None` — reached readers unreported while the table told them not
to look.

What no rule in that selection covers, and what the reading is therefore
**for**: `X | None` rather than `Optional[X]`, missing or wrong docstrings,
missing type hints, whether a comment explains *why*, whether a name means
anything, duplication across files, a value written in two places, and every
correctness question below — the logic, the data shapes, the silent no-ops.
Those are the manual checks a second reader can add.

## Review checklist

The checklist is deliberately **unnumbered**. Refer to an item by its
bold name and to a group by its section heading — never by position.
Numbered cross-references become wrong when an item is added or removed; the
same section has already been cited three different ways across sibling repos.

### Correctness & bugs

Real defects are the top priority — every confirmed bug is a **Must Fix**.
Read for behavior, not just style: trace the changed code with concrete
inputs and ask "what input makes this do the wrong thing?"

- **Wrong logic** — inverted conditionals, `and`/`or` mix-ups, off-by-one
  bounds, wrong comparison operator, a mishandled edge case (empty input,
  single element, zero, negative).
- **None / missing-key / index errors** — a value that can be `None` used
  without a guard; `dict[key]` / `df[col]` / `list[i]` that can raise
  `KeyError`/`IndexError`; a `.get()` result not checked.
- **Data-shape bugs** — a join/merge on the wrong keys or that silently
  multiplies rows; a filter that drops or duplicates records; a group-by
  missing a dimension; a value that leaks across a partition it should
  have been scoped to.
- **Resource / lifetime bugs** — using an object after it is closed or
  garbage-collected; reading a result after the thing that produced it is
  gone; a file/connection/handle not released.
- **Error handling** — an exception swallowed (`except: pass`) that hides a
  failure; catching too broad a type; the wrong exception type; a `finally`
  that masks the original error.
- **Async / concurrency** — a missing `await`, a coroutine never awaited,
  shared state mutated without protection, or ordering assumptions a
  `gather`/thread pool breaks.
- **State & mutation** — mutating a shared or input object as a side effect;
  a mutable default argument; aliasing that surprises the caller.
- **Contract mismatch** — a call whose argument order/types don't match the
  callee's signature; a return value the caller uses wrongly; units or sign
  conventions that don't line up.
- **Silent no-ops** — an operation that fails by doing nothing and reports
  success: a replace whose pattern never matched, a loop over a list that is
  always empty, a filter that excludes everything, a scan pointed at a
  missing or unreadable input, a check whose condition cannot be false.
  Nothing raises, nothing changes, and every downstream step passes for the
  wrong reason. Ask of each one: **what would I see if this had done
  nothing?** If the answer is "the same thing I see now", it is unverified.

For each suspected bug, state the **concrete input/state that triggers it**
and the **wrong result** (a crash, a silent wrong value, a corrupted
output). Per **Verify by running**, trigger it rather than arguing it, and
say which findings you confirmed that way and which you did not. If you
cannot construct a failing case, mark it a lower-confidence **Consider**,
not a Must Fix. Ruff `F`/`B` findings (`F821` undefined name,
`B006` mutable default, `B008`, …) are mechanical bugs — fold them in here as
Must Fix.

### Readability

- **Function length** — flag functions > 50 lines. Can they be split?
- **Variable names** — are they descriptive? Flag single-letter names
  except a conventional loop index.
- **Hollowed-out names** — an identifier that condensed a multi-word domain
  term until what is left names something else: `scale_*` for a Weibull
  scale parameter, `factor` for the min-of-n reduction factor. The remainder
  is usually an ordinary word, so the name looks descriptive while pointing
  to the wrong concept. Report it with the full term and a
  suggested name, and stop there — renaming changes every caller, which is
  why `/comment-docstring` restores the term in prose and deliberately
  leaves the identifier alone.
- **Nesting depth** — flag > 3 levels of nesting. Can early returns
  or guard clauses simplify?
- **Magic numbers** — flag hardcoded values without explanation.
- **Dead code** — commented-out code, unused imports, unreachable branches.
- **Silent failures** — missing file/dir checks that skip without logging
  a warning.
- **Arbitrary decisions** — thresholds, caps, multipliers, or logic
  branches with no comment explaining *why* that value or approach was
  chosen (e.g. a 1.15 scaling factor or a 50-unit floor with no
  justification).

### Documentation

- **Missing docstrings** — every public function and class needs one.
- **Outdated docstrings** — does the docstring match the current code?
- **Stale claims** — prose asserting behavior the code no longer has: a
  flag's effect, a file's contents, what a tool outputs, which values a
  constant may hold. Settle each by running it (see **Verify by
  running**). Doc drift is invisible to reading because stale prose stays
  well-formed after the code changes.
- **Missing type hints** — all function signatures need types.
- **Contract the signature does not carry** — a parameter or return whose
  shape cannot be worked out from its annotation, with nothing showing one.
  `pd.DataFrame`, `dict` and nested containers name the container and
  nothing about the contents, and two parameters of the same type can be
  swapped by a caller reading only the signature. The fix is an `Examples:`
  block with real columns and units, not more prose — flag which of the two
  it is, since a docstring paragraph restating the annotation is what this
  usually gets instead.
- **Confusing comments** — comments that describe *what* instead of *why*.
- **Missing comments** — complex logic without explanation.

### Style (per CLAUDE.md)

- **Google-style docstrings** with summary, Args, Returns.
- **`X | None`** syntax (not `Optional[X]`).
- **Direct imports** for type hints (no forward references).
- **No `_` prefix** on function names. Internal helpers still get
  real names.

### Code duplication & helper functions

Flag repeated patterns and recommend concrete extractions. This is
a **Should Fix** at 2 copies and a **Must Fix** at 3+.

- **Near-identical functions** — two or more functions that share
  >50% of their logic with only minor parameter differences (e.g.
  different format strings, different column names). Extract the
  shared body into a parameterized helper and make the public
  functions thin wrappers. Example: two policy scorers differing only in how
  they rank should share a `score_candidates(segments, rank_by)` helper
  rather than duplicating the scoring loop.
- **Repeated multi-line patterns in notebooks** — if the same 3+
  line sequence appears in multiple cells (build a figure → add a
  series → reorder → restyle), extract it into a notebook-local helper
  or a module-level function. Marimo cells must remain separate
  `@app.cell` defs for the reactive graph, but the *body* of each cell
  can call a shared helper.
- **Copy-pasted logic with small variations** — loops, conditions,
  or data-processing blocks that were clearly copied and tweaked (the
  same filter-iterate-append pattern applied to two related
  collections). Extract into a function parameterized on the varying
  parts.
- **Hardcoded values repeated across files** — the same magic number
  or string literal appearing in 2+ files without a shared constant.
  Extract to a module-level constant or config field.
- **When NOT to extract** — do not flag single-use patterns shorter
  than 3 lines, marimo cell signatures (they must list dependencies
  explicitly), or test setup code (test clarity > DRY).

When flagging duplication, always include:
- Which functions/blocks are duplicated
- How many copies exist
- A concrete suggested helper signature (name, parameters, return type)

For the broader minimalism lens (code that shouldn't exist, reinvention
of built-ins, premature abstraction), see the **Simplification** section
below.

### Single source of truth for parameter values

Often the highest-priority class of finding. Parameter values (numbers,
dicts, config entries) get ONE home; other modules read from it.
Duplicate-source-of-truth bugs drift silently and affect every caller that
uses the wrong copy. Flag every instance, with severity calibrated to blast
radius:

- **Must Fix** when the duplication is in a production code path, or
  when the two copies have already drifted to different values.
- **Should Fix** when the duplication is in diagnostics, charts, or
  tests and the values currently agree — drift is a matter of time.

- **Compile-step duplication of a config dict** — one shape this bug takes,
  and the one to check for first. A config dict plus a parallel mirror in a
  runtime module, kept in sync by a `derive_*.py` / `compile_*.py` script
  someone has to remember to rerun. **This pattern is itself the anti-pattern,
  not an architecture that needs guardrails.** Editing the source dict without
  rerunning the compile step silently diverges runtime behavior from what the
  editor expected — no warning, no error, just wrong output.
  Adding a startup assertion that compares the two dicts is **not** a fix:
  the duplicate is still there; the assertion only catches the drift later.
  **Must Fix. The correct
  resolution is to delete the compile step and collapse to one dict** —
  if the split was forced by a circular import, restructure the imports
  (move shared constants into a leaf module) so the duplicate is not
  needed. Challenge any new compile-step proposal on the same grounds.
- **Hardcoded scalar where a config constant exists** — a literal in a
  runtime module that also lives as a named constant in the config
  module. Two sources for one value drift apart on the next edit. Fix:
  read the config constant and delete the literal. Watch especially for
  rates, factors, fallbacks, and unit-conversion constants.
- **Module-level constant imported directly when a config field wraps
  it** — one caller does `from ...config import SOME_MAP` while the
  runtime path reads `cfg.some_map`. Per-instance overrides then reach
  the runtime path but not the direct importer, so a scenario override
  silently applies in one place and not the other. Fix: route both
  through the config object, or document why one deliberately ignores
  overrides.
- **Mismatched defaults across functions for the same logical value** —
  e.g. a config field defaulting to `"quarterly"` while a function
  signature defaults the same concept to `None`. A caller that omits
  the argument gets a silent behavior split between tests and
  production. Fix: align the defaults, or make the parameter required.
- **A function reimplements logic a runtime function already has** — a
  chart or report recomputing a value from raw inputs when the
  production path already computed it with extra handling (resampling,
  gap-filling, unit conversion). Even a tiny difference compounds. Fix:
  delegate to the function that produced the canonical value, or read
  its stored output, rather than recomputing.
- **Two functions compute the same per-key value independently** — the
  same derived quantity calculated in two places from *different* source
  sets. Cache the canonical computation once and have both callers read
  it. Whichever function wrote the canonical value stays the sole
  computer.

When flagging a source-of-truth violation, include:
- The two (or more) source locations
- Whether they currently agree (same value, two copies) or already drift
- Which one is canonical (the runtime/production source usually wins)
- The signature change required to consolidate

### On-boarding ease

- **Would a new team member understand this?** — flag sections that
  need context comments.
- **Are error messages helpful?** — do assertions explain what went wrong?
- **Are log messages informative?** — do they help debug a pipeline run?
- **Are READMEs up to date?** — do they reflect the current code?

### Simplification (per minimalism rules)

Apply the **Minimalism (write less)** hierarchy from `CLAUDE.md` to the
changed code: walk it top to bottom and flag where the diff skipped an
earlier Minimalism step. This lens is about the *form* of the new code (is it
minimal?), as distinct from **Code Duplication & Helper Functions** above
(is it repeated?) and the `simplify-audit` skill (is there dead/excess
code across the *whole repo*?). Report only — do not edit.

- **Existence / YAGNI** — does the new code need to exist? Flag
  speculative helpers, unused parameters, config knobs, or branches the
  diff adds "just in case" with no current caller.
- **Reinvention** — hand-rolled logic that duplicates a built-in from the
  stdlib or from a library this project already imports. **Read the
  imports at the top of the file under review** rather than assuming a
  fixed library set, and cite the specific built-in that replaces the
  hand-rolled code (e.g. a manual accumulation loop that a single
  group-and-sum call does in one line).
- **Single-use abstraction** — a wrapper, helper, or class the diff
  introduces for exactly one call site. Recommend inlining. This is the
  inverse of the **Code Duplication & Helper Functions** section above:
  extract at 2–3 copies, inline at one.
- **Premature generalization** — parameters, `**kwargs`, or branches that
  handle cases which do not occur in the codebase yet.
- **Smallest correct form** — multi-line constructs that collapse to a
  comprehension, vectorized op, or single call; needless intermediate
  variables.

For each simplification finding, cite the Minimalism step (1–6) it maps
to so the user sees which rule applies. Those steps *are* numbered — they
are an ordered hierarchy you walk until one solves the problem.

### Project-specific additions

- **The Python reference and the Rust kernel are a deliberate mirror, so
  "duplication" here means *drift*, not repetition.**
  `python/cablesim/policies.py` and `src/policies.rs` implement the same
  scoring, and `python/cablesim/simulate.py` and `src/simulate.rs` the same annual
  loop. Do not propose collapsing them — the parity tests are what validate
  the kernel. Do report any place the two have diverged, and treat a numeric
  literal appearing in both as a Must Fix: the copies drift, and the parity
  test is the only thing that would notice.
- **Every run knob lives on the pydantic config model.** A tunable read from
  a module-level global, or a default written in Python and again in Rust, is
  a source-of-truth finding. Fixed constants a caller overriding would be a
  *bug* are the exception and belong in the package's constants module.
- **Notebooks and the app import from `cablesim` and define no modeling
  logic.** A function defined in `notebooks/` or `app/` that computes
  anything about failure, cost, or reliability belongs in the package; report
  it with the module it should move to.

## Output format

Group findings by severity. **Number findings sequentially across
all three buckets** (1, 2, 3, ...) starting at 1 in Must Fix and
continuing through Should Fix and Consider. Sequential numbering
gives every finding a unique short ID the user can reference in
conversation ("apply 1, 4, 7", "skip #11"). These numbers are generated
per run and describe *findings*, not checklist items — they are the only
numbering in this skill that carries meaning.

### Must Fix
- **Confirmed correctness bugs** (a concrete input produces a crash or wrong
  result) — always the highest priority
- Critical issues (wrong docstrings, misleading comments, missing types)

### Should Fix
- Important readability issues (long functions, missing comments)

### Consider
- Style suggestions, minor improvements

For each finding, include:
- A leading sequential number (continuing from prior bucket)
- File and line number
- What the issue is
- Suggested fix (brief)
- **The harness that confirmed it**, where one was built and a test would be
  small enough to keep — named as the test it wants to become, so the next
  reader can reuse the proof rather than rebuilding it, and with the mutation
  that must make it fail

Close with **what was not run**, named individually — the test suite this
skill does not own, a claim left unsettled because settling it needed
something out of reach, anything a caller scoped out before invoking this. A
reader deciding how far to trust the findings needs to know which runs stand
behind them; an omitted check can otherwise be mistaken for a pass. Say so
explicitly when nothing was left out.

Example layout:

```
## Must Fix

### 1. `path/to/file.py:42` — function raises but lacks Raises: section
...

## Should Fix

### 2. `path/to/file.py:100-150` — function is 80 lines, split into ...
...

### 3. `path/to/other.py:5` — duplicated logic with file.py:42
...

## Consider

### 4. `path/to/file.py:60` — variable name `x` could be `row_count`
...
```

## Steps

1. Identify files to review:
   - If a file or directory argument is provided, review those files
   - **If no argument is provided**, you MUST run this command to find
     all changed files and review every one of them:
     ```bash
     git diff --name-only HEAD
     git ls-files --others --exclude-standard
     ```
     **Union both lists — the second is not a fallback for the first.**
     A change set that edits one tracked file and adds twenty untracked ones
     is ordinary; an "only if the first is empty" rule reviews the one and
     skips the twenty. Review ALL files returned — do not skip any.
2. Run static checks on the changed files first — these surface
   issues mechanically before you start reading:
   - `uv run ruff check <changed-files>` — unused imports, undefined
     names, style violations, marker for X | None vs Optional[X]
   - `cargo clippy` and `cargo fmt --check` — where the changed set holds
     `.rs` files. Ruff does not read Rust, so without these the kernel gets
     no mechanical pass. Cite the lint name in each finding the way ruff
     codes are cited, and treat a `clippy::correctness` lint as **Must
     Fix**. `test.yml` fails the build on either, so a review that skips
     them cannot catch what CI rejects.

   **This skill does not run the test suite.** Whether the change set passes
   is a fact about the change, not a review finding, and its caller
   establishes it — the `code-reviewer` agent's full pass does, after its
   docstring pass, where one run also covers the edits that pass makes. What
   belongs here is the targeted run that settles a particular claim, which
   *Verify by running* above describes. **Its `commit` mode does not run the
   suite**, and neither does a direct invocation; wherever no run follows,
   name the suite among what was not run, rather than letting silence imply a
   pass.

   Treat any ruff finding as **at least Should Fix**; F821 (undefined
   name) and most B-class rules are **Must Fix** since they're real
   bugs. Cite the rule code (e.g. `F401`, `B008`) in each finding so
   the user knows what `--fix` would do. Skip the **Dead code** and
   **Direct imports** checklist items — ruff covers them more reliably
   than human review. **Do not report `S` (flake8-bandit) findings
   here** — those are security concerns owned by the `security-scan`
   skill (the `code-reviewer` agent runs it as a separate phase);
   reporting them here would double-count.

   Formatting is deliberately **not** checked: this project does not
   enforce `ruff format`, so `ruff format --check` would report drift
   on files nobody intends to reformat. Do not run it or report it.

   Note on marimo notebooks: check `pyproject.toml` for what
   `[tool.ruff.lint.per-file-ignores]` actually sets under `notebooks/**`
   rather than assuming a list — it is currently B018 and S101, for marimo's
   bare trailing expression (a cell's displayed output) and for a notebook
   asserting on a value it has already computed. Don't manually re-flag what
   that block ignores; do flag anything it does not.
3. Read each file fully — do not skip any changed files. As you read, keep
   a running list of the claims the file makes — what a comment says a flag
   does, what a docstring says a function returns, what a doc says a command
   prints — and settle them per **Verify by running** before moving on.
4. Apply the checklist to every changed file, checking for **correctness bugs
   first** (the **Correctness & Bugs** section — every confirmed bug is a
   Must Fix), then paying special attention to the **Single Source of
   Truth for Parameter Values**, **Code Duplication & Helper Functions**,
   and **Simplification** sections. For each duplication finding, include
   a concrete helper signature so the fix is actionable; for each
   simplification finding, cite the Minimalism step (1–6) it maps to; for
   each source-of-truth finding, say whether the copies have already
   drifted.
5. **Where the change touches one side of the Python/Rust mirror, open the
   other.** A change to `src/policies.rs` or `src/simulate.rs` means reading
   `python/cablesim/policies.py` or `simulate.py`, and the reverse. Report a
   one-sided change as a finding even when both sides still compile — the
   parity test is statistical and may not catch a small divergence.
6. Present findings grouped by severity, ruff findings cited inline
7. Do NOT make edits — let the user decide what to fix

## After the review: present, then gate

This skill is report-only: **present every finding (numbered) for the
user to review — never auto-fix.** But the findings are a tracked
checklist, not advisory prose:

- Treat each **Must Fix** and **Should Fix** as an open item referenced
  by its number. Do **not** open the branch's pull request until every such
  item is either fixed or **explicitly waived by the user** — the deadline
  `CLAUDE.md` sets. A finding raised at the commit tier stays open until
  then rather than blocking that commit.
- "Pre-existing" / "out of scope" is never a reason to omit a finding — name
  the finding and get the user's waiver.
- **Consider** items are optional and do not block.
