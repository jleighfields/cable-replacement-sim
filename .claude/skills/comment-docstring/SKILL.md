---
name: comment-docstring
description: Review Python and Rust files for missing docstrings, doc comments, type hints, and inline comments. Generate Google-style docstrings and Rust `///` doc comments, and suggest helpful comments for onboarding. Sweeps nearby READMEs for prose the change invalidated. Edits in place.
disable-model-invocation: false
allowed-tools: Read, Glob, Grep, Edit, Bash
argument-hint: <file-or-directory>
---
# Comment & Docstring Review

Scan Python files for missing or incomplete documentation, and Rust files
under `src/` for missing `///` doc comments on public items. Sweep nearby
READMEs for stale prose the changes invalidate. Fix issues in place and
report what was changed.

Everything below is written for Python; the rules about *what a comment may
say* — self-contained, factual, why not what — apply unchanged to Rust doc
comments. The mechanical rules that name a Python construct (Google-style
`Args`/`Returns` sections, type hints) map onto `///` prose and the signature
Rust already carries in its types.

## Arguments

- **file-or-directory** (required): Path to a `.py` file or directory
  to scan. If a directory, scan all `.py` files recursively.

## What to check

For each function and class:

- **Docstring exists** — every public function needs one
- **Google style** — summary line, Args section, Returns section
- **Type hints** — all parameters and return types annotated
- **Summary is accurate** — matches what the function actually does
- **Parameters documented** — every parameter listed with type and
  description
- **Returns documented** — return type and meaning described
- **An `Examples:` block where the signature leaves the contract open**, and
  nowhere else (see **Examples in docstrings** below)

For the file body:

- **Inline comments** — add comments that explain *why*, not *what*
- **Section headers** — use `# --- Section Name ---` for logical blocks
- **Magic numbers** — explain any hardcoded values
- **Complex logic** — add comments for non-obvious algorithms
- **Self-contained** — no prose that only resolves outside the repo (see
  **Dangling references** below). This is a **fix**, not a note: rewrite
  the sentence around what it was pointing at.
- **Factual** — no rating the reader cannot check: superlatives, invented
  cost figures, personified programs (see **Editorializing** below). Also a
  **fix**: replace the rating with the mechanism it obscures.
- **Terms still name their concept** — no domain term condensed until the
  remainder means something else, "the scale" for a Weibull scale parameter
  as the example (see **Hollowed-out terms** below). A **fix** in prose; an
  identifier that lost its qualifier is reported instead, because renaming
  it changes callers.

## Style rules

CLAUDE.md's **Code style** section is the authority — docstring format, type
hints, import style, and the no-`_`-prefix rule all live there, and nothing
mechanical enforces the last two. Read it rather than a copy of it: `select` in
`pyproject.toml` carries no `UP` rules, so `Optional[X]` where the project
requires `X | None` reaches you unreported.

## Example output

```python
def summarize_runs(run_dir: Path, metric: str) -> pl.DataFrame:
    """Collect one metric from every run under a directory.

    Each run writes a ``config.json`` beside its outputs; this pairs the
    two so runs can be compared without reloading the full result set.

    Args:
        run_dir: Directory holding one subdirectory per run.
        metric: Column to extract from each run's metrics table.

    Returns:
        One row per run: ``[run_label, timestamp, <metric>]``, sorted
        oldest first.

    Raises:
        FileNotFoundError: If ``run_dir`` holds no run subdirectories. An
            empty result is far more often a wrong path than a genuine
            "no runs yet", so this fails loudly rather than returning an
            empty frame.
    """
```

## Dangling references

Comments and docs must stand alone for a reader who has the repo and
nothing else — CLAUDE.md's **Comments & docstrings are self-contained**
states the rule; this is how you find and fix violations.

These references usually come from the work that produced the code. Code
written against a plan inherits the plan's vocabulary, and "the Step 4
loader" is clear while the plan is open. A month later, the phrase has no
referent in the repo, and plans are ephemeral by design — superseded,
moved to `tasks/completed/`, deleted.

**Grep for the ones with a shape.** Plans, plan steps, tickets, commits, and
the stock "as discussed" / "per review feedback" phrasings all match a
regex. Run this over the target files, including `.py`, `.md`, and marimo
cells:

```bash
grep -rnEi 'tasks/|see the plan|step [0-9]|phase [0-9]|sprint [0-9]|as discussed|per (the )?(review|feedback|discussion)|#[0-9]{2,}|(commit|sha) [0-9a-f]{7,}|[0-9a-f]{40}' <paths>
```

**Read for the two classes that have no shape.** The grep cannot find these
and never will — they are ordinary English:

- **The code's own past** — "now uses the closed form instead of sampling",
  "replaces the old two-pass approach", "no longer reads the cache".
- **Dates** — "retired 2026-06-03".

A named person slips through as well ("Justin asked for this"), though the
commoner attributions above do match. So **a clean grep means the greppable
classes are clear, not that the file is** — treat it as one pass over a
file you still have to read.

**Point the grep at the files under review, not the whole repo.** Expect most
hits to be legitimate: a docstring saying "step 3" of an algorithm whose
steps are right there is self-contained, and `tasks/` is a normal path to
name in a file whose subject *is* the tasks directory. Skill, agent, and
plan files number their own steps by design. The test is never the phrase
— it is whether the **referent is in the same file or adjacent context**.

**Fix them** — replace the pointer with the thing it pointed at:

| Instead of | Write |
|---|---|
| "Implements Step 4 of the migration plan." | "Loads the raw extract and normalizes column names." |
| "See `tasks/refactor-2026-05-02.md` §3 for why." | "Two passes: the second needs the first's totals." |
| "Changed in #142 to fix the join." | "Joins on `(site_id, date)` — `site_id` alone multiplies rows." |
| "Now uses the closed form instead of sampling." | *(delete — the code says which library it uses)* |
| "Per review feedback, retry 3 times." | "Retries 3×; the upstream API 502s under load." |

The last two need extra attention. Prose about what the code *used to be*
requires readers to compare against code they cannot see; delete it rather
than rewriting it. And a rationale attributed to a person or a review
still has a real cause; recover the cause and write that instead of
dropping the sentence.

If you cannot recover what a reference meant, say so in your summary
rather than inventing a rationale — an unsupported "why" misleads the
next reader.

## Editorializing

Prose states what is true and how the reader can check it — CLAUDE.md's
**Prose is professional and factual** states the rule; this is how you find
and fix violations. A rating with no evidence describes the author's
opinion rather than the code's behavior. When the code changes,
unsupported ratings do not update with it.

**Grep for the ones with a shape.** Praise words, personified verbs, and
invented cost figures all match a regex:

```bash
grep -rnEi 'powerful|robust|seamless|blazing|elegant|hacky|nicely|gracefully|beautifully|unfortunately|sadly|nightmare|obviously|easily|of course|amazing|awesome|the (most|least|worst|biggest)( [a-z]+|$)|(worst|best) (way|direction|case)|(happily|cheerfully|politely) [a-z]+s|costs? an? (afternoon|day|week|hour)|!!' <paths>
```

**The trailing `|$` in the superlative branch is load-bearing.** These files
hard-wrap near 76 columns and grep matches one line at a time, so a phrase
split across a wrap — "is the most" ending a line, "common defect" beginning
the next — is invisible without it. Any phrase pattern added here needs the
same treatment or it misses wrapped phrases.

**Read for the classes that have no shape.** The grep cannot find these:

- **Personification not on the word list** — "the two things that bite
  newcomers", "the parser is happy to accept". Any verb that gives a program
  a mood replaces the mechanism with that mood.
- **Filler** — "it is worth saying plainly that", "note that", "it is
  important to understand". What follows is the content; the run-up is not.
- **Aphorisms** — "`utils.py` is where code goes to become unfindable". They
  sound conclusive but state no fact.
- **Effort verdicts naming no duration** — "trivial", "a quick fix", "just a
  rename". The grep reaches only the ones quoting a figure (`costs an
  afternoon`); these are the same claim with the figure left out, and nobody
  measured either.

**Expect legitimate hits, and leave them.** `the most recent` is factual
ordering, not a ranking. This file's own offender list quotes every word in
the pattern, so run it over the files under review rather than the repo, or
this file dominates the output. Do **not** add `clean`, `just`, `simply`,
`perfectly`, `deliberately`, or `quietly` to the pattern — they are
common here and essentially every use is ordinary technical usage (`clean
tree`, `lint clean`, `not just the first`, `simply` meaning "and nothing
more"). Adding them creates a triage pass with no findings.

**Fix them** — replace the rating with the underlying mechanism:

| Instead of | Write |
|---|---|
| "Doc drift is the most common defect." | "Doc drift is invisible to reading — that is why it accumulates." |
| "Fails quietly, in the worst way." | "Fails quietly: the suite stays green and the gate reports success." |
| "Which happily succeeds on a file no parser accepts." | "Which succeeds on a file no parser accepts." |
| "The kind of bug that costs an afternoon." | *(delete — the preceding clause already says what the bug does)* |
| "If a fix feels hacky, find the elegant solution." | "If a change only stops the symptom, find what removes the cause." |

**Try deletion first.** Where the mechanism is already in the next clause, the
rating is the unsupported part and cutting it loses nothing. Where it is
not, the rewrite has to supply the mechanism the rating replaced.

**Argument is not editorializing, and must survive this pass.** A claim that
gives its reason in the same sentence belongs here: "two copies of the same
value drift apart over time" is checkable because it explains the failure
mode. Removing the reasoning from a sentence like that damages the
rule. Ask whether the sentence gives the reader something to check — not
whether it is strongly worded.

## Hollowed-out terms

A domain term shortened until the remainder no longer names the concept.
"Customer minutes interrupted (CMI)" written as "the minutes" is the shape, and
the failure is hard to spot because `minutes` is an ordinary English word:
a reader without the domain can parse the sentence and take the everyday
meaning. A term shortened to an abbreviation is visibly unexplained by
comparison — nobody mistakes "the SAIFI" for something they already know.

Short forms appear for the same reason dangling references do. After
repeated use, people already in the conversation stop writing the full
term, and everyone knows which minutes the short form names. That context
is absent for the next reader, and no marker identifies where it stopped.

**Grep for the terms this repo defines.** A term introduced with a
parenthetical expansion is the greppable half, and it tells you which
multi-word terms matter here rather than guessing:

```bash
grep -rnoEi '[a-z]+( [a-z]+){1,3} \([A-Z]{2,6}\)' <paths> \
  | sed 's/^[^:]*:[0-9]*://' | sort -u
```

Then grep each returned term's head noun standing alone. Expect noise in
both directions: the first pass matches ordinary parentheticals, and a hit
on the second is a candidate rather than a finding — "the minutes" is correct
in a sentence that established the term two lines above.

**Read for the ones no file spells out.** A term nobody wrote in full has no
expansion to find. This case needs manual review because a repo where the
short form has fully taken over is also the one where no expansion exists.
The signal is a bare noun that is also common English — scale, shape,
minutes, load, factor, spend, bands — carrying a meaning specific to this
project. Ask what a reader who has the repo and no domain background would
take it to mean, then whether that is what it means.

**Fix them** — restore the qualifier the term lost:

| Instead of | Write |
|---|---|
| "the minutes for each year" | "the customer minutes interrupted (CMI) for each year" |
| "grows with the factor" | "grows with the min-of-n scale reduction factor" |
| "stays inside the budget" | "stays inside the annual replacement budget — the capital the policy may spend that year" |

**Expand on first use per file, then use the short form.** Every occurrence
expanded adds no new information after the first, and a reader who has met
the term once does not need it again. The first mention — in the module
docstring, or in the docstring of the first function that uses it — is
where the expansion belongs.

**This skill does not rename identifiers.** Where the hollowing reached the
identifiers — `minutes_by_year()` where `cmi_by_year()` names the series —
the repair is a rename, which changes every caller. This skill edits in place,
so renaming would change the API inside a pass nobody expects to touch it.
Name it in your summary and leave it alone; `/code-quality-review` reports
it, where a human decides.

## Examples in docstrings

Google style has an `Examples:` section, and most functions should not have
one. It is prose that can drift, with a trap the rest of a docstring does not
have: it looks executable, so readers treat it as tested.

**Add one where the signature leaves the contract open** — and nowhere else:

- **The types do not say what is in them.** `pl.DataFrame`, `dict`, or a nested
  container names the container and nothing about its contents. Show the
  columns or keys, with units.
- **A plausible wrong call exists.** Two parameters of the same type that can
  be swapped, a value in MW where MWh is meant, a ratio that reads as a
  percentage.
- **The calling convention is not obvious** — a parameter that only does
  anything when another is set, an ordering requirement, a resource the caller
  must hold open.
- **It is an entry point** callers reach for directly.

`add(a: int, b: int) -> int` needs none, and one showing `add(1, 2)` adds no
contract information.

**Write it as a real call with real shapes, and run it before leaving it in.**
Invented column names document names the code does not produce. An example is a
claim about behaviour, so *Verify by running* covers it — and that cost is the
reason to keep them scarce: each one is a claim someone must re-check when the
function changes, and a stale one misleads because it still looks tested.

## README sweep

Every pass under this skill should also sweep nearby READMEs for prose
the diff just invalidated — stale formulas, removed function or column
names, retired flags, outdated tables. No automated check reports most
README drift: the code changes, the doc still describes the old behavior,
and the next developer relies on outdated instructions.

**Which READMEs to check.** For each touched file, sweep the README in
its own directory and the one directly above it, plus the root
`README.md` when the change affects usage, the public API, or a
documented workflow. Directory READMEs are orientation notes that point
at whatever owns the detail — if a sweep finds one restating detail that
lives elsewhere, cut it down to a pointer rather than updating both.

**What to look for.** Grep the README set for the name of every removed,
renamed, or refactored symbol in the diff:

- **Removed constants, dicts, or functions** still cited by name.
- **Renamed columns or fields** referenced by their old name in tables
  and bullet lists.
- **Outdated formulas** describing a calculation the code no longer does.
- **Stale config tables** — a field list mirroring a config model drifts
  every time a field is added or removed. Cross-check it against the
  model.

**What not to do.** Do not add dates or "removed in" history. Prose
describes the current state; `git log` carries the history. Write "the
cost path carries no transport term", not "transport was retired on
<date>". Do not add a migration-notes or changelog section to a README to
track a refactor — that belongs in `tasks/`.

## Project-specific additions

- **Rust doc comments follow the same rules as docstrings.** `///` on every
  public item in `src/`, saying what it computes and what its arguments mean.
  A `#[pyfunction]` signature is a contract two languages read, so it gets a
  doc comment on the Rust side and a docstring on the Python wrapper, and
  they must not say different things.
- **A change to the config schema reaches three files.** The pydantic model,
  `configs/base.yaml`, and whatever section of `README.md` documents the
  knob. A field added to the model without a commented entry in the YAML is
  invisible to anyone reading the config as documentation.
- **`PLAN.md` is a durable in-repo reference and may be cited by section —
  with the section title alongside the number.** A bare `§9` silently became
  `§10` the day a section was inserted ahead of it; `§10, Phased roadmap`
  shows the mismatch.
  It is checked in and is where the modeling decisions are argued — a comment
  saying "min-of-n scale reduction, PLAN.md §2.3" points somewhere that will
  still exist. This is the exception to the rule against pointing outward;
  the rule still bans references to `tasks/` plans, commits and tickets.

## Steps

1. Read the target file(s)
2. For each function/class, check docstring completeness
3. For each function signature, check type hints
4. Scan for places where inline comments would help
5. Find **dangling references** — grep for the classes that have a regex
   shape, read for the two that do not — and rewrite each one around its
   referent
6. Find **editorializing** — grep for the shaped classes, read for the four
   that have no shape — and replace each rating with the mechanism under it
7. Find **hollowed-out terms** — grep the parenthetical definitions, then
   read for the bare common nouns the grep cannot reach — and restore the
   qualifier in prose. List any *identifier* that lost one rather than
   renaming it
8. Add an **`Examples:` block** to each function whose contract its signature
   leaves open, and to no others — then run each example against the current
   code before leaving it in
9. Apply the **README sweep** above — grep the relevant READMEs for every
   removed or renamed symbol in the diff and fix the prose it invalidated
10. **Where the diff changes the config schema, sweep `configs/base.yaml` and
   `README.md` too**, and where it changes a `#[pyfunction]` signature, sweep
   the Python wrapper's docstring — the pairings above.
11. Make edits directly (don't just report — fix)
12. Report a summary of changes made, including any README updates, any
   reference whose meaning could not be recovered, and any name whose
   qualifier this skill deliberately did not restore

**This skill does not run the test suite**, though its edits are the kind that
can break one — a docstring is inert, but a type hint corrected in passing is
not. The `code-reviewer` agent's full pass runs the suite once after these
edits, which is the run that covers them; nothing else does. Wherever no run
follows, name the suite among what you did not run, so the summary does not
imply the edits were checked.
