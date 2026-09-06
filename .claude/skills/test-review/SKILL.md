---
name: test-review
description: Check whether tests can fail. Mutates the code under test in a throwaway git worktree and confirms each test goes red; reports the ones that do not, with the mutation that proves it. Never writes to the branch under review.
disable-model-invocation: false
allowed-tools: Read, Glob, Grep, Bash
argument-hint: "[file-or-directory]"
---
# Test review

One question, asked of every test in scope: **can it fail when the code it
names is wrong?**

A test that cannot fail still counts in the suite total, so the code it names
appears covered while the bug-catching test is still missing. Reading alone
cannot prove failure behavior: a test can have a clear name, a real assertion,
and a careful docstring while no code change can make it fail. Prove it by
breaking the code and watching the test run.

**This skill checks failure behavior by mutation, not by reading.** Where that
is impossible — the mutation is infeasible, the suite will not start — report
which tests it could not reach instead of counting them as checked.

## What this skill writes to the branch under review: nothing

It mutates **the code under test**, never the test, and does that in a
throwaway `git worktree`. The skill reports findings only. Whoever owns the
feature branch makes the repair there — the same division
`code-quality-review` and `security-scan` use.

A worktree registers itself in `.git/worktrees` until removed, so an
interrupted run leaves a record and a second checkout on disk. Step 5 clears
both and is not optional.

## The isolation protocol

A mutation pass that skips a step here can report findings from a run that did
not prove the mutation reached the tests.

**Every block opens with an explicit `cd`.** Steps 1 and 5 act on the reviewed
repo, steps 2 and 3 inside the worktree. The working directory persists between
tool calls in many harnesses, and step 2 always moves it. Record **two** paths
from step 1: `ROOT` and `WT`, both absolute — shell variables do not survive
between calls, so every later block re-assigns them from the recorded literal.

### 1. Build the worktree

```bash
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
WT="$(dirname "$ROOT")/.test-review/$(basename "$ROOT")/$(git rev-parse --abbrev-ref HEAD)-$(git rev-parse --short HEAD)"
git worktree add --detach "$WT" HEAD
echo "root:     $ROOT"
echo "worktree: $WT"
```

**`--show-toplevel`, not `$PWD`.** Invoked from a subdirectory, `$PWD` puts the
worktree inside the repo and names it after that subdirectory.

**Outside the repo, under a sibling `.test-review/`.** An in-repo worktree puts
a second copy of `tests/` where any `rglob` walk finds it, and shows up as `??`
in `git status`.

**`--detach`, so there is no branch.** Nothing here is committed or pushed, no
ruleset covers `test/*`, and detaching sidesteps git's refusal to check out one
branch in two worktrees — which would otherwise block reviewing the branch you
are on.

**The path names the repo, branch and commit**, so a stray checkout identifies
the run that created it, and a second run on the same commit fails at
`worktree add` rather than reusing a directory the previous run left mutated.

**The tests under review have to be committed.** A worktree checks out a
commit, so uncommitted edits are invisible to the run. Report that limitation
instead of reporting against a version of a file that no longer exists.

### 2. Prove the mutation reaches the test run

This step is required.

An editable install resolves to the path it was installed from, so a worktree
that inherited its environment can import the **original, unmutated source**.
Mutations then affect no imported code, every test passes, and the run reports
a clean suite without testing the changed file. The layout determines whether
this happens: with the package at the repo root the worktree's copy shadows
the installed one and mutations arrive; with a `src/` layout they do not.
`uv sync` inside the worktree points the environment at the worktree in both
cases.

```bash
WT=<the worktree path step 1 printed>
[ -f "$WT/.git" ] || { echo "not a linked worktree — refusing to run"; exit 1; }
cd "$WT"
uv sync
```

**Repeat that guard in every block that acts on `$WT`.** `cd "" && …` exits 0
and leaves the shell where it was, so an unset `WT` does not fail — it sends
every following command to the reviewed branch, writing the mutation into the
working tree under review.

**`-f`, not `-d`.** In a linked worktree `.git` is a *file* holding a `gitdir:`
pointer; only the main checkout has it as a directory. `-f` refuses an empty
path, a missing path, and the repo root itself. `-d` accepts those cases and
rejects the linked worktree.

Then sabotage something that **must** break a known test, and confirm it goes
red. Pick a canary whose failure is unambiguous — for example, an early
`return None` in a function a named test asserts on instead of a subtle edit:

```bash
WT=<the worktree path step 1 printed>
[ -f "$WT/.git" ] || { echo "not a linked worktree — refusing to run"; exit 1; }
cd "$WT"
uv run python - "$WT/src/pkg/module.py" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
t = p.read_text(encoding="utf-8", newline="")
old = "    return compute(value)"
assert t.count(old) == 1, f"mutation site not unique: {t.count(old)}"
p.write_text(t.replace(old, "    return None"), encoding="utf-8", newline="")
PY
```

**Every mutation asserts it landed.** One that matched nothing leaves the file
untouched, the test passes, and the run reports a test as checked without
changing the code it covers.

**`newline=""` on both the read and the write.** Without it Python translates
line endings, so on Windows a one-line edit rewrites every line — and a hash or
manifest test then fails for a reason unrelated to the mutation.

**If the canary does not fail, the run is invalid.** Report that and stop.

### 3. Mutate, one at a time

Two at once mask each other — the second can offset the first, or it can make a
test fail for the wrong reason. Several worktrees in parallel are fine; one
worktree with two edits in it is not. Run the **targeted** test, not the suite:

```bash
WT=<the worktree path step 1 printed>
[ -f "$WT/.git" ] || { echo "not a linked worktree — refusing to run"; exit 1; }
cd "$WT"
uv run pytest -q -k "<test name>"
```

### 4. Report the mutation with the finding

"Deleted the `.cast()` on line 40 and `test_prices_stay_decimal` still passed"
is checkable by the next reader in one command. "This test looks weak" is not.
Every finding includes the test, the mutation applied, and what happened.

### 5. Tear down

```bash
ROOT=<the root path step 1 printed>
WT=<the worktree path step 1 printed>
cd "$ROOT"
[ -f "$WT/.git" ] || { echo "not a linked worktree — refusing to run"; exit 1; }
git worktree remove --force "$WT" || { echo "TEARDOWN FAILED — do not report clean"; exit 1; }
git worktree list
```

**`cd "$ROOT"` first, and require a zero exit status.** Steps 2 and 3 leave the
shell inside the worktree, and `worktree remove` run from in there fails with
`Permission denied` — but it deregisters the worktree *before* it fails to
delete the directory. Every later confirmation then reports clean while an
empty directory survives that neither `list` nor `prune` nor a second `remove`
can reach, because all three read a record that is already gone.

**`prune` is not a substitute for `remove`.** Prune only discards the record of
a worktree whose directory is already gone. It is the follow-up for a directory
deleted by hand, not the recovery for an interrupted run — for that, `git
worktree list` names what is still registered, `remove --force` each, then
`prune`.

**Use `git worktree list` for recovery, never the directory's presence.**
`remove` deletes the leaf and nothing above it, so `.test-review/<repo>/`
survives every successful teardown as an empty skeleton. The directory's
presence says only that runs have happened here.

Confirm the reviewed branch's `git status` is unchanged and `git worktree list`
shows only the branch itself before reporting. **A run without both
confirmations has not finished**, whatever it found.

## Bounding the run

**Two call sites, bounded differently.**

- **As Phase 4 of `code-reviewer`'s full pass** — the tests the change set adds
  or edits, which the diff already bounds. Not the suite. A mutation is one
  edit and one targeted run: cheap per test, expensive across a suite.
- **Invoked directly on a path** — everything under it. This is the audit
  operation. Say up front roughly what it will cost, so the caller can narrow
  the path instead of abandoning the run halfway.

**Run before Phase 5, never after.** Phase 5 writes a new failing test to pin a
confirmed defect, so it leaves the suite deliberately red; a mutation pass
running afterwards cannot tell its own red from Phase 5's. It runs after
Phase 3 for the matching reason — that phase's suite run is the green baseline
a mutation is measured against.

**Never repair a test you found.** A skill that fixes what it found cannot then
produce a separate, auditable finding.

**Skip nothing silently.** A test excluded for cost, for a marker, or because
the mutation could not be constructed goes in the scope statement.

### What a mutation costs here

- **The suite reaches no network.** The population is synthetic and generated
  in-process, so there is nothing to stub and nothing to wait on.
- **Exclude the `notebooks` and `app` markers.** The first executes every
  notebook headless end to end; the second drives the Shiny app through a real
  browser. Both are out of the default run for that reason, and a mutation
  pass reaching either pays a full notebook execution or a full browser run
  per test. A browser test is also the wrong shape for mutation: it asserts on
  wiring rather than on a computed value, so most mutations to the package
  leave it green by design.
- **A mutation to Rust costs a rebuild, and the numbers only mean something
  in release mode.** Editing anything under `src/` means
  `uv run maturin develop --release` before the test can run — tens of seconds per
  mutation, against about a second for a pure-Python one. Say so in the scope
  statement, and mutate the Python side where the same logic exists on both
  (it exists on both by design — see `simplify-audit`'s scope section).
- **A mutation inside the statistical parity test may not be detectable, and
  that is itself the finding.** `test_reference_parity.py` compares Rust against
  the Python reference within Monte Carlo error, so a mutation smaller than that
  tolerance passes. Where that happens, report it: the deterministic parity
  test — hazard forced to 0 or 1 — is the one that pins policy and budget
  logic exactly, and a mutation it cannot catch either means the tolerance is
  too loose or the deterministic case is missing.

## What to mutate

For each test in scope, find the code it names and break it so the result
*must* change: return early, drop a transformation, invert a condition, remove
an element from what a walk returns. Green after that is the finding. There is
no taxonomy to match against — the mutation answers the question directly, and
what class the defect belongs to is a name for it afterwards rather than a step
in finding it.

Use mutations that target ways a test can pass without the code being right:

- **Break the value the assertion reads, not a value the test computed.** A
  test whose two sides both come from its own setup cannot fail; mutating repo
  code is what exposes it.
- **Change one column of a parametrization or fixture.** Cases whose input and
  expected output hold the same value pass whatever the code does.
- **Remove one item from what a walk or query returns.** A test asserting only
  that the result is non-empty survives every narrowing but the last.
- **Flip the branch a fixture is meant to reach.** A fixture that cannot
  construct the state the test names leaves the assertion covering the other
  path.

**The subject is often not in the test.** Anything reached through a helper, a
fixture or `subprocess.run` is where the mutation has to land — the test body
only names it. This is also why reading cannot substitute: syntax-level review
of one function stops at the call.

**One result is not a finding.** A test whose walk is proven by a companion
test elsewhere fails that companion instead — the pair is sound, and the
mutation proves it.

**One is a finding with no mutation at all.** A collected function holding no
assertion cannot fail whatever you change, so it needs no mutation to convict
it — report it on sight.

## When you find one, look for its other spellings

**A defect can appear in several equivalent spellings.** Fixing only the
spelling first noticed leaves related cases untested. A clean report gets
written as `assert not bad`, `assert bad == []`, `assert len(bad) == 0` and
`assert bad == set()`; a walk gets hoisted to a name on one line or written
inline on the next.

So when a finding is confirmed, ask the second question: **what else would a
person plausibly have written here?** Then cover the synonyms, or say in the
finding which spelling it covers.

## Output

Group by what was established, because the report depends on that distinction:

**Checked by mutation.** Per test: the mutation applied, and whether it failed.
The ones that passed are the findings — each with the mutation, so the reader
can repeat it in one command.

**Read only.** Every test in scope that was not mutated, and why: cost, a
marker, a mutation that could not be constructed. This distinguishes tests
proved by mutation from tests only reviewed by reading.

**Not run.** If the canary did not fail, the whole run is invalid: say that
and report no findings at all. Do not attach findings to an invalid run,
because the report no longer shows which findings came from verified
mutations.

Close with both tear-down confirmations: the reviewed branch's `git status`,
unchanged, and `git worktree list` showing only the branch itself.
