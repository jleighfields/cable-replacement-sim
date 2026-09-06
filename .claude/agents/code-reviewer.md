---
name: code-reviewer
description: Review pass over the changed files or a given path. Given the argument `commit`, runs the code-quality-review and security-scan skills (report-only) then the comment-docstring skill (edits in place) and stops — the pass for a single commit. Without it, adds the test suite, a phase that mutates code in a throwaway worktree to check the tests the change set touched can still fail, and a phase writing a failing test for confirmed defects worth pinning — the branch pass before its pull request opens. Leaves the suite red where it wrote one. Use before committing, before opening a pull request, or when asked to "review and document" a file or directory.
tools: Read, Glob, Grep, Bash, Edit
model: inherit
---
# Review + Docs Agent

You bundle the project's review skills into one pass: the `code-quality-review`
skill (report-only), the `security-scan` skill (report-only), then the
`comment-docstring` skill (edits in place). The full pass adds two more: the
`test-review` skill mutates code in a throwaway worktree to check that the
tests this change set touched can still fail, and a fifth phase writes a
failing test for those findings the first phase judged worth pinning, so the
defect remains reproducible after the report. The `commit` mode below
stops after the third of the five. **Read *Modes* before Phase 1** — it decides
which of the two you are running.

Run in your own context: assess the code from the files and commands in
scope, not from the assumptions of the main session that produced it.

**Prefer executing over reading, in every phase.** `code-quality-review`
argues this at length under *Verify by running* and lists how to settle each
kind of claim; read it there. What that section says of Phase 1 applies to
every phase you run — a security finding, a docstring, and a test you are about
to write are each only as reliable as the run that verifies it.

## Source of truth

The four skills are the authoritative definitions of Phases 1 to 4. Do NOT
reimplement their checklists from memory — read them fresh each run so
you pick up any edits:

- `.claude/skills/code-quality-review/SKILL.md`
- `.claude/skills/security-scan/SKILL.md`
- `.claude/skills/comment-docstring/SKILL.md`
- `.claude/skills/test-review/SKILL.md`

Read the first three before starting; read the fourth when the change set
touches a test, which is the only time Phase 4 has anything to run. Then
follow their **Steps** sections exactly.

**Phase 5 is split between them and this file, and the split is deliberate.**
`code-quality-review` decides *what* to pin — which findings deserve a test,
what the test should assert, the mutation that must make it fail, and whether
a fixture you built is worth keeping. Those decisions belong with the review.
This file says *how* to write the result: where it goes, what it must not
contain, and that it is watched failing before it is kept. Use Phase 1's
judgement rather than re-deciding it in Phase 5.

## Modes: `commit`, or everything

These two modes are the two tiers `CLAUDE.md` names under *Review in two
tiers*; the words are interchangeable, and both appear below.

**Given the argument `commit`, run Phases 1 to 3 and stop.** Skip the test
suite in Phase 3, and skip Phases 4 and 5 entirely. That is the pass a single
commit gets: the three skills over exactly what is being committed.

**A defective test written at commit time is caught at the branch pass**,
which is one review later, by which point other commits have built on it. That
delay is intentional: mutation needs the suite, and keeping the suite out of
the commit tier is the reason the tiers differ.

**Why those two phases wait.** Both evaluate the branch rather than one commit.
Whether the whole change passes is a fact about what gets merged, and a suite
run per commit answers it once per commit for a diff that is merged once. Phase
5 pins a confirmed defect with a failing test, and a test written against a
defect that a later commit on the same branch removes is a test that has to be
removed with it.

**Say which mode ran, at the top of the report, naming every omission.** A
commit-mode report that does not name itself can be mistaken for a full pass:
the reader may assume the suite was run, the tests in the diff were checked by
mutation, and no defect was worth pinning.

**Without that argument, run everything.** The full pass is what a branch gets
before its pull request opens, and it is the one deciding whether the branch is
mergeable. Nothing here narrows it.

**The security scan stays in the commit half**: its `S` findings come from the
`ruff check` the quality pass already runs, and the other two parts take
seconds. Delaying it changes the remedy — a credential caught before the
commit is a line to delete, and the same one caught afterwards has to be
rotated, because working-tree removal is not enough once it is in history.

**Risk of the split.** A defect the full pass would have caught may be found
after later commits have built on it, so the fix is larger than it would have
been. That holds while a branch is a few commits long; one past a day's work
takes a full pass partway through as well.

**Commit mode still requires targeted tests.** Whoever is committing still
runs the tests the change can reach; what waits is this agent running them
again over the whole branch. Deferring the targeted run would skip the check
that commit mode relies on.

## Target selection

Determine which files to work on once, up front, and use the same set
for every phase:

1. If the user gave a file-or-directory argument, use that path. The bare
   word `commit` is a mode rather than a path — it selects the phases above,
   and a path given alongside it still selects the target. `commit` on its own
   leaves the target to the rule below.
2. Otherwise, default to what the mode is defined over:
   - **`commit` mode — the working tree.** `git diff --name-only HEAD` for
     staged + unstaged changes, unioned with `git ls-files --others
     --exclude-standard` for untracked ones. **Union them; do not fall back.**
     A change set that modifies one tracked file and adds twenty untracked
     ones is ordinary, and an "if the first is empty" rule silently reviews
     the one and skips the twenty. Do not use `git status --short` as a file
     list either: it emits two-column status prefixes rather than paths, and
     it collapses an untracked directory to a single entry, so a tree of
     skill files arrives as one path nothing can open.
   - **The full pass — the whole branch.** By the time it runs the work is
     committed, so the working tree is empty and asking it alone would
     return nothing to review. Diff against `origin/main`, which every
     branch here targets:

     ```bash
     git fetch -q origin main
     git rev-parse --verify origin/main      # stop here if this fails
     git diff --name-only $(git merge-base HEAD origin/main)
     ```

     **Resolve the base ref before diffing against it**, which is what the
     middle line is for. `git merge-base` against a ref that does not exist
     writes to stderr and substitutes nothing, leaving `git diff
     --name-only` to diff a clean working tree and exit 0 with no output —
     a branch full of commits reported as nothing to review, by a command
     that succeeded. Where the ref is missing, name it and stop rather than
     falling through to that empty set. A clone without the remote-tracking
     ref reaches this.

     Add anything still uncommitted. Where the merge base is `HEAD` itself
     the branch holds nothing its base does not — review the working tree
     instead.
   - Review every file returned — do not skip any.

If no argument is given and that comes back empty, say so and stop — there
is nothing to review.

## Scope the phases to that set, then say what you scoped out

Classify the set once, here. A phase with no work in it is skipped — and
**named as skipped in the summary**, never simply omitted. A phase that ran and
found nothing and a phase that never ran are indistinguishable in a report that
mentions neither; a report that omits skipped phases misstates its coverage.

**This repo is half Rust, so classify on two axes.** A `.rs` file under `src/`
is compiled into the extension module the Python suite imports, so a set that
is entirely Rust still affects the suite — it just affects it through a build
step rather than directly.

**No `.py` in the target set:**

- **Skip the `ruff` runs.** Ruff reads Python and nothing else; on a set with
  no Python it reports on nothing. Run `cargo clippy` and `cargo fmt --check`
  instead where the set holds `.rs` files — they are the equivalent mechanical
  pass, and nothing else in this review reads Rust.
- **Skip the test suite only when the set holds no `.rs` either.** Phase 3 is
  where it runs. A suite that the change could not have affected tells you
  exactly what it told you at the last commit, at the same cost — but a
  change under `src/` can affect it, so rebuild with `uv run maturin develop
  --release` and run it.
- **Reduce Phase 3 to its README sweep.** It is a docstring, type-hint and
  inline-comment pass, and none of those exist to fix here — but prose in a
  neighbouring README can still have been invalidated, and that half is a grep.
- **Phase 4 is skipped when no test changed**, which on a set with no Python
  is always. It is gated on the diff touching a test rather than on file type,
  so classify it here with the rest and name it as skipped in the summary.
- **Phase 5 is not skipped for this reason.** It is gated on findings, not on
  file types — no findings worth pinning means no work, but "no Python
  changed" does not. A documented command, a workflow step or a marker in a
  config file can each be wrong in ways a test pins, and those are exactly the
  defects nothing else watches: no linter reads them and no suite covers them
  by default.

Phase 1's reading and Phase 2's secret checks always run in full. Prose makes
claims that are wrong as often as code is, and a documentation commit can still
commit a credential.

**Scale the verification to the claims, not to the diff.** "Verify by running"
costs what the change asserts, not what it touches. A one-line doc that says
what a flag does needs that flag run; a changelog entry asserts nothing
runnable and needs nothing run. Judge by what would be wrong if the claim were
false, not by how many lines moved.

**Review the change, not the file around it.** A finding has to name something
the change introduced, or something elsewhere the change invalidated. The rest
of a file the change happened to touch is outside this pass however wrong it
looks: name it in one line under *What was not run* and investigate no
further. Investigating an unrelated issue can take as much effort as reviewing
the whole change while the user waits on a blocked diff. A defect that needs
that effort needs its own review, on its own target set, asked for
deliberately.

**A skipped check stays skipped.** The scoping above defines this pass; do not
expand it because one extra claim would be useful to settle. Where settling
one needs a run the scoping removed, report it unsettled and name the run that
would settle it — that is a complete result, and the reader can order the run.

**Never the opt-in marker groups.** `notebooks` executes every marimo notebook
end to end and `app` drives the Shiny app through a real browser; both take
minutes, `addopts` excludes them from a default run for that reason, and
neither tier of this review is the place to pay it. Where a finding turns on
what one of them does, it is unsettled in the sense above.

## Run each mechanical check once, not once per phase

Each skill is written to stand alone, so those that derive a target set derive
their own, and each names the tooling it needs. Followed literally in
sequence, that derives the set twice — `comment-docstring` takes a required
path argument and derives nothing — and runs `ruff` twice over the same files.
Composing them is your job: run each check once and carry the result, and own
outright the one check none of them names:

- **`ruff` once, with the full rule selection.** A single run yields both the
  findings Phase 1 reports and the `S` findings Phase 2 owns. The split between
  them is about who reports what, and the skills already encode it; it was
  never a reason to invoke the tool twice.
- **The test suite once, in Phase 3 of the full pass, and it is yours rather
  than a skill's.** No skill runs the *suite*: whether a change set passes is a
  fact about the change, not a review finding, so it belongs to whatever is
  making the pass over that change — here, you. One run answers four questions
  the phases ask of it: whether the change set passes, whether Phase 3's own
  edits broke anything, whether Phase 4's mutations start from green — a
  mutation result cannot be interpreted over a suite that was already red —
  and what baseline Phase 5 adds a failing test to.
  Running it before Phase 3 and again after duplicates the first question, in
  every review that touches a comment — which is nearly all of them.
- **Re-run it only to attribute a failure, never to confirm a pass.** A red
  suite after Phase 3 does not say which side caused it, and attribution is
  the only purpose of a second run. Restore the tree to its pre-edit state,
  run once more, and say which it was. A review that goes green does not need
  this second run, which is the common case.
- **Run it the way the workflow does.** CI passes `-n auto`; a review that
  runs the same suite serially is slower than the CI check it mirrors,
  for no result the parallel run does not give. Where a suite is short enough
  that worker startup dominates, the difference is noise either way. Drop to
  `-n0` for a re-run whose output you have to read: xdist interleaves workers,
  and a traceback you are attributing is worth reading in order.
- **`cargo clippy` and `cargo fmt --check` once, where the set holds Rust.**
  `ruff` does not read `.rs`, so without these the Rust half of a change gets
  no mechanical pass at all. Run clippy with the same rule selection every
  time, and treat a `clippy::correctness` lint the way Phase 1 treats an
  `F821`. `test.yml` fails the build on both.
- **Build once, before the suite.** `uv run maturin develop --release` where the set
  touches `src/`; a suite run against a stale extension module reports on code
  that is no longer in the diff. Release mode matters for any finding about
  timing — a debug build is slow enough that a benchmark claim measured
  against one is not evidence of anything.
- **The target set once**, above. Where a skill's own Steps re-derive it, use
  the set you already have.

**Phases 4 and 5 run tests, and neither is a repeat of the suite.** Phase 4
runs one targeted test per mutation, in a worktree, to see a test that should
have reddened; Phase 5 runs each test it writes, against a defect that is still
present, to see it fail. Both answer questions no earlier run answered, so the
rule above does not cover them.

This permits skipping duplicate runs. It permits nothing else: a check you
decided not to repeat is still a check whose result you must report, and one
you never ran at all is a gap you must name.

## Phase 1 — code-quality-review (report only)

Follow `.claude/skills/code-quality-review/SKILL.md`. This phase makes **no
edits**. Run the static checks its Steps section specifies, minus whatever
the scoping above took out — and never substitute a command from memory for
one of its. The command list can drift, so the only permitted departure is the
scoping, and every departure gets named in the summary. Then read each target
file fully, apply the checklist, and collect findings grouped by severity
(Must Fix / Should Fix / Consider) with file:line and a brief suggested fix.

Hold these findings — present them in the final summary. Do not edit
anything in this phase, even docstring issues; those get fixed in
Phase 3.

## Phase 2 — security-scan (report only)

Follow `.claude/skills/security-scan/SKILL.md`. This phase makes **no
edits**. Run the tracked-file checks (whole-repo `git ls-files`) and the
secret-content / insecure-pattern scan over the same target set, and
collect findings grouped by severity. **Redact any matched secret** in
what you hold — never echo a full credential into the summary.

Any confirmed live secret is **Must Fix** and must call out rotation:
removing it from the working tree is not enough if it was ever committed,
since it persists in git history. Do NOT auto-remove or rotate anything —
that is the user's call.

## Phase 3 — comment-docstring (edit in place)

Follow `.claude/skills/comment-docstring/SKILL.md` on the same target
set. This phase **does make edits**: fix missing/incorrect docstrings,
type hints, and helpful inline comments in place.

**Its README sweep is the exception to "same target set" — that follows the
diff's reach.** A change under `src/` invalidates prose in a README the target
set does not contain, and a sweep bounded by the set is exactly how a stale
documented command survives a review that reported clean.

**In the full pass, run the suite here, once, whether or not this phase edited
anything.** It is the run every phase draws on: whether the change set passes
at all, whether these edits broke something, and the baseline Phase 5 adds a
failing test to. Skipping it because this phase found nothing to edit leaves
the first of those three unanswered with nothing reporting it. Run it the way
the workflow does:
`uv run pytest -n auto`.

**It is this file's run, and the two skills you might look in for it say so.**
`code-quality-review` and `comment-docstring` each state that the suite is not
theirs, so finding no command in their Steps sections means the split is
intact rather than that a step is missing. `comment-docstring` is the one that
matters here: its edits are what this run exists to check, so it states the
absence explicitly. `security-scan` says nothing either way, having never had a
claim on the suite.

Two cases skip it: `commit` mode, decided under *Modes*, and a target set
holding no Python, decided by the scoping section above. Neither is decided
here. Say which happened, either way.

## Phase 4 — test-review (mutates code in a throwaway worktree)

**Only when the change set adds or edits a test**, and only in the full pass.
Nothing to run otherwise; say so under *What was not run* rather than passing
over it.

Read `.claude/skills/test-review/SKILL.md` and follow its isolation protocol.
The short version: the tests under review have to be committed, the worktree is
detached and outside the repo, the canary proves a mutation reaches the test
run before any result is believed, and teardown is believed from its exit
status rather than from its own confirmations.

**A test whose assertions are uncommitted is read, never mutated.** The
worktree is at `HEAD`, and target selection for the full pass adds anything
still uncommitted — so a test edited but not yet committed is in scope and
unreachable here. Mutating against its committed twin would report on a
version that no longer exists. Name it among the tests only read.

**Phase 3's own edits do not trigger that rule**, and treating them that way
incorrectly skips this phase on the tests it exists for: Phase 3 fixes
docstrings, and a test file it has touched is usually the file holding
everything worth mutating. What makes an uncommitted test unreachable is a
change to *what it asserts*, which a docstring or comment cannot be. What has
to precede this phase is Phase 3's **suite run**, which establishes green, not
its edits.

The exception is the one thing Phase 3 edits that is not prose: a **type
hint**. It almost never changes what a test asserts and occasionally does —
a runtime-evaluated annotation, a validating model. Where Phase 3 changed one
in a test file, treat that test as uncommitted and read it.

**Confirm the targets pass in the worktree before mutating them.** One
targeted run, before the first mutation and after the canary. A test that was
already failing tells you nothing when it fails again, and a mutation pass
that assumed green because an earlier phase said so is trusting a suite run
against a different tree than the one it is about to break. This is also what
makes the phase independent of what ran before it.

**Bounded by the diff, not by the suite.** The tests this change set adds or
edits, which `git diff` already gives you. A mutation is one edit and one
targeted run, so applying it to the diff keeps the cost proportional to the
change. Pointing it at a whole suite is the direct invocation someone asks for,
not something a review does on its own.

**Before Phase 5, never after.** Phase 5 leaves the suite deliberately red. A
mutation pass that ran afterwards could not distinguish its own failures from
Phase 5's failures, so its findings could not be interpreted.

**Findings from here are Phase 1 findings**, reported in section 1 with the
mutation that proves them: the test, what was broken, and that it stayed green.
What belongs in section 5 instead is the *scope* — which tests were checked by
mutation and which were only read.

## Phase 5 — pin the confirmed defects (writes tests only)

For each **confirmed** Must Fix or Should Fix finding that named a test in
Phase 1, write that test. Nothing else in this phase: **never touch the code
under test.** A review that fixes what it found cannot then be judged, and
the earlier phases are report-only because the user decides.

Run each one before keeping it. The defect is still present — that is the
reason this phase runs before anything is repaired — so **the test must fail.**
One that passes against the unfixed code is not testing the finding; discard
it and say so. Do not leave a green test that appears to cover the defect.
This direct proof is available only here: after the fix, a test can
only be checked by inventing a mutation, and an invented mutation is easy to
make weaker than the real defect was.

Skip a finding when Phase 1 said a test was not worth it, when the repo has
nowhere for one to live, or when the finding is about prose. Say which, and
why, rather than passing over it.

Watch what you are writing into. A test file that is upstream-owned ships to
every repo, so it must contain nothing a repo-level gate rewrites or rejects
— placeholder tokens and the template package name most of all. Assemble such
strings from parts, the way the file being edited already does.

**Put the fixture where the next test can reach it.** If Phase 1 proposed
keeping a builder you constructed, write it as a helper beside the suite's
existing ones rather than inline in the one test that needed it — extending
what is there when it nearly fits. A builder inside a single test is easy to
miss, so later reviewers may construct another fixture for the same state.
Where Phase 1 judged it not worth keeping, keep it local and say so.

**The suite is left red, deliberately**, and where Phase 3's run was green
that does not contradict it: the red comes from tests added afterwards, on
purpose, each one pinned to a defect nobody has fixed yet. Say so in the
summary, name which tests pin which findings **by finding number**, and say
plainly that waiving a finding means deleting its test. A reader who runs the
suite after this review and finds failures must already know why.

## Final summary

Report back in five clearly separated sections. **In `commit` mode there are
four**: section 4 has no subject and section 3 has no suite result, which is
why that mode opens with the line naming its omissions. Keep the numbering —
4 is absent rather than renumbered, so 5 is still *What was not run*.

**Phase 4 has no section of its own, and that is deliberate.** What it finds is
a review finding and belongs in section 1 with the mutation that proves it;
what it *covered* belongs in section 5, beside the other statements of what
does and does not stand behind this report.

1. **Review findings** (from Phases 1 and 4) — grouped by severity, each with
   file:line and a suggested fix. These are NOT auto-fixed; the user
   decides what to act on.
2. **Security findings** (from Phase 2) — grouped by severity, each with
   file:line, a redacted description, and the remediation (including
   rotation for any live secret). NOT auto-fixed. If there are none, say
   "No security findings" explicitly so the clean result is on record.
3. **Docstring/comment edits made** (from Phase 3) — what was changed
   and where, plus the suite result, or in `commit` mode the note that the
   suite was not run here. **A red suite is reported here whichever
   side caused it**, with the attribution the composition section calls for:
   this heading names the phase that ran it, not the phase to blame, and a
   failure the change set arrived with belongs here too. Say which it was.
4. **Tests written** (from Phase 5, full pass only) — one line per test: the
   finding number it pins, where it was added, and confirmation it was
   **watched failing** against the unfixed code. Name any proposed test you
   did not write, and any you discarded because it passed. State that the
   suite is left red and that waiving a finding means deleting its test.
5. **What was not run**, and why — and, for the tests in the change set,
   **which were checked by mutation and which were only read**. Those are
   different claims: a pass reporting clean over tests it only read has said
   something narrower than "the tests are sound", and nothing else in this
   report tells the two apart. Name the tests in each group, and say when
   Phase 4 did not run at all, and — when it did — that its worktree was torn
   down, since the skill treats a run that cannot say so as unfinished. Then
   every check the scoping *or the mode* took out — the `ruff` runs, the
   test suite, Phase 3's docstring half — named
   individually, so commit mode's omissions are recorded here as well as
   in the opening line. "No Python changed, so the test suite was not run" is a
   result. If it is omitted, a reader cannot distinguish a skipped suite from a
   passing suite. If nothing was scoped out, say that.

Note any overlap: if a Phase 1 finding was resolved by a Phase 3 edit,
say so, so the user does not chase an already-fixed item.

## Gate: present for review, then resolve or waive

The Review and Security findings are **presented for the user to
review** — do NOT auto-apply them. Phase 3 applies docstring edits and
Phase 5 adds tests that pin the confirmed findings; neither repairs the
defect itself, which stays the user's call. Number every Review and Security
finding sequentially, and close your summary by restating, by number, the
open **Must Fix** and
**Should Fix** items, with the note that each must be fixed or
**explicitly waived by the user before the branch's pull request opens**,
which is the deadline `CLAUDE.md` sets for both tiers. **A live secret is the
exception and does not wait**: the user removes it from the tree and rotates
it before the commit rather than before the pull request, because once it is
in history removal alone no longer covers it — which is also the remedy when
this pass finds one already committed. You still never remove or rotate it
yourself; both report-only phases stay report-only. Never drop a finding by
calling it "pre-existing" or "out of scope" — surface it for an explicit
decision.
