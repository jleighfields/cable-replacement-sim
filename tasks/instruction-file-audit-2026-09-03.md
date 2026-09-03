# Instruction-file audit — 2026-09-03

Two review passes over `CLAUDE.md`, the five skills and the two agents: one for
factual correctness (every command, grep pattern and path run or resolved), one
for minimalism and content placement. This file records what was found and what
was done about it, so the decisions are recoverable without the review reports.

## Why

The instruction set had grown to ~2,600 lines against ~430 lines of source.
Three problems justified a pass rather than incremental fixes:

1. Several statements were **false about this repo**, and a reviewer acting on
   them would suppress real findings.
2. Rules were written out in full in up to five files each, so changing one
   meant finding every copy.
3. Contributor-facing documentation was reachable only inside the
   agent-instruction file, and `.claude/` had no map at all.

## Done

### Corrections — statements that were false

- [x] The marimo per-file-ignores do not exist. `pyproject.toml` has one entry,
      `S101` under `tests/**`. Three files described a seven-rule ignore list
      for `notebooks/**/*.py` and told reviewers not to flag those rules there.
      All three now point at `pyproject.toml` as the authority.
- [x] `UP045` was listed as already-settled-by-ruff. `select` carries no `UP`
      rules, so `Optional[X]` reaches the reader unreported — the opposite of
      what the section said. Moved to the manual-reading list.
- [x] `maturin` resolves only through the environment. Five instruction sites
      and the pull-request template said `maturin develop --release`; all now
      say `uv run maturin develop --release`, matching the workflows.
- [x] The secret-scanning fallback patterns were unusable as printed: markdown
      table cells escape `|` as `\|`, and a pattern pasted with the escapes
      intact matches a literal pipe and finds nothing. Moved to a fenced block,
      unescaped, and each verified against a planted fixture.
- [x] The worktree-teardown failure mode did not reproduce on Linux with git
      2.43 — `remove` from inside the worktree succeeds. Rewritten as a
      platform-dependent outcome rather than a certainty.

### Placement

- [x] New `.claude/README.md`: what each skill and agent is, each agent's
      default target, and which file owns which kind of rule. Nothing in the
      repo listed them before.
- [x] New `tests/README.md`: layout and import style, fixtures, the three
      validation layers, the two opt-in marker groups.
- [x] New `README.md` §Contributing: branch protection and the pull-request
      flow, previously reachable only inside `CLAUDE.md`.

### Deduplication

- [x] Target-set derivation: five copies to one, owned by
      `code-quality-review` §Arguments.
- [x] "Who runs the test suite": four copies to one, owned by `code-reviewer`.
- [x] Function-length threshold: two numbers (50 and ~80) for one concept, in
      two files. `simplify-audit` now points at the one in
      `code-quality-review`.
- [x] The Python/Rust mirror argument, the PyO3 practicalities and the config
      schema: `CLAUDE.md` now points at `PLAN.md` §4, §11 and §3 rather than
      restating them.

### Drift

- [x] Phase-numbered sentences restated as conditions that go false on their
      own — the required-check ruleset, `benches/`, the deployment manifest.
- [x] The mirror table named six files, none of which exists yet, and the same
      names were hard-coded in five more places. Replaced with the pairing by
      role plus a pointer to `PLAN.md` §4, Repo layout, which is maintained
      per phase.

## Deliberately not done

- **The secret-logging check stays**, though nothing in this repo currently
  holds a secret to log. Removing a security check because today's code has no
  subject is how it is missing on the day one appears.
- **`cargo fmt --check` and `cargo clippy` were not run.** This machine's
  rustup toolchain ships neither component; CI installs them. Every other
  command named in these files was run.

## Review

The instruction set went from ~2,630 lines to ~2,480, with 115 lines of new
README carrying what moved out. `CLAUDE.md` went from 406 to 355 — the line
count that matters most, since it loads on every session.

The audit's own lesson: these files had never been read against the repo since
they were written. Two of the four false claims (the marimo ignores, `UP045`)
would have been caught by running the tool once. The rule that prevents a
repeat is already in `CLAUDE.md` — *Verify by running, not by reading* — and it
applies to the instruction files themselves, not only to the code they govern.
