# Skills and agents

Committed Claude Code configuration. Names and descriptions from these files
are injected at session start, so a newly added agent or skill is invisible
until the session restarts.

## Skills — `skills/`

Each is a slash command, invocable directly, and each defines one phase of a
review. All five are report-only except `comment-docstring`.

| Skill | What it asks | Edits? |
|---|---|---|
| `code-quality-review` | Is this correct, clear, documented, and in its minimal form? | no |
| `security-scan` | Does this leak a credential or open a hole? | no |
| `comment-docstring` | Are the docstrings, doc comments, type hints and nearby READMEs right? | **yes** |
| `test-review` | Can these tests fail when the code they name is wrong? | no (mutates a throwaway worktree) |
| `simplify-audit` | Should this code exist, and is it minimal? | no |

## Agents — `agents/`

| Agent | Runs | Default target |
|---|---|---|
| `code-reviewer` | the four review skills as an ordered pass, plus a phase that pins confirmed defects with a failing test | `commit` mode: changed files (`git diff --name-only HEAD` unioned with the untracked ones). Full pass: the branch diff against `origin/main` |
| `simplify-auditor` | `simplify-audit` in its own context | the whole repo |

Both exist to keep repo-wide grep and read output out of the main session; an
agent returns its finished report, not its search transcript.

## Which file owns which rule

Editing these files means choosing where a rule goes, and the split is what
keeps them from duplicating each other:

- **What to look for, and what counts as a finding** belongs to the **skill**
  running that phase — where it also applies when someone invokes the skill
  directly.
- **How a pass is ordered, what it composes, and what it escalates** belongs
  to the **agent**.
- **What every session needs before starting work** belongs to `CLAUDE.md`,
  which loads on every session and pays for its length each time.
- **What a contributor needs in order to run or extend something** belongs in
  the README next to it — this file, `tests/README.md`, or the root
  `README.md`.

Agents read their skill's `SKILL.md` at runtime rather than copying its
checklist, so a skill edit takes effect without touching the agent.
