---
name: security-scan
description: Scan for leaked secrets and insecure patterns — hardcoded passwords/tokens/keys, tracked .env or credential files, secrets in logs, and unsafe defaults. Drives ruff S + detect-secrets with a grep fallback; report-only, makes no edits.
disable-model-invocation: false
allowed-tools: Read, Glob, Grep, Bash
argument-hint: "[file-or-directory]"
---
# Security Scan

Find secrets and unsafe patterns before they get committed, and report a
findings list. **Report-only** — never edit or delete a secret yourself.
Removing a secret and (critically) **rotating** it is a deliberate,
human-driven step.

Run tooling first, the same way `code-quality-review` runs ruff, then cover
the remaining cases with `git` checks and `grep`:

1. **`ruff` `S` rules (flake8-bandit)** — mechanical insecure-pattern
   detection (hardcoded passwords, `eval`/`exec`, `shell=True`, unsafe
   deserialization). Configured in `pyproject.toml`.
2. **`detect-secrets`** — entropy + regex secret detection with a committed
   `.secrets.baseline` so known false positives stay suppressed.
3. **`git ls-files` + `grep`** — tracked-credential-file checks and a regex
   fallback for token shapes the tools miss.

Security scan findings answer "does this leak a credential or open a hole?"
`code-quality-review` answers "is it clear?" and `simplify-audit` answers
"should it exist?"

## Arguments

- **file-or-directory** (optional): Path to scan for secret *content* /
  insecure patterns. If omitted, scan the changed files
  (`git diff --name-only HEAD` unioned with `git ls-files --others
  --exclude-standard` — union, not fallback).
  The tracked-file checks always run against the whole repo regardless of
  the argument.

## Scope

- **Content / pattern scan:** the target files (changed files by default,
  or the given path).
- **Tracked-file checks:** always whole-repo via `git ls-files` — a
  committed `.env` is a repo-wide fact, not a diff fact.
- **Never scan gitignored or untracked artifacts** for content. Confirm
  with `git ls-files` rather than guessing from directory names — but DO
  still confirm those paths are actually ignored.

## What to check

### Tracked credential files (whole repo, always)

Run `git ls-files` and flag any tracked file that should never be committed:

- `.env`, `.env.local`, `.env.*` **except** `.env.example` / `.env.template`
  / `.env.sample` (those are intended templates — see the caveat below).
- Private keys / certs: `*.pem`, `*.key`, `*.pfx`, `*.p12`, `*.keytab`,
  `id_rsa`, `id_dsa`.
- Credential dumps: `credentials.json`, `service-account*.json`,
  `*.kdbx`, `.netrc`, `.pgpass`, `.htpasswd`.

A tracked secret file is **Must Fix**: `git rm --cached` it, add it to
`.gitignore`, and **rotate** anything it exposed (it is already in history).

### `.gitignore` coverage

Confirm `.env` and the patterns above are gitignored, not merely absent. A
secret that is untracked today but not ignored can be committed by
`git add -A`.

### Hardcoded secrets — detect-secrets + grep fallback

Primary: run `detect-secrets` against the baseline (see Steps). It catches
token shapes AND high-entropy strings the regexes below would miss, and the
baseline suppresses audited false positives.

Fallback / supplement (also useful for explaining a finding): scan for
assignments of a secret-looking name to a literal, and known token shapes.

**These are in a fenced block, not a table, and that is not cosmetic.** A
markdown table cell has to escape `|` as `\|`, and `\|` in a regex is a
*literal pipe* rather than alternation — so a pattern copied out of a table
matches the pipe character and nothing else, reports clean, and has checked
nothing. That is the silent no-op this whole phase exists to catch, and it is
invisible precisely because a clean result is what you expect. Copy from here,
where the patterns are unescaped and run as written.

```
# secret-named literal assigned to a string
(?i)(pass(word|wd)?|secret|token|api[_-]?key|client[_-]?secret|access[_-]?key|auth[_-]?token|private[_-]?key)\s*[:=]\s*["'][^"']{6,}["']

# private key block
-----BEGIN (RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----

# AWS access key id
AKIA[0-9A-Z]{16}

# GitHub token
gh[pousr]_[A-Za-z0-9]{36,}
github_pat_[A-Za-z0-9_]{60,}

# Slack token
xox[baprs]-[A-Za-z0-9-]{10,}

# bearer token and JWT
(?i)bearer\s+[A-Za-z0-9._\-]{20,}
eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}

# URL with embedded credentials
[a-z][a-z0-9+.\-]*://[^/\s:@]+:[^/\s:@]+@

# connection-string password
(?i)(password|pwd)=[^;"'\s]{4,}
```

**Watch one of these match before trusting a clean run.** Plant a fixture — a
line like `api_key = "sk-live-0123456789abcdef"` in a scratch file — and
confirm the scan reports it. A regex that cannot match is indistinguishable
from a repository with no secrets in it, and this one already shipped broken
once.

For each hit, **redact the value in your report** — show the variable name
and first few characters only, never the full secret.

### Secrets passed to logging or print

ruff `S` does not cover this. Flag log and print statements that
interpolate a secret-bearing value: `Authorization` headers, anything named
`*secret*` / `*token*` / `*password*`, or a whole `headers` / `auth` dict.
Even at DEBUG a logged token ends up in the deployment's log store, which
is usually retained longer and read by more people than the code.

### Insecure patterns — ruff `S` rules

`ruff check --select S` reports these patterns; cite the rule code in each
finding (as `code-quality-review` cites ruff codes):

- **Hardcoded password** — `S105` (string), `S106` (func arg), `S107`
  (default arg).
- **`eval` / `exec`** — `S307` (eval), `S102` (exec).
- **Unsafe deserialization** — `S301` (`pickle`), `S506` (`yaml.load`
  without `SafeLoader`).
- **Shell injection surface** — `S602`/`S604`/`S605` (`shell=True`),
  `S607` (partial executable path).
- **TLS verification disabled** — `S501` (`verify=False`).
- **Other** — `S104` (bind all interfaces), `S324` (weak hash).

Treat any `S`-rule hit on the target files as **at least Should Fix**;
`S105-S107` (hardcoded secret) is **Must Fix**. Do not re-flag the
configured ignores below.

## Configured exceptions (don't re-flag these)

`pyproject.toml` encodes the legitimate exceptions and `.secrets.baseline`
records audited non-secrets — respect both. **Read the actual per-file
ignores in `pyproject.toml` rather than assuming**; they are the authority,
and this section describes only the fleet default.

- **`tests/**` ignores `S101` only.** pytest tests use `assert`, so
  `S101` is expected there. **The hardcoded-secret rules `S105-S107` stay
  ACTIVE in tests**, because a real credential pasted into a fixture is
  exactly as leaked as one in production code. Never suppress an
  `S105-S107` hit under `tests/` as a configured exception — it is a
  **Must Fix**.

## The `.env.example` caveat

- **`.env.example` is supposed to be committed.** Do NOT flag it as a
  tracked secret file. Instead, **open it and confirm every value is a
  placeholder** — a real value leaked into the template IS a finding, and a
  Must Fix.

## Repo-specific caveats

- **There is no `.env` here and none is expected.** Nothing in this project
  authenticates to anything: the cable population is generated in-process
  from a seed, and no input is read from a remote service. `.gitignore`
  covers the pattern defensively, so a tracked `.env` appearing is a real
  finding rather than the usual false positive, and there is no
  `.env.example` to audit.
- **RNG seeds in `configs/*.yaml` are not credentials.** `simulation.seed`
  and anything derived from it exist so a run reproduces; they are meant to
  be committed and read. The same goes for every other value in that file —
  it holds simulation tunables, and a high-entropy-looking number there is a
  parameter.
- **No input data is real.** The population is fully synthetic and
  generated by this repo, so a fixture
  that looks like utility asset data carries nothing to leak. A file that
  appears to hold *observed* failure records is the exception worth flagging —
  it would mean real data entered a repo that is documented as having none.
- **`ruff` does not read Rust, so the `S` pass covers half this repo.** Add a
  `cargo clippy` run over `src/` and read every `unsafe` block by hand: PyO3
  bindings are the place `unsafe` legitimately appears, and one outside a
  binding needs a comment saying what invariant it upholds.

Each caveat has to be anchored to something durable: a file that exists, a
documented convention. A caveat naming a deleted module authorizes an
exception that no longer applies.

## Output format

Group findings by severity, same buckets as `code-quality-review` so the
`code-reviewer` agent's report stays uniform. **Number findings
sequentially**, continuing from the review findings rather than restarting
at 1, so a closing list of open items is unambiguous about which section
each belongs to.

### Must Fix
- Confirmed live secret in code, a tracked `.env`/key file, a real value
  in `.env.example`, or an `S105-S107` hit. **Always include the
  remediation:** remove it, move it to an environment variable / `.env`,
  and **rotate the exposed credential** — state explicitly that
  working-tree removal is not enough if it was ever committed, since it
  persists in git history; scrub with `git filter-repo` / BFG if needed.

### Should Fix
- Other insecure-pattern `S`-rule hits (`verify=False`, `eval`/`exec`,
  unsafe deserialization, `shell=True`) and secret logging.

### Consider
- Possible-but-uncertain matches (entropy hits not in the baseline that
  might be fixtures), `.gitignore` gaps with nothing currently leaked.

For each finding: a sequential number, `file:line`, the rule code where
applicable, what matched (**redacted**), why it is a risk, and the
suggested fix.

## Steps

1. **Pick targets.** Path argument → that path. Otherwise
   `git diff --name-only HEAD` unioned with `git ls-files --others
   --exclude-standard`. Union, not fallback — see `code-quality-review`.
   The tracked-file and `.gitignore` checks run whole-repo regardless.
2. **Mechanical passes first** — reuse the configured tooling:
   - `uv run ruff check --select S <targets>` — insecure patterns. Cite
     each rule code.
   - `uv run detect-secrets-hook --baseline .secrets.baseline <targets>` —
     entropy/regex secret detection. The `-hook` entry point **reads** the
     baseline to suppress audited false positives, then exits non-zero and
     prints each new secret's type and location. Crucially it **does not
     rewrite `.secrets.baseline`** — unlike `detect-secrets scan
     --baseline .secrets.baseline`, which rewrites the file in place AND
     drops the recorded entries for every file outside `<targets>`, wiping
     audited false positives for everything unscanned. Treat every
     reported hit as a candidate and open it to confirm a real secret
     versus a placeholder or fixture.
   - **Check the baseline is there and intact before trusting a clean
     run.** It must exist, be tracked, and still carry its detector list.
     If it is missing or truncated, treat the scan as invalid even when the
     hook exits cleanly; a clean exit then does not prove the detector list
     or audited false positives were used.
   - **This skill is report-only: never modify `.secrets.baseline`.**
     Recording a new audited false positive (via `scan` plus an
     interactive `detect-secrets audit`) is a deliberate, separate human
     step — and **never** commit a real secret into the baseline to
     silence it. If the baseline shows as modified after a scan, restore
     it with `git checkout -- .secrets.baseline`.
   - **When you report a false positive, recommend the inline fix, not the
     baseline.** Marking the line `# pragma: allowlist secret` with the
     reason beside it clears the hit and keeps the reason beside the code.
     A baseline entry stores a hash without the reason it was allowed. So
     the recorded findings should be zero; the baseline should carry the
     detector list and the exclude pattern, not accumulated exceptions.
     Recommend excluding a whole file only where its values change on every
     run, since those are stale in a baseline by the next commit.
   - **Regenerating it deliberately keeps the exclude pattern**, which is
     not carried forward from the existing file:

     ```bash
     uv run detect-secrets scan --exclude-files 'manifest\.json$' > .secrets.baseline
     ```

     Dropping the flag brings back findings that change on every deploy,
     and a value that changes every run is stale in the baseline by the
     next commit — exclude the file instead. A Connect `manifest.json`,
     whose per-file md5s all read as high-entropy hex, is the case the
     pattern exists for — this repo grows one at Phase 7, when the Shiny app
     deploys, and not before. Commit the result: the pattern is stored *in* the
     baseline as a filter, so every later caller inherits it without
     repeating the flag.
3. **Tracked-file check.** `git ls-files`, filtered for the credential-file
   patterns above, applying the `.env.example` exception. Confirm
   `.gitignore` coverage.
4. **Fallback content scan.** Run the grep patterns over the target files
   for anything the tools did not surface, then check for secret logging.
   Open each hit to confirm it is a real secret versus a placeholder or
   fixture; redact before reporting.
5. **Emit the report** grouped by severity. Make **no edits** — this skill
   is report-only. If anything is Must Fix, lead with it. If there are no
   findings, say "No security findings" explicitly so the clean result is
   on record.

## Note on marimo notebooks

Scan notebook `.py` files for secrets the same as any source file — a
hardcoded token in a marimo cell is just as committed. `S101` is the only
`S` rule ignored for notebooks, so every other one — including the
hardcoded-secret rules `S105-S107` — still fires there. Read
`[tool.ruff.lint.per-file-ignores]` in `pyproject.toml` rather than trusting
a list here; that file is the authority.
