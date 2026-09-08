# Lessons

One rule per correction, written so it prevents the repeat rather than
describing the incident. Read this when starting related work.

## Never poll for a background process by a pattern that matches the poller

`until ! pgrep -f remeasure.py; do sleep 20; done` never terminates. The shell
running the loop has `remeasure.py` in *its own* command line, so `pgrep -f`
finds the waiter and the condition is true forever. Every such loop becomes a
permanent stale process the moment it starts, and killing them does not help
because the next one does the same.

**Do not poll.** Start the work with `run_in_background` and read its output
file when the completion notification arrives. That is what the notification is
for, and it costs nothing while waiting.

**The `[r]emeasure.py` bracket trick does not fix this, and recommending it here
was wrong.** It stops `pgrep` matching its own process, which is a different
problem. The waiter's command line still contains the plain literal — the launch
line that started the job has it — so `pgrep -f` matches the waiter whatever the
pattern is bracketed to. A task written that way ran for an hour after its work
finished, and the bracket was in it.

If a wait is genuinely unavoidable, wait on the process rather than on a
pattern:

```bash
uv run python remeasure.py > out.log 2>&1 &
wait $!
```

`wait` on a captured PID cannot match the wrong process, because it matches no
text at all. Note also that the harness kills a foreground command at ten
minutes, so a loop longer than that is doubly wrong.

## An edit script that validates late discards every edit before it

A script that applies several replacements and asserts on each as it goes will,
on the first failed assertion, exit before writing — so the edits that already
succeeded are lost with it. Nothing says so: the traceback names the assertion
that failed, not the two replacements that are now missing. Reporting the change
as made, on the strength of a later command that printed something reassuring,
is how it reaches a commit message.

It happened three times in one branch. Once as above, where an assertion on the
second replacement discarded the first, and a review three passes later found
the commit describing a change the diff did not contain. Once as a conditional
replacement — `if s.count(old) == 1:` — that silently matched nothing because
the target text was wrapped differently than expected, and the script reported
success anyway. And once as a `sed` whose pattern missed a line carrying a
trailing comment, which is the worst of the three: the edit was a *setup* step
for a verification run, so the run went ahead against unchanged input and
reported a pass that proved nothing.

**Verify the file on disk after writing it**, in the same command:

```python
p.write_text(s.replace(old, new))
assert "the new text" in p.read_text()
```

And prefer `assert s.count(old) == 1` over `if s.count(old) == 1:` — a
replacement that matches nothing should stop, not shrug.

## Never `git add -A` while a review agent is editing

A `code-reviewer` pass edits docstrings in place. Committing everything the
working tree holds sweeps those edits into whatever commit is being made, under
a message that does not describe them — and the reviewer then reports its own
work as having appeared from outside the session. This has now happened twice.

**Stage by path when an agent is running**: `git add src/ python/` and so on, or
wait for it to finish. `git status` before committing shows what is about to be
taken.

## A standing rule states a constraint, never what the code currently does

`CLAUDE.md` carried "`py.allow_threads` around the compute" and "random draws
arrive as an array". PyO3 renamed the first; this branch removed the second.
Both had to be found and corrected by a reviewer, because a file that loads on
every session was describing code that no longer existed.

`CLAUDE.md` already says why: *"`README.md` is how a person runs this. `PLAN.md`
is what the project is and why. This file is how work is done here. A rule
belongs in exactly one of the three; restated in a second, the copies drift."*

**A constraint survives a refactor; a description of current behaviour goes
stale the moment the behaviour moves.** "Never call back into Python inside the
loop" stays true whatever the method is called. Put the mechanism in the module
that implements it, where the code and the sentence are edited together.

## A wasted pass in the baseline is a correctness problem for the claim

The batched NumPy loop is the baseline every speedup is measured against. It
was first written drawing a replacement lifetime for every segment when three
percent are replaced, and totalling by rearranging whole arrays — which made it
up to 3.8x slower than the scalar reference it was supposed to beat.

Nothing was wrong with any *number* it produced. Everything measured against it
was flattered. **Write the baseline as carefully as the fast path**, and treat
"the baseline is slower than the reference" as a defect report rather than a
curiosity.

## Do not transliterate an implementation from one whose constraints it lacks

The polars loop sorted the whole frame into rank order and back, because that
is what the rectangular NumPy form must do — its candidate sets differ between
replications, so compacting them would leave a ragged array. A long frame has
no such constraint: filtering to the candidates first took it from 3.1x slower
than the array form to 1.2x, and made it *faster* on the policy where few
segments are eligible.

**Ask what the new form is good at before porting the old form's workarounds.**

## Publish a measurement only after asking what else it could be measuring

Two headline findings on one branch had to be withdrawn — "the batched loop is
the honest baseline" and "the frame form costs three and a half times the array
form". Both were measurements of code written badly, not of the thing named.

Before a number goes into `PLAN.md`, the README or a pull request: profile it,
or say in the text that it has not been. **The claim is about the tool; the
measurement is of one program, and those are the same thing only when the
program is known to be competent.**

## Move the measurement boundary before the change, not after

Draw generation used to sit outside the timed region. Moving generation into
the kernel would have shown a 3.5 ms/replication gain that was really work
ceasing to be counted. The boundary was moved first, so the same work is
counted on both sides and the improvement is a measurement rather than an
artefact of where the clock went.

## Report the mean of several runs, with the count beside it

The minimum estimates the work and the mean estimates what someone waits for.
The mean is the more conservative choice for a speedup claim, because
interruption inflates numerator and denominator alike rather than only the row
one is pleased with. Report both; neither means anything without the repeat
count.

## Watch what an expensive dependency does to the edit loop, not just the build

Adding polars to the compute crate took a cold release build from 9 seconds to
297. The figure that actually mattered was different: **142 seconds for a
rebuild after any Rust edit**, paid on every iteration rather than on a cache
miss, and by a wide margin the largest cost of the session that added it.

When a build gets slow, iterate with `cargo check` and rebuild only to run
Python. If it stays slow, question the dependency.

## A test that compares two constants cannot fail

`assert!(12_000 < MAX_SEGMENT)` reads like a guard on the shipped size and is
decided at compile time. Clippy caught it; nothing else would have.

Where the requirement is real, state it as a compile-time assertion —
`const _: () = assert!(...)` — so narrowing the field is a build error. Where it
is not, delete it.
