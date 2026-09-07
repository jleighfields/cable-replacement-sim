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

If a wait is genuinely unavoidable, match on something that cannot match the
waiter — `pgrep -f "[r]emeasure.py"`, or the interpreter rather than the script.
And note the harness kills a foreground command at ten minutes, so a loop
longer than that is doubly wrong.

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
