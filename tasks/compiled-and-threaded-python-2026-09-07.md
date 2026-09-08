# Numba and threaded NumPy, against plain NumPy and the Rust kernel

## The question

The kernel is reported as beating the fastest Python by 18.5x, 78.4x and 22.6x
across the three policies. Those figures compare **one Python thread against
forty-eight Rust threads**, because every Python implementation refuses a thread
count above one and nothing in this workload reaches a threaded BLAS.

At 100,000 segments under `risk_ranked` the kernel on one thread runs 0.386 s
per replication against the batched loop's 0.432 — 12%, not a multiple — and
0.044 on forty-eight. Almost the whole advantage at that size is the threads.

So the headline number cannot currently distinguish three explanations:

1. Rust generates better code than anything available in Python.
2. *Compiled* code beats interpreted code, and the language is incidental.
3. *Threads* beat one thread, and compilation is incidental.

This measures which it is, by building the two implementations that separate
them and running them against the ones already here.

**They are built as real implementations, not as a benchmark script.** They live
in `python/cablesim/` beside the others, take the same arguments, join the same
parity tests at no tolerance, and are named at the call the same way — which is
how every implementation in this repo is validated, and the only way their
timings mean anything. The polars pair is the precedent: it was built in the
package, measured, and moved to `deprecated/` only once the measurement said it
had not earned its place. Whether these two go the same way is a question for
after the numbers, not before them.

## What gets measured

Four rows, every one of them computing the same model on the same draws:

| Row | What it isolates |
|---|---|
| Batched NumPy, 1 thread | the baseline as published today |
| Batched NumPy, T threads | threads without compilation |
| Numba `njit`, 1 thread | compilation without threads |
| Numba `prange`, T threads | both, against the kernel's both |

The kernel's existing one-thread and forty-eight-thread rows are the comparison
they all run against.

### Threaded NumPy

`ThreadPoolExecutor` over chunks of replications, each chunk running the batched
loop that already exists. The measured basis for expecting it to work: of the
operations the batched loop uses, `lexsort` dominates at seven times the cost of
everything else combined, and it releases the interpreter lock, scaling 3.94x on
four threads. `argsort`, `cumsum`, `where` and the elementwise Weibull chain
release it too, at 2.8x to 3.8x. The three that hold it — `nonzero`, `bincount`,
`isnan` — are together about 1.8% of the total.

Chunking is safe here for a reason worth stating: draws are computed from the
run key and the position being read, so a chunk produces its own draws with no
coordination, and a replication reads identical values whichever thread runs it.

Expect well below 3.94x. Python-level orchestration sits between the array
operations holding the lock throughout, and for a policy as cheap per
replication as `run_to_failure` the vectorised generator's fixed
260-microsecond floor is already 6% of the work.

### Numba

`njit` over the scalar shape — the reference's structure, not the batched one.
That choice is the point of including Numba at all: vectorising costs the
ability to skip, which is why the batched loop loses to the reference under
`age_threshold`, where 655 of 12,000 segments are eligible in the first year and
the batched form sorts all 12,000 anyway. A compiled scalar loop keeps the skip.

**Thread count is set with `numba.set_num_threads`**, which moves a running
process between the one-thread and forty-eight-thread rows without restarting
it. It can only move *down* from `NUMBA_NUM_THREADS`, which is read once at
import, so that variable is set to at least the largest count wanted before
numba is imported — otherwise the wide row runs narrow and reports a speedup
that is really a thread cap. `numba.threading_layer()` names the layer that was
actually used, and belongs in the recorded provenance beside the build profile.
None of this is verified yet; it is the documented interface, and step 1
confirms it.

**`prange` may reorder a reduction as the thread count changes**, which would
break exact agreement and show up as a parity failure rather than as a wrong
number. Replications are independent, so the parallel region should contain no
reduction across them — anything accumulated across replications stays outside
it or runs in a fixed order. This is the specific thing step 4's thread-count
assertion is looking for.

**The bulk of the work is Philox, not the policy loop.** The study has to
reproduce the generator at the same draw indices, bit for bit, inside Numba's
supported subset with matching uint64 wrapping arithmetic. If that cannot be
made to agree, the study stops there and reports that as its finding — a Numba
row computing different numbers is not a comparison.

## Where it lives

`python/cablesim/compiled.py` for the Numba loop — not `numba.py`, which reads
as the third-party package at every import site. Threaded NumPy is a thread
count the batched loop accepts rather than a fourth module, since it runs the
same code over a chunk of the replications.

`check_arguments` opens to a thread count above one for exactly the
implementations that can use it, and keeps refusing it for the ones that
cannot — the message it raises today already names which is which, so that
distinction stays and gains two entries.

Both join `benchmarks` as named configurations. `05_parity_and_bench.py` builds
its table from a list of `benchmarks.Configuration(name, threads, reps)`, so the
new rows are entries in that list and the notebook learns nothing else — it
already renders whatever it is given, and already checks every row against the
reference. `scripts/run_benchmarks.py` picks them up the same way, which is what
keeps the script and the notebook from disagreeing about what was measured.

## Steps

- [ ] 1. `uv add --group benchmarks numba`, and confirm `uv sync --locked` still
      passes.
- [ ] 2. Philox in Numba, checked against NumPy's `np.random.Philox` at the
      positions the existing draw tests use — against the published
      implementation, not against this repo's other two copies of it. **Stop
      here and report if it will not agree exactly.**
- [ ] 3. The annual loop under `njit` in `compiled.py`, joined to the existing
      parity tests at no tolerance rather than given tests of its own.
- [ ] 4. `prange` over replications, with the answer asserted independent of the
      thread count, the way the kernel's already is.
- [ ] 5. A thread count on the batched loop, same two assertions, and
      `check_arguments` widened to admit it for the implementations that can.
- [ ] 6. Add both to `05_parity_and_bench.py`'s configuration list, so the
      notebook renders the four rows against the kernel's two and checks each
      against the reference. Run it both ways — headless and interactive — and
      run `pytest -m notebooks`, which nothing triggers automatically and which
      step 5 obliges by changing `check_arguments`.
- [ ] 7. Run the table: both sizes, three policies, one thread and forty-eight,
      kernel rows alongside.
- [ ] 8. Write the conclusion, and say plainly which of the three explanations
      the numbers support.
- [ ] 9. If the conclusion changes what the published speedup row means, say so
      in `README.md` and `PLAN.md` — the row currently compares one Python
      thread against forty-eight Rust threads without saying that it does.
- [ ] 10. Review passes.

## What would invalidate the comparison

- **`fastmath` anywhere.** It licenses reassociation, which changes reduction
  order and breaks exact agreement. Listed here so it is not reached for when a
  number disappoints.
- **Compilation inside the timed region.** Numba compiles on first call. Every
  configuration is warmed before timing, the way the existing rows are, and
  compile time is reported as its own figure rather than folded in or dropped.
- **Different thread counts between rows.** The point is to hold that fixed and
  vary the language.
- **A debug build of the kernel in the same table.** The profile is written into
  the benchmark output for this reason.

## Open questions, for the user

**Do they stay?** Answerable only after step 7. A Numba loop that lands close to
the kernel is worth keeping as a fourth implementation and worth saying so in
the README; one that does not is the polars situation again, and
`deprecated/README.md` is where that gets recorded. Either way the measurement
is the thing being produced here.

**How many threads is the fair comparison?** Forty-eight matches the published
kernel row, but replications cap the speedup — ten replications cap it at ten
however many threads exist, which is why the 100,000-segment run reached 8.8x
and not 20x. Either the study runs enough replications that forty-eight threads
are not starved, or it reports a lower thread count and says why.
