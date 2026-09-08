# A batch size the implementation chooses

## The finding

`run.run` splits a run's replications into chunks of `DEFAULT_BATCH_SIZE = 50`
and calls the named implementation once per chunk. One number currently serves
two implementations whose constraints point in opposite directions.

**What batching saves each implementation**, computed from what each allocates:

| | Holds per chunk | At 1,000 replications |
|---|---|---|
| Batched NumPy | `(replications, segments)` arrays, eight or more of them | one array is 96 MB at 12,000 segments, **800 MB at 100,000** |
| Rust kernel | per-worker scratch sized by *segments*, plus the results array | results are **5.04 MB for the whole run**, 0.25 MB per chunk of 50 |
| Scalar reference | one replication at a time | nothing that grows with the chunk |

So the knob saves the batched loop hundreds of megabytes and saves the kernel
about five.

**Every figure in that table is computed rather than measured**, and the one
doing the most work is the batched loop's — it is the whole reason the default
cannot simply be raised. Step 5 measures peak resident memory for each
implementation at each batch size and either supports those numbers or replaces
them. An implementation may allocate temporaries the arithmetic above does not
know about, and `polars` and NumPy both keep allocator arenas that resident
memory sees and a size calculation does not.

**What it costs the kernel**, measured end to end through `run.run` at 1,000
replications over five policies, release build, 48 workers:

| Segments | Chunks of 50 | One chunk | |
|---|---|---|---|
| 12,000 | 6.71 s | 3.82 s | **1.76x** |
| 50,000 | 30.43 s | 19.96 s | **1.52x** |

Replications are the axis the kernel parallelises over, so a chunk of 50 across
48 workers is about one replication each, twenty times in sequence, with a pool
built per chunk. The scalar reference is indifferent — it loops one replication
at a time whatever the chunk is.

## What changes

`run.run` takes `batch_size: int | None = None`, and `None` means *the
implementation decides*. An explicit value is still honoured, unchanged, so
nothing that passes one behaves differently.

The resolution belongs beside `RUNNABLE` in `run.py`, for the reason
`CONCURRENT` lives there: it is a property of the implementations, declared
rather than guessed, and every name in it has to be one `RUNNABLE` holds.

`reference` and `kernel` resolve to the whole chunk; `batched_numpy` keeps a
bounded value. The manifest records the **resolved** number rather than `None`,
because `batch_size` is provenance and a manifest that records "whatever the
default was" cannot be read a year later.

## Why this is safe to change now, and was not before

Batch size can only change a result if an implementation mishandles the offset
it is told a chunk starts at. Three assertions now cover that, and the last two
did not exist before this month:

- `test_run.py` runs the same configuration whole, in chunks of two, and in
  chunks of four, and compares the saved rows.
- `test_parity.py` asserts a chunk gives the same rows wherever it starts, for
  every implementation.
- `test_parity.py` asserts the same for a *threaded* chunk that does not start
  at zero, which is what a real run does and what neither of the other two
  covered.

The middle and last were watched failing with the defect planted in two
implementations each.

## What has to be re-measured, and what does not

**Not affected:** every benchmark figure in `README.md`, `PLAN.md` and
`docs/compiled-and-threaded-python.md`. `benchmarks.time_once` calls
`run.RUNNABLE[name]` directly with the whole replication count, so the published
tables never went through `run.run` and were never batched. This is worth
stating in the pull request, because "we changed batching and the benchmark
table did not move" reads as an error otherwise.

**Affected, and to be retaken:**

- `notebooks/06_production_run.py` — its timings and the throughput it prints.
  It currently passes `batch_size=N_REPS` by hand; that argument comes out once
  the default does the same thing, and the paragraph explaining why becomes a
  paragraph about what the default now resolves to.
- `notebooks/04_policy_explorer.py` — three `run.run` calls at the reduced size.
  No published timing, but the notebook suite's wall time moves.
- `scripts/budget_sweep.py` — its `--batch-size` default is
  `run.DEFAULT_BATCH_SIZE`, which becomes `None`. A sweep is eight budget levels
  times five policies, so this is where the change is worth the most.
- The notebook suite's runtime, which gates nothing but is the number someone
  waits for.

## Steps

- [ ] 1. `batch_size: int | None = None` on `run.run`, resolved per
      implementation beside `RUNNABLE`, with the manifest recording the resolved
      value.
- [ ] 2. `scripts/budget_sweep.py`'s `--batch-size` default follows, so the
      command-line default is the implementation's rather than a fixed 50.
- [ ] 3. Tests: that the resolved default differs by implementation and is
      recorded in the manifest; that an explicit `batch_size` still overrides;
      that a resolved default names only implementations `RUNNABLE` holds. The
      equivalence tests above already cover the results, and should be watched
      failing once against the new path rather than assumed to still apply.
- [ ] 4. Drop `batch_size=N_REPS` from notebook 06 and rewrite the paragraph
      that explains it.
- [ ] 5. Re-measure time **and peak resident memory**, on a quiet machine, and
      record before and after:
      - notebook 06 at 12,000 and at 50,000 segments;
      - a budget sweep;
      - a grid of batch size against implementation against population size,
        which is what says whether the resolved defaults are the right ones and
        whether the batched loop's constraint is where the arithmetic puts it.

      Peak resident memory through `resource.getrusage(RUSAGE_SELF).ru_maxrss`,
      one implementation per process so the figure attributes to one of them —
      the 0.32 GB already recorded for a 100,000-segment run covers a process
      that ran three implementations in turn and cannot be split between them.
      A run that would exceed memory has to be run under a cap rather than left
      to swap, the way the draw-index test bounds a child's address space.
- [ ] 6. Check `PLAN.md` for anything that describes batching as a single
      default, and `README.md` for the same.
- [ ] 7. Review passes.

## Open questions

**Is the batched loop's memory where the arithmetic says?** The case against
simply raising the default rests on an 800 MB array at 1,000 replications and
100,000 segments, times eight or more arrays. If measured peak memory comes in
far below that, the whole trade changes and a single large default may be right
after all. This is the first thing step 5 should answer, because it decides
whether the rest of the change is necessary.

**Does `batched_numpy` want a fixed 50, or one that scales with the
population?** Its constraint is `batch x segments`, so 50 is 5 MB per array at
12,000 segments and 40 MB at 100,000. A fixed batch means the memory it was
chosen to bound grows with the fleet anyway. Scaling it to hold megabytes rather
than replications constant would be the honest form, and is more machinery than
the problem may deserve.

**Should the kernel resolve to the whole chunk, or to a large fixed number?**
The whole chunk is what the measurement supports and holds 5 MB at 1,000
replications. A study running 100,000 replications would hold 500 MB of results,
which is fine, but the number grows without a stated bound and nobody has run
one.
