# A batch size the implementation chooses

## The finding

`run.run` splits a run's replications into chunks of `DEFAULT_BATCH_SIZE = 50`
and calls the named implementation once per chunk. (That constant is the
pre-change state this plan sets out to fix; `BATCH_SIZES` replaced it, and the
name `DEFAULT_BATCH_SIZE` no longer exists.) One number currently serves
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
replications over five policies, release build, 48 workers. These are the
figures that motivated the change, each a single run:

| Segments | Chunks of 50 | One chunk | |
|---|---|---|---|
| 12,000 | 6.71 s | 3.82 s | **1.76x** |
| 50,000 | 30.43 s | 19.96 s | **1.52x** |

**Superseded by step 5, and not comparable to it cell by cell.** Step 5 lands at
1.66x and 1.67x, so the size-dependence this table appears to show is not there.
But every absolute cell moved as well, by 14% to 34% — more than either
measurement's own spread accounts for — and the two were taken in different
sessions on a shared machine, so nothing here says how much of that is load and
how much is anything else. The table is kept because it is what the decision was
made on. **Read step 5's numbers, not these.**

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
  `DEFAULT_BATCH_SIZE`, the single constant this change replaces, and it becomes
  `None`. Eight budget levels times five policies looked like where the change
  would be worth the most; step 5 measured it unchanged, because the sweep's
  reduced defaults run 40 replications and that is one chunk at either size.
- The notebook suite's runtime, which gates nothing but is the number someone
  waits for.

## Steps

- [x] 1. `batch_size: int | None = None` on `run.run`, resolved per
      implementation beside `RUNNABLE`, with the manifest recording the resolved
      value. `BATCH_SIZES` holds them and an import-time guard refuses a
      runnable implementation that has none.
- [x] 2. `scripts/budget_sweep.py`'s `--batch-size` default follows, so the
      command-line default is the implementation's rather than a fixed 50.
- [x] 3. Tests: that the resolved default differs by implementation and is
      recorded in the manifest; that an explicit `batch_size` still overrides;
      that a resolved default names only implementations `RUNNABLE` holds. All
      three watched failing. The equivalence test was watched failing against
      the new path too, by shifting the chunk boundaries, rather than assumed
      to still apply.
- [x] 4. Dropped `batch_size=N_REPS` from notebook 06; the paragraph now
      explains what each implementation resolves to rather than why the
      notebook overrode one.
- [x] 5. Re-measured, time and peak resident memory. Through `run.run` on the
      kernel at 1,000 replications, forty-eight threads, batch fifty against
      the implementation's own:

      | Segments | Batch 50 | Its own | | Peak, 50 | Peak, own | Runs |
      |---|---|---|---|---|---|---|
      | 12,000 | 5.54 s | 3.34 s | **1.66x** | 431.4 MB | 385.6 MB | 6 timed, 3 for memory |
      | 50,000 | 24.85 s | 14.90 s | **1.67x** | 651.8 MB | 625.5 MB | 12 timed, 3 for memory |

      Every cell is a mean and the count is beside it, because an earlier
      version of this table reported single runs and its 50,000 row did not
      reproduce — read at 24.25 s once and at 23.22 s by a second measurement,
      against a mean of 24.85 s over twelve. **That cell is the noisy one**,
      spanning 23.1 s to 27.9 s within these runs, while its unchunked
      denominator stays inside 14.7 to 16.0 s and the 12,000 row varies by
      under 3% across its six. Those spreads describe the runs behind this
      table and say nothing about the earlier single ones above, which do not
      fall inside them. What
      moved with the repeat count is the *shape* of the result: the two sizes
      cost the same 1.66x rather than 1.76x falling to 1.52x.

      Peak resident memory is one process per configuration, since a peak
      belongs to the process. It falls slightly rather than rising, which the
      estimate did not predict: each chunk writes its own parquet part, so
      twenty chunks hold twenty parts where one holds one, and that outweighs
      the 4.8 MB the larger results array costs.

      **The scalar reference is close to indifferent on both counts**, which is
      what its one-replication-at-a-time loop predicts and is now measured
      rather than argued. Two independent runs of three at 2,000 segments and
      200 replications on one thread disagree about how close: 8.26 s chunked
      against 8.20 s whole, and 8.23 s against 7.83 s, the second separating
      completely. So chunking costs it between under 1% and about 5%, against
      the 66% the kernel pays, and memory does not move — 230.8 MB against
      230.6 MB. Either figure supports the same decision.

      Notebook 06 is not a measurement of this change and its timing is not
      quoted as one. It passed `batch_size=N_REPS` by hand before and resolves
      to the same size now, so it does identical work either way; the run-to-run
      difference between the two readings is session variance, and an earlier
      version of this line offered it as a speedup.

      **The budget sweep is unchanged, 1.70 s against 1.68 s, and that is not a
      disappointment but arithmetic**: it runs at the reduced 40 replications,
      which is a single chunk at fifty and a single chunk at the kernel's own
      size. The change can only pay where a run has more replications than the
      bounded batch, which the sweep at its reduced defaults does not.

      Original scope, for the record:
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
- [x] 6. `PLAN.md` had two: a paragraph deriving the batch's useful range from
      one number, and a claim of twenty boundary crossings per policy, which is
      now one. `README.md` had none.
- [x] 7. Review passes, two of them. The first found the 50,000-segment timing
      row did not reproduce, and that every figure in step 5 was a single run.
      All of them are retaken above as means with their repeat counts, which
      changed the conclusion: the cost is the same at both population sizes
      rather than falling with size. It also found that both batching
      assertions read the manifest, which records what was resolved and not
      what the run did — two
      mutations that made a run record one size and perform another left the
      suite green, and the test now wraps the registry entry and asserts on the
      calls.

      The second found that repair incomplete in two more places: the
      explicit override and the sweep script's default were asserted through
      the manifest and through the parsed flag, and both were green against a
      mutation that made the run chunk at some other size. Both now assert on
      the calls. It also found that the recorder changed what it watched: the
      wrapper's module is not the annual loop's, so `run.run` recorded no build
      profile for any run made inside it, and `functools.wraps` is what fixes
      that. Two claims here were wrong and are corrected above: the growth of
      an unbatched run counted the result arrays and not the rows frame built
      from them, understating the peak about twelvefold, and the reference's
      indifference was measured once at 0.7% where a second measurement put it
      at 5%. The bounded batch size is pinned as a memory bound, which an
      earlier round argued was not possible; the measurement it is pinned
      against is the one this change took.

## Open questions

**Is the batched loop's memory where the arithmetic says?** **Answered first,
and yes — slightly worse.** Measured with `scripts/measure_memory.py`, one
implementation per process:

| Segments | Batch | Peak | Above import | One array | Ratio |
|---|---|---|---|---|---|
| 12,000 | 50 | 224.8 MB | 98.6 | 4.8 | 20.5x |
| 12,000 | 250 | 475.3 MB | 349.7 | 24.0 | 14.6x |
| 12,000 | 1000 | 1,370.9 MB | 1,245.8 | 96.0 | 13.0x |
| 100,000 | 50 | 835.8 MB | 710.0 | 40.0 | 17.8x |
| 100,000 | 250 | 2,774.9 MB | 2,649.8 | 200.0 | 13.2x |
| 100,000 | 1000 | 9,991.0 MB | 9,865.3 | 800.0 | 12.3x |

The loop carries twelve to twenty times one `(replications, segments)` array,
not the eight this plan estimated, so a single large default would peak at ten
gigabytes on the largest run this project has done. The bound is necessary and
the rest of the change follows.

**Does `batched_numpy` want a fixed 50, or one that scales with the
population?** **Fixed, for now, and the reason is stated rather than assumed.**
The concern was real — a fixed batch lets the memory it bounds grow with the
fleet, measured at 99 MB above import at 12,000 segments and 710 MB at 100,000,
a sevenfold rise for a bound that did not move. But 710 MB is not a constraint
on any machine this runs on, and a batch that scales inversely with the
population is a second rule to keep in step with the first. Revisit if a
population arrives where 710 MB matters; the measurement above is what to
revisit it against.

**Should the kernel resolve to the whole chunk, or to a large fixed number?**
**The whole chunk**, which is what the measurement supports: at the 1,000
replications this project runs, the peak went *down* rather than up, because
twenty chunks each write a parquet part.

The unbounded growth is real and larger than this plan first said. Measured
through `run.run` over five policies, a run peaks 220 MB above the interpreter
at 1,000 replications and 2,499 MB at 40,000 — about 58 MB per thousand, where
the result arrays alone account for 5. The rows frame is built from those arrays
and held while they are still live, and the peak carries both. So a hundred
thousand replications would be several gigabytes rather than the 500 MB the
arrays suggest. It is stated where the constant is declared rather than capped,
because nobody has run one and a cap chosen without that run would be a number
with no measurement behind it.
