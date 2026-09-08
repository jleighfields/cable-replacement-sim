# Make the array loops draw the way the new generator wants

Computing draws from their position was the right change, but both Python
implementations still ask for them the way they asked for slices of an array.
Measured, that is now where their time goes.

## What the profile says

12,000 segments, 10 replications, `run_to_failure`, which is the policy that
does the least work and so shows the draw cost most clearly:

| | total | in the generator |
|---|---|---|
| Scalar reference | 165 ms | **108 ms (65%)** |
| Batched NumPy | 62 ms | 26 ms (42%) |

The reference is not drawing more numbers than the batched loop. It draws them
in **300 calls where the batched loop uses 30** — one per replication-year
rather than one per year.

## Why that costs so much

**The vectorised Philox in `random_draws.py` has a 255 microsecond floor
whatever the array size.** It is about 320 NumPy operations — ten rounds of a
four-word round function, each word built from shifts, masks and multiplies —
and below a few thousand elements the dispatch overhead is the whole cost:

| counters | 4 | 40 | 360 | 3,600 | 36,000 | 360,000 |
|---|---|---|---|---|---|---|
| microseconds | 255 | 256 | 300 | 762 | 5,964 | 110,425 |

**NumPy ships the same generator in C, and it has no such floor**: 13 us for 360
draws, 63 us for 12,000, 520 us for 120,000 — roughly ten times faster per draw
and no fixed cost. It was not used originally because it is a Python object per
counter, which is useless for scattered positions. It is not useless for a
**consecutive run**, which is what `uniforms_over` already is.

## The two changes, both measured

1. **`uniforms_over` should use NumPy's Philox.** It is defined as a run of
   consecutive positions, which is exactly what NumPy's generator produces, and
   the two agree bit for bit — checked. Every dense draw goes through it: the
   starting lifetime and the fixed priority for every segment of every
   replication. Estimated 11 times faster on that path.

2. **Scattered positions inside one replication-year should be a run and a
   gather.** The reference asks for about 360 scattered positions per year; they
   all lie inside one replication-year, hence inside one consecutive span. Doing
   the span and gathering measures **77 us against 323** — 4.2 times, same
   numbers. It is more draws and less time, because the alternative pays the
   dispatch floor.

   The batched loop's positions span every replication, so they are not one run
   and it keeps the scattered path. That is why this belongs in `uniforms_at` as
   a case it detects rather than in a caller.

## What this is expected to buy

The reference from 165 ms to about 69 ms, and the batched loop from 62 ms to
about 51 ms, at that shape. **The reference matters more than it looks**: it is
what every other implementation is compared against for correctness, and it went
from level with the batched loop to 2.8 times slower when draws became computed.
A speedup ratio quoted against "the fastest Python" moved denominator partly
because the reference regressed, so this is about the honesty of the table as
much as about the runtime.

## Steps

- [x] 1. `uniforms_over` through NumPy's Philox, with the equality against the
      vectorised one asserted in `test_draws.py` rather than assumed.
      `test_a_run_starting_mid_block_drops_the_words_before_it` carries the
      direct assertion, over the block offsets no production caller reaches.
- [x] 2. `uniforms_at` takes the run-and-gather path when every position shares
      a replication, with the measurement in the comment that justifies
      producing draws nobody reads. Bounded by `MAX_RUN_DRAWS`: the run is
      charged for the largest segment asked for, so at the largest segment a
      draw index can carry it would ask for 32 GB. The bound is a memory cap,
      not a tuned crossover — that constant's docstring says why.
- [x] 3. Re-measure both loops and the whole table, at 12,000 segments and
      again at 100,000.
- [x] 4. Check what remains: `order_by_rank` and
      `conditional_failure_probability` are the next largest and are the model
      rather than the plumbing. Left alone — changing the model to make it
      faster is a different decision from changing the plumbing.
- [x] 5. Review passes.
