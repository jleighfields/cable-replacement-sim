# Single precision, and whether it can exist here

## The question

`f32` halves the memory and is worth about twice the arithmetic throughput. Both
are measured below. Whether this project can have it turns on something neither
of those numbers touches: **whether NumPy and Rust compute the same `f32`
answer**, because the validation strategy is bit-for-bit agreement between
implementations with no tolerance anywhere.

That question is cheap to settle and gates everything else, so it is step 1 and
the plan stops there if it fails.

## What is already measured

**Memory.** Floats are **91.8%** of the kernel's per-worker scratch — seven
`f64` arrays per segment against one bool and one index. Halving them saves 46%
of it. The integers are the other 8.2%, which is why narrowing them is not worth
doing: `u32` would save 4 bytes of 61, and `i16` cannot hold a segment
identifier at the 50,000 and 100,000 populations this project runs.

**Speed**, timed on the operations this simulation actually spends time in.
These are NumPy's vectorised ufuncs, which is an **upper bound** rather than a
forecast for the kernel — its loop is scalar per segment and would need LLVM to
auto-vectorise to see anything similar:

| Operation | 12,000 | 100,000 |
|---|---|---|
| Weibull hazard (`**`, `expm1`) | 1.86x | 2.28x |
| Lifetime draw (`log1p`, `**`) | 1.50x | 1.93x |
| Elementwise multiply | 1.93x | 1.97x |
| `argsort` | 0.95x | 0.98x |
| `lexsort` | 1.00x | 1.01x |
| `cumsum` | 1.00x | 1.01x |

**So `f32` is worth about twice the arithmetic and nothing at all on sorting**,
which is latency- and comparison-bound rather than throughput-bound.

**That lands badly for this workload.** `risk_ranked` makes every segment a
candidate, so it is dominated by the sort — `lexsort` alone is 16 ms at 100,000
segments against roughly 5 ms of elementwise work — and it is both the slowest
policy by an order of magnitude and the one the roadmap calls the deliverable.
`f32` would deliver near 2x on the policies that are already fast and something
closer to 1.15x on the one that costs.

**Precision, at least where it was expected to bite.** The greedy budget fill
compares a running total against a budget, and which segment lands last is a
discrete outcome the parity tests compare exactly. Over 2,000 draws of 4,000
candidates from the shipped population, **the funding cutoff never differed**
between `f32` and `f64`, and the running total had drifted $1.71 by the time it
reached a $12M budget. Costs run $7k to $1M, so a candidate has to land within a
couple of dollars of the remaining budget for this to matter. This was expected
to be the blocker and is not one.

## Step 1, which decides whether there is a step 2

**Do NumPy and Rust agree on `f32` transcendentals?**

In `f64` they do, which is the only reason exact parity works at all today. In
`f32` there is no such guarantee: NumPy carries hand-written SIMD loops for
single-precision `expm1`, `log1p` and `pow`, and Rust calls the system libm's
`expm1f`, `log1pf` and `powf`. These are different implementations and neither is
required to be correctly rounded.

The test is one Rust function computing a column in `f32` and an exact
comparison against NumPy over a wide sample of the arguments the model actually
produces — ages 0 to 60, uniforms in `[0, 1)`, the shipped shape and scale — for
each of the three functions the annual loop uses:

- `conditional_failure_probability`, which is `**` then `expm1`
- `draw_lifetime`, which is `log1p` then `**`
- `draw_remaining_life`, the same with a division first

**If any of them disagrees anywhere in that sample, stop and record it.** An
`f32` mode cannot exist under a no-tolerance rule if the two languages compute
different numbers, and no amount of plumbing changes that.

## If step 1 passes: what the flag looks like

**The array dtype is the flag.** Nothing needs a new argument. The population
arrays are built in one dtype or the other, every implementation reads their
dtype, and the Rust side takes `PyReadonlyArray1<f32>` alongside its existing
`f64`. The choice belongs on the config model, since a caller legitimately picks
it per run, and the population generator is where it takes effect.

**Philox stays in `f64` and the draw is cast at the point of use.** The
generator produces `u64` words and converts them by NumPy's rule, and that
conversion is what every parity test against `numpy.random.Philox` checks.
Narrowing an `f64` uniform to `f32` is a deterministic IEEE rounding that both
languages do identically, so the generator, its constants and its validation are
untouched. Generating draws directly in `f32` would break agreement with NumPy's
own Philox for no gain, since the draws are a small share of the arithmetic.

**Parity comes for free, and stays exact.** `tests/test_parity.py` takes its
implementations from `run.RUNNABLE`; that is the property that made adding and
then removing a fourth implementation cost zero test edits. A dtype parameter
threaded through the same registry doubles the matrix without a new test, and
the comparison stays `f32` against `f32` — **no tolerance is introduced
anywhere.** An earlier reading of this plan said otherwise and was wrong.

## Steps

- [x] 1. The transcendental agreement test, in Rust and NumPy, over the three
      functions. **Passes.** Identical over 2,000,000 samples of ages 0-90 and
      uniforms on `[0, 1)`; identical again across all 160 distinct
      (shape, scale) pairs the shipped fleet carries, at 50,000 samples each;
      identical at the edges — a zero uniform, the smallest and largest a draw
      can produce, and age zero. The comparison is known to discriminate:
      computing the hazard with `exp` instead of `exp_m1` on the Rust side
      makes 76% of it differ.

      The reason it agrees was guessed here as both sides widening to double
      and rounding once, and that guess is wrong — see the review section
      below. They agree because both reach the same C library.
- [x] 2. `precision` on the config model, taking effect in
      `run.segment_arrays` rather than in `population.py` as planned — the
      population frame is built at one width and narrowed as its arrays are
      taken, so the choice reaches an implementation without the generator
      knowing about it.
- [x] 3. The scalar reference and the batched loop read the dtype they are
      given. Python needs care that no literal silently promotes back to `f64`
      — a `float` in an expression with an `f32` array does not, but `np.float64`
      scalars from configuration do.
- [x] 4. The kernel generic over the float type, and the binding accepting both.
      This is the bulk of the work and the reason to be sure of step 1 first.
- [x] 5. Parity at both dtypes, which the registry should give without new tests
      — confirmed by watching a planted divergence redden it.
- [x] 6. Re-measure time **and peak resident memory**, both dtypes, both
      population sizes, all three policies. The prediction to check: near 2x on
      `age_threshold` and `run_to_failure`, near 1.15x on `risk_ranked`, memory
      down by about 46%.
- [x] 7. Decide whether it stays, on those numbers. **It stays**, as a run
      option rather than a default. `docs/single-precision.md` has the
      measurement.

## What the measurement said, against what this plan predicted

**The prediction was wrong in its mechanism and roughly right in its
scepticism.** It expected about 2x where a policy is arithmetic-bound and
nothing where it is sort-bound, drawn from NumPy microbenchmarks. The split is
by population size instead: nothing at 12,000 segments, the range in `docs/single-precision.md`'s tables at 100,000,
and it helps the sort-bound policy nearly as much as the other. Halving the
working set matters when forty-eight workers pull their scratch through a
shared cache and does not matter when it already fits. Those microbenchmarks
timed one thread on isolated arrays, which is the wrong shape for the thing
being predicted.

**Step 1's first explanation was wrong, and its own test is what refuted it.**
The guess was that both widths evaluate `powf`, `expm1` and `log1p` by widening
to double, which would have explained the agreement and the absent speedup at
once. It does not hold: a double computation rounded once matches NumPy for
17,936 of 20,000 `expm1` inputs, not all of them. They agree because both sides
reach the same C library, and the arithmetic is no faster because a scalar call
into that library does not get cheaper for being narrower — two mechanisms
rather than one. `docs/single-precision.md` carries the corrected account, and
`test_numpy_and_the_system_library_agree_in_single_precision` is what holds it:
its second assertion exists to rule out the guess above, and deleting it as
redundant would leave the test unable to tell the two apart.

**Memory falls**, which is what the arithmetic did predict, though by less than
it suggested: 4.6% to 20.7% of the whole process across implementations and
sizes, or 15.4% to 33.4% of what the model adds above the interpreter's own
~126 MB. The arithmetic predicted 46% of the kernel's per-worker scratch, and
that scratch is a fraction of either figure. `docs/single-precision.md` carries
the table these come from; an earlier version of this line quoted 25% to 87%,
read off a memory table since retaken at one replication count.
- [ ] 8. Review passes.

## What would make this not worth doing

Worth writing down now, because the measurements above already argue against it
and the argument should not quietly disappear once someone has built it:

- **Memory is not binding.** The 100,000-segment run peaks at 0.32 GB on a
  machine with 251 GB. Halving something that is not a constraint buys nothing.
- **The speedup misses the bottleneck.** It is worth least on the policy that
  costs the most.
- **It is a second numeric path through four implementations**, and the project
  has just finished retiring a fourth implementation on the grounds that
  reaching parity is not a reason to maintain one.
- **The batch-size change is 1.5–1.8x on the whole production path** for a
  signature change and no new failure modes, and it is not done yet.

The honest ordering is: batch size first, then this only if `risk_ranked` stops
dominating or a population arrives where memory binds.

## Open questions

**Should `f32` be a run knob or a build?** As a config value both dtypes ship in
one wheel and a run records which it used, which is what provenance wants. As a
build flag the kernel stays one code path and the wheel doubles. The config
value is more code and the better artifact.

**What happens to a saved run's schema?** Results written in `f32` and results
written in `f64` are different files claiming the same shape. Either the
manifest records the precision and readers check it, or results are always
widened to `f64` on the way out — the second costs nothing measurable and keeps
one schema.
