# Phase 5 — parallel kernel, batched baselines, and benchmarks

Make the kernel parallel over replications, build the two batched Python
implementations that the speedup is honestly measured against, and produce the
benchmark table and the notebook that renders it.

The single-threaded kernel that landed in the previous phase is the fixed
baseline the parallel number is claimed against, and it is deliberately not
rewritten by this work beyond being made callable in parallel.

## Decisions taken before starting

1. **Thread count is an argument, and one thread skips rayon entirely.** A
   `threads` argument on the binding flows from `run.run` into the manifest
   field that already exists for it. At one thread the loop is a plain
   sequential iterator with no rayon involvement, so the baseline the speedup
   is denominated in carries no work-stealing overhead. Above one, a scoped
   rayon pool is built for that call. The manifest records the number actually
   used rather than the number asked for.
2. **The batched NumPy implementation is held to bit-for-bit agreement with the
   reference**, the same bar the kernel meets. The reduction orders can be made
   to match exactly: a cumulative sum with interleaved zeros equals the
   cumulative sum of the compacted array, because adding zero changes no
   floating-point value, and a `bincount` over offset bins accumulates in the
   same sequential order as one `bincount` per replication.
   For polars, attempt the same bar; where a polars reduction reorders
   internally, pin the quantities that are exact regardless — the counts — and
   use a stated tolerance only for the specific sums that reorder, naming in
   the test which they are and why.
3. **The benchmark table is measured at the full 12,000-segment population**,
   with the replication count chosen per row and reported with it. The scalar
   reference is shown for scale rather than as the baseline, so it runs a small
   count and does not pin the whole table's runtime to itself. Per-replication
   time is the comparable number.
4. **The budget sweep script defaults to the kernel**, keeping `--full` opt-in
   and the reduced size unchanged. The default sweep gets faster without
   redrawing every figure at a different size, and the reference stays one flag
   away for anyone arbitrating a result.
5. **A polars implementation is written in Rust as well as in Python**, so the
   two frame implementations can be compared directly rather than through the
   cheaper proxy of timing the same expressions from both languages. This is a
   sixth mirror of the annual loop and carries the cost every mirror carries —
   another place divergence can hide, another implementation every parity test
   has to cover — and it answers a question the Python pair alone cannot: the
   Python polars package binds the Rust polars crate, so a loss there could be
   the engine being wrong for this shape of work or could be the cost of
   crossing into it, and only having both separates them.

   It needs no frame to cross the boundary. It takes the same arrays every
   other implementation takes, builds its frames on the Rust side, and returns
   the same seven arrays, so nothing about passing a DataFrame through the FFI
   layer arises.
6. **The comparison table lives in the notebook that Phase 5 owns**, which is
   the parity-and-benchmark notebook rather than the policy explorer. Its rows
   are the implementations — the scalar Python reference, the Rust kernel, the
   Python polars loop and the Rust polars loop, with the Rust kernel appearing
   at more than one thread count — and its columns are how long each took and
   whether it reproduced the scalar Python reference's numbers exactly. The
   policy explorer stays about the reliability-against-budget curve; a
   benchmark table there would be a second place the same claim is made.

## Steps

- [ ] 1. `cargo add rayon`. Restructure `src/simulate.rs` so one replication is
      a function returning its own small result block, then run those blocks
      sequentially at one thread and through rayon above one. Per-replication
      scratch buffers move inside that function or into per-thread state; they
      currently live outside the replication loop, which no parallel iterator
      can share.
- [ ] 2. `src/lib.rs`: `py.allow_threads` around the compute, and the `threads`
      argument. Nothing inside the released region may touch a Python object.
- [ ] 3. `python/cablesim/kernel.py` and `run.py`: thread count through to the
      manifest, replacing the hard-coded one.
- [ ] 4. `python/cablesim/batched.py`: the batched NumPy annual loop, state as
      `(replications, segments)` arrays so the year loop runs once per year
      rather than once per replication-year.
- [ ] 5. The batched polars loop in the same module, ranking and the greedy
      fill as a sort plus a cumulative sum within a replication grouping. Its
      uniforms arrive from NumPy as columns, because polars has no addressable
      per-element generator.
- [ ] 6. Parity tests for both, through the same fixture every other parity
      test reads.
- [ ] 7. `scripts/run_benchmarks.py`: the timing harness, writing a table with
      each row's replication count, chunk size, build profile and thread count
      beside its time.
- [ ] 8. The Rust polars annual loop, taking the same arrays as every other
      implementation and returning the same seven, so it is held to the same
      parity tests.
- [ ] 9. `notebooks/05_parity_and_bench.py`: the agreement plots, and the table
      whose rows are the implementations and whose columns are the wall time
      and whether the result matched the scalar Python reference exactly.
- [ ] 10. `scripts/budget_sweep.py` default, and a `--threads` argument.
- [ ] 11. Update the plan document with the measured figures, which it
      currently leaves blank, and record what the polars comparison found.
- [ ] 12. Review passes until no Must Fix or Should Fix remains.
