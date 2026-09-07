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
5. **Whether a polars implementation is written in Rust as well as in Python.**
   Open pending the dependency research; the standing plan says to first time
   the same polars expressions from both languages on the grouping step alone,
   and only then decide whether a sixth mirror of the annual loop is worth
   maintaining.

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
- [ ] 8. `notebooks/05_parity_and_bench.py`: the agreement plots and the
      benchmark table.
- [ ] 9. `scripts/budget_sweep.py` default, and a `--threads` argument.
- [ ] 10. Update the plan document with the measured figures, which it
      currently leaves blank, and record what the polars comparison found.
- [ ] 11. Review passes until no Must Fix or Should Fix remains.
