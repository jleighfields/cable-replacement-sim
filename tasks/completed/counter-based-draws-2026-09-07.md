# Counter-based draws, generated inside the kernel

Move random-number generation for the simulation from NumPy into the Rust
kernel, using a counter-based generator so that a draw is a pure function of
its index rather than of a stream position.

Cut from the Phase 5 branch rather than added to it. Phase 5 is the parallel
kernel and the benchmark table; this moves every seeded number in the project
and deserves its own review.

## Why

**The draw array is the binding memory constraint.** It is
`(replications, segments, years + 1)` — 149 MB for a fifty-replication chunk at
the shipped population, and 14.6 GB at a million segments with forty-eight
workers. Measured, **only 1.45% to 1.74% of its year entries are ever read**: a
year's draw is consumed only by a segment replaced that year.

It exists because generation lives in NumPy while the workers run in Rust with
the interpreter lock released, so a worker cannot ask for a draw and the array
is the handoff. Deferring generation does not help — within a chunk every
replication needs its draws at once — and skipping the unread 98% needs random
access rather than laziness.

**A counter-based generator gives random access and is stateless**, which is
what makes it safe under rayon without a lock and independent of scheduling:

    draw(purpose, replication, segment, year)
        = to_double(philox(key(seed), counter = pack(purpose, replication, segment, year)))

Two workers computing draws for different replications share nothing. Results
stay identical at any thread count, which is the property the existing
thread-count parity test already checks.

## What it buys, measured or computed

- The draw array leaves the kernel path: 14.6 GB to about 2.7 GB of per-worker
  scratch at a million segments, and that scratch scales with threads rather
  than with the chunk size.
- **Chunking stops being load-bearing for the kernel.** `batch_size` exists to
  bound the draw array; without one, a call could carry every replication.
- The 98.5% of draws nobody reads are never generated — about 8% of runtime.

## What it costs

- A generator implementation enters the crate, contradicting a rule stated in
  both `CLAUDE.md` and `PLAN.md`. The rule is rewritten rather than bent.
- **Every seeded number in the project moves.** Draws currently come from PCG64
  raw streams spawned per purpose and per replication; Philox indexed by
  position is a different stream.
- Cross-language draw parity stops being structurally impossible to break and
  becomes a test. That test is cheap and strong, but it is a test.

## Decisions taken before starting

1. **Philox-4x64-10**, the Random123 standard, because NumPy ships it. Both
   sides then agree with a published reference rather than with each other.
2. **Implemented by hand rather than taken from a crate.** It is about forty
   lines of well-specified arithmetic, and the crate this lives in already pays
   five minutes of cold build for polars. Correctness is established by
   comparing against NumPy over a grid, which is a stronger check than a
   dependency's own tests.
3. **The counter packs `(purpose, replication, segment, year)`** and must be
   injective. The purpose field replaces the current spawn-by-purpose
   separation, which is load-bearing: without it the replacement-policy
   priorities would correlate with the same segments' lifetime draws, and the
   random policy would stop being a control.
4. **NumPy keeps a matching path.** The Python reference and the batched
   implementations still take draws as arrays, so `random_draws` must be able
   to materialize exactly what the kernel would generate. That is what keeps
   the parity tests exact, and it only ever runs at test sizes.

## Steps

- [x] 1. Establish NumPy's Philox mapping from raw-stream index to
      `(counter, lane)`. **Done before writing anything: counter `c` yields
      four `u64`, so index `i` is `counter = i // 4`, `lane = i % 4`.**
- [x] 2. `src/draws.rs`: Philox-4x64-10, the counter packing, and the
      `u64` to double conversion NumPy uses.
- [x] 3. A Rust test against vectors taken from NumPy, and a Python test
      comparing the two implementations over a grid of indices. `draws.rs`
      carries eight Rust tests and `tests/test_draws.py` seventeen, and the
      comparison is three-cornered: Rust against NumPy, Python against NumPy,
      and Python against Rust.
- [x] 4. `random_draws.py`: a Philox path producing the identical values, so
      the reference and the batched loops read what the kernel generates.
      **Both designs now live in that module** — the spawned `SeedSequence`
      sources stayed for the population table and the failure history, which
      are drawn once per run and which nothing crosses a language boundary to
      reproduce.
- [x] 5. The kernel generates its own draws; the draw array leaves its
      signature. It takes `draw_key` and `first_replication` instead, and
      `n_reps` became explicit because there is no longer an array whose shape
      it could be recovered from. No test-only entry point was needed: the
      Python side computes the same uniforms from the same positions, so parity
      compares two generators rather than one generator and one reader.
- [x] 6. Whether `batch_size` survives on the kernel path. **It does not** —
      the kernel resolves to the whole run, which is a separate plan of its own
      and is where that was measured.
- [x] 7. Update `PLAN.md`: the rule that the kernel neither seeds nor
      generates, the settled item recording this as declined, and the memory
      figures that follow from it. **Finished later than the rest, and it was
      the only step still outstanding.** The settled item and the opening
      summary were rewritten when the change landed, but a pocket of section
      2.11 was not: it still headed its consequences "of materializing the
      draws", still said "passing an array is the simplest way to be indexed",
      still showed the `PCG64`/`SeedSequence` code building a
      `(chunk, segments, years + 1)` array, and still gave the random policy a
      second array called `policy_uniforms`. The Rust-versus-Python table
      described the draws as "built once per chunk and handed to every
      implementation", and the kernel's own bullet said it "generates nothing".
      All of that described the superseded design while section 2.11's opening
      paragraph, three hundred lines above, correctly said there is no draw
      array.
- [x] 8. Review passes, as part of the pull request that carried the work.
