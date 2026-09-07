# Phase 4 — the Rust kernel, single-threaded

Port the annual loop to Rust, bind it with PyO3, and validate it against the
Python reference that landed in Phase 3. The reference is not replaced: it is
what a parity failure is arbitrated against.

## Decisions taken before starting

Four questions were open where the roadmap and the Phase 3 code disagreed, or
where the roadmap left the choice to this phase. All four are settled:

1. **The Rust policy struct is `Resolved`, not `PolicyConfig`.** A mirrored
   pair carries the same name on both sides, so `policies::Resolved` mirrors
   `policies.Resolved` field for field. The plan's section on the call is
   updated to match rather than the other way round.
2. **The kernel returns a seven-element tuple, and a wrapper builds
   `simulate.Results` from it.** Both implementations then return the identical
   Python type, so `run.py`, `metrics.py` and every test call either without
   knowing which they hold. A `#[pyclass]` would be a second result type that
   unpacking, `_fields` and indexing all break on.
3. **Strictly single-threaded.** No rayon, no `py.allow_threads`. Phase 5 adds
   both and measures them against this phase's kernel, which is the only way
   that number means anything.
4. **Wired through the sweep and notebook 04**, not left as a library with
   tests. `--implementation kernel` on the sweep script, and a closing notebook
   section showing the two implementations agree.

## Steps

- [x] 1. `cargo add numpy` at the version matching the PyO3 pin. No rayon, no
      `rand` — every uniform arrives from NumPy.
- [x] 2. `src/weibull.rs`: `conditional_failure_probability`, `draw_lifetime`,
      `draw_remaining_life`. Scalar, same names and same argument order as
      `weibull.py`.
- [x] 3. `src/policies.rs`: `Resolved`, the policy tags, `planned_cost`,
      `rank_key`, `eligible`, `order_by_rank`, `fund` — mirroring
      `policies.py`. The sort comparator is hand-written and total:
      `(rank descending, segment_id ascending)`, NaN last and asserted absent.
- [x] 4. `src/simulate.rs`: the year loop, mirroring `simulate.py` step for
      step. Seven flat result buffers, indexed `(replication, year, class)`.
- [x] 5. `src/lib.rs`: the `#[pyfunction]`, argument names identical to
      `simulate.run_chunk`'s, contiguity and shape checked at the boundary.
- [x] 6. `python/cablesim/kernel.py`: the wrapper, whose docstring and the Rust
      doc comment must not say different things.
- [x] 7. `tests/conftest.py`: the one fixture every parity test reads, so
      "the same draws" is true by construction rather than by coincidence.
- [x] 8. `tests/test_parity.py`: the deterministic test with lifetimes forced
      through the ordinary `scale` array, and the statistical test at 50
      replications and 2,000 segments.
- [x] 9. `scripts/budget_sweep.py`: `--implementation`, recording the build
      profile in the manifest when the kernel ran.
- [x] 10. `notebooks/04_policy_explorer.py`: a closing section running one
      chunk through both implementations and asserting they agree. No timing
      claim — that is notebook 05's job, and a speedup without a stated
      baseline, profile and thread count is not a measurement.
- [x] 11. A pass over every Rust comment and doc comment for a reader who
      knows Python and is learning Rust: where the two languages differ and why
      the Rust is shaped the way it is — borrowing and `&`, `Result` against
      exceptions, `match` against `if`/`elif`, why `f64` has no total order,
      why buffers are allocated once outside the loop. The audience is someone
      reading `simulate.rs` beside `simulate.py` to learn the language, not
      only to check the model.
- [x] 12. `PLAN.md` and `README.md`: the struct name, the return type, the
      wrapper module, and the status paragraph.
- [ ] 13. `code-reviewer` per commit, then the full branch pass, repeated until
      no Must Fix or Should Fix remains.

## Review

Every step is done and the branch is green: 246 tests, four notebooks in 32
seconds, nine Rust tests, `ruff`, `cargo fmt`, `cargo clippy` and the sweep
through the kernel end to end.

### What the kernel is worth, measured

At 12,000 segments, 50 replications and a 30-year horizon, release build, one
thread, against the reference on the same draws: **1.9x** with no candidates,
**5.8x** under an age threshold that leaves 655 of 12,000 eligible, and
**1.0x** for the three policies whose neutral threshold makes every segment a
candidate every year. The kernel scores only candidates while the reference
scores the whole population, which is the whole of that spread.

So the single-threaded case for the kernel rests on policies that fund a
minority of the fleet, and the case at large rests on rayon. That is a weaker
result than the roadmap anticipates, and it is recorded in the plan rather than
left for the benchmark phase to discover.

One exact saving was identified and deliberately not taken: this year's
subtrahend in the annual failure probability is next year's minuend, so
carrying the accumulated hazard forward halves the `powf` calls. It is deferred
because it puts per-segment state in the year loop needing invalidation in
three places, and because it would stop `simulate.rs` reading line for line
against `simulate.py`.

### What the reviews cost, and what they were worth

Five passes. Four rounds of fixes, covering **5 Must Fix and 20 Should Fix**.
The defects clustered in three places, and the pattern is worth carrying
forward:

**A guard added to one side of the mirror is a defect until the other side has
it.** Four separate rounds fixed boundary checks that landed on the kernel
alone — an empty population, a class index past the axis, an unrecognized
policy tag, a wrong-length array, an over-long per-year series. Each made the
two implementations answer differently to input neither should accept, which is
exactly what the mirror is supposed to make impossible.

**A test that cannot fail is worse than no test.** Six of them: an invariant
whose breach stopped the suite at collection rather than reddening; a guard
whose message was the only thing distinguishing it from the lookup below it; a
budget-overrun test that never reached the greedy fill; a set comparison true
by construction; a set-minus-itself; a fixture guard nothing ever called with a
bad value. Mutation testing found all six. Reading found none of them.

**A number in prose needs the same evidence as a number in code.** The
saturation claim in the annual failure probability was wrong three times: the
threshold named the wrong mechanism, then the right threshold was measured
against the wrong quantity, twice. The cause was an identifier — `accumulated`
named a within-year increment — and both wrong measurements followed the word
rather than the expression. Renaming it to `annual_hazard` is the actual fix;
the prose was a symptom. Two performance comments also claimed magnitudes that
had not been measured, and one was out by a factor of twenty.

### Left open, deliberately

- The hazard carry-forward above, for Phase 5.
- The sweep script's reduced default, chosen when the reference was the only
  implementation and worth revisiting now the kernel makes full size cheap.
- `PLAN.md`'s full-size benchmark figures are measured but reproduced only at
  reduced size in review; the shape holds at both.
