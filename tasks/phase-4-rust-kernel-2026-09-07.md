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

To be written when the branch is finished.
