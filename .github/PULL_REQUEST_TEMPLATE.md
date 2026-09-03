<!-- Write this so it stands on its own. Your reviewer has the diff and
     nothing else: no plan file, no skill checklist, no numbered phase list, no
     memory of the conversation that produced this. Say what a reference means
     instead of naming it — "the greedy budget allocation in the Rust kernel",
     not "policy.rs step 4"; "the min-of-n scale reduction" rather than a
     section number. Those numbers live only inside the file that defines
     them, they shift when it is edited, and half the time that file is not
     even in the diff.

     Same for the rest: no "as discussed", no ticket number standing in for
     the reason, no comparison to a version of the code the reviewer cannot
     see. CLAUDE.md's "Comments & docstrings are self-contained" is the same
     rule; this is where it applies to a PR. -->

## Summary
<!-- What does this PR do and why? Include the context needed to judge it. -->

## Changes
<!-- List the main changes in this PR -->
-

## Both sides of the mirror
<!-- The Python oracle and the Rust kernel implement the same model twice, on
     purpose — the parity tests are what validate the kernel. If this PR
     touches one side, say what happened to the other, and if it touches
     neither, delete this section. -->
- [ ] Touches `src/` — the matching `python/cablesim/` module was read and
      updated, or is confirmed unaffected
- [ ] Touches `python/cablesim/policies.py`, `reference.py` or `weibull.py` —
      likewise for `src/policy.rs`, `sim.rs`, `weibull.rs`
- [ ] Parity tests still pass, and any changed numeric default appears in
      exactly one place

## Test plan
- [ ] Unit tests pass (`uv run pytest`)
- [ ] Lint clean (`uv run ruff check`)
- [ ] Rust clean (`cargo fmt --check`,
      `cargo clippy --all-targets --no-default-features -- -D warnings`,
      `cargo test --no-default-features`)
- [ ] Rebuilt before testing where `src/` changed (`maturin develop --release`)

<!-- The test check runs all of these, so tick them from a local run and let
     CI be the second opinion rather than the only one. "Tests pass" with
     nothing behind it is the claim CI exists to stop anyone having to take on
     trust.

     Rebuild before you tick the first box if you touched src/: pytest imports
     whatever extension module is currently installed, so a stale one reports
     a pass on code that is not in this diff. Release mode matters for the
     same reason it matters in CI — the parity tests run many replications,
     and a debug build makes that slow enough that people skip it. -->

## Timing claims
<!-- Delete unless this PR claims something got faster. A speedup is a
     measurement or it is nothing: give the baseline it was measured against,
     the build profile, and the thread count. -->

## Manual testing
<!-- Describe any manual testing performed and results -->
