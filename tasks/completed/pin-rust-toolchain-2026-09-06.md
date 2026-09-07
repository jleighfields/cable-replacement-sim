# Pin the Rust toolchain, and record that one is installed

Branch `feature/pin-rust-toolchain`, cut from `main` after the Phase 2 pull
request merges. Independent of Phase 2: it changes how the project builds and
what continuous integration installs, and belongs in a review about that rather
than one about the maximum-likelihood fit.

## Why

Continuous integration runs `cargo clippy --all-targets -- -D warnings`, and
clippy gains lints with every stable release. With the toolchain floating on
whatever stable is current, a clippy release turns a green branch red with no
change to the code.

That is worse than the wasted run. `CLAUDE.md` says a red check is a finding
and not a flake, and that rule survives only while it is true. The first time
an unrelated pull request goes red because clippy got stricter, the rule starts
being read as advice. Pinning turns that event into a deliberate one-line bump
with its own commit and its own review.

There is room to pin: PyO3 0.27 declares `rust-version = "1.74"` and the
installed toolchain is 1.98.1.

## What changes

1. **`rust-toolchain.toml`** at the repository root:

       [toolchain]
       channel = "1.98"
       components = ["clippy", "rustfmt"]
       profile = "minimal"

   The channel is the minor version rather than an exact patch, so patch
   releases arrive on their own while the minor stays put. That is the line
   that matters for this repository: clippy gains its lints with minor
   releases, so holding the minor is what prevents a green branch going red,
   and following patches costs nothing in return.

   Verified rather than assumed: rustup accepts a minor-only channel, resolves
   it here to cargo 1.98.1, and installs `clippy` and `rustfmt` under the
   minimal profile because the file declares them. `cargo fmt --check` and
   `cargo clippy --all-targets --no-default-features -- -D warnings` both pass
   on the crate under that toolchain.

2. **All three workflows.** `.github/workflows/test.yml`,
   `notebooks.yml` and `app.yml` each say `dtolnay/rust-toolchain@stable`, and
   `test.yml` additionally passes `components: clippy, rustfmt`.

   **These must change in the same commit as the file, or the change makes
   things worse rather than better.** With the pin present and the workflows
   unchanged, the action installs stable, then cargo reads the pin and
   downloads a second toolchain on every run: slower, and the `@stable` line
   now misdescribes what actually compiles. Drop the `components:` input too,
   which the file now owns.

   **Use bare `rustup install`, not `rustup show`.** GitHub runners ship
   rustup, and running it inside the checkout reads the file and installs what
   it names. `rustup show` has that effect too, but only by way of the
   auto-install fallback that fires for any proxy when the active toolchain is
   missing — and rustup 1.28.0 removed that fallback outright before 1.28.1
   restored it with a warning attached. Building the pipeline on a path that
   has already been taken away once would trade one surprise breakage for
   another. Bare `rustup install` is what 1.28.0's release notes name as the
   replacement, and it reports the active toolchain as installed. Pass
   `--no-self-update`: rustup updates itself by default here, and a job that
   upgrades its own toolchain manager is the drift this pin exists to stop.

   `actions-rust-lang/setup-rust-toolchain@v1` reads the file natively and is
   the alternative if a cached action is preferred to a shell line.

3. **`PLAN.md` §14, First actions in the next session**, and the line near the
   top of the document that says the extension module is not currently
   rebuildable because the machine has no Rust toolchain. Both are now false:
   rustup 1.29.1 with rustc and cargo 1.98.1, `clippy` and `rustfmt` included,
   is installed. The checklist item in §12, PyO3 / maturin practicalities,
   asking to confirm a toolchain before Phase 0 can go too.

## How to verify, rather than assume

- `cargo fmt --check`, `cargo clippy --all-targets --no-default-features --
  -D warnings` and `cargo test --no-default-features` all pass locally on the
  pinned toolchain. The third needs the system package below.
- `rustup install` inside the checkout reports the pinned version as the
  active toolchain, and `cargo --version` there reads 1.98.x rather than the
  default toolchain's version.
- Temporarily set the pin to a version that is installed but different, and
  confirm `cargo --version` inside the project reports the pin rather than the
  default. A pin nobody has watched take effect is not known to work.
- Read a continuous integration run's log and confirm one toolchain is
  installed rather than two. This is the failure the change exists to avoid,
  and it is invisible except in the log.

## Blocked on, and not part of this branch

`cargo test --no-default-features` fails on this machine with `unable to find
library -lpython3.12`. `/usr/lib/x86_64-linux-gnu` holds
`libpython3.12.so.1.0` and the `.so.1` symlink from `libpython3.12t64`, but not
the unversioned `libpython3.12.so` that the linker name needs; that comes from
`libpython3.12-dev`, which is not installed.

This is environmental and predates the branch. It shows up only in this gate
because `cargo fmt` and `cargo clippy` never link, and `maturin develop` builds
with `pyo3/extension-module`, where CPython's symbols come from the interpreter
at import time. `cargo test --no-default-features` drops that feature by
design, so the test binary has to find libpython itself. Fix with
`sudo apt install libpython3.12-dev`.

## Open questions

1. **Exact version or minor channel?** Settled: the minor channel, `1.98`.
   Patch releases carry fixes and no new lints, so following them costs
   nothing, and the minor is the boundary clippy's lints arrive on — which is
   the whole reason for pinning.
2. **Who bumps it, and when?** Settled: when something needs a newer
   compiler, and not on a schedule. A bump is a commit that does nothing else,
   so the lints a new release brings are read as that commit's diff rather than
   as an unrelated branch going red. The accepted cost is that the gap can grow
   until one bump arrives with a wall of new findings, in exchange for never
   doing the work speculatively. Recorded in `PLAN.md` §12, PyO3 / maturin
   practicalities, since it outlives this branch.

## Done when

- The three gates pass locally against the pinned toolchain.
- A continuous integration run is green and its log shows one toolchain.
- Nothing in `PLAN.md` still says the machine has no Rust.

## Review

Done, with one item deferred and one addition.

**The pin is watched working, not assumed.** With the file in place,
`rustup show active-toolchain` reports
`1.98-x86_64-unknown-linux-gnu (overridden by '<repo>/rust-toolchain.toml')`,
naming the file as the reason. Setting the channel to a name no toolchain
carries makes cargo refuse outright — `custom toolchain
'1.98.1-not-a-real-channel' specified in override file ... is not installed` —
which is what proves the file is being read rather than ignored. The
"temporarily pin a different installed version" check in the section above was
not usable: the default toolchain and the pinned one are both 1.98.1 here, so
`cargo --version` cannot distinguish them. The break-it check replaces it and
is the stronger of the two.

**Bare `rustup install` is the right entry point.** Run inside the checkout it
prints `the active toolchain '1.98-x86_64-unknown-linux-gnu' has been
installed` and `it's active because: overridden by '<repo>/rust-toolchain.toml'`
— the second line is what a continuous integration log needs to show that one
toolchain was installed and which file chose it. With `--no-self-update` the
`checking for self-update` line it otherwise prints is gone, confirmed by
running it both ways.

**All five gates pass on the pinned toolchain**: `cargo fmt --check`,
`cargo clippy --all-targets --no-default-features -- -D warnings`,
`cargo test --no-default-features`, `uv run ruff check .`, and
`uv run pytest -n auto` at 71 passed. The `cargo test` link failure recorded
under *Blocked on* above is gone; `libpython3.12-dev` is installed.

**Exit codes were checked without a pipe.** `cargo clippy ... | tail` reports
`tail`'s status and not cargo's, so a failing lint reads as a pass. Both Rust
commands were re-run redirecting to `/dev/null` and their own `$?` read.

**Added beyond the plan: two corrections in `PLAN.md` §10.4, Branch protection
on `main`, and §14, First actions in the next session.** Rewriting §14 to drop
the finished toolchain item left its other items visibly stale, and one was
false: the required-status-check ruleset is described as checked in and not yet
applied. It is applied. `gh api repos/{owner}/{repo}/rulesets` returns both
rulesets active, and the required-check one's live rules match the checked-in
JSON exactly — one required context, `test`, with the strict up-to-date policy
on. The status line at the top of the document, which still said Phase 1 was
landing and that the extension module could not be rebuilt, is corrected in the
same commit. What has *not* been watched is the rule refusing a merge, so that
survives as an item rather than being deleted.

**The run log shows one toolchain, which is the check this change exists to
satisfy and the only place the failure would be visible.** In the pull
request's `test` run: one `syncing channel updates for
1.98-x86_64-unknown-linux-gnu`, one `the active toolchain ... has been
installed`, and `it's active because: overridden by
'/home/runner/work/.../rust-toolchain.toml'` — the runner naming the file as the
reason. Zero auto-install warnings, so nothing fell through to the fallback,
and no self-update line. Five components downloaded, which is clippy and
rustfmt arriving with the compiler as the file asks. `Swatinem/rust-cache`
lists `rust-toolchain.toml` among the files in its cache key, so a future bump
invalidates the cache without anyone remembering to. The whole job took 1m1s
and every gate passed inside it, 73 tests included.

### What the branch review changed

The review's one Must Fix and four Should Fixes are all fixed, and its cheap
Consider items with them.

- **`README.md` claimed the fit is checked against an independent
  implementation.** It is not: that sentence and the removal of the outside
  cross-check landed in the same pull request, and the sentence outlived it.
  Replaced with what the recovery ladder actually does — fit data generated at
  parameters the configuration states, and check the estimates come back at
  them. The same paragraph pointed at `PLAN.md` §14 for "what is not yet
  verified on a fresh machine", which this branch deleted; that clause is gone.
- **"Every change so far has arrived through a pull request with a green
  `test`" was false.** Two commits predate continuous integration and went in
  by direct push. The sentence is load-bearing — it is why §14 can say the rule
  has never been *watched* blocking anything — so it now says "every change
  since continuous integration existed".
- **`rustup install` self-updates rustup by default**, which the action it
  replaced suppressed with `--no-self-update`. A job that upgrades its own
  toolchain manager is the drift this pin exists to stop. The flag is now
  passed in all three workflows, and running it both ways confirms it removes
  the `checking for self-update` line.
- **§11's Phase 0 still told the reader to apply the required-check ruleset**,
  above a code block that POSTs — which creates a duplicate rather than
  updating one. Past-tensed, pointing at §10.4 for the current state.
- **The deprecation claim above was wrong.** `rustup show` prints no
  deprecation warning; what fires is rustup's auto-install warning, for any
  proxy rather than for `rustup show` in particular, and it points at
  `RUSTUP_AUTO_INSTALL=0`. The conclusion holds on better grounds: 1.28.0
  removed auto-install and named the no-argument install as its replacement,
  and 1.28.1 restored it only as a warned-about fallback.
- Three spellings of the clippy command across the toml and `PLAN.md` collapsed
  to the full one and a citation; "each release" corrected to "each minor
  release", which is the claim the minor-channel pin rests on; the bare "four
  Rust gates" count and the machine-specific "a Rust toolchain is installed"
  replaced in the status line; the plan's own citation of "Section 7" corrected
  to §12, which is where that checklist item was.

**`tests/test_workflows.py` is new**, from the review's last Consider. The
condition that produces two toolchains — a workflow naming a channel while
`rust-toolchain.toml` names another — is checkable even though the log is not.
Two tests: no workflow references a toolchain-installing action, and every
workflow that compiles installs the pin explicitly rather than resting on
rustup's auto-install fallback. Both were watched failing before being
believed: re-adding `dtolnay/rust-toolchain@stable` reddens the first, replacing
the install step reddens the second, and pointing the directory constant at a
directory holding no workflows reddens both rather than passing over an empty
list. Suite: 73 passed.
