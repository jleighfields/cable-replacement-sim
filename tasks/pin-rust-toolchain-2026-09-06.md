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
   it names. `rustup show` also has that effect today, by auto-installing, but
   rustup 1.29.1 prints a deprecation warning for exactly that and points at
   `rustup install` — building the pipeline on a path its own tool says is
   going away would trade one surprise breakage for another. Bare
   `rustup install` reads the file and reports the active toolchain as
   installed, which is the supported form.

   `actions-rust-lang/setup-rust-toolchain@v1` reads the file natively and is
   the alternative if a cached action is preferred to a shell line.

3. **`PLAN.md` §14, First actions in the next session**, and the line near the
   top of the document that says the extension module is not currently
   rebuildable because the machine has no Rust toolchain. Both are now false:
   rustup 1.29.1 with rustc and cargo 1.98.1, `clippy` and `rustfmt` included,
   is installed. Section 7's checklist item asking to confirm a toolchain
   before Phase 0 can go too.

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

## Open questions to settle with the user

1. **Exact version or minor channel?** `1.98.1` pins reproducibly and needs a
   bump for every patch release; `1.98` follows patches and still holds clippy
   still, since lints arrive with minor releases. Recommendation: `1.98.1`,
   because the point is that nothing changes without a commit saying so.
2. **Who bumps it, and when?** A pin nobody updates is a project on an ageing
   compiler. Worth deciding whether this is reviewed on a schedule or left
   until something needs a newer feature.

## Done when

- The three gates pass locally against the pinned toolchain.
- A continuous integration run is green and its log shows one toolchain.
- Nothing in `PLAN.md` still says the machine has no Rust.
