"""Structural checks on the continuous integration workflow files.

The Rust toolchain version is pinned in `rust-toolchain.toml` and nowhere else.
A workflow that also names a channel does not fail and does not warn: the job
installs that channel, cargo then reads the pin and downloads a second
toolchain, and every run is slower while the workflow's own line misdescribes
what compiled. The only evidence is in the run log, so it is checked here
instead.
"""

import pathlib

from cablesim import constants

WORKFLOW_DIR: pathlib.Path = constants.PROJECT_ROOT / ".github" / "workflows"

TOOLCHAIN_ACTIONS: tuple[str, ...] = (
    "dtolnay/rust-toolchain",
    "actions-rust-lang/setup-rust-toolchain",
)
"""Actions that install a Rust toolchain of their own choosing.

Either reads a channel from the workflow rather than from `rust-toolchain.toml`
— `actions-rust-lang` reads the file when given no channel, but naming one
overrides it — so neither belongs here while the pin does the choosing.
"""


def workflows() -> list[pathlib.Path]:
    """Every workflow file, as paths.

    Returns:
        The workflow files under `.github/workflows`, sorted by name. Both
        extensions, because GitHub runs `.yaml` and `.yml` alike and a guard
        that reads one of them reports a clean pass on a file it never opened.

    Raises:
        AssertionError: If the directory holds none, which would make every
            check below pass over an empty list.
    """
    found = sorted(p for p in WORKFLOW_DIR.iterdir() if p.suffix in (".yml", ".yaml"))
    assert found, f"no workflow files under {WORKFLOW_DIR}"
    return found


def test_no_workflow_names_a_rust_toolchain() -> None:
    """The pin is the only thing that chooses a compiler version."""
    offenders = {
        path.name: action
        for path in workflows()
        for action in TOOLCHAIN_ACTIONS
        if action in path.read_text(encoding="utf-8")
    }

    assert not offenders, (
        f"a workflow installs its own toolchain, so a run resolves two: "
        f"{offenders}. The version belongs in rust-toolchain.toml alone."
    )


def test_every_workflow_that_builds_rust_installs_the_pinned_toolchain() -> None:
    """A job that compiles asks rustup for the pin rather than relying on it.

    Leaving the step out still works today, because rustup auto-installs a
    missing active toolchain. That fallback was removed in rustup 1.28.0 and
    restored in 1.28.1 with a warning attached, so a pipeline resting on it is
    resting on something already taken away once.
    """
    missing = [
        path.name
        for path in workflows()
        for text in [path.read_text(encoding="utf-8")]
        if ("cargo " in text or "maturin " in text)
        and "rustup install --no-self-update" not in text
    ]

    assert not missing, (
        f"these workflows compile Rust without installing the pinned "
        f"toolchain first: {missing}"
    )
