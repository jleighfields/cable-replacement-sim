"""Structural checks on the continuous integration workflow files.

The Rust toolchain version is pinned in `rust-toolchain.toml` and nowhere else.
A workflow that also names a channel does not fail and does not warn: the job
installs that channel, cargo then reads the pin and downloads a second
toolchain, and every run is slower while the workflow's own line misdescribes
what compiled. The only evidence is in the run log, so it is checked here
instead.
"""

import os
import pathlib
import subprocess

import yaml
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


def app_gate_script() -> str:
    """The shell the app workflow runs to decide whether an app suite exists.

    Read out of the workflow rather than restated here, so this checks the
    line that will run in the job instead of a copy that can drift from it.

    Returns:
        The body of the `run:` block belonging to the step the `detect` job
        takes its `present` output from.

    Raises:
        AssertionError: If that job or that step is not there under the names
            the output expression uses, which would mean the gate has been
            restructured and this check is pointing at nothing.
    """
    workflow = yaml.safe_load((WORKFLOW_DIR / "app.yml").read_text(encoding="utf-8"))
    detect = workflow["jobs"]["detect"]
    assert detect["outputs"]["present"] == "${{ steps.suite.outputs.present }}", (
        "the detect job no longer publishes its answer from a step called "
        "`suite`, so this check cannot find the script that produces it"
    )
    scripts = [step["run"] for step in detect["steps"] if step.get("id") == "suite"]
    assert len(scripts) == 1, f"expected one step with id `suite`, found {len(scripts)}"
    return scripts[0]


def gate_answers_present(tree: pathlib.Path) -> bool:
    """Runs the gate script against a directory and reads what it decided.

    Args:
        tree: The directory to run it in, standing in for a checkout.

    Returns:
        True when the script wrote `present=true`, which is the value the
        `app` job's `if:` compares against.

    Raises:
        AssertionError: If the script wrote no `present=` line at all, which
            leaves the job condition reading an empty string and the suite
            skipped for a reason nothing reports.
    """
    written = tree / "github_output"
    written.write_text("", encoding="utf-8")
    # `bash -e`, because that is the shell GitHub gives a `run:` block on a
    # Linux runner, and `compgen` is a bash builtin that a `sh` fallback does
    # not carry.
    subprocess.run(  # noqa: S603
        # Resolved from PATH, the way the runner resolves the shell it gives a
        # `run:` block. The script is read out of a file tracked in this repo,
        # so it is no more untrusted input than the workflow itself.
        ["bash", "-e", "-c", app_gate_script()],  # noqa: S607
        cwd=tree,
        env={**os.environ, "GITHUB_OUTPUT": str(written)},
        check=True,
    )
    answer = written.read_text(encoding="utf-8")
    assert "present=" in answer, f"the gate wrote no answer at all: {answer!r}"
    return "present=true" in answer


APP_SUITE_LAYOUTS: dict[str, str] = {
    "flat under tests/app": "tests/app/test_smoke.py",
    "a subdirectory of tests/app": "tests/app/ui/test_smoke.py",
    "the app marker outside tests/app": "tests/test_app_smoke.py",
}
"""Places an app test can sit that ``pytest -m app`` would collect from.

`testpaths` is `tests`, `python_files` is left at pytest's `test_*.py`, and
`app` is a marker declared for the whole project — so a marked test is in the
app suite wherever under `tests/` it is written. The second entry is the shape
`tests/` mirroring the package produces; the third is what a single wiring test
looks like before anyone makes a directory for it.
"""


def test_the_app_gate_finds_an_app_suite_wherever_pytest_would(
    tmp_path: pathlib.Path,
) -> None:
    """The gate decides whether the app job runs, so it must not be narrower.

    The `app` job is skipped whenever this script answers anything but
    `present=true`, and a skipped job leaves the workflow green. So a suite the
    gate cannot see is a suite that never runs, reported as a pass, for as long
    as nobody reads the job list — the failure the guard was added to prevent,
    arriving through the guard itself.
    """
    empty = tmp_path / "empty"
    (empty / "tests").mkdir(parents=True)
    assert not gate_answers_present(empty), (
        "the gate answered `present=true` for a checkout holding no tests at "
        "all, so it cannot be distinguishing anything"
    )

    unseen = []
    for description, relative in APP_SUITE_LAYOUTS.items():
        tree = tmp_path / relative.replace("/", "-")
        marked = tree / relative
        marked.parent.mkdir(parents=True)
        marked.write_text(
            "import pytest\n\n\n@pytest.mark.app\ndef test_the_app_starts() -> None:\n"
            "    assert True\n",
            encoding="utf-8",
        )
        if not gate_answers_present(tree):
            unseen.append(f"{description} ({relative})")

    assert not unseen, (
        f"`pytest -m app` would collect these and the workflow gate would not, "
        f"so the app job stays skipped and the run stays green: {unseen}. The "
        f"gate has to select the way the job does — on the marker, not on one "
        f"directory's immediate children."
    )
