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
import tempfile

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
    # Somewhere other than the directory being inspected. The runner points
    # `GITHUB_OUTPUT` at a file outside the checkout, and writing it inside
    # `tree` would leave one in the repository root the moment this is asked
    # about the real one.
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w", suffix=".github_output", delete=False
    )
    handle.close()
    written = pathlib.Path(handle.name)
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
    written.unlink()
    assert "present=" in answer, f"the gate wrote no answer at all: {answer!r}"
    return "present=true" in answer


APP_SUITE_LAYOUTS: dict[str, str] = {
    "flat under tests/app": "tests/app/test_smoke.py",
    "a subdirectory of tests/app": "tests/app/ui/test_smoke.py",
}
"""Places an app test can sit that the gate has to see.

The second is the shape `tests/` mirroring the package produces, and it is why
the gate searches at any depth rather than globbing one directory's immediate
children.
"""


def collected_app_tests() -> list[str]:
    """What ``pytest -m app`` actually selects in this repository.

    Asks pytest rather than reading the source for the marker. A search of the
    text finds the marker wherever it is written, including inside a string
    literal in a test that builds fixtures containing it — which is how the
    gate came to answer "there are app tests" for a repository with none.

    Returns:
        The node id of every test the ``app`` marker selects, empty when the
        suite does not exist yet.
    """
    found = subprocess.run(  # noqa: S603
        ["uv", "run", "pytest", "-m", "app", "--collect-only", "-q"],  # noqa: S607
        cwd=constants.PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return [line for line in found.stdout.splitlines() if "::" in line]


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
        f"so the app job stays skipped and the run stays green: {unseen}"
    )


def test_the_gate_answers_this_repository_the_way_pytest_does() -> None:
    """Run against the repository it actually gates, not only against fixtures.

    The gate's other test builds trees under a temporary directory, and a gate
    can be right about every one of those and wrong here. It was: a search for
    the marker text matched this file, which writes the marker into the
    fixtures above, so the gate answered `present=true` for a repository with
    no app tests — the job ran, `pytest -m app` selected nothing, exited 5, and
    the scheduled workflow failed on every push to the default branch.

    Comparing the gate against pytest on this checkout is the check that was
    missing, and it is the one that cannot be satisfied by a gate that is right
    about hypothetical trees alone.
    """
    collected = collected_app_tests()
    present = gate_answers_present(constants.PROJECT_ROOT)

    assert present == bool(collected), (
        f"the gate says the app suite is "
        f"{'present' if present else 'absent'} and `pytest -m app` collects "
        f"{len(collected)} tests, so the app job "
        f"{'runs with nothing to do' if present else 'skips a suite that exists'}"
    )


def test_no_app_test_hides_where_the_gate_cannot_see_it() -> None:
    """The convention the gate rests on: app tests live under ``tests/app``.

    A cheap gate cannot reproduce pytest's collection, so it reads a convention
    instead — and a convention nothing enforces is one the next person breaks
    without hearing about it. `pytest -m app` would collect a marked test
    anywhere under `tests/`; the gate only looks under `tests/app`; this is
    what keeps those two answers the same.
    """
    stray = [
        node
        for node in collected_app_tests()
        if not node.startswith("tests/app/")
    ]

    assert not stray, (
        f"these carry the `app` marker outside `tests/app/`, where the "
        f"workflow gate cannot see them, so the app job would skip while they "
        f"exist: {stray}"
    )
