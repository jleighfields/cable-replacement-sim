"""Executes every marimo notebook headless, so they cannot silently rot.

A notebook breaks when the package interface it imports changes, and nothing
else notices. These run in minutes rather than seconds, so they carry the
`notebooks` marker and are excluded from a default run; the workflow that
executes them runs on pushes to the default branch and weekly.

The notebooks carry their own assertions about what they compute. This module
only asserts that they run at all.
"""

import pathlib
import subprocess
import sys

import pytest
from cablesim import constants

NOTEBOOKS: list[pathlib.Path] = sorted(
    (constants.PROJECT_ROOT / "notebooks").glob("[0-9]*.py")
)


def test_there_are_notebooks_to_run() -> None:
    """The glob found something.

    A scan that matches nothing passes every test below it by doing nothing,
    which is exactly how a suite reports green on an empty directory.
    """
    assert NOTEBOOKS, "no notebooks matched; the parametrisation below is empty"


@pytest.mark.notebooks
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_runs_headless(notebook: pathlib.Path) -> None:
    """The notebook executes end to end and its own assertions hold.

    Args:
        notebook: The marimo notebook to execute.
    """
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(notebook)],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )

    assert result.returncode == 0, f"{notebook.name} failed:\n{result.stderr[-3000:]}"
