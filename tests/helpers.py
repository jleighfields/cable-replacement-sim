"""Plain functions shared by the test suite.

Nothing here needs pytest to be readable; fixtures and other pytest machinery
belong in `conftest.py`.
"""

import pathlib
import re

from cablesim import constants

PLAN_PATH: pathlib.Path = constants.PROJECT_ROOT / "PLAN.md"
"""The standing project plan, whose structure the document tests check."""


def headings(text: str) -> list[tuple[int, str]]:
    """Finds the level-2 and level-3 headings outside fenced code blocks.

    Fenced blocks are skipped because the configuration examples contain YAML
    comments, and a comment is one "#" away from looking like a heading.

    Args:
        text: The whole markdown document.

    Returns:
        One `(level, heading text)` pair per heading, in document order.
    """
    found: list[tuple[int, str]] = []
    fenced = False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced:
            match = re.match(r"^(#{2,3}) (.+)$", line)
            if match is not None:
                found.append((len(match.group(1)), match.group(2).strip()))
    return found


def section_numbers(text: str) -> set[str]:
    """Collects the section numbers a document actually defines.

    Args:
        text: The whole markdown document.

    Returns:
        Numbers such as `{"2", "2.11", "10.4"}`, taken from numbered headings.
    """
    numbers: set[str] = set()
    for _, heading in headings(text):
        match = re.match(r"^(\d+(?:\.\d+)?)[.\s]", heading)
        if match is not None:
            numbers.add(match.group(1).rstrip("."))
    return numbers


def section_citations(text: str) -> set[str]:
    """Finds every "Section N" or "§N.M" citation in a document.

    Whitespace is normalized first because a citation that wraps across a line
    break is invisible to a line-by-line scan, which is how two stale
    references once survived a check that found every other one.

    Args:
        text: The whole markdown document.

    Returns:
        The cited section numbers, for example `{"6", "10.4"}`.
    """
    flat = re.sub(r"\s+", " ", text)
    return set(re.findall(r"(?:Section |§)(\d+(?:\.\d+)?)", flat))
