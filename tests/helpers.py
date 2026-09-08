"""Plain functions shared by the test suite.

Nothing here needs pytest to be readable; fixtures and other pytest machinery
belong in `conftest.py`.
"""

import pathlib
import re
import subprocess

import numpy as np
from cablesim import config, constants, policies, simulate

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
    prefixed = re.findall(r"(?:Section |§)(\d+(?:\.\d+)?)", flat)
    # The document cites its own subsections far more often in the bare form
    # "2.9, Annual simulation loop" than with a prefix, so an extractor that
    # matches only the prefixed spellings leaves most references unchecked.
    bare = re.findall(r"(?<![\w.])(\d+\.\d+), [A-Z]", flat)
    return set(prefixed) | set(bare)


def plan_citations(text: str) -> list[str]:
    """Finds citations of a numbered section of the standing plan.

    Matches the spellings the project has actually used — ``PLAN.md`` followed
    by a section number, with or without a "section" or "§" between them.

    Args:
        text: The file contents to scan.

    Returns:
        Every citation found, in document order.
    """
    pattern = r"PLAN\.md`?[^.\n]{0,12}?(?:§|[Ss]ection\s*)\d+(?:\.\d+)*"
    return re.findall(pattern, text)


def phase_citations(text: str) -> list[str]:
    """Finds citations of a numbered phase of the standing plan's roadmap.

    The roadmap renumbers whenever a phase is inserted, exactly as the numbered
    sections do, so a phase number quoted anywhere else goes stale the same way
    and with nothing to report it. Matched case-insensitively because prose
    writes both "Phase 6" and "phase 6".

    Args:
        text: The file contents to scan.

    Returns:
        Every citation found, in document order.
    """
    return re.findall(r"[Pp]hase [0-9]+", text)


def tracked_documents(suffix: str = ".md") -> list[pathlib.Path]:
    """Every tracked document, minus the finished plans under ``tasks/``.

    Asks git what is tracked rather than globbing, so a document added in a new
    directory is covered the day it is added. ``tasks/`` is excluded because a
    finished plan is a dated record of what was done: editing it to track a
    figure that has since been retaken would falsify the record rather than
    repair it.

    Args:
        suffix: The file extension to keep.

    Returns:
        Absolute paths, in the order git lists them.
    """
    root = PLAN_PATH.parent
    listed = subprocess.run(  # noqa: S603
        # Resolved from PATH, the same way `results.git_provenance` invokes it.
        ["git", "ls-files", "-z"],  # noqa: S607
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    return [
        root / name
        for name in listed
        if name.endswith(suffix)
        and not name.startswith("tasks/")
        and (root / name).is_file()
    ]


REPEATED_FIGURES: dict[str, str] = {
    "what threads are worth": r"threads[^.]{0,10}?([0-9.]+)[-\u2013\u2014]([0-9.]+)x",
    "what compiling is worth": (
        r"compiling[^.]{0,10}?([0-9.]+)[-\u2013\u2014]([0-9.]+)x"
    ),
    "what the language is worth": (
        r"language[^.]{0,10}?([0-9.]+)[-\u2013\u2014]([0-9.]+)x"
    ),
    "how many of the six columns threading NumPy was slower in": (
        r"([a-z]+) of (?:the )?six columns"
    ),
}
"""Figures the benchmark study quotes in more than one document, by pattern.

Each is a measurement read off the tables in
``docs/compiled-and-threaded-python.md`` and then restated in prose elsewhere.
A value written in two places drifts, and here the drift is silent: every copy
stays well-formed markdown, and only a reader who recomputes the ratio notices
that two documents disagree about what was measured.

The patterns capture the number rather than matching a literal, so correcting a
figure needs no edit here — what is asserted is that the copies agree, not what
they say.
"""


def quoted_figures(text: str, pattern: str) -> set[tuple[str, ...]]:
    """Every value one document quotes for a repeated figure.

    Whitespace is normalized first because these documents hard-wrap near 80
    columns, so a label and the number it introduces routinely land on
    different lines — "threads" ending one and "8-36x" beginning the next.
    A scan that did not normalize would miss exactly those and report clean.

    Args:
        text: The whole markdown document.
        pattern: One of the patterns in ``REPEATED_FIGURES``.

    Returns:
        One tuple per distinct value quoted, empty if the document is silent
        on this figure.
    """
    flat = re.sub(r"\s+", " ", text)
    return {
        found if isinstance(found, tuple) else (found,)
        for found in re.findall(pattern, flat)
    }


CALLERS_THAT_NAME_IMPLEMENTATIONS: tuple[pathlib.Path, ...] = (
    constants.PROJECT_ROOT / "scripts" / "run_benchmarks.py",
    constants.PROJECT_ROOT / "notebooks" / "05_parity_and_bench.py",
)
"""Files that build a benchmark table by naming implementations in source.

Neither is reached by a default test run — one is a driver script and the other
a marimo notebook behind the ``notebooks`` marker — so a name that stopped
existing goes unreported in both until someone runs them by hand.
"""


def requested_implementations(path: pathlib.Path) -> set[str]:
    """Reads the implementation names a file asks the benchmark harness for.

    Matches the literal first argument of ``benchmarks.Configuration``, which
    is how both callers spell the request. Reading the source rather than
    importing it is what lets a marimo notebook be checked without executing
    its cells, which costs minutes.

    Args:
        path: The file to read.

    Returns:
        Every implementation name it names, or an empty set if it names none.

    Raises:
        FileNotFoundError: If the path does not exist. An empty result would
            otherwise report a moved file as a file that requests nothing.
    """
    return set(
        re.findall(
            r"""Configuration\(\s*["']([^"']+)["']""",
            path.read_text(encoding="utf-8"),
        )
    )


def resolved(name: str, **params: float | str) -> policies.Resolved:
    """Validates a policy and reduces it to what the annual loop reads.

    Going through the schema rather than building the NamedTuple directly is
    what makes a test's policy the same object a run's policy is, including the
    neutral threshold values that decide eligibility.

    Args:
        name: The policy name.
        **params: Policy parameters.

    Returns:
        The resolved policy.
    """
    return policies.resolve(config.PolicySpec(name=name, params=params))


def first_segments(
    arguments: dict[str, object], n_segments: int
) -> dict[str, object]:
    """Copies one call's arguments down to its first ``n_segments`` segments.

    A segment's array position is its identifier, so a prefix is still a valid
    population: the identifiers stay 0 upwards and the tie-break still means
    what it meant. This is what lets a test that needs a handful of segments
    reuse the fixture's population instead of building a second one that would
    drift away from it.

    Nothing has to be done about the draws. They are computed from the position
    they sit at rather than handed over as an array, so keeping the first
    ``n_segments`` segments keeps exactly the draws those segments would have
    had in the larger population — which is what makes a cut-down fixture a
    smaller version of the same run rather than a different one.

    Args:
        arguments: The arguments to copy, as the parity fixtures build them.
        n_segments: How many segments to keep, counting from segment 0.

    Returns:
        A new argument dictionary; the original is untouched.
    """
    kept = {name: arguments[name][:n_segments] for name in simulate.SEGMENT_ARGUMENTS}
    return {**arguments, **kept}


FAILS_AT_ONCE = 1e-3
"""A Weibull scale that puts every segment's remaining life inside year one.

Randomness is removed through the ordinary ``scale`` and ``replacement_scale``
arrays rather than through an argument only tests pass, so a test using this
exercises the shipped path rather than a branch nothing else reaches.

**Bounded below by what single precision can represent, not by what forces the
outcome.** The hazard is a difference of two ``(age / scale) ** shape`` terms,
and at the horizon's oldest age and the fleet's largest shape that term is
``(120 / scale) ** 6.5``. Single precision tops out at 3.4e38, so a scale under
about 1.4e-4 overflows it, both terms become infinite, and their difference is
a NaN the ranking refuses — while double precision, with room to 1.8e308,
carries the same fixture without noticing. This value is three orders of
magnitude inside that bound and still puts every remaining life at about a
thousandth of a year. No population this project generates comes near it: the
shipped scales are decades, where the same term is around a thousand.
"""

NEVER_FAILS = 1e6
"""A scale that puts the first failure hundreds of thousands of years out."""


def at_call_width(arguments: dict[str, object]) -> dict[str, object]:
    """Casts every float array in a call to the width its population carries.

    A test that overrides an argument builds the replacement at NumPy's default
    width, and an implementation reads the run's precision off the dtype of what
    it is handed — so a call mixing widths is a call at neither. The kernel's
    binding refuses one outright, naming the array; a Python implementation
    would widen everything the wider array touched and go on agreeing with the
    others, which is the failure that has no symptom.

    Args:
        arguments: A call's arguments, some of them possibly rebuilt.

    Returns:
        The same arguments with every float array at the width ``age0`` carries.
    """
    width = np.asarray(arguments["age0"]).dtype
    return {
        name: (
            value.astype(width)
            if isinstance(value, np.ndarray) and value.dtype.kind == "f"
            else value
        )
        for name, value in arguments.items()
    }


def forced_lifetimes(
    arguments: dict[str, object], scale: float, replacement: float | None = None
) -> dict[str, object]:
    """Copies a call's arguments with every Weibull scale replaced.

    Args:
        arguments: The arguments to copy.
        scale: What to put in ``scale``.
        replacement: What to put in ``replacement_scale``, or None to use
            ``scale`` for both.

    Returns:
        A new argument dictionary; the original is untouched, which matters
        because the fixture it usually comes from is shared by every test in
        the session.
    """
    # At the width the rest of the call carries. An implementation reads the
    # precision off the dtype of the arrays it is handed, so a replacement
    # array built at the default would hand it two widths at once — which the
    # kernel's binding refuses outright rather than converting, and which would
    # make a Python implementation quietly widen everything it touched.
    reference = np.asarray(arguments["age0"])
    segments = reference.shape
    return {
        **arguments,
        "scale": np.full(segments, scale, dtype=reference.dtype),
        "replacement_scale": np.full(
            segments,
            scale if replacement is None else replacement,
            dtype=reference.dtype,
        ),
    }
