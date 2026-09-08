"""Plain functions shared by the test suite.

Nothing here needs pytest to be readable; fixtures and other pytest machinery
belong in `conftest.py`.
"""

import contextlib
import functools
import importlib.util
import pathlib
import platform
import re
import subprocess
import types
from collections.abc import Iterator

import numpy as np
from cablesim import config, constants, policies, run, simulate

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
    "what chunking the kernel costs": (
        r"costs the kernel ([0-9.]+)x at 12,000 segments and ([0-9.]+)x at "
        r"50,000"
    ),
}
"""Figures the benchmark study quotes in more than one document, by pattern.

Each is a measurement read off the tables in
``docs/compiled-and-threaded-python.md`` and then restated in prose elsewhere,
or — for the cost of chunking the kernel — taken through ``run.run`` and
restated in a module docstring and in the plan. A value written in two places
drifts, and here the drift is silent: every copy stays well-formed prose,
nothing recomputes it, and only a reader who goes back to the tables notices
that two files disagree about what was measured. **Source files are scanned
alongside the documents**, because one of these figures lives in a docstring
and a scan of markdown alone would compare a document against nothing.

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
"""A Weibull scale that puts a *new* segment's remaining life inside year one.

**Pass ``from_new=True`` to ``forced_lifetimes`` with this.** On an aged
population it forces nothing at single precision, for the reason that argument
documents, and a test asserting that everything fails will assert it of a
population where half of it never does.

Randomness is removed through the ordinary ``scale`` and ``replacement_scale``
arrays rather than through an argument only tests pass, so a test using this
exercises the shipped path rather than a branch nothing else reaches.

**Bounded below by what single precision can represent, not by what forces the
outcome.** The hazard is a difference of two ``(age / scale) ** shape`` terms,
and at the horizon's oldest age and the fleet's largest shape that term is
``(90 / scale) ** 6.8``. Single precision tops out at 3.4e38, so a scale under
about 1.9e-4 overflows it, both terms become infinite, and their difference is
a NaN the ranking refuses — while double precision, with room to 1.8e308,
carries the same fixture without noticing. This value clears that bound by a
factor of 5.2 and no more, so lowering it is the change to refuse: two orders
of magnitude below it, most of the shipped population scores NaN at its own age.
It needs no lowering, because 1e-3 puts every remaining life at about a
thousandth of a year already. No population this project generates comes near
it: at the shipped scales the same term peaks at 178.

``tests/test_weibull.py::test_the_forced_scale_stays_inside_single_precision``
recomputes the age, the shape, the bound and the margin from the fleet, because
a margin quoted in prose is what a later edit reads before deciding how far it
may lower this.
"""

NEVER_FAILS = 1e6
"""A scale that puts the first failure hundreds of thousands of years out."""


def assert_at_width(arguments: dict[str, object], precision: str) -> None:
    """Checks that every float array in a call carries the named width.

    An argument builder that misses one produces a call at a width nobody asked
    for, and nothing comparing implementations can see it because they all widen
    together. Every builder that names a precision should end with this.

    Args:
        arguments: A call's arguments.
        precision: The key of ``constants.PRECISIONS`` they should carry.

    Raises:
        AssertionError: If any float array is at another width.
        KeyError: If ``precision`` names no width, which is a caller error
            rather than something about the arguments.
    """
    wanted = constants.PRECISIONS[precision]
    wrong = {
        name: str(value.dtype)
        for name, value in arguments.items()
        if isinstance(value, np.ndarray)
        and value.dtype.kind == "f"
        and value.dtype != wanted
    }
    if wrong:
        # Raised rather than asserted, because a bare `assert` disappears under
        # `python -O` and this is the check that catches a whole width running
        # twice under two names.
        raise AssertionError(
            f"the call was built for {precision} and carries {wrong}; a test "
            f"parametrised on a width it does not produce pins nothing"
        )


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


def alternating_lifetimes(arguments: dict[str, object]) -> dict[str, object]:
    """Copies a call's arguments so even-numbered segments fail and odd ones do not.

    The scenario two parity tests are built on: a year that has both an
    emergency bill and a candidate list, so ``emergency_charged_to_budget``
    decides how far down the second the money reaches. Half the population
    carries ``FAILS_AT_ONCE`` and half ``NEVER_FAILS``, alternating on segment
    identifier so the split is the same whatever the population size.

    Both scales are applied through the ordinary ``scale`` and
    ``replacement_scale`` arrays and then cast to the width ``age0`` carries, so
    the call names one precision rather than two.

    **The failing half starts from new and the surviving half keeps its age.**
    A lifetime is drawn conditional on survival to the current age, and for a
    segment far past its scale the accumulated hazard swamps the draw at single
    precision, so a short scale on an aged segment does not put its remaining
    life inside year one — measured, 273 of 600 failed rather than all 600, and
    the tests built on this describe half the population failing. Zeroing only
    the half meant to fail fixes that without making the survivors ineligible:
    an all-new population is never past an age threshold, and one of the two
    tests here ranks on exactly that.

    Args:
        arguments: The arguments to copy, as the parity fixtures build them.

    Returns:
        A new argument dictionary; the original is untouched.
    """
    n_segments = np.size(arguments["age0"])
    fails = np.arange(n_segments) % 2 == 0
    scales = np.where(fails, FAILS_AT_ONCE, NEVER_FAILS).astype(float)
    return at_call_width(
        {
            **arguments,
            "scale": scales,
            "replacement_scale": scales,
            "age0": np.where(fails, 0.0, np.asarray(arguments["age0"])),
        }
    )


def forced_lifetimes(
    arguments: dict[str, object],
    scale: float,
    replacement: float | None = None,
    from_new: bool = False,
) -> dict[str, object]:
    """Copies a call's arguments with every Weibull scale replaced.

    Args:
        arguments: The arguments to copy.
        scale: What to put in ``scale``.
        replacement: What to put in ``replacement_scale``, or None to use
            ``scale`` for both.
        from_new: Also start every segment at age zero. **Required for
            ``FAILS_AT_ONCE`` to force what its name says at single
            precision**, and harmless at double. A lifetime is drawn
            conditional on survival to the current age, as
            ``scale * ((age / scale) ** shape - ln1p(-u)) ** (1 / shape) -
            age``. For a segment far past its scale the first term inside the
            bracket is enormous — 8e30 at age 57 with a scale of a thousandth
            — and adding the draw to it changes nothing a single-precision
            float can hold, so the remaining life comes back as whatever the
            round trip happens to round to rather than as something inside year
            one. Half the population then never fails, silently, and a test
            named for everything failing asserts against a scenario it did not
            build. At age zero there is no accumulated hazard to swamp the
            draw, and the same scale gives a remaining life of about a
            thousandth of a year at both widths.

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
    forced = {
        **arguments,
        "scale": np.full(segments, scale, dtype=reference.dtype),
        "replacement_scale": np.full(
            segments,
            scale if replacement is None else replacement,
            dtype=reference.dtype,
        ),
    }
    if from_new:
        forced["age0"] = np.zeros(segments, dtype=reference.dtype)
    return forced


@contextlib.contextmanager
def recorded_batches(implementation: str) -> Iterator[list[int]]:
    """Records the replication count of every call to one implementation.

    **What a manifest says about the batch size and what the run did are two
    claims, and reading the manifest checks only one of them.** The resolved
    size is computed in one place and the chunking is done in another, so a run
    can record fifty and call the loop once with the whole thousand, or record a
    thousand and call it twenty times with fifty. Both were tried against the
    suite and neither reddened anything, because every assertion about batching
    went through the manifest.

    This wraps the callable ``run.run`` will reach for and collects the
    ``n_reps`` each call was given, which is the chunking itself rather than a
    record of it.

    Args:
        implementation: The name it is keyed into ``run.RUNNABLE`` under.

    Yields:
        The replication counts, filled in as the calls happen and complete once
        the block exits. Empty until a run is started inside the block.
    """
    called: list[int] = []
    wrapped = run.RUNNABLE[implementation]

    # Carrying the wrapped loop's identity, ``__module__`` above all: ``run.run``
    # reads the Rust build profile from the module the loop was defined in, so a
    # wrapper defined here makes a kernel run record no profile — as ``None``
    # rather than as a refusal, which is the shape that goes unnoticed. Every
    # manifest written inside this block would then be missing the one field a
    # timing has to be read against.
    @functools.wraps(wrapped)
    def recording(*, n_reps: int, **arguments: object) -> simulate.Results:
        called.append(n_reps)
        return wrapped(n_reps=n_reps, **arguments)

    run.RUNNABLE[implementation] = recording
    try:
        yield called
    finally:
        # Restored however the block exits, since the registry is module state
        # shared by every test in the session and a spy left in it would make
        # whichever test ran next depend on the order it ran in.
        run.RUNNABLE[implementation] = wrapped


def script(name: str) -> types.ModuleType:
    """Loads one of the driver scripts as a module.

    ``scripts/`` is not part of the importable package, so the file is loaded
    by path. Reading it this way is what lets a test drive a script's argument
    handling without starting a subprocess.

    Args:
        name: The file's stem, without the extension.

    Returns:
        The loaded module, whose ``main`` takes an argument list.

    Raises:
        ImportError: If the file could not be loaded as a module.
    """
    path = constants.PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} could not be loaded as a module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def total_order(values: np.ndarray) -> np.ndarray:
    """Maps floats onto integers that sort the way the floats do.

    Two floats one representable step apart differ by one here, whatever their
    exponent, which is what makes "how far apart" a meaningful question across
    a sample spanning several orders of magnitude. Subtracting the raw bit
    patterns does not give that across the sign boundary, because a negative
    float's pattern grows as the value falls.

    Args:
        values: Any floating array, read through its bit pattern.

    Returns:
        One integer per value, monotone in the value, with both zeros at 0.
    """
    width = np.dtype(f"int{values.dtype.itemsize * 8}")
    bits = values.view(width).astype(np.int64)
    lowest = np.int64(-(2 ** (values.dtype.itemsize * 8 - 1)))
    return np.where(bits < 0, lowest - bits, bits)


def steps_apart(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Counts representable steps between two arrays, elementwise.

    Args:
        left: One array.
        right: The other, same dtype and shape.

    Returns:
        How many representable values separate each pair; 0 where equal.
    """
    return np.abs(total_order(left) - total_order(right))


def disagreement_report(
    name: str, inputs: np.ndarray, from_numpy: np.ndarray, from_library: np.ndarray
) -> str:
    """Describes where and by how much two implementations differ.

    Written for the failure message of the single-precision agreement check,
    whose whole difficulty is that "they disagree" does not say whether the
    disagreement matters. What decides that is **which inputs disagree**: a
    difference at an input the annual loop never reaches is a property of the
    sample, and one at an input it reaches every year is a property of the
    model. So the offending inputs are reported, not just the count.

    Args:
        name: The function being compared, for the report's first line.
        inputs: What each result was computed from.
        from_numpy: NumPy's results.
        from_library: The C library's results, same dtype.

    Returns:
        A multi-line report, or a line saying they agree everywhere.
    """
    steps = steps_apart(from_numpy, from_library)
    differing = np.flatnonzero(steps)
    if differing.size == 0:
        return f"{name}: agrees on all {inputs.size:,} inputs"

    offending = inputs[differing]
    lines = [
        f"{name}: {differing.size:,} of {inputs.size:,} inputs disagree, "
        f"by at most {int(steps.max())} representable step(s)",
        f"{name}: those inputs span {offending.min():.6g} to "
        f"{offending.max():.6g}, within a sample spanning "
        f"{inputs.min():.6g} to {inputs.max():.6g}",
    ]
    # A handful is enough to recognise the values; the span above is what says
    # whether they cluster.
    for index in differing[:5]:
        # Nine significant digits, which is what a single-precision value needs
        # to be read back as itself.
        lines.append(
            f"{name}:   f({float(inputs[index]):.9g}) = "
            f"{float(from_numpy[index]):.9g} from NumPy, "
            f"{float(from_library[index]):.9g} from the C library"
        )
    return "\n".join(lines)


CPUINFO_PATH: pathlib.Path = pathlib.Path("/proc/cpuinfo")
"""Where Linux publishes the processor this run is on."""


def platform_fingerprint(cpuinfo: pathlib.Path = CPUINFO_PATH) -> list[str]:
    """What a run needs to record to explain a floating-point disagreement.

    Reported on every run rather than only on a failing one. A fingerprint from
    the machine that disagrees says nothing on its own — what identifies the
    difference is comparing it against the machines that agree, and those are
    the runs that passed.

    Each fact that cannot be read says so rather than being left out, since a
    missing line and an absent feature would otherwise look the same in a log.

    Args:
        cpuinfo: Where to read the processor from. An argument only so that a
            test can reach the branch taken where the file is absent, which on
            the platform this runs on it otherwise never would.

    Returns:
        One line per fact, for a test report header.
    """
    missing = f"unreadable ({cpuinfo} is not present)"
    model = f"cpu model: {missing}"
    flags = f"cpu features: {missing}"
    if cpuinfo.exists():
        text = cpuinfo.read_text()
        named = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
        model = f"cpu model: {named.group(1).strip() if named else 'not named'}"
        listed = re.search(r"^flags\s*:\s*(.+)$", text, re.MULTILINE)
        present = set(listed.group(1).split()) if listed else set()
        # The features that decide which kernel a vectorised or dispatched
        # implementation runs, on either side of the comparison.
        watched = ("avx", "avx2", "fma", "avx512f", "avx512dq", "avx512vl")
        flags = "cpu features: " + " ".join(
            f"{feature}={'yes' if feature in present else 'no'}"
            for feature in watched
        )

    dispatch = numpy_dispatch()
    return [
        model,
        flags,
        f"libc: {' '.join(platform.libc_ver()) or 'unknown'}",
        f"numpy: {np.__version__}, baseline {dispatch['baseline']}, "
        f"enabled {dispatch['enabled'] or 'none beyond baseline'}",
    ]


def numpy_dispatch() -> dict[str, str]:
    """Which vectorised code paths this NumPy build is allowed to take.

    NumPy compiles several instruction-set variants of a loop and picks one at
    import from what the processor reports. That choice is the first candidate
    explanation for two machines computing different bits from one build, so it
    is recorded beside the processor rather than inferred from it.

    Returns:
        The always-compiled baseline and the dispatched sets this machine
        enabled, each as a space-separated string, or a note where this NumPy
        does not expose them.
    """
    umath = getattr(np._core, "_multiarray_umath", None)
    baseline = getattr(umath, "__cpu_baseline__", None)
    dispatch = getattr(umath, "__cpu_dispatch__", None)
    features = getattr(umath, "__cpu_features__", None)
    if baseline is None or dispatch is None or features is None:
        return {"baseline": "not exposed by this numpy", "enabled": ""}
    return {
        "baseline": " ".join(baseline),
        "enabled": " ".join(name for name in dispatch if features.get(name)),
    }
