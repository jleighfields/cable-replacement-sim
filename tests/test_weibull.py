"""Analytical checks on the Weibull forms.

These compare against the closed form rather than against another
implementation, so they build their own draws at their own sizes and seeds.
An analytical check is worth more than a parity check because it can be wrong
in only one way.
"""

import ctypes
import ctypes.util
import pathlib
import re

import numpy as np
import pytest
from cablesim import config, constants, population, random_draws, run, weibull
from numpy.random import SeedSequence
from scipy import stats

from tests import helpers

KS_DRAWS: int = 100_000
KS_ALPHA: float = 0.001
"""Sample size and significance chosen together.

The size gives power to detect an exponent wrong in its second decimal; the
small alpha keeps a correct implementation from failing by chance. The upper
bound on size is runtime, not sensitivity — floating-point differences perturb
the distribution far below the rejection threshold at any size used here.
"""


def weibull_draws(
    source: SeedSequence, shape: float, scale: float, n: int
) -> np.ndarray:
    """Draws lifetimes from a Weibull by inverting its survivor function.

    Args:
        source: Seed sequence to draw from.
        shape: Weibull shape parameter.
        scale: Weibull scale parameter.
        n: How many lifetimes to draw.

    Returns:
        Ages at failure, in years.
    """
    return weibull.draw_lifetime(random_draws.uniforms(source, n), shape, scale)


def test_numpy_and_the_system_library_agree_in_single_precision() -> None:
    """The assumption single precision rests on, and the reason it holds.

    Two implementations of a single-precision ``expm1``, ``log1p`` or ``pow``
    may legitimately differ in the last bit — none of them is required to be
    correctly rounded. This project compares implementations at no tolerance, so
    a single-precision run is only possible while NumPy and the crate produce
    the same bits, and they do because **both reach the same C library**.

    The second half is what makes this a test rather than a restatement. The
    other available explanation — that each computes in double and rounds once —
    is ruled out, not merely unnecessary: over this sample it gives a different
    answer for 2,064 of 20,000 `expm1` inputs and 1,478 of 20,000 `log1p`, but
    for only **15 of 20,000** on `pow`. That last margin is what the `pow` arm
    turns on, so narrowing its range or shrinking the sample can disarm that arm
    while the other two go on passing. Without it, a NumPy that grew its own
    vectorised single-precision loops would pass the first assertion by
    accident only until it did not.

    This is a property of the platform, so this is where a build against a
    different C library would report rather than the parity suite failing
    somewhere less obvious. **The failure carries which inputs disagreed**,
    because "they disagree" does not say whether it matters: a difference at an
    input the annual loop never reaches is a fact about this sample, and one at
    an input it reaches every year is a fact about the model. The machine is in
    the run header, printed whether this passes or fails, since a fingerprint
    from the machine that disagreed means nothing without the ones that did
    not.
    """
    library = ctypes.CDLL(ctypes.util.find_library("m"))
    for name in ("expm1f", "log1pf"):
        getattr(library, name).restype = ctypes.c_float
        getattr(library, name).argtypes = [ctypes.c_float]
    library.powf.restype = ctypes.c_float
    library.powf.argtypes = [ctypes.c_float, ctypes.c_float]

    generator = np.random.default_rng(0)
    # Each over the range the annual loop actually reaches it in: the hazard
    # exponent over a plausible age-to-scale ratio, the lifetime draw over the
    # open unit interval a uniform lands in.
    hazard = generator.uniform(-3.0, 3.0, 20_000).astype(np.float32)
    uniform = generator.uniform(0.01, 0.99, 20_000).astype(np.float32)
    ratio = generator.uniform(0.1, 50.0, 20_000).astype(np.float32)
    exponent = np.float32(6.5)

    # Each entry carries the argument as well as the three results, so a
    # failure can name the inputs that disagreed rather than only the count.
    checks = {
        "expm1": (
            hazard,
            np.expm1(hazard),
            [library.expm1f(value) for value in hazard],
            np.expm1(hazard.astype(np.float64)),
        ),
        "log1p": (
            -uniform,
            np.log1p(-uniform),
            [library.log1pf(-value) for value in uniform],
            np.log1p(-uniform.astype(np.float64)),
        ),
        "pow": (
            ratio,
            ratio**exponent,
            [library.powf(value, 6.5) for value in ratio],
            ratio.astype(np.float64) ** 6.5,
        ),
    }
    for name, (inputs, from_numpy, from_library, in_double) in checks.items():
        from_c = np.array(from_library, dtype=np.float32)
        assert np.array_equal(from_numpy, from_c), (
            f"NumPy and the system C library disagree on single-precision "
            f"{name}, so the crate and NumPy would compute different numbers "
            f"and no run at that precision could be compared against another "
            f"implementation\n"
            + helpers.disagreement_report(name, inputs, from_numpy, from_c)
        )
        assert not np.array_equal(from_numpy, in_double.astype(np.float32)), (
            f"for {name}, computing in double and rounding once gives the same "
            f"answer over this sample, so this test no longer distinguishes the "
            f"two explanations and would pass for an implementation that does "
            f"not share the C library"
        )


def test_min_of_n_matches_the_reduced_scale() -> None:
    """The minimum of three conductor lifetimes is Weibull at the reduced scale.

    This is the closed form the whole three-phase treatment rests on, and it is
    checked against sampled minima rather than assumed.
    """
    shape, scale, n = 2.4, 55.0, KS_DRAWS
    source = random_draws.spawn_sources(11).population
    draws = weibull.draw_lifetime(
        random_draws.uniforms(source, 3 * n).reshape(3, n), shape, scale
    )

    reduced = weibull.effective_scale(
        np.array(shape), np.array(scale), np.array(3.0), np.array(500.0), 500.0, 1.0
    )
    result = stats.kstest(
        draws.min(axis=0), "weibull_min", args=(shape, 0.0, float(reduced))
    )

    assert result.pvalue > KS_ALPHA, f"min of 3 is not Weibull({shape}, {reduced})"


def test_length_enters_through_the_exponent() -> None:
    """A longer segment fails sooner, by the configured power of its length.

    With an exponent of 1 this is the spatial-Poisson case; the point of the
    check is that whatever exponent is configured is the one applied.
    """
    shape, scale, n = 2.4, 55.0, KS_DRAWS
    length, reference, exponent = 2000.0, 500.0, 0.5
    reduced = weibull.effective_scale(
        np.array(shape),
        np.array(scale),
        np.array(1.0),
        np.array(length),
        reference,
        exponent,
    )
    expected = scale * ((length / reference) ** exponent) ** (-1.0 / shape)
    assert reduced == np.float64(expected)

    source = random_draws.spawn_sources(12).population
    draws = weibull_draws(source, shape, float(reduced), n)
    result = stats.kstest(draws, "weibull_min", args=(shape, 0.0, float(reduced)))
    assert result.pvalue > KS_ALPHA


def test_annual_probability_telescopes_to_the_survival_curve() -> None:
    """Surviving every year in turn equals surviving the horizon.

    Deterministic, so it is asserted exactly rather than statistically: the
    product of ``1 - p(t)`` over the horizon is ``S(30)``.
    """
    shape, scale, horizon = 2.2, 50.0, 30
    ages = np.arange(horizon, dtype=float)

    survived = np.prod(
        1.0 - weibull.conditional_failure_probability(ages, shape, scale)
    )

    closed_form = np.exp(-((horizon / scale) ** shape))
    assert (
        survived == np.float64(closed_form) or abs(survived / closed_form - 1.0) < 1e-12
    )


def test_left_truncated_draw_matches_conditional_survival() -> None:
    """Remaining life given survival to an age follows ``S(t)/S(a)``.

    Checked at three entry ages spanning the range, because the correction's
    error grows with entry age and a single young cohort passes even with the
    correction missing altogether.
    """
    shape, scale, n = 2.2, 50.0, KS_DRAWS
    for index, entry_age in enumerate((5.0, 30.0, 70.0)):
        source = random_draws.spawn_sources(20 + index).population
        remaining = weibull.draw_remaining_life(
            random_draws.uniforms(source, n), entry_age, shape, scale
        )
        total = entry_age + remaining

        # Conditioning on survival to `entry_age` is a truncation of the same
        # Weibull, so the truncated distribution is the reference.
        reference = stats.weibull_min(shape, scale=scale)
        expected = (reference.sf(total) / reference.sf(entry_age)).clip(0.0, 1.0)
        result = stats.kstest(1.0 - expected, "uniform")

        assert result.pvalue > KS_ALPHA, f"entry age {entry_age} is not conditional"
        assert (remaining > 0).all()


def test_a_censored_episode_of_zero_age_does_not_poison_the_likelihood() -> None:
    """Zero exposure contributes nothing rather than making the sum undefined.

    The failure term is multiplied by whether the episode failed, so a censored
    row should contribute none of it. The logarithm inside is still evaluated,
    though, and at zero age it is negative infinity: multiplied by zero that is
    `nan`, not zero, and one such row makes the whole log-likelihood undefined.
    The fit downstream then fails naming nothing that points here.

    Not reachable through `records.lifetimes`, where a censored episode is
    measured to the study end and so has positive age. `log_likelihood` is
    public and takes arrays directly, which is what makes it worth pinning.
    """
    age = np.array([0.0, 10.0, 20.0])
    entry = np.zeros(3)

    censored = weibull.log_likelihood(6.0, 50.0, age, entry, np.array([0.0, 1.0, 1.0]))
    assert np.isfinite(censored)
    # The zero-age row is the only difference, and it contributes nothing, so
    # dropping it must leave the same value.
    assert censored == pytest.approx(
        weibull.log_likelihood(6.0, 50.0, age[1:], entry[1:], np.array([1.0, 1.0]))
    )

    # An observed failure at age zero is a degenerate lifetime rather than a
    # rounding artefact, and must still be refused rather than smoothed over.
    with np.errstate(divide="ignore"):
        degenerate = weibull.log_likelihood(
            6.0, 50.0, age, entry, np.array([1.0, 1.0, 1.0])
        )
    assert not np.isfinite(degenerate)


FORCED_SCALE_BOUND = re.compile(
    r"that term is ``\((?P<age>[0-9.]+) / scale\) \*\* (?P<shape>[0-9.]+)``\."
    r".*?a scale under about (?P<bound>[0-9.e-]+) overflows it"
    r".*?clears that bound by a factor of (?P<margin>[0-9.]+)"
    r".*?the same term peaks at (?P<peak>[0-9.]+)",
    re.DOTALL,
)
"""The arithmetic ``helpers.FAILS_AT_ONCE`` quotes for its own lower bound.

Read out of the docstring rather than restated here, so this fails when the
prose and the fleet disagree rather than when someone forgets a second copy —
the same arrangement ``test_parity.py`` uses for the premium comment in
``src/policies.rs``.
"""


def test_the_forced_scale_stays_inside_single_precision() -> None:
    """The margin ``helpers.FAILS_AT_ONCE`` quotes is the one the fleet gives.

    That constant is a Weibull scale small enough to force a failure inside
    year one, and how small it may go is bounded by single precision rather
    than by the outcome: the hazard is a difference of two ``(age / scale) **
    shape`` terms, and past the ceiling both terms are infinite and their
    difference is a NaN the ranking refuses. The docstring states that bound by
    naming an age and a shape, then quotes how far the constant sits above it.

    Both inputs are facts about the shipped configuration, so both can go stale
    without anything reporting it — a vintage added to the class table moves
    the largest shape, and a longer horizon moves the oldest age. A margin
    quoted in prose is what a later edit reads before deciding how far it may
    lower the constant, and this is the only thing that reads it back.

    The age is the oldest the hazard is evaluated at: a segment never replaced
    reaches ``max(age0) + n_years``, since the year's hazard reads ``age + 1``.
    The shape is the largest any segment carries, in the ground or after
    replacement. Pairing the two overstates the bound, because no one segment
    has both — which is the direction a lower bound should err in.
    """
    settings = config.load_config(constants.DEFAULT_CONFIG_PATH)
    segments = run.segment_arrays(
        population.generate(settings), constants.DEFAULT_PRECISION
    )
    oldest = float(segments["age0"].max()) + settings.simulation.n_years
    largest_shape = float(
        max(segments["shape"].max(), segments["replacement_shape"].max())
    )
    # The scale at which ``(oldest / scale) ** largest_shape`` reaches the
    # single-precision ceiling; anything smaller overflows it.
    bound = oldest / float(np.finfo(np.float32).max) ** (1.0 / largest_shape)

    source = (constants.PROJECT_ROOT / "tests" / "helpers.py").read_text(
        encoding="utf-8"
    )
    quoted = FORCED_SCALE_BOUND.search(" ".join(source.split()).replace("` `", "``"))
    assert quoted is not None, (
        "tests/helpers.py no longer states FAILS_AT_ONCE's lower bound in the "
        "form this reads; either the docstring was reworded, in which case "
        "update this pattern, or the bound was dropped"
    )

    assert (float(quoted["age"]), float(quoted["shape"])) == (
        oldest,
        largest_shape,
    ), (
        f"FAILS_AT_ONCE's docstring bounds itself with "
        f"({quoted['age']} / scale) ** {quoted['shape']}, but the shipped "
        f"fleet's oldest evaluated age is {oldest:g} and its largest shape is "
        f"{largest_shape:g}; the bound it derives is not this fleet's"
    )
    assert float(quoted["bound"]) == pytest.approx(bound, rel=0.05), (
        f"the docstring says a scale under {quoted['bound']} overflows single "
        f"precision; from this fleet the bound is {bound:.3g}"
    )
    assert float(quoted["margin"]) == pytest.approx(
        helpers.FAILS_AT_ONCE / bound, rel=0.05
    ), (
        f"the docstring says {helpers.FAILS_AT_ONCE:g} clears that bound by "
        f"{quoted['margin']}x; it clears {bound:.3g} by "
        f"{helpers.FAILS_AT_ONCE / bound:.3g}x"
    )


    # Per segment, each against the oldest age it reaches, which is the term
    # the model computes — not every segment paired with the fleet's oldest.
    reached = segments["age0"] + settings.simulation.n_years
    peak = float(((reached / segments["scale"]) ** segments["shape"]).max())
    assert float(quoted["peak"]) == pytest.approx(peak, rel=0.05), (
        f"the docstring says the hazard term peaks at {quoted['peak']} at the "
        f"shipped scales; over this fleet it peaks at {peak:.3g}"
    )

    # The bound is a claim about arithmetic, so it is run as well as compared.
    # At the constant the hazard is finite everywhere; two orders of magnitude
    # below it, it is not, which is the refusal the docstring asks for. One
    # order is not enough, because the bound pairs an age and a shape no single
    # segment has — the oldest segments carry neither of the largest shapes.
    narrow = np.float32
    at_own_age = segments["age0"].astype(narrow)
    shapes = segments["shape"].astype(narrow)
    with np.errstate(over="ignore", invalid="ignore"):
        finite = weibull.conditional_failure_probability(
            at_own_age, shapes, narrow(helpers.FAILS_AT_ONCE)
        )
        overflowed = weibull.conditional_failure_probability(
            at_own_age, shapes, narrow(helpers.FAILS_AT_ONCE / 100.0)
        )
    assert np.isfinite(finite).all(), (
        f"{helpers.FAILS_AT_ONCE:g} already overflows single precision "
        f"somewhere in this fleet, so it sits below its own documented bound"
    )
    assert np.isnan(overflowed).mean() > 0.5, (
        f"two orders of magnitude below {helpers.FAILS_AT_ONCE:g} leaves "
        f"{np.isnan(overflowed).mean():.0%} of the fleet at NaN; the docstring "
        f"says most of it, which is the reason it gives for not lowering this"
    )


def test_representable_steps_are_counted_across_the_sign_boundary() -> None:
    """Subtracting raw bit patterns is wrong either side of zero.

    A negative float's bit pattern grows as the value falls, so the two
    smallest numbers either side of zero — one representable step from it
    apiece — sit at opposite ends of the integer range. Read naively they are
    two thousand million steps apart rather than two, which would turn every
    report of a sign-straddling disagreement into a number that means nothing.
    """
    below = np.nextafter(np.float32(0.0), np.float32(-1.0))
    above = np.nextafter(np.float32(0.0), np.float32(1.0))

    assert helpers.steps_apart(np.array([below]), np.array([above]))[0] == 2
    assert helpers.steps_apart(np.array([above]), np.array([above]))[0] == 0
    # The two zeros are one value as far as this is concerned, and their bit
    # patterns are as far apart as the representation allows.
    assert helpers.steps_apart(
        np.array([np.float32(-0.0)]), np.array([np.float32(0.0)])
    )[0] == 0


def test_the_disagreement_report_names_the_inputs_that_disagreed() -> None:
    """The count alone cannot say whether a disagreement matters.

    Which inputs disagree is what separates a fact about the sample from a
    fact about the model, and the report is read from a continuous integration
    log for a machine that cannot be borrowed, so it has to carry the values
    rather than an index into an array nobody has.
    """
    inputs = np.linspace(-3.0, 3.0, 100, dtype=np.float32)
    computed = np.expm1(inputs)
    other = computed.copy()
    other[7] = np.nextafter(other[7], np.float32(1e30))

    report = helpers.disagreement_report("expm1", inputs, computed, other)

    assert "1 of 100 inputs disagree" in report
    assert "at most 1 representable step" in report
    # The offending input, to nine significant digits, and both answers.
    assert f"{float(inputs[7]):.9g}" in report
    assert f"{float(computed[7]):.9g}" in report
    assert f"{float(other[7]):.9g}" in report
    # The span of the whole sample, so a reader can see where in it they fell.
    assert f"{float(inputs.min()):.6g}" in report


def test_the_disagreement_report_says_so_when_nothing_disagrees() -> None:
    """A report that is empty when the arrays match reads as a broken report.

    This one is called only from a failing assertion today, so the agreeing
    branch would otherwise never run and would rot unnoticed.
    """
    inputs = np.linspace(-3.0, 3.0, 100, dtype=np.float32)
    computed = np.expm1(inputs)

    report = helpers.disagreement_report("expm1", inputs, computed, computed)

    assert report == "expm1: agrees on all 100 inputs"


FINGERPRINT_FACTS = ["cpu model", "cpu features", "libc", "numpy"]
"""What the run header records about the machine, in the order it records it."""


def test_the_platform_fingerprint_reports_every_fact_it_claims_to() -> None:
    """The fingerprint is compared across machines, so a missing line is what
    makes the comparison ambiguous exactly when it is needed."""
    lines = helpers.platform_fingerprint()

    assert [line.partition(":")[0] for line in lines] == FINGERPRINT_FACTS
    assert all(line.partition(":")[2].strip() for line in lines), lines


def test_the_platform_fingerprint_says_which_facts_it_could_not_read() -> None:
    """A fact left out and a fact that is absent look the same in a log.

    Driven at a path that does not exist, because on the platform this suite
    runs on the processor file is always there and the branch that handles its
    absence would never be reached — an untaken branch in a diagnostic that
    only ever runs on a machine nobody can borrow.
    """
    lines = helpers.platform_fingerprint(pathlib.Path("/proc/cpuinfo.absent"))

    assert [line.partition(":")[0] for line in lines] == FINGERPRINT_FACTS
    unreadable = [line for line in lines if "unreadable" in line]
    assert len(unreadable) == 2, lines
    assert all("/proc/cpuinfo.absent" in line for line in unreadable), unreadable
