"""Checks that the four random streams are independent and stable.

Both properties fail silently. Correlated streams still produce plausible
numbers, and a stream that moves under a library upgrade still produces
plausible numbers — they are just different ones from the archived run.
"""

import numpy as np
from cablesim import streams

DRAWS: int = 20_000

PINNED_POPULATION_DRAWS: list[float] = [
    0.8382711479571602,
    0.08372444856512495,
    0.6176152913826177,
    0.7028375859931597,
]
"""The first four population uniforms at seed 0.

Pinned so that a change in the underlying stream — a NumPy release altering a
bit generator, or an edit to the conversion — fails here rather than quietly
moving every archived result. NumPy guarantees version-to-version stream
compatibility for bit generators and explicitly permits `Generator` methods to
change, which is why the conversion is written out rather than delegated.
"""


def test_the_four_roots_are_independent() -> None:
    """No two purposes draw the same numbers.

    A fresh `SeedSequence` has spawned nothing, so two separately constructed
    ones return identical children. The separation therefore has to be made
    once at a root spawn, and if it collapses the replacement policy's
    priorities become a deterministic function of the same segments' lifetime
    draws — the control stops controlling, and every parity test still passes,
    because every implementation reads the same array.
    """
    roots = streams.spawn_roots(20260902)
    drawn = [streams.uniforms(root, DRAWS) for root in roots]

    for index, left in enumerate(drawn):
        for right in drawn[index + 1 :]:
            assert not np.array_equal(left, right)
            assert abs(float(np.corrcoef(left, right)[0, 1])) < 0.05


def test_the_uniform_stream_is_pinned() -> None:
    """The draws are the ones every archived result was computed against."""
    drawn = streams.uniforms(streams.spawn_roots(0).population, 4)

    assert [float(value) for value in drawn] == PINNED_POPULATION_DRAWS


def test_uniforms_are_on_the_unit_interval() -> None:
    """The conversion lands in [0, 1) and covers it.

    The shift-and-scale is the one `Generator.random` performs internally; an
    error in it would show up as a distribution that is uniform on the wrong
    interval, which nothing downstream would raise on.
    """
    drawn = streams.uniforms(streams.spawn_roots(7).lifetimes, DRAWS)

    assert drawn.min() >= 0.0
    assert drawn.max() < 1.0
    assert abs(float(drawn.mean()) - 0.5) < 0.01


def test_the_same_seed_gives_the_same_roots() -> None:
    """Reproducibility, which is the reason the seed is recorded with a run."""
    first = streams.uniforms(streams.spawn_roots(11).records, DRAWS)
    second = streams.uniforms(streams.spawn_roots(11).records, DRAWS)

    assert np.array_equal(first, second)
    assert not np.array_equal(
        first, streams.uniforms(streams.spawn_roots(12).records, DRAWS)
    )
