"""The counter-based generator, against NumPy's implementation of the same one.

The kernel computes its own uniforms from their index; the Python reference and
the batched loops still take theirs as arrays. **Both have to be the same
numbers, to the last bit**, or the parity tests are comparing two simulations
that saw different randomness — which would show up as a difference somewhere
plausible-looking and be attributed to the wrong thing.

Nothing here checks that the draws look random. Philox is a published algorithm
with published statistical properties, and re-testing those would be testing
NumPy. What is worth testing is the part this project could get wrong: whether
its implementation is the same algorithm, indexed the same way.
"""

import numpy as np
import pytest
from cablesim import _cablesim

KEYS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 0),
    (0, 1),
    (0xDEADBEEF, 0x0BADC0DE),
    (2**64 - 1, 2**64 - 1),
)
"""Keys to check across.

Zero and all-ones are included because they are where an implementation that
mishandles the key schedule is most likely to coincide with a correct one, and
the two single-bit keys distinguish the low word from the high one — swapping
them is a mistake that leaves the output looking perfectly random.
"""


def numpy_uniforms(key: tuple[int, int], start: int, count: int) -> np.ndarray:
    """The uniforms NumPy produces at a stretch of the stream.

    Args:
        key: The two key words, low first.
        start: The first position in the stream.
        count: How many consecutive positions to take.

    Returns:
        ``count`` doubles in ``[0, 1)``.
    """
    lanes = 4
    block, lane = divmod(start, lanes)
    # Drawn from the bit generator's raw stream rather than a distribution
    # method, for the reason `random_draws` gives: NumPy guarantees stream
    # compatibility for bit generators and explicitly does not for `Generator`
    # methods, so a routine upgrade could otherwise move every archived result.
    raw = np.random.Philox(key=list(key), counter=block).random_raw(lane + count)
    return (raw[lane:] >> 11) * (1.0 / 9007199254740992.0)


@pytest.mark.parametrize("key", KEYS, ids=lambda pair: f"{pair[0]:x}-{pair[1]:x}")
def test_the_kernel_and_numpy_produce_the_same_draws(key: tuple[int, int]) -> None:
    """Bit for bit, over a stretch spanning many blocks.

    Exact equality rather than a tolerance: these are two implementations of one
    published algorithm, so anything but equality is a defect in one of them
    rather than a difference to be accommodated. A tolerance here would pass an
    implementation that was one block out, which is the mistake this was
    actually written after making.
    """
    count = 1000
    expected = numpy_uniforms(key, 0, count)
    produced = _cablesim.philox_uniforms(key[0], key[1], 0, count)

    assert np.array_equal(produced, expected)


@pytest.mark.parametrize("start", [0, 1, 3, 4, 5, 7, 8, 4096, 2**32 + 1])
def test_a_draw_does_not_depend_on_where_reading_started(start: int) -> None:
    """Position, not order.

    This is the property the whole design rests on: a draw is a function of its
    index alone, so a worker can compute the one it needs without producing the
    ones before it, and two policies consuming different numbers of draws still
    see the same value at the same place. Asking for a stretch beginning
    part-way through a block is what would catch an implementation that had
    quietly become sequential.
    """
    key = (0xDEADBEEF, 0x0BADC0DE)
    count = 16
    from_here = _cablesim.philox_uniforms(key[0], key[1], start, count)
    from_the_beginning = _cablesim.philox_uniforms(key[0], key[1], 0, start + count)

    assert np.array_equal(from_here, from_the_beginning[start:])
    assert np.array_equal(from_here, numpy_uniforms(key, start, count))


def test_every_draw_can_be_used_as_a_uniform() -> None:
    """In ``[0, 1)``, with 1 excluded rather than merely unlikely.

    ``draw_lifetime`` takes ``log1p(-u)``, which is infinite at exactly 1, so a
    generator that could return it would produce an infinite lifetime for one
    segment in a few billion — a run that completes, with one cable that never
    fails.
    """
    draws = _cablesim.philox_uniforms(1, 0, 0, 100_000)

    assert draws.min() >= 0.0
    assert draws.max() < 1.0


def test_asking_for_nothing_gives_nothing() -> None:
    """The empty case returns an empty array rather than failing.

    A chunk covering no positions is not an error, and a caller looping over
    ranges should not have to special-case it.
    """
    assert _cablesim.philox_uniforms(1, 0, 0, 0).shape == (0,)
