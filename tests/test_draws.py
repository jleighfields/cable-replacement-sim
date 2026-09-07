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
from cablesim import _cablesim, random_draws

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

    # Against NumPy at the same positions, rather than against a run from zero.
    # Reading from zero would be the more direct statement of the property and
    # is unaffordable at the large starts: asking for the first `2**32` draws to
    # compare sixteen of them produces thirty-four gigabytes.
    assert np.array_equal(from_here, numpy_uniforms(key, start, count))
    # And that a position is reached the same way whether it was arrived at or
    # jumped to, which is the property itself, checked where it is cheap.
    if start < 4096:
        run_from_zero = _cablesim.philox_uniforms(key[0], key[1], 0, start + count)
        assert np.array_equal(from_here, run_from_zero[start:])


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


@pytest.mark.parametrize("key", KEYS, ids=lambda pair: f"{pair[0]:x}-{pair[1]:x}")
def test_the_python_generator_matches_numpys(key: tuple[int, int]) -> None:
    """The package's vectorised Philox against the one NumPy ships.

    NumPy's is a Python object per counter, which is why the package has its
    own: eighteen thousand scattered positions would be eighteen thousand
    constructions. What it must not be is a different algorithm, so it is
    checked against the implementation it replaces rather than against the
    Rust one — that comparison is the next test, and two implementations
    agreeing with each other proves less than each agreeing with a third.
    """
    blocks = 64
    counters = np.arange(1, blocks + 1, dtype=np.uint64)
    produced = random_draws.philox(counters, key).ravel()
    expected = np.random.Philox(key=list(key), counter=0).random_raw(blocks * 4)

    assert np.array_equal(produced, expected)


@pytest.mark.parametrize("year", [0, 1, 29])
def test_python_and_rust_draw_the_same_numbers_at_the_same_positions(
    year: int,
) -> None:
    """The property the whole design rests on.

    Every implementation reads its draws from wherever it likes and they must
    still be the same numbers, because a comparison between two policies is a
    paired difference only if replication ``r`` met identical lifetimes under
    each. Handing one array to all of them made that true by construction;
    computing it from an index makes it true only if the two generators agree,
    which is what this establishes.
    """
    key = random_draws.draw_key(20260907)
    replications = np.array([0, 0, 1, 7, 7, 999], dtype=np.uint32)
    segments = np.array([0, 11999, 3, 0, 42, 11999], dtype=np.uint32)

    from_python = random_draws.uniforms_at(
        key, random_draws.PURPOSE["lifetimes"], replications, segments, year
    )
    from_rust = _cablesim.uniforms_at(
        key[0], key[1], random_draws.PURPOSE["lifetimes"], replications, segments, year
    )

    assert np.array_equal(from_python, from_rust)


def test_the_dense_and_sparse_paths_agree_with_each_other() -> None:
    """Two ways of asking for the same positions give the same numbers.

    The dense path exists because every segment reads the starting draw and
    there is nothing to select; the sparse path because a year's replacement
    draw is read by a few percent. They are different code, so a defect in the
    dense one's position arithmetic would otherwise show up only as a run whose
    numbers moved for no reason.
    """
    key = random_draws.draw_key(7)
    n_reps, n_segments = 3, 5
    dense = random_draws.uniforms_dense(
        key, random_draws.PURPOSE["policies"], 0, n_reps, n_segments, 0
    )
    replications = np.repeat(np.arange(n_reps, dtype=np.uint32), n_segments)
    segments = np.tile(np.arange(n_segments, dtype=np.uint32), n_reps)
    sparse = random_draws.uniforms_at(
        key, random_draws.PURPOSE["policies"], replications, segments, 0
    )

    assert np.array_equal(dense.ravel(), sparse)
    assert np.array_equal(
        dense.ravel(),
        _cablesim.uniforms_dense(
            key[0], key[1], random_draws.PURPOSE["policies"], 0, n_reps, n_segments, 0
        ),
    )


def test_a_chunk_draws_the_same_numbers_wherever_it_sits() -> None:
    """Chunking must change no number, which is why the offset is an argument.

    A run splits its replications into chunks to bound memory, and the split is
    provenance rather than a parameter of the model. Replication 7 has to meet
    the same draws whether it was the seventh of one chunk or the second of
    another.
    """
    key = random_draws.draw_key(11)
    whole = random_draws.uniforms_dense(
        key, random_draws.PURPOSE["lifetimes"], 0, 8, 4, 0
    )
    later = random_draws.uniforms_dense(
        key, random_draws.PURPOSE["lifetimes"], 5, 3, 4, 0
    )

    assert np.array_equal(later, whole[5:8])


def test_different_purposes_do_not_share_draws() -> None:
    """The separation the spawned streams used to provide.

    Without it the replacement-policy priorities come out equal to the same
    segments' lifetime draws. A larger uniform gives a shorter lifetime, so the
    random policy would then rank segments by imminence of failure and stop
    being the control the study reads it as — a run that completes, with a
    control that silently is not one.
    """
    key = random_draws.draw_key(3)
    lifetimes = random_draws.uniforms_dense(
        key, random_draws.PURPOSE["lifetimes"], 0, 4, 100, 0
    )
    priorities = random_draws.uniforms_dense(
        key, random_draws.PURPOSE["policies"], 0, 4, 100, 0
    )

    assert not np.array_equal(lifetimes, priorities)
    # Not merely unequal somewhere: no position may coincide by construction.
    assert not np.any(lifetimes == priorities)
