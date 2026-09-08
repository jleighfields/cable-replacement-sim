"""Where every random number in this project comes from.

**Two designs live here, and which one applies depends on what is being
drawn.** The simulation's own uniforms — segment lifetimes and the random
policy's priorities — are computed from their position by the counter-based
generator in the second half of this module, because the Rust kernel has to
produce the identical numbers without a stream to share. Everything drawn once
per run rather than per replication — the synthetic segment table and the
synthetic failure history — still comes from the spawned ``SeedSequence``
sources in the first half, which nothing crosses a language boundary to
reproduce.

Two rules govern the spawned sources, and both are easy to get wrong in a way
nothing detects:

**Spawn by purpose first, then by replication.** A fresh ``SeedSequence`` has
spawned nothing, so calling ``spawn`` on two separately constructed ones
returns *identical* children. Asking for "its own child" without saying where
from therefore produces the correlation the separation exists to prevent — the
replacement-policy priorities would come out bit-identical to the same
segments' first lifetime draws, and since a larger uniform gives a shorter
lifetime, the random policy would rank segments by imminence of failure and
stop being a control.

**Draw from the bit generator's raw stream, not from a distribution method.**
This is also why the module is not named for a generator: it deliberately does
not hand back a ``numpy.random.Generator``.
NumPy guarantees version-to-version stream compatibility for ``BitGenerator``
classes and explicitly permits ``Generator`` methods to change on feature
releases, so a routine upgrade could otherwise move every archived result with
nothing failing.
"""

from typing import NamedTuple

import numpy as np
from numpy.random import PCG64, SeedSequence

from cablesim import _cablesim


class Sources(NamedTuple):
    """The four independent sources of randomness, one per purpose.

    Attributes:
        lifetimes: Segment lifetime draws. Unused by the annual loop, which
            computes its lifetimes from their positions instead; kept so the
            four purposes stay at fixed spawn positions, since renumbering
            them would move the population and history draws of every archived
            run.
        policies: Replacement-policy randomness. Unused for the same reason.
        population: The synthetic segment table.
        records: The synthetic censored failure history.
    """

    lifetimes: SeedSequence
    policies: SeedSequence
    population: SeedSequence
    records: SeedSequence


def spawn_sources(seed: int) -> Sources:
    """Derives the four independent root streams from the configured seed.

    The order is part of the contract: changing it changes every result, so it
    lives here and nowhere else.

    Args:
        seed: ``simulation.seed`` from the configuration.

    Returns:
        One sequence per purpose. ``population`` and ``records`` are the two a
        run reads; the annual loop's own uniforms come from ``uniforms_dense``
        and ``uniforms_at`` below instead.
    """
    return Sources(*SeedSequence(seed).spawn(4))


def uniforms(source: SeedSequence, size: int) -> np.ndarray:
    """Draws uniforms on [0, 1) from a sequence's raw bit stream.

    The conversion is the one ``Generator.random`` performs internally, written
    out so that it rests on the guarantee NumPy gives for bit generators rather
    than the weaker one it gives for distribution methods.

    Args:
        source: The sequence to draw from.
        size: How many uniforms to produce.

    Note:
        Calling this twice on the same sequence returns the *same* array. A
        sequence is a description of a stream rather than a position in one, so
        two draws that must differ come from two children of it, never from two
        calls.

    Returns:
        A one-dimensional array of ``size`` floats on [0, 1).
    """
    raw = PCG64(source).random_raw(size)
    return (raw >> np.uint64(11)) * 2.0**-53


PURPOSE: dict[str, int] = dict(_cablesim.DRAW_PURPOSES)
"""Integer tag per purpose, which the draw index carries as a field.

Read from the compiled crate rather than written here, for the reason the field
widths are: a tag written twice would put the two languages on different draws
for the same position, and only a parity test would notice.


Replaces the spawn-by-purpose separation above for the simulation's own draws,
and does the same job: without it the replacement-policy priorities would come
out correlated with the same segments' lifetime draws, and since a larger
uniform gives a shorter lifetime the random policy would rank segments by
imminence of failure and stop being a control.

A field rather than four streams, because there are no streams to keep apart
once a draw is a function of where it sits.
"""

MAX_REPLICATION, MAX_SEGMENT, MAX_YEAR, MAX_PURPOSE = _cablesim.DRAW_INDEX_LIMITS
"""The largest position each field of a draw index can carry."""

PURPOSE_SHIFT, REPLICATION_SHIFT, YEAR_SHIFT, SEGMENT_SHIFT = (
    _cablesim.DRAW_INDEX_SHIFTS
)
"""Where each field sits in a draw index.

The segment is in the low bits, which is what lets four adjacent segments come
out of one call to the generator: it produces four words at a time, so a block
is shared by four consecutive indices. Anywhere else and every draw would cost
a full encryption of which three quarters was discarded.


**Read from the compiled crate rather than written here.** The packing decides
which draw a position gets, so a copy of these numbers that drifted would put
the two languages on different draws — or worse, alias two positions onto one,
which is a correlation nothing downstream could detect. The crate authors them
and this reads them, so there is one copy.
"""

PHILOX_LANES = 4
"""Values one Philox counter yields, so index ``i`` is block ``i // 4``."""

MAX_RUN_DRAWS = 16_384
"""Longest run ``uniforms_at`` will produce to gather a few positions out of.

**A memory bound, not a tuned crossover.** A run is charged for the largest
segment asked for and not for how many positions were asked for, so without a
bound one draw high in the segment field produces every position beneath it:
``check_positions`` admits segments up to ``MAX_SEGMENT``, where the run would
be 32 GB.

The value sits above the 12,000-segment population this project ships, where
producing the run is about three times cheaper than drawing each position its
own block, and below 100,000, where it is two to three times more expensive.
Where the two cross in between was measured twice, in and out of the loop that
calls this, and came out at 16,000 one way and 80,000 the other — the run
path's cost per draw moves with how the allocator handles the size being asked
for. Nothing here runs a population in that range, and any value across it
behaves the same on the two sizes that are run.
"""

PHILOX_MULTIPLIERS = (np.uint64(0xD2E7470EE14C6C93), np.uint64(0xCA5A826395121157))
"""The round function's two multipliers, from Random123."""

PHILOX_WEYL = (np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBB67AE8584CAA73B))
"""The key increments per round, from the golden ratio and the square root of 3."""

PHILOX_ROUNDS = 10
"""Rounds in Philox-4x64-10, the variant NumPy implements."""

LOW_32 = np.uint64(0xFFFFFFFF)
"""Mask for the low half of a 64-bit word."""

HALF_WORD = np.uint64(32)
"""Bits in half a word, for splitting a product."""

MANTISSA_SHIFT = np.uint64(11)
"""Bits dropped to leave the 53 a double represents exactly."""

TWO_TO_THE_FIFTY_THIRD = 1.0 / 9007199254740992.0
"""What a 53-bit integer is scaled by to land in ``[0, 1)``."""


def draw_key(seed: int) -> tuple[int, int]:
    """Turns a run's seed into the two key words a draw is computed under.

    Derived through ``SeedSequence`` rather than by using the seed directly, so
    that a small integer seed still spreads across both key words. Its
    ``generate_state`` is a documented, stream-stable entry point, which matters
    for the same reason the raw bit-generator stream is used above: an archived
    run has to reproduce.

    Args:
        seed: The run's seed.

    Returns:
        The low and high key words.
    """
    low, high = SeedSequence(seed).generate_state(2, dtype=np.uint64)
    return int(low), int(high)


def multiply_wide(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """High and low halves of a 64-by-64 bit product, over arrays.

    Rust takes the product in 128 bits and shifts. NumPy has no 128-bit integer,
    so the factors are split into 32-bit halves and recombined — the schoolbook
    method, arranged so no intermediate exceeds 64 bits.

    Args:
        left: One factor.
        right: The other.

    Returns:
        The high and low words of the product.
    """
    left_low, left_high = left & LOW_32, left >> HALF_WORD
    right_low, right_high = right & LOW_32, right >> HALF_WORD
    low_low = left_low * right_low
    middle = left_low * right_high + (low_low >> HALF_WORD)
    carried = left_high * right_low + (middle & LOW_32)
    high = left_high * right_high + (middle >> HALF_WORD) + (carried >> HALF_WORD)
    # The low word is the wrapping product, which is what unsigned multiplication
    # already gives.
    return high, left * right


def philox(counters: np.ndarray, key: tuple[int, int]) -> np.ndarray:
    """Encrypts each counter under the key, four words apiece.

    The whole generator: no state advances, so the value at any counter is
    available without producing the ones before it. **The Rust crate implements
    the same algorithm**, and both are checked against NumPy's own Philox.

    Args:
        counters: Counters to encrypt, as ``uint64``.
        key: The two key words.

    Returns:
        ``(len(counters), 4)`` of ``uint64``.
    """
    zero = np.zeros_like(counters)
    state = [counters, zero, zero, zero]
    schedule = [
        np.full_like(counters, np.uint64(key[0])),
        np.full_like(counters, np.uint64(key[1])),
    ]
    for round_index in range(PHILOX_ROUNDS):
        if round_index > 0:
            schedule = [
                schedule[0] + PHILOX_WEYL[0],
                schedule[1] + PHILOX_WEYL[1],
            ]
        high_0, low_0 = multiply_wide(PHILOX_MULTIPLIERS[0], state[0])
        high_1, low_1 = multiply_wide(PHILOX_MULTIPLIERS[1], state[2])
        state = [
            high_1 ^ state[1] ^ schedule[0],
            low_1,
            high_0 ^ state[3] ^ schedule[1],
            low_0,
        ]
    return np.stack(state, axis=-1)


def draw_index(
    purpose: int,
    replications: np.ndarray,
    segments: np.ndarray,
    year: int,
) -> np.ndarray:
    """Where in the stream the draws for these positions live.

    Args:
        purpose: Which stream, from ``PURPOSE``.
        replications: One entry per wanted draw.
        segments: The matching segment of each.
        year: The year these draws belong to.

    Returns:
        One index per position, as ``uint64``.

    Raises:
        ValueError: If a position is past what its field can carry, where two
            positions would otherwise share one draw.
    """
    if replications.shape != segments.shape:
        raise ValueError(
            f"replications has {replications.size} entries against "
            f"{segments.size} segments; a draw is named by both, so they pair "
            f"up one for one"
        )
    check_positions(
        int(replications.max()) if replications.size else 0,
        int(segments.max()) if segments.size else 0,
        year,
        purpose,
    )
    return (
        (np.uint64(purpose) << np.uint64(PURPOSE_SHIFT))
        | (replications.astype(np.uint64) << np.uint64(REPLICATION_SHIFT))
        | (np.uint64(year) << np.uint64(YEAR_SHIFT))
        | (segments.astype(np.uint64) << np.uint64(SEGMENT_SHIFT))
    )


def check_positions(
    replication: int, segment: int, year: int, purpose: int = 0
) -> None:
    """Refuses a position the index cannot represent.

    Called once with the largest position a chunk will reach, rather than per
    draw, because the largest is known from the shape and checking every draw
    would cost more than producing it.

    Args:
        replication: The largest replication this chunk covers.
        segment: The largest segment identifier.
        year: The last year drawn for.
        purpose: Which stream. Bounded like the other three because it folds the
            same way: it sits in the top bits, so one past the limit shifts out
            of the word and lands on a purpose that fits — identically on both
            sides of the boundary, which is why no comparison between
            implementations could see it.

    Raises:
        ValueError: If any is past what its field can carry, where two positions
            would otherwise share one draw.
    """
    for name, value, limit in (
        ("purpose", purpose, MAX_PURPOSE),
        ("replication", replication, MAX_REPLICATION),
        ("segment", segment, MAX_SEGMENT),
        ("year", year, MAX_YEAR),
    ):
        if value > limit:
            raise ValueError(
                f"{name} {value} is past the {limit} a draw index can carry; "
                f"beyond it two positions would share one draw"
            )


def uniforms_at(
    key: tuple[int, int],
    purpose: int,
    replications: np.ndarray,
    segments: np.ndarray,
    year: int,
) -> np.ndarray:
    """The uniforms at named positions, and nothing else.

    The sparse case, and the reason the generator is indexed: a year's
    replacement draw is consumed only by a segment replaced that year, which is
    a few percent of them.

    Args:
        key: The two key words, from ``draw_key``.
        purpose: Which stream, from ``PURPOSE``.
        replications: One entry per wanted draw.
        segments: The matching segment of each.
        year: The year these draws belong to.

    Returns:
        One double in ``[0, 1)`` per position, in the order given.
    """
    index = draw_index(purpose, replications, segments, year)
    if index.size == 0:
        return np.empty(0)

    # **Positions sharing a replication lie inside one consecutive run**, so the
    # run can be produced and the wanted entries taken out of it. At the shipped
    # population that is more draws and less time — 108 microseconds against
    # 325 for the scattered path, asking for 200 positions across 12,000
    # segments. It is the reference implementation that takes this branch,
    # drawing one replication's replaced segments at a time.
    #
    # Both conditions are needed. Positions spanning several replications are
    # several runs separated by a wide stride, and a run longer than
    # ``MAX_RUN_DRAWS`` is refused for the reason that constant carries. The
    # scattered path's cost grows with the number of positions asked for; the
    # run's grows with the largest segment among them, which is why the two
    # cross at all. Detected here rather than asked of the caller: which branch
    # is cheaper is a fact about this module's generators, not about the loop.
    largest = int(segments.max())
    if largest < MAX_RUN_DRAWS and (replications == replications[0]).all():
        start = draw_index(purpose, replications[:1], np.zeros(1, np.uint64), year)
        run = uniforms_over(key, int(start[0]), largest + 1)
        return run[segments]

    lanes = np.uint64(PHILOX_LANES)
    # One block per draw, because the positions share nothing.
    words = philox(index // lanes + np.uint64(1), key)
    chosen = words[np.arange(index.size), (index % lanes).astype(np.intp)]
    return to_double(chosen)


def to_double(words: np.ndarray) -> np.ndarray:
    """Turns uniform 64-bit words into doubles in ``[0, 1)``.

    The top 53 bits are kept, which is every bit a double represents without
    rounding. **This is NumPy's own conversion and the crate's**, and it has to
    be all three: a different rounding would put the languages one bit apart on
    every draw.

    Args:
        words: Uniform 64-bit words.

    Returns:
        Doubles in ``[0, 1)``.
    """
    return (words >> MANTISSA_SHIFT) * TWO_TO_THE_FIFTY_THIRD


def uniforms_over(
    key: tuple[int, int], first_index: int, count: int
) -> np.ndarray:
    """The uniforms at a run of consecutive positions.

    **Four consecutive positions share one block**, so this encrypts a quarter as
    many counters as there are draws. That is the whole reason the segment field
    sits in the low bits, and it is what the dense case is made of: a
    replication's segments are consecutive, so its starting lifetimes are one
    run and its priorities another.

    Args:
        key: The two key words.
        first_index: Where the run starts.
        count: How many consecutive positions to produce.

    Returns:
        ``count`` doubles in ``[0, 1)``.
    """
    if count == 0:
        return np.empty(0)
    first_block, offset = divmod(first_index, PHILOX_LANES)
    # **NumPy's own Philox, not the one in this module.** A run of consecutive
    # positions is exactly what a stream produces, so the C implementation
    # applies. The vectorised `philox` above exists for *scattered* positions,
    # where a stream is no use, and it carries about 320 NumPy operations — a
    # floor of 255 microseconds whatever the array size. NumPy's has no such
    # floor: 13 microseconds for 360 draws against 300, and roughly ten times
    # the throughput besides.
    #
    # The two produce the same stream, which `test_draws.py` asserts rather than
    # assumes. `counter` is the block index directly, because NumPy increments
    # before producing: its first word at `counter=b` is the one Philox defines
    # at block `b + 1`, so it already carries the `+ 1` that `uniforms_at`
    # above adds by hand.
    words = np.random.Philox(
        key=np.array(key, dtype=np.uint64), counter=first_block
    ).random_raw(offset + count)
    return to_double(words[offset:])


def uniforms_dense(
    key: tuple[int, int],
    purpose: int,
    first_replication: int,
    n_reps: int,
    n_segments: int,
    year: int,
) -> np.ndarray:
    """The uniforms every segment of every replication reads.

    The dense case: the left-truncated draw taken at the start of a run, and the
    fixed per-segment priority the random policy ranks on.

    Args:
        key: The two key words, from ``draw_key``.
        purpose: Which stream, from ``PURPOSE``.
        first_replication: Where this chunk starts in the run, so a chunk draws
            the same numbers wherever it sits.
        n_reps: Replications in this chunk.
        n_segments: Segments in the population.
        year: The year these draws belong to.

    Returns:
        ``(n_reps, n_segments)`` doubles in ``[0, 1)``.
    """
    if n_reps == 0:
        # Worded exactly as the binding words it: the two are documented as
        # computing the same values, so they must refuse the same input the
        # same way.
        raise ValueError("n_reps is 0, so there are no positions to draw at")
    if n_reps < 0 or n_segments < 0:
        # A separate branch rather than a reworded message, because the one
        # above has to stay byte-identical to the binding's. The binding gets
        # these refused by PyO3's own extraction, which will not take a
        # negative into an unsigned argument.
        raise ValueError(
            f"n_reps is {n_reps} and n_segments is {n_segments}; neither counts "
            f"anything below zero"
        )
    check_positions(first_replication + n_reps - 1, n_segments - 1, year, purpose)
    # One run per replication: a replication's segments are consecutive, so its
    # draws are consecutive too. Across replications they are not, because the
    # replication field changes, so this is a loop of runs rather than one run.
    return np.stack(
        [
            uniforms_over(
                key,
                int(
                    (np.uint64(purpose) << np.uint64(PURPOSE_SHIFT))
                    | (np.uint64(replication) << np.uint64(REPLICATION_SHIFT))
                    | (np.uint64(year) << np.uint64(YEAR_SHIFT))
                ),
                n_segments,
            )
            for replication in range(first_replication, first_replication + n_reps)
        ]
    )
