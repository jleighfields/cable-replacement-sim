"""Where every random number in this project comes from.

Two rules make the results reproducible and the controls honest, and both are
easy to get wrong in a way nothing detects:

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


class Sources(NamedTuple):
    """The four independent sources of randomness, one per purpose.

    Attributes:
        lifetimes: Segment lifetime draws, spawned again per replication.
        policies: Replacement-policy randomness, spawned again per replication.
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
        One sequence per purpose, to be spawned again per replication where a
        stream needs to be addressable by replication.
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


def child_of(source: SeedSequence, index: int) -> SeedSequence:
    """Derives one numbered child of a stream, without consuming the stream.

    ``SeedSequence.spawn`` is **stateful**: it counts how many children it has
    handed out, so calling it twice on the same sequence returns two different
    sets. A chunked run calling it once per chunk would therefore give
    replication ``r`` different draws depending on how the run was batched,
    which is exactly the property the per-replication children exist to
    provide. This builds the child by index instead, which is what ``spawn``
    does internally and is reproducible from the index alone.

    Args:
        source: The purpose-level stream.
        index: Which child to derive, counting from zero.

    Returns:
        The numbered child, identical to the one ``spawn`` would return at that
        position on an unused sequence.
    """
    return SeedSequence(
        source.entropy,
        spawn_key=(*source.spawn_key, index),
        pool_size=source.pool_size,
    )


def replication_uniforms(
    source: SeedSequence, replications: range, per_replication: tuple[int, ...]
) -> np.ndarray:
    """Builds one chunk's draws, one replication's block at a time.

    Each replication takes its **own child** of the stream rather than reading
    further along a shared one. That is what makes a chunk addressable: a chunk
    builds its replications from their own children without consuming the ones
    before, so replication ``r`` holds the same draws whatever size the chunks
    were. Advancing a single stream by a computed offset would work too, and it
    would put the arithmetic in the caller, where an error is silent.

    Args:
        source: The purpose-level stream to spawn replication children from.
        replications: Which replications this chunk covers, as indices into the
            run.
        per_replication: Shape of one replication's block.

    Returns:
        An array of shape ``(len(replications), *per_replication)``.
    """
    size = int(np.prod(per_replication))
    return np.stack(
        [
            uniforms(child_of(source, index), size).reshape(per_replication)
            for index in replications
        ]
    )
