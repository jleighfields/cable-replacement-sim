"""Random streams, spawned by purpose so that they cannot correlate.

Every uniform in this project comes from here. Two rules make the results
reproducible and the controls honest, and both are easy to get wrong in a way
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
NumPy guarantees version-to-version stream compatibility for ``BitGenerator``
classes and explicitly permits ``Generator`` methods to change on feature
releases, so a routine upgrade could otherwise move every archived result with
nothing failing.
"""

from typing import NamedTuple

import numpy as np
from numpy.random import PCG64, SeedSequence


class Streams(NamedTuple):
    """The four independent random streams, one per purpose.

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


def spawn_roots(seed: int) -> Streams:
    """Derives the four independent root streams from the configured seed.

    The order is part of the contract: changing it changes every result, so it
    lives here and nowhere else.

    Args:
        seed: ``simulation.seed`` from the configuration.

    Returns:
        One root sequence per purpose.
    """
    return Streams(*SeedSequence(seed).spawn(4))


def uniforms(source: SeedSequence, size: int) -> np.ndarray:
    """Draws uniforms on [0, 1) from a sequence's raw bit stream.

    The conversion is the one ``Generator.random`` performs internally, written
    out so that it rests on the guarantee NumPy gives for bit generators rather
    than the weaker one it gives for distribution methods.

    Args:
        source: The sequence to draw from.
        size: How many uniforms to produce.

    Returns:
        A one-dimensional array of ``size`` floats on [0, 1).
    """
    raw = PCG64(source).random_raw(size)
    return (raw >> np.uint64(11)) * 2.0**-53
