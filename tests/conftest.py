"""The fixtures every parity test reads.

**One builder makes the population, the draw arrays and the configuration, and
both implementations under test are handed that one set of objects.** That is
what makes "the same draws" true by construction rather than by coincidence: a
stored draw file would be bypassed by a test that built its own inputs exactly
as easily as a fixture would, and two builders that drift apart turn a parity
failure into a question about the fixtures first.

Across test functions the draws need not match. A parity test asserts that two
implementations agree with *each other* on whatever they were handed, not that
a number equals a literal, so a test that needs four replications does not pay
for fifty. Within one test they must match exactly, which the shared fixture
gives, and across runs they must not move, which the pinned seed gives.

The analytical checks elsewhere in this suite build their own draws at their
own sizes: they compare against a closed form rather than against another
implementation, so they have nothing to share.
"""

import numpy as np
import pytest
from cablesim import config as config_module
from cablesim import constants, population, random_draws, run

DETERMINISTIC_SEGMENTS = 400
"""Population for the deterministic tests.

Large enough that the greedy fill runs off the end of the budget with many
candidates behind it, and that ``age_threshold`` ties thousands of ways on an
integer age, which is what the total sort key exists for. Small enough that
three replications of the scalar reference cost a fraction of a second.
"""

DETERMINISTIC_REPS = 3
"""Replications for those tests.

Three rather than one, because the replication axis is where the two
implementations index the draw arrays differently — the reference takes a
NumPy row, the kernel offsets into a flat slice — so a run with a single
replication would leave that arithmetic reading the same bytes either way.
"""

STATISTICAL_SEGMENTS = 2_000
"""Population for the paired comparison over a real population."""

STATISTICAL_REPS = 50
"""Replications for that comparison.

The scalar reference is what makes this the binding cost, which is why it is
fifty rather than the thousand the batched baseline will carry once it exists.
"""


def simulation_arguments(
    settings: config_module.Config, n_reps: int
) -> dict[str, object]:
    """Builds one call's arguments from a configuration.

    Assembled through the same package functions a real run uses — the
    population generator, the purpose-spawned draw sources and the escalation
    series — so what the parity tests hand an implementation is what a run
    hands it.

    Args:
        settings: The configuration to build from.
        n_reps: Replications in the chunk.

    Returns:
        Every argument of ``simulate.run_chunk`` except ``policy``, keyed by
        name.
    """
    simulation = settings.simulation
    segments = run.segment_arrays(population.generate(settings))
    n_segments = segments["age0"].size
    sources = random_draws.spawn_sources(simulation.seed)
    replications = range(n_reps)

    return {
        **segments,
        "lifetime_uniforms": random_draws.replication_uniforms(
            sources.lifetimes, replications, (n_segments, simulation.n_years + 1)
        ),
        "policy_uniforms": random_draws.replication_uniforms(
            sources.policies, replications, (n_segments,)
        ),
        "budget": settings.budget.annual
        * run.escalation_series(settings.budget.escalation, simulation.n_years),
        "cost_escalation": run.escalation_series(
            settings.costs.escalation_rate, simulation.n_years
        ),
        "emergency_multiplier": settings.costs.emergency_multiplier,
        "mobilization_per_segment": settings.costs.mobilization_per_segment,
        "emergency_charged_to_budget": settings.budget.emergency_charged_to_budget,
        "n_classes": len(settings.population.classes),
        "n_years": simulation.n_years,
    }


@pytest.fixture(scope="session")
def deterministic_arguments() -> dict[str, object]:
    """A small run's arguments, for the tests that force the lifetimes.

    Session-scoped and returned by reference: a test that alters the dictionary
    would alter it for every later test, so each one copies what it changes.

    Returns:
        Every argument of ``simulate.run_chunk`` except ``policy``.
    """
    settings = config_module.resize_population(
        config_module.load_config(constants.DEFAULT_CONFIG_PATH),
        DETERMINISTIC_SEGMENTS,
        n_reps=DETERMINISTIC_REPS,
    )
    return simulation_arguments(settings, DETERMINISTIC_REPS)


@pytest.fixture(scope="session")
def statistical_arguments() -> dict[str, object]:
    """A real population's arguments, for the paired comparison.

    Returns:
        Every argument of ``simulate.run_chunk`` except ``policy``.
    """
    settings = config_module.resize_population(
        config_module.load_config(constants.DEFAULT_CONFIG_PATH),
        STATISTICAL_SEGMENTS,
        n_reps=STATISTICAL_REPS,
    )
    return simulation_arguments(settings, STATISTICAL_REPS)


def forced_lifetimes(
    arguments: dict[str, object], scale: float, replacement: float | None = None
) -> dict[str, object]:
    """Copies the arguments with every Weibull scale replaced.

    Randomness is removed through the ordinary ``scale`` and
    ``replacement_scale`` arrays rather than through an argument only tests
    pass, so what runs is the shipped path. A scale near zero makes every
    segment fail inside its first year; one far past the horizon makes none
    fail at all.

    Args:
        arguments: The arguments to copy.
        scale: What to put in ``scale``.
        replacement: What to put in ``replacement_scale``, or None for
            ``scale``.

    Returns:
        A new argument dictionary; the original is untouched.
    """
    segments = np.shape(arguments["age0"])
    return {
        **arguments,
        "scale": np.full(segments, scale),
        "replacement_scale": np.full(
            segments, scale if replacement is None else replacement
        ),
    }
