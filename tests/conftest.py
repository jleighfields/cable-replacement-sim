"""The fixtures every parity test reads.

**One builder makes the population, the draw key and the configuration, and
every implementation under test is handed that one set of objects.** Two
builders that drifted apart would turn a parity failure into a question about
the fixtures first.

The draws themselves are no longer among those objects: each implementation
computes them from the key and the position it is at. That they come out
identical is established by `test_draws.py` rather than by construction, which
is the one guarantee this design traded away.

Across test functions the draws need not match. A parity test asserts that two
implementations agree with *each other* on whatever they were handed, not that
a number equals a literal, so a test that needs four replications does not pay
for fifty. Within one test they must match exactly, which the shared fixture
gives, and across runs they must not move, which the pinned seed gives.

The analytical checks elsewhere in this suite build their own draws at their
own sizes: they compare against a closed form rather than against another
implementation, so they have nothing to share.
"""

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
implementations address their draws differently — the reference builds a
``(replications, segments)`` block and takes a row of it, the kernel computes
each position on its own — so a run with a single replication would leave that
arithmetic reading position zero either way.
"""

STATISTICAL_SEGMENTS = 2_000
"""Population for the paired comparison over a real population."""

STATISTICAL_REPS = 50
"""Replications for that comparison.

The scalar reference is run here alongside every other implementation and is
the slowest of them by a wide margin, so it is what sets this number: fifty
rather than the thousand a shipped run uses. The comparison is paired, so it
does not need the replication count a confidence interval on a single run
would.
"""


def simulation_arguments(settings: config_module.Config) -> dict[str, object]:
    """Builds one call's arguments from a configuration.

    Assembled through the same package functions a real run uses — the
    population generator, the key derivation and the escalation series — so
    what the parity tests hand an implementation is what a run hands it.

    Args:
        settings: The configuration to build from, which carries the
            replication count and the seed. Taken from there rather than passed
            alongside them, so a fixture cannot resize the population to one
            count and set ``n_reps`` to another.

    Returns:
        Every argument of ``simulate.run_chunk`` except ``policy``, keyed by
        name.
    """
    simulation = settings.simulation
    if simulation.n_reps < 2:
        # One replication removes what this fixture exists for without failing
        # anything: the replication is a field of every draw's address, and at
        # a single replication that field is 0 on both sides whatever the
        # arithmetic around it does. Mutating either side's offset away reddens
        # the parity tests at three replications and none at one.
        raise ValueError(
            f"the parity fixtures need at least 2 replications, got "
            f"{simulation.n_reps}: the replication axis is where the two "
            f"implementations address their draws differently"
        )

    segments = run.segment_arrays(population.generate(settings))

    return {
        **segments,
        "draw_key": random_draws.draw_key(simulation.seed),
        "first_replication": 0,
        "n_reps": simulation.n_reps,
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
    return simulation_arguments(settings)


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
    return simulation_arguments(settings)
