"""Smoke tests for the build pipeline and the configuration loader.

These assert that the Rust extension module compiles, imports and round-trips a
value, and that the checked-in configuration satisfies its schema. They are
deliberately cheap: their job is to fail loudly when the build is broken,
before any test that depends on a working kernel runs at all.
"""

import pathlib

import numpy as np
import pytest
from cablesim import config, constants, kernel

from tests import helpers


def test_the_kernel_runs_one_chunk_through_the_extension_module() -> None:
    """The Rust kernel is reachable from Python and arrays survive both ways.

    This is the end-to-end check that maturin built the crate, that the
    resulting module imported, and that a run crossed the boundary in both
    directions. Randomness is removed by forcing the Weibull scale near zero
    through the ordinary `scale` array, so both segments fail in the first
    year and the answer is known without simulating anything.
    """
    n_segments, n_years, n_classes = 2, 1, 1
    results = kernel.run_chunk(
        length_ft=np.full(n_segments, 100.0),
        customers=np.array([10.0, 20.0]),
        customer_minutes_per_failure=np.array([100.0, 200.0]),
        customer_minutes_per_planned=np.zeros(n_segments),
        outage_cost_per_failure=np.zeros(n_segments),
        class_index=np.zeros(n_segments, dtype=np.uint8),
        age0=np.array([10.0, 20.0]),
        shape=np.full(n_segments, 6.2),
        scale=np.full(n_segments, helpers.FAILS_AT_ONCE),
        replacement_shape=np.full(n_segments, 6.2),
        replacement_scale=np.full(n_segments, helpers.FAILS_AT_ONCE),
        cost_per_ft=np.full(n_segments, 10.0),
        lifetime_uniforms=np.full((1, n_segments, n_years + 1), 0.5),
        policy_uniforms=np.full((1, n_segments), 0.5),
        budget=np.zeros(n_years),
        cost_escalation=np.ones(n_years),
        policy=helpers.resolved("run_to_failure"),
        emergency_multiplier=2.5,
        mobilization_per_segment=500.0,
        emergency_charged_to_budget=False,
        n_classes=n_classes,
        n_years=n_years,
    )

    assert results.failures.shape == (1, n_years, n_classes)
    assert results.failures[0, 0, 0] == 2.0
    assert results.customers_interrupted[0, 0, 0] == 30.0
    # Planned cost is 100 feet at 10 dollars plus 500 of mobilization, and an
    # emergency replacement is 2.5 times that.
    assert results.emergency_spend[0, 0, 0] == 2 * 1_500.0 * 2.5


def test_base_config_satisfies_the_schema() -> None:
    """The checked-in default configuration validates.

    The base file is the documented default that sweeps override, so a schema
    change that invalidates it breaks every entry point at once.
    """
    loaded = config.load_config()

    assert loaded.simulation.n_years == 30
    assert loaded.simulation.start_year == 2026
    assert len(loaded.policies) == 5
    assert len(loaded.population.classes) == 3


def test_a_missing_config_file_raises() -> None:
    """A mistyped path fails at the load rather than falling back.

    Silently substituting the default would surface three steps downstream as a
    result nobody could explain.
    """
    with pytest.raises(FileNotFoundError):
        config.load_config(pathlib.Path("configs/does-not-exist.yaml"))


def test_project_root_resolves_to_the_repo() -> None:
    """The project root is derived from the package, not the caller's cwd.

    A notebook run from `notebooks/` and the app run from the repo root must
    resolve the same configuration file.
    """
    assert (constants.PROJECT_ROOT / "pyproject.toml").is_file()
    assert constants.DEFAULT_CONFIG_PATH.is_file()
