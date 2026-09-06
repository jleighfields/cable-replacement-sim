"""Smoke tests for the build pipeline and the configuration loader.

These assert that the Rust extension module compiles, imports and round-trips a
value, and that the checked-in configuration satisfies its schema. They are
deliberately cheap: their job is to fail loudly when the build is broken,
before any test that depends on a working kernel runs at all.
"""

import pathlib

import pytest
from cablesim import add, config, constants


def test_add_round_trips_through_the_extension_module() -> None:
    """The Rust `add` is reachable from Python and returns its sum.

    This is the end-to-end check that maturin built the crate, that the
    resulting module imported, and that an integer survived the boundary in
    both directions.
    """
    assert add(2, 3) == 5
    assert add(-1, 1) == 0


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
