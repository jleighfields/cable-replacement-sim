"""Smoke tests for the build pipeline and the configuration loader.

These assert that the Rust extension module compiles, imports, and round-trips
a value, and that the checked-in configuration satisfies its schema. They are
deliberately cheap: their job is to fail loudly when the build is broken,
before any test that depends on a working kernel runs at all.
"""

import pathlib

import pydantic
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

    The base file is the documented default that sweeps override, so a
    schema change that invalidates it breaks every entry point at once.
    """
    loaded = config.load_config()

    assert loaded.simulation.n_years == 30
    assert loaded.population.n_segments == 40000
    assert len(loaded.policies) == 5


def test_class_shares_partition_the_population() -> None:
    """The configured segment-class shares sum to one.

    Shares that do not partition the population silently change the class mix
    rather than raising, which would move every result without explaining
    why.
    """
    loaded = config.load_config()

    assert sum(c.share for c in loaded.population.classes) == pytest.approx(1.0)


def test_a_missing_config_file_raises() -> None:
    """A mistyped path fails at the load rather than falling back.

    Silently substituting the default would surface three steps downstream as
    a result nobody could explain.
    """
    with pytest.raises(FileNotFoundError):
        config.load_config(pathlib.Path("configs/does-not-exist.yaml"))


def test_shares_that_do_not_sum_to_one_are_rejected() -> None:
    """The share validator rejects a class mix that does not partition."""
    raw = {
        "name": "only_class",
        "share": 0.5,
        "n_conductors": 1,
        "length_ft": {"dist": "lognormal", "median": 350, "sigma": 0.6},
        "customers": {"dist": "lognormal", "median": 12, "sigma": 0.85},
        "cost_per_ft": 95.0,
        "weibull": {"shape": 2.0, "scale": 45.0},
    }
    with pytest.raises(pydantic.ValidationError):
        config.PopulationConfig.model_validate(
            {
                "n_segments": 10,
                "total_customers": 100,
                "classes": [raw],
                "initial_age": {
                    "dist": "empirical_install_years",
                    "install_year_range": [1965, 2020],
                    "weights": "build_out_curve",
                },
            }
        )


def test_project_root_resolves_to_the_repo() -> None:
    """The project root is derived from the package, not the caller's cwd.

    A notebook run from `notebooks/` and the app run from the repo root must
    resolve the same configuration file.
    """
    assert (constants.PROJECT_ROOT / "pyproject.toml").is_file()
    assert constants.DEFAULT_CONFIG_PATH.is_file()
