"""Checks that each configuration validator rejects what it is meant to.

Every validator here guards a failure that would otherwise surface as a wrong
number rather than an error, so each test breaks the checked-in configuration
in exactly one way and confirms the loader refuses it. A validator nobody has
watched refuse something is not known to work.
"""

import copy
from typing import Any

import pydantic
import pytest
from cablesim import config


def raw_config() -> dict[str, Any]:
    """The checked-in configuration as a plain dictionary.

    Returns:
        A deep copy, safe for a test to break.
    """
    return copy.deepcopy(config.load_config().model_dump())


def test_class_shares_must_partition_the_population() -> None:
    """Shares that do not sum to one are rejected.

    Shares that do not partition silently change the class mix rather than
    raising, which moves every result without explaining why.
    """
    broken = raw_config()
    broken["population"]["classes"][0]["share"] += 0.05

    with pytest.raises(pydantic.ValidationError, match="shares must sum to 1"):
        config.Config.model_validate(broken)


def test_a_customer_type_missing_from_a_class_is_rejected() -> None:
    """A type in one block and absent from another would contribute zero."""
    broken = raw_config()
    del broken["population"]["classes"][0]["customer_mix"]["industrial"]

    with pytest.raises(pydantic.ValidationError, match="customer_mix covers"):
        config.Config.model_validate(broken)


def test_a_customer_type_missing_a_value_of_lost_load_is_rejected() -> None:
    """Scoring would weight that type at zero and nothing would say so."""
    broken = raw_config()
    del broken["reliability"]["voll_per_customer_hour"]["commercial"]

    with pytest.raises(pydantic.ValidationError, match="voll_per_customer_hour"):
        config.Config.model_validate(broken)


def test_a_gap_in_the_technology_vintages_is_rejected() -> None:
    """A gap leaves segments with no technology and therefore no parameters."""
    broken = raw_config()
    broken["population"]["technologies"][1]["vintage"] = (1990, 2004)

    with pytest.raises(pydantic.ValidationError, match="no gap or overlap"):
        config.Config.model_validate(broken)


def test_an_unconfigured_replacement_technology_is_rejected() -> None:
    """Replacements would otherwise install a technology that does not exist."""
    broken = raw_config()
    broken["population"]["replacement_technology"] = "not_a_technology"

    with pytest.raises(pydantic.ValidationError, match="replacement_technology"):
        config.Config.model_validate(broken)


def test_a_missing_class_restoration_time_is_rejected() -> None:
    """A class absent from the map would contribute a zero-duration outage.

    That reads as a reliability improvement, which is the worst way for a
    configuration error to present.
    """
    broken = raw_config()
    del broken["reliability"]["outage_hours_emergency"]["lateral_1ph"]

    with pytest.raises(pydantic.ValidationError, match="outage_hours_emergency"):
        config.Config.model_validate(broken)


def test_a_study_ending_after_the_simulation_begins_is_rejected() -> None:
    """The fit would otherwise use data the simulation is meant to predict."""
    broken = raw_config()
    broken["records"]["study_end"] = broken["simulation"]["start_year"] + 1

    with pytest.raises(pydantic.ValidationError, match="monitoring_start < study_end"):
        config.Config.model_validate(broken)


def test_a_policy_parameter_the_policy_does_not_take_is_rejected() -> None:
    """A misspelled parameter would leave the policy on its default."""
    broken = raw_config()
    broken["policies"][1]["params"] = {"threshhold_years": 45}

    with pytest.raises(pydantic.ValidationError, match="does not take"):
        config.Config.model_validate(broken)


def test_an_unconfigured_baseline_policy_is_rejected() -> None:
    """Avoided metrics would have nothing to be measured against."""
    broken = raw_config()
    broken["reporting"]["baseline_policy"] = "worst_first_but_misspelled"

    with pytest.raises(pydantic.ValidationError, match="baseline_policy"):
        config.Config.model_validate(broken)


def test_install_volume_must_reach_both_endpoints() -> None:
    """Interpolation would otherwise extrapolate past the configured range."""
    broken = raw_config()
    del broken["population"]["initial_age"]["install_volume"][1965]

    with pytest.raises(pydantic.ValidationError, match="include both endpoints"):
        config.Config.model_validate(broken)


def test_an_unimplemented_dependence_model_is_rejected() -> None:
    """Accepting it would leave the run silently using independence."""
    broken = raw_config()
    broken["failure"]["conductor_dependence"] = "shared_frailty"

    with pytest.raises(pydantic.ValidationError):
        config.Config.model_validate(broken)


def test_a_lognormal_median_of_zero_is_rejected() -> None:
    """A median of zero is not a lognormal: it would need the log of zero."""
    broken = raw_config()
    broken["population"]["classes"][0]["length_ft"]["median"] = 0.0

    with pytest.raises(pydantic.ValidationError):
        config.Config.model_validate(broken)
