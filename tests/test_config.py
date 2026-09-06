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
import yaml
from cablesim import config, constants


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

    That reads as a reliability improvement, so nothing downstream looks wrong.
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

    with pytest.raises(pydantic.ValidationError, match="'iid'"):
        config.Config.model_validate(broken)


def test_a_lognormal_median_of_zero_is_rejected() -> None:
    """A median of zero is not a lognormal: it would need the log of zero."""
    broken = raw_config()
    broken["population"]["classes"][0]["length_ft"]["median"] = 0.0

    with pytest.raises(pydantic.ValidationError, match="greater than 0"):
        config.Config.model_validate(broken)


def test_a_negative_value_of_lost_load_is_rejected() -> None:
    """A negative value of lost load makes an interruption look beneficial.

    ``outage_cost_per_failure`` is the product of the per-type counts, the
    value of lost load and the restoration time, and a risk-ranked policy
    maximises it. A negative entry therefore ranks the segments it applies to
    as the ones most worth leaving in the ground, and nothing downstream
    raises.
    """
    broken = raw_config()
    broken["reliability"]["voll_per_customer_hour"]["residential"] = -8.0

    with pytest.raises(pydantic.ValidationError, match="voll_per_customer_hour"):
        config.Config.model_validate(broken)


def test_a_negative_restoration_time_is_rejected() -> None:
    """A negative restoration time drives SAIDI and CMI negative.

    Both minute columns are the customer count times the configured hours, so
    a negative entry subtracts from the reliability indices rather than adding
    to them — an outage that improves the numbers.
    """
    broken = raw_config()
    broken["reliability"]["outage_hours_emergency"]["lateral_1ph"] = -5.0

    with pytest.raises(pydantic.ValidationError, match="outage_hours_emergency"):
        config.Config.model_validate(broken)


def test_install_years_after_the_simulation_starts_are_rejected() -> None:
    """Cable cannot be installed after year 0 of the run.

    Age is ``start_year - install_year``, so an install-year range reaching
    past ``simulation.start_year`` gives some segments a negative age. Both
    Weibull forms then raise a negative number to a fractional power and
    return NaN, which propagates through every result behind a
    ``RuntimeWarning`` and no exception.
    """
    broken = raw_config()
    first, _ = broken["population"]["initial_age"]["install_year_range"]
    beyond = broken["simulation"]["start_year"] + 10
    broken["population"]["initial_age"]["install_year_range"] = (first, beyond)
    broken["population"]["initial_age"]["install_volume"][beyond] = (
        broken["population"]["initial_age"]["install_volume"].pop(2020)
    )
    broken["population"]["technologies"][-1]["vintage"] = (2005, beyond)

    with pytest.raises(pydantic.ValidationError, match="start_year"):
        config.Config.model_validate(broken)


def test_a_policy_missing_a_parameter_it_requires_is_rejected() -> None:
    """An omitted parameter leaves the policy on a default just as a typo does.

    The whitelist rejects a key the policy does not take, which catches a
    misspelling because the misspelled key is unexpected. Omitting the key
    outright produces the same silent fallback and has to be rejected on the
    same grounds.
    """
    broken = raw_config()
    broken["policies"][1] = {"name": "age_threshold", "params": {}}

    with pytest.raises(pydantic.ValidationError, match="threshold_years"):
        config.Config.model_validate(broken)


def test_every_configured_number_parses_as_a_number() -> None:
    """No numeric field reaches pydantic as a string.

    YAML 1.1 requires a signed exponent, so `12.0e6` is a *string* to the
    parser and only becomes a float because pydantic coerces it. That works
    until something reads the file without pydantic — a driver script, a diff,
    another language — and it hides in plain sight, because the coerced value
    is correct. Writing the digits out avoids the trap entirely; this checks
    nobody reintroduces it.
    """
    raw = yaml.safe_load(constants.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))

    numeric: list[tuple[str, Any]] = [
        ("population.n_segments", raw["population"]["n_segments"]),
        ("population.total_customers", raw["population"]["total_customers"]),
        ("population.length_ref_ft", raw["population"]["length_ref_ft"]),
        ("population.length_exponent", raw["population"]["length_exponent"]),
        ("records.n_segments", raw["records"]["n_segments"]),
        ("budget.annual", raw["budget"]["annual"]),
        ("budget.escalation", raw["budget"]["escalation"]),
        ("costs.mobilization_per_segment", raw["costs"]["mobilization_per_segment"]),
        ("costs.emergency_multiplier", raw["costs"]["emergency_multiplier"]),
        ("costs.discount_rate", raw["costs"]["discount_rate"]),
    ]
    for name, value in numeric:
        assert isinstance(value, (int, float)), (
            f"{name} parsed as {type(value).__name__}"
        )

    for technology in raw["population"]["technologies"]:
        for field, value in technology["weibull"].items():
            assert isinstance(value, (int, float)), f"{technology['name']}.{field}"
