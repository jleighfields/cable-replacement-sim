"""Configuration schema and loader.

One pydantic model is the single source of truth for every run knob. The same
validated object feeds the Python reference, the Rust kernel and the Shiny app,
so a knob the app can set and a notebook cannot is a knob that has escaped
this model.
"""

import pathlib
from typing import Literal

import pydantic
import yaml

from cablesim import constants


class LogNormalSpec(pydantic.BaseModel):
    """A lognormal draw described by its median and log-scale sigma.

    Median rather than mean because the median is the parameter an engineer
    can estimate from a plan set by eye.

    Attributes:
        dist: Discriminator naming the distribution family.
        median: Median of the distribution, in the caller's units.
        sigma: Standard deviation of the underlying normal.
    """

    dist: Literal["lognormal"]
    median: float = pydantic.Field(gt=0)
    sigma: float = pydantic.Field(gt=0)


class WeibullSpec(pydantic.BaseModel):
    """Conductor-level Weibull lifetime parameters.

    These describe a single conductor, not a segment. A segment with several
    conductors fails when its first conductor does, which shifts the effective
    scale down — the reduction the reference and the kernel both apply.

    Attributes:
        shape: Weibull shape parameter, above 1 for a wear-out hazard.
        scale: Weibull scale parameter, in years.
    """

    shape: float = pydantic.Field(gt=0)
    scale: float = pydantic.Field(gt=0)


class SegmentClass(pydantic.BaseModel):
    """One class of cable segment in the synthetic population.

    Attributes:
        name: Identifier for the class, used in result tables.
        share: Fraction of the population in this class.
        n_conductors: Conductors per segment; three-phase segments carry 3.
        length_ft: Segment length distribution, in feet.
        customers: Customers served per segment.
        cost_per_ft: Planned replacement cost per foot, in dollars.
        weibull: Conductor-level lifetime parameters.
    """

    name: str
    share: float = pydantic.Field(gt=0, le=1)
    n_conductors: int = pydantic.Field(ge=1)
    length_ft: LogNormalSpec
    customers: LogNormalSpec
    cost_per_ft: float = pydantic.Field(gt=0)
    weibull: WeibullSpec


class InitialAgeSpec(pydantic.BaseModel):
    """How the starting age of each segment is drawn.

    Attributes:
        dist: Discriminator naming the age model.
        install_year_range: Inclusive first and last install year.
        weights: Named weighting curve over install years.
    """

    dist: Literal["empirical_install_years"]
    install_year_range: tuple[int, int]
    weights: Literal["build_out_curve", "uniform"]

    @pydantic.field_validator("install_year_range")
    @classmethod
    def range_is_ordered(cls, value: tuple[int, int]) -> tuple[int, int]:
        """Rejects a reversed install-year range.

        Args:
            value: The first and last install year.

        Returns:
            The validated range.

        Raises:
            ValueError: If the first year is not before the last.
        """
        first, last = value
        if first >= last:
            raise ValueError(f"install_year_range must be increasing, got {value}")
        return value


class SimulationConfig(pydantic.BaseModel):
    """Replication and horizon settings.

    Attributes:
        n_years: Years simulated per replication.
        n_reps: Monte Carlo replications.
        seed: Base seed. The kernel derives a per-replication seed from it, so
            results are reproducible and independent of the order replications
            complete in under rayon.
    """

    n_years: int = pydantic.Field(ge=1)
    n_reps: int = pydantic.Field(ge=1)
    seed: int = pydantic.Field(ge=0)


class PopulationConfig(pydantic.BaseModel):
    """The synthetic segment population.

    Attributes:
        n_segments: Total segments generated.
        total_customers: System-wide customer count, the denominator for
            SAIFI and SAIDI.
        classes: The segment classes and their population shares.
        initial_age: How starting ages are drawn.
    """

    n_segments: int = pydantic.Field(ge=1)
    total_customers: int = pydantic.Field(ge=1)
    classes: list[SegmentClass] = pydantic.Field(min_length=1)
    initial_age: InitialAgeSpec

    @pydantic.field_validator("classes")
    @classmethod
    def shares_sum_to_one(cls, value: list[SegmentClass]) -> list[SegmentClass]:
        """Requires the class shares to partition the population.

        Args:
            value: The configured segment classes.

        Returns:
            The validated classes.

        Raises:
            ValueError: If the shares do not sum to 1 within floating-point
                tolerance.
        """
        total = sum(c.share for c in value)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"class shares must sum to 1, got {total}")
        return value


class FailureConfig(pydantic.BaseModel):
    """How conductor failures within a segment relate to one another.

    Attributes:
        conductor_dependence: ``iid`` treats conductors as independent. A
            shared-frailty model is a possible extension and is not
            implemented.
    """

    conductor_dependence: Literal["iid"]


class CostConfig(pydantic.BaseModel):
    """Replacement cost parameters.

    Attributes:
        emergency_multiplier: Emergency cost as a multiple of planned cost.
        mobilization_per_segment: Fixed cost per segment touched, in dollars.
        escalation_rate: Annual cost escalation, as a fraction.
    """

    emergency_multiplier: float = pydantic.Field(ge=1)
    mobilization_per_segment: float = pydantic.Field(ge=0)
    escalation_rate: float


class BudgetConfig(pydantic.BaseModel):
    """The annual capital constraint a policy spends against.

    Attributes:
        annual: Capital available in the first year, in dollars.
        escalation: Annual growth of that budget, as a fraction.
        emergency_charged_to_budget: Whether emergency replacements consume
            the capital budget or a separate operations bucket.
    """

    annual: float = pydantic.Field(ge=0)
    escalation: float
    emergency_charged_to_budget: bool


class ReliabilityConfig(pydantic.BaseModel):
    """Outage duration and customer-cost parameters.

    Attributes:
        outage_hours_emergency: Customer-hours lost per emergency failure.
        outage_hours_planned: Customer-hours lost per planned replacement,
            zero where the work is done under a planned transfer.
        voll_per_customer_hour: Value of lost load, in dollars per customer
            hour.
    """

    outage_hours_emergency: float = pydantic.Field(ge=0)
    outage_hours_planned: float = pydantic.Field(ge=0)
    voll_per_customer_hour: float = pydantic.Field(ge=0)


class PolicySpec(pydantic.BaseModel):
    """One replacement policy to simulate.

    Attributes:
        name: Policy identifier.
        params: Policy-specific settings, empty for policies that take none.
    """

    name: Literal[
        "run_to_failure",
        "age_threshold",
        "risk_ranked",
        "worst_first",
        "random",
    ]
    params: dict[str, float | str] = pydantic.Field(default_factory=dict)


class Config(pydantic.BaseModel):
    """The whole validated run configuration.

    Attributes:
        simulation: Replication and horizon settings.
        population: The synthetic segment population.
        failure: Conductor dependence model.
        costs: Replacement cost parameters.
        budget: The annual capital constraint.
        reliability: Outage duration and customer-cost parameters.
        policies: The policies to compare.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    simulation: SimulationConfig
    population: PopulationConfig
    failure: FailureConfig
    costs: CostConfig
    budget: BudgetConfig
    reliability: ReliabilityConfig
    policies: list[PolicySpec] = pydantic.Field(min_length=1)


def load_config(path: pathlib.Path | None = None) -> Config:
    """Loads and validates a configuration file.

    Args:
        path: YAML file to read. Defaults to the checked-in base
            configuration.

    Returns:
        The validated configuration.

    Raises:
        FileNotFoundError: If the file does not exist. Raised rather than
            falling back to a default, so a mistyped path surfaces here
            instead of three steps downstream as a surprising result.
        pydantic.ValidationError: If the file does not satisfy the schema.
    """
    config_path = constants.DEFAULT_CONFIG_PATH if path is None else pathlib.Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Config.model_validate(raw)
