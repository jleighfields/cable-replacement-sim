"""Configuration schema and loader.

One pydantic model is the single source of truth for every run knob. The same
validated object feeds the Python reference, the Rust kernel and the Shiny app,
so a knob the app can set and a notebook cannot is a knob that has escaped this
model.

Most validators here guard a failure that would otherwise surface as a wrong
number rather than an error: a customer type configured in one block and
missing from another contributes zero, a technology vintage gap leaves segments
with no parameters, and a misspelled policy parameter silently leaves the
policy on its default.
"""

import pathlib
from typing import Annotated, Literal

import pydantic
import yaml

from cablesim import constants


class LogNormalSpec(pydantic.BaseModel):
    """A lognormal draw described by its median and log-scale sigma.

    Median rather than mean because the median is the parameter an engineer can
    estimate from a plan set by eye. The mean is ``median * exp(sigma**2 / 2)``,
    which is the quantity to use when reasoning about totals.

    Attributes:
        dist: Discriminator naming the distribution family.
        median: Median of the distribution, in the caller's units. Strictly
            positive, because a lognormal with a median of zero would need the
            logarithm of zero; a quantity that should almost never occur takes
            a small median and rounds to zero instead.
        sigma: Standard deviation of the underlying normal.
    """

    dist: Literal["lognormal"]
    median: float = pydantic.Field(gt=0)
    sigma: float = pydantic.Field(gt=0)


class WeibullSpec(pydantic.BaseModel):
    """Weibull lifetime parameters for one conductor at the reference length.

    These describe a single conductor of length ``population.length_ref_ft``,
    not a segment. The effective-scale reduction turns the pair into the
    per-segment value the simulation uses; applying it to a segment-level scale
    would count conductor number and length twice.

    Attributes:
        shape: Weibull shape parameter, above 1 for a wear-out hazard.
        scale: Weibull scale parameter, in years.
    """

    shape: float = pydantic.Field(gt=0)
    scale: float = pydantic.Field(gt=0)


class Technology(pydantic.BaseModel):
    """One insulation technology and the vintage it was installed in.

    Failure behaviour follows technology and vintage, where class drives cost
    and customer exposure, so the Weibull pair keys here rather than on the
    class.

    Attributes:
        name: Identifier used in result tables and by ``replacement_technology``.
        vintage: Inclusive first and last install year for this technology.
        weibull: Conductor-level lifetime parameters at the reference length.
    """

    name: str
    vintage: tuple[int, int]
    weibull: WeibullSpec

    @pydantic.field_validator("vintage")
    @classmethod
    def vintage_is_ordered(cls, value: tuple[int, int]) -> tuple[int, int]:
        """Rejects a reversed vintage range.

        Args:
            value: The first and last install year.

        Returns:
            The validated range.

        Raises:
            ValueError: If the first year is after the last.
        """
        first, last = value
        if first > last:
            raise ValueError(f"vintage must be increasing, got {value}")
        return value


class SegmentClass(pydantic.BaseModel):
    """One class of cable segment in the synthetic population.

    Attributes:
        name: Identifier for the class, used in result tables and as the key
            for the per-class restoration times.
        share: Fraction of the population in this class.
        n_conductors: Conductors per segment; three-phase segments carry 3.
        length_ft: Segment length distribution, in feet.
        cost_per_ft: Planned replacement cost per foot, in dollars. Keyed on
            class rather than technology because it is the cost of installing
            new cable and does not depend on what is being removed.
        weibull_shape: Weibull shape for this class's cable, where it differs
            from the technology default. Technology carries the parameters
            because failure behaviour follows insulation compound and vintage,
            but conductor size is a second axis the technology does not
            capture: large-conductor feeder cable is a different product from
            residential lateral cable of the same vintage. Left unset, the
            class takes the technology's shape.
        customer_mix: Customers served downstream, per customer type.
    """

    name: str
    share: float = pydantic.Field(gt=0, le=1)
    n_conductors: int = pydantic.Field(ge=1)
    length_ft: LogNormalSpec
    cost_per_ft: float = pydantic.Field(gt=0)
    weibull_shape: float | None = pydantic.Field(default=None, gt=0)
    customer_mix: dict[str, LogNormalSpec]


class InitialAgeSpec(pydantic.BaseModel):
    """How the starting install year of each segment is drawn.

    Attributes:
        dist: Discriminator naming the age model.
        install_year_range: Inclusive first and last install year.
        install_volume: Relative install volume at each breakpoint year,
            linearly interpolated between them and normalized to a probability
            mass function over integer years.
    """

    dist: Literal["empirical_install_years"]
    install_year_range: tuple[int, int]
    install_volume: dict[int, float]

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

    @pydantic.model_validator(mode="after")
    def volume_spans_the_range(self) -> "InitialAgeSpec":
        """Requires breakpoints to cover the range so interpolation never extrapolates.

        Returns:
            The validated specification.

        Raises:
            ValueError: If a breakpoint falls outside the range, if either
                endpoint is missing, or if any weight is negative.
        """
        first, last = self.install_year_range
        years = sorted(self.install_volume)
        if not years:
            raise ValueError("install_volume needs at least the two endpoints")
        if years[0] != first or years[-1] != last:
            raise ValueError(
                f"install_volume must include both endpoints {first} and {last}, "
                f"got {years[0]} to {years[-1]}"
            )
        if any(weight < 0 for weight in self.install_volume.values()):
            raise ValueError("install_volume weights must not be negative")
        if sum(self.install_volume.values()) <= 0:
            raise ValueError("install_volume weights must not all be zero")
        return self


class SimulationConfig(pydantic.BaseModel):
    """Replication, horizon and chunking settings.

    Attributes:
        n_years: Years simulated per replication.
        n_reps: Monte Carlo replications.
        seed: Base seed. Independent streams are spawned from it by purpose,
            then per replication, so a result reproduces and does not depend on
            the chunk size it was computed at.
        start_year: Year 0. A segment's age is ``start_year - install_year``,
            and present values are discounted to this year.
        chunk_reps: Replications per kernel call, which sizes the draw array.
            It is recorded because a run cannot be reproduced without it, and
            it changes no result.
    """

    n_years: int = pydantic.Field(ge=1)
    n_reps: int = pydantic.Field(ge=1)
    seed: int = pydantic.Field(ge=0)
    start_year: int
    chunk_reps: int = pydantic.Field(ge=1)


class PopulationConfig(pydantic.BaseModel):
    """The synthetic segment population.

    Attributes:
        n_segments: Total segments generated.
        total_customers: System-wide customer count, the denominator for SAIFI
            and SAIDI. Scaling ``n_segments`` without scaling this understates
            every index by the ratio.
        customer_types: The customer types every class and every value-of-lost-
            load entry must cover.
        length_ref_ft: The length the configured Weibull scales describe.
        length_exponent: How strongly length drives failure in the
            effective-scale reduction. 1 is the spatial-Poisson case; below 1
            reflects faults concentrating at splices and terminations.
        technologies: Insulation technologies and the vintages they cover.
        replacement_technology: What a replacement installs.
        classes: The segment classes and their population shares.
        initial_age: How install years are drawn.
    """

    n_segments: int = pydantic.Field(ge=1)
    total_customers: int = pydantic.Field(ge=1)
    customer_types: list[str] = pydantic.Field(min_length=1)
    length_ref_ft: float = pydantic.Field(gt=0)
    length_exponent: float = pydantic.Field(ge=0, le=1)
    technologies: list[Technology] = pydantic.Field(min_length=1)
    replacement_technology: str
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
            ValueError: If the shares do not sum to 1 within tolerance.
        """
        total = sum(c.share for c in value)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"class shares must sum to 1, got {total}")
        return value

    @pydantic.model_validator(mode="after")
    def cross_checks(self) -> "PopulationConfig":
        """Checks the parts of the population against one another.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If class or technology names repeat, if a class's
                customer mix does not cover exactly the configured types, if
                ``replacement_technology`` names nothing, or if the technology
                vintages do not partition the install-year range.
        """
        class_names = [c.name for c in self.classes]
        if len(set(class_names)) != len(class_names):
            raise ValueError(f"class names must be unique, got {class_names}")

        technology_names = [t.name for t in self.technologies]
        if len(set(technology_names)) != len(technology_names):
            raise ValueError(f"technology names must be unique, got {technology_names}")
        if self.replacement_technology not in technology_names:
            raise ValueError(
                f"replacement_technology {self.replacement_technology!r} is not a "
                f"configured technology: {technology_names}"
            )

        expected = set(self.customer_types)
        for segment_class in self.classes:
            if set(segment_class.customer_mix) != expected:
                raise ValueError(
                    f"class {segment_class.name!r} customer_mix covers "
                    f"{sorted(segment_class.customer_mix)}, expected "
                    f"{sorted(expected)}"
                )

        first, last = self.initial_age.install_year_range
        spans = sorted((t.vintage[0], t.vintage[1]) for t in self.technologies)
        if spans[0][0] != first or spans[-1][1] != last:
            raise ValueError(
                f"technology vintages must span {first} to {last}, "
                f"got {spans[0][0]} to {spans[-1][1]}"
            )
        for (_, prev_last), (next_first, _) in zip(spans, spans[1:], strict=False):
            if next_first != prev_last + 1:
                raise ValueError(
                    f"technology vintages must partition the install-year range "
                    f"with no gap or overlap; {prev_last} is followed by {next_first}"
                )
        return self


class RecordsConfig(pydantic.BaseModel):
    """The synthetic censored failure history the MLE fits.

    Sized by how many observed failures the recovery test needs rather than by
    how large a system is being modeled, which is why it is separate from the
    population.

    Attributes:
        n_segments: Segments drawn. The table has more rows than this, because
            a segment that fails inside the window contributes one row per
            installation episode.
        monitoring_start: Left-truncation point; no record exists before it.
        study_end: Right-censoring point.
    """

    n_segments: int = pydantic.Field(ge=1)
    monitoring_start: int
    study_end: int


class FailureConfig(pydantic.BaseModel):
    """How conductor failures within a segment relate to one another.

    Attributes:
        conductor_dependence: ``iid`` treats conductors as independent. A
            shared-frailty model is a documented extension and is not
            implemented, so it is not accepted here — a run that silently used
            independence while asking for dependence would look fine.
    """

    conductor_dependence: Literal["iid"]


class CostConfig(pydantic.BaseModel):
    """Replacement cost parameters.

    Attributes:
        emergency_multiplier: Emergency cost as a multiple of planned cost.
        mobilization_per_segment: Fixed cost per segment touched, in dollars.
            This is what makes short segments expensive per foot, and what
            makes ranking by score per dollar differ from ranking by score.
        escalation_rate: Annual cost escalation, as a fraction. It applies to
            every dollar quantity in the year.
        discount_rate: Annual discount rate for present-value reporting.
    """

    emergency_multiplier: float = pydantic.Field(ge=1)
    mobilization_per_segment: float = pydantic.Field(ge=0)
    escalation_rate: float
    discount_rate: float


class BudgetConfig(pydantic.BaseModel):
    """The annual capital constraint a policy spends against.

    Attributes:
        annual: Capital available in the first year, in dollars.
        escalation: Annual growth of that budget, as a fraction.
        emergency_charged_to_budget: Whether emergency replacements consume the
            capital budget or a separate operations bucket. When true,
            emergency spend is charged before the planned pass is scored.
    """

    annual: float = pydantic.Field(ge=0)
    escalation: float
    emergency_charged_to_budget: bool


class ReliabilityConfig(pydantic.BaseModel):
    """Customer restoration times and the value of lost load.

    Attributes:
        outage_hours_emergency: Restoration time per class after a failure, in
            hours — time until the customer is back on, not time to repair.
        outage_hours_planned: Restoration time per class for planned work, zero
            where the class can be switched out without interrupting anyone.
        voll_per_customer_hour: Value of lost load per customer type, in
            dollars per customer-hour.
    """

    # Non-negative throughout: a risk-ranked policy maximises the product of
    # these, so a negative entry ranks the segments it applies to as the ones
    # most worth leaving in the ground, and nothing downstream would raise.
    outage_hours_emergency: dict[str, Annotated[float, pydantic.Field(ge=0)]]
    outage_hours_planned: dict[str, Annotated[float, pydantic.Field(ge=0)]]
    voll_per_customer_hour: dict[str, Annotated[float, pydantic.Field(ge=0)]]


class PolicySpec(pydantic.BaseModel):
    """One replacement policy to simulate.

    Attributes:
        name: Policy identifier.
        params: Policy-specific settings, empty for policies that take none.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    name: Literal[
        "run_to_failure",
        "age_threshold",
        "risk_ranked",
        "worst_first",
        "random",
    ]
    params: dict[str, float | str] = pydantic.Field(default_factory=dict)

    @pydantic.model_validator(mode="after")
    def params_match_the_policy(self) -> "PolicySpec":
        """Rejects a parameter the named policy does not take.

        A misspelled parameter that is silently dropped leaves the policy
        running on its default and the run still looks fine, which is the
        failure this rejects rather than ignores.

        Returns:
            The validated policy.

        Raises:
            ValueError: If a key is not accepted by this policy name, or if
                ``rank_by`` is not one of the two ranking modes.
        """
        accepted: dict[str, set[str]] = {
            "run_to_failure": set(),
            "age_threshold": {"threshold_years"},
            "risk_ranked": {"rank_by"},
            "worst_first": set(),
            "random": set(),
        }
        # A parameter left out produces the same silent fallback to a default
        # as one misspelled, so both are rejected. `rank_by` is absent here
        # because ranking on the raw score is a meaningful default; an age
        # threshold has none, since it is the policy.
        required: dict[str, set[str]] = {"age_threshold": {"threshold_years"}}

        unexpected = set(self.params) - accepted[self.name]
        if unexpected:
            raise ValueError(
                f"policy {self.name!r} does not take {sorted(unexpected)}; "
                f"it accepts {sorted(accepted[self.name])}"
            )
        missing = required.get(self.name, set()) - set(self.params)
        if missing:
            raise ValueError(
                f"policy {self.name!r} requires {sorted(missing)}"
            )
        rank_by = self.params.get("rank_by")
        if rank_by is not None and rank_by not in ("score", "score_per_dollar"):
            raise ValueError(
                f"rank_by must be 'score' or 'score_per_dollar', got {rank_by!r}"
            )
        return self


class ReportingConfig(pydantic.BaseModel):
    """What the reported metrics are measured against.

    Attributes:
        baseline_policy: The policy that "avoided" quantities are compared to.
    """

    baseline_policy: str


class Config(pydantic.BaseModel):
    """The whole validated run configuration.

    Attributes:
        simulation: Replication, horizon and chunking settings.
        population: The synthetic segment population.
        records: The synthetic censored failure history.
        failure: Conductor dependence model.
        costs: Replacement cost parameters.
        budget: The annual capital constraint.
        reliability: Restoration times and the value of lost load.
        policies: The policies to compare.
        reporting: What "avoided" is measured against.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    simulation: SimulationConfig
    population: PopulationConfig
    records: RecordsConfig
    failure: FailureConfig
    costs: CostConfig
    budget: BudgetConfig
    reliability: ReliabilityConfig
    policies: list[PolicySpec] = pydantic.Field(min_length=1)
    reporting: ReportingConfig

    @pydantic.model_validator(mode="after")
    def cross_block_checks(self) -> "Config":
        """Checks the parts of the configuration that reference one another.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If a restoration-time map does not cover exactly the
                configured classes, if the value of lost load does not cover
                exactly the configured customer types, if the study window does
                not sit inside the install range and end by the simulation's
                first year, or if ``baseline_policy`` names no configured
                policy.
        """
        class_names = {c.name for c in self.population.classes}
        for field in ("outage_hours_emergency", "outage_hours_planned"):
            configured = set(getattr(self.reliability, field))
            if configured != class_names:
                raise ValueError(
                    f"reliability.{field} covers {sorted(configured)}, expected one "
                    f"entry per class: {sorted(class_names)}"
                )

        # A customer type sharing a name with a derived column would overwrite
        # it: the frame would carry that type's count where the sum belongs,
        # while the minute and cost columns still used the true sum.
        derived = {
            "customers",
            "customer_minutes_per_failure",
            "customer_minutes_per_planned",
            "outage_cost_per_failure",
        }
        colliding = derived & set(self.population.customer_types)
        if colliding:
            raise ValueError(
                f"customer types {sorted(colliding)} collide with derived column "
                f"names and would overwrite them"
            )

        types = set(self.population.customer_types)
        voll = set(self.reliability.voll_per_customer_hour)
        if voll != types:
            raise ValueError(
                f"reliability.voll_per_customer_hour covers {sorted(voll)}, expected "
                f"{sorted(types)}"
            )

        first, last = self.population.initial_age.install_year_range
        if last > self.simulation.start_year:
            raise ValueError(
                f"install_year_range ends at {last}, after simulation.start_year "
                f"{self.simulation.start_year}: cable cannot be installed after "
                f"year 0, and the negative age it produces makes every Weibull "
                f"form return NaN rather than raising"
            )
        if not first <= self.records.monitoring_start <= last:
            raise ValueError(
                f"records.monitoring_start {self.records.monitoring_start} must lie "
                f"in the install-year range {first} to {last}"
            )
        if not (
            self.records.monitoring_start
            < self.records.study_end
            <= self.simulation.start_year
        ):
            raise ValueError(
                f"records must satisfy monitoring_start < study_end <= start_year, "
                f"got {self.records.monitoring_start}, {self.records.study_end}, "
                f"{self.simulation.start_year}"
            )

        policy_names = {p.name for p in self.policies}
        if self.reporting.baseline_policy not in policy_names:
            raise ValueError(
                f"reporting.baseline_policy {self.reporting.baseline_policy!r} is not "
                f"a configured policy: {sorted(policy_names)}"
            )
        return self


def load_config(path: pathlib.Path | None = None) -> Config:
    """Loads and validates a configuration file.

    Args:
        path: YAML file to read. Defaults to the checked-in base configuration.

    Returns:
        The validated configuration.

    Raises:
        FileNotFoundError: If the file does not exist. Raised rather than
            falling back to a default, so a mistyped path surfaces here instead
            of three steps downstream as a surprising result.
        pydantic.ValidationError: If the file does not satisfy the schema.
    """
    config_path = constants.DEFAULT_CONFIG_PATH if path is None else pathlib.Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Config.model_validate(raw)
