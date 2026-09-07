"""Reducing saved results to the numbers a figure is drawn from.

Everything here is a pure function of the frame a run wrote. Nothing reads a
file, and nothing is computed inside an implementation of the annual loop:
implementations return counts and spend, and the ratios are derived once, here,
from either of them. That also means the reference and the kernel are compared
on the quantities they actually compute rather than on ratios that could agree
by cancellation.

**The reliability indices cover unplanned interruptions only.** That is how
they are conventionally defined, and it is the only way they stay coherent
here: a planned replacement that interrupts a lateral's customers would
otherwise land in the duration index while contributing nothing to the
frequency index, and their ratio would divide two different populations of
outage. The customer-minutes planned work does cost are reported beside them
instead of being dropped — a customer out for four hours does not care that the
work was scheduled — so aggressive replacement shows as a customer-minute cost
now against a reliability gain later, with neither distorting the other.
"""

import polars as pl

UNPLANNED_COLUMNS: tuple[str, ...] = (
    "failures",
    "customers_interrupted",
    "customer_minutes",
)
"""What the reliability indices are built from."""


def per_replication(frame: pl.LazyFrame, total_customers: float) -> pl.LazyFrame:
    """Totals over segment classes and forms the indices, per replication year.

    The class axis is summed away here rather than in an implementation,
    because a system total cannot be decomposed afterwards and the per-class
    detail is what failure-by-class figures are drawn from.

    The system customer count is a configured figure and **not** the sum over
    segments: customers are counted downstream of each asset, so a feeder's
    count already contains the laterals' below it and summing would
    double-count along every radial path.

    Args:
        frame: Saved rows, as a run wrote them.
        total_customers: System-wide customers served, the denominator of both
            indices.

    Returns:
        One row per policy, replication and year, carrying the indices and the
        totals they came from.
    """
    keys = ["policy", "replication", "year"]
    totals = frame.group_by(keys).agg(
        pl.col(
            *UNPLANNED_COLUMNS,
            "planned_customer_minutes",
            "planned_replacements",
            "planned_spend",
            "emergency_spend",
        ).sum()
    )
    return totals.with_columns(
        saifi=pl.col("customers_interrupted") / total_customers,
        saidi=pl.col("customer_minutes") / total_customers,
        cmi=pl.col("customer_minutes"),
        total_spend=pl.col("planned_spend") + pl.col("emergency_spend"),
    ).with_columns(
        # Average duration per customer interrupted. Undefined rather than zero
        # in a year nothing failed: dividing two zeros would report a
        # restoration time for outages that did not happen, and a mean over
        # replications would then pull toward a number nothing measured.
        caidi=pl.when(pl.col("saifi") > 0)
        .then(pl.col("saidi") / pl.col("saifi"))
        .otherwise(None)
    )


def discount(frame: pl.LazyFrame, rate: float) -> pl.LazyFrame:
    """Adds present-value columns for every dollar quantity.

    Costs accumulate in nominal terms inside the annual loop and are discounted
    here, because discounting is a reduction over a saved stream and the loop
    does not need to know about it. Over a thirty-year horizon an undiscounted
    total overstates late spending badly enough that cost per customer-minute
    avoided is hard to defend without it, since utility planning is done on a
    present-value basis.

    Args:
        frame: Rows carrying a ``year`` column and the spend columns.
        rate: Annual discount rate, as a fraction.

    Returns:
        The frame with a discounted counterpart for each spend column.
    """
    factor = (1.0 + rate) ** (-pl.col("year").cast(pl.Float64))
    return frame.with_columns(
        [
            (pl.col(name) * factor).alias(f"{name}_discounted")
            for name in ("planned_spend", "emergency_spend", "total_spend")
        ]
    )


def bands(
    frame: pl.LazyFrame, columns: list[str], quantiles: tuple[float, float] = (0.1, 0.9)
) -> pl.LazyFrame:
    """Summarizes across replications into a mean and an interval.

    The replication axis cannot be recovered once it is averaged away, which is
    why it is saved rather than summarized, and why this is a reduction over
    the saved frame rather than something a run writes.

    Args:
        frame: Per-replication rows.
        columns: Which quantities to summarize.
        quantiles: Lower and upper quantile for the band.

    Returns:
        One row per policy and year.
    """
    lower, upper = quantiles
    return (
        frame.group_by(["policy", "year"])
        .agg(
            [pl.col(name).mean().alias(f"{name}_mean") for name in columns]
            + [
                pl.col(name).quantile(lower).alias(f"{name}_p{int(lower * 100)}")
                for name in columns
            ]
            + [
                pl.col(name).quantile(upper).alias(f"{name}_p{int(upper * 100)}")
                for name in columns
            ]
        )
        .sort(["policy", "year"])
    )


def horizon_totals(frame: pl.LazyFrame, rate: float) -> pl.LazyFrame:
    """Sums each replication over the whole horizon, then averages.

    Summing before averaging is what keeps the replication as the unit: the
    mean of a total is not the total of a mean once a policy's spend varies
    between replications.

    Args:
        frame: Per-replication rows, already discounted.
        rate: Annual discount rate, used only to name what was applied.

    Returns:
        One row per policy.
    """
    summed = frame.group_by(["policy", "replication"]).agg(
        pl.col(
            "failures",
            "customer_minutes",
            "planned_customer_minutes",
            "planned_replacements",
            "planned_spend",
            "emergency_spend",
            "total_spend",
            "planned_spend_discounted",
            "emergency_spend_discounted",
            "total_spend_discounted",
        ).sum()
    )
    return (
        summed.group_by("policy")
        .agg(pl.exclude("policy", "replication").mean())
        .with_columns(discount_rate=pl.lit(rate))
        .sort("policy")
    )


def against_baseline(totals: pl.LazyFrame, baseline_policy: str) -> pl.LazyFrame:
    """Measures each policy against the one avoided quantities are relative to.

    The comparison is a difference of means over the same draws rather than an
    independent comparison: every policy reads the identical random field, so
    the difference carries far less noise than either side does. That pairing
    is the entire reason the draws are held fixed, and it is why the saved rows
    keep their replication axis.

    Args:
        totals: Horizon totals, one row per policy.
        baseline_policy: What "avoided" is measured against.

    Returns:
        The totals with avoided customer-minutes, avoided failures, and cost
        per customer-minute avoided.

    Raises:
        ValueError: If the baseline policy is not among the rows, which would
            otherwise silently produce nulls in every avoided column.
    """
    collected = totals.collect()
    matching = collected.filter(pl.col("policy") == baseline_policy)
    if matching.height != 1:
        raise ValueError(
            f"baseline policy {baseline_policy!r} is not in these results "
            f"({collected['policy'].to_list()}); every avoided quantity would "
            f"be null"
        )
    baseline = matching.row(0, named=True)

    return (
        collected.lazy()
        .with_columns(
            customer_minutes_avoided=pl.lit(baseline["customer_minutes"])
            - pl.col("customer_minutes"),
            failures_avoided=pl.lit(baseline["failures"]) - pl.col("failures"),
            additional_spend_discounted=pl.col("total_spend_discounted")
            - pl.lit(baseline["total_spend_discounted"]),
        )
        .with_columns(
            # Undefined where a policy avoided nothing, rather than infinite or
            # zero: a cost per unit of nothing is not a large number, it is not
            # a number.
            cost_per_customer_minute_avoided=pl.when(
                pl.col("customer_minutes_avoided") > 0
            )
            .then(
                pl.col("additional_spend_discounted")
                / pl.col("customer_minutes_avoided")
            )
            .otherwise(None)
        )
    )
