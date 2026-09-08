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

The three index columns carry their standard utility names, spelled out here
once because the column names are the abbreviations alone::

    saifi  System Average Interruption Frequency Index
           interruptions per customer served in the year
    saidi  System Average Interruption Duration Index
           customer-minutes interrupted per customer served in the year
    caidi  Customer Average Interruption Duration Index
           SAIDI / SAIFI: minutes per customer actually interrupted

Customer minutes interrupted (CMI) — the undivided total the duration index
divides — is not a fourth column. It is the saved ``customer_minutes`` column
unchanged, so it is read under that name rather than copied under another one.

Both indices divide by the configured system-wide customer count, **not** by a
sum over segments: customers are counted downstream of each asset, so a
feeder's count already contains the laterals' below it and summing would
double-count along every radial path.
"""

import polars as pl

BAND_QUANTILES: tuple[float, float] = (0.1, 0.9)
"""The band's lower and upper quantiles.

Not a parameter: ``plots.trajectory`` builds the column names it reads as
``_p10`` and ``_p90`` literally, so any other pair produces columns it raises
on. Changing the band means changing both.
"""

UNPLANNED_COLUMNS: tuple[str, ...] = (
    "failures",
    "customers_interrupted",
    "customer_minutes",
)
"""What the reliability indices are built from."""


def indices_per_replication(
    frame: pl.LazyFrame, total_customers: float, by: tuple[str, ...] = ()
) -> pl.LazyFrame:
    """Totals over segment classes and forms the indices, per replication year.

    The class axis is summed away here rather than in an implementation,
    because a system total cannot be decomposed afterwards and the per-class
    detail is what failure-by-class figures are drawn from.

    Args:
        frame: Saved rows, as a run wrote them.
        total_customers: System-wide customers served, the denominator of both
            indices; a configured figure rather than a sum over segments.
        by: Extra columns to keep as grouping keys, for a frame holding more
            than one run. A sweep's rows differ only in a swept column, so
            without naming it here two budget levels for one policy,
            replication and year are summed into a single row: nothing raises,
            the shape stays plausible, and the swept column disappears.

    Returns:
        One row per policy, replication and year, carrying every saved
        quantity summed over classes plus ``saifi``, ``saidi``, ``caidi``,
        ``total_spend`` and ``failure_cost``. Customer minutes interrupted
        stays under its saved name, ``customer_minutes``.

    Examples:
        The input is the saved frame, one row per class::

            >>> saved.select("policy", "replication", "year", "class",
            ...              "customers_interrupted", "customer_minutes")
            risk_ranked  0  0  main_feeder  100.0  6000.0
            risk_ranked  0  0  lateral_1ph  150.0  1500.0

            >>> indices_per_replication(saved.lazy(), 1000.0).collect().select(
            ...     "customers_interrupted", "saifi", "saidi", "caidi")
            250.0  0.25  7.5  30.0

        The class axis is gone and the surviving keys are exactly ``policy``,
        ``replication`` and ``year``.
    """
    keys = ["policy", "replication", "year", *by]
    totals = frame.group_by(keys).agg(
        pl.col(
            *UNPLANNED_COLUMNS,
            "planned_customer_minutes",
            "planned_replacements",
            "planned_spend",
            "emergency_spend",
            "voll",
        ).sum()
    )
    return totals.with_columns(
        saifi=pl.col("customers_interrupted") / total_customers,
        saidi=pl.col("customer_minutes") / total_customers,
        # No `cmi` column: customer minutes interrupted is `customer_minutes`
        # unchanged, and the module docstring says so once.
        total_spend=pl.col("planned_spend") + pl.col("emergency_spend"),
        # What a year of failures costs, which is not what it costs the
        # utility: the emergency bill is money spent and the value of lost
        # load is value destroyed. They are added here because the frontier
        # plots their sum against the planned spend that bought it down, and
        # kept out of `total_spend` because that column is the utility's own
        # outlay and the reliability-against-budget figures divide by it.
        failure_cost=pl.col("emergency_spend") + pl.col("voll"),
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
    does not need to know about it. Utility planning is done on a present-value
    basis, and over a thirty-year horizon an undiscounted total weights a
    year-29 dollar the same as a year-0 one.

    Args:
        frame: Rows carrying a ``year`` column and the dollar columns, as
            ``indices_per_replication`` leaves them.
        rate: Annual discount rate, as a fraction.

    Returns:
        The frame with a discounted counterpart for each spend column.
    """
    factor = (1.0 + rate) ** (-pl.col("year").cast(pl.Float64))
    return frame.with_columns(
        [
            (pl.col(name) * factor).alias(f"{name}_discounted")
            for name in (
                "planned_spend",
                "emergency_spend",
                "total_spend",
                "voll",
                "failure_cost",
            )
        ]
    )


def summarize_replications(
    frame: pl.LazyFrame, columns: list[str], by: tuple[str, ...] = ()
) -> pl.LazyFrame:
    """Summarizes across replications into a mean and an interval.

    The replication axis cannot be recovered once it is averaged away, which is
    why it is saved rather than summarized, and why this is a reduction over
    the saved frame rather than something a run writes.

    Args:
        frame: Per-replication rows.
        columns: Which quantities to summarize.
        by: Extra columns to keep as grouping keys, as in ``indices_per_replication``.

    Returns:
        One row per policy and year, plus one per extra grouping key. Each
        name in ``columns`` becomes three: ``<name>_mean``, and a pair named
        for ``BAND_QUANTILES`` as whole percents — ``<name>_p10`` and
        ``<name>_p90``, which is what ``plots.trajectory`` looks for.
    """
    lower, upper = BAND_QUANTILES
    return (
        frame.group_by(["policy", "year", *by])
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
        .sort(["policy", "year", *by])
    )


def horizon_totals(frame: pl.LazyFrame, by: tuple[str, ...] = ()) -> pl.LazyFrame:
    """Sums each replication over the whole horizon, then averages.

    Summing before averaging is what keeps the replication as the unit: the
    mean of a total is not the total of a mean once a policy's spend varies
    between replications.

    Args:
        frame: Per-replication rows, already discounted.
        by: Extra columns to keep as grouping keys, as in ``indices_per_replication``.

    Returns:
        One row per policy, and per extra grouping key.
    """
    summed = frame.group_by(["policy", "replication", *by]).agg(
        pl.col(
            "failures",
            "customer_minutes",
            "planned_customer_minutes",
            "planned_replacements",
            "planned_spend",
            "emergency_spend",
            "total_spend",
            "voll",
            "failure_cost",
            "planned_spend_discounted",
            "emergency_spend_discounted",
            "total_spend_discounted",
            "voll_discounted",
            "failure_cost_discounted",
        ).sum()
    )
    return (
        summed.group_by(["policy", *by])
        .agg(pl.exclude("policy", "replication", *by).mean())
        .sort(["policy", *by])
    )


def against_baseline(
    totals: pl.LazyFrame, baseline_policy: str, by: tuple[str, ...] = ()
) -> pl.LazyFrame:
    """Measures each policy against the one avoided quantities are relative to.

    The comparison is a difference of means over the same draws rather than an
    independent comparison: every policy reads the identical random field, so
    the difference carries far less noise than either side does. That pairing
    is the entire reason the draws are held fixed, and it is why the saved rows
    keep their replication axis.

    Args:
        totals: Horizon totals, one row per policy and, where ``by`` is given,
            one per policy and level.
        baseline_policy: What "avoided" is measured against.
        by: Extra grouping keys the totals carry. Each level gets its own
            baseline row, because a policy at one budget level is not measured
            against run-to-failure at a different one.

    Returns:
        The totals with avoided customer-minutes, avoided failures, and cost
        per customer-minute avoided.

        **Cost per customer-minute avoided is negative where prevention pays
        for itself**, and that is a result rather than an error: replacing a
        segment before it fails costs the planned price where letting it fail
        costs a multiple of it, so a programme can avoid more emergency spend
        than the planned work it buys. It turns positive once the cheap
        opportunities are used up — which need not happen inside any particular
        budget range, and does not inside the one the shipped configuration
        brackets.

    Raises:
        ValueError: If the baseline policy does not appear once per level of
            ``by`` — not at all, or at only some of them. Either would leave
            rows whose avoided columns are null or attached to the wrong
            level, and both look computed.
    """
    collected = totals.collect()
    matching = collected.filter(pl.col("policy") == baseline_policy)
    # Every level needs exactly one baseline row. Counting baseline rows alone
    # is not enough — two at one level and none at another counts correctly
    # while duplicating the first and dropping the second — so the count of
    # distinct levels is compared against both the baseline's row count and the
    # levels present. With no grouping keys there is one level, so the same two
    # comparisons hold with the counts fixed at one rather than derived, and
    # the duplicate case is caught in that path too.
    levels = collected.select(by).unique().height if by else 1
    baseline_levels = matching.select(by).unique().height if by else 1
    if matching.is_empty():
        present = (
            collected["policy"].unique().to_list()
            if not collected.is_empty()
            else "no rows at all"
        )
        raise ValueError(
            f"baseline policy {baseline_policy!r} is not in these results "
            f"({present}); every avoided quantity would be null"
        )
    if baseline_levels != matching.height:
        raise ValueError(
            f"baseline policy {baseline_policy!r} has {matching.height} rows "
            f"across {baseline_levels} level(s) of {list(by)}; each level needs "
            f"exactly one baseline row or its rows would be duplicated"
        )
    if baseline_levels != levels:
        missing = (
            collected.select(by)
            .unique()
            .join(matching.select(by).unique(), on=list(by), how="anti")
        )
        raise ValueError(
            f"baseline policy {baseline_policy!r} is missing at "
            f"{missing.height} level(s) of {list(by)}: "
            f"{missing.head(5).to_dicts()}; those levels would be dropped from "
            f"the comparison rather than reported as incomparable"
        )

    # Joined on the extra keys rather than read as scalars, so each swept level
    # is measured against its own baseline. Comparing a policy at one budget
    # level against run-to-failure at another would attribute the difference
    # between two budgets to the difference between two policies.
    reference = matching.select(
        [*by, "customer_minutes", "failures", "total_spend_discounted"]
    ).rename(
        {
            "customer_minutes": "baseline_customer_minutes",
            "failures": "baseline_failures",
            "total_spend_discounted": "baseline_total_spend_discounted",
        }
    )
    # A cross join with no keys attaches the single baseline row to every row;
    # with keys it attaches each level's own. Both are the same statement about
    # what a baseline is, so they are written as one.
    joined = (
        collected.join(reference, on=list(by), how="inner")
        if by
        else collected.join(reference, how="cross")
    )

    return (
        joined.lazy()
        .with_columns(
            customer_minutes_avoided=pl.col("baseline_customer_minutes")
            - pl.col("customer_minutes"),
            failures_avoided=pl.col("baseline_failures") - pl.col("failures"),
            additional_spend_discounted=pl.col("total_spend_discounted")
            - pl.col("baseline_total_spend_discounted"),
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
        .drop(
            "baseline_customer_minutes",
            "baseline_failures",
            "baseline_total_spend_discounted",
        )
    )
