"""Reliability indices, discounting, and the comparison against a baseline.

Every case is a small frame whose answers can be worked out by hand. That is
worth more than comparing one reduction against another: an analytical check
can be wrong in only one way, where two computations agreeing can both be wrong
in the same way.
"""

import polars as pl
import pytest
from cablesim import metrics

TOTAL_CUSTOMERS = 1_000.0


def rows(**overrides: object) -> pl.LazyFrame:
    """Builds saved-shaped rows for one policy, replication and year.

    Args:
        **overrides: Column values to replace.

    Returns:
        A one-row lazy frame in the saved schema.
    """
    row: dict[str, object] = {
        "policy": "risk_ranked",
        "replication": 0,
        "year": 0,
        "class": "main_feeder",
        "failures": 2.0,
        "customers_interrupted": 100.0,
        "customer_minutes": 6_000.0,
        "planned_customer_minutes": 500.0,
        "planned_replacements": 3.0,
        "planned_spend": 4_500.0,
        "emergency_spend": 7_500.0,
    }
    row.update(overrides)
    return pl.LazyFrame([row])


def test_the_frequency_index_divides_customers_by_the_system_total() -> None:
    """One hundred customers out of a thousand is 0.1 interruptions each."""
    result = metrics.per_replication(rows(), TOTAL_CUSTOMERS).collect()

    assert result["saifi"].item() == pytest.approx(0.1)


def test_the_duration_index_divides_customer_minutes_by_the_same_total() -> None:
    """Six thousand customer-minutes over a thousand customers is six each."""
    result = metrics.per_replication(rows(), TOTAL_CUSTOMERS).collect()

    assert result["saidi"].item() == pytest.approx(6.0)
    assert result["cmi"].item() == pytest.approx(6_000.0)


def test_planned_work_enters_no_reliability_index() -> None:
    """Its minutes are reported beside them, never inside them.

    Adding planned minutes to the duration index while they contribute nothing
    to the frequency index would make their ratio divide two different
    populations of outage.
    """
    without = metrics.per_replication(rows(), TOTAL_CUSTOMERS).collect()
    with_more = metrics.per_replication(
        rows(planned_customer_minutes=999_999.0), TOTAL_CUSTOMERS
    ).collect()

    assert without["saidi"].item() == with_more["saidi"].item()
    assert without["saifi"].item() == with_more["saifi"].item()
    assert with_more["planned_customer_minutes"].item() == pytest.approx(999_999.0)


def test_average_duration_is_the_ratio_of_the_two_indices() -> None:
    """Six minutes each over 0.1 interruptions each is sixty minutes apiece."""
    result = metrics.per_replication(rows(), TOTAL_CUSTOMERS).collect()

    assert result["caidi"].item() == pytest.approx(60.0)


def test_average_duration_is_undefined_in_a_year_with_no_failures() -> None:
    """Zero over zero is not a restoration time of zero.

    Reported as zero it would drag a mean over replications toward a number
    nothing measured.
    """
    quiet = rows(customers_interrupted=0.0, customer_minutes=0.0, failures=0.0)

    result = metrics.per_replication(quiet, TOTAL_CUSTOMERS).collect()

    assert result["caidi"].item() is None


def test_classes_are_summed_before_the_indices_are_formed() -> None:
    """The indices are system-wide; the class axis is what figures split on."""
    two_classes = pl.concat(
        [
            rows(**{"class": "main_feeder", "customers_interrupted": 100.0}),
            rows(**{"class": "lateral_1ph", "customers_interrupted": 150.0}),
        ]
    )

    result = metrics.per_replication(two_classes, TOTAL_CUSTOMERS).collect()

    assert result.height == 1, "one row per policy, replication and year"
    assert result["saifi"].item() == pytest.approx(0.25)


def test_discounting_leaves_year_zero_alone_and_shrinks_later_years() -> None:
    """Present values are quoted at year 0, so year 0 is its own present value."""
    frame = pl.concat([rows(year=0), rows(year=2)])

    result = (
        metrics.discount(
            metrics.per_replication(frame, TOTAL_CUSTOMERS), rate=0.06
        )
        .collect()
        .sort("year")
    )

    assert result["total_spend"].to_list() == pytest.approx([12_000.0, 12_000.0])
    assert result["total_spend_discounted"].to_list() == pytest.approx(
        [12_000.0, 12_000.0 / 1.06**2]
    )


def test_a_horizon_total_sums_each_replication_before_averaging() -> None:
    """The mean of a total is not the total of a mean once replications differ.

    The extents are deliberately unequal: two replications over three years.
    With as many replications as years, grouping by the wrong one splits the
    same grand total into the same number of groups, and the mean of the group
    sums comes out identical — so a reduction over entirely the wrong axis
    passes.

    Here the replications total 600 and 1,400, for a mean of 1,000. Summed by
    year instead the groups are 1,000, 400 and 600, and their mean is 667.
    """
    spend = {0: (100.0, 300.0, 200.0), 1: (900.0, 100.0, 400.0)}
    frame = pl.concat(
        [
            rows(
                replication=replication,
                year=year,
                planned_spend=amount,
                emergency_spend=0.0,
            )
            for replication, amounts in spend.items()
            for year, amount in enumerate(amounts)
        ]
    )

    totals = metrics.horizon_totals(
        metrics.discount(metrics.per_replication(frame, TOTAL_CUSTOMERS), rate=0.0),
        rate=0.0,
    ).collect()

    assert totals["planned_spend"].item() == pytest.approx(1_000.0)


def test_bands_summarize_across_replications_rather_than_collapsing_years() -> None:
    """One row per policy and year, with the spread the replications showed."""
    frame = pl.concat(
        [
            rows(replication=index, customer_minutes=1_000.0 * index)
            for index in range(5)
        ]
    )

    banded = metrics.bands(
        metrics.per_replication(frame, TOTAL_CUSTOMERS), ["saidi"]
    ).collect()

    assert banded.height == 1
    assert banded["saidi_mean"].item() == pytest.approx(2.0)
    assert banded["saidi_p10"].item() < banded["saidi_p90"].item()


def test_avoided_minutes_are_measured_against_the_configured_baseline() -> None:
    """A policy that avoids nothing avoids zero, and the baseline avoids zero."""
    frame = pl.concat(
        [
            rows(policy="run_to_failure", customer_minutes=10_000.0),
            rows(policy="risk_ranked", customer_minutes=4_000.0),
        ]
    )

    compared = metrics.against_baseline(
        metrics.horizon_totals(
            metrics.discount(
                metrics.per_replication(frame, TOTAL_CUSTOMERS), rate=0.06
            ),
            rate=0.06,
        ),
        "run_to_failure",
    ).collect()

    avoided = dict(
        zip(
            compared["policy"].to_list(),
            compared["customer_minutes_avoided"].to_list(),
            strict=True,
        )
    )
    assert avoided["risk_ranked"] == pytest.approx(6_000.0)
    assert avoided["run_to_failure"] == pytest.approx(0.0)


def test_cost_per_minute_avoided_is_undefined_where_nothing_was_avoided() -> None:
    """A cost per unit of nothing is not a large number; it is not a number."""
    frame = pl.concat(
        [
            rows(policy="run_to_failure", customer_minutes=10_000.0),
            rows(policy="worst_first", customer_minutes=10_000.0),
        ]
    )

    compared = metrics.against_baseline(
        metrics.horizon_totals(
            metrics.discount(
                metrics.per_replication(frame, TOTAL_CUSTOMERS), rate=0.06
            ),
            rate=0.06,
        ),
        "run_to_failure",
    ).collect()

    assert compared["cost_per_customer_minute_avoided"].to_list() == [None, None]


def test_a_baseline_that_is_not_in_the_results_is_refused() -> None:
    """Otherwise every avoided column comes back null and looks computed."""
    totals = metrics.horizon_totals(
        metrics.discount(metrics.per_replication(rows(), TOTAL_CUSTOMERS), rate=0.06),
        rate=0.06,
    )

    with pytest.raises(ValueError, match="not in these results"):
        metrics.against_baseline(totals, "no_such_policy")
