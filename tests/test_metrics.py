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
        metrics.discount(metrics.per_replication(frame, TOTAL_CUSTOMERS), rate=0.0)
    ).collect()

    assert totals["planned_spend"].item() == pytest.approx(1_000.0)


def test_bands_summarize_across_replications_rather_than_collapsing_years() -> None:
    """One row per policy and year, with the spread the replications showed.

    The years have to differ, and there have to be several. Built entirely at
    one year, this cannot tell grouping by year from not grouping by year at
    all — which is the one property its name claims.
    """
    frame = pl.concat(
        [
            rows(
                replication=replication,
                year=year,
                customer_minutes=1_000.0 * replication + 10_000.0 * year,
            )
            for replication in range(5)
            for year in range(3)
        ]
    )

    banded = metrics.bands(
        metrics.per_replication(frame, TOTAL_CUSTOMERS), ["saidi"]
    ).collect().sort("year")

    assert banded.height == 3, "one row per year, not one row overall"
    # Year means are 2, 12 and 22 customer-minutes per customer.
    assert banded["saidi_mean"].to_list() == pytest.approx([2.0, 12.0, 22.0])
    assert (banded["saidi_p10"] < banded["saidi_p90"]).all()


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
            )
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
            )
        ),
        "run_to_failure",
    ).collect()

    assert compared["cost_per_customer_minute_avoided"].to_list() == [None, None]


def test_a_baseline_that_is_not_in_the_results_is_refused() -> None:
    """Otherwise every avoided column comes back null and looks computed."""
    totals = metrics.horizon_totals(
        metrics.discount(metrics.per_replication(rows(), TOTAL_CUSTOMERS), rate=0.06)
    )

    with pytest.raises(ValueError, match="is not in these results"):
        metrics.against_baseline(totals, "no_such_policy")


def test_a_sweep_of_two_budget_levels_is_not_reduced_into_one_row() -> None:
    """A swept column has to be named, or the levels are summed together.

    A sweep is a directory of runs differing in a swept parameter, and that
    value is written into the rows as a column so the frame is readable without
    the layout that produced it. The reduction groups on policy, replication
    and year, so unless the swept column is named as a key too, two budget
    levels for one policy, replication and year collapse into a single row
    whose customer-minutes are their sum — nothing raises, the shape stays
    plausible, and the swept column disappears.
    """
    sweep = pl.concat(
        [
            rows(customer_minutes=6_000.0).with_columns(annual_budget=pl.lit(0.0)),
            rows(customer_minutes=12_000.0).with_columns(annual_budget=pl.lit(1e6)),
        ]
    )

    reduced = metrics.per_replication(
        sweep, TOTAL_CUSTOMERS, by=("annual_budget",)
    ).collect()

    assert reduced.height == 2, "the two budget levels were summed together"
    assert sorted(reduced["customer_minutes"].to_list()) == [6_000.0, 12_000.0]
    assert sorted(reduced["annual_budget"].to_list()) == [0.0, 1e6]


def test_every_reduction_keeps_the_swept_column_apart() -> None:
    """Not only the first: the bands and the horizon totals compose after it."""
    sweep = pl.concat(
        [
            rows(year=year, customer_minutes=1_000.0).with_columns(
                annual_budget=pl.lit(0.0)
            )
            for year in range(2)
        ]
        + [
            rows(year=year, customer_minutes=4_000.0).with_columns(
                annual_budget=pl.lit(1e6)
            )
            for year in range(2)
        ]
    )
    keys = ("annual_budget",)

    per = metrics.per_replication(sweep, TOTAL_CUSTOMERS, by=keys)
    banded = metrics.bands(per, ["saidi"], by=keys).collect()
    totals = metrics.horizon_totals(metrics.discount(per, 0.0), by=keys).collect()

    assert banded.height == 4, "two levels times two years"
    assert totals.height == 2, "one row per policy per budget level"
    assert sorted(totals["customer_minutes"].to_list()) == [2_000.0, 8_000.0]


def test_each_budget_level_is_measured_against_its_own_baseline() -> None:
    """Not against run-to-failure at some other level.

    Comparing across levels would charge the difference between two budgets to
    the difference between two policies, which is the whole quantity the figure
    is meant to show.
    """
    # The two levels' baselines must differ, or which one gets attached makes
    # no difference and a join that ignores the level entirely still passes.
    levels = []
    for budget, baseline_minutes, ranked_minutes in (
        (0.0, 10_000.0, 10_000.0),
        (1e6, 5_000.0, 2_000.0),
    ):
        for policy, minutes in (
            ("run_to_failure", baseline_minutes),
            ("risk_ranked", ranked_minutes),
        ):
            levels.append(
                rows(policy=policy, customer_minutes=minutes).with_columns(
                    annual_budget=pl.lit(budget)
                )
            )
    keys = ("annual_budget",)

    compared = metrics.against_baseline(
        metrics.horizon_totals(
            metrics.discount(
                metrics.per_replication(pl.concat(levels), TOTAL_CUSTOMERS, by=keys),
                rate=0.0,
            ),
            by=keys,
        ),
        "run_to_failure",
        by=keys,
    ).collect()

    # Four rows in, four rows out. A join that attaches every baseline to every
    # row duplicates them, and reading the result into a dictionary keyed on
    # policy and level would quietly collapse the duplicates again.
    assert compared.height == 4, "one row in, one row out"
    avoided = {
        (row["policy"], row["annual_budget"]): row["customer_minutes_avoided"]
        for row in compared.iter_rows(named=True)
    }
    assert avoided[("run_to_failure", 0.0)] == pytest.approx(0.0)
    assert avoided[("run_to_failure", 1e6)] == pytest.approx(0.0)
    assert avoided[("risk_ranked", 0.0)] == pytest.approx(0.0)
    # 5,000 against this level's own baseline; 8,000 against the other one.
    assert avoided[("risk_ranked", 1e6)] == pytest.approx(3_000.0)


def test_extra_spend_is_measured_in_the_direction_it_is_named() -> None:
    """A policy that spends more than the baseline shows a positive figure.

    Reversed, extra spending reads as a saving and the headline economic
    number — cost per customer-minute avoided — comes out negative, which
    reads as a policy that was paid to improve reliability.
    """
    frame = pl.concat(
        [
            rows(
                policy="run_to_failure",
                customer_minutes=10_000.0,
                planned_spend=0.0,
                emergency_spend=1_000.0,
            ),
            rows(
                policy="risk_ranked",
                customer_minutes=6_000.0,
                planned_spend=3_000.0,
                emergency_spend=500.0,
            ),
        ]
    )

    compared = metrics.against_baseline(
        metrics.horizon_totals(
            metrics.discount(
                metrics.per_replication(frame, TOTAL_CUSTOMERS), rate=0.0
            )
        ),
        "run_to_failure",
    ).collect()

    ranked = compared.filter(pl.col("policy") == "risk_ranked").row(0, named=True)
    # 3,500 spent against the baseline's 1,000, for 4,000 customer-minutes.
    assert ranked["additional_spend_discounted"] == pytest.approx(2_500.0)
    assert ranked["customer_minutes_avoided"] == pytest.approx(4_000.0)
    assert ranked["cost_per_customer_minute_avoided"] == pytest.approx(0.625)


def test_a_baseline_missing_at_one_level_is_refused() -> None:
    """Otherwise that level is dropped from the comparison without a word.

    Counting rows cannot see this: a baseline present twice at one level and
    absent at another counts correctly, duplicates the level it is present at,
    and loses the other.
    """
    frame = pl.concat(
        [
            rows(policy="run_to_failure").with_columns(annual_budget=pl.lit(0.0)),
            rows(policy="run_to_failure").with_columns(annual_budget=pl.lit(0.0)),
            rows(policy="risk_ranked").with_columns(annual_budget=pl.lit(1e6)),
        ]
    )
    keys = ("annual_budget",)
    totals = metrics.horizon_totals(
        metrics.discount(
            metrics.per_replication(frame, TOTAL_CUSTOMERS, by=keys), rate=0.0
        ),
        by=keys,
    )

    with pytest.raises(ValueError, match="missing at"):
        metrics.against_baseline(totals, "run_to_failure", by=keys)


def test_an_empty_set_of_totals_is_refused_rather_than_returned_empty() -> None:
    """An empty frame coming back looks exactly like a computed comparison."""
    # The column has to exist for the grouping to be valid; what is empty is
    # the set of rows, which is the state this refuses.
    no_rows = rows().with_columns(annual_budget=pl.lit(0.0)).filter(
        pl.col("policy") == "no such policy"
    )
    empty = metrics.horizon_totals(
        metrics.discount(
            metrics.per_replication(no_rows, TOTAL_CUSTOMERS, by=("annual_budget",)),
            rate=0.0,
        ),
        by=("annual_budget",),
    )

    with pytest.raises(ValueError, match="is not in these results"):
        metrics.against_baseline(empty, "run_to_failure", by=("annual_budget",))


def totals_row(policy: str, budget: float, minutes: float) -> dict[str, object]:
    """One row in the shape horizon totals produce.

    Built directly rather than through the reductions, because the grouping in
    ``horizon_totals`` collapses a duplicated baseline before it can reach the
    comparison — so the duplicate case is unreachable from that direction and
    would go untested.

    Args:
        policy: The policy name.
        budget: The swept budget level.
        minutes: Customer-minutes over the horizon.

    Returns:
        The row.
    """
    return {
        "policy": policy,
        "annual_budget": budget,
        "customer_minutes": minutes,
        "failures": 1.0,
        "total_spend_discounted": 100.0,
    }


def test_a_level_carrying_two_baseline_rows_is_refused() -> None:
    """Joining on it would duplicate every row at that level.

    Unreachable through the reductions, which group it away, and reachable by
    anyone calling the comparison with a frame of their own — which is what a
    public function has to survive.
    """
    totals = pl.LazyFrame(
        [
            totals_row("run_to_failure", 0.0, 10_000.0),
            totals_row("run_to_failure", 0.0, 9_000.0),
            totals_row("risk_ranked", 0.0, 4_000.0),
        ]
    )

    with pytest.raises(ValueError, match="more than once"):
        metrics.against_baseline(totals, "run_to_failure", by=("annual_budget",))


def test_one_baseline_a_level_is_accepted() -> None:
    """The same shape without the duplicate goes through, one row per row in."""
    totals = pl.LazyFrame(
        [
            totals_row("run_to_failure", 0.0, 10_000.0),
            totals_row("risk_ranked", 0.0, 4_000.0),
            totals_row("run_to_failure", 1e6, 8_000.0),
            totals_row("risk_ranked", 1e6, 3_000.0),
        ]
    )

    compared = metrics.against_baseline(
        totals, "run_to_failure", by=("annual_budget",)
    ).collect()

    assert compared.height == 4
    avoided = {
        (row["policy"], row["annual_budget"]): row["customer_minutes_avoided"]
        for row in compared.iter_rows(named=True)
    }
    assert avoided[("risk_ranked", 0.0)] == pytest.approx(6_000.0)
    assert avoided[("risk_ranked", 1e6)] == pytest.approx(5_000.0)
