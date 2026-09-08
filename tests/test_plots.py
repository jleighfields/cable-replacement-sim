"""Figures, checked on their data rather than their pixels.

An image comparison fails on a library upgrade that changed an anti-aliasing
default and passes on a chart plotting the wrong column, which is the wrong way
round. These assert the number of traces and the values attached to each, so a
figure that draws the wrong series fails and a figure that draws the right one
in a new shade does not.
"""

import polars as pl
import pytest
from cablesim import plots


def banded() -> pl.DataFrame:
    """Summary rows for two policies over three years.

    Returns:
        One row per policy and year, with mean and interval columns.
    """
    return pl.DataFrame(
        {
            "policy": ["run_to_failure"] * 3 + ["risk_ranked"] * 3,
            "year": [0, 1, 2] * 2,
            "saidi_mean": [1.0, 2.0, 3.0, 1.0, 1.5, 1.8],
            "saidi_p10": [0.5, 1.0, 2.0, 0.5, 1.0, 1.2],
            "saidi_p90": [1.5, 3.0, 4.0, 1.5, 2.0, 2.4],
            "planned_spend_mean": [0.0, 0.0, 0.0, 900.0, 950.0, 1_000.0],
            "emergency_spend_mean": [500.0, 700.0, 900.0, 400.0, 450.0, 500.0],
        }
    )


def test_a_trajectory_draws_a_line_and_a_band_for_every_policy() -> None:
    """Two traces apiece: the interval behind, the mean in front."""
    figure = plots.trajectory(banded(), "saidi", "Duration index", "Minutes")

    assert len(figure.data) == 4
    lines = [trace for trace in figure.data if trace.mode == "lines"]
    assert [trace.name for trace in lines] == ["run_to_failure", "risk_ranked"]


def test_a_trajectory_plots_the_values_it_was_given() -> None:
    """The means reach the figure, in year order, per policy."""
    figure = plots.trajectory(banded(), "saidi", "Duration index", "Minutes")

    lines = {
        trace.name: list(trace.y) for trace in figure.data if trace.mode == "lines"
    }
    assert lines["run_to_failure"] == [1.0, 2.0, 3.0]
    assert lines["risk_ranked"] == [1.0, 1.5, 1.8]


def test_a_band_traces_the_upper_bound_out_and_the_lower_bound_back() -> None:
    """A filled region needs its boundary as one closed path.

    Concatenated the wrong way, the shape crosses itself and the figure shades
    an area that is not the interval.
    """
    figure = plots.trajectory(banded(), "saidi", "Duration index", "Minutes")

    band = next(trace for trace in figure.data if trace.fill == "toself")
    assert list(band.x) == [0, 1, 2, 2, 1, 0]
    assert list(band.y) == [1.5, 3.0, 4.0, 2.0, 1.0, 0.5]


def test_a_policy_keeps_one_colour_across_figures() -> None:
    """Two figures read side by side need the same policy to look the same."""
    frame = banded()

    trajectory_colors = {
        trace.name: trace.line.color
        for trace in plots.trajectory(frame, "saidi", "t", "y").data
        if trace.mode == "lines"
    }
    sweep = pl.DataFrame(
        {
            "policy": ["run_to_failure", "risk_ranked"],
            "annual_budget": [0.0, 1.0],
            "customer_minutes": [10.0, 5.0],
        }
    )
    sweep_colors = {
        trace.name: trace.line.color
        for trace in plots.reliability_against_budget(
            sweep, "annual_budget", "customer_minutes"
        ).data
    }

    assert trajectory_colors == sweep_colors
    assert len(set(trajectory_colors.values())) == len(trajectory_colors), (
        "policies sharing a colour are consistent and unreadable"
    )


def test_plotting_an_unsummarized_frame_is_refused() -> None:
    """An empty figure looks exactly like a result showing no difference."""
    unsummarized = pl.DataFrame({"policy": ["risk_ranked"], "year": [0]})

    with pytest.raises(KeyError, match="summarize across replications"):
        plots.trajectory(unsummarized, "saidi", "t", "y")


def test_failures_are_stacked_one_series_per_class() -> None:
    """Which class is failing is the difference between two stories."""
    frame = pl.DataFrame(
        {
            "policy": ["risk_ranked"] * 4,
            "year": [0, 0, 1, 1],
            "class": ["main_feeder", "lateral_1ph"] * 2,
            "failures": [1.0, 5.0, 2.0, 6.0],
        }
    )

    figure = plots.failures_by_class(frame, "risk_ranked")

    assert figure.layout.barmode == "stack"
    series = {trace.name: list(trace.y) for trace in figure.data}
    assert series == {"lateral_1ph": [5.0, 6.0], "main_feeder": [1.0, 2.0]}


def test_only_the_requested_policy_is_drawn() -> None:
    """Otherwise every figure silently averages policies together."""
    frame = pl.DataFrame(
        {
            "policy": ["risk_ranked", "run_to_failure"],
            "year": [0, 0],
            "class": ["main_feeder", "main_feeder"],
            "failures": [1.0, 99.0],
        }
    )

    figure = plots.failures_by_class(frame, "risk_ranked")

    assert [list(trace.y) for trace in figure.data] == [[1.0]]


def test_spend_is_drawn_against_the_budget_line_it_competes_for() -> None:
    """Planned spend below the line means the constraint is not binding."""
    figure = plots.spend_against_budget(
        banded(), [1_000.0, 1_030.0, 1_060.9], "risk_ranked"
    )

    series = {trace.name: list(trace.y) for trace in figure.data}
    assert series["Planned"] == [900.0, 950.0, 1_000.0]
    assert series["Emergency"] == [400.0, 450.0, 500.0]
    assert series["Planned budget"] == [1_000.0, 1_030.0, 1_060.9]


def test_the_deliverable_draws_one_line_per_policy_against_the_budget() -> None:
    """Every policy at zero budget must land on the same point."""
    sweep = pl.DataFrame(
        {
            "policy": ["run_to_failure"] * 3 + ["risk_ranked"] * 3,
            "annual_budget": [0.0, 1e6, 2e6] * 2,
            "customer_minutes": [100.0, 100.0, 100.0, 100.0, 80.0, 70.0],
        }
    )

    figure = plots.reliability_against_budget(
        sweep, "annual_budget", "customer_minutes"
    )

    assert len(figure.data) == 2
    at_zero = {
        trace.name: list(trace.y)[list(trace.x).index(0.0)] for trace in figure.data
    }
    assert at_zero["run_to_failure"] == at_zero["risk_ranked"] == 100.0


def frontier_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Mean and per-replication horizon totals for two policies at two levels.

    The means are not the midpoints of the replications they summarize, so a
    figure that drew the cloud where the means belong, or averaged the cloud
    itself, does not pass by coincidence.

    Returns:
        The policy means, and the replications behind them.
    """
    totals = pl.DataFrame(
        {
            "policy": ["run_to_failure"] * 2 + ["risk_ranked"] * 2,
            "annual_budget": [0.0, 2e6] * 2,
            "planned_spend": [0.0, 0.0, 0.0, 1.8e6],
            "failure_cost": [9e6, 9e6, 9e6, 6e6],
        }
    )
    per_replication = pl.DataFrame(
        {
            "policy": ["run_to_failure"] * 4 + ["risk_ranked"] * 4,
            "replication": [0, 1] * 4,
            "annual_budget": [0.0, 0.0, 2e6, 2e6] * 2,
            "planned_spend": [0.0] * 6 + [1.7e6, 1.9e6],
            "failure_cost": [8e6, 1e7, 8e6, 1e7, 8e6, 1e7, 5e6, 7e6],
        }
    )
    return totals, per_replication


def test_the_frontier_draws_a_curve_and_a_cloud_for_every_policy() -> None:
    """The cloud is the figure's reason for existing, not decoration.

    Two costs that rise together within a replication are what error bars on
    each axis would hide, so a frontier that lost its cloud would still look
    like a finished figure.
    """
    totals, per_replication = frontier_frames()

    figure = plots.cost_frontier(totals, per_replication, "annual_budget")

    assert len(figure.data) == 4
    curves = [trace for trace in figure.data if trace.mode == "lines+markers"]
    clouds = [trace for trace in figure.data if trace.mode == "markers"]
    assert [trace.name for trace in curves] == ["run_to_failure", "risk_ranked"]
    assert len(clouds) == 2
    assert all(len(trace.x) == 4 for trace in clouds)


def test_the_frontier_plots_spend_incurred_rather_than_the_budget_offered() -> None:
    """A policy that cannot spend its allowance has to show that.

    ``run_to_failure`` is offered two million and funds nothing, so both its
    points sit at zero on the x-axis. Plotting the budget instead would walk
    it rightwards across a figure whose whole subject is what the money
    bought.
    """
    totals, per_replication = frontier_frames()

    figure = plots.cost_frontier(totals, per_replication, "annual_budget")

    curves = {
        trace.name: (list(trace.x), list(trace.y))
        for trace in figure.data
        if trace.mode == "lines+markers"
    }
    assert curves["run_to_failure"] == ([0.0, 0.0], [9e6, 9e6])
    assert curves["risk_ranked"] == ([0.0, 1.8e6], [9e6, 6e6])


def test_every_cloud_is_drawn_before_every_curve() -> None:
    """One loop apiece, so a policy's line is not buried under the next
    policy's replications.

    Interleaved, the last policy drawn covers the first, and at a thousand
    replications the curve underneath is invisible rather than merely faint.
    """
    totals, per_replication = frontier_frames()

    figure = plots.cost_frontier(totals, per_replication, "annual_budget")

    modes = [trace.mode for trace in figure.data]
    assert modes == ["markers", "markers", "lines+markers", "lines+markers"]


def test_a_frontier_frame_missing_its_columns_is_refused() -> None:
    """An empty figure looks exactly like a policy that bought nothing."""
    totals, per_replication = frontier_frames()

    with pytest.raises(KeyError, match="failure_cost"):
        plots.cost_frontier(
            totals.drop("failure_cost"), per_replication, "annual_budget"
        )
    with pytest.raises(KeyError, match="annual_budget"):
        plots.cost_frontier(totals, per_replication, "annual_budget_offered")
