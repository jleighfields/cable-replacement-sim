"""Figures the notebooks and the application share.

Every function takes a frame and returns a figure. None reads a file and none
draws to a global figure, which is what lets a notebook and a web application
call exactly the same code and get exactly the same picture. A figure earns a
place here when a second caller wants it; a one-off diagnostic stays in the
notebook that needs it.

One plotting backend, and it is an interactive one. The policy comparison is
read by hovering a year and comparing values across policies, which a static
image cannot do, and rendering the same figure twice for two front ends is the
duplication this module exists to remove.

Importing this module without the plotting extra installed fails here, at the
import, saying what is missing — rather than part way through building a
figure at the end of a long run.
"""

import plotly.graph_objects as go
import polars as pl

from cablesim import metrics

BAND_OPACITY = 0.18
"""How solid the shaded interval is behind its line."""


def band_colors(policies: list[str]) -> dict[str, str]:
    """Assigns one colour per policy, stable across figures.

    A policy keeping its colour between the reliability figure and the spend
    figure is what makes them readable side by side.

    Args:
        policies: Policy names, in the order they should be coloured.

    Returns:
        A colour per policy.
    """
    palette = [
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#C44E52",
        "#8172B3",
        "#937860",
    ]
    return {
        policy: palette[index % len(palette)] for index, policy in enumerate(policies)
    }


def trajectory(
    banded: pl.DataFrame, quantity: str, title: str, y_title: str
) -> go.Figure:
    """Draws one quantity against year, per policy, with its interval.

    Args:
        banded: Rows carrying ``policy``, ``year`` and the mean and quantile
            columns for ``quantity``.
        quantity: The base column name, without the summary suffix.
        title: Figure title.
        y_title: Axis label, including its unit.

    Returns:
        One line per policy, each behind a shaded interval.

    Raises:
        KeyError: If the summary columns for this quantity are absent, which
            would otherwise draw an empty figure that looks like a result.
    """
    # Derived from the quantiles the summary was built with, rather than
    # written out again here: two spellings of one pair drift apart, and the
    # drift shows up as a figure that raises rather than as a wrong number.
    lower, upper = (int(share * 100) for share in metrics.BAND_QUANTILES)
    needed = [f"{quantity}_mean", f"{quantity}_p{lower}", f"{quantity}_p{upper}"]
    missing = [name for name in needed if name not in banded.columns]
    if missing:
        raise KeyError(
            f"{missing} are not in the frame; summarize across replications "
            f"before plotting {quantity!r}"
        )

    policies = banded["policy"].unique(maintain_order=True).to_list()
    colors = band_colors(policies)
    figure = go.Figure()
    for policy in policies:
        rows = banded.filter(pl.col("policy") == policy).sort("year")
        years = rows["year"].to_list()
        figure.add_trace(
            go.Scatter(
                x=years + years[::-1],
                y=rows[f"{quantity}_p{upper}"].to_list()
                + rows[f"{quantity}_p{lower}"].to_list()[::-1],
                fill="toself",
                fillcolor=colors[policy],
                opacity=BAND_OPACITY,
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
                name=f"{policy} band",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=years,
                y=rows[f"{quantity}_mean"].to_list(),
                mode="lines",
                line={"color": colors[policy], "width": 2},
                name=policy,
            )
        )
    figure.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title=y_title,
        hovermode="x unified",
    )
    return figure


def failures_by_class(frame: pl.DataFrame, policy: str) -> go.Figure:
    """Stacks failures per year by segment class, for one policy.

    The class axis is carried all the way from the annual loop for this: a
    system total cannot be decomposed after the fact, and which class is
    failing is the difference between a story about feeders and a story about
    laterals.

    Args:
        frame: Saved rows, averaged over replications by class and year.
        policy: Which policy to show.

    Returns:
        One stacked bar series per segment class.
    """
    rows = (
        frame.filter(pl.col("policy") == policy)
        .group_by(["year", "class"])
        .agg(pl.col("failures").mean())
        .sort(["class", "year"])
    )
    figure = go.Figure()
    for name in rows["class"].unique(maintain_order=True).to_list():
        subset = rows.filter(pl.col("class") == name)
        figure.add_trace(
            go.Bar(
                x=subset["year"].to_list(),
                y=subset["failures"].to_list(),
                name=name,
            )
        )
    figure.update_layout(
        barmode="stack",
        title=f"Failures per year by class — {policy}",
        xaxis_title="Year",
        yaxis_title="Failures (replication mean)",
    )
    return figure


def spend_against_budget(
    banded: pl.DataFrame, budget: list[float], policy: str
) -> go.Figure:
    """Shows planned and emergency spend against the capital available.

    Planned spend sitting below the budget line means the constraint is not
    binding, which is the case where every ranking policy gives the same
    answer and the comparison has nothing to show.

    Args:
        banded: Rows carrying ``policy``, ``year`` and the mean spend columns.
        budget: Capital available per year, escalated, indexed by year.
        policy: Which policy to show.

    Returns:
        Two spend series and the budget line.
    """
    rows = banded.filter(pl.col("policy") == policy).sort("year")
    years = rows["year"].to_list()
    figure = go.Figure()
    for column, label in (
        ("planned_spend_mean", "Planned"),
        ("emergency_spend_mean", "Emergency"),
    ):
        figure.add_trace(go.Bar(x=years, y=rows[column].to_list(), name=label))
    figure.add_trace(
        go.Scatter(
            x=years,
            y=[budget[year] for year in years],
            mode="lines",
            line={"dash": "dash", "color": "#444444"},
            name="Planned budget",
        )
    )
    figure.update_layout(
        barmode="group",
        title=f"Spend against budget — {policy}",
        xaxis_title="Year",
        yaxis_title="Dollars (nominal, replication mean)",
    )
    return figure


def reliability_against_budget(
    sweep: pl.DataFrame, budget_column: str, quantity: str
) -> go.Figure:
    """The deliverable: what each policy buys as the budget varies.

    Every policy at zero budget must land on the same point, because none of
    them funds anything there. A curve that does not is reporting a difference
    the model cannot produce.

    Args:
        sweep: One row per policy and budget level.
        budget_column: The swept parameter's column.
        quantity: What to plot against it.

    Returns:
        One line per policy.
    """
    policies = sweep["policy"].unique(maintain_order=True).to_list()
    colors = band_colors(policies)
    figure = go.Figure()
    for policy in policies:
        rows = sweep.filter(pl.col("policy") == policy).sort(budget_column)
        figure.add_trace(
            go.Scatter(
                x=rows[budget_column].to_list(),
                y=rows[quantity].to_list(),
                mode="lines+markers",
                line={"color": colors[policy]},
                name=policy,
            )
        )
    figure.update_layout(
        title=f"{quantity} against annual budget",
        xaxis_title="Annual planned budget (dollars)",
        yaxis_title=quantity,
        hovermode="x unified",
    )
    return figure
