"""The benchmark figures, checked against each other wherever they are quoted.

`docs/compiled-and-threaded-python.md` holds the measurements; `README.md`,
`PLAN.md` and `deprecated/README.md` restate them in prose. That is a value
written in four places, and the drift is silent: every copy stays well-formed
markdown, nothing imports any of them, and the only reader who notices two
documents disagreeing is one who recomputes the ratio off the tables.

This asserts the copies agree, not what they say, so retaking a measurement
needs no edit here.
"""

import pytest

from tests import helpers


@pytest.mark.parametrize(
    "figure, pattern",
    [
        pytest.param(figure, pattern, id=figure)
        for figure, pattern in sorted(helpers.REPEATED_FIGURES.items())
    ],
)
def test_a_figure_quoted_in_several_documents_is_quoted_the_same_way(
    figure: str, pattern: str
) -> None:
    """Every document quoting one measurement quotes the same value for it.

    Args:
        figure: What the measurement is, for the failure message.
        pattern: The spelling to look for, from ``helpers.REPEATED_FIGURES``.
    """
    quoted: dict[tuple[str, ...], list[str]] = {}
    for path in helpers.tracked_documents():
        for value in helpers.quoted_figures(
            path.read_text(encoding="utf-8"), pattern
        ):
            quoted.setdefault(value, []).append(
                str(path.relative_to(helpers.PLAN_PATH.parent))
            )

    # A pattern that has stopped matching reports agreement by finding nothing,
    # which is the one way this check can pass without having checked anything.
    assert quoted, (
        f"no document quotes {figure}; the wording has moved and this pattern "
        f"now matches nothing, so it is asserting agreement over an empty set"
    )
    assert len(quoted) == 1, (
        f"documents disagree about {figure}: "
        + "; ".join(
            f"{'-'.join(value)} in {', '.join(sorted(where))}"
            for value, where in sorted(quoted.items())
        )
    )
