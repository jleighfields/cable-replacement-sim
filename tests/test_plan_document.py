"""Structural checks on the standing project plan.

`PLAN.md` is long, heavily cross-referenced, and renumbered whenever a section
is inserted. This invariant has already been broken once: a citation pointed at
a section that no longer existed, and it survived a line-by-line scan because
it wrapped across a line break.
"""

from tests import helpers


def test_every_section_citation_resolves() -> None:
    """Every cited section number exists as a heading.

    The plan's convention is to cite a section by number *and* title, so a
    mismatch shows on sight. This checks the number half mechanically, which
    is the half that moves when a section is inserted.
    """
    text = helpers.PLAN_PATH.read_text(encoding="utf-8")

    unresolved = sorted(
        helpers.section_citations(text) - helpers.section_numbers(text)
    )

    assert not unresolved, f"citations name sections that do not exist: {unresolved}"
