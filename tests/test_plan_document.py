"""Structural checks on the standing project plan.

`PLAN.md` is long, heavily cross-referenced, and renumbered whenever a section
is inserted. This invariant has already been broken once: a citation pointed at
a section that no longer existed, and it survived a line-by-line scan because
it wrapped across a line break.
"""

from cablesim import constants

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


def test_the_plan_quotes_the_configuration_verbatim() -> None:
    """Section 3's YAML block is `configs/base.yaml`, not a copy of it.

    A schema restated in prose drifts from the file it describes, and the drift
    is silent: both look right in isolation. Embedding the file and checking it
    is the only version of this that stays true.
    """
    plan = helpers.PLAN_PATH.read_text(encoding="utf-8")
    config_text = constants.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").rstrip()

    fenced = [
        block.split("\n```", 1)[0]
        for block in plan.split("```yaml\n")[1:]
    ]

    assert config_text in fenced, (
        "PLAN.md section 3 is out of step with configs/base.yaml"
    )
