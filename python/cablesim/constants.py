"""Fixed values that no caller may override.

The test for membership here rather than on the config model: a value belongs
in this module if a caller overriding it would be a *bug*, and on the config
model if a caller legitimately overrides it for one run. Nothing here is a
tunable.
"""

import pathlib

PROJECT_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent.parent
"""Repo root, derived from this file rather than the working directory.

A notebook run from ``notebooks/`` and the app run from the repo root have
different working directories, and anchoring to ``__file__`` makes both
resolve the same path.
"""

CONFIG_DIR: pathlib.Path = PROJECT_ROOT / "configs"
"""Directory holding the checked-in configuration files."""

DEFAULT_CONFIG_PATH: pathlib.Path = CONFIG_DIR / "base.yaml"
"""The documented default configuration."""

HOURS_PER_YEAR: float = 8760.0
"""Hours in a non-leap year, used to convert outage durations to rates."""
