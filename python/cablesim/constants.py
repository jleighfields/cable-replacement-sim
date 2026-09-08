"""Fixed values that no caller may override.

The test for membership here rather than on the config model: a value belongs
in this module if a caller overriding it would be a *bug*, and on the config
model if a caller legitimately overrides it for one run. Nothing here is a
tunable.
"""

import pathlib

import numpy as np

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

PRECISIONS: dict[str, type[np.floating]] = {
    "f64": np.float64,
    "f32": np.float32,
}
"""The working precisions an annual loop can be run at, by the name a run
records.

Here rather than on the configuration model because the *mapping* is fixed — a
caller who changed what ``"f32"`` means would be writing a bug — while *which*
of them a run uses is a choice, and that choice is a field on the model.

Both are exact for every implementation: an implementation reads the dtype of
the arrays it is handed, and two implementations at the same precision are held
to agreeing in every cell, with no tolerance either way. What changes between
them is the answer, not how closely the answers are compared.
"""

DEFAULT_PRECISION: str = "f64"
"""The precision a run uses unless one is named.

Double, because it is what every published figure was measured at and what the
saved results carry. Single is an option a run opts into, not a default that
would quietly restate existing numbers.
"""

MINUTES_PER_HOUR: float = 60.0
"""Converts configured restoration hours to the customer-minutes the
reliability indices are defined in.

The conversion happens once, where the per-segment outage columns are built,
so every name downstream says which unit it carries.
"""
