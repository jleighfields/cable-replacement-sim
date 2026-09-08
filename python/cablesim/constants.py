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

An implementation reads the width off the dtype of the arrays it is handed,
so nothing downstream branches on the name.

**The two are not held to the same standard, and that is deliberate.** At
``"f64"`` two implementations must agree in every cell with no tolerance, and
that comparison is what validates the kernel against the reference. At
``"f32"`` they are held to four significant figures, because exactness at that
width would need NumPy and the crate to return the same bits from the C
library's single-precision ``expm1``, ``log1p`` and ``pow`` — none of which is
required to be correctly rounded, and which were measured disagreeing by one
representable step on some machines and not others. ``docs/single-precision.md``
has the measurement and what the looser comparison gives up.
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
