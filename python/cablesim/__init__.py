"""Simulation of underground cable failure and replacement policy.

The population this package simulates is entirely synthetic and generated
in-process from a seed. No observed utility data is used anywhere in this
project.

The compute kernel is a Rust extension module built by maturin; the pure
Python reference implementation mirrors it so the two can be compared. That
duplication is the validation strategy rather than an oversight.
"""

from cablesim.config import Config, load_config

__all__ = ["Config", "load_config"]
