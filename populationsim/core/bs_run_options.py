# PopulationSim
# See full license in LICENSE.txt.
"""Fork options for run time and memory.

Both default to the behaviour of upstream v0.10.0 and neither changes the
synthetic population: with an option on, ``expanded_household_ids`` is the
table an unchanged run produces, row for row.

``STORE_POSITIVE_WEIGHT_ROWS_ONLY``
    The weight tables of the sub-geographies keep only the rows whose balanced
    or integer weight is positive. Upstream stores one row per seed household
    group for every zone, almost all of them zero; the next geography then
    rebuilds the full-length parent weight vector with zeros when it reads the
    table.

``FAST_PATHS``
    Use the fork's faster implementations of steps whose result is identical
    bit for bit to the upstream code path (which stays in place and is what a
    run uses while the option is off).
"""

from __future__ import annotations

import numpy as np

from populationsim.core import config

STORE_POSITIVE_SETTING = "STORE_POSITIVE_WEIGHT_ROWS_ONLY"
FAST_PATHS_SETTING = "FAST_PATHS"


def _flag(key):
    value = config.setting(key, False)
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{key} must be true or false, got {value!r}")
    return bool(value)


def store_positive_weight_rows_only():
    """``STORE_POSITIVE_WEIGHT_ROWS_ONLY`` (default False)."""
    return _flag(STORE_POSITIVE_SETTING)


def fast_paths():
    """``FAST_PATHS`` (default False)."""
    return _flag(FAST_PATHS_SETTING)
