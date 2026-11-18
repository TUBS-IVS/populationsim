# PopulationSim
# See full license in LICENSE.txt.
"""Fork option INTEGERIZER_PATH_RECORD: a machine-readable solver-path record.

Every integerizer LP call and every parent outcome of the simultaneous
integerizer wrapper is appended as one row to ``integerizer_paths.csv`` in the
output directory. All functions are no-ops while the option is off (default).

The file is only ever appended to, never truncated: the caller must start each
run with an empty output directory (a resumed run appends duplicate rows).
The file name never carries the output file prefix. With the OR-Tools solver
``build_s`` / ``solve_s`` are measured; with CVXPY they stay empty.

Columns: :data:`COLUMNS` while ``INTEGERIZER_ROUNDING`` is ``round`` (the
header of ``v0.10.0+bs.1``, byte for byte), :data:`CONTROLLED_COLUMNS` while it
is ``controlled``. In controlled mode every row carries ``rounding``
(``controlled`` on simultaneous call rows and on outcome rows, ``round`` on
single-integerizer call rows, which keep upstream's rounding), and the
simultaneous call rows carry the stats of
:func:`populationsim.integerizing.controlled_lp.solve_controlled` (which
documents them); ``build_s`` is the LP assembly there and ``solve_s`` the LP
solves of all attempts plus the repair and its pricing solve. A NaN objective
(failed solve or failed repair) is written as an empty field.
"""

from __future__ import annotations

import csv
import logging
import math
import os

import numpy as np

from populationsim.core import config
from populationsim.integerizing import bs_options

logger = logging.getLogger(__name__)

FILE_NAME = "integerizer_paths.csv"
COLUMNS = (
    "row_type",
    "kind",
    "geography",
    "trace_label",
    "status",
    "zones",
    "rows",
    "build_s",
    "solve_s",
    "outcome",
    "rounded_zones",
    "draw",
    "random_seed",
)
# the stats of a controlled simultaneous call (controlled_lp.solve_controlled)
CONTROLLED_STATS = (
    "solver",
    "attempts",
    "x_count",
    "fractional_x",
    "repaired_x",
    "fractional_after_repair",
    "repair_failed",
    "slack_cell",
    "slack_parent_sub",
    "slack_parent",
    "slack_conserve",
    "hh_unreachable",
    "hh_outside_lp",
    "hh_excess",
    "objective_lp",
    "objective_repaired",
)
CONTROLLED_COLUMNS = COLUMNS + ("rounding",) + CONTROLLED_STATS

_logged = False


def check_usable():
    """Raise if the record is on while several processes would write one file."""
    if bs_options.path_record_enabled() and config.setting("multiprocess", False):
        raise ValueError(
            "INTEGERIZER_PATH_RECORD is not supported with multiprocess: true "
            "(one record file, several writers)"
        )


def _append(row):
    global _logged
    # never prefixed: the pipeline reader expects <output>/integerizer_paths.csv
    path = config.build_output_file_path(FILE_NAME, use_prefix=None)
    if not _logged:
        _logged = True
        logger.info("integerizer path record: appending to %s", path)
    columns = _columns()
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow({c: ("" if row.get(c) is None else row[c]) for c in columns})


def _columns():
    """The columns of the record in the configured rounding mode."""
    if bs_options.rounding_mode() == "controlled":
        return CONTROLLED_COLUMNS
    return COLUMNS


def _fmt_seconds(value):
    return "" if value is None else "%.6f" % value


def _fmt_stat(value):
    """A stat as a field: integers as such, other numbers with 9 significant
    digits, NaN and None empty."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_, int, np.integer)):
        return str(int(value))
    value = float(value)
    return "" if math.isnan(value) else "%.9g" % value


def _rounding(kind):
    """The ``rounding`` field of a row (``kind`` None for an outcome row);
    None (no column) in ``round`` mode."""
    if bs_options.rounding_mode() != "controlled":
        return None
    # the single integerizer always rounds as upstream
    return "round" if kind == "single" else "controlled"


def record_call(
    kind, trace_label, status, zones=None, rows=None, timing=None, extra=None
):
    """Append one ``call`` row (one LP call); no-op unless the record is on.

    ``extra``: the stats of a controlled simultaneous call; the keys in
    :data:`CONTROLLED_STATS` are written (in controlled mode only, as the
    columns exist only there), any other key is ignored.
    """
    if not bs_options.path_record_enabled():
        return
    check_usable()
    timing = timing or {}
    draw = bs_options.draw_method()
    row = {
        "row_type": "call",
        "kind": kind,
        "trace_label": trace_label,
        "status": status,
        "zones": zones,
        "rows": rows,
        "build_s": _fmt_seconds(timing.get("build_s")),
        "solve_s": _fmt_seconds(timing.get("solve_s")),
        "draw": draw,
        "random_seed": bs_options.random_seed() if draw != "off" else "",
        "rounding": _rounding(kind),
    }
    for name in CONTROLLED_STATS:
        if extra and name in extra:
            row[name] = _fmt_stat(extra[name])
    _append(row)


def record_outcome(outcome, geography="", trace_label="", zones=None, rounded_zones=None):
    """Append one ``outcome`` row (one parent); no-op unless the record is on."""
    if not bs_options.path_record_enabled():
        return
    check_usable()
    draw = bs_options.draw_method()
    _append(
        {
            "row_type": "outcome",
            "geography": geography,
            "trace_label": trace_label,
            "zones": zones,
            "outcome": outcome,
            "rounded_zones": rounded_zones,
            "draw": draw,
            "random_seed": bs_options.random_seed() if draw != "off" else "",
            "rounding": _rounding(None),
        }
    )


def reset_for_tests():
    """Forget that the path was logged (test helper)."""
    global _logged
    _logged = False
