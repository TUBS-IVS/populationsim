# PopulationSim
# See full license in LICENSE.txt.
"""Fork option INTEGERIZER_PATH_RECORD: a machine-readable solver-path record.

Every integerizer LP call and every parent outcome of the simultaneous
integerizer wrapper is appended as one row to ``integerizer_paths.csv`` in the
output directory. All functions are no-ops while the option is off (default).
"""

from __future__ import annotations

import csv
import logging
import os

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
    path = config.output_file_path(FILE_NAME)
    if not _logged:
        _logged = True
        logger.info("integerizer path record: appending to %s", path)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow({c: ("" if row.get(c) is None else row[c]) for c in COLUMNS})


def _fmt_seconds(value):
    return "" if value is None else "%.6f" % value


def record_call(kind, trace_label, status, zones=None, rows=None, timing=None):
    """Append one ``call`` row (one LP call); no-op unless the record is on."""
    if not bs_options.path_record_enabled():
        return
    check_usable()
    timing = timing or {}
    draw = bs_options.draw_method()
    _append(
        {
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
        }
    )


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
        }
    )


def reset_for_tests():
    """Forget that the path was logged (test helper)."""
    global _logged
    _logged = False
