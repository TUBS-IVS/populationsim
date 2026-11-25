# PopulationSim
# See full license in LICENSE.txt.
"""Shared inputs for the tests that drive the ``summarize`` step directly
(``tests/test_bs_positive_weight_rows.py``, ``tests/test_summarize.py``).

A small four-level problem (REGION > PUMA > TRACT > TAZ, PUMA the seed
geography) with dense sub-geography weight tables, and a driver that runs
``summarize`` on it with the fork's ``STORE_POSITIVE_WEIGHT_ROWS_ONLY`` off or
on (on: the tables keep only their positive rows).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SETTING = "STORE_POSITIVE_WEIGHT_ROWS_ONLY"

GEOGRAPHIES = ["REGION", "PUMA", "TRACT", "TAZ"]


class Table:
    """Stand-in for a pipeline table: ``to_frame`` returns a copy."""

    def __init__(self, df):
        self.df = df

    def to_frame(self):
        return self.df.copy()


def positive_rows(df):
    """The rows a weight table keeps with ``STORE_POSITIVE_WEIGHT_ROWS_ONLY``."""
    mask = df["balanced_weight"] > 0
    if "integer_weight" in df:
        mask = mask | (df["integer_weight"] > 0)
    return df[mask]


def summarize_problem():
    """PUMA 1 (households 1-6) holds TRACT 10 (TAZ 1, 2) and TRACT 11 (TAZ 3);
    PUMA 2 (households 7-9) holds TRACT 12 (TAZ 4), whose controls are zero, so
    none of its zones is balanced and its households have no dense row. TRACT 11
    and TAZ 2 have only zero weights.

    Returns ``(incidence, crosswalk, spec, control_tables, weights)``.
    """
    hh = pd.Index(range(1, 10), name="hh_id", dtype=np.int64)
    incidence = pd.DataFrame(
        {
            "num_hh": 1,
            "hh_size_1": [1, 0, 1, 0, 0, 1, 1, 0, 1],
            "PUMA": [1] * 6 + [2] * 3,
            "REGION": 1,
            "sample_weight": 10.0,
        },
        index=hh,
    )
    crosswalk = pd.DataFrame(
        {"TAZ": [1, 2, 3, 4], "TRACT": [10, 10, 11, 12], "PUMA": [1, 1, 1, 2]}
    )
    crosswalk["REGION"] = 1
    spec = pd.DataFrame(
        {
            "target": ["num_hh", "hh_size_1"],
            "geography": ["TAZ", "TAZ"],
            "seed_table": "households",
            "importance": [1000000, 1000],
            "control_field": ["HHS", "HH1"],
        }
    )

    def controls(name, ids, num_hh, size_1):
        return pd.DataFrame(
            {"num_hh": num_hh, "hh_size_1": size_1},
            index=pd.Index(ids, name=name, dtype=np.int64),
        )

    control_tables = {
        "TAZ": controls("TAZ", [1, 2, 3], [5, 3, 4], [2, 1, 1]),
        "TRACT": controls("TRACT", [10, 11], [8, 4], [3, 1]),
        "PUMA": controls("PUMA", [1], [12], [4]),
        "REGION": controls("REGION", [1], [12], [4]),
    }
    seed = pd.DataFrame(
        {
            "preliminary_balanced_weight": np.linspace(0.5, 4.5, 9),
            "balanced_weight": np.linspace(0.25, 4.25, 9),
            "integer_weight": np.array([1, 0, 2, 3, 1, 4, 0, 0, 0]),
        },
        index=hh,
    )
    seed["PUMA"] = incidence["PUMA"]
    rng = np.random.default_rng(7)

    def dense(geography, zones):
        frames = []
        for zone, parent, zero in zones:
            df = pd.DataFrame(index=list(range(0, 6)))
            df["hh_id"] = hh.values[:6]
            df[geography] = zone
            weights = rng.uniform(0.1, 3.0, 6) * (rng.uniform(size=6) > 0.4)
            df["balanced_weight"] = 0.0 if zero else weights
            df["integer_weight"] = np.round(0.0 * weights if zero else weights).astype(
                int
            )
            if geography == "TAZ":
                df["TRACT"] = parent
            df["REGION"] = 1
            df["PUMA"] = 1
            frames.append(df)
        return pd.concat(frames)

    weights = {
        "PUMA": seed,
        "TRACT": dense("TRACT", [(10, 1, False), (11, 1, True)]),
        "TAZ": dense("TAZ", [(1, 10, False), (2, 10, True), (3, 11, False)]),
    }
    return incidence, crosswalk, spec, control_tables, weights


def run_summarize(monkeypatch, settings, problem, positive_only):
    """Run the ``summarize`` step on ``problem`` (see :func:`summarize_problem`)
    and return its output tables by name. ``settings`` is the dict of
    ``tests.settings_isolation.patch_settings``; the step's table readers and
    ``out_table`` are replaced through ``monkeypatch``."""
    from populationsim.core import inject
    from populationsim.steps import summarize as step

    incidence, crosswalk, spec, control_tables, weights = problem
    settings.update(
        {
            "geographies": GEOGRAPHIES,
            "seed_geography": "PUMA",
            "household_id_col": "hh_id",
            "total_hh_control": "num_hh",
            SETTING: positive_only,
        }
    )
    stored = dict(weights)
    if positive_only:
        for g in ("TRACT", "TAZ"):
            stored[g] = positive_rows(weights[g])
    out = {}
    monkeypatch.setattr(inject, "get_step_arg", lambda name, default=None: default)
    monkeypatch.setattr(step, "get_control_table", lambda g: control_tables[g])
    monkeypatch.setattr(
        step, "get_weight_table", lambda g, sparse=False: stored[g].copy()
    )
    monkeypatch.setattr(step, "out_table", lambda name, df: out.__setitem__(name, df))
    step.summarize(Table(crosswalk), Table(incidence), spec)
    return out
