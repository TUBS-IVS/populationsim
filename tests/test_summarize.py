# PopulationSim
# See full license in LICENSE.txt.
"""Fork: the ``_diff`` columns of the zone summaries (``summarize_geography``) are
``_result - _control`` of the same zone, whatever the order of the control table.

The results are aggregated with ``groupby`` (sorted zone ids) and the controls are
in the order of the control table; the difference is taken after aligning the
results to the control order by zone id.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populationsim.steps import summarize as step
from tests.settings_isolation import patch_settings
from tests.test_bs_positive_weight_rows import _run_summarize, _summarize_problem


def _controls(ids, counts):
    return pd.DataFrame(
        {"count": counts}, index=pd.Index(ids, name="zone", dtype=np.int64)
    )


def _summarize(monkeypatch, controls, results):
    """``summarize_geography`` of geography ``zone`` with one control ``count``;
    every household has incidence 1, so a zone's result is its weight sum."""
    monkeypatch.setattr(step, "get_control_table", lambda g: controls)
    crosswalk = pd.DataFrame({"zone": controls.index.to_numpy()})
    incidence = pd.DataFrame(
        {"count": 1}, index=pd.Index([1, 2, 3, 4], name="hh_id", dtype=np.int64)
    )
    results_df = pd.DataFrame(results, columns=["zone", "hh_id", "integer_weight"])
    return step.summarize_geography(
        "zone", "integer_weight", "hh_id", crosswalk, results_df, incidence
    )


def test_diff_aligned_by_zone_id_when_control_order_differs(monkeypatch):
    controls = _controls([20, 10, 30], [5, 6, 7])
    # zone 10: 2 + 1, zone 20: 4; the groupby gives them as 10, 20
    results = [[10, 1, 2], [20, 2, 4], [10, 3, 1]]
    out = _summarize(monkeypatch, controls, results)

    assert out.index.tolist() == ["zone_20", "zone_10", "zone_30"]
    assert out.columns.tolist() == [
        "geography",
        "id",
        "count_control",
        "count_result",
        "count_diff",
    ]
    assert out["id"].tolist() == [20, 10, 30]
    assert out["count_control"].tolist() == [5, 6, 7]
    assert out["count_result"].tolist() == [4, 3, 0]
    assert out["count_diff"].tolist() == [-1, -3, -7]
    pd.testing.assert_series_equal(
        out["count_diff"],
        (out["count_result"] - out["count_control"]).astype(out["count_diff"].dtype),
        check_names=False,
    )


def test_zone_without_rows_gets_zero_result(monkeypatch):
    controls = _controls([10, 20, 30], [5, 6, 7])
    results = [[10, 1, 2], [30, 2, 4]]  # zone 20 has controls but no weight rows
    out = _summarize(monkeypatch, controls, results)

    assert out.index.tolist() == ["zone_10", "zone_20", "zone_30"]
    assert out.loc["zone_20", "count_result"] == 0
    assert out.loc["zone_20", "count_diff"] == -6
    assert out["count_diff"].tolist() == [-3, -6, -3]
    # the integer result keeps its dtype (no NaN introduced by the alignment)
    assert out["count_result"].dtype == np.int64


@pytest.mark.parametrize("taz_order", [[1, 2, 3], [3, 1, 2]])
def test_positive_rows_layout_gives_same_diff(monkeypatch, taz_order):
    """Dense weight tables and positive-only weight tables (the fork's
    ``STORE_POSITIVE_WEIGHT_ROWS_ONLY``, filled by ``_DenseLayout``) give the same
    zone summaries, with ``_diff == _result - _control`` row by row, also when the
    control table is not sorted."""
    settings = patch_settings(monkeypatch)
    problem = _summarize_problem()
    controls = problem[3]
    controls["TAZ"] = controls["TAZ"].loc[taz_order]

    dense = _run_summarize(monkeypatch, settings, problem, False)
    positive = _run_summarize(monkeypatch, settings, problem, True)

    zone_tables = [
        n for n in dense if n not in ("hh_weights",) and "aggregate" not in n
    ]
    zone_tables = [n for n in zone_tables if not n.startswith("REGION_")]
    assert sorted(zone_tables) == sorted(
        ["TRACT", "TRACT_PUMA", "TRACT_REGION", "TAZ", "TAZ_PUMA", "TAZ_REGION"]
    )
    assert dense["TAZ"]["id"].tolist() == taz_order
    for name in zone_tables:
        diff_cols = [c for c in dense[name] if c.endswith("_diff")]
        assert diff_cols == ["num_hh_diff", "hh_size_1_diff"]
        pd.testing.assert_frame_equal(
            positive[name][diff_cols], dense[name][diff_cols], check_exact=True
        )
        for table in (dense[name], positive[name]):
            for col in diff_cols:
                target = col[: -len("_diff")]
                np.testing.assert_array_equal(
                    table[col].to_numpy(),
                    table["%s_result" % target].to_numpy()
                    - table["%s_control" % target].to_numpy(),
                )
