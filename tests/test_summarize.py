# PopulationSim
# See full license in LICENSE.txt.
"""Fork: the ``_diff`` columns of the zone summaries (``summarize_geography``) are
``_result - _control`` of the same zone, whatever the order of the control table.

The results are aggregated with ``groupby`` (sorted zone ids) and the controls are
in the order of the control table; the difference is taken after aligning the
results to the control order by zone id.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from populationsim.steps import summarize as step
from tests.settings_isolation import patch_settings
from tests.summary_helpers import run_summarize, summarize_problem


def _controls(ids, counts):
    return pd.DataFrame(
        {"count": counts}, index=pd.Index(ids, name="zone", dtype=np.int64)
    )


def _summarize(monkeypatch, controls, results, weight_col="integer_weight"):
    """``summarize_geography`` of geography ``zone`` with one control ``count``;
    every household has incidence 1, so a zone's result is its weight sum.
    ``results`` are ``[zone, hh_id, weight]`` rows."""
    monkeypatch.setattr(step, "get_control_table", lambda g: controls)
    crosswalk = pd.DataFrame({"zone": controls.index.to_numpy()})
    incidence = pd.DataFrame(
        {"count": 1}, index=pd.Index([1, 2, 3, 4], name="hh_id", dtype=np.int64)
    )
    results_df = pd.DataFrame(results, columns=["zone", "hh_id", weight_col])
    results_df = results_df.astype({"zone": np.int64, "hh_id": np.int64})
    if weight_col == "integer_weight":
        results_df[weight_col] = results_df[weight_col].astype(np.int64)
    else:
        results_df[weight_col] = results_df[weight_col].astype(np.float64)
    return step.summarize_geography(
        "zone", weight_col, "hh_id", crosswalk, results_df, incidence
    )


def _summary_log(caplog):
    return [
        r
        for r in caplog.records
        if r.name == step.logger.name and r.getMessage().startswith("[summarize]")
    ]


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


def test_missing_zone_is_logged_as_a_warning_with_its_rate(
    monkeypatch, caplog, fork_logging
):
    controls = _controls([10, 20, 30], [5, 6, 7])
    with caplog.at_level(logging.INFO, logger=step.logger.name):
        _summarize(monkeypatch, controls, [[10, 1, 2], [30, 2, 4]])
    (record,) = _summary_log(caplog)
    assert record.levelno == logging.WARNING
    assert record.getMessage() == (
        "[summarize] zone: zones in the weight table 2/3 (66.7 %), "
        "missing 1 (33.3 %) filled with result 0"
    )


def test_no_missing_zone_is_logged_as_info(monkeypatch, caplog, fork_logging):
    controls = _controls([20, 10], [5, 6])
    with caplog.at_level(logging.INFO, logger=step.logger.name):
        _summarize(monkeypatch, controls, [[10, 1, 2], [20, 2, 4]])
    (record,) = _summary_log(caplog)
    assert record.levelno == logging.INFO
    assert record.getMessage() == (
        "[summarize] zone: zones in the weight table 2/2 (100.0 %), "
        "missing 0 (0.0 %) filled with result 0"
    )


def test_float_weights_with_unsorted_controls_keep_int_ids(monkeypatch):
    """The float-weight path (``balanced_weight``, as with
    ``NO_INTEGERIZATION_EVER: true``): the groupby index is float, the ids and
    labels follow the int control index."""
    controls = _controls([20, 10, 30], [5, 6, 7])
    results = [[10, 1, 2.5], [20, 2, 4.25], [30, 3, 1.0]]
    out = _summarize(monkeypatch, controls, results, weight_col="balanced_weight")

    assert out.index.tolist() == ["zone_20", "zone_10", "zone_30"]
    assert out["id"].tolist() == [20, 10, 30]
    assert out["id"].dtype == np.int64
    assert out["count_result"].dtype == np.float64
    assert out["count_result"].tolist() == [4.25, 2.5, 1.0]
    np.testing.assert_array_equal(
        out["count_diff"].to_numpy(),
        out["count_result"].to_numpy() - out["count_control"].to_numpy(),
    )
    assert out["count_diff"].tolist() == [-0.75, -3.5, -6.0]


@pytest.mark.parametrize(
    "weight_col, weight_dtype",
    [("integer_weight", np.int32), ("balanced_weight", np.float64)],
)
def test_uncast_weight_table_dtypes(monkeypatch, weight_col, weight_dtype):
    """The weight table is passed as is: int32 ids and int32 integer weights (the
    Windows default integer), float64 balanced weights."""
    controls = _controls([20, 10, 30], [5, 6, 7])
    monkeypatch.setattr(step, "get_control_table", lambda g: controls)
    crosswalk = pd.DataFrame({"zone": controls.index.to_numpy()})
    incidence = pd.DataFrame(
        {"count": 1}, index=pd.Index([1, 2, 3], name="hh_id", dtype=np.int32)
    )
    results_df = pd.DataFrame(
        {
            "zone": np.array([10, 20, 10], dtype=np.int32),
            "hh_id": np.array([1, 2, 3], dtype=np.int32),
            weight_col: np.array([2, 4, 1], dtype=weight_dtype),
        }
    )
    out = step.summarize_geography(
        "zone", weight_col, "hh_id", crosswalk, results_df, incidence
    )

    assert out["id"].tolist() == [20, 10, 30]
    assert out["count_result"].tolist() == [4, 3, 0]
    np.testing.assert_array_equal(
        out["count_diff"].to_numpy(),
        out["count_result"].to_numpy() - out["count_control"].to_numpy(),
    )
    assert out["count_diff"].tolist() == [-1, -3, -7]


@pytest.mark.parametrize(
    "weight_col, result_dtype",
    [("integer_weight", np.int64), ("balanced_weight", np.float64)],
)
def test_all_empty_geography_keeps_the_result_dtype(
    monkeypatch, weight_col, result_dtype
):
    """No weight rows in any zone: every zone gets result 0 in the dtype the
    non-empty path gives (no extra zero row, int ids)."""
    controls = _controls([20, 10], [5, 6])
    empty = np.zeros((0, 3))
    out = _summarize(monkeypatch, controls, empty, weight_col=weight_col)
    nonempty = _summarize(monkeypatch, controls, [[10, 1, 2]], weight_col=weight_col)

    assert out.index.tolist() == ["zone_20", "zone_10"]
    assert out["id"].tolist() == [20, 10]
    assert out["count_result"].tolist() == [0, 0]
    assert out["count_diff"].tolist() == [-5, -6]
    assert out["count_result"].dtype == result_dtype
    pd.testing.assert_series_equal(out.dtypes, nonempty.dtypes)


@pytest.mark.parametrize("taz_order", [[1, 2, 3], [3, 1, 2]])
def test_positive_rows_layout_gives_same_diff(monkeypatch, taz_order):
    """Dense weight tables and positive-only weight tables (the fork's
    ``STORE_POSITIVE_WEIGHT_ROWS_ONLY``, filled by ``_DenseLayout``) give the same
    zone summaries, with ``_diff == _result - _control`` row by row, also when the
    control table is not sorted."""
    settings = patch_settings(monkeypatch)
    problem = summarize_problem()
    controls = problem[3]
    controls["TAZ"] = controls["TAZ"].loc[taz_order]

    dense = run_summarize(monkeypatch, settings, problem, False)
    positive = run_summarize(monkeypatch, settings, problem, True)

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
