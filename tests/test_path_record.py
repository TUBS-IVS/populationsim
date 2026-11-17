# PopulationSim
# See full license in LICENSE.txt.
"""Fork option INTEGERIZER_PATH_RECORD: integerizer_paths.csv."""

from __future__ import annotations

import ast
import csv
import subprocess

import numpy as np
import pandas as pd
import pytest

from populationsim.core import config
from populationsim.integerizing import bs_options, path_record, wrappers
from populationsim.integerizing.simul_integerizer import SimulIntegerizer
from populationsim.integerizing.single_integerizer import Integerizer
from tests.bs_differential import REPO_ROOT, run_example

COLUMNS = [
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
]
OUTCOMES = {
    "simultaneous",
    "retry_simultaneous",
    "sequential_no_infeasible_subzones",
    "sequential_single_feasible_subzone",
    "sequential_retry_failed",
    "smart_round_all_subzones",
    "single_failed_smart_round",
}


@pytest.fixture
def settings(monkeypatch, tmp_path):
    """Inject a settings dict and redirect the record file into tmp_path."""
    values = {}
    monkeypatch.setattr(
        config, "setting", lambda key, default=None: values.get(key, default)
    )
    record = tmp_path / "integerizer_paths.csv"
    monkeypatch.setattr(config, "output_file_path", lambda name: str(record))
    bs_options.reset_for_tests()
    path_record.reset_for_tests()
    values["_record_path"] = record
    yield values
    bs_options.reset_for_tests()
    path_record.reset_for_tests()


def _rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _single(n=6, label="lbl"):
    rng = np.random.RandomState(3)
    float_weights = rng.randint(1, 4, size=n) + rng.uniform(0.1, 0.9, size=n)
    total = int(np.floor(float_weights).sum()) + 2
    incidence = pd.DataFrame(
        {"num_hh": np.ones(n, dtype=np.int64)}, index=pd.RangeIndex(n, name="hh_id")
    )
    return Integerizer(
        incidence_table=incidence,
        control_importance_weights=pd.Series([1000.0], index=["num_hh"]),
        float_weights=pd.Series(float_weights, index=incidence.index),
        relaxed_control_totals=pd.Series([float(total)], index=["num_hh"]),
        total_hh_control_value=total,
        total_hh_control_index=0,
        control_is_hh_based=pd.Series([True], index=["num_hh"]),
        trace_label=label,
    )


def _simul(zones=2, n=6, label="lbl"):
    rng = np.random.RandomState(4)
    float_weights = rng.randint(0, 3, size=(zones, n)) + rng.uniform(
        0.1, 0.9, size=(zones, n)
    )
    zone_ids = list(range(1, zones + 1))
    totals = np.round(float_weights.sum(axis=1)).astype(np.int64)
    incidence = pd.DataFrame(
        {"num_hh": np.ones(n, dtype=np.int64)}, index=pd.RangeIndex(n, name="hh_id")
    )
    sub_weights = pd.DataFrame(float_weights.T, index=incidence.index, columns=zone_ids)
    sub_controls = pd.DataFrame({"num_hh": totals}, index=zone_ids)
    spec = pd.DataFrame({"target": ["num_hh"], "importance": [1000.0]})
    return SimulIntegerizer(incidence, sub_weights, sub_controls, spec, "num_hh", label)


def test_record_off_writes_nothing(settings):
    assert _single().integerize() in ("OPTIMAL", "FEASIBLE")
    assert _simul().integerize() in ("OPTIMAL", "FEASIBLE")
    path_record.record_outcome("simultaneous", geography="TAZ")
    assert not settings["_record_path"].exists()


def test_call_rows_single_and_simul(settings):
    settings["INTEGERIZER_PATH_RECORD"] = True
    _single(label="backstopped_x").integerize()
    _simul(zones=2, label="retry_y").integerize()
    rows = _rows(settings["_record_path"])
    assert list(rows[0].keys()) == COLUMNS
    assert [(r["row_type"], r["kind"], r["trace_label"]) for r in rows] == [
        ("call", "single", "backstopped_x"),
        ("call", "simul", "retry_y"),
    ]
    for r in rows:
        assert r["status"] in ("OPTIMAL", "FEASIBLE")
        assert r["geography"] == "" and r["outcome"] == ""
        assert r["draw"] == "off" and r["random_seed"] == ""
        assert float(r["build_s"]) >= 0 and float(r["solve_s"]) >= 0
    assert rows[1]["zones"] == "2" and rows[1]["rows"] == "6"
    assert rows[0]["rows"] == "6"


def test_call_row_carries_draw_and_seed(settings):
    settings["INTEGERIZER_PATH_RECORD"] = True
    settings["INTEGERIZER_DRAW"] = "pareto"
    settings["INTEGERIZER_RANDOM_SEED"] = 77
    _single().integerize()
    (row,) = _rows(settings["_record_path"])
    assert (row["draw"], row["random_seed"]) == ("pareto", "77")


def test_all_residuals_zero_writes_optimal_zero_seconds(settings):
    settings["INTEGERIZER_PATH_RECORD"] = True
    n = 4
    incidence = pd.DataFrame(
        {"num_hh": np.ones(n, dtype=np.int64)}, index=pd.RangeIndex(n, name="hh_id")
    )
    integerizer = Integerizer(
        incidence_table=incidence,
        control_importance_weights=pd.Series([1000.0], index=["num_hh"]),
        float_weights=pd.Series([1.0, 2.0, 3.0, 1.0], index=incidence.index),
        relaxed_control_totals=pd.Series([7.0], index=["num_hh"]),
        total_hh_control_value=7,
        total_hh_control_index=0,
        control_is_hh_based=pd.Series([True], index=["num_hh"]),
        trace_label="zero",
    )
    assert integerizer.integerize() == "OPTIMAL"
    (row,) = _rows(settings["_record_path"])
    assert row["status"] == "OPTIMAL" and row["trace_label"] == "zero"
    assert float(row["build_s"]) == 0.0 and float(row["solve_s"]) == 0.0


def test_header_written_once_and_after_empty_file(settings):
    settings["INTEGERIZER_PATH_RECORD"] = True
    settings["_record_path"].write_text("")  # empty file: header still needed
    path_record.record_outcome("simultaneous", geography="TAZ")
    path_record.record_outcome("simultaneous", geography="TAZ")
    text = settings["_record_path"].read_text()
    assert text.count("row_type,kind") == 1
    assert "\r" not in text
    assert len(_rows(settings["_record_path"])) == 2


def test_record_with_multiprocess_raises(settings):
    settings["INTEGERIZER_PATH_RECORD"] = True
    settings["multiprocess"] = True
    with pytest.raises(ValueError, match="multiprocess"):
        path_record.record_outcome("simultaneous", geography="TAZ")
    with pytest.raises(ValueError, match="multiprocess"):
        _single()
    assert not settings["_record_path"].exists()


def test_multiprocess_without_record_is_fine(settings):
    settings["multiprocess"] = True
    assert _single().integerize() in ("OPTIMAL", "FEASIBLE")


def test_record_logs_one_info_line(settings, caplog):
    settings["INTEGERIZER_PATH_RECORD"] = True
    with caplog.at_level("INFO"):
        _single().integerize()
        _single().integerize()
    lines = [r.getMessage() for r in caplog.records if "path record" in r.getMessage()]
    assert len(lines) == 1
    assert str(settings["_record_path"]) in lines[0]
    assert "do_simul_integerizing" not in lines[0]


# ------------------------------------------------------- wrapper branches


def _wrapper_inputs(zone_ids):
    n = 3
    incidence = pd.DataFrame(
        {"num_hh": np.ones(n, dtype=np.int64)}, index=pd.RangeIndex(n, name="hh_id")
    )
    names = ["TAZ_%s" % z for z in zone_ids]
    sub_weights = pd.DataFrame(1.0, index=incidence.index, columns=names)
    sub_controls = pd.DataFrame({"num_hh": [3] * len(zone_ids)}, index=zone_ids)
    zones = pd.Series(names, index=zone_ids)
    spec = pd.DataFrame({"target": ["num_hh"], "importance": [1000.0]})
    return dict(
        trace_label="PUMA_9",
        incidence_df=incidence,
        sub_weights=sub_weights,
        sub_controls_df=sub_controls,
        control_spec=spec,
        total_hh_control_col="num_hh",
        sub_geography="TAZ",
        sub_control_zones=zones,
    )


def _df(tag):
    return pd.DataFrame({"x": [tag]})


@pytest.mark.parametrize(
    "case, zone_ids, try_results, seq, outcome, rounded",
    [
        ("simultaneous", [1, 2], [("OPTIMAL", "S")], None, "simultaneous", 0),
        (
            "retry",
            [1, 2, 3],
            [("INFEASIBLE", None), ("FEASIBLE", _df("R"))],
            ([1, 2], [3], _df("A"), _df("B")),
            "retry_simultaneous",
            1,
        ),
        (
            "no_infeasible",
            [1, 2],
            [("INFEASIBLE", None)],
            ([1, 2], [], _df("A"), None),
            "sequential_no_infeasible_subzones",
            0,
        ),
        (
            "single_feasible",
            [1, 2],
            [("INFEASIBLE", None)],
            ([1], [2], _df("A"), _df("B")),
            "sequential_single_feasible_subzone",
            1,
        ),
        (
            "retry_failed",
            [1, 2, 3],
            [("INFEASIBLE", None), ("INFEASIBLE", None)],
            ([1, 2], [3], _df("A"), _df("B")),
            "sequential_retry_failed",
            1,
        ),
        (
            "all_rounded",
            [1, 2],
            [("INFEASIBLE", None)],
            ([], [1, 2], None, _df("B")),
            "smart_round_all_subzones",
            2,
        ),
    ],
)
def test_each_wrapper_branch_writes_its_outcome(
    settings, monkeypatch, case, zone_ids, try_results, seq, outcome, rounded
):
    settings["INTEGERIZER_PATH_RECORD"] = True
    queue = list(try_results)
    labels = []

    def fake_try(trace_label, *args, **kwargs):
        labels.append(trace_label)
        status, result = queue.pop(0)
        return status, (_df("S") if isinstance(result, str) else result)

    monkeypatch.setattr(wrappers, "try_simul_integerizing", fake_try)
    monkeypatch.setattr(
        wrappers, "do_sequential_integerizing", lambda *a, **k: seq
    )
    wrappers.do_simul_integerizing(**_wrapper_inputs(zone_ids))
    assert not queue
    rows = _rows(settings["_record_path"])
    assert len(rows) == 1
    (row,) = rows
    assert row["row_type"] == "outcome" and row["outcome"] == outcome
    assert row["geography"] == "TAZ" and row["trace_label"] == "PUMA_9"
    assert row["rounded_zones"] == str(rounded)
    assert row["zones"] == str(len(zone_ids))
    assert row["kind"] == "" and row["status"] == "" and row["solve_s"] == ""
    if case in ("retry", "retry_failed"):
        assert labels == ["PUMA_9", "retry_PUMA_9"]


def test_single_failed_smart_round_outcome(settings, monkeypatch):
    settings["INTEGERIZER_PATH_RECORD"] = True
    settings["INTEGERIZE_WITH_BACKSTOPPED_CONTROLS"] = False

    class FailingIntegerizer:
        def __init__(self, **kwargs):
            self.weights = pd.DataFrame(
                {"integerized_weight": [1, 2, 3]}, index=kwargs["float_weights"].index
            )

        def integerize(self):
            return "INFEASIBLE"

    monkeypatch.setattr(wrappers, "Integerizer", FailingIntegerizer)
    n = 3
    incidence = pd.DataFrame(
        {"num_hh": np.ones(n, dtype=np.int64)}, index=pd.RangeIndex(n, name="hh_id")
    )
    spec = pd.DataFrame(
        {"target": ["num_hh"], "importance": [1000.0], "seed_table": ["households"]}
    )
    wrappers.do_integerizing(
        "TAZ_5",
        spec,
        pd.Series({"num_hh": 6}),
        incidence,
        pd.Series([1.5, 2.5, 2.0], index=incidence.index),
        "num_hh",
    )
    (row,) = _rows(settings["_record_path"])
    assert row["row_type"] == "outcome"
    assert row["outcome"] == "single_failed_smart_round"
    assert row["geography"] == "" and row["trace_label"] == "TAZ_5"


def test_single_success_writes_no_outcome(settings, monkeypatch):
    settings["INTEGERIZER_PATH_RECORD"] = True
    settings["INTEGERIZE_WITH_BACKSTOPPED_CONTROLS"] = False

    class OkIntegerizer:
        def __init__(self, **kwargs):
            self.weights = pd.DataFrame(
                {"integerized_weight": [1, 2, 3]}, index=kwargs["float_weights"].index
            )

        def integerize(self):
            return "FEASIBLE"

    monkeypatch.setattr(wrappers, "Integerizer", OkIntegerizer)
    n = 3
    incidence = pd.DataFrame(
        {"num_hh": np.ones(n, dtype=np.int64)}, index=pd.RangeIndex(n, name="hh_id")
    )
    spec = pd.DataFrame(
        {"target": ["num_hh"], "importance": [1000.0], "seed_table": ["households"]}
    )
    wrappers.do_integerizing(
        "TAZ_5",
        spec,
        pd.Series({"num_hh": 6}),
        incidence,
        pd.Series([1.5, 2.5, 2.0], index=incidence.index),
        "num_hh",
    )
    assert not settings["_record_path"].exists()


# ------------------------------------------------------------- log texts


def _logger_calls(source):
    calls = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "logger"
        ):
            arg = node.args[0]
            fmt = ast.unparse(arg) if hasattr(ast, "unparse") else ast.dump(arg)
            calls.append((node.lineno, node.func.attr, fmt))
    return [c[1:] for c in sorted(calls)]


def test_wrapper_log_texts_unchanged():
    old = subprocess.run(
        ["git", "show", "v0.10.0:populationsim/integerizing/wrappers.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    new = (REPO_ROOT / "populationsim/integerizing/wrappers.py").read_text(
        encoding="utf-8"
    )
    old_calls, new_calls = _logger_calls(old), _logger_calls(new)
    assert len(old_calls) == 21
    assert new_calls == old_calls


# --------------------------------------------------------- example runs


def test_record_off_example_writes_no_file(tmp_path):
    run_example(REPO_ROOT, tmp_path / "out")
    assert not (tmp_path / "out" / "integerizer_paths.csv").exists()


def test_record_on_example(tmp_path):
    out = tmp_path / "out"
    run_example(REPO_ROOT, out, {"INTEGERIZER_PATH_RECORD": True})
    path = out / "integerizer_paths.csv"
    rows = _rows(path)
    assert list(rows[0].keys()) == COLUMNS
    assert {r["row_type"] for r in rows} <= {"call", "outcome"}
    calls = [r for r in rows if r["row_type"] == "call"]
    outcomes = [r for r in rows if r["row_type"] == "outcome"]
    assert calls and outcomes
    assert {r["kind"] for r in calls} <= {"single", "simul"}
    assert "simul" in {r["kind"] for r in calls}
    assert {r["status"] for r in calls} <= {
        "OPTIMAL",
        "FEASIBLE",
        "INFEASIBLE",
        "NOT_SOLVED",
        "UNBOUNDED",
        "ABNORMAL",
    }
    assert {r["outcome"] for r in outcomes} <= OUTCOMES
    assert {r["geography"] for r in outcomes} <= {"TAZ", "TRACT", ""}
    # one outcome row per parent: no duplicate (geography, trace_label)
    keys = [(r["geography"], r["trace_label"]) for r in outcomes if r["geography"]]
    assert keys and len(keys) == len(set(keys))
    # every parent solved once (first simul call, not a retry) has its outcome
    parents = [
        r for r in calls if r["kind"] == "simul" and not r["trace_label"].startswith("retry_")
    ]
    assert len(keys) == len(parents)
    for geography in ("TRACT", "TAZ"):
        assert any(g == geography for g, _ in keys)


def test_record_does_not_change_results(tmp_path):
    base = run_example(REPO_ROOT, tmp_path / "out_default")
    rec = run_example(
        REPO_ROOT, tmp_path / "out_record", {"INTEGERIZER_PATH_RECORD": True}
    )
    assert (tmp_path / "out_record" / "integerizer_paths.csv").exists()

    def canon(df):
        return df.sort_values(list(df.columns)).reset_index(drop=True)

    assert len(base) > 0
    assert canon(base).equals(canon(rec))
