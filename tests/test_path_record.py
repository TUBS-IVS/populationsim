# PopulationSim
# See full license in LICENSE.txt.
"""Fork option INTEGERIZER_PATH_RECORD: integerizer_paths.csv."""

from __future__ import annotations

import ast
import csv
import logging
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
    monkeypatch.setattr(
        config,
        "build_output_file_path",
        lambda name, use_prefix=None: str(record.with_name(name)),
    )
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


def test_record_logs_one_info_line(settings, caplog, fork_logging):
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
    monkeypatch.setattr(wrappers, "do_sequential_integerizing", lambda *a, **k: seq)
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
        r
        for r in calls
        if r["kind"] == "simul" and not r["trace_label"].startswith("retry_")
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


def test_all_rounded_branch_writes_no_single_failed_rows(settings, monkeypatch):
    """Five failing zones: one parent outcome row, no per-zone failure rows."""
    settings["INTEGERIZER_PATH_RECORD"] = True
    settings["INTEGERIZE_WITH_BACKSTOPPED_CONTROLS"] = False

    class FailingIntegerizer:
        def __init__(self, **kwargs):
            self.weights = pd.DataFrame(
                {"integerized_weight": [1, 1, 1]}, index=kwargs["float_weights"].index
            )

        def integerize(self):
            return "INFEASIBLE"

    monkeypatch.setattr(wrappers, "Integerizer", FailingIntegerizer)
    monkeypatch.setattr(
        wrappers, "try_simul_integerizing", lambda *a, **k: ("INFEASIBLE", None)
    )
    inputs = _wrapper_inputs([1, 2, 3, 4, 5])
    inputs["control_spec"] = inputs["control_spec"].assign(seed_table="households")
    wrappers.do_simul_integerizing(**inputs)
    rows = _rows(settings["_record_path"])
    assert [r["outcome"] for r in rows] == ["smart_round_all_subzones"]
    assert rows[0]["rounded_zones"] == "5"


def test_record_ignores_output_file_prefix(settings, monkeypatch, tmp_path):
    """The record name carries no output file prefix."""
    from populationsim.core import inject

    monkeypatch.undo()  # use the real config helpers, not the fixture's patches
    out = tmp_path / "pref_out"
    out.mkdir()
    inject.add_injectable("output_dir", out)
    inject.add_injectable("output_file_prefix", "PREFIX")
    try:
        monkeypatch.setattr(
            config,
            "setting",
            lambda key, default=None: (
                True if key == "INTEGERIZER_PATH_RECORD" else default
            ),
        )
        bs_options.reset_for_tests()
        path_record.record_outcome("simultaneous", geography="TAZ")
    finally:
        inject.add_injectable("output_file_prefix", None)
        bs_options.reset_for_tests()
    assert (out / "integerizer_paths.csv").exists()
    assert not list(out.glob("PREFIX*"))


# --------------------------------------- INTEGERIZER_ROUNDING: controlled

# the counts a controlled simultaneous call row carries (stats of
# controlled_lp.solve_controlled)
CONTROLLED_COUNTS = (
    "attempts",
    "x_count",
    "fractional_x",
    "repaired_x",
    "fractional_after_repair",
    "repair_failed",
    "hh_unreachable",
    "hh_outside_lp",
    "hh_excess",
)


def test_round_mode_path_record_has_bs1_columns(settings):
    settings["INTEGERIZER_PATH_RECORD"] = True
    settings["INTEGERIZER_ROUNDING"] = "round"
    _single(label="s").integerize()
    _simul(zones=2, label="m").integerize()
    path_record.record_outcome("simultaneous", geography="TAZ", trace_label="m")
    header = settings["_record_path"].read_text().splitlines()[0]
    assert tuple(header.split(",")) == path_record.COLUMNS
    assert list(path_record.COLUMNS) == COLUMNS


def test_controlled_columns_are_the_documented_tuple():
    assert path_record.CONTROLLED_COLUMNS == tuple(COLUMNS) + (
        "rounding",
        "method",
        "solver",
        "attempts",
        "x_count",
        "swaps",
        "sweeps",
        "converged",
        "distance_start",
        "distance_final",
        "flow_wanted",
        "flow_achieved",
        "zone_shortfall",
        "row_shortfall",
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


# the columns only the LP method fills, and those only the search fills
LP_ONLY_COLUMNS = (
    "solver",
    "attempts",
    "fractional_x",
    "repaired_x",
    "fractional_after_repair",
    "repair_failed",
    "slack_cell",
    "slack_parent_sub",
    "slack_parent",
    "slack_conserve",
    "objective_lp",
    "objective_repaired",
)
SEARCH_ONLY_COLUMNS = (
    "swaps",
    "sweeps",
    "converged",
    "distance_start",
    "distance_final",
    "flow_wanted",
    "flow_achieved",
    "zone_shortfall",
    "row_shortfall",
)


@pytest.mark.parametrize("method", ["search", "lp", None])
def test_controlled_path_record_columns_and_counts(settings, method):
    settings["INTEGERIZER_PATH_RECORD"] = True
    settings["INTEGERIZER_ROUNDING"] = "controlled"
    if method is not None:
        settings["INTEGERIZER_CONTROLLED_METHOD"] = method
    _single(label="s").integerize()
    _simul(zones=2, label="m").integerize()
    path_record.record_outcome("simultaneous", geography="TAZ", trace_label="m")
    header = settings["_record_path"].read_text().splitlines()[0]
    assert tuple(header.split(",")) == path_record.CONTROLLED_COLUMNS
    assert path_record.CONTROLLED_COLUMNS[: len(COLUMNS)] == tuple(COLUMNS)
    single, simul, outcome = _rows(settings["_record_path"])
    assert (single["kind"], single["rounding"]) == ("single", "round")
    assert single["solver"] == "" and single["x_count"] == ""
    assert single["method"] == "" and single["swaps"] == ""
    assert (simul["kind"], simul["rounding"]) == ("simul", "controlled")
    assert float(simul["build_s"]) >= 0 and float(simul["solve_s"]) >= 0
    for column in ("x_count", "hh_unreachable", "hh_outside_lp", "hh_excess"):
        assert int(simul[column]) >= 0, column
    if method == "lp":
        assert simul["method"] == "lp"
        assert simul["status"] in ("OPTIMAL", "FEASIBLE")
        assert simul["solver"] == bs_options.controlled_solver()
        assert simul["attempts"] == "1"
        for column in CONTROLLED_COUNTS:
            assert int(simul[column]) >= 0, column
        for column in (
            "slack_cell",
            "slack_parent_sub",
            "slack_parent",
            "slack_conserve",
        ):
            assert float(simul[column]) >= 0.0, column
        assert float(simul["objective_lp"]) >= float(simul["objective_repaired"]) - 1e-6
        assert all(simul[column] == "" for column in SEARCH_ONLY_COLUMNS)
    else:
        # the default method
        assert simul["method"] == "search"
        assert simul["status"] == "FEASIBLE"
        for column in (
            "swaps",
            "sweeps",
            "converged",
            "flow_wanted",
            "flow_achieved",
            "zone_shortfall",
            "row_shortfall",
        ):
            assert int(simul[column]) >= 0, column
        assert simul["converged"] == "1"
        assert float(simul["distance_final"]) <= float(simul["distance_start"])
        assert all(simul[column] == "" for column in LP_ONLY_COLUMNS)
    assert (outcome["row_type"], outcome["rounding"]) == ("outcome", "controlled")
    assert outcome["solver"] == "" and outcome["attempts"] == ""
    assert outcome["method"] == "" and outcome["swaps"] == ""


def test_search_failure_is_recorded_and_handed_to_the_wrapper(
    settings, monkeypatch, caplog, fork_logging
):
    from populationsim.integerizing import controlled_search

    settings["INTEGERIZER_PATH_RECORD"] = True
    settings["INTEGERIZER_ROUNDING"] = "controlled"

    def failing(*args, **kwargs):
        return "BAD_COST_RANGE", None

    monkeypatch.setattr(controlled_search, "_min_cost_flow", failing)
    with caplog.at_level(logging.WARNING):
        assert _simul(zones=2, label="flowfail").integerize() == "ABNORMAL"
    (row,) = _rows(settings["_record_path"])
    assert (row["kind"], row["status"], row["method"]) == (
        "simul",
        "ABNORMAL",
        "search",
    )
    assert row["swaps"] == "0" and row["distance_final"] == ""
    assert any(
        r.levelno == logging.WARNING
        and "[simul controlled] flowfail" in r.getMessage()
        and "status ABNORMAL" in r.getMessage()
        for r in caplog.records
    )


def _stub_stats(**changes):
    stats = {
        "x_count": 0,
        "fractional_x": 0,
        "slack_cell": 0.0,
        "slack_parent_sub": 0.0,
        "slack_parent": 0.0,
        "slack_conserve": 0.0,
        "solver": "highs-ds",
        "attempts": 1,
        "repaired_x": 0,
        "fractional_after_repair": 0,
        "repair_failed": 0,
        "repair_seconds": 0.0,
        "conserve_deviation_lp": 0.0,
        "conserve_deviation_repaired": 0.0,
        "objective_lp": float("nan"),
        "objective_repaired": float("nan"),
        "objective_seconds": 0.0,
        "hh_unreachable": 0,
        "hh_outside_lp": 0,
        "hh_excess": 0,
    }
    stats.update(changes)
    return stats


def test_path_record_counts_fractional_lp_values(
    settings, monkeypatch, caplog, fork_logging
):
    """A fractional LP solution: the count is recorded and warned about, and
    smart_round still gives every zone its household control."""
    from populationsim.integerizing import controlled_lp

    settings["INTEGERIZER_PATH_RECORD"] = True
    settings["INTEGERIZER_ROUNDING"] = "controlled"
    settings["INTEGERIZER_CONTROLLED_METHOD"] = "lp"

    def fractional(problem, sub_incidence, parent_incidence, resid, *args, **kwargs):
        out = np.zeros_like(resid)
        out[0, 0] = 0.5
        out[1, 0] = 0.5
        stats = _stub_stats(
            x_count=int(problem.x_allowed.sum()),
            fractional_x=2,
            fractional_after_repair=2,
            objective_lp=1.0,
            objective_repaired=1.0,
        )
        return out, "OPTIMAL", stats

    monkeypatch.setattr(controlled_lp, "solve_controlled", fractional)
    integerizer = _simul(zones=2, label="frac")
    with caplog.at_level(logging.INFO):
        assert integerizer.integerize() == "OPTIMAL"
    (row,) = _rows(settings["_record_path"])
    assert row["rounding"] == "controlled"
    assert (row["fractional_x"], row["fractional_after_repair"]) == ("2", "2")
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "[simul controlled]" in r.getMessage()
    ]
    assert len(warnings) == 1 and "frac" in warnings[0]
    infos = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO and "[simul controlled]" in r.getMessage()
    ]
    assert len(infos) == 1
    controls = integerizer.sub_controls_df["num_hh"].to_numpy()
    np.testing.assert_array_equal(
        integerizer.integerized_weights.sum(axis=0).to_numpy(), controls
    )


def test_controlled_lp_failure_is_recorded_and_handed_to_the_wrapper(
    settings, monkeypatch, caplog, fork_logging
):
    from populationsim.integerizing import controlled_lp

    settings["INTEGERIZER_PATH_RECORD"] = True
    settings["INTEGERIZER_ROUNDING"] = "controlled"
    settings["INTEGERIZER_CONTROLLED_METHOD"] = "lp"
    settings["INTEGERIZE_WITH_BACKSTOPPED_CONTROLS"] = False

    def not_solved(problem, sub_incidence, parent_incidence, resid, *args, **kwargs):
        return resid, "NOT_SOLVED", _stub_stats(x_count=int(problem.x_allowed.sum()))

    monkeypatch.setattr(controlled_lp, "solve_controlled", not_solved)
    with caplog.at_level(logging.WARNING):
        assert _simul(zones=2, label="fail").integerize() == "NOT_SOLVED"
    (row,) = _rows(settings["_record_path"])
    assert (row["kind"], row["status"], row["rounding"]) == (
        "simul",
        "NOT_SOLVED",
        "controlled",
    )
    assert row["objective_lp"] == "" and row["objective_repaired"] == ""
    assert any(
        r.levelno == logging.WARNING and "[simul controlled] fail" in r.getMessage()
        for r in caplog.records
    )

    settings["_record_path"].unlink()
    inputs = _wrapper_inputs([1, 2])
    inputs["control_spec"] = inputs["control_spec"].assign(seed_table="households")
    wrappers.do_simul_integerizing(**inputs)
    rows = _rows(settings["_record_path"])
    simul_calls = [r for r in rows if r["row_type"] == "call" and r["kind"] == "simul"]
    assert simul_calls and all(
        (r["status"], r["rounding"]) == ("NOT_SOLVED", "controlled")
        for r in simul_calls
    )
    single_calls = [r for r in rows if r["kind"] == "single"]
    assert single_calls and all(r["rounding"] == "round" for r in single_calls)
    outcomes = [r for r in rows if r["row_type"] == "outcome"]
    assert len(outcomes) == 1
    assert outcomes[0]["outcome"] in OUTCOMES - {"simultaneous", "retry_simultaneous"}
    assert outcomes[0]["rounding"] == "controlled"
