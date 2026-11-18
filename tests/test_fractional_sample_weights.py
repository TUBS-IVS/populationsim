# PopulationSim
# See full license in LICENSE.txt.
"""Fractional grouped sample weights (KEEP_FRACTIONAL_SAMPLE_WEIGHTS)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from populationsim.core import config
from populationsim.integerizing import bs_options
from populationsim.steps.setup_data_structures import build_grouped_incidence_table

REPO_ROOT = Path(__file__).resolve().parent.parent

WEIGHTS = [0.4, 0.3, 0.2, 1.6, 2.0, 5.0]


@pytest.fixture
def settings(monkeypatch):
    values = {"household_id_col": "hh_id"}
    monkeypatch.setattr(
        config, "setting", lambda key, default=None: values.get(key, default)
    )
    bs_options.reset_for_tests()
    yield values
    bs_options.reset_for_tests()


def _incidence(weights):
    table = pd.DataFrame(
        {
            "c1": np.array([0, 0, 0, 1, 1, 1], dtype=np.int64),
            "c2": np.array([0, 0, 1, 1, 1, 1], dtype=np.int64),
            "PUMA": np.array([1] * 6, dtype=np.int64),
            "sample_weight": np.array(weights),
        },
        index=pd.Index(np.arange(10, 16), name="hh_id"),
    )
    return table


def _build(weights):
    spec = pd.DataFrame({"target": ["c1", "c2"]})
    return build_grouped_incidence_table(_incidence(weights), spec, "PUMA")


def test_flag_off_truncates_like_upstream(settings):
    grouped, _ = _build(WEIGHTS)
    assert grouped["sample_weight"].dtype == np.int64
    assert grouped["sample_weight"].tolist() == [0, 0, 8]


def test_flag_on_keeps_sum(settings):
    off, off_groups = _build(WEIGHTS)
    settings["KEEP_FRACTIONAL_SAMPLE_WEIGHTS"] = True
    on, on_groups = _build(WEIGHTS)
    assert on["sample_weight"].dtype == np.float64
    assert on["sample_weight"].tolist() == pytest.approx([0.7, 0.2, 8.6])
    for col in on.columns.drop("sample_weight"):
        assert on[col].dtype == np.int64, col
    assert on["group_id"].tolist() == off["group_id"].tolist()
    assert list(on.columns) == list(off.columns)
    pd.testing.assert_frame_equal(on_groups, off_groups)


def test_flag_on_integer_weights_identical(settings):
    weights = [1, 2, 3, 4, 5, 6]
    off, _ = _build(weights)
    settings["KEEP_FRACTIONAL_SAMPLE_WEIGHTS"] = True
    on, _ = _build(weights)
    assert on["sample_weight"].tolist() == off["sample_weight"].tolist()
    for col in off.columns.drop("sample_weight"):
        assert on[col].tolist() == off[col].tolist()


def test_flag_on_logs_rescued_groups(settings, caplog, fork_logging):
    settings["KEEP_FRACTIONAL_SAMPLE_WEIGHTS"] = True
    with caplog.at_level(logging.INFO):
        _build(WEIGHTS)
    assert (
        "fractional sample weights kept: 3 groups, 2 with 0 < weight < 1 "
        "(the int64 cast would set them to 0)"
    ) in caplog.text
    assert "do_simul_integerizing" not in caplog.text


def test_flag_off_logs_no_fractional_line(settings, caplog, fork_logging):
    with caplog.at_level(logging.INFO):
        _build(WEIGHTS)
    assert "fractional sample weights kept" not in caplog.text


# Runs the example (scaled seed weights) up to the seed-level integerizer and
# records the grouped table and what the integerizer receives.
_RUNNER = """
import json, sys
from pathlib import Path

example_dir, configs_tmp, tests_configs, out_dir, result, n_models = sys.argv[1:7]

from populationsim.core import inject, pipeline, tracing
import populationsim.steps.integerize_final_seed_weights as ifsw

calls = []
real = ifsw.do_integerizing


def spy(**kwargs):
    calls.append(len(kwargs["incidence_table"].index))
    return real(**kwargs)


ifsw.do_integerizing = spy

example = Path(example_dir)
inject.reinject_decorated_tables()
inject.add_injectable(
    "configs_dir", [Path(configs_tmp), Path(tests_configs), example / "configs"]
)
inject.add_injectable("output_dir", Path(out_dir))
inject.add_injectable("data_dir", example / "data")
inject.clear_cache()
tracing.config_logger()

error = None
try:
    models = [
            "input_pre_processor",
            "setup_data_structures",
            "initial_seed_balancing",
            "meta_control_factoring",
            "final_seed_balancing",
            "integerize_final_seed_weights",
    ]
    pipeline.run(models=models[: int(n_models)], resume_after=None)
except Exception as exc:  # reported to the test, which decides
    error = repr(exc)
inc = None
integer_weights = None
if error is None:
    inc = pipeline.get_table("incidence_table")
    if int(n_models) >= 6:
        wt = pipeline.get_table("PUMA_weights")
        integer_weights = wt["integer_weight"].reindex(inc.index).tolist()
    inc = inc["sample_weight"].tolist()
    pipeline.close_pipeline()
Path(result).write_text(
    json.dumps(
        {
            "calls": calls,
            "weights": inc,
            "integer_weights": integer_weights,
            "error": error,
        }
    )
)
"""


def _run_scaled_example(tmp_path, flag, n_models, factor=0.01):
    tmp_path = tmp_path / f"flag_{flag}"
    tmp_path.mkdir()
    example = tmp_path / "example_test"
    shutil.copytree(REPO_ROOT / "examples" / "example_test", example)
    seed = example / "data" / "seed_households.csv"
    df = pd.read_csv(seed)
    df["WGTP"] = df["WGTP"] * factor
    df.to_csv(seed, index=False)

    configs_tmp = tmp_path / "configs_override"
    configs_tmp.mkdir()
    (configs_tmp / "settings.yaml").write_text(
        yaml.safe_dump(
            {"inherit_settings": True, "KEEP_FRACTIONAL_SAMPLE_WEIGHTS": flag}
        )
    )
    out_dir = tmp_path / f"out_{flag}"
    out_dir.mkdir()
    result = tmp_path / f"result_{flag}.json"
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _RUNNER,
            str(example),
            str(configs_tmp),
            str(REPO_ROOT / "tests" / "configs"),
            str(out_dir),
            str(result),
            str(n_models),
        ],
        cwd=out_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(result.read_text())


def test_flag_on_example_runs_to_seed_integerization(tmp_path):
    # Positive control: without the flag the scaled input yields zero-weight
    # groups. With the flag OFF balancing raises ZeroDivisionError on this input
    # (all weights cast to 0), which is why the control stops after
    # setup_data_structures.
    off = _run_scaled_example(tmp_path, False, n_models=2)
    assert off["error"] is None, off["error"]
    assert sum(1 for w in off["weights"] if w == 0) > 0

    on = _run_scaled_example(tmp_path, True, n_models=6)
    assert on["error"] is None, on["error"]
    assert len(on["weights"]) > 0
    assert all(w > 0 for w in on["weights"])
    assert sum(on["calls"]) == len(on["weights"])
    # Rescued groups exist, and at least one is actually chosen (primary path).
    rescued = [i for i, w in enumerate(on["weights"]) if 0 < w < 1]
    assert len(rescued) > 0
    assert any(on["integer_weights"][i] >= 1 for i in rescued)
