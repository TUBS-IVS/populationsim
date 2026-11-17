# PopulationSim
# See full license in LICENSE.txt.

import pandas as pd

from populationsim.integerizing import path_record
from tests.bs_differential import REPO_ROOT, run_example

KNOWN_OUTCOMES = {
    "simultaneous",
    "smart_round_all_subzones",
    "sequential_no_infeasible_subzones",
    "sequential_single_feasible_subzone",
    "retry_simultaneous",
    "sequential_retry_failed",
    "single_failed_smart_round",
}

ALL_ON = {
    "KEEP_FRACTIONAL_SAMPLE_WEIGHTS": True,
    "INTEGERIZER_DRAW": "pareto",
    "INTEGERIZER_RANDOM_SEED": 7,
    "INTEGERIZER_PATH_RECORD": True,
}


def test_example_all_on_runs(tmp_path):
    """Every fork option on at once: the example finishes, the record is well formed.

    The example's seed-level integerizer is INFEASIBLE for some zones also on the
    unmodified tag, so ``single_failed_smart_round`` may legitimately appear.
    """
    default = run_example(REPO_ROOT, tmp_path / "out_default")
    all_on = run_example(REPO_ROOT, tmp_path / "out_all_on", ALL_ON)
    assert len(default) > 0
    assert len(all_on) == len(default), "household total differs from the default run"

    record = pd.read_csv(
        tmp_path / "out_all_on" / path_record.FILE_NAME, dtype=str, keep_default_na=False
    )
    assert list(record.columns) == list(path_record.COLUMNS)
    assert len(record) > 0
    assert set(record["row_type"]) <= {"call", "outcome"}
    calls = record[record["row_type"] == "call"]
    outcomes = record[record["row_type"] == "outcome"]
    assert len(calls) > 0
    assert set(outcomes["outcome"]) <= KNOWN_OUTCOMES
    assert (outcomes["outcome"] != "").all()
    assert (calls["draw"] == "pareto").all()
    assert (calls["random_seed"] == "7").all()
    assert not (tmp_path / "out_default" / path_record.FILE_NAME).exists()
