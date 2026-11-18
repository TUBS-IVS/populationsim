# PopulationSim
# See full license in LICENSE.txt.
"""Release checks of ``0.10.0+bs.2`` against the example model.

* ``round`` mode of this tree reproduces ``v0.10.0+bs.1`` row for row, in the
  synthetic population and in the solver-path record (except the two timing
  columns), also with the run-time options of #656 on.
* The example finishes with every option on, in controlled rounding with both
  methods, and the run-time options do not change the controlled result either.

Each example run is a subprocess; every run is made once per module.
"""

import pandas as pd
import pytest

from populationsim.integerizing import path_record
from tests.bs_differential import REPO_ROOT, export_tag, run_example

BS1_TAG = "v0.10.0+bs.1"

# the options that v0.10.0+bs.1 knows
BS1_OPTIONS = {
    "KEEP_FRACTIONAL_SAMPLE_WEIGHTS": True,
    "INTEGERIZER_DRAW": "gumbel",
    "INTEGERIZER_RANDOM_SEED": 7,
    "INTEGERIZER_PATH_RECORD": True,
}
RUN_TIME_OPTIONS = {"STORE_POSITIVE_WEIGHT_ROWS_ONLY": True, "FAST_PATHS": True}
RUN_TIME_OPTIONS_OFF = {"STORE_POSITIVE_WEIGHT_ROWS_ONLY": False, "FAST_PATHS": False}
CONTROLLED = {**BS1_OPTIONS, "INTEGERIZER_ROUNDING": "controlled"}

TIMING_COLUMNS = ["build_s", "solve_s"]
SUCCESS_STATUSES = {"OPTIMAL", "FEASIBLE"}
FALLBACK_OUTCOMES = {"smart_round_all_subzones"}
FALLBACK_OUTCOME_PREFIX = "sequential_"


def _read_record(out_dir):
    return pd.read_csv(
        out_dir / path_record.FILE_NAME, dtype=str, keep_default_na=False
    )


def _assert_same_record(expected, actual):
    """Same header, same rows (in file order) except the timing columns."""
    assert list(actual.columns) == list(expected.columns)
    assert len(actual) == len(expected)
    pd.testing.assert_frame_equal(
        actual.drop(columns=TIMING_COLUMNS),
        expected.drop(columns=TIMING_COLUMNS),
        check_dtype=False,
    )


@pytest.fixture(scope="module")
def bs1_run(tmp_path_factory):
    """The example on the code of the tag v0.10.0+bs.1."""
    base = tmp_path_factory.mktemp("bs1")
    tag_root = export_tag(BS1_TAG, base / "tag_code")
    out = base / "out"
    ids = run_example(tag_root, out, BS1_OPTIONS)
    return ids, _read_record(out)


@pytest.fixture(scope="module")
def default_run(tmp_path_factory):
    """The example on this tree with no fork option."""
    return run_example(REPO_ROOT, tmp_path_factory.mktemp("default") / "out")


@pytest.fixture(scope="module")
def controlled_run(tmp_path_factory):
    """Controlled rounding (search) with every option on."""
    out = tmp_path_factory.mktemp("controlled") / "out"
    ids = run_example(REPO_ROOT, out, {**CONTROLLED, **RUN_TIME_OPTIONS})
    return ids, _read_record(out)


def test_round_mode_equals_bs1(tmp_path, bs1_run):
    """Round mode with the options of bs.1 gives bs.1's population and path record."""
    bs1_ids, bs1_record = bs1_run
    out = tmp_path / "out"
    ids = run_example(REPO_ROOT, out, BS1_OPTIONS)
    assert len(bs1_ids) > 0
    pd.testing.assert_frame_equal(ids, bs1_ids)  # row for row, no sorting
    _assert_same_record(bs1_record, _read_record(out))


def test_run_options_do_not_change_round_mode(tmp_path, bs1_run):
    """The run-time options of #656 leave round mode equal to bs.1."""
    bs1_ids, bs1_record = bs1_run
    out = tmp_path / "out"
    ids = run_example(REPO_ROOT, out, {**BS1_OPTIONS, **RUN_TIME_OPTIONS})
    assert len(bs1_ids) > 0
    pd.testing.assert_frame_equal(ids, bs1_ids)
    _assert_same_record(bs1_record, _read_record(out))


def test_example_controlled_with_every_option_on(tmp_path, default_run, controlled_run):
    """Controlled rounding (search) with every option on finishes cleanly."""
    ids, record = controlled_run
    assert len(default_run) > 0
    assert len(ids) > 0
    assert list(record.columns) == list(path_record.CONTROLLED_COLUMNS)

    calls = record[record["row_type"] == "call"]
    simul = calls[calls["kind"] == "simul"]
    assert len(simul) > 0
    assert (simul["rounding"] == "controlled").all()
    assert (simul["method"] == "search").all()
    assert set(simul["status"]) <= SUCCESS_STATUSES, set(simul["status"])

    outcomes = record[record["row_type"] == "outcome"]
    assert len(outcomes) > 0
    for outcome in set(outcomes["outcome"]):
        assert outcome not in FALLBACK_OUTCOMES, outcome
        assert not outcome.startswith(FALLBACK_OUTCOME_PREFIX), outcome

    # a search that recorded swaps or sweeps ended in a local optimum
    searched = simul[(simul["swaps"] != "") | (simul["sweeps"] != "")]
    assert len(searched) > 0
    assert (searched["converged"] == "1").all(), searched["converged"].tolist()

    # the run-time options do not change the controlled result either
    off = run_example(
        REPO_ROOT, tmp_path / "out_off", {**CONTROLLED, **RUN_TIME_OPTIONS_OFF}
    )
    pd.testing.assert_frame_equal(off, ids)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "OPEN FINDING (#646): the controlled rounding (search) gives 1502 expanded "
        "households in the example, the default run 1500: tracts 41 and 61 get 151 "
        "instead of 150 (TAZ 402 and 602 get 15 instead of 14). The zone household "
        "counts follow the integer household control while the rows keep their parent "
        "weights, and the two do not add up in the example (hh_excess > 0 in the "
        "record). Whether that is intended is for the owner of controlled_rounding "
        "to decide; this test states the expectation of the release check."
    ),
)
def test_controlled_household_total_equals_default_run(default_run, controlled_run):
    ids, _ = controlled_run
    assert len(default_run) > 0
    assert len(ids) == len(default_run), "household total differs from the default run"


def test_example_controlled_lp_with_every_option_on(tmp_path, default_run):
    """The linear programme method finishes too, with success statuses."""
    out = tmp_path / "out"
    ids = run_example(
        REPO_ROOT,
        out,
        {
            **CONTROLLED,
            **RUN_TIME_OPTIONS,
            "INTEGERIZER_CONTROLLED_METHOD": "lp",
        },
    )
    assert len(ids) > 0
    record = _read_record(out)
    calls = record[record["row_type"] == "call"]
    simul = calls[calls["kind"] == "simul"]
    assert len(simul) > 0
    assert (simul["rounding"] == "controlled").all()
    assert (simul["method"] == "lp").all()
    assert set(simul["status"]) <= SUCCESS_STATUSES, set(simul["status"])
