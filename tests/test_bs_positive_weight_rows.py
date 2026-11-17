# PopulationSim
# See full license in LICENSE.txt.
"""``STORE_POSITIVE_WEIGHT_ROWS_ONLY``: the sub-geography weight tables keep only
their positive rows, and the synthetic population does not change.

Unit tests of the filter and of the parent-weight rebuild, and differential runs of
the fork's example with the setting off and on.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from populationsim.core import config
from populationsim.core import positive_weight_rows as pwr
from populationsim.integerizing import wrappers
from tests.bs_differential import REPO_ROOT, run_example

SETTING = "STORE_POSITIVE_WEIGHT_ROWS_ONLY"


@pytest.fixture
def settings(monkeypatch):
    """Inject a settings dict by patching config.setting."""
    values = {}
    monkeypatch.setattr(
        config, "setting", lambda key, default=None: values.get(key, default)
    )
    return values


class _ListHandler(logging.Handler):
    """Collect records directly on the module logger (independent of caplog)."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def module_log():
    logger = logging.getLogger(pwr.__name__)
    handler = _ListHandler()
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler
    logger.removeHandler(handler)
    logger.setLevel(old_level)


def _zone_frame():
    """A zone frame shaped like the builders' (labels 0..n-1, int32 integer weight)."""
    df = pd.DataFrame(index=list(range(0, 6)))
    df["hh_id"] = np.array([11, 12, 13, 14, 15, 16], dtype=np.int64)
    df["TAZ"] = 7
    df["balanced_weight"] = [0.0, 0.25, 0.0, 3.0, 0.0, 1e-300]
    df["integer_weight"] = np.array([0, 0, 1, 3, 0, 0], dtype=np.int32)
    return df


# --- the filter -------------------------------------------------------------------


def test_filter_keeps_union_of_positive_rows_with_labels_and_dtypes():
    df = _zone_frame()
    kept = pwr.keep_positive_rows(df, 1)
    expected = df[(df["balanced_weight"] > 0) | (df["integer_weight"] > 0)]
    pd.testing.assert_frame_equal(kept, expected)
    assert list(kept.index) == [1, 2, 3, 5]
    assert kept.dtypes.equals(df.dtypes)


def test_filter_without_integer_column_uses_balanced_weight():
    df = _zone_frame().drop(columns="integer_weight")
    kept = pwr.keep_positive_rows(df, 1)
    assert list(kept.index) == [1, 3, 5]
    assert kept.dtypes.equals(df.dtypes)


def test_filter_raises_on_nan_before_dropping():
    df = _zone_frame()
    df.loc[0, "balanced_weight"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        pwr.keep_positive_rows(df, 1)


def test_concat_drops_empty_frames_and_keeps_dtypes_when_all_are_empty():
    df = _zone_frame()
    zero = df.copy()
    zero["balanced_weight"] = 0.0
    zero["integer_weight"] = np.zeros(len(df), dtype=np.int32)
    empty = pwr.keep_positive_rows(zero, 1)
    assert len(empty) == 0
    both = pwr.concat_kept([empty, pwr.keep_positive_rows(df, 1), empty])
    pd.testing.assert_frame_equal(both, pwr.keep_positive_rows(df, 1))
    only_empty = pwr.concat_kept([empty, empty])
    assert len(only_empty) == 0
    assert list(only_empty.columns) == list(df.columns)
    assert only_empty.dtypes.equals(df.dtypes)


def _sub_weights(all_zero_zone=True):
    """Balancer output: one row per hh (index hh_id), one column per sub zone."""
    index = pd.Index([21, 22, 23, 24, 25], name="hh_id", dtype=np.int64)
    float_weights = pd.DataFrame(
        {
            "TAZ_1": [0.0, 1.5, 0.0, 2.25, 0.0],
            "TAZ_2": [0.0, 0.0, 0.0, 0.0, 0.0] if all_zero_zone else [1.0] * 5,
            "TAZ_3": [0.5, 0.0, 0.0, 1.0, 0.0],
        },
        index=index,
    )
    integerized = pd.DataFrame(
        {
            "TAZ_1": [0, 2, 0, 2, 0],
            "TAZ_2": [0, 0, 0, 0, 0] if all_zero_zone else [1] * 5,
            "TAZ_3": [1, 0, 0, 1, 0],
        },
        index=index,
    )
    zones = pd.Series(["TAZ_1", "TAZ_2", "TAZ_3"], index=[1, 2, 3])
    return float_weights, integerized, zones


def _positive(df):
    mask = df["balanced_weight"] > 0
    if "integer_weight" in df:
        mask = mask | (df["integer_weight"] > 0)
    return df[mask]


def test_reshape_result_on_equals_positive_rows_of_off(settings):
    float_weights, integerized, zones = _sub_weights()
    off = wrappers.reshape_result(float_weights, integerized, "TAZ", zones)
    settings[SETTING] = True
    on = wrappers.reshape_result(float_weights, integerized, "TAZ", zones)
    pd.testing.assert_frame_equal(on, _positive(off))
    assert 2 not in set(on["TAZ"]), "the all-zero zone must be dropped"
    assert len(off) == 15 and len(on) == 4


def test_reshape_result_all_zero_parent_keeps_columns_and_dtypes(settings):
    float_weights, integerized, zones = _sub_weights()
    float_weights[:] = 0.0
    integerized[:] = 0
    off = wrappers.reshape_result(float_weights, integerized, "TAZ", zones)
    settings[SETTING] = True
    on = wrappers.reshape_result(float_weights, integerized, "TAZ", zones)
    assert len(on) == 0
    assert list(on.columns) == list(off.columns)
    assert on.dtypes.equals(off.dtypes)


def test_reshape_result_on_raises_on_nan(settings):
    float_weights, integerized, zones = _sub_weights()
    float_weights.iloc[0, 0] = np.nan
    settings[SETTING] = True
    with pytest.raises(ValueError, match="NaN"):
        wrappers.reshape_result(float_weights, integerized, "TAZ", zones)


def test_no_integerizing_on_equals_positive_rows_of_off(settings):
    float_weights, _, zones = _sub_weights()
    off = wrappers.do_no_integerizing(float_weights, zones, "TAZ")
    settings[SETTING] = True
    on = wrappers.do_no_integerizing(float_weights, zones, "TAZ")
    pd.testing.assert_frame_equal(on, _positive(off))
    assert len(on) == 4


# --- the parent rebuild -----------------------------------------------------------


def _dense_parent_table():
    """A parent weight table as the dense path stores it, for two parent zones.

    Each zone holds one row per seed household in the seed incidence order.
    """
    seed_index = pd.Index([31, 32, 33, 34, 35, 36], name="hh_id", dtype=np.int64)
    frames = []
    for zone, bal, integer in [
        (1, [0.0, 2.5, 0.0, 0.0, 1.25, 0.0], [0, 3, 0, 0, 1, 0]),
        (2, [0.0] * 6, [0] * 6),
    ]:
        df = pd.DataFrame(index=list(range(0, 6)))
        df["hh_id"] = seed_index.values
        df["TRACT"] = zone
        df["balanced_weight"] = bal
        df["integer_weight"] = np.array(integer, dtype=np.int32)
        frames.append(df)
    return seed_index, pd.concat(frames)


def _dense_read(table, parent_id, column):
    """What upstream sub_balancing passes to the balancer for one parent."""
    rows = table[table["TRACT"] == parent_id].set_index("hh_id")
    return rows[column]


@pytest.mark.parametrize("column", ["balanced_weight", "integer_weight"])
@pytest.mark.parametrize("parent_id", [1, 2])
def test_rebuild_equals_dense_read_when_stored_order_differs(column, parent_id):
    seed_index, dense = _dense_parent_table()
    stored = pwr.keep_positive_rows(dense, 1)
    # store the kept rows in another order than the seed incidence index
    stored = stored.iloc[::-1]
    assert len(stored) == 2
    assert list(stored["hh_id"]) == [35, 32]
    rebuilt = pwr.rebuild_parent_weights(
        stored[stored["TRACT"] == parent_id], "hh_id", column, seed_index
    )
    expected = _dense_read(dense, parent_id, column)
    pd.testing.assert_series_equal(rebuilt, expected)
    assert rebuilt.index.equals(seed_index)


def test_rebuild_refuses_duplicate_households():
    seed_index, dense = _dense_parent_table()
    stored = pwr.keep_positive_rows(dense, 1)
    stored = pd.concat([stored, stored])
    with pytest.raises(ValueError, match="unique"):
        pwr.rebuild_parent_weights(
            stored[stored["TRACT"] == 1], "hh_id", "balanced_weight", seed_index
        )


def test_rebuild_refuses_households_of_another_seed_zone():
    seed_index, dense = _dense_parent_table()
    stored = pwr.keep_positive_rows(dense, 1)
    stored.loc[stored["hh_id"] == 35, "hh_id"] = 99
    with pytest.raises(AssertionError, match="overlapping zones"):
        pwr.rebuild_parent_weights(
            stored[stored["TRACT"] == 1], "hh_id", "balanced_weight", seed_index
        )


# --- the rate line ----------------------------------------------------------------


def test_rate_line_once_per_geography_and_nothing_per_zone(module_log):
    pwr.reset_row_tally()
    for zone_id in (1, 2, 3):
        pwr.keep_positive_rows(_zone_frame(), zone_id)
    assert module_log.records == [], "the filter must not log per zone"
    assert pwr.log_row_tally("TAZ") == (12, 18)
    infos = [r for r in module_log.records if r.levelno == logging.INFO]
    assert len(infos) == 1
    message = infos[0].getMessage()
    assert "TAZ_weights" in message
    assert "kept 12 of 18 rows (66.7 %)" in message


def test_rate_counts_a_zone_once_when_its_frame_is_replaced():
    """do_simul_integerizing's retry replaces a zone's sequential frame."""
    pwr.reset_row_tally()
    pwr.keep_positive_rows(_zone_frame(), 1)
    zero = _zone_frame()
    zero["balanced_weight"] = 0.0
    zero["integer_weight"] = np.zeros(len(zero), dtype=np.int32)
    pwr.keep_positive_rows(zero, 2)
    pwr.keep_positive_rows(_zone_frame(), 2)
    assert pwr.log_row_tally("TAZ") == (8, 12)
    pwr.reset_row_tally()
    assert pwr.log_row_tally("TAZ") == (0, 0)


# --- differential runs of the example ---------------------------------------------

VARIANTS = {
    "default": {},
    "sequential": {"USE_SIMUL_INTEGERIZER": False},
    "no_integerizing": {"NO_INTEGERIZATION_EVER": True},
    "float_seed_weights": {"SUB_BALANCE_WITH_FLOAT_SEED_WEIGHTS": True},
    "no_grouping": {"GROUP_BY_INCIDENCE_SIGNATURE": False},
}
SUB_GEOGRAPHIES = ["TRACT", "TAZ"]


def _final_table(store, table):
    """The last version of ``table`` in a checkpoint store."""
    checkpoint = store["checkpoints"].iloc[-1][table]
    return store[f"/{table}/{checkpoint}"]


def _run_off_on(tmp_path, overrides):
    no_int = overrides.get("NO_INTEGERIZATION_EVER", False)
    off = run_example(
        REPO_ROOT, tmp_path / "off", {**overrides, SETTING: False}, allow_empty=no_int
    )
    on = run_example(
        REPO_ROOT, tmp_path / "on", {**overrides, SETTING: True}, allow_empty=no_int
    )
    return off, on


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_example_on_equals_off(tmp_path, variant):
    overrides = VARIANTS[variant]
    off_ids, on_ids = _run_off_on(tmp_path, overrides)
    pd.testing.assert_frame_equal(on_ids, off_ids, check_exact=True)
    no_int = overrides.get("NO_INTEGERIZATION_EVER", False)
    if not no_int:
        assert len(off_ids) > 0

    off_store = pd.HDFStore(tmp_path / "off" / "pipeline.h5", mode="r")
    on_store = pd.HDFStore(tmp_path / "on" / "pipeline.h5", mode="r")
    try:
        dropped = 0
        rows = {}
        for geography in SUB_GEOGRAPHIES:
            name = f"{geography}_weights"
            off_dense = _final_table(off_store, name)
            on_dense = _final_table(on_store, name)
            pd.testing.assert_frame_equal(
                on_dense, _positive(off_dense), check_exact=True
            )
            rows[geography] = (len(on_dense), len(off_dense))
            dropped += len(off_dense) - len(on_dense)
            if not no_int:
                pd.testing.assert_frame_equal(
                    _final_table(on_store, f"{name}_sparse"),
                    _final_table(off_store, f"{name}_sparse"),
                    check_exact=True,
                )
            trace = f"trace_{name}"
            pd.testing.assert_frame_equal(
                _final_table(on_store, trace),
                _positive(_final_table(off_store, trace)),
                check_exact=True,
            )
        assert dropped > 0, "the example must have zero rows to drop"

        # the seed-geography table is never filtered
        pd.testing.assert_frame_equal(
            _final_table(on_store, "PUMA_weights"),
            _final_table(off_store, "PUMA_weights"),
            check_exact=True,
        )

        # summarize still runs; its sums may differ only in the last bits
        summaries = [
            k.split("/")[1] for k in off_store.keys() if k.startswith("/summary_")
        ]
        assert summaries
        for table in summaries:
            pd.testing.assert_frame_equal(
                _final_table(on_store, table),
                _final_table(off_store, table),
                check_exact=False,
                rtol=1e-12,
                atol=1e-9,
            )
    finally:
        off_store.close()
        on_store.close()

    log = (tmp_path / "on" / "populationsim.log").read_text()
    for geography in SUB_GEOGRAPHIES:
        lines = [
            line
            for line in log.splitlines()
            if "[positive weight rows]" in line and f" {geography}_weights:" in line
        ]
        assert len(lines) == 1, lines
        kept, total = rows[geography]
        assert f"kept {kept} of {total} rows" in lines[0], (lines[0], rows)
    off_log = (tmp_path / "off" / "populationsim.log").read_text()
    assert "[positive weight rows]" not in off_log


def test_dense_parent_rows_follow_the_seed_incidence_order(tmp_path):
    """The premise of the rebuild: with the setting off, every parent zone's rows in
    the TRACT weight table (the TAZ step's parent) are the seed zone's households in
    the order of the incidence table index."""
    run_example(REPO_ROOT, tmp_path / "off")
    store = pd.HDFStore(tmp_path / "off" / "pipeline.h5", mode="r")
    try:
        incidence = _final_table(store, "incidence_table")
        tract = _final_table(store, "TRACT_weights")
    finally:
        store.close()
    checked = 0
    for tract_id, rows in tract.groupby("TRACT", sort=False):
        seed_ids = rows["PUMA"].unique()
        assert len(seed_ids) == 1
        seed_index = incidence.index[incidence["PUMA"] == seed_ids[0]]
        assert list(rows["hh_id"]) == list(seed_index), f"TRACT {tract_id}"
        checked += 1
    assert checked > 1
