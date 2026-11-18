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


@pytest.mark.parametrize("column", ["balanced_weight", "integer_weight"])
def test_filter_raises_on_negative_weight_before_dropping(column):
    df = _zone_frame()
    df.loc[2, column] = -1
    pwr.reset_row_tally()
    with pytest.raises(ValueError, match="negative"):
        pwr.keep_positive_rows(df, 1)
    assert pwr.log_row_tally("TAZ") == (0, 0), "nothing may be tallied"


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
def test_rebuild_equals_dense_read_when_stored_order_differs(column):
    seed_index, dense = _dense_parent_table()
    stored = pwr.keep_positive_rows(dense, 1)
    # store the kept rows in another order than the seed incidence index
    stored = stored.iloc[::-1]
    assert len(stored) == 2
    assert list(stored["hh_id"]) == [35, 32]
    rebuilt = pwr.rebuild_parent_weights(
        stored[stored["TRACT"] == 1], "hh_id", column, seed_index
    )
    expected = _dense_read(dense, 1, column)
    pd.testing.assert_series_equal(rebuilt, expected)
    assert rebuilt.index.equals(seed_index)


def test_rebuild_refuses_a_parent_without_stored_rows():
    """Parent 2 has only zero weights, so no row is stored: refused by name (the
    dense path's all-zero vector fails in the balancer; see the sub_balancing
    tests below)."""
    seed_index, dense = _dense_parent_table()
    stored = pwr.keep_positive_rows(dense, 1)
    with pytest.raises(RuntimeError, match="TAZ: parent TRACT 2 has no stored row"):
        pwr.rebuild_parent_weights(
            stored[stored["TRACT"] == 2],
            "hh_id",
            "balanced_weight",
            seed_index,
            geography="TAZ",
            parent_geography="TRACT",
            parent_id=2,
        )


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


# --- sub_balancing on a small problem ---------------------------------------------

GEOGRAPHIES = ["REGION", "PUMA", "TRACT", "TAZ"]


class _Table:
    def __init__(self, df):
        self.df = df

    def to_frame(self):
        return self.df.copy()


def _sub_balancing_problem(tract_weights):
    """Seed zone PUMA 1 with six households; TAZ 1 and 2 in TRACT 10, TAZ 3 in
    TRACT 11. ``tract_weights`` maps a TRACT id to its six balanced weights (the
    integer weights are their rounded values)."""
    hh = pd.Index([1, 2, 3, 4, 5, 6], name="hh_id", dtype=np.int64)
    incidence = pd.DataFrame(
        {
            "num_hh": 1,
            "hh_size_1": [1, 0, 1, 0, 0, 1],
            "PUMA": 1,
            "sample_weight": 10.0,
        },
        index=hh,
    )
    crosswalk = pd.DataFrame(
        {"TAZ": [1, 2, 3], "TRACT": [10, 10, 11], "PUMA": 1, "REGION": 1}
    )
    spec = pd.DataFrame(
        {
            "target": ["num_hh", "hh_size_1"],
            "geography": ["TAZ", "TAZ"],
            "seed_table": "households",
            "importance": [1000000, 1000],
            "control_field": ["HHS", "HH1"],
        }
    )
    controls = {
        "TAZ": pd.DataFrame(
            {"num_hh": [5, 3, 4], "hh_size_1": [2, 1, 1]},
            index=pd.Index([1, 2, 3], name="TAZ"),
        ),
        "TRACT": pd.DataFrame(
            {"num_hh": [8, 4], "hh_size_1": [3, 1]},
            index=pd.Index([10, 11], name="TRACT"),
        ),
    }
    frames = []
    for tract, weights in tract_weights.items():
        df = pd.DataFrame(index=list(range(0, 6)))
        df["hh_id"] = hh.values
        df["TRACT"] = tract
        df["balanced_weight"] = np.asarray(weights, dtype=float)
        df["integer_weight"] = np.round(weights).astype(int)
        df["REGION"] = 1
        df["PUMA"] = 1
        frames.append(df)
    return incidence, crosswalk, spec, controls, pd.concat(frames)


def _run_sub_balancing(monkeypatch, settings, problem, positive_only, stored=None):
    from populationsim.balancing import wrappers as balancing_wrappers
    from populationsim.core import inject
    from populationsim.steps import sub_balancing as step

    incidence, crosswalk, spec, controls, dense_tract = problem
    settings.update(
        {
            "geographies": GEOGRAPHIES,
            "seed_geography": "PUMA",
            "household_id_col": "hh_id",
            "total_hh_control": "num_hh",
            SETTING: positive_only,
        }
    )
    monkeypatch.setattr(balancing_wrappers, "setting", config.setting)
    tract = dense_tract
    if stored is not None:
        tract = stored
    elif positive_only:
        tract = pwr.concat_kept(
            [pwr.keep_positive_rows(g, t) for t, g in dense_tract.groupby("TRACT")]
        )
    added = {}
    monkeypatch.setattr(inject, "get_step_arg", lambda name, default=None: "TAZ")
    monkeypatch.setattr(
        inject, "add_table", lambda name, df, **kw: added.__setitem__(name, df)
    )
    monkeypatch.setattr(step, "get_control_table", lambda g: controls[g])
    monkeypatch.setattr(step, "get_weight_table", lambda g, sparse=False: tract.copy())
    step.sub_balancing(settings, _Table(crosswalk), _Table(spec), _Table(incidence))
    return added


INTEGERIZERS = pytest.mark.parametrize(
    "integerizer",
    [
        {},
        {"USE_SIMUL_INTEGERIZER": False},
        {"NO_INTEGERIZATION_EVER": True},
        {"SUB_BALANCE_WITH_FLOAT_SEED_WEIGHTS": False},
        {"USE_NUMBA": True},
    ],
    ids=["simul", "sequential", "no_integerizing", "integer_parent_weights", "numba"],
)


@INTEGERIZERS
def test_sub_balancing_on_equals_off(monkeypatch, settings, module_log, integerizer):
    problem = _sub_balancing_problem(
        {10: [2.0, 1.5, 0.0, 3.0, 1.0, 0.5], 11: [1.0, 0.0, 0.0, 2.0, 0.0, 1.0]}
    )
    settings.update(integerizer)
    off = _run_sub_balancing(monkeypatch, settings, problem, False)
    module_log.records.clear()
    on = _run_sub_balancing(monkeypatch, settings, problem, True)
    assert len(on["TAZ_weights"]) < len(off["TAZ_weights"]) == 18
    pd.testing.assert_frame_equal(
        on["TAZ_weights"], _positive(off["TAZ_weights"]), check_exact=True
    )
    if "TAZ_weights_sparse" in off:
        pd.testing.assert_frame_equal(
            on["TAZ_weights_sparse"], off["TAZ_weights_sparse"], check_exact=True
        )
    messages = [(r.levelno, r.getMessage()) for r in module_log.records]
    expected = "TAZ_weights: kept %d of %d rows" % (
        len(on["TAZ_weights"]),
        len(off["TAZ_weights"]),
    )
    assert len(messages) == 1 and expected in messages[0][1], messages
    assert messages[0][0] == logging.INFO


@INTEGERIZERS
@pytest.mark.parametrize(
    "tract_weights, missing",
    [
        ({10: [0.0] * 6, 11: [0.0] * 6}, 10),
        ({10: [2.0, 1.5, 0.0, 3.0, 1.0, 0.5], 11: [0.0] * 6}, 11),
    ],
    ids=["no_positive_row", "one_parent_without_rows"],
)
def test_sub_balancing_on_refuses_a_parent_without_stored_rows(
    monkeypatch, settings, integerizer, tract_weights, missing
):
    """A parent zone with controls (so sub_balancing balances it) whose weights are
    all zero has no stored row. With the setting off its all-zero parent vector
    fails inside the balancer (no positive row to balance: NaN or division by
    zero); with the setting on the step refuses it by name."""
    problem = _sub_balancing_problem(tract_weights)
    settings.update(integerizer)
    with pytest.raises((AssertionError, ZeroDivisionError)):
        _run_sub_balancing(monkeypatch, settings, problem, False)
    with pytest.raises(RuntimeError, match=r"TAZ.*TRACT %d\b" % missing):
        _run_sub_balancing(monkeypatch, settings, problem, True)


# --- summarize --------------------------------------------------------------------


def _summarize_problem():
    """PUMA 1 (households 1-6) holds TRACT 10 (TAZ 1, 2) and TRACT 11 (TAZ 3);
    PUMA 2 (households 7-9) holds TRACT 12 (TAZ 4), whose controls are zero, so
    none of its zones is balanced and its households have no dense row. TRACT 11
    and TAZ 2 have only zero weights."""
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


def _run_summarize(monkeypatch, settings, problem, positive_only):
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
            stored[g] = _positive(weights[g])
    out = {}
    monkeypatch.setattr(inject, "get_step_arg", lambda name, default=None: default)
    monkeypatch.setattr(step, "get_control_table", lambda g: control_tables[g])
    monkeypatch.setattr(
        step, "get_weight_table", lambda g, sparse=False: stored[g].copy()
    )
    monkeypatch.setattr(step, "out_table", lambda name, df: out.__setitem__(name, df))
    step.summarize(_Table(crosswalk), _Table(incidence), spec)
    return out


def test_summarize_on_equals_off_and_keeps_unbalanced_households_missing(
    monkeypatch, settings
):
    problem = _summarize_problem()
    off = _run_summarize(monkeypatch, settings, problem, False)
    on = _run_summarize(monkeypatch, settings, problem, True)
    assert sorted(on) == sorted(off)
    for name in off:
        pd.testing.assert_frame_equal(on[name], off[name], check_exact=True)
    # households 7-9 (PUMA 2, never balanced below the seed) are missing, not zero
    hh = off["hh_weights"]
    assert hh.loc[[7, 8, 9], "TAZ_integer_weight"].isna().all()
    assert (hh.loc[range(1, 7), "TAZ_integer_weight"] >= 0).all()
    assert 7 not in off["TAZ_aggregate"].index


def test_summarize_on_refuses_stored_rows_outside_the_dense_table(
    monkeypatch, settings
):
    problem = _summarize_problem()
    taz = problem[-1]["TAZ"]
    taz.loc[taz["TAZ"] == 3, "TRACT"] = 10  # TAZ 3 does not lie in TRACT 10
    with pytest.raises(RuntimeError, match="TAZ_weights"):
        _run_summarize(monkeypatch, settings, problem, True)


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


def test_rate_line_refuses_a_stored_row_count_other_than_the_tally():
    pwr.reset_row_tally()
    pwr.keep_positive_rows(_zone_frame(), 1)
    assert pwr.log_row_tally("TAZ", stored_rows=4, dense_rows=6) == (4, 6)
    with pytest.raises(RuntimeError, match="TAZ_weights"):
        pwr.log_row_tally("TAZ", stored_rows=5)
    with pytest.raises(RuntimeError, match="TAZ_weights"):
        pwr.log_row_tally("TAZ", dense_rows=7)


def test_rate_line_of_a_sub_process_says_it_counts_a_slice(module_log, monkeypatch):
    monkeypatch.setattr(
        pwr, "_is_slice_process", lambda: True
    )  # as in a multiprocess sub-process
    pwr.reset_row_tally()
    pwr.keep_positive_rows(_zone_frame(), 1)
    pwr.log_row_tally("TAZ")
    (record,) = [r for r in module_log.records if r.levelno == logging.INFO]
    assert "slice of this sub-process" in record.getMessage()


# --- every path of do_simul_integerizing, with a stubbed integerizer -----------


class _StubSimulIntegerizer:
    """Stands in for SimulIntegerizer: statuses come from a queue; the integer
    weights are the ceiling of the balanced weights."""

    statuses = []

    def __init__(
        self,
        incidence_df,
        sub_weights,
        sub_controls_df,
        control_spec,
        total_hh_control_col,
        trace_label,
    ):
        self.sub_weights = sub_weights

    def integerize(self):
        status = _StubSimulIntegerizer.statuses.pop(0)
        if status in wrappers.STATUS_SUCCESS:
            self.integerized_weights = np.ceil(self.sub_weights).astype(np.int64)
        return status


def _stub_do_integerizing(zone_status):
    """Stands in for the single-zone integerizer: status per zone id; the integer
    weights are the floor of the balanced weights, plus one household with zero
    balanced weight (so a zone's sequential frame keeps other rows than its
    simultaneous frame, and a tally that does not replace it shows)."""

    def do_integerizing(
        trace_label,
        control_spec,
        control_totals,
        incidence_table,
        float_weights,
        total_hh_control_col,
        record_failure=True,
    ):
        integer = np.floor(float_weights).astype(int)
        integer.iloc[2] = 1  # balanced weight 0 in every zone of _simul_inputs
        return integer, zone_status[control_totals.name]

    return do_integerizing


def _simul_inputs():
    float_weights, _, zones = _sub_weights(all_zero_zone=False)
    float_weights["TAZ_2"] = [0.0, 0.0, 0.0, 2.5, 0.75]
    incidence = pd.DataFrame(
        {"num_hh": 1, "hh_size_1": [1, 0, 1, 0, 1]}, index=float_weights.index
    )
    controls = pd.DataFrame(
        {"num_hh": [4, 3, 2], "hh_size_1": [1, 1, 1]}, index=zones.index
    )
    spec = pd.DataFrame(
        {
            "target": ["num_hh", "hh_size_1"],
            "importance": [1000, 100],
            "seed_table": "households",
        }
    )
    return float_weights, incidence, controls, spec, zones


# outcome -> (statuses of the simultaneous attempts, sequential status per zone)
PATHS = {
    "simultaneous": (["OPTIMAL"], {}),
    "retry_simultaneous": (
        ["INFEASIBLE", "OPTIMAL"],
        {1: "OPTIMAL", 2: "OPTIMAL", 3: "INFEASIBLE"},
    ),
    "sequential_single_feasible_subzone": (
        ["INFEASIBLE"],
        {1: "OPTIMAL", 2: "INFEASIBLE", 3: "INFEASIBLE"},
    ),
    "smart_round_all_subzones": (
        ["INFEASIBLE"],
        {1: "INFEASIBLE", 2: "INFEASIBLE", 3: "INFEASIBLE"},
    ),
    "sequential_no_infeasible_subzones": (
        ["INFEASIBLE"],
        {1: "OPTIMAL", 2: "OPTIMAL", 3: "OPTIMAL"},
    ),
    "sequential_retry_failed": (
        ["INFEASIBLE", "INFEASIBLE"],
        {1: "OPTIMAL", 2: "OPTIMAL", 3: "INFEASIBLE"},
    ),
}


def _run_simul_path(monkeypatch, path, parent_id=10):
    simul_statuses, zone_status = PATHS[path]
    _StubSimulIntegerizer.statuses = list(simul_statuses)
    monkeypatch.setattr(wrappers, "SimulIntegerizer", _StubSimulIntegerizer)
    monkeypatch.setattr(wrappers, "do_integerizing", _stub_do_integerizing(zone_status))
    outcomes = []
    monkeypatch.setattr(
        wrappers,
        "_record_parent_outcome",
        lambda outcome, *args: outcomes.append(outcome),
    )
    float_weights, incidence, controls, spec, zones = _simul_inputs()
    with pwr.tally_scope("TAZ", parent_id):
        result = wrappers.do_simul_integerizing(
            trace_label="TRACT_%s" % parent_id,
            incidence_df=incidence,
            sub_weights=float_weights,
            sub_controls_df=controls,
            control_spec=spec,
            total_hh_control_col="num_hh",
            sub_geography="TAZ",
            sub_control_zones=zones,
        )
    assert outcomes == [path], "the stub must drive do_simul_integerizing to " + path
    assert _StubSimulIntegerizer.statuses == [], "every queued status must be used"
    return result


@pytest.mark.parametrize("path", list(PATHS))
def test_every_simul_integerizing_path_stores_the_positive_rows_of_off(
    settings, monkeypatch, path
):
    pwr.reset_row_tally()
    off = _run_simul_path(monkeypatch, path)
    assert pwr.log_row_tally("TAZ") == (0, 0), "the setting off must not tally"
    assert len(off) == 15
    settings[SETTING] = True
    on = _run_simul_path(monkeypatch, path)
    pd.testing.assert_frame_equal(on, _positive(off), check_exact=True)
    assert len(on) < len(off)
    kept_total = pwr.log_row_tally("TAZ", stored_rows=len(on), dense_rows=len(off))
    assert kept_total == (len(on), len(off))


def test_rate_counts_a_zone_once_when_its_frame_is_replaced(settings, monkeypatch):
    """The retry of do_simul_integerizing replaces the sequential frames of the
    feasible zones by their simultaneous frames: each zone counts once, with the
    frame that is stored."""
    settings[SETTING] = True
    pwr.reset_row_tally()
    on = _run_simul_path(monkeypatch, "retry_simultaneous")
    # zones 1 and 2 kept a zero-balanced household in their sequential frames;
    # only the rounded zone 3 keeps it in the stored table
    assert len(on[on["balanced_weight"] == 0]) == 1
    assert pwr.log_row_tally("TAZ") == (len(on), 15)


def test_rate_counts_the_same_zone_id_under_two_parents_twice(settings, monkeypatch):
    settings[SETTING] = True
    pwr.reset_row_tally()
    first = _run_simul_path(monkeypatch, "simultaneous", parent_id=10)
    second = _run_simul_path(monkeypatch, "simultaneous", parent_id=11)
    assert pwr.log_row_tally("TAZ") == (len(first) + len(second), 30)


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

        # summarize gives the same tables: the zero terms it no longer adds do not
        # change a sum, and what was missing (NaN) with the setting off stays so
        summaries = [
            k.split("/")[1] for k in off_store.keys() if k.startswith("/summary_")
        ]
        assert summaries
        for table in summaries:
            pd.testing.assert_frame_equal(
                _final_table(on_store, table),
                _final_table(off_store, table),
                check_exact=True,
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


# --- the multiprocess example -----------------------------------------------------

_MP_RUNNER = """
import sys
from pathlib import Path

import populationsim  # noqa: F401  (registers the steps)
from populationsim.core import inject, mp_tasks, pipeline, tracing

if __name__ == "__main__":
    configs_tmp, mp_configs, tests_configs, example_dir, out_dir = sys.argv[1:6]
    example = Path(example_dir)
    inject.add_injectable(
        "configs_dir",
        [Path(configs_tmp), Path(mp_configs), Path(tests_configs), example / "configs"],
    )
    inject.add_injectable("output_dir", Path(out_dir))
    inject.add_injectable("data_dir", example / "data")
    tracing.config_logger()
    injectables = {
        k: inject.get_injectable(k) for k in ["data_dir", "configs_dir", "output_dir"]
    }
    mp_tasks.run_multiprocess(injectables)
    pipeline.open_pipeline("_")
    ids = pipeline.get_table("expanded_household_ids")
    ids.to_csv(Path(out_dir) / "bs_expanded_household_ids.csv", index=False)
    pipeline.close_pipeline()
"""


def _run_mp_example(tmp_path, name, positive_only):
    import os
    import subprocess
    import sys

    import yaml

    out_dir = (tmp_path / name).resolve()
    configs_tmp = out_dir / "_configs_override"
    configs_tmp.mkdir(parents=True)
    (configs_tmp / "settings.yaml").write_text(
        yaml.safe_dump({"inherit_settings": True, SETTING: positive_only})
    )
    runner = out_dir / "run_mp.py"
    runner.write_text(_MP_RUNNER)
    example = REPO_ROOT / "examples" / "example_test"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [
            sys.executable,
            str(runner),
            str(configs_tmp),
            str(example / "configs_mp"),
            str(REPO_ROOT / "tests" / "configs"),
            str(example),
            str(out_dir),
        ],
        cwd=out_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (proc.stdout[-1500:], proc.stderr[-3000:])
    ids = pd.read_csv(out_dir / "bs_expanded_household_ids.csv")
    log = "\n".join(p.read_text() for p in sorted(out_dir.glob("*.log")))
    return ids, log


def test_multiprocess_example_on_equals_off_and_reports_the_coalesced_table(
    tmp_path,
):
    """The multiprocess example (two sub-processes balance the TAZ level): the
    population is the same, each sub-process reports its slice, and one line
    reports the coalesced TAZ_weights table against its dense row count."""
    off_ids, off_log = _run_mp_example(tmp_path, "off", False)
    on_ids, on_log = _run_mp_example(tmp_path, "on", True)
    assert len(off_ids) > 0
    pd.testing.assert_frame_equal(on_ids, off_ids, check_exact=True)
    assert "[positive weight rows]" not in off_log

    off_store = pd.HDFStore(tmp_path / "off" / "pipeline.h5", mode="r")
    on_store = pd.HDFStore(tmp_path / "on" / "pipeline.h5", mode="r")
    try:
        for name in ("TRACT_weights", "TAZ_weights"):
            off_dense = _final_table(off_store, name)
            on_dense = _final_table(on_store, name)
            pd.testing.assert_frame_equal(
                on_dense, _positive(off_dense), check_exact=True
            )
            pd.testing.assert_frame_equal(
                _final_table(on_store, name + "_sparse"),
                _final_table(off_store, name + "_sparse"),
                check_exact=True,
            )
        kept, total = len(on_dense), len(off_dense)
    finally:
        off_store.close()
        on_store.close()

    lines = [line for line in on_log.splitlines() if "[positive weight rows]" in line]
    coalesced = [line for line in lines if "TAZ_weights (coalesced" in line]
    assert len(coalesced) == 1, lines
    assert f"kept {kept} of {total} rows" in coalesced[0], (coalesced, kept, total)
    slices = [
        line for line in lines if "TAZ_weights (slice of this sub-process" in line
    ]
    assert len(slices) == 2, lines
    # no line may present a slice's count as the whole TAZ_weights table
    assert not [line for line in lines if " TAZ_weights: " in line], lines
