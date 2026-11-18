# PopulationSim
# See full license in LICENSE.txt.
"""``FAST_PATHS``: the fast seed integerizer LP build and the fast household
expansion give results identical bit for bit to the upstream code paths.

A. ``lp_single_fast.np_integerizer_ortools_fast`` against the upstream
   ``lp_ortools.np_integerizer_ortools``: the model handed to the solver (LP and
   MPS exports and the deterministic serialisation of the model proto, which also
   shows explicitly stored zero coefficients) and the solved residual weights and
   status are equal, on the single-integerizer problems of the fork's example and
   on seeded random problems. The upstream model is exported from upstream's own
   function, unchanged, by wrapping ``Solver.Solve`` while it runs.
B. ``expand_households`` with the setting on against off: the expanded household
   ids table and the state of the step's random generator after the step are
   equal, on a constructed group table and on the fork's example.
"""

from __future__ import annotations

import copy
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from populationsim.integerizing import lp_cvx, lp_ortools, lp_single_fast
from populationsim.integerizing.single_integerizer import Integerizer
from populationsim.steps import expand_households as eh
from tests.bs_differential import REPO_ROOT, run_example
from tests.settings_isolation import patch_settings

SETTING = "FAST_PATHS"


@pytest.fixture
def settings(monkeypatch):
    """Inject a settings dict by patching config.setting and every module that
    imported it by name (see tests/settings_isolation.py); restored at teardown."""
    return patch_settings(monkeypatch)


# --- A. single integerizer LP build -------------------------------------------


def _exports(solver):
    """Every form of the model the solver interface offers, as comparable values."""
    from ortools.linear_solver import linear_solver_pb2

    proto = linear_solver_pb2.MPModelProto()
    solver.ExportModelToProto(proto)
    return {
        "lp": solver.ExportModelAsLpFormat(False),
        "mps": solver.ExportModelAsMpsFormat(True, False),
        "proto": proto.SerializeToString(deterministic=True),
        "terms": [len(c.var_index) for c in proto.constraint],
    }


def _run_with_exports(func, kwargs, monkeypatch):
    """Call an integerizer LP function and export its model when it is solved."""
    from ortools.linear_solver import pywraplp

    captured = []
    original_solve = pywraplp.Solver.Solve

    def solve(self, *args):
        captured.append(_exports(self))
        return original_solve(self, *args)

    with monkeypatch.context() as patch:
        patch.setattr(pywraplp.Solver, "Solve", solve)
        weights, status = func(**copy.deepcopy(kwargs))
    assert len(captured) == 1
    return weights, status, captured[0]


def _assert_same_model_and_result(kwargs, monkeypatch):
    up_w, up_status, up_model = _run_with_exports(
        lp_ortools.np_integerizer_ortools, kwargs, monkeypatch
    )
    fast_w, fast_status, fast_model = _run_with_exports(
        lp_single_fast.np_integerizer_ortools_fast, kwargs, monkeypatch
    )
    # the private builder the public fast function uses gives the same model
    built = _exports(lp_single_fast._build_model(**copy.deepcopy(kwargs))[0])
    for key in ("lp", "mps", "proto", "terms"):
        assert fast_model[key] == up_model[key], key
        assert built[key] == up_model[key], key
    # no zero coefficient is stored: one term per non-zero incidence value plus the
    # relaxation variable (inequalities) or one per household (equality)
    incidence = kwargs["incidence"]
    total = kwargs["total_hh_control_index"]
    expected_terms = []
    for c in range(incidence.shape[0]):
        if c != total:
            n = int(np.count_nonzero(incidence[c]))
            expected_terms += [n + 1, n + 1]
    expected_terms.append(incidence.shape[1])
    assert up_model["terms"] == expected_terms
    assert fast_status == up_status
    assert fast_w.dtype == up_w.dtype
    assert np.array_equal(fast_w, up_w)
    return up_status


def _random_problem(seed, density, with_draw_keys, shape=None):
    """An integerizer LP shaped as ``Integerizer.integerize`` builds it."""
    rng = np.random.RandomState(seed)
    sample_count = int(rng.randint(20, 160))
    control_count = int(rng.randint(2, 10))
    if shape is not None:
        control_count, sample_count = shape
    total = int(rng.randint(0, control_count))
    values = rng.randint(1, 5, size=(control_count, sample_count)).astype(np.float64)
    incidence = np.where(
        rng.random_sample((control_count, sample_count)) < density, values, 0.0
    )
    incidence[total] = 1.0
    # first household non-zero in some controls and zero in others
    incidence[:, 0] = np.where(rng.random_sample(control_count) < 0.5, 0.0, 2.0)
    incidence[total, 0] = 1.0
    float_weights = rng.random_sample(sample_count) * 3.0 + 0.05
    float_weights[rng.random_sample(sample_count) < 0.1] = 2.0  # zero residuals
    int_weights = float_weights.astype(int)
    resid_weights = float_weights % 1.0
    totals = np.round(np.dot(float_weights, incidence.T))
    lp_rhs = np.maximum(totals - np.dot(int_weights, incidence.T), 0.0)
    max_inc = np.amax(incidence, axis=1)
    num_households = totals[total]
    kwargs = dict(
        incidence=incidence,
        resid_weights=resid_weights,
        log_resid_weights=np.log(np.maximum(resid_weights, np.exp(-725))),
        control_importance_weights=rng.random_sample(control_count) * 1000.0,
        total_hh_control_index=total,
        lp_right_hand_side=lp_rhs,
        relax_ge_upper_bound=np.maximum(max_inc * num_households - lp_rhs, 0),
        hh_constraint_ge_bound=np.maximum(num_households * max_inc, lp_rhs),
        timeout_in_seconds=60,
    )
    if with_draw_keys:
        kwargs["draw_keys"] = (
            rng.gumbel(size=sample_count) + kwargs["log_resid_weights"]
        )
    return kwargs


DENSITIES = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]


@pytest.mark.parametrize("seed", list(range(36)))
def test_random_problems_build_the_same_model(seed, monkeypatch):
    density = DENSITIES[seed % len(DENSITIES)]
    kwargs = _random_problem(seed, density, with_draw_keys=bool(seed % 2))
    _assert_same_model_and_result(kwargs, monkeypatch)


@pytest.mark.parametrize("with_draw_keys", [False, True])
def test_seed_sized_problem_builds_the_same_model(with_draw_keys, monkeypatch):
    """Larger and sparser, like a seed-level LP: 14 % non-zero incidence."""
    kwargs = _random_problem(100, 0.14, with_draw_keys, shape=(40, 3000))
    _assert_same_model_and_result(kwargs, monkeypatch)


def test_timing_is_filled_like_upstream():
    kwargs = _random_problem(1, 0.3, with_draw_keys=False)
    timing = {}
    lp_single_fast.np_integerizer_ortools_fast(**copy.deepcopy(kwargs), timing=timing)
    assert sorted(timing) == ["build_s", "solve_s"]


_CAPTURE_RUNNER = """
import pickle
import sys
from pathlib import Path

code_root, configs_tmp, tests_configs, example_dir, out_dir = sys.argv[1:6]
sys.path.insert(0, code_root)

from populationsim.core import inject, pipeline, tracing
from populationsim.integerizing import lp_ortools

problems = []
original = lp_ortools.np_integerizer_ortools


def capture(**kwargs):
    problems.append({k: (v.copy() if hasattr(v, "copy") else v) for k, v in kwargs.items()})
    return original(**kwargs)


lp_ortools.np_integerizer_ortools = capture

example = Path(example_dir)
inject.reinject_decorated_tables()
inject.add_injectable(
    "configs_dir", [Path(configs_tmp), Path(tests_configs), example / "configs"]
)
inject.add_injectable("output_dir", Path(out_dir))
inject.add_injectable("data_dir", example / "data")
inject.clear_cache()
tracing.config_logger()
pipeline.run(
    models=[
        "input_pre_processor",
        "setup_data_structures",
        "initial_seed_balancing",
        "meta_control_factoring",
        "final_seed_balancing",
        "integerize_final_seed_weights",
    ],
    resume_after=None,
)
pipeline.close_pipeline()
with open(Path(out_dir) / "problems.pkl", "wb") as f:
    pickle.dump(problems, f)
"""


@pytest.fixture(scope="module")
def example_problems(tmp_path_factory):
    """The single-integerizer LPs of the example's seed level, as called."""
    out_dir = tmp_path_factory.mktemp("lp_capture")
    configs_tmp = out_dir / "_configs_override"
    configs_tmp.mkdir()
    (configs_tmp / "settings.yaml").write_text(
        yaml.safe_dump({"inherit_settings": True})
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _CAPTURE_RUNNER,
            str(REPO_ROOT),
            str(configs_tmp),
            str(REPO_ROOT / "tests" / "configs"),
            str(REPO_ROOT / "examples" / "example_test"),
            str(out_dir),
        ],
        cwd=out_dir,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    with open(out_dir / "problems.pkl", "rb") as f:
        return pickle.load(f)


def test_example_problems_build_the_same_model(example_problems, monkeypatch):
    # the example's seed zones: the backstopped LP of each, and for the zones
    # where it is infeasible the retry with the explicit controls only
    assert len(example_problems) == 6
    statuses = []
    for kwargs in example_problems:
        assert "draw_keys" not in kwargs
        statuses.append(_assert_same_model_and_result(kwargs, monkeypatch))
        # the same problems with draw keys
        keyed = dict(kwargs)
        keyed["draw_keys"] = np.random.RandomState(7).gumbel(
            size=kwargs["incidence"].shape[1]
        )
        _assert_same_model_and_result(keyed, monkeypatch)
    # both the solved and the failed branch of the return value are compared
    assert "OPTIMAL" in statuses and "INFEASIBLE" in statuses


def _integerizer():
    incidence = pd.DataFrame({"num_hh": [1, 1], "a": [1, 0]})
    return Integerizer(
        incidence_table=incidence,
        control_importance_weights=pd.Series([1.0, 1.0]),
        float_weights=pd.Series([1.5, 0.5]),
        relaxed_control_totals=pd.Series([2.0, 1.0]),
        total_hh_control_value=2,
        total_hh_control_index=0,
        control_is_hh_based=pd.Series([True, True]),
    )


@pytest.mark.parametrize(
    "fast, cvxpy, expected",
    [
        (False, False, lp_ortools.np_integerizer_ortools),
        (True, False, lp_single_fast.np_integerizer_ortools_fast),
        (False, True, lp_cvx.np_integerizer_cvx),
        (True, True, lp_cvx.np_integerizer_cvx),
    ],
)
def test_dispatch(settings, fast, cvxpy, expected):
    settings[SETTING] = fast
    settings["USE_CVXPY"] = cvxpy
    assert _integerizer().integerizer_func is expected


def test_cvxpy_with_fast_paths_logs_that_the_fast_build_does_not_apply(
    settings, caplog, fork_logging
):
    import logging

    from populationsim.integerizing import single_integerizer

    settings[SETTING] = True
    settings["USE_CVXPY"] = True
    single_integerizer._LOGGED_FAST_BUILD_SKIPPED.clear()
    with caplog.at_level(logging.INFO, logger=single_integerizer.logger.name):
        _integerizer()
        _integerizer()
    messages = [
        r.getMessage() for r in caplog.records if "FAST_PATHS" in r.getMessage()
    ]
    assert len(messages) == 1
    assert "USE_CVXPY" in messages[0]

    # no message without FAST_PATHS, none for the OR-Tools solver
    single_integerizer._LOGGED_FAST_BUILD_SKIPPED.clear()
    caplog.clear()
    settings[SETTING] = False
    with caplog.at_level(logging.INFO, logger=single_integerizer.logger.name):
        _integerizer()
    settings[SETTING] = True
    settings["USE_CVXPY"] = False
    with caplog.at_level(logging.INFO, logger=single_integerizer.logger.name):
        _integerizer()
    assert not [r for r in caplog.records if "FAST_PATHS" in r.getMessage()]


# --- B. household expansion ---------------------------------------------------

GEOGRAPHIES = ["REGION", "PUMA", "TAZ"]


def _expansion_tables(group_count, used_count, seed, int_sample_weights=False):
    """A household group table and a sparse TAZ weight table using few groups.

    Group sizes vary from 1 to 40 households, with one group of 20,000; the
    household rows are shuffled so that the rows of a group are not contiguous,
    and the used groups are spread over the whole id range.
    """
    rng = np.random.RandomState(seed)
    sizes = rng.randint(1, 41, size=group_count)
    sizes[group_count // 3] = 20000
    group_id = np.repeat(np.arange(group_count, dtype=np.int64), sizes)
    rng.shuffle(group_id)
    if int_sample_weights:
        sample_weight = rng.randint(1, 300, size=len(group_id)).astype(np.int64)
    else:
        sample_weight = rng.random_sample(len(group_id)) * 250.0 + 1e-3
    household_groups = pd.DataFrame(
        {
            "group_id": group_id,
            "sample_weight": sample_weight,
        },
        index=pd.Index(np.arange(len(group_id), dtype=np.int64) * 3 + 5, name="hh_id"),
    )
    household_groups["hh_id"] = household_groups.index.astype(np.int64)
    used = np.sort(rng.choice(group_count, size=used_count, replace=False))
    used = np.union1d(used, [group_count // 3])
    rows = len(used) * 2
    weights = pd.DataFrame(
        {
            "REGION": np.ones(rows, dtype=np.int64),
            "PUMA": rng.randint(1, 4, size=rows).astype(np.int64),
            "TAZ": rng.randint(100, 140, size=rows).astype(np.int64),
            "hh_id": np.concatenate([used, rng.permutation(used)]).astype(np.int64),
            "integer_weight": rng.randint(1, 4, size=rows).astype(np.int32),
        },
        index=pd.Index(rng.permutation(rows * 3)[:rows]),
    )
    return household_groups, weights


def _run_expand(monkeypatch, household_groups, weights, fast, group_by=True):
    """Run the real step with stubbed pipeline access; return table and rng."""
    values = {
        "NO_INTEGERIZATION_EVER": False,
        "geographies": GEOGRAPHIES,
        "household_id_col": "hh_id",
        "seed_geography": "PUMA",
        "GROUP_BY_INCIDENCE_SIGNATURE": group_by,
        SETTING: fast,
    }
    rng = np.random.RandomState([11, 42])
    added = {}

    class _Generator:
        def get_external_rng(self, name):
            assert name == "expand_households"
            return rng

    def get_table(name):
        assert name == "household_groups"
        return household_groups.copy()

    def add_table(name, table, replace=False):
        added[name] = table

    with monkeypatch.context() as patch:
        patch.setattr(
            eh.config, "setting", lambda key, default=None: values.get(key, default)
        )
        patch.setattr(eh, "get_weight_table", lambda geo, sparse=False: weights.copy())
        patch.setattr(eh.pipeline, "get_table", get_table)
        patch.setattr(eh.pipeline, "get_rn_generator", lambda: _Generator())
        patch.setattr(eh.inject, "get_step_arg", lambda name, default=None: default)
        patch.setattr(eh.inject, "add_table", add_table)
        eh.expand_households()
    return added["expanded_household_ids"], rng


def _assert_same_rng_state(a, b):
    sa, sb = a.get_state(), b.get_state()
    assert sa[0] == sb[0]
    assert np.array_equal(sa[1], sb[1])
    assert sa[2:] == sb[2:]


def _assert_same_table(on, off):
    pd.testing.assert_frame_equal(on, off, check_exact=True)
    assert on.index.equals(off.index)
    assert list(on.dtypes) == list(off.dtypes)


@pytest.mark.parametrize("int_sample_weights", [False, True])
def test_expansion_fast_equals_upstream_on_sparse_groups(
    monkeypatch, int_sample_weights
):
    household_groups, weights = _expansion_tables(
        3000, 60, seed=3, int_sample_weights=int_sample_weights
    )
    off, off_rng = _run_expand(monkeypatch, household_groups, weights, fast=False)
    on, on_rng = _run_expand(monkeypatch, household_groups, weights, fast=True)
    assert len(off) == weights["integer_weight"].sum()
    _assert_same_table(on, off)
    _assert_same_rng_state(on_rng, off_rng)
    # the generator was used (one draw per expanded household)
    _assert_rng_advanced(off_rng)


def _assert_rng_advanced(rng):
    fresh = np.random.RandomState([11, 42])
    assert not np.array_equal(fresh.get_state()[1], rng.get_state()[1]) or (
        fresh.get_state()[2] != rng.get_state()[2]
    )


def test_expansion_fast_lists_equal_upstream_lists():
    """The per-group id and probability lists equal upstream's, for every used group."""
    household_groups, weights = _expansion_tables(500, 40, seed=5)
    upstream = {}
    for group_id, df in household_groups.groupby("group_id"):
        upstream[group_id] = [
            list(df["hh_id"]),
            list(df.sample_weight / df.sample_weight.sum()),
        ]
    used = weights["hh_id"].unique()
    fast = eh.household_lists_for_groups(household_groups, used, "hh_id")
    assert sorted(fast) == sorted(int(g) for g in used)
    for group_id, (ids, probs) in fast.items():
        assert ids == upstream[group_id][0]
        assert [type(i) for i in ids] == [type(i) for i in upstream[group_id][0]]
        assert np.array_equal(np.array(probs), np.array(upstream[group_id][1]))
        assert [type(p) for p in probs] == [type(p) for p in upstream[group_id][1]]


def _lists(household_groups, used):
    lists = eh.household_lists_for_groups(household_groups, used, "hh_id")
    return {int(group_id): value for group_id, value in lists.items()}


@pytest.mark.parametrize("dtype", [object, np.float64, np.int32])
def test_expansion_lists_do_not_depend_on_the_dtype_of_the_used_ids(dtype):
    """The step hands over an object array of group ids (the expanded table is
    built from mixed dtypes); object, float and int ids select the same groups
    and give the same lists as int64 ids."""
    household_groups, weights = _expansion_tables(300, 30, seed=3)
    used = weights["hh_id"].to_numpy()
    expected = _lists(household_groups, used)
    assert len(expected) > 0
    got = _lists(household_groups, used.astype(dtype))
    assert list(got) == list(expected)
    for group_id, (ids, probs) in expected.items():
        assert got[group_id][0] == ids
        assert np.array_equal(np.array(got[group_id][1]), np.array(probs))


def test_expansion_refuses_a_group_id_the_cast_would_change():
    household_groups, weights = _expansion_tables(100, 10, seed=4)
    used = weights["hh_id"].to_numpy().astype(np.float64)
    used[0] = 2.5
    with pytest.raises(ValueError, match="group id"):
        eh.household_lists_for_groups(household_groups, used, "hh_id")
    with pytest.raises(ValueError, match="group id"):
        eh.household_lists_for_groups(
            household_groups, np.array([1, "x"], dtype=object), "hh_id"
        )


def test_expansion_without_grouping_is_unchanged(monkeypatch):
    household_groups, weights = _expansion_tables(200, 20, seed=9)
    off, off_rng = _run_expand(
        monkeypatch, household_groups, weights, fast=False, group_by=False
    )
    on, on_rng = _run_expand(
        monkeypatch, household_groups, weights, fast=True, group_by=False
    )
    _assert_same_table(on, off)
    _assert_same_rng_state(on_rng, off_rng)


# --- differential runs of the example -------------------------------------------

VARIANTS = {
    "default": {},
    "no_grouping": {"GROUP_BY_INCIDENCE_SIGNATURE": False},
    "sequential": {"USE_SIMUL_INTEGERIZER": False},
}


def _final_tables(path):
    """Every table of the last checkpoint of a run's pipeline store."""
    with pd.HDFStore(path, mode="r") as store:
        last = store["checkpoints"].iloc[-1]
        tables = {}
        for name, checkpoint in last.items():
            if name in ("checkpoint_name", "timestamp") or not checkpoint:
                continue
            key = f"/{name}/{checkpoint}"
            if key in store:
                tables[name] = store[key]
        return tables


def _assert_example_equal(tmp_path, off_settings, on_settings, expect_ids=True):
    """Run the example with ``off_settings`` and ``on_settings`` and compare the
    expanded household ids and every weight, sparse and id table exactly."""
    off_ids = run_example(
        REPO_ROOT, tmp_path / "off", off_settings, allow_empty=not expect_ids
    )
    on_ids = run_example(
        REPO_ROOT, tmp_path / "on", on_settings, allow_empty=not expect_ids
    )
    if expect_ids:
        assert len(off_ids) > 0
    pd.testing.assert_frame_equal(on_ids, off_ids, check_exact=True)

    off_tables = _final_tables(tmp_path / "off" / "pipeline.h5")
    on_tables = _final_tables(tmp_path / "on" / "pipeline.h5")
    if expect_ids:
        assert "expanded_household_ids" in off_tables
    assert sorted(on_tables) == sorted(off_tables)
    compared = 0
    for name in off_tables:
        if (
            name.endswith("_weights")
            or name.endswith("_sparse")
            or (name == "expanded_household_ids")
        ):
            _assert_same_table(on_tables[name], off_tables[name])
            compared += 1
    assert compared > 0


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_example_on_equals_off(tmp_path, variant):
    overrides = VARIANTS[variant]
    _assert_example_equal(
        tmp_path, {**overrides, SETTING: False}, {**overrides, SETTING: True}
    )


# The example configs leave USE_NUMBA off, so the runs above never reach the
# lookup-table kernels of the balancers. These runs switch numba on in both
# runs; the fast kernels are used on the "on" side only.
NUMBA_VARIANTS = {
    "default": {},
    "sequential": {"USE_SIMUL_INTEGERIZER": False},
    "no_integerization": {"NO_INTEGERIZATION_EVER": True},
    "no_grouping": {"GROUP_BY_INCIDENCE_SIGNATURE": False},
    "float_seed_weights": {"SUB_BALANCE_WITH_FLOAT_SEED_WEIGHTS": True},
}


@pytest.mark.parametrize("variant", list(NUMBA_VARIANTS))
def test_example_with_numba_on_equals_off(tmp_path, variant):
    overrides = {**NUMBA_VARIANTS[variant], "USE_NUMBA": True}
    _assert_example_equal(
        tmp_path,
        {**overrides, SETTING: False},
        {**overrides, SETTING: True},
        expect_ids=variant != "no_integerization",
    )


def test_example_with_numba_and_both_run_time_options_equals_both_off(tmp_path):
    """Numba on, FAST_PATHS and STORE_POSITIVE_WEIGHT_ROWS_ONLY together against
    both off: the expanded household ids and the sparse tables are equal (the
    weight tables of the positive-rows option hold fewer rows by design)."""
    off = {"USE_NUMBA": True, SETTING: False, "STORE_POSITIVE_WEIGHT_ROWS_ONLY": False}
    on = {"USE_NUMBA": True, SETTING: True, "STORE_POSITIVE_WEIGHT_ROWS_ONLY": True}
    off_ids = run_example(REPO_ROOT, tmp_path / "off", off)
    on_ids = run_example(REPO_ROOT, tmp_path / "on", on)
    assert len(off_ids) > 0
    pd.testing.assert_frame_equal(on_ids, off_ids, check_exact=True)
    off_tables = _final_tables(tmp_path / "off" / "pipeline.h5")
    on_tables = _final_tables(tmp_path / "on" / "pipeline.h5")
    assert sorted(on_tables) == sorted(off_tables)
    sparse = [name for name in off_tables if name.endswith("_sparse")]
    assert sparse
    for name in sparse + ["expanded_household_ids"]:
        _assert_same_table(on_tables[name], off_tables[name])
