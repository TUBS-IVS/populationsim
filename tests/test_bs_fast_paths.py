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


def _upstream_draw(household_groups, group_ids, rng):
    """Upstream's draw, as ``expand_households`` runs it with FAST_PATHS off."""
    lists = {}
    for group_id, df in household_groups.groupby("group_id"):
        lists[group_id] = [
            list(df["hh_id"]),
            list(df.sample_weight / df.sample_weight.sum()),
        ]

    def chooser(group_id):
        return rng.choice(lists[group_id][0], p=lists[group_id][1])

    return pd.Series(group_ids).apply(chooser)


@pytest.mark.parametrize("int_sample_weights", [False, True])
def test_fast_draw_equals_upstream_choice(int_sample_weights):
    """Same ids, same dtype and the same next random number as a per-row
    ``RandomState.choice``, on groups that recur in many rows and have 1 to
    20,000 households."""
    household_groups, weights = _expansion_tables(
        500, 40, seed=5, int_sample_weights=int_sample_weights
    )
    group_ids = np.repeat(
        weights["hh_id"].to_numpy(), weights["integer_weight"].to_numpy()
    )
    group_ids = np.random.RandomState(1).permutation(np.tile(group_ids, 20))
    upstream_rng = np.random.RandomState([11, 42])
    fast_rng = np.random.RandomState([11, 42])
    expected = _upstream_draw(household_groups, group_ids, upstream_rng)
    got = eh.draw_household_ids(
        pd.Series(group_ids.astype(object)), household_groups, "hh_id", fast_rng
    )
    assert len(got) == len(group_ids) > 1000
    np.testing.assert_array_equal(got, expected.to_numpy())
    assert got.dtype == expected.dtype
    _assert_same_rng_state(fast_rng, upstream_rng)


def test_fast_draw_takes_a_singleton_group_and_zero_weights_like_upstream():
    household_groups = pd.DataFrame(
        {
            "hh_id": [10, 11, 12, 13, 14],
            "group_id": [3, 0, 3, 3, 7],
            "sample_weight": [0.5, 2.0, 0.0, 1.5, 4.0],
        }
    )
    group_ids = np.array([3, 7, 0, 3, 3, 7, 7, 0, 3] * 50)
    upstream_rng = np.random.RandomState(8)
    fast_rng = np.random.RandomState(8)
    expected = _upstream_draw(household_groups, group_ids, upstream_rng)
    got = eh.draw_household_ids(
        pd.Series(group_ids), household_groups, "hh_id", fast_rng
    )
    np.testing.assert_array_equal(got, expected.to_numpy())
    assert 12 not in set(got.tolist())
    _assert_same_rng_state(fast_rng, upstream_rng)


@pytest.mark.parametrize("dtype", [object, np.float64, np.int32])
def test_fast_draw_does_not_depend_on_the_dtype_of_the_group_ids(dtype):
    """The step hands over an object array of group ids (the expanded table is
    built from mixed dtypes); object, float and int ids select the same groups
    and draw the same households as int64 ids."""
    household_groups, weights = _expansion_tables(300, 30, seed=3)
    used = weights["hh_id"].to_numpy()
    reference = eh.draw_household_ids(
        pd.Series(used), household_groups, "hh_id", np.random.RandomState(2)
    )
    got = eh.draw_household_ids(
        pd.Series(used.astype(dtype)),
        household_groups,
        "hh_id",
        np.random.RandomState(2),
    )
    np.testing.assert_array_equal(got, reference)
    assert got.dtype == reference.dtype


def test_expansion_refuses_a_group_id_the_cast_would_change():
    household_groups, weights = _expansion_tables(100, 10, seed=4)
    used = weights["hh_id"].to_numpy().astype(np.float64)
    used[0] = 2.5
    rng = np.random.RandomState(1)
    with pytest.raises(ValueError, match="group id"):
        eh.draw_household_ids(pd.Series(used), household_groups, "hh_id", rng)
    with pytest.raises(ValueError, match="group id"):
        eh.draw_household_ids(
            pd.Series([1, "x"], dtype=object), household_groups, "hh_id", rng
        )
    with pytest.raises(ValueError, match="group id"):
        eh.draw_household_ids(
            pd.Series([1.0, np.nan], dtype=object), household_groups, "hh_id", rng
        )
    # nothing was drawn before the refusal
    _assert_same_rng_state(rng, np.random.RandomState(1))


def test_fast_draw_refuses_a_group_without_households():
    household_groups, weights = _expansion_tables(100, 10, seed=4)
    rng = np.random.RandomState(1)
    with pytest.raises(ValueError, match="group 100000"):
        eh.draw_household_ids(pd.Series([100000]), household_groups, "hh_id", rng)
    _assert_same_rng_state(rng, np.random.RandomState(1))


@pytest.mark.parametrize(
    "weights", [[0, 0], [-1, 2], [1, np.nan], [1, np.inf], [np.nan, np.nan]]
)
def test_fast_draw_names_a_group_with_invalid_weights(weights):
    """Upstream fails inside ``RandomState.choice`` (a ValueError for negative or
    non-normalised probabilities); the fast draw fails with a ValueError that
    names the group, before it draws anything."""
    household_groups = pd.DataFrame(
        {
            "hh_id": [1, 2, 3],
            "group_id": [4, 4, 5],
            "sample_weight": weights + [1.0],
        }
    )
    rng = np.random.RandomState(1)
    with pytest.raises(ValueError, match="group 4"):
        eh.draw_household_ids(pd.Series([5, 4, 5]), household_groups, "hh_id", rng)
    _assert_same_rng_state(rng, np.random.RandomState(1))


@pytest.mark.parametrize("id_offset", [0, 2**40])
def test_fast_draw_keeps_upstream_id_dtype(monkeypatch, id_offset):
    """The fast draw returns the ``hh_id`` dtype the upstream path returns on this
    platform (int32 on Windows with NumPy 1.x, which builds a Python list of ids
    and lets ``RandomState.choice`` pick the default integer; int64 where the ids
    do not fit), not the dtype of the id column."""
    household_groups, weights = _expansion_tables(200, 20, seed=6)
    household_groups["hh_id"] = household_groups["hh_id"] + id_offset
    off, off_rng = _run_expand(monkeypatch, household_groups, weights, fast=False)
    on, on_rng = _run_expand(monkeypatch, household_groups, weights, fast=True)
    assert on["hh_id"].dtype == off["hh_id"].dtype
    assert list(on.dtypes) == list(off.dtypes)
    _assert_same_table(on, off)
    _assert_same_rng_state(on_rng, off_rng)


@pytest.mark.parametrize("dtype", [object, np.int64, np.float64])
def test_fast_draw_of_no_rows_has_the_dtype_of_the_upstream_result(dtype):
    """With no expanded row upstream's apply returns an empty Series of the dtype
    of the group id Series (object in the step); the fast draw does the same and
    draws nothing."""
    household_groups, _ = _expansion_tables(50, 5, seed=2)
    group_ids = np.array([], dtype=dtype)
    expected = _upstream_draw(household_groups, group_ids, np.random.RandomState(3))
    rng = np.random.RandomState(3)
    got = eh.draw_household_ids(group_ids, household_groups, "hh_id", rng)
    assert len(got) == 0
    assert got.dtype == expected.dtype
    _assert_same_rng_state(rng, np.random.RandomState(3))


def test_expansion_with_no_rows_is_equal_on_and_off(monkeypatch):
    household_groups, weights = _expansion_tables(50, 5, seed=2)
    weights = weights.iloc[:0]
    off, off_rng = _run_expand(monkeypatch, household_groups, weights, fast=False)
    on, on_rng = _run_expand(monkeypatch, household_groups, weights, fast=True)
    assert len(on) == 0
    assert on["hh_id"].dtype == off["hh_id"].dtype
    _assert_same_table(on, off)
    _assert_same_rng_state(on_rng, off_rng)


def test_fast_draw_names_the_first_invalid_group_in_row_order():
    """Upstream's per-row choice fails at the first row whose group is invalid;
    the group named is the one of that row, not the lowest id or the last one."""
    household_groups = pd.DataFrame(
        {
            "hh_id": [1, 2, 3, 4, 5],
            "group_id": [4, 5, 6, 7, 7],
            "sample_weight": [0.0, 1.0, -1.0, 1.0, 1.0],
        }
    )
    rng = np.random.RandomState(1)
    with pytest.raises(ValueError, match="group 6"):
        eh.draw_household_ids(
            pd.Series([7, 5, 6, 4, 6]), household_groups, "hh_id", rng
        )
    with pytest.raises(ValueError, match="group 4"):
        eh.draw_household_ids(pd.Series([7, 4, 6, 5]), household_groups, "hh_id", rng)
    # a group without households counts the same way
    with pytest.raises(ValueError, match="group 99"):
        eh.draw_household_ids(pd.Series([5, 99, 6]), household_groups, "hh_id", rng)
    _assert_same_rng_state(rng, np.random.RandomState(1))


def test_expansion_off_path_does_not_use_the_fast_draw(monkeypatch):
    household_groups, weights = _expansion_tables(100, 10, seed=7)

    def refuse(*args, **kwargs):
        raise AssertionError("draw_household_ids called with FAST_PATHS off")

    monkeypatch.setattr(eh, "draw_household_ids", refuse)
    off, _ = _run_expand(monkeypatch, household_groups, weights, fast=False)
    assert len(off) == weights["integer_weight"].sum()
    with pytest.raises(AssertionError, match="FAST_PATHS off"):
        _run_expand(monkeypatch, household_groups, weights, fast=True)


def test_expansion_fast_path_logs_its_batch(monkeypatch, caplog, fork_logging):
    import logging

    household_groups, weights = _expansion_tables(100, 10, seed=7)
    with caplog.at_level(logging.INFO, logger=eh.logger.name):
        _run_expand(monkeypatch, household_groups, weights, fast=True)
    lines = [r.getMessage() for r in caplog.records if "FAST_PATHS" in r.getMessage()]
    assert len(lines) == 1 and "batch" in lines[0]


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


# --- D. result frames of the integerizers (reshape_result) -----------------------


def _reshape_inputs(
    zone_count,
    household_count=40,
    seed=0,
    zero_zone=None,
    integer_dtype=np.int64,
    id_dtype=np.int64,
):
    """Float and integer weights, one column per zone, hh ids unsorted and named."""
    rng = np.random.RandomState(seed)
    ids = rng.permutation(household_count * 3)[:household_count].astype(id_dtype)
    index = pd.Index(ids, name="hh_id")
    names = ["zone_%d" % (i + 1) for i in range(zone_count)]
    float_values = np.where(
        rng.rand(household_count, zone_count) < 0.2,
        rng.rand(household_count, zone_count) * 5.0,
        0.0,
    )
    integer_values = np.where(float_values > 2.5, np.round(float_values), 0.0)
    # a row positive in the integer weight only and one in the float weight only
    float_values[0, 0] = 0.0
    integer_values[0, 0] = 1.0
    float_values[1, 0] = 0.25
    integer_values[1, 0] = 0.0
    if zero_zone is not None:
        float_values[:, zero_zone] = 0.0
        integer_values[:, zero_zone] = 0.0
    float_weights = pd.DataFrame(float_values, index=index, columns=names)
    integerized = pd.DataFrame(
        integer_values.astype(integer_dtype), index=index, columns=names
    )
    return float_weights, integerized, names


def _zones(names, zone_type):
    return pd.Series(names, index=[zone_type(10 + 3 * i) for i in range(len(names))])


def _run_reshape(settings, fast, float_weights, integerized, zones, positive):
    """(result, tally) of the upstream loop (fast=False) or of the fast build."""
    from populationsim.core import positive_weight_rows as pwr
    from populationsim.integerizing import wrappers as iw

    pwr._TALLY.clear()
    try:
        if fast:
            result = iw._reshape_result_fast(
                float_weights, integerized, "TAZ", zones, positive
            )
        else:
            settings.clear()
            settings.update(
                {SETTING: False, "STORE_POSITIVE_WEIGHT_ROWS_ONLY": positive}
            )
            result = iw.reshape_result(float_weights, integerized, "TAZ", zones)
        return result, dict(pwr._TALLY)
    finally:
        pwr._TALLY.clear()


def _assert_reshape_equal(settings, float_weights, integerized, zones, positive):
    up, up_tally = _run_reshape(
        settings, False, float_weights, integerized, zones, positive
    )
    fast, fast_tally = _run_reshape(
        settings, True, float_weights, integerized, zones, positive
    )
    pd.testing.assert_frame_equal(
        fast, up, check_exact=True, check_index_type=True, check_column_type=True
    )
    assert list(fast.columns) == list(up.columns)
    assert fast.index.dtype == up.index.dtype
    assert list(fast.dtypes) == list(up.dtypes)
    assert fast_tally == up_tally
    assert bool(up_tally) == positive  # the tally is part of what is compared
    return up


@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("zone_type", [int, np.int64, np.int32, float, str])
@pytest.mark.parametrize("zone_count", [1, 7])
@pytest.mark.parametrize("integer_dtype", [np.int64, np.float64])
def test_reshape_result_fast_equals_upstream(
    settings, positive, zone_type, zone_count, integer_dtype
):
    float_weights, integerized, names = _reshape_inputs(
        zone_count, seed=zone_count, integer_dtype=integer_dtype
    )
    zones = _zones(names, zone_type)
    values = float_weights.to_numpy()
    assert (values > 0).any() and (values == 0).any()  # the filter has work to do
    _assert_reshape_equal(settings, float_weights, integerized, zones, positive)


@pytest.mark.parametrize("positive", [False, True])
def test_reshape_result_fast_keeps_an_all_zero_zone(settings, positive):
    float_weights, integerized, names = _reshape_inputs(5, seed=3, zero_zone=2)
    result = _assert_reshape_equal(
        settings, float_weights, integerized, _zones(names, int), positive
    )
    assert (result["TAZ"] == 16).sum() == (0 if positive else 40)


@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("zone_count", [1, 4])
def test_reshape_result_fast_with_nothing_positive_keeps_the_dense_dtypes(
    settings, positive, zone_count
):
    float_weights, integerized, names = _reshape_inputs(zone_count)
    float_weights[:] = 0.0
    integerized[:] = 0
    result = _assert_reshape_equal(
        settings, float_weights, integerized, _zones(names, int), positive
    )
    assert len(result.index) == (0 if positive else 40 * zone_count)


@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("id_kind", ["int64", "int32", "object", "descending"])
def test_reshape_result_fast_keeps_the_household_id_dtype(settings, positive, id_kind):
    float_weights, integerized, names = _reshape_inputs(3, seed=5)
    if id_kind == "object":
        index = pd.Index(["h%d" % i for i in range(40)], name="hh_id")
    elif id_kind == "int32":
        index = pd.Index(np.arange(40, 0, -1).astype(np.int32), name="hh_id")
    elif id_kind == "descending":
        index = pd.Index(np.arange(40, 0, -1), name="hh_id")
    else:
        index = pd.Index(np.arange(40, dtype=np.int64) * 7, name="hh_id")
    float_weights.index = index
    integerized.index = index
    _assert_reshape_equal(
        settings, float_weights, integerized, _zones(names, int), positive
    )


def test_reshape_result_fast_repeated_zone_id_overwrites_its_tally_entry(settings):
    float_weights, integerized, names = _reshape_inputs(3, seed=8)
    zones = pd.Series(names, index=[10, 20, 10])
    _assert_reshape_equal(settings, float_weights, integerized, zones, True)


# an integer weight table cannot hold NaN (astype(int) refuses it), so no such case
@pytest.mark.parametrize(
    "bad, column", [(np.nan, "float"), (-1.0, "float"), (-1.0, "integer")]
)
@pytest.mark.parametrize("bad_zone", [0, 2])
def test_reshape_result_fast_raises_the_upstream_error_and_leaves_its_tally(
    settings, bad, column, bad_zone
):
    float_weights, integerized, names = _reshape_inputs(4, seed=2)
    if column == "float":
        float_weights.iloc[3, bad_zone] = bad
    else:
        integerized.iloc[3, bad_zone] = bad
    # a second, later offender must not change which zone is reported
    float_weights.iloc[5, 3] = -2.0
    zones = _zones(names, int)

    outcomes = []
    for fast in (False, True):
        try:
            _run_reshape(settings, fast, float_weights, integerized, zones, True)
        except ValueError as error:
            outcomes.append(str(error))
    assert len(outcomes) == 2 and outcomes[0] == outcomes[1]
    assert "zone %d " % (10 + 3 * bad_zone) in outcomes[0]


def _tally_after_error(settings, fast, float_weights, integerized, zones):
    from populationsim.core import positive_weight_rows as pwr
    from populationsim.integerizing import wrappers as iw

    pwr._TALLY.clear()
    settings.update({SETTING: False, "STORE_POSITIVE_WEIGHT_ROWS_ONLY": True})
    try:
        with pytest.raises(ValueError):
            if fast:
                iw._reshape_result_fast(float_weights, integerized, "TAZ", zones, True)
            else:
                iw.reshape_result(float_weights, integerized, "TAZ", zones)
        return dict(pwr._TALLY)
    finally:
        pwr._TALLY.clear()


@pytest.mark.parametrize("bad_zone", [0, 1, 3])
def test_reshape_result_fast_counts_the_zones_before_an_offender_like_upstream(
    settings, bad_zone
):
    float_weights, integerized, names = _reshape_inputs(4, seed=2)
    float_weights.iloc[3, bad_zone] = np.nan
    zones = _zones(names, int)
    upstream = _tally_after_error(settings, False, float_weights, integerized, zones)
    fast = _tally_after_error(settings, True, float_weights, integerized, zones)
    assert len(upstream) == bad_zone
    assert fast == upstream


def _tally_of_failing_call(settings, fast, float_weights, integerized, zones, positive):
    from populationsim.core import positive_weight_rows as pwr
    from populationsim.integerizing import wrappers as iw

    pwr._TALLY.clear()
    settings.clear()
    settings.update({SETTING: False, "STORE_POSITIVE_WEIGHT_ROWS_ONLY": positive})
    try:
        with pytest.raises(Exception) as error:
            if fast:
                iw._reshape_result_fast(
                    float_weights, integerized, "TAZ", zones, positive
                )
            else:
                iw.reshape_result(float_weights, integerized, "TAZ", zones)
        return type(error.value), str(error.value), dict(pwr._TALLY)
    finally:
        pwr._TALLY.clear()


@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("bad_zone", [0, 2, 3])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_reshape_result_fast_integer_cast_error_and_tally_are_upstreams(
    settings, positive, bad_zone, bad
):
    """A float integerized_weights table with a value astype(int) refuses: upstream
    casts zone by zone, so the zones before it are tallied and its error is raised."""
    float_weights, integerized, names = _reshape_inputs(
        4, seed=2, integer_dtype=np.float64
    )
    integerized.iloc[5, bad_zone] = bad
    zones = _zones(names, int)
    upstream = _tally_of_failing_call(
        settings, False, float_weights, integerized, zones, positive
    )
    fast = _tally_of_failing_call(
        settings, True, float_weights, integerized, zones, positive
    )
    assert fast == upstream
    assert "non-finite" in upstream[1]
    assert len(upstream[2]) == (bad_zone if positive else 0)


def test_reshape_result_fast_invalid_weight_before_a_cast_error_wins_like_upstream(
    settings,
):
    float_weights, integerized, names = _reshape_inputs(
        4, seed=2, integer_dtype=np.float64
    )
    float_weights.iloc[3, 1] = -1.0  # zone 1: negative balanced weight
    integerized.iloc[5, 2] = np.nan  # zone 2: not castable
    zones = _zones(names, int)
    upstream = _tally_of_failing_call(
        settings, False, float_weights, integerized, zones, True
    )
    fast = _tally_of_failing_call(
        settings, True, float_weights, integerized, zones, True
    )
    assert "negative" in upstream[1] and len(upstream[2]) == 1
    assert fast == upstream
    # and the other way round: the cast error of zone 1 comes before zone 2's NaN
    float_weights.iloc[3, 1] = 0.0
    float_weights.iloc[3, 2] = np.nan
    integerized.iloc[5, 1] = np.nan
    upstream = _tally_of_failing_call(
        settings, False, float_weights, integerized, zones, True
    )
    fast = _tally_of_failing_call(
        settings, True, float_weights, integerized, zones, True
    )
    assert "non-finite" in upstream[1] and len(upstream[2]) == 1
    assert fast == upstream


def test_reshape_result_fast_checks_columns_in_upstream_order(settings):
    """Within a zone: balanced_weight before integer_weight, NaN before negative."""
    float_weights, integerized, names = _reshape_inputs(3, seed=4)
    zones = _zones(names, int)
    float_weights.iloc[2, 1] = -1.0
    integerized.iloc[3, 1] = -1
    messages = []
    for fast in (False, True):
        try:
            _run_reshape(settings, fast, float_weights, integerized, zones, True)
        except ValueError as error:
            messages.append(str(error))
    assert len(messages) == 2 and messages[0] == messages[1]
    assert "column balanced_weight" in messages[0] and "negative" in messages[0]

    float_weights.iloc[2, 1] = 0.0
    float_weights.iloc[4, 1] = np.nan
    integerized.iloc[3, 1] = -1
    messages = []
    for fast in (False, True):
        try:
            _run_reshape(settings, fast, float_weights, integerized, zones, True)
        except ValueError as error:
            messages.append(str(error))
    assert len(messages) == 2 and messages[0] == messages[1]
    assert "NaN" in messages[0]


def test_reshape_result_dispatches_on_fast_paths(settings, monkeypatch):
    from populationsim.integerizing import wrappers as iw

    float_weights, integerized, names = _reshape_inputs(3, seed=6)
    zones = _zones(names, int)
    calls = []
    real = iw._reshape_result_fast

    def spy(*args):
        calls.append(args[-1])
        return real(*args)

    monkeypatch.setattr(iw, "_reshape_result_fast", spy)
    for fast, positive in [(False, False), (True, False), (True, True)]:
        settings.clear()
        settings.update({SETTING: fast, "STORE_POSITIVE_WEIGHT_ROWS_ONLY": positive})
        iw.reshape_result(float_weights, integerized, "TAZ", zones)
    assert calls == [False, True]


@pytest.mark.parametrize("positive", [False, True])
def test_reshape_result_without_households_is_the_upstream_result(settings, positive):
    """No rows: equal to upstream (the upstream loop builds it)."""
    from populationsim.integerizing import wrappers as iw

    float_weights, integerized, names = _reshape_inputs(2, seed=1)
    zones = _zones(names, int)
    results = []
    for fast in (True, False):
        settings.clear()
        settings.update({SETTING: fast, "STORE_POSITIVE_WEIGHT_ROWS_ONLY": positive})
        results.append(
            iw.reshape_result(
                float_weights.iloc[:0], integerized.iloc[:0], "TAZ", zones
            )
        )
    pd.testing.assert_frame_equal(results[0], results[1], check_exact=True)


@pytest.mark.parametrize("positive", [False, True])
def test_reshape_result_without_zones_fails_like_upstream(settings, positive):
    from populationsim.integerizing import wrappers as iw

    float_weights, integerized, _ = _reshape_inputs(2, seed=1)
    zones = _zones([], int)
    errors = []
    for fast in (True, False):
        settings.clear()
        settings.update({SETTING: fast, "STORE_POSITIVE_WEIGHT_ROWS_ONLY": positive})
        with pytest.raises((ValueError, IndexError)) as error:
            iw.reshape_result(float_weights[[]], integerized[[]], "TAZ", zones)
        errors.append((type(error.value), str(error.value)))
    assert errors[0] == errors[1]
