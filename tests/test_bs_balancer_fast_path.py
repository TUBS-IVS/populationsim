# PopulationSim
# See full license in LICENSE.txt.
"""The ``FAST_PATHS`` numba balancer kernels are identical to the originals, bit for bit.

The fast kernels skip ``exp(log_gamma * inc)`` for a zero incidence cell
(``w * exp(0.0) == w`` exactly). The balanced weights feed an integerizer whose
choices can flip on a last-bit difference, so every comparison here is on the
raw bytes (``-0.0`` against ``0.0`` would fail), never within a tolerance.
All kernels run compiled (no ``py_func``): the claim is about the compiled code.
"""

import logging
import struct

import numpy as np
import pandas as pd
import pytest

from populationsim.balancing import balancers_numba, wrappers
from populationsim.balancing.balancers import np_balancer_py, np_simul_balancer_py
from populationsim.balancing.balancers_numba import (
    np_balancer_numba,
    np_balancer_numba_fast,
    np_simul_balancer_numba,
    np_simul_balancer_numba_fast,
)
from populationsim.balancing.single_balancer import ListBalancer
from populationsim.balancing.simul_balancer import SimultaneousListBalancer
from populationsim.core import config
from tests.bs_differential import REPO_ROOT, run_example

# Row counts that are and are not multiples of the SIMD widths (2, 4, 8).
EDGE_SIZES = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 63, 65, 127, 129, 5000]


def _bytes_equal(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _status_key(status):
    converged, iteration, delta, max_gamma_dif = status
    return (
        bool(converged),
        int(iteration),
        struct.pack("<d", float(delta)),
        struct.pack("<d", float(max_gamma_dif)),
    )


def _incidence(rng, control_count, sample_count, densities):
    """Integer counts 0..4; one density per control row (0 = all-zero row)."""
    rows = []
    for density in densities:
        present = rng.random(sample_count) < density
        rows.append(present * rng.integers(1, 5, size=sample_count))
    return np.array(rows, dtype=np.int64).reshape(control_count, sample_count)


def _densities(rng, control_count):
    choices = rng.random(control_count)
    densities = np.where(choices < 0.15, 0.0, np.where(choices > 0.85, 1.0, choices))
    return densities


def _weights_and_bounds(rng, sample_count):
    """Weights at, inside and outside their bounds, including -0.0 and 0.0."""
    lb = np.where(
        rng.random(sample_count) < 0.5, 0.0, rng.uniform(0.0, 2.0, sample_count)
    )
    ub = lb + np.exp(rng.uniform(0.0, 8.0, sample_count))
    w = np.exp(rng.uniform(-3.0, 7.0, sample_count))
    kind = rng.integers(0, 7, size=sample_count)
    w = np.where(kind == 1, lb, w)  # exactly at the lower bound
    w = np.where(kind == 2, ub, w)  # exactly at the upper bound
    w = np.where(kind == 3, ub * 1.5 + 1.0, w)  # above the upper bound on entry
    w = np.where((kind == 4) & (lb > 0), lb * 0.5, w)  # below the lower bound
    w = np.where(kind == 5, -0.0, w)  # negative zero
    w = np.where(kind == 6, 0.0, w)
    neg_zero_bound = rng.random(sample_count) < 0.05
    lb = np.where(neg_zero_bound & (lb == 0.0), -0.0, lb)
    return w, lb, ub


def _single_problem(seed, sample_count=None):
    rng = np.random.default_rng(seed)
    if sample_count is None:
        sample_count = int(np.exp(rng.uniform(0.0, np.log(5000.0))))
    control_count = int(rng.integers(1, 31))
    incidence = _incidence(
        rng, control_count, sample_count, _densities(rng, control_count)
    )
    master = -1
    if rng.random() < 0.7:
        master = int(rng.integers(0, control_count))
        incidence[master, :] = 1
    w, lb, ub = _weights_and_bounds(rng, sample_count)
    target = np.exp(rng.uniform(-1.0, 4.0, sample_count))
    totals = (incidence * target).sum(axis=1) * rng.uniform(0.5, 1.5, control_count)
    totals[rng.random(control_count) < 0.1] = 0.0
    constraint = np.maximum(totals, 0.1)
    importance = np.maximum(np.exp(rng.uniform(0.0, np.log(1e9), control_count)), 1.0)
    max_iterations = int(rng.integers(1, 250))

    # The callers pass incidence_table.values.T, which is C-contiguous for a
    # pandas frame (its block is stored transposed) and int64 or float64 with
    # NUMBA_PRECISION float64; both layouts are tested.
    variant = seed % 6
    fortran = variant % 2 == 1
    inc = np.ascontiguousarray(incidence.T).T if fortran else incidence
    if variant in (2, 3):
        inc = inc.astype(np.float64)
    elif variant in (4, 5):
        # NUMBA_PRECISION float32: every array is cast to float32 by the caller
        inc = inc.astype(np.float32)
        w, lb, ub = (a.astype(np.float32) for a in (w, lb, ub))
        constraint = constraint.astype(np.float32)
        importance = importance.astype(np.float32)
    return (
        sample_count,
        control_count,
        master,
        inc,
        w,
        lb,
        ub,
        constraint,
        importance,
        max_iterations,
    )


def _simul_problem(seed, sample_count=None):
    rng = np.random.default_rng(seed)
    if sample_count is None:
        sample_count = int(np.exp(rng.uniform(0.0, np.log(5000.0))))
    control_count = int(rng.integers(1, 31))
    zone_count = int(rng.integers(1, 13))
    incidence = _incidence(
        rng, control_count, sample_count, _densities(rng, control_count)
    )
    master = int(rng.integers(0, control_count))
    incidence[master, :] = 1
    parent = np.exp(rng.uniform(-2.0, 5.0, sample_count))
    parent[rng.random(sample_count) < 0.05] = 0.0
    shares = rng.dirichlet(np.ones(zone_count), size=sample_count)  # (n, zones)
    sub_weights = parent[:, None] * shares
    kind = rng.integers(0, 6, size=sub_weights.shape)
    sub_weights = np.where(kind == 1, parent[:, None], sub_weights)  # at the bound
    sub_weights = np.where(kind == 2, parent[:, None] * 2.0 + 1.0, sub_weights)
    sub_weights = np.where(kind == 3, -0.0, sub_weights)
    sub_weights = np.where(kind == 4, 0.0, sub_weights)
    lb = np.zeros(sample_count)
    lb[rng.random(sample_count) < 0.05] = -0.0
    some_lb = rng.random(sample_count) < 0.2
    lb[some_lb] = parent[some_lb] * 0.1
    ub = parent.copy()
    zone_targets = rng.dirichlet(np.ones(zone_count), size=control_count)
    parent_controls = np.maximum(
        (incidence * parent).sum(axis=1) * rng.uniform(0.5, 1.5, control_count), 0.1
    )
    sub_controls = (parent_controls[:, None] * zone_targets).astype("float").T
    importance = np.maximum(np.exp(rng.uniform(0.0, np.log(1e9), control_count)), 1.0)
    max_iterations = int(rng.integers(1, 120))
    if seed % 2 == 0:
        # the caller's layouts: incidence and sub_weights are transposed views
        inc = np.ascontiguousarray(incidence.T).T.astype(np.float64)
        sub_weights = np.ascontiguousarray(sub_weights).T
    else:
        inc = incidence.astype(np.float64)
        sub_weights = np.ascontiguousarray(sub_weights.T)
        sub_controls = np.ascontiguousarray(sub_controls)
    return (
        sample_count,
        control_count,
        zone_count,
        master,
        inc,
        parent,
        lb,
        ub,
        sub_weights,
        parent_controls,
        importance,
        sub_controls,
        max_iterations,
    )


def _copy_args(args):
    # order="K" keeps the Fortran layout of the callers' transposed arrays
    # (ndarray.copy() would make them C-contiguous and compile another kernel).
    return tuple(a.copy(order="K") if isinstance(a, np.ndarray) else a for a in args)


SINGLE_CASES = [(seed, None) for seed in range(48)] + [
    (1000 + i, n) for i, n in enumerate(EDGE_SIZES)
]
SIMUL_CASES = [(seed, None) for seed in range(44)] + [
    (2000 + i, n) for i, n in enumerate(EDGE_SIZES)
]


@pytest.mark.parametrize("seed, sample_count", SINGLE_CASES)
def test_single_fast_kernel_is_bitwise_identical(seed, sample_count):
    args = _single_problem(seed, sample_count)
    w_orig, rf_orig, st_orig = np_balancer_numba(*_copy_args(args))
    w_fast, rf_fast, st_fast = np_balancer_numba_fast(*_copy_args(args))
    assert _bytes_equal(w_orig, w_fast)
    assert _bytes_equal(rf_orig, rf_fast)
    assert _status_key(st_orig) == _status_key(st_fast)


@pytest.mark.parametrize("seed, sample_count", SIMUL_CASES)
def test_simul_fast_kernel_is_bitwise_identical(seed, sample_count):
    args = _simul_problem(seed, sample_count)
    w_orig, rf_orig, st_orig = np_simul_balancer_numba(*_copy_args(args))
    w_fast, rf_fast, st_fast = np_simul_balancer_numba_fast(*_copy_args(args))
    assert _bytes_equal(w_orig, w_fast)
    assert _bytes_equal(rf_orig, rf_fast)
    assert _status_key(st_orig) == _status_key(st_fast)


def test_random_problems_exercise_the_edge_cases():
    """The generators really produce what the bitwise tests claim to cover."""
    single = [_single_problem(seed) for seed, _ in SINGLE_CASES[:48]]
    simul = [_simul_problem(seed) for seed, _ in SIMUL_CASES[:44]]
    assert {p[3].dtype for p in single} == {
        np.dtype(np.int64),
        np.dtype(np.float64),
        np.dtype(np.float32),
    }
    for dtype in (np.int64, np.float64, np.float32):
        layouts = {p[3].flags.c_contiguous for p in single if p[3].dtype == dtype}
        assert layouts == {True, False}, dtype
    assert {p[8].flags.c_contiguous for p in simul} == {True, False}
    assert any(p[0] > 2000 for p in single) and any(p[0] < 10 for p in single)
    incidences = [np.asarray(p[3]) for p in single] + [p[4] for p in simul]
    assert any((inc == 0).all(axis=1).any() for inc in incidences)
    assert any((inc != 0).all(axis=1).any() for inc in incidences)
    assert any((inc > 1).any() for inc in incidences)
    weights = np.concatenate([p[4].astype(np.float64) for p in single])
    assert (np.signbit(weights) & (weights == 0)).any()
    assert any((p[4] > p[6]).any() and (p[4] == p[5]).any() for p in single)
    assert any((p[4] < p[5]).any() for p in single)
    importances = np.concatenate([p[8].astype(np.float64) for p in single])
    assert importances.min() < 10 and importances.max() > 1e8
    assert {p[2] for p in simul} >= {1, 12} or len({p[2] for p in simul}) > 8


@pytest.fixture
def settings(monkeypatch):
    """Inject a settings dict by patching config.setting (and the wrapper's import)."""
    values = {}

    def fake_setting(key, default=None):
        return values.get(key, default)

    monkeypatch.setattr(config, "setting", fake_setting)
    monkeypatch.setattr(wrappers, "setting", fake_setting)
    return values


def _list_balancer(use_numba, fast_paths):
    incidence = pd.DataFrame({"hh": [1, 1, 1], "p": [1, 0, 2]})
    return ListBalancer(
        incidence_table=incidence,
        initial_weights=np.ones(3),
        control_totals=[3, 3],
        control_importance_weights=[10, 10],
        lb_weights=0,
        ub_weights=30,
        master_control_index=0,
        max_iterations=100,
        use_numba=use_numba,
        numba_precision="float64",
        fast_paths=fast_paths,
    )


def _simul_balancer(use_numba, fast_paths):
    incidence = pd.DataFrame({"hh": [1, 1, 1], "p": [1, 0, 2]})
    zones = pd.Series(["Z_1", "Z_2"], index=[1, 2])
    controls = pd.DataFrame(
        {
            "name": ["hh", "p"],
            "importance": [10, 10],
            "total": [3, 3],
            "Z_1": [2, 1],
            "Z_2": [1, 2],
        }
    )
    return SimultaneousListBalancer(
        incidence_table=incidence,
        parent_weights=pd.Series([1.0, 1.0, 1.0]),
        controls=controls,
        sub_control_zones=zones,
        total_hh_control_col="hh",
        use_numba=use_numba,
        numba_precision="float64",
        fast_paths=fast_paths,
    )


@pytest.mark.parametrize(
    "use_numba, fast_paths, single, simul",
    [
        (True, False, np_balancer_numba, np_simul_balancer_numba),
        (True, True, np_balancer_numba_fast, np_simul_balancer_numba_fast),
        (False, False, np_balancer_py, np_simul_balancer_py),
        (False, True, np_balancer_py, np_simul_balancer_py),
    ],
)
def test_dispatch_selects_the_fast_kernel_only_with_numba_and_fast_paths(
    use_numba, fast_paths, single, simul
):
    assert _list_balancer(use_numba, fast_paths).balancer is single
    assert _simul_balancer(use_numba, fast_paths).balancer is simul


def test_balancers_default_to_the_upstream_kernels():
    incidence = pd.DataFrame({"hh": [1, 1]})
    balancer = ListBalancer(
        incidence_table=incidence,
        initial_weights=np.ones(2),
        control_totals=[2],
        control_importance_weights=[1],
        lb_weights=0,
        ub_weights=30,
        master_control_index=0,
        max_iterations=10,
        use_numba=True,
        numba_precision="float64",
    )
    assert balancer.balancer is np_balancer_numba


class _Captured(Exception):
    pass


@pytest.mark.parametrize("flag", [None, False, True])
def test_wrappers_pass_the_fast_paths_setting(settings, monkeypatch, flag):
    if flag is not None:
        settings["FAST_PATHS"] = flag
    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        raise _Captured()

    monkeypatch.setattr(wrappers, "ListBalancer", capture)
    monkeypatch.setattr(wrappers, "SimultaneousListBalancer", capture)

    control_spec = pd.DataFrame(
        {"target": ["hh", "p"], "importance": [10, 10], "geography": ["Z", "Z"]}
    )
    incidence = pd.DataFrame({"hh": [1, 1], "p": [1, 0]})
    with pytest.raises(_Captured):
        wrappers.do_balancing(
            control_spec=control_spec,
            total_hh_control_col="hh",
            max_expansion_factor=None,
            min_expansion_factor=None,
            absolute_upper_bound=None,
            absolute_lower_bound=None,
            incidence_df=incidence,
            control_totals=pd.Series([2, 1], index=["hh", "p"]),
            initial_weights=pd.Series([1.0, 1.0]),
            use_hard_constraints=False,
            use_numba=True,
            numba_precision="float64",
        )
    assert seen["fast_paths"] is bool(flag)
    seen.clear()
    with pytest.raises(_Captured):
        wrappers.do_simul_balancing(
            incidence_df=incidence,
            parent_weights=pd.Series([1.0, 1.0]),
            sub_controls_df=pd.DataFrame({"hh": [1, 1], "p": [1, 0]}, index=[1, 2]),
            control_spec=control_spec,
            total_hh_control_col="hh",
            parent_geography="P",
            parent_id=1,
            sub_geographies=["Z"],
            sub_control_zones=pd.Series(["Z_1", "Z_2"], index=[1, 2]),
            use_numba=True,
            numba_precision="float64",
        )
    assert seen["fast_paths"] is bool(flag)


def test_fast_kernel_choice_is_logged_once(monkeypatch, caplog):
    monkeypatch.setattr(balancers_numba, "_LOGGED_KERNEL_CHOICES", set())
    with caplog.at_level(logging.INFO, logger=balancers_numba.__name__):
        for _ in range(3):
            _list_balancer(True, True)
            _simul_balancer(True, True)
            _list_balancer(True, False)
    lines = [r for r in caplog.records if "FAST_PATHS" in r.getMessage()]
    assert len(lines) == 1
    assert lines[0].levelno == logging.INFO
    assert "np_balancer_numba_fast" in lines[0].getMessage()
    assert "np_simul_balancer_numba_fast" in lines[0].getMessage()


def test_fast_paths_without_numba_is_logged_once(monkeypatch, caplog):
    monkeypatch.setattr(balancers_numba, "_LOGGED_KERNEL_CHOICES", set())
    with caplog.at_level(logging.INFO, logger=balancers_numba.__name__):
        for _ in range(3):
            _list_balancer(False, True)
            _simul_balancer(False, True)
    lines = [r for r in caplog.records if "FAST_PATHS" in r.getMessage()]
    assert len(lines) == 1
    assert "USE_NUMBA" in lines[0].getMessage()


@pytest.mark.parametrize(
    "precision", [{}, {"NUMBA_PRECISION": "float32"}], ids=["float64", "float32"]
)
def test_example_run_unchanged_with_fast_paths(tmp_path, precision):
    base = {"USE_NUMBA": True, **precision}
    off = run_example(REPO_ROOT, tmp_path / "off", base)
    on = run_example(REPO_ROOT, tmp_path / "on", {**base, "FAST_PATHS": True})
    assert len(off) > 0
    pd.testing.assert_frame_equal(on, off, check_exact=True)

    def logged_fast_kernels(out):
        logs = list((tmp_path / out).rglob("*.log"))
        assert logs, f"no log file written under {out}"
        return any("np_balancer_numba_fast" in p.read_text() for p in logs)

    assert logged_fast_kernels("on"), "the fast kernels did not run"
    assert not logged_fast_kernels("off")
