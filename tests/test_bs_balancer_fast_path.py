# PopulationSim
# See full license in LICENSE.txt.
"""The ``FAST_PATHS`` numba balancers give the weights of the originals, bit for bit.

The fast balancers (``np_balancer_numba_fast``, ``np_simul_balancer_numba_fast``)
take ``exp(log_gamma * inc)`` from a per-control lookup table of the distinct
incidence values. The balanced weights feed an integerizer whose choices can
flip on a last-bit difference, so weights, relaxation factors, the converged
flag and the iteration count are compared on the raw bytes, never within a
tolerance. Only ``delta`` (the third status value) is compared with a relative
tolerance: the fast update loop may sum ``|new_w - w_old|`` in another order,
and the guard re-runs the original kernel whenever ``delta`` comes near a
convergence threshold, so only the last bits of the reported number can differ.

Every comparison goes through the public fast functions, so the guard and the
fallbacks are part of what is tested. All kernels run compiled (no
``py_func``): the claim is about the compiled code.
"""

import logging
import struct

import numpy as np
import pandas as pd
import pytest

from populationsim.balancing import balancers_numba, simul_balancer, wrappers
from populationsim.balancing.balancers import np_balancer_py, np_simul_balancer_py
from populationsim.balancing.balancers_numba import (
    fast_path_counts,
    np_balancer_numba,
    np_balancer_numba_fast,
    np_simul_balancer_numba,
    np_simul_balancer_numba_fast,
    reset_fast_path_counts,
)
from populationsim.balancing.single_balancer import ListBalancer
from populationsim.balancing.simul_balancer import SimultaneousListBalancer
from populationsim.core import config
from tests.bs_differential import REPO_ROOT, run_example
from tests.settings_isolation import patch_settings

# Row counts that are and are not multiples of the SIMD widths (2, 4, 8).
EDGE_SIZES = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 63, 65, 127, 129, 5000]
# Non-integer incidence values mixed into the float problems (1/3 is not exact
# in float32 or float64, so the conversion to float64 matters).
NON_INTEGER_VALUES = np.array([0.1, 0.5, 1.5, 2.25, 1.0 / 3.0])
FAST_OUTCOMES = {"fast", "rerun_delta_band"}


@pytest.fixture(autouse=True)
def _clean_counts():
    reset_fast_path_counts()
    yield
    reset_fast_path_counts()


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


def _assert_same_result(orig, fast):
    w_orig, rf_orig, st_orig = orig
    w_fast, rf_fast, st_fast = fast
    assert _bytes_equal(w_orig, w_fast), "weights differ"
    assert _bytes_equal(rf_orig, rf_fast), "relaxation factors differ"
    assert bool(st_orig[0]) == bool(st_fast[0]), "converged flag differs"
    assert int(st_orig[1]) == int(st_fast[1]), "iteration count differs"
    # delta only decides convergence; the fast update loop may sum it in
    # another order (relative difference about n * 2.2e-16), and the guard
    # re-runs the original kernel whenever it is within 1e-6 relative of a
    # threshold. So the decisions above are identical, the number may differ
    # in its last bits.
    assert float(st_fast[2]) == pytest.approx(float(st_orig[2]), rel=1e-9, abs=0.0)
    # max_gamma_dif is derived from the identical gammas
    assert struct.pack("<d", float(st_orig[3])) == struct.pack("<d", float(st_fast[3]))


def _only_outcome(kind):
    counts = fast_path_counts()[kind]
    used = {outcome: n for outcome, n in counts.items() if n}
    assert sum(used.values()) == 1, counts
    return next(iter(used))


def _incidence(rng, control_count, sample_count, densities, non_integer=False):
    """Integer counts 0..6, one density per control row (0 = all-zero row).

    With ``non_integer``, about 3 % of the non-zero cells get a value from
    NON_INTEGER_VALUES (float64 result).
    """
    rows = []
    for density in densities:
        present = rng.random(sample_count) < density
        rows.append(present * rng.integers(1, 7, size=sample_count))
    incidence = np.array(rows, dtype=np.int64).reshape(control_count, sample_count)
    if not non_integer:
        return incidence
    incidence = incidence.astype(np.float64)
    odd = (incidence > 0) & (rng.random(incidence.shape) < 0.03)
    incidence[odd] = rng.choice(NON_INTEGER_VALUES, size=int(odd.sum()))
    return incidence


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
    variant = seed % 6
    incidence = _incidence(
        rng,
        control_count,
        sample_count,
        _densities(rng, control_count),
        non_integer=variant >= 2,
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
    fortran = variant % 2 == 1
    inc = np.ascontiguousarray(incidence.T).T if fortran else incidence
    if variant in (4, 5):
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
        rng,
        control_count,
        sample_count,
        _densities(rng, control_count),
        non_integer=seed % 3 == 0,
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


SINGLE_CASES = [(seed, None) for seed in range(66)] + [
    (1000 + i, n) for i, n in enumerate(EDGE_SIZES)
]
SIMUL_CASES = [(seed, None) for seed in range(62)] + [
    (2000 + i, n) for i, n in enumerate(EDGE_SIZES)
]


@pytest.mark.parametrize("seed, sample_count", SINGLE_CASES)
def test_single_fast_balancer_gives_identical_weights(seed, sample_count):
    args = _single_problem(seed, sample_count)
    orig = np_balancer_numba(*_copy_args(args))
    fast = np_balancer_numba_fast(*_copy_args(args))
    _assert_same_result(orig, fast)
    assert _only_outcome("single") in FAST_OUTCOMES


@pytest.mark.parametrize("seed, sample_count", SIMUL_CASES)
def test_simul_fast_balancer_gives_identical_weights(seed, sample_count):
    args = _simul_problem(seed, sample_count)
    orig_args = _copy_args(args)
    fast_args = _copy_args(args)
    orig = np_simul_balancer_numba(*orig_args)
    fast = np_simul_balancer_numba_fast(*fast_args)
    _assert_same_result(orig, fast)
    assert _only_outcome("simul") in FAST_OUTCOMES
    # the original balances sub_weights and scales sub_controls in place
    assert fast[0] is fast_args[8]
    assert _bytes_equal(orig_args[8], fast_args[8])
    assert _bytes_equal(orig_args[11], fast_args[11])


def test_random_problems_exercise_the_edge_cases():
    """The generators really produce what the bitwise tests claim to cover."""
    single = [_single_problem(seed) for seed, _ in SINGLE_CASES[:66]]
    simul = [_simul_problem(seed) for seed, _ in SIMUL_CASES[:62]]
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
    assert {p[1] for p in single} >= {1} and max(p[1] for p in single) >= 28
    assert min(p[2] for p in simul) == 1 and max(p[2] for p in simul) >= 11
    incidences = [np.asarray(p[3]) for p in single] + [p[4] for p in simul]
    assert any((inc == 0).all(axis=1).any() for inc in incidences)
    assert any((inc != 0).all(axis=1).any() for inc in incidences)
    assert any((inc == 6).any() for inc in incidences)
    assert any((inc != np.round(inc)).any() for inc in incidences)
    assert any((p[4] != np.round(p[4])).any() for p in simul)
    weights = np.concatenate([p[4].astype(np.float64) for p in single])
    assert (np.signbit(weights) & (weights == 0)).any()
    assert any((p[4] > p[6]).any() and (p[4] == p[5]).any() for p in single)
    assert any((p[4] < p[5]).any() for p in single)
    importances = np.concatenate([p[8].astype(np.float64) for p in single])
    assert importances.min() < 10 and importances.max() > 1e8


def test_the_lookup_kernels_are_the_primary_path():
    """Almost every random problem uses the fast result, not a re-run."""
    for seed in range(20):
        np_balancer_numba_fast(*_single_problem(seed + 500))
        np_simul_balancer_numba_fast(*_simul_problem(seed + 500))
    counts = fast_path_counts()
    for kind in ("single", "simul"):
        assert counts[kind]["fast"] >= 18, counts
        assert sum(counts[kind].values()) == 20, counts


def test_encoding_holds_the_exact_float64_values():
    inc = np.array(
        [[0, 0.1, 2, 0.1, 0, 1.0 / 3.0], [1, 1, 1, 1, 1, 1]], dtype=np.float32
    )
    values, n_values, codes, status = balancers_numba._encode_incidence(inc, 256)
    assert status == balancers_numba._ENCODE_OK
    assert codes.dtype == np.uint8
    assert list(n_values) == [4, 1]
    expected = sorted(float(x) for x in set(inc[0].tolist()))
    assert values[0, :4].tolist() == expected
    assert values[0, :4].tobytes() == np.array(expected).tobytes()
    assert values[1, 0] == 1.0
    for c in range(2):
        assert (values[c, codes[c]] == inc[c].astype(np.float64)).all()
    assert (codes[1] == 0).all()


def _problem_with_distinct_values(distinct, seed=5, sample_count=600):
    rng = np.random.default_rng(seed)
    inc = np.ones((3, sample_count))
    inc[1] = (np.arange(sample_count) % distinct) * 0.01
    inc[2] = rng.integers(0, 3, sample_count)
    w = np.exp(rng.uniform(0.0, 2.0, sample_count))
    lb = np.zeros(sample_count)
    ub = np.full(sample_count, 1e6)
    constraint = np.maximum(
        (inc * rng.uniform(0.5, 2.0, sample_count)).sum(axis=1), 0.1
    )
    importance = np.array([1e9, 10.0, 10.0])
    return (sample_count, 3, 0, inc, w, lb, ub, constraint, importance, 100)


def test_value_cap_runs_the_original_kernel_and_is_counted(caplog, fork_logging):
    args = _problem_with_distinct_values(257)
    with caplog.at_level(logging.WARNING, logger=balancers_numba.__name__):
        fast = np_balancer_numba_fast(*_copy_args(args))
    orig = np_balancer_numba(*_copy_args(args))
    assert _status_key(orig[2]) == _status_key(fast[2])  # the original ran
    _assert_same_result(orig, fast)
    assert _only_outcome("single") == "fallback_value_cap"
    assert any("distinct incidence values" in r.getMessage() for r in caplog.records)


def test_value_cap_boundary_uses_the_lookup_kernel():
    args = _problem_with_distinct_values(256)
    fast = np_balancer_numba_fast(*_copy_args(args))
    _assert_same_result(np_balancer_numba(*_copy_args(args)), fast)
    assert _only_outcome("single") in FAST_OUTCOMES


def _simul_problem_with_many_values(distinct):
    for seed in range(100, 200):
        args = list(_simul_problem(seed, 600))
        if args[1] >= 2:
            break
    control = (args[3] + 1) % args[1]
    inc = args[4].copy()
    inc[control] = (np.arange(600) % distinct) * 0.01
    args[4] = inc
    return tuple(args)


def test_simul_value_cap_runs_the_original_kernel_and_is_counted():
    args = _simul_problem_with_many_values(300)
    orig_args = _copy_args(args)
    fast_args = _copy_args(args)
    orig = np_simul_balancer_numba(*orig_args)
    fast = np_simul_balancer_numba_fast(*fast_args)
    assert _status_key(orig[2]) == _status_key(fast[2])
    _assert_same_result(orig, fast)
    assert _bytes_equal(orig_args[8], fast_args[8])
    assert _bytes_equal(orig_args[11], fast_args[11])
    assert _only_outcome("simul") == "fallback_value_cap"


def test_nonfinite_incidence_runs_the_original_kernel():
    args = list(_problem_with_distinct_values(3))
    inc = args[3].copy()
    inc[2, 7] = np.inf
    args[3] = inc
    fast = np_balancer_numba_fast(*_copy_args(args))
    orig = np_balancer_numba(*_copy_args(args))
    assert _bytes_equal(orig[0], fast[0])
    assert _only_outcome("single") == "fallback_nonfinite"


def test_svml_runs_the_original_kernel(monkeypatch):
    monkeypatch.setattr(balancers_numba, "_svml_active", lambda: True)
    args = _single_problem(3, 300)
    fast = np_balancer_numba_fast(*_copy_args(args))
    orig = np_balancer_numba(*_copy_args(args))
    assert _status_key(orig[2]) == _status_key(fast[2])
    _assert_same_result(orig, fast)
    assert _only_outcome("single") == "fallback_svml"


@pytest.mark.parametrize("seed", [2, 3, 4, 7])
def test_delta_band_reruns_the_original_kernel(monkeypatch, caplog, fork_logging, seed):
    """A band wide enough to hold every delta forces the re-run on every call."""
    monkeypatch.setattr(balancers_numba, "FAST_PATH_DELTA_BAND", 1e300)
    args = _single_problem(seed, 700)
    with caplog.at_level(logging.WARNING, logger=balancers_numba.__name__):
        fast = np_balancer_numba_fast(*_copy_args(args))
    orig = np_balancer_numba(*_copy_args(args))
    assert _status_key(orig[2]) == _status_key(fast[2])  # the original's delta
    _assert_same_result(orig, fast)
    assert _only_outcome("single") == "rerun_delta_band"
    assert any("re-run" in r.getMessage() for r in caplog.records)

    sargs = _simul_problem(seed, 700)
    orig_args = _copy_args(sargs)
    fast_args = _copy_args(sargs)
    orig = np_simul_balancer_numba(*orig_args)
    fast = np_simul_balancer_numba_fast(*fast_args)
    assert _status_key(orig[2]) == _status_key(fast[2])
    _assert_same_result(orig, fast)
    # the re-run started from the untouched inputs and mutated them like the
    # original does
    assert _bytes_equal(orig_args[8], fast_args[8])
    assert _bytes_equal(orig_args[11], fast_args[11])
    assert _only_outcome("simul") == "rerun_delta_band"


def test_lookup_kernel_reports_a_delta_inside_the_band():
    """The kernel flag itself: off with a zero band, on with a huge one."""
    args = _single_problem(2, 300)
    values, n_values, codes, _ = balancers_numba._encode_incidence(args[3], 256)
    for band, expected in ((0.0, False), (1e300, True)):
        result = balancers_numba._np_balancer_lookup(
            *_copy_args(args), values, n_values, codes, band
        )
        assert result[3] is expected


def test_fallback_warnings_are_rate_limited(caplog, fork_logging):
    args = _problem_with_distinct_values(300, sample_count=400)
    with caplog.at_level(logging.WARNING, logger=balancers_numba.__name__):
        for _ in range(7):
            np_balancer_numba_fast(*_copy_args(args))
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 5
    assert "fast_path_counts()" in warnings[-1].getMessage()
    assert fast_path_counts()["single"]["fallback_value_cap"] == 7


def test_first_call_of_each_balancer_is_logged_once(caplog, fork_logging):
    with caplog.at_level(logging.INFO, logger=balancers_numba.__name__):
        for _ in range(3):
            np_balancer_numba_fast(*_single_problem(1, 50))
            np_simul_balancer_numba_fast(*_simul_problem(1, 50))
    single = [r for r in caplog.records if "single balancer: kernel" in r.getMessage()]
    simul = [r for r in caplog.records if "simul balancer: kernel" in r.getMessage()]
    assert len(single) == 1 and "np_balancer_numba_fast" in single[0].getMessage()
    assert len(simul) == 1 and "np_simul_balancer_numba_fast" in simul[0].getMessage()


@pytest.fixture
def settings(monkeypatch):
    """Inject a settings dict by patching config.setting and every module that
    imported it by name (see tests/settings_isolation.py); restored at teardown."""
    return patch_settings(monkeypatch)


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


def test_balancer_classes_give_identical_weights_with_fast_paths(settings, monkeypatch):
    monkeypatch.setattr(simul_balancer, "setting", config.setting)
    status_off, weights_off, controls_off = _list_balancer(True, False).balance()
    status_on, weights_on, controls_on = _list_balancer(True, True).balance()
    assert _bytes_equal(weights_off["final"].values, weights_on["final"].values)
    pd.testing.assert_frame_equal(controls_off, controls_on, check_exact=True)
    assert status_off["iter"] == status_on["iter"]

    off = _simul_balancer(True, False)
    on = _simul_balancer(True, True)
    status_off = off.balance()
    status_on = on.balance()
    pd.testing.assert_frame_equal(
        off.sub_zone_weights, on.sub_zone_weights, check_exact=True
    )
    assert status_off["iter"] == status_on["iter"]
    assert fast_path_counts()["single"]["fast"] == 1
    assert fast_path_counts()["simul"]["fast"] == 1


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


def test_fast_kernel_choice_is_logged_once(monkeypatch, caplog, fork_logging):
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


def test_fast_paths_without_numba_is_logged_once(monkeypatch, caplog, fork_logging):
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

    def log_text(out):
        logs = list((tmp_path / out).rglob("*.log"))
        assert logs, f"no log file written under {out}"
        return "\n".join(p.read_text() for p in logs)

    on_log = log_text("on")
    # the lookup kernels really ran (first-call lines of both balancers)
    assert "single balancer: kernel np_balancer_numba_fast" in on_log
    assert "simul balancer: kernel np_simul_balancer_numba_fast" in on_log
    assert "np_balancer_numba_fast" not in log_text("off")


# ---------------------------------------------------------------------------
# Rate summary at exit, delta guard on a real threshold, value cap check (#656)
# ---------------------------------------------------------------------------


def _rate_records(caplog):
    return [r for r in caplog.records if "balancer fast path:" in r.getMessage()]


def test_rate_line_is_a_warning_with_the_exact_numbers_after_a_fallback(
    caplog, fork_logging
):
    np_balancer_numba_fast(*_copy_args(_single_problem(1, 50)))
    np_balancer_numba_fast(*_copy_args(_problem_with_distinct_values(257)))
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=balancers_numba.__name__):
        balancers_numba.log_fast_path_rate()
    records = _rate_records(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].getMessage() == (
        "balancer fast path: fast 1 of 2 calls (50.0 %), fallback value cap 1, "
        "fallback SVML 0, fallback non-finite 0, re-run after the delta guard 0"
    )


def test_rate_line_is_info_when_every_call_was_fast(caplog, fork_logging):
    np_balancer_numba_fast(*_copy_args(_single_problem(1, 50)))
    np_simul_balancer_numba_fast(*_copy_args(_simul_problem(1, 50)))
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=balancers_numba.__name__):
        balancers_numba.log_fast_path_rate()
    records = _rate_records(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert "fast 2 of 2 calls (100.0 %)" in records[0].getMessage()


def test_rate_line_counts_a_delta_guard_rerun_as_a_warning(
    monkeypatch, caplog, fork_logging
):
    monkeypatch.setattr(balancers_numba, "FAST_PATH_DELTA_BAND", 1e300)
    np_balancer_numba_fast(*_copy_args(_single_problem(2, 300)))
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=balancers_numba.__name__):
        balancers_numba.log_fast_path_rate()
    (record,) = _rate_records(caplog)
    assert record.levelno == logging.WARNING
    assert "fast 0 of 1 calls (0.0 %)" in record.getMessage()
    assert "re-run after the delta guard 1" in record.getMessage()


def test_rate_line_goes_to_a_given_logger(caplog, fork_logging):
    np_balancer_numba_fast(*_copy_args(_single_problem(1, 50)))
    caplog.clear()
    other = logging.getLogger("populationsim.test_rate_target")
    with caplog.at_level(logging.INFO, logger=other.name):
        balancers_numba.log_fast_path_rate(other)
    (record,) = _rate_records(caplog)
    assert record.name == other.name


def test_rate_line_is_skipped_when_no_call_was_counted(caplog, fork_logging):
    """All counts zero (nothing ran, or the counts were reset): no line at all."""
    assert all(
        n == 0 for counts in fast_path_counts().values() for n in counts.values()
    )
    with caplog.at_level(logging.DEBUG, logger=balancers_numba.__name__):
        balancers_numba.log_fast_path_rate()
    assert _rate_records(caplog) == []


def test_rate_line_at_exit_survives_a_closed_stream(capsys, monkeypatch):
    """At interpreter exit the handler's stream may be closed already; logging
    would then print ``--- Logging error ---`` on stderr. The summary must be
    silent there and must restore ``logging.raiseExceptions``."""
    import io

    np_balancer_numba_fast(*_copy_args(_single_problem(1, 50)))
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    stream.close()
    other = logging.getLogger("populationsim.test_closed_stream_target")
    other.addHandler(handler)
    other.propagate = False
    other.setLevel(logging.INFO)
    monkeypatch.setattr(logging, "raiseExceptions", True)
    capsys.readouterr()
    try:
        balancers_numba.log_fast_path_rate(other)
    finally:
        other.removeHandler(handler)
    assert "Logging error" not in capsys.readouterr().err
    assert logging.raiseExceptions is True


def test_rate_summary_is_registered_with_atexit_once(monkeypatch):
    registered = []
    monkeypatch.setattr(
        balancers_numba.atexit, "register", lambda *a, **k: registered.append(a)
    )
    monkeypatch.setitem(balancers_numba._FAST_PATH_STATE, "atexit_registered", False)
    for _ in range(3):
        np_balancer_numba_fast(*_copy_args(_single_problem(1, 50)))
        np_simul_balancer_numba_fast(*_copy_args(_simul_problem(1, 50)))
    assert registered == [(balancers_numba.log_fast_path_rate,)]


def _original_deltas(args, last):
    """delta of iterations 1..last of the original kernel, via max_iterations=k."""
    deltas = []
    for k in range(1, last + 1):
        run_args = list(_copy_args(args))
        run_args[-1] = k
        status = np_balancer_numba(*run_args)[2]
        assert not status[0], "problem converged inside the probed iterations"
        deltas.append(float(status[2]))
    return deltas


def test_delta_guard_fires_on_one_real_near_threshold_iteration(monkeypatch):
    """The guard fires when exactly one iteration's delta is inside the band."""
    args = list(_single_problem(2, 700))
    total_iterations = 12
    args[-1] = total_iterations
    deltas = _original_deltas(args, total_iterations)
    band = balancers_numba.FAST_PATH_DELTA_BAND
    assert band == 1.0e-6  # the shipped relative width
    chosen = 5
    # a centre 5e-7 (relative) above the chosen delta: inside the band for it
    center = deltas[chosen] * (1.0 + 0.5 * band)
    inside = [i for i, d in enumerate(deltas) if abs(d - center) <= band * center]
    assert inside == [chosen], "the probe problem must have one delta in the band"
    monkeypatch.setattr(balancers_numba, "FAST_PATH_BAND_CENTER_1", center)
    monkeypatch.setattr(balancers_numba, "FAST_PATH_BAND_CENTER_2", 1e300)
    orig = np_balancer_numba(*_copy_args(args))
    fast = np_balancer_numba_fast(*_copy_args(args))
    assert fast_path_counts()["single"]["rerun_delta_band"] == 1
    assert fast_path_counts()["single"]["fast"] == 0
    # the original's result, bit for bit including delta
    assert _bytes_equal(orig[0], fast[0])
    assert _bytes_equal(orig[1], fast[1])
    assert _status_key(orig[2]) == _status_key(fast[2])


def test_lookup_delta_agrees_with_the_original_far_inside_the_band():
    """Relative delta difference below 1e-9 on 50+ seeded problems (band 1e-6)."""
    worst = 0.0
    compared = 0
    for seed in range(60):
        args = _single_problem(seed)
        orig = np_balancer_numba(*_copy_args(args))
        values, n_values, codes, status = balancers_numba._encode_incidence(
            args[3], 256
        )
        if status != 0:
            continue
        # a zero band never flags, so the lookup kernel runs to its own end
        fast = balancers_numba._np_balancer_lookup(
            *_copy_args(args), values, n_values, codes, 0.0
        )
        d_orig = float(orig[2][2])
        d_fast = float(fast[2][2])
        if d_orig > 0.0:
            worst = max(worst, abs(d_fast - d_orig) / d_orig)
        compared += 1
    assert compared >= 50
    assert worst < 1e-9


def test_value_cap_above_the_code_dtype_raises_value_error(monkeypatch):
    monkeypatch.setattr(balancers_numba, "FAST_PATH_MAX_VALUES", 257)
    with pytest.raises(ValueError, match="FAST_PATH_MAX_VALUES"):
        np_balancer_numba_fast(*_copy_args(_single_problem(1, 50)))
    with pytest.raises(ValueError, match="FAST_PATH_MAX_VALUES"):
        np_simul_balancer_numba_fast(*_copy_args(_simul_problem(1, 50)))
