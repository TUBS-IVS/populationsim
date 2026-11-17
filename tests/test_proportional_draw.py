# PopulationSim
# See full license in LICENSE.txt.
"""Seeded weight-proportional draw in both integerizers (INTEGERIZER_DRAW).

The OR-Tools LPs maximise ``sum x * ln(residual)``; with the draw on, ``ln(r)``
is replaced by a random key per household (order sampling) while every control
constraint stays in force. These tests prove (1) the key formulas and their
inclusion probabilities, (2) that the LP really selects the rows with the
largest keys, (3) that no row-position term is left in the objective, (4) that
controls still bind, and (5) that the draw is reproducible and active end to
end in the example run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populationsim.core import config
from populationsim.integerizing import bs_options, lp_ortools
from populationsim.integerizing.draw import draw_keys
from populationsim.integerizing.simul_integerizer import SimulIntegerizer
from populationsim.integerizing.single_integerizer import Integerizer
from tests.bs_differential import REPO_ROOT, run_example

METHODS = ("pareto", "gumbel")
CVXPY_MESSAGE = "INTEGERIZER_DRAW requires the OR-Tools integerizer (USE_CVXPY false)"
LOG_OVERFLOW = -725  # same floor as upstream's integerizers


@pytest.fixture
def settings(monkeypatch):
    """Inject a settings dict by patching config.setting."""
    values = {}
    monkeypatch.setattr(
        config, "setting", lambda key, default=None: values.get(key, default)
    )
    bs_options.reset_for_tests()
    yield values
    bs_options.reset_for_tests()


def _top_k(values, k):
    return set(np.argsort(values)[-k:].tolist()) if k > 0 else set()


# ---------------------------------------------------------------- builders


def _single(float_weights, totals, extra_cols=None, importance=1000.0, label="lbl"):
    """Integerizer with the household-total control (index 0) plus ``extra_cols``.

    ``totals`` are the relaxed control totals, total households first.
    """
    float_weights = np.asarray(float_weights, dtype=np.float64)
    n = len(float_weights)
    cols = {"num_hh": np.ones(n, dtype=np.int64)}
    cols.update(extra_cols or {})
    incidence = pd.DataFrame(cols, index=pd.RangeIndex(n, name="hh_id"))
    names = list(incidence.columns)
    return Integerizer(
        incidence_table=incidence,
        control_importance_weights=pd.Series([importance] * len(names), index=names),
        float_weights=pd.Series(float_weights, index=incidence.index),
        relaxed_control_totals=pd.Series(
            np.asarray(totals, dtype=np.float64), index=names
        ),
        total_hh_control_value=int(totals[0]),
        total_hh_control_index=0,
        control_is_hh_based=pd.Series([True] * len(names), index=names),
        trace_label=label,
    )


def _single_total_only(rng, n, k, label="lbl"):
    """Single integerizer, only the household-total control, k rows to round up."""
    float_weights = rng.randint(1, 4, size=n) + rng.uniform(0.02, 0.98, size=n)
    total = int(np.floor(float_weights).sum()) + k
    return float_weights, _single(float_weights, [total], label=label)


def _simul_total_only(rng, zones, n, label="lbl"):
    """SimulIntegerizer, only the household-total control per zone."""
    float_weights = rng.randint(0, 3, size=(zones, n)) + rng.uniform(
        0.02, 0.98, size=(zones, n)
    )
    zone_ids = list(range(1, zones + 1))
    totals = np.round(float_weights.sum(axis=1)).astype(np.int64)
    incidence = pd.DataFrame(
        {"num_hh": np.ones(n, dtype=np.int64)}, index=pd.RangeIndex(n, name="hh_id")
    )
    sub_weights = pd.DataFrame(float_weights.T, index=incidence.index, columns=zone_ids)
    sub_controls = pd.DataFrame({"num_hh": totals}, index=zone_ids)
    control_spec = pd.DataFrame({"target": ["num_hh"], "importance": [1000.0]})
    integerizer = SimulIntegerizer(
        incidence, sub_weights, sub_controls, control_spec, "num_hh", label
    )
    return float_weights, integerizer


def _capture_single_lp_inputs(integerizer):
    """Run the integerizer (draw off) and return the LP kwargs it built."""
    captured = {}
    real = lp_ortools.np_integerizer_ortools

    def spy(**kwargs):
        captured.update(kwargs)
        return real(**kwargs)

    integerizer.integerizer_func = spy
    integerizer.integerize()
    return captured


def _capture_simul_lp_inputs(integerizer):
    """Run the simul integerizer (draw off) and return the LP args it built."""
    captured = {}
    real = lp_ortools.np_simul_integerizer_ortools

    def spy(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return real(*args, **kwargs)

    integerizer.integerizer_func = spy
    integerizer.integerize()
    return captured


def _log_parent(sub_resid):
    parent = np.sum(sub_resid, axis=0)
    return np.log(np.maximum(parent, np.exp(LOG_OVERFLOW)))


# ---------------------------------------------------------------- draw_keys


@pytest.mark.parametrize("method", METHODS)
def test_draw_keys_reproducible(method):
    resid = np.array([0.0, 0.05, 0.3, 0.6, 0.9, 1.0])
    before = resid.copy()
    a = draw_keys(resid, method, np.random.RandomState([5, 1]))
    b = draw_keys(resid, method, np.random.RandomState([5, 1]))
    c = draw_keys(resid, method, np.random.RandomState([5, 2]))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == resid.shape
    assert np.isfinite(a).all(), "residual 0 and 1 must give finite keys"
    np.testing.assert_array_equal(resid, before)

    # The exact definition (clip to [exp(-725), 1 - 1e-12], then the key).
    r = np.clip(resid, np.exp(LOG_OVERFLOW), 1 - 1e-12)
    twin = np.random.RandomState([5, 1])
    if method == "gumbel":
        expected = np.log(r) + twin.gumbel(size=r.shape)
    else:
        expected = np.log(r) - np.log1p(-r) + twin.logistic(size=r.shape)
    np.testing.assert_array_equal(a, expected)

    two_d = np.array([[0.1, 0.0, 0.7], [1.0, 0.3, 0.5]])
    k2 = draw_keys(two_d, method, np.random.RandomState([5, 1]))
    assert k2.shape == (2, 3)
    assert np.isfinite(k2).all()
    np.testing.assert_array_equal(
        k2, draw_keys(two_d, method, np.random.RandomState([5, 1]))
    )


def test_draw_keys_rejects_bad_input():
    rng = np.random.RandomState(0)
    with pytest.raises(ValueError):
        draw_keys(np.array([0.5]), "off", rng)
    with pytest.raises(ValueError):
        draw_keys(np.array([0.5]), "Pareto", rng)
    with pytest.raises(ValueError):
        draw_keys(np.array([0.5, np.nan]), "pareto", rng)


# Order sampling of k = 128 out of 400 rows whose residuals sum to k.
RESID_CLASSES = (0.05, 0.3, 0.6, 0.9)
CLASS_SIZES = (160, 120, 80, 40)
SAMPLE_K = 128
REPLICATIONS = 2000


def _inclusion_by_class(method):
    resid = np.repeat(RESID_CLASSES, CLASS_SIZES)
    labels = np.repeat(np.arange(len(CLASS_SIZES)), CLASS_SIZES)
    assert np.isclose(resid.sum(), SAMPLE_K)
    hits = np.zeros(len(CLASS_SIZES))
    for i in range(REPLICATIONS):
        keys = draw_keys(resid, method, np.random.RandomState([1, i]))
        top = np.argsort(keys)[-SAMPLE_K:]
        hits += np.bincount(labels[top], minlength=len(CLASS_SIZES))
    return hits / (REPLICATIONS * np.asarray(CLASS_SIZES))


def test_pareto_inclusion_matches_residual():
    """Primary path: Pareto keys give inclusion probability close to r."""
    inclusion = _inclusion_by_class("pareto")
    print("pareto inclusion by class", dict(zip(RESID_CLASSES, inclusion)))
    for r, p in zip(RESID_CLASSES, inclusion):
        assert abs(p - r) < 0.01, (r, p)


def test_gumbel_inclusion_bias_is_known():
    """Gumbel keys (successive sampling) under-select large residuals.

    This known bias is why the Pareto keys exist.
    """
    inclusion = _inclusion_by_class("gumbel")
    print("gumbel inclusion by class", dict(zip(RESID_CLASSES, inclusion)))
    assert inclusion[-1] < 0.80, inclusion
    assert inclusion[0] > 0.06, inclusion


# ---------------------------------------------------------------- LP = top-k


@pytest.mark.parametrize("method", METHODS)
def test_lp_selects_top_k_keys_single(settings, method):
    k = 15
    rng = np.random.RandomState(20261117)
    float_weights, integerizer = _single_total_only(rng, 60, k, label="lp_single")
    lp = _capture_single_lp_inputs(integerizer)
    resid = lp["resid_weights"]
    assert ((resid > 0) & (resid < 1)).all()
    assert lp["lp_right_hand_side"].tolist() == [k]

    keys = draw_keys(resid, method, np.random.RandomState([11, 0]))
    out, status = lp_ortools.np_integerizer_ortools(**lp, draw_keys=keys)
    assert status == "OPTIMAL"
    np.testing.assert_allclose(out, np.round(out), atol=1e-9)
    chosen = set(np.flatnonzero(out > 0.5).tolist())
    assert chosen == _top_k(keys, k)
    # Positive control: the keys change the choice against upstream's ln(r).
    assert chosen != _top_k(resid, k)

    # The class draws its keys from call_rng("single", trace_label).
    settings["INTEGERIZER_DRAW"] = method
    settings["INTEGERIZER_RANDOM_SEED"] = 11
    _, drawn = _single_total_only(
        np.random.RandomState(20261117), 60, k, label="lp_single"
    )
    drawn.integerize()
    class_keys = draw_keys(
        float_weights % 1.0, method, bs_options.call_rng("single", "lp_single")
    )
    expected = np.floor(float_weights).astype(int)
    expected[sorted(_top_k(class_keys, k))] += 1
    np.testing.assert_array_equal(
        drawn.weights["integerized_weight"].to_numpy(), expected
    )


@pytest.mark.parametrize("method", METHODS)
def test_lp_selects_top_k_keys_simul(settings, method):
    rng = np.random.RandomState(20261118)
    float_weights, integerizer = _simul_total_only(rng, 2, 40, label="lp_simul")
    lp = _capture_simul_lp_inputs(integerizer)
    args = lp["args"]
    sub_resid = args[5]
    total_rhs = args[10]
    assert sub_resid.shape == (2, 40)
    assert ((sub_resid > 0) & (sub_resid < 1)).all()
    assert args[9].shape == (40, 0), "only the total control, no parent control"

    keys = draw_keys(sub_resid, method, np.random.RandomState([12, 0]))
    out, status = lp_ortools.np_simul_integerizer_ortools(
        *args, **lp["kwargs"], draw_keys=keys
    )
    assert status == "OPTIMAL"
    np.testing.assert_allclose(out, np.round(out), atol=1e-9)
    log_parent = _log_parent(sub_resid)
    for z in range(2):
        k = int(total_rhs[z])
        assert 0 < k < 40
        chosen = set(np.flatnonzero(out[z] > 0.5).tolist())
        assert chosen == _top_k(keys[z] + log_parent, k), z
        assert chosen != _top_k(sub_resid[z], k), z

    # The class draws its keys from call_rng("simul", trace_label).
    settings["INTEGERIZER_DRAW"] = method
    settings["INTEGERIZER_RANDOM_SEED"] = 12
    _, drawn = _simul_total_only(
        np.random.RandomState(20261118), 2, 40, label="lp_simul"
    )
    drawn.integerize()
    resid = float_weights % 1.0
    class_keys = draw_keys(resid, method, bs_options.call_rng("simul", "lp_simul"))
    expected = np.floor(float_weights).astype(int)
    for z in range(2):
        k = int(round(float_weights[z].sum())) - int(np.floor(float_weights[z]).sum())
        expected[z, sorted(_top_k(class_keys[z] + _log_parent(resid), k))] += 1
    np.testing.assert_array_equal(drawn.integerized_weights.to_numpy().T, expected)


def test_draw_off_passes_no_keys(settings):
    """Draw off: the LP call is exactly upstream's (no draw_keys argument)."""
    _, integerizer = _single_total_only(np.random.RandomState(5), 10, 3)
    assert "draw_keys" not in _capture_single_lp_inputs(integerizer)
    _, simul = _simul_total_only(np.random.RandomState(5), 2, 10)
    assert "draw_keys" not in _capture_simul_lp_inputs(simul)["kwargs"]


def test_lp_rejects_bad_draw_keys(settings):
    _, integerizer = _single_total_only(np.random.RandomState(5), 10, 3)
    lp = _capture_single_lp_inputs(integerizer)
    with pytest.raises(ValueError, match="shape"):
        lp_ortools.np_integerizer_ortools(**lp, draw_keys=np.zeros(9))
    bad = np.zeros(10)
    bad[2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        lp_ortools.np_integerizer_ortools(**lp, draw_keys=bad)

    _, simul = _simul_total_only(np.random.RandomState(5), 2, 10)
    cap = _capture_simul_lp_inputs(simul)
    with pytest.raises(ValueError, match="shape"):
        lp_ortools.np_simul_integerizer_ortools(
            *cap["args"], **cap["kwargs"], draw_keys=np.zeros((10, 2))
        )
    bad = np.zeros((2, 10))
    bad[1, 3] = -np.inf
    with pytest.raises(ValueError, match="non-finite"):
        lp_ortools.np_simul_integerizer_ortools(
            *cap["args"], **cap["kwargs"], draw_keys=bad
        )


def test_no_row_position_term_with_draw(settings):
    """Permuting rows (inputs and keys alike) permutes the LP result alike.

    Upstream adds 1e-6 * row_index to each coefficient as a tie-break; with
    the draw on that would be a row-position bias, so it must be gone.
    """
    n, k = 600, 20
    rng = np.random.RandomState(7)
    _, integerizer = _single_total_only(rng, n, k, label="perm")
    lp = _capture_single_lp_inputs(integerizer)

    # Keys built so that a 1e-6 * row_index term would flip the choice: rows
    # 0..19 hold the top keys, rows 580..599 sit just below them.
    keys = -10.0 - rng.uniform(size=n)
    keys[:20] = 1e-4 * np.arange(20)
    keys[-20:] = -1e-4 - 1e-4 * np.arange(20)
    reverse = np.arange(n)[::-1]
    shuffle = np.random.RandomState(8).permutation(n)

    # Positive control: with the upstream position term the reversal changes
    # the selected households, so this test would catch the term.
    position = 1e-6 * np.arange(n)
    with_term = _top_k(keys + position, k)
    with_term_reversed = {int(reverse[i]) for i in _top_k(keys[reverse] + position, k)}
    assert with_term != with_term_reversed

    def solve(perm):
        permuted = dict(lp)
        permuted["incidence"] = lp["incidence"][:, perm]
        permuted["resid_weights"] = lp["resid_weights"][perm]
        permuted["log_resid_weights"] = lp["log_resid_weights"][perm]
        out, status = lp_ortools.np_integerizer_ortools(
            **permuted, draw_keys=keys[perm]
        )
        assert status == "OPTIMAL"
        return out

    base = solve(np.arange(n))
    assert set(np.flatnonzero(base > 0.5).tolist()) == set(range(20))
    for perm in (reverse, shuffle):
        np.testing.assert_allclose(solve(perm), base[perm], atol=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_draw_respects_controls(settings, method):
    """A binding control (exactly 10 class-A households) holds in every draw."""
    settings["INTEGERIZER_DRAW"] = method
    is_a = np.array([1] * 20 + [0] * 20, dtype=np.int64)
    # Class A has the larger residuals: unconstrained, ~16 A rows would win.
    float_weights = np.where(is_a == 1, 0.8, 0.45)
    totals = [25, 10]
    selections = set()
    unconstrained_differs = 0
    for seed in range(50):
        settings["INTEGERIZER_RANDOM_SEED"] = seed
        integerizer = _single(
            float_weights, totals, extra_cols={"class_a": is_a}, label="ctrl"
        )
        integerizer.integerize()
        weights = integerizer.weights["integerized_weight"].to_numpy()
        assert weights.sum() == 25, seed
        assert (weights * is_a).sum() == 10, seed
        selections.add(tuple(weights.tolist()))
        keys = draw_keys(
            float_weights % 1.0, method, bs_options.call_rng("single", "ctrl")
        )
        if is_a[sorted(_top_k(keys, 25))].sum() != 10:
            unconstrained_differs += 1
    # Positive controls: the control binds, and the draw varies with the seed.
    assert unconstrained_differs >= 40, unconstrained_differs
    assert len(selections) >= 2


def test_draw_reproducible_and_label_specific(settings):
    settings["INTEGERIZER_DRAW"] = "pareto"
    settings["INTEGERIZER_RANDOM_SEED"] = 3

    def run(label, seed=3):
        settings["INTEGERIZER_RANDOM_SEED"] = seed
        _, integerizer = _single_total_only(
            np.random.RandomState(99), 60, 15, label=label
        )
        integerizer.integerize()
        return tuple(integerizer.weights["integerized_weight"].tolist())

    first = run("TAZ_1")
    assert run("TAZ_1") == first
    assert run("TAZ_1", seed=4) != first
    by_label = {run("TAZ_%s" % i) for i in range(20)}
    assert len(by_label) >= 2


def test_cvxpy_with_draw_raises(settings):
    settings["USE_CVXPY"] = True
    # Positive control: draw off + CVXPY constructs as upstream.
    _single_total_only(np.random.RandomState(1), 10, 3)
    _simul_total_only(np.random.RandomState(1), 2, 10)

    settings["INTEGERIZER_DRAW"] = "pareto"
    with pytest.raises(ValueError) as exc:
        _single_total_only(np.random.RandomState(1), 10, 3)
    assert str(exc.value) == CVXPY_MESSAGE
    with pytest.raises(ValueError) as exc:
        _simul_total_only(np.random.RandomState(1), 2, 10)
    assert str(exc.value) == CVXPY_MESSAGE


# ---------------------------------------------------------------- example


def _canonical(df):
    return df.sort_values(list(df.columns)).reset_index(drop=True)


def test_example_draw_runs(tmp_path):
    """The draw runs end to end, keeps the household total and is reproducible."""
    default = run_example(REPO_ROOT, tmp_path / "out_default")
    draw = {"INTEGERIZER_DRAW": "pareto", "INTEGERIZER_RANDOM_SEED": 7}
    first = run_example(REPO_ROOT, tmp_path / "out_draw_1", draw)
    second = run_example(REPO_ROOT, tmp_path / "out_draw_2", draw)

    log_text = (tmp_path / "out_draw_1" / "populationsim.log").read_text()
    assert "draw=pareto seed=7" in log_text
    assert len(default) > 0
    assert len(first) == len(default)
    assert not _canonical(first).equals(_canonical(default))
    pd.testing.assert_frame_equal(first, second)
