# PopulationSim
# See full license in LICENSE.txt.
"""Controlled rounding: targets and costs (pure numpy) and the controlled LP."""

import dataclasses
import logging

import numpy as np
import pytest
import scipy.sparse as sp

from populationsim.integerizing import controlled_lp
from populationsim.integerizing import controlled_rounding as cr
from populationsim.integerizing import lp_ortools
from populationsim.integerizing.bs_options import PenaltySettings
from populationsim.integerizing.constants import STATUS_OPTIMAL, STATUS_SUCCESS
from populationsim.integerizing.smart_round import smart_round
from tests.bs_controlled_reference import solve_dense_reference

PENALTIES = PenaltySettings(
    floor=100.0, cap=100000.0, parent_total_factor=10.0, conserve_penalty=100000.0
)


def _inputs(
    weights,
    incidence,
    household_controls,
    importance=None,
    parent_incidence=None,
    parent_importance=None,
    keys=None,
    total_index=0,
):
    """Keyword arguments of build_problem; weights are (zones, rows)."""
    weights = np.asarray(weights, dtype=np.float64)
    incidence = np.asarray(incidence, dtype=np.float64)
    zones, rows = weights.shape
    if importance is None:
        importance = np.full(incidence.shape[1], 1000.0)
    if parent_incidence is None:
        parent_incidence = np.zeros((rows, 0))
        parent_importance = np.zeros(0)
    if keys is None:
        keys = np.zeros((zones, rows))
    return dict(
        sub_float_weights=weights,
        sub_incidence=incidence,
        parent_incidence=np.asarray(parent_incidence, dtype=np.float64),
        sub_importance=np.asarray(importance, dtype=np.float64),
        parent_importance=np.asarray(parent_importance, dtype=np.float64),
        household_controls=np.asarray(household_controls),
        total_hh_sub_control_index=total_index,
        keys=np.asarray(keys, dtype=np.float64),
        penalties=PENALTIES,
    )


def _build(*args, **kwargs):
    return cr.build_problem(**_inputs(*args, **kwargs))


def test_targets_band_and_parent_totals():
    # column 0 = household total, column 1 = a control (e.g. persons) with
    # incidence 2 in row 0 and 0 in row 1: F = 2 * [0.3, 0.4, 0.3]. The
    # incidence is asymmetric, so a transposed incidence (column 0 = w0 + 2 w1)
    # changes every total. Row 1 carries only the household total and has
    # integer parts.
    weights = [[0.3, 2.0], [0.4, 1.25], [0.3, 0.75]]
    incidence = [[1, 2], [1, 0]]
    problem = _build(
        weights,
        incidence,
        household_controls=[2, 2, 1],
        parent_incidence=[[1], [0]],
        parent_importance=[500.0],
    )

    np.testing.assert_array_equal(problem.band_lo[:, 1], [0, 0, 0])
    np.testing.assert_array_equal(problem.band_hi[:, 1], [1, 1, 1])
    np.testing.assert_allclose(problem.frac[:, 1], [0.6, 0.8, 0.6], atol=1e-12)
    assert problem.parent_sub_target[1] == 2  # round(2.0) - 0
    assert problem.parent_target.tolist() == [1]  # round(1.0) - 0
    # household total: F = [2.3, 1.65, 1.05], integer part A = [2, 1, 0]
    np.testing.assert_array_equal(problem.band_lo[:, 0], [0, 0, 1])
    np.testing.assert_array_equal(problem.band_hi[:, 0], [1, 1, 2])
    assert problem.parent_sub_target[0] == 2  # round(5.0) - 3
    np.testing.assert_array_equal(
        problem.x_allowed, [[True, False], [True, True], [True, True]]
    )
    assert problem.conserve_target.tolist() == [1, 1]


def test_targets_snap_float_noise():
    # columns: household total, c1 (rows 0 and 1), c2 (row 2)
    weights = [[2.9999999997, 0.0, 0.4999999], [0.0, 3.0000000002, 0.5000002]]
    incidence = [[1, 1, 0], [1, 1, 0], [1, 0, 1]]
    problem = _build(weights, incidence, household_controls=[3, 4])

    # F[:, c1] = [2.9999999997, 3.0000000002]: one-point band at 3, no fraction
    np.testing.assert_array_equal(problem.frac[:, 1], [0.0, 0.0])
    lo = problem.band_lo[:, 1] + np.array([2, 3])  # back to absolute units
    hi = problem.band_hi[:, 1] + np.array([2, 3])
    np.testing.assert_array_equal(lo, [3, 3])
    np.testing.assert_array_equal(hi, [3, 3])
    # residuals [0.4999999, 0.5000002] of row 2: integer conservation target
    assert problem.conserve_target[2] == 1
    assert problem.conserve_target[0] == 1  # residual 0.9999999997
    assert problem.conserve_target[1] == 0  # residual 2e-10
    assert problem.parent_sub_target[1] == 1  # round(6.0) - (2 + 3)
    assert problem.parent_sub_target[2] == 1  # round(1.0000001) - 0
    np.testing.assert_allclose(problem.frac[:, 2], [0.4999999, 0.5000002], atol=1e-12)


def test_household_count_is_what_smart_round_enforces():
    # three zones with the same four rows: integer part 3, every row a residual
    row_weights = [1.5, 1.25, 1.5, 0.75]
    weights = [row_weights] * 3
    controls = [5, 2, 9]
    problem = _build(weights, [[1]] * 4, household_controls=controls)

    # Integer parts [1, 1, 1, 0] sum to 3; H = 4 rows, all with a residual.
    # need = N - 3 = [2, -1, 6]:
    # - hh_count = clip(need, 0, n_avail = 4) = [2, 0, 4];
    # - hh_excess = max(-need, 0) summed = 1 (zone 1 already holds 3 > 2);
    # - smart_round adds min(max(need, 0), H) = [2, 0, 4] rows, all of them
    #   rows with a residual, so hh_outside_lp = [2, 0, 4] - hh_count = 0;
    # - zone 2 asks for 6 but smart_round adds at most H = 4 rows:
    #   hh_unreachable = max(need - H, 0) summed = 2.
    np.testing.assert_array_equal(problem.hh_count, [2, 0, 4])
    assert problem.hh_count.dtype.kind == "i"
    assert problem.hh_excess == 1
    assert problem.hh_outside_lp == 0
    assert problem.hh_unreachable == 2

    # the count is the number of rows smart_round rounds up
    int_weights = np.array(row_weights).astype(int)
    resid = np.array(row_weights) - int_weights
    for z, control in enumerate(controls):
        added = smart_round(int_weights, resid, control).sum() - int_weights.sum()
        assert added == problem.hh_count[z]


def test_household_count_counts_rows_not_household_total_incidence(
    caplog, fork_logging
):
    # row 0 has integer part 2 but household-total incidence 0: smart_round
    # counts rows, so it needs 3 - 2 = 1 more household, not 3 - 0 = 3 (clipped
    # to the 2 rows with a residual)
    weights = [[2.0, 0.5, 0.5]]
    incidence = [[0], [1], [1]]
    with caplog.at_level("WARNING", logger=cr.__name__):
        problem = _build(weights, incidence, household_controls=[3])

    np.testing.assert_array_equal(problem.hh_count, [1])
    assert problem.hh_excess == 0
    assert problem.hh_outside_lp == 0
    assert problem.hh_unreachable == 0
    assert "household-total incidence is not 1" in caplog.text
    int_weights = np.array(weights[0]).astype(int)
    resid = np.array(weights[0]) - int_weights
    added = smart_round(int_weights, resid, 3).sum() - int_weights.sum()
    assert added == problem.hh_count[0]


def test_household_count_split_outside_the_lp_and_unreachable():
    # integer parts [1, 0, 0], H = 3 rows, one with a residual; N = 10.
    # need = 10 - 1 = 9; the LP rounds up hh_count = clip(9, 0, 1) = 1 row;
    # smart_round's shortfall is clipped at H: it adds min(9, 3) = 3 rows, the
    # 2 beyond the LP's among the rows without a residual (hh_outside_lp = 2);
    # the remaining 9 - 3 = 6 are never added (hh_unreachable = 6).
    weights = [[1.5, 0.0, 0.0]]
    problem = _build(weights, [[1]] * 3, household_controls=[10])

    np.testing.assert_array_equal(problem.hh_count, [1])
    assert problem.hh_outside_lp == 2
    assert problem.hh_unreachable == 6
    assert problem.hh_excess == 0
    int_weights = np.array(weights[0]).astype(int)
    resid = np.array(weights[0]) - int_weights
    rounded = smart_round(int_weights, resid, 10)
    added = rounded.sum() - int_weights.sum()
    assert added == problem.hh_count[0] + problem.hh_outside_lp
    assert 10 - rounded.sum() == problem.hh_unreachable


def test_x_coefficients_match_a_brute_force_fold():
    # 3 zones x 5 rows x 4 controls (household total in column 2), incidence
    # in {0, 1, 2} differing per row and control, non-zero keys and a
    # different fraction in every cell
    rng = np.random.default_rng(646)
    zones, rows, controls, t = 3, 5, 4, 2
    incidence = np.array(
        [[2, 0, 1, 1], [0, 1, 1, 2], [1, 2, 1, 0], [2, 1, 1, 1], [0, 2, 1, 2]],
        dtype=np.float64,
    )
    weights = rng.uniform(0.05, 2.95, size=(zones, rows))
    keys = rng.uniform(-25.0, 25.0, size=(zones, rows))
    keys[0, 0] = 80.0  # clipped to KEY_CLIP
    importance = np.array([150.0, 50.0, 1e9, 3000.0])  # 50 -> floor 100
    problem = _build(
        weights,
        incidence,
        household_controls=[4] * zones,
        importance=importance,
        keys=keys,
        total_index=t,
    )

    # brute force: float cell totals, fractions and the folded cell slope
    float_totals = np.zeros((zones, controls))
    for z in range(zones):
        for c in range(controls):
            for h in range(rows):
                float_totals[z, c] += weights[z, h] * incidence[h, c]
    distance = np.abs(float_totals - np.round(float_totals))
    assert distance.min() > 1e-3  # no snapping, every cell has a fraction
    frac = float_totals - np.floor(float_totals)
    assert len(np.unique(np.round(frac, 9))) == frac.size
    q = np.clip(importance, PENALTIES.floor, PENALTIES.cap)
    expected = np.zeros((zones, rows))
    for z in range(zones):
        for h in range(rows):
            value = float(np.clip(keys[z, h], -cr.KEY_CLIP, cr.KEY_CLIP))
            for c in range(controls):
                if c != t:
                    value -= q[c] * (1.0 - 2.0 * frac[z, c]) * incidence[h, c]
            expected[z, h] = value

    np.testing.assert_allclose(problem.frac, frac, atol=1e-12)
    np.testing.assert_allclose(problem.x_coef, expected, rtol=0, atol=1e-9)


def test_penalty_tiers():
    importance = [0.0, 200.0, 20000.0, 1e9]
    incidence = [[1, 0, 1, 1], [0, 1, 1, 1]]
    problem = _build(
        [[0.5, 0.5]],
        incidence,
        household_controls=[1],
        importance=importance,
        parent_incidence=incidence,
        parent_importance=importance,
        total_index=3,
    )

    np.testing.assert_array_equal(problem.cell_penalty, [100, 200, 20000, 100000])
    np.testing.assert_array_equal(
        problem.parent_sub_penalty, [1000, 2000, 100000, 100000]
    )
    np.testing.assert_array_equal(problem.parent_penalty, [1000, 2000, 100000, 100000])
    assert problem.conserve_penalty == 100000.0


def test_x_coefficients_reproduce_cell_cost():
    # one row per zone with integer part 2 and f in {0, 0.09, 0.5, 0.91};
    # column 0 is the household total (must not enter x_coef), column 1 a control
    q = 500.0
    float_totals = [2.0, 2.09, 2.5, 2.91]
    integer_part = 2
    problem = _build(
        [[f] for f in float_totals],
        [[1, 1]],
        household_controls=[3, 3, 3, 3],
        importance=[1e9, q],
    )

    for z, float_total in enumerate(float_totals):
        f = problem.frac[z, 1]
        assert f == pytest.approx(float_total - integer_part, abs=1e-12)
        # keys are zero, so x_coef is the folded cell term of the control only
        assert problem.x_coef[z, 0] == pytest.approx(-q * (1 - 2 * f), abs=1e-9)
        assert problem.under_price[z, 1] == pytest.approx(2 * q * (1 - f), abs=1e-9)
        assert problem.over_price[z, 1] == pytest.approx(2 * q * f, abs=1e-9)
        lo = int(problem.band_lo[z, 1]) + integer_part
        hi = int(problem.band_hi[z, 1]) + integer_part
        for y in range(lo - 1, hi + 2):
            under = max(lo - y, 0)
            over = max(y - hi, 0)
            cost = (
                -problem.x_coef[z, 0] * (y - lo)
                + problem.under_price[z, 1] * under
                + problem.over_price[z, 1] * over
            )
            expected = cr.cell_cost(y, float_total, q) - q * f
            assert cost == pytest.approx(expected, abs=1e-9), (float_total, y)


def test_keys_are_clipped():
    problem = _build(
        [[0.5, 0.5]], [[1], [1]], household_controls=[1], keys=[[-725.0, 80.0]]
    )
    np.testing.assert_array_equal(problem.x_coef, [[-30.0, 30.0]])
    assert cr.KEY_CLIP == 30.0


def _set(name, value):
    def change(kwargs):
        kwargs[name] = np.asarray(value, dtype=np.float64)

    return change


def _set_entry(name, index, value):
    def change(kwargs):
        array = np.array(kwargs[name], dtype=np.float64)
        array[index] = value
        kwargs[name] = array

    return change


def _set_index(value):
    def change(kwargs):
        kwargs["total_hh_sub_control_index"] = value

    return change


@pytest.mark.parametrize(
    "change, name",
    [
        (_set("sub_incidence", [[1, 1]] * 3), "sub_incidence"),
        (_set("parent_incidence", [[1]] * 3), "parent_incidence"),
        (_set("sub_importance", [1.0]), "sub_importance"),
        (_set("parent_importance", [1.0, 2.0]), "parent_importance"),
        (_set("household_controls", [1.0]), "household_controls"),
        (_set("household_controls", [1.0, 2.5]), "household_controls"),
        (_set("keys", [[0.0, 0.0]]), "keys"),
        (_set_entry("sub_float_weights", (0, 1), np.nan), "sub_float_weights"),
        (_set_entry("sub_float_weights", (1, 0), -0.5), "sub_float_weights"),
        (_set_entry("sub_incidence", (0, 0), np.nan), "sub_incidence"),
        (_set_entry("keys", (0, 0), np.nan), "keys"),
        (_set_entry("sub_importance", 1, np.nan), "sub_importance"),
        (_set_index(2), "total_hh_sub_control_index"),
        (_set_index(True), "total_hh_sub_control_index"),
    ],
)
def test_build_problem_rejects_invalid_inputs(change, name):
    kwargs = _inputs(
        [[0.5, 1.5], [0.5, 0.5]],
        [[1, 1], [1, 0]],
        household_controls=[1, 1],
        parent_incidence=[[1], [0]],
        parent_importance=[500.0],
    )
    cr.build_problem(**kwargs)  # the unchanged inputs are valid
    change(kwargs)
    with pytest.raises(ValueError, match=name):
        cr.build_problem(**kwargs)


# --- the controlled LP -------------------------------------------------------

STAT_KEYS = {
    "x_count",
    "fractional_x",
    "slack_cell",
    "slack_parent_sub",
    "slack_parent",
    "slack_conserve",
    "solver",
}

# the solver of the behaviour tests; test_all_solvers_agree covers the others
SOLVER = "highs-ds"


def _log_keys(weights):
    """Upstream's household keys: ln(sub residual) + ln(summed residual)."""
    resid = np.asarray(weights, dtype=np.float64) % 1.0
    floor = np.exp(-725)  # upstream's LOG_OVERFLOW
    return np.log(np.maximum(resid, floor)) + np.log(
        np.maximum(resid.sum(axis=0), floor)
    )


def _controlled(weights, incidence, household_controls, **kwargs):
    """Build the problem and solve the controlled LP as the integerizer will."""
    inputs = _inputs(weights, incidence, household_controls, **kwargs)
    problem = cr.build_problem(**inputs)
    timing = {}
    resid_out, status, stats = controlled_lp.solve_relaxation(
        problem,
        inputs["sub_incidence"],
        inputs["parent_incidence"],
        inputs["sub_float_weights"] % 1.0,
        inputs["total_hh_sub_control_index"],
        SOLVER,
        timeout_seconds=60,
        timing=timing,
    )
    assert set(stats) == STAT_KEYS
    assert stats["solver"] == SOLVER
    return problem, resid_out, status, stats, timing


# stats of solve_controlled: those of solve_relaxation plus the attempts, the
# repair and the problem's household counts
CONTROLLED_STAT_KEYS = STAT_KEYS | {
    "attempts",
    "repaired_x",
    "fractional_after_repair",
    "repair_failed",
    "repair_seconds",
    "conserve_deviation_lp",
    "conserve_deviation_repaired",
    "objective_lp",
    "objective_repaired",
    "objective_seconds",
    "hh_unreachable",
    "hh_outside_lp",
    "hh_excess",
}


def _controlled_repaired(weights, incidence, household_controls, **kwargs):
    """As _controlled, through solve_controlled with the repair on."""
    inputs = _inputs(weights, incidence, household_controls, **kwargs)
    problem = cr.build_problem(**inputs)
    timing = {}
    resid_out, status, stats = controlled_lp.solve_controlled(
        problem,
        inputs["sub_incidence"],
        inputs["parent_incidence"],
        inputs["sub_float_weights"] % 1.0,
        inputs["total_hh_sub_control_index"],
        solver=SOLVER,
        retry_solver="none",
        repair=True,
        timeout_seconds=60,
        timing=timing,
    )
    assert set(stats) == CONTROLLED_STAT_KEYS
    assert stats["solver"] == SOLVER
    return problem, resid_out, status, stats, timing


def _rounded(weights, resid_out, household_controls):
    """Integer weights (zones, rows) after smart_round per zone, as upstream."""
    weights = np.asarray(weights, dtype=np.float64)
    int_weights = weights.astype(int)
    return np.array(
        [
            smart_round(int_weights[z], resid_out[z], household_controls[z])
            for z in range(weights.shape[0])
        ]
    )


def _upstream_inputs(
    weights, sub_incidence, sub_importance, parent_incidence, parent_importance
):
    """Arguments of np_simul_integerizer_ortools, derived exactly as
    SimulIntegerizer.integerize derives them (household total in column 0)."""
    total_hh_sub_control_index = 0
    sub_float_weights = np.asarray(weights, dtype=np.float64)
    sub_incidence = np.asarray(sub_incidence, dtype=np.float64)
    parent_incidence = np.asarray(parent_incidence, dtype=np.float64)
    sub_int_weights = sub_float_weights.astype(int)
    sub_resid_weights = sub_float_weights % 1.0
    sub_control_importance = np.array(sub_importance, dtype=np.float64)

    relaxed_sub_control_totals = np.dot(sub_float_weights, sub_incidence)
    lp_right_hand_side = np.round(relaxed_sub_control_totals) - np.dot(
        sub_int_weights, sub_incidence
    )
    lp_right_hand_side = np.maximum(lp_right_hand_side, 0.0)
    sub_num_households = relaxed_sub_control_totals[:, (total_hh_sub_control_index,)]
    sub_max_control_values = np.amax(sub_incidence, axis=0) * sub_num_households
    relax_ge_upper_bound = np.maximum(sub_max_control_values - lp_right_hand_side, 0)
    hh_constraint_ge_bound = np.maximum(sub_max_control_values, lp_right_hand_side)
    total_hh_right_hand_side = lp_right_hand_side[:, total_hh_sub_control_index]

    parent_float_weights = np.sum(sub_float_weights, axis=0)
    parent_int_weights = np.sum(sub_int_weights, axis=0)
    parent_resid_weights = np.sum(sub_resid_weights, axis=0)
    relaxed_parent_control_totals = np.dot(parent_float_weights, parent_incidence)
    parent_countrol_importance = np.array(parent_importance, dtype=np.float64)
    parent_lp_right_hand_side = np.round(relaxed_parent_control_totals) - np.dot(
        parent_int_weights, parent_incidence
    )
    parent_lp_right_hand_side = np.maximum(parent_lp_right_hand_side, 0.0)
    parent_num_households = np.sum(sub_num_households)
    parent_max_possible_control_values = (
        np.amax(parent_incidence, axis=0) * parent_num_households
    )
    parent_relax_ge_upper_bound = np.maximum(
        parent_max_possible_control_values - parent_lp_right_hand_side, 0
    )
    parent_hh_constraint_ge_bound = np.maximum(
        parent_max_possible_control_values, parent_lp_right_hand_side
    )
    return dict(
        sub_int_weights=sub_int_weights,
        parent_countrol_importance=parent_countrol_importance,
        parent_relax_ge_upper_bound=parent_relax_ge_upper_bound,
        sub_control_importance=sub_control_importance,
        sub_float_weights=sub_float_weights,
        sub_resid_weights=sub_resid_weights,
        lp_right_hand_side=lp_right_hand_side,
        parent_hh_constraint_ge_bound=parent_hh_constraint_ge_bound,
        sub_incidence=sub_incidence,
        parent_incidence=parent_incidence,
        total_hh_right_hand_side=total_hh_right_hand_side,
        relax_ge_upper_bound=relax_ge_upper_bound,
        parent_lp_right_hand_side=parent_lp_right_hand_side,
        hh_constraint_ge_bound=hh_constraint_ge_bound,
        parent_resid_weights=parent_resid_weights,
        total_hh_sub_control_index=total_hh_sub_control_index,
        total_hh_parent_control_index=-1,
        timeout_in_seconds=60,
    )


def test_controlled_keeps_parent_total_of_rare_group():
    # columns: household total, rare control, kind_a; ten zones of 3 households.
    # Row 0 is the rare group (parent weight 1, F = 0.1 per zone); rows 1 and 2
    # fill the zones (parent weights 14 and 15).
    rare = 1
    incidence = [[1, 1, 0], [1, 0, 1], [1, 0, 0]]
    row_1 = [1.4, 1.2, 1.6, 1.3, 1.5, 1.4, 1.2, 1.6, 1.3, 1.5]
    row_2 = [1.5, 1.7, 1.3, 1.6, 1.4, 1.5, 1.7, 1.3, 1.6, 1.4]
    weights = np.array([[0.1] * 10, row_1, row_2]).T
    controls = [3] * 10
    importance = [1e9, 1000.0, 1000.0]

    problem, resid_out, status, stats, _ = _controlled(
        weights,
        incidence,
        controls,
        importance=importance,
        keys=_log_keys(weights),
    )
    assert status in STATUS_SUCCESS
    rounded = _rounded(weights, resid_out, controls)
    assert (rounded @ np.asarray(incidence))[:, rare].sum() == 1
    np.testing.assert_array_equal(rounded.sum(axis=1), controls)
    np.testing.assert_array_equal(rounded.sum(axis=0), [1, 14, 15])
    assert stats["slack_parent_sub"] == pytest.approx(0.0, abs=1e-6)

    # Upstream on the same data: its cell target np.round(F) of the rare
    # control is 0 in every zone, and it bounds the slack relax_le by that
    # target, so the rare row cannot be rounded up anywhere.
    upstream = _upstream_inputs(
        weights, incidence, importance, np.zeros((3, 0)), np.zeros(0)
    )
    np.testing.assert_array_equal(upstream["lp_right_hand_side"][:, rare], 0.0)
    up_out, up_status = lp_ortools.np_simul_integerizer_ortools(**upstream)
    assert up_status in STATUS_SUCCESS
    up_rounded = _rounded(weights, up_out, controls)
    assert (up_rounded @ np.asarray(incidence))[:, rare].sum() == 0


def test_controlled_is_feasible_where_upstream_is_infeasible():
    # columns: household total, c1, c2, c3. In zone 0 every row with a residual
    # carries a control whose float total (0.3, 0.3, 0.4) rounds to 0, while the
    # zone needs one more household (F_t = 3, integer part 2).
    incidence = [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 0]]
    weights = [[0.3, 0.3, 0.4, 2.0], [0.7, 0.7, 0.6, 1.0]]
    controls = [3, 3]
    importance = [1e9, 1000.0, 1000.0, 1000.0]

    upstream = _upstream_inputs(
        weights, incidence, importance, np.zeros((4, 0)), np.zeros(0)
    )
    assert upstream["total_hh_right_hand_side"][0] == 1
    np.testing.assert_array_equal(upstream["lp_right_hand_side"][0, 1:], 0.0)
    _, up_status = lp_ortools.np_simul_integerizer_ortools(**upstream)
    assert up_status == "INFEASIBLE"

    problem, resid_out, status, stats, _ = _controlled(
        weights, incidence, controls, importance=importance
    )
    assert status in STATUS_SUCCESS
    rounded = _rounded(weights, resid_out, controls)
    np.testing.assert_array_equal(rounded.sum(axis=1), controls)
    np.testing.assert_array_equal(rounded.sum(axis=0), [1, 1, 1, 3])
    assert stats["slack_conserve"] == pytest.approx(0.0, abs=1e-6)
    # the extra household of zone 0 is the row with the largest fraction (c3)
    np.testing.assert_array_equal(rounded[0], [0, 0, 1, 2])


def _conservation_case(solve=_controlled):
    """Four zones; columns household total, c1, c2, persons; one parent control.
    Rows 0 and 2 have parent weight 1, rows 1 and 3 parent weight 2, row 4 7.
    ``solve`` is _controlled (the LP alone) or _controlled_repaired."""
    incidence = np.array(
        [[1, 1, 0, 1], [1, 0, 1, 2], [1, 1, 1, 3], [1, 0, 0, 2], [1, 0, 1, 4]],
        dtype=np.float64,
    )
    parent_incidence = np.array([[0], [1], [1], [0], [1]], dtype=np.float64)
    weights = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.6, 0.4, 0.5, 0.5],
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.7, 0.2, 0.6],
            [1.55, 1.45, 2.75, 1.25],
        ]
    ).T
    controls = [3, 3, 4, 3]
    solved = solve(
        weights,
        incidence,
        controls,
        importance=[1e9, 500.0, 2000.0, 300.0],
        parent_incidence=parent_incidence,
        parent_importance=[800.0],
        keys=_log_keys(weights),
    )
    return weights, np.hstack([incidence, parent_incidence]), controls, solved


def test_controlled_conserves_parent_weight():
    # the LP solution itself: every row keeps its parent weight and every
    # control (sub and parent) its float total over the zones
    weights, all_incidence, controls, solved = _conservation_case()
    problem, resid_out, status, stats, _ = solved
    assert status in STATUS_SUCCESS
    assert stats["x_count"] == 20
    assert stats["slack_conserve"] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(resid_out.sum(axis=1), problem.hh_count, atol=1e-6)
    lp_weights = weights.astype(int) + resid_out
    np.testing.assert_allclose(lp_weights.sum(axis=0), [1, 2, 1, 2, 7], atol=1e-6)
    np.testing.assert_allclose(
        lp_weights.sum(axis=0) @ all_incidence,
        weights.sum(axis=0) @ all_incidence,
        atol=1e-6,
    )


def test_controlled_rounding_keeps_parent_weight_at_a_fractional_vertex():
    # The LP optimum of this case is a fractional vertex (x = 0.5 for rows 1
    # and 4 in zones 0 and 1). Handed to smart_round as it is, each zone is
    # rounded on its own and one unit moves from row 1 to row 4. The repair
    # rounds the fractional values jointly over zones and rows first.
    *_, (_, _, _, lp_stats, _) = _conservation_case()
    assert lp_stats["fractional_x"] > 0, "the case must have a fractional vertex"
    weights, all_incidence, controls, solved = _conservation_case(_controlled_repaired)
    problem, resid_out, status, stats, _ = solved
    assert status in STATUS_SUCCESS
    assert stats["repaired_x"] == lp_stats["fractional_x"]
    assert stats["fractional_after_repair"] == 0
    assert stats["repair_failed"] == 0
    assert stats["conserve_deviation_repaired"] == 0.0
    rounded = _rounded(weights, resid_out, controls)
    np.testing.assert_array_equal(rounded.sum(axis=1), controls)
    np.testing.assert_array_equal(rounded.sum(axis=0), [1, 2, 1, 2, 7])
    np.testing.assert_allclose(
        rounded.sum(axis=0) @ all_incidence,
        weights.sum(axis=0) @ all_incidence,
        atol=1e-9,
    )


def test_controlled_prefers_the_cell_with_the_larger_fraction():
    # one unit of c1 to place, F = [0.7, 0.3]; row 1 carries only the
    # household total. The keys favour the wrong zone at the clip limit, so
    # the cell distance at the penalty floor (q = 100) must outrank them.
    incidence = [[1, 1], [1, 0]]
    weights = [[0.7, 0.3], [0.3, 0.7]]
    controls = [1, 1]
    keys = [[0.0, cr.KEY_CLIP], [cr.KEY_CLIP, 0.0]]

    problem, resid_out, status, stats, _ = _controlled(
        weights, incidence, controls, importance=[1e9, 100.0], keys=keys
    )
    assert status in STATUS_SUCCESS
    rounded = _rounded(weights, resid_out, controls)
    np.testing.assert_array_equal(rounded[:, 0], [1, 0])
    np.testing.assert_array_equal(rounded[:, 1], [0, 1])


def test_controlled_records_household_count_mismatch():
    # the zones hold 2 + 2 households, their controls allow 2 + 1: one
    # household of the parent cannot be kept
    incidence = [[1, 1], [1, 0], [1, 0]]
    weights = [[0.5, 0.5, 1.0], [0.5, 0.5, 1.0]]
    controls = [2, 1]

    problem, resid_out, status, stats, _ = _controlled(weights, incidence, controls)
    assert status in STATUS_SUCCESS
    np.testing.assert_array_equal(problem.hh_count, [1, 0])
    assert problem.hh_excess == 0
    assert problem.hh_outside_lp == 0
    assert problem.hh_unreachable == 0
    assert stats["slack_conserve"] == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(resid_out.sum(axis=1), problem.hh_count, atol=1e-6)
    rounded = _rounded(weights, resid_out, controls)
    np.testing.assert_array_equal(rounded.sum(axis=1), controls)


def test_controlled_without_parent_controls():
    incidence = [[1, 1, 0], [1, 0, 1], [1, 1, 1]]
    weights = np.array([[0.5, 0.2, 0.3], [0.4, 0.4, 1.2], [1.1, 0.4, 0.5]]).T
    controls = [2, 1, 2]

    problem, resid_out, status, stats, timing = _controlled(
        weights, incidence, controls, keys=_log_keys(weights)
    )
    assert problem.parent_target.shape == (0,)
    assert status in STATUS_SUCCESS
    assert stats["slack_parent"] == 0.0
    assert stats["slack_conserve"] == pytest.approx(0.0, abs=1e-6)
    assert set(timing) == {"build_s", "solve_s"}
    rounded = _rounded(weights, resid_out, controls)
    np.testing.assert_array_equal(rounded.sum(axis=1), controls)
    np.testing.assert_array_equal(rounded.sum(axis=0), [1, 2, 2])


def test_controlled_single_zone_with_integer_weights(monkeypatch):
    import scipy.optimize
    from ortools.linear_solver import pywraplp

    def no_solver(*args, **kwargs):
        raise AssertionError("no solver may be called without variables")

    monkeypatch.setattr(scipy.optimize, "linprog", no_solver)
    monkeypatch.setattr(pywraplp.Solver, "CreateSolver", no_solver)
    weights = [[2.0, 1.0, 3.0]]
    controls = [6]

    problem, resid_out, status, stats, timing = _controlled(
        weights, [[1, 1], [1, 0], [1, 1]], controls
    )
    assert status == STATUS_OPTIMAL
    np.testing.assert_array_equal(resid_out, np.zeros((1, 3)))
    assert stats == {
        "x_count": 0,
        "fractional_x": 0,
        "slack_cell": 0.0,
        "slack_parent_sub": 0.0,
        "slack_parent": 0.0,
        "slack_conserve": 0.0,
        "solver": SOLVER,
    }
    assert timing["solve_s"] == 0.0
    rounded = _rounded(weights, resid_out, controls)
    np.testing.assert_array_equal(rounded, [[2, 1, 3]])


# --- the sparse LP against the dense reference, and the solver backends ------


def _random_case(seed):
    """A random small parent: (problem, sub_incidence, parent_incidence,
    sub_resid_weights, keys, weights). Column 0 is the household total.

    Every row's weights over the zones sum to an integer parent weight (as the
    balancer leaves them); some rows sit wholly in one zone (no residual,
    hence no variable); some zones' household controls are off by one."""
    rng = np.random.default_rng(seed)
    zones = int(rng.integers(2, 6))
    rows = int(rng.integers(5, 26))
    controls = int(rng.integers(3, 7))
    parent_controls = int(rng.integers(0, 4))
    parent_weight = rng.integers(1, 5, size=rows).astype(np.float64)
    weights = rng.dirichlet(np.full(zones, 0.8), size=rows).T * parent_weight
    whole = rng.random(rows) < 0.2
    weights[:, whole] = 0.0
    weights[rng.integers(0, zones, size=whole.sum()), np.flatnonzero(whole)] = (
        parent_weight[whole]
    )
    incidence = rng.integers(0, 4, size=(rows, controls)) * (
        rng.random((rows, controls)) < 0.5
    )
    incidence[:, 0] = 1
    parent_incidence = rng.integers(0, 4, size=(rows, parent_controls)) * (
        rng.random((rows, parent_controls)) < 0.5
    )
    levels = [0.0, 50.0, 200.0, 1000.0, 5000.0, 20000.0, 1e6]
    importance = np.concatenate([[1e9], rng.choice(levels, controls - 1)])
    parent_importance = rng.choice(levels, parent_controls)
    household_controls = np.rint(weights.sum(axis=1))
    off = rng.random(zones) < 0.3
    household_controls[off] += rng.integers(-1, 2, size=off.sum())
    household_controls = np.maximum(household_controls, 0)
    keys = rng.uniform(-40.0, 40.0, size=(zones, rows))
    penalties = PENALTIES
    if seed % 4 == 3:
        # conservation free of charge: the dense form drops its rows
        penalties = PENALTIES._replace(conserve_penalty=0.0)
    inputs = _inputs(
        weights,
        incidence,
        household_controls,
        importance=importance,
        parent_incidence=parent_incidence,
        parent_importance=parent_importance,
        keys=keys,
    )
    inputs["penalties"] = penalties
    problem = cr.build_problem(**inputs)
    return (
        problem,
        inputs["sub_incidence"],
        inputs["parent_incidence"],
        weights % 1.0,
        keys,
        weights,
    )


def _sparse_and_dense(case, solver=SOLVER):
    problem, inc, pinc, resid, _, _ = case
    lp = controlled_lp.assemble(problem, inc, pinc, 0)
    sparse = controlled_lp.solve_lp(lp, solver, 60)
    dense = solve_dense_reference(problem, inc, pinc, resid, 0, 60)
    return lp, sparse, dense


@pytest.mark.parametrize("seed", range(20))
def test_sparse_lp_matches_the_dense_reference(seed):
    case = _random_case(seed)
    problem, inc, pinc, resid, _, _ = case
    lp, sparse, (dense_x, dense_status, dense_objective) = _sparse_and_dense(case)
    assert sparse.status_text in STATUS_SUCCESS
    assert dense_status in STATUS_SUCCESS
    assert sparse.objective == pytest.approx(dense_objective, rel=1e-6, abs=1e-6)
    # the sparse solution meets the hard zone counts
    x = np.zeros(problem.x_allowed.shape)
    x[problem.x_allowed] = sparse.values[lp.blocks["x"]]
    np.testing.assert_allclose(x.sum(axis=1), problem.hh_count, atol=1e-6)
    # and solve_relaxation reports the same success
    _, status, stats = controlled_lp.solve_relaxation(
        problem, inc, pinc, resid, 0, SOLVER, 60
    )
    assert status in STATUS_SUCCESS
    assert stats["x_count"] == int(problem.x_allowed.sum())


def _snapped(values):
    nearest = np.round(values)
    return np.where(np.abs(values - nearest) <= 1e-6, nearest, values)


def _cost_of(x, problem, weights, inc, pinc, keys, conserve_penalty, total_index=0):
    """Objective of the controlled LP at an integral x, computed from the
    definitions, not from the LP arrays.

    Cell (z, c != t): the LP's cell term is minus
    ``cell_cost(y) - q f + q (1 - 2f) band_lo``: the ranged-row cost equals
    ``cell_cost(y) - q f`` (``controlled_rounding`` docstring), and the
    fold into ``x_coef`` measures ``y`` from the integer part ``A`` instead
    of from ``lo = floor(F')``, a constant ``q (1 - 2f) (lo - A)`` per cell.
    """
    a = np.floor(weights)
    y = (a + x) @ inc
    float_totals = weights @ inc
    snapped = _snapped(float_totals)
    lo = np.floor(snapped)
    f = snapped - lo
    band_lo = lo - a @ inc
    q = problem.cell_penalty
    cell = 0.0
    for z in range(weights.shape[0]):
        for c in range(inc.shape[1]):
            if c == total_index:
                continue
            cell += cr.cell_cost(y[z, c], float_totals[z, c], q[c])
            cell += -q[c] * f[z, c] + q[c] * (1 - 2 * f[z, c]) * band_lo[z, c]
    conserve = (
        conserve_penalty
        * np.abs(x.sum(axis=0) - np.round((weights % 1.0).sum(axis=0))).sum()
    )
    sub_dev = np.abs(y.sum(axis=0) - np.round(float_totals.sum(axis=0)))
    sub_dev[total_index] = 0.0
    parent_sub = (problem.parent_sub_penalty * sub_dev).sum()
    parent_dev = np.abs(
        (a + x).sum(axis=0) @ pinc - np.round(weights.sum(axis=0) @ pinc)
    )
    parent = (problem.parent_penalty * parent_dev).sum()
    key_term = (x * np.clip(keys, -cr.KEY_CLIP, cr.KEY_CLIP)).sum()
    return key_term - cell - conserve - parent_sub - parent


# seeds of _random_case whose LP optimum is integral under every solver of
# SOLVERS (checked in the test), with slack in cells and conservation
INTEGRAL_SEEDS = (34, 36, 53)


@pytest.mark.parametrize("seed", INTEGRAL_SEEDS)
def test_reported_objective_equals_the_cost_of_the_solution(seed):
    case = _random_case(seed)
    problem, inc, pinc, resid, keys, weights = case
    lp = controlled_lp.assemble(problem, inc, pinc, 0)
    result = controlled_lp.solve_lp(lp, SOLVER, 60)
    assert result.status_text in STATUS_SUCCESS
    assert result.objective == pytest.approx(lp.objective @ result.values, rel=1e-9)
    x = np.zeros(problem.x_allowed.shape)
    x[problem.x_allowed] = result.values[lp.blocks["x"]]
    assert np.abs(x - np.round(x)).max() < 1e-6, "pick a seed with integral x"
    x = np.round(x)
    expected = _cost_of(x, problem, weights, inc, pinc, keys, problem.conserve_penalty)
    assert result.objective == pytest.approx(expected, rel=1e-9, abs=1e-6)


def _infeasible_lp():
    """x0 + x1 = 3 with x in [0, 1]."""
    empty = slice(2, 2)
    blocks = {name: empty for name in controlled_lp.VARIABLE_BLOCKS}
    blocks["x"] = slice(0, 2)
    no_index = np.zeros(0, dtype=np.int64)
    return controlled_lp.LinearProgram(
        objective=np.array([1.0, 1.0]),
        matrix=sp.csr_matrix(np.array([[1.0, 1.0]])),
        rhs=np.array([3.0]),
        lower=np.zeros(2),
        upper=np.ones(2),
        x_index=np.array([[0, 1]], dtype=np.int64),
        blocks=blocks,
        row_blocks={
            name: slice(0, 1) if name == "zone_count" else slice(1, 1)
            for name in controlled_lp.ROW_BLOCKS
        },
        cell_zone=no_index,
        cell_control=no_index,
        count_zone=np.array([0], dtype=np.int64),
        conserve_row=no_index,
        parent_sub_control=no_index,
        parent_control=no_index,
    )


def _agreement_cases():
    weights = np.array([[0.1] * 10, [1.4, 1.2] * 5, [1.5, 1.7] * 5]).T
    rare = (
        cr.build_problem(
            **_inputs(
                weights,
                [[1, 1, 0], [1, 0, 1], [1, 0, 0]],
                [3] * 10,
                importance=[1e9, 1000.0, 1000.0],
                keys=_log_keys(weights),
            )
        ),
        np.array([[1, 1, 0], [1, 0, 1], [1, 0, 0]], dtype=np.float64),
        np.zeros((3, 0)),
        weights % 1.0,
    )
    weights, _, _, (problem, _, _, _, _) = _conservation_case()
    incidence = np.array(
        [[1, 1, 0, 1], [1, 0, 1, 2], [1, 1, 1, 3], [1, 0, 0, 2], [1, 0, 1, 4]],
        dtype=np.float64,
    )
    parent_incidence = np.array([[0], [1], [1], [0], [1]], dtype=np.float64)
    conservation = (problem, incidence, parent_incidence, weights % 1.0)
    random_problem, inc, pinc, resid, _, _ = _random_case(7)
    return [rare, conservation, (random_problem, inc, pinc, resid)]


@pytest.mark.parametrize("solver", controlled_lp.SOLVERS)
def test_all_solvers_agree(solver):
    for problem, inc, pinc, resid in _agreement_cases():
        lp = controlled_lp.assemble(problem, inc, pinc, 0)
        result = controlled_lp.solve_lp(lp, solver, 60)
        _, dense_status, dense_objective = solve_dense_reference(
            problem, inc, pinc, resid, 0, 60
        )
        assert dense_status in STATUS_SUCCESS
        assert result.status_text in STATUS_SUCCESS
        assert result.objective == pytest.approx(dense_objective, rel=1e-6)
        assert result.values.shape == lp.objective.shape
        assert result.solve_seconds >= 0.0

    infeasible = controlled_lp.solve_lp(_infeasible_lp(), solver, 60)
    assert infeasible.status_text == "INFEASIBLE"
    assert infeasible.values is None
    assert np.isnan(infeasible.objective)


@pytest.mark.parametrize(
    "solver", [name for name in controlled_lp.SOLVERS if not name.startswith("highs")]
)
def test_a_stop_before_the_optimum_is_not_solved(solver, monkeypatch):
    # An OR-Tools LP backend reports FEASIBLE only when it stops at a limit
    # with a feasible, non-optimal point (clp-primal and glop-primal do so at
    # the time limit on a 40 x 3000 parent): that point must not pass as a
    # success.
    from ortools.linear_solver import pywraplp

    monkeypatch.setattr(
        pywraplp.Solver, "Solve", lambda self, *args: pywraplp.Solver.FEASIBLE
    )
    problem, inc, pinc, resid, _, _ = _random_case(0)
    lp = controlled_lp.assemble(problem, inc, pinc, 0)
    result = controlled_lp.solve_lp(lp, solver, 60)
    assert result.status_text == "NOT_SOLVED"
    assert result.values is None
    resid_out, status, stats = controlled_lp.solve_relaxation(
        problem, inc, pinc, resid, 0, solver, 60
    )
    assert status == "NOT_SOLVED"
    assert resid_out is resid
    assert stats["slack_conserve"] == 0.0


def test_unknown_solver_is_refused():
    problem, inc, pinc, resid, _, _ = _random_case(0)
    lp = controlled_lp.assemble(problem, inc, pinc, 0)
    with pytest.raises(ValueError, match="INTEGERIZER_CONTROLLED_SOLVER"):
        controlled_lp.solve_lp(lp, "gurobi", 60)
    with pytest.raises(ValueError, match="INTEGERIZER_CONTROLLED_SOLVER"):
        controlled_lp.solve_relaxation(problem, inc, pinc, resid, 0, "highs", 60)


def test_assemble_shapes_and_blocks():
    # two zones, three rows; columns household total, c1, c2, c3; one parent
    # control. Variables x0 = (0, 0), x1 = (0, 1), x2 = (1, 0), x3 = (1, 2).
    # Cell (1, c2) has no variable of non-zero incidence and the band [0, 0]:
    # left out. c3 has no incidence and target 0: no cell, no parent row.
    zeros = np.zeros((2, 4))
    problem = cr.ControlledProblem(
        x_allowed=np.array([[True, True, False], [True, False, True]]),
        hh_count=np.array([1, 1]),
        band_lo=np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        band_hi=np.array([[1.0, 1.0, 2.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        frac=zeros,
        under_price=np.array([[0.0, 150.0, 200.0, 0.0], [0.0, 300.0, 0.0, 0.0]]),
        over_price=np.array([[0.0, 50.0, 210.0, 0.0], [0.0, 7.0, 0.0, 0.0]]),
        cell_penalty=np.array([1e5, 100.0, 200.0, 300.0]),
        parent_sub_target=np.array([0.0, 1.0, 1.0, 0.0]),
        parent_sub_penalty=np.array([1e5, 1000.0, 2000.0, 3000.0]),
        parent_target=np.array([1.0]),
        parent_penalty=np.array([800.0]),
        conserve_target=np.array([1.0, 0.0, 1.0]),
        conserve_penalty=1e5,
        x_coef=np.array([[1.5, -2.0, 9.0], [3.0, 9.0, -4.0]]),
        hh_unreachable=0,
        hh_outside_lp=0,
        hh_excess=0,
    )
    inc = np.array([[1, 1, 0, 0], [1, 0, 2, 0], [1, 1, 0, 0]], dtype=np.float64)
    pinc = np.array([[0], [1], [1]], dtype=np.float64)

    lp = controlled_lp.assemble(problem, inc, pinc, 0)

    sizes = {
        "x": 4,
        "cell_band": 3,
        "cell_under": 3,
        "cell_over": 3,
        "conserve_under": 3,
        "conserve_over": 3,
        "parent_sub_under": 2,
        "parent_sub_over": 2,
        "parent_under": 1,
        "parent_over": 1,
    }
    assert list(lp.blocks) == list(controlled_lp.VARIABLE_BLOCKS)
    start = 0
    for name, size in sizes.items():
        assert lp.blocks[name] == slice(start, start + size), name
        start += size
    row_sizes = {"cell": 3, "zone_count": 2, "conserve": 3, "parent_sub": 2}
    row_sizes["parent"] = 1
    assert list(lp.row_blocks) == list(controlled_lp.ROW_BLOCKS)
    start = 0
    for name, size in row_sizes.items():
        assert lp.row_blocks[name] == slice(start, start + size), name
        start += size
    np.testing.assert_array_equal(lp.x_index, [[0, 1, -1], [2, -1, 3]])
    np.testing.assert_array_equal(lp.cell_zone, [0, 0, 1])
    np.testing.assert_array_equal(lp.cell_control, [1, 2, 1])
    np.testing.assert_array_equal(lp.count_zone, [0, 1])
    np.testing.assert_array_equal(lp.conserve_row, [0, 1, 2])
    np.testing.assert_array_equal(lp.parent_sub_control, [1, 2])
    np.testing.assert_array_equal(lp.parent_control, [0])

    # columns: x 0-3, band 4-6, cell under 7-9, cell over 10-12, conserve
    # under 13-15, conserve over 16-18, parent-sub under 19-20, over 21-22,
    # parent under 23, over 24
    expected = np.zeros((11, 25))
    # cell (0, c1): x0 (row 0, inc 1); cell (0, c2): x1 (row 1, inc 2);
    # cell (1, c1): x2 (row 0, inc 1), x3 (row 2, inc 1)
    expected[0, [0, 4, 7, 10]] = [1, -1, 1, -1]
    expected[1, [1, 5, 8, 11]] = [2, -1, 1, -1]
    expected[2, [2, 3, 6, 9, 12]] = [1, 1, -1, 1, -1]
    # zone counts
    expected[3, [0, 1]] = 1
    expected[4, [2, 3]] = 1
    # conservation of rows 0, 1, 2
    expected[5, [0, 2, 13, 16]] = [1, 1, 1, -1]
    expected[6, [1, 14, 17]] = [1, 1, -1]
    expected[7, [3, 15, 18]] = [1, 1, -1]
    # parent-sub c1 on rows 0 and 2: inc * (over - under)
    expected[8, [13, 15, 16, 18, 19, 21]] = [-1, -1, 1, 1, 1, -1]
    # parent-sub c2 on row 1 (inc 2)
    expected[9, [14, 17, 20, 22]] = [-2, 2, 1, -1]
    # parent control on rows 1 and 2
    expected[10, [14, 15, 17, 18, 23, 24]] = [-1, -1, 1, 1, 1, -1]
    assert sp.isspmatrix_csr(lp.matrix)
    np.testing.assert_array_equal(lp.matrix.toarray(), expected)

    # rhs: cells 0; counts; K; c1: 1 - (1 + 1); c2: 1 - 2 * 0; p: 1 - 1
    np.testing.assert_array_equal(lp.rhs, [0, 0, 0, 1, 1, 1, 0, 1, -1, 1, 0])
    big = 1e5
    np.testing.assert_array_equal(
        lp.objective,
        [1.5, -2.0, 3.0, -4.0]
        + [0.0] * 3
        + [-150.0, -200.0, -300.0]
        + [-50.0, -210.0, -7.0]
        + [-big] * 6
        + [-1000.0, -2000.0, -1000.0, -2000.0]
        + [-800.0, -800.0],
    )
    inf = np.inf
    np.testing.assert_array_equal(lp.lower, [0.0] * 4 + [0.0, 1.0, 1.0] + [0.0] * 18)
    np.testing.assert_array_equal(lp.upper, [1.0] * 4 + [1.0, 2.0, 1.0] + [inf] * 18)


def test_assemble_refuses_a_conservation_target_without_variables():
    problem, inc, pinc, _, _, _ = _random_case(0)
    no_x = np.flatnonzero(~problem.x_allowed.any(axis=0))
    assert no_x.size, "pick a seed with a row without variables"
    target = problem.conserve_target.copy()
    target[no_x[0]] = 1.0
    broken = dataclasses.replace(problem, conserve_target=target)
    with pytest.raises(ValueError, match="conserve_target"):
        controlled_lp.assemble(broken, inc, pinc, 0)


# --- repair of fractional LP values, retry with a second solver --------------


def _two_by_two(keys):
    """Two zones, two rows of parent weight 1 (weights 0.7 / 0.3 and 0.3 /
    0.7), only the household total: (problem, lp). One household per zone,
    every row conserves 1; x in the order x[0, 0], x[0, 1], x[1, 0], x[1, 1]."""
    problem = _build([[0.7, 0.3], [0.3, 0.7]], [[1], [1]], [1, 1], keys=keys)
    lp = controlled_lp.assemble(problem, np.ones((2, 1)), np.zeros((2, 0)), 0)
    return problem, lp


# keys at the clip limit for the points (x[0, 1], x[1, 0]) and (x[0, 0], x[1, 1])
KEYS_OFF_DIAGONAL = [[0.0, cr.KEY_CLIP], [cr.KEY_CLIP, 0.0]]
KEYS_DIAGONAL = [[cr.KEY_CLIP, 0.0], [0.0, cr.KEY_CLIP]]


@pytest.mark.parametrize(
    "values,expected",
    [([0.7, 0.3, 0.3, 0.7], [1, 0, 0, 1]), ([0.3, 0.7, 0.7, 0.3], [0, 1, 1, 0])],
)
def test_repair_minimises_the_distance_to_the_lp_values(values, expected):
    # Two integer points keep the zone counts and the row totals. The L1
    # distance to the LP values decides (1.2 against 2.8), whatever the keys
    # favour: they only break ties.
    for keys in (KEYS_OFF_DIAGONAL, KEYS_DIAGONAL):
        problem, lp = _two_by_two(keys)
        x, stats = controlled_lp.repair_fractional(
            lp, np.array(values), problem, SOLVER, 60
        )
        np.testing.assert_array_equal(x, expected)
        assert stats["repaired_x"] == 4
        assert stats["fractional_after_repair"] == 0
        assert stats["repair_failed"] == 0
        assert stats["conserve_deviation_lp"] == pytest.approx(0.0, abs=1e-12)
        assert stats["conserve_deviation_repaired"] == 0.0
        assert stats["repair_seconds"] >= 0.0


@pytest.mark.parametrize(
    "keys,expected",
    [(KEYS_OFF_DIAGONAL, [0, 1, 1, 0]), (KEYS_DIAGONAL, [1, 0, 0, 1])],
)
def test_repair_breaks_ties_by_the_lp_coefficients(keys, expected):
    # x = 0.5 everywhere: both integer points are at L1 distance 2, the LP's
    # own x coefficients (here the keys) decide
    problem, lp = _two_by_two(keys)
    x, stats = controlled_lp.repair_fractional(lp, np.full(4, 0.5), problem, SOLVER, 60)
    np.testing.assert_array_equal(x, expected)


def test_repair_without_fractional_values_only_snaps(monkeypatch):
    def no_solver(*args, **kwargs):
        raise AssertionError("no repair LP without fractional values")

    problem, lp = _two_by_two(KEYS_DIAGONAL)
    monkeypatch.setattr(controlled_lp, "solve_lp", no_solver)
    values = np.array([1.0 - 1e-9, 1e-9, -1e-12, 1.0])
    x, stats = controlled_lp.repair_fractional(lp, values, problem, SOLVER, 60)
    np.testing.assert_array_equal(x, [1, 0, 0, 1])
    assert stats["repaired_x"] == 0
    assert stats["fractional_after_repair"] == 0
    assert stats["repair_failed"] == 0


def test_repair_refuses_values_that_miss_the_zone_counts():
    problem, lp = _two_by_two(KEYS_DIAGONAL)
    with pytest.raises(ValueError, match="zone"):
        controlled_lp.repair_fractional(
            lp, np.array([0.7, 0.3, 0.3, 0.3]), problem, SOLVER, 60
        )
    with pytest.raises(ValueError, match="x_values"):
        controlled_lp.repair_fractional(
            lp, np.array([0.5, 0.5, 0.5]), problem, SOLVER, 60
        )


def test_repair_returns_integers_with_exact_zone_counts():
    # The bound on conservation that the repair guarantees, per household row
    # h with LP total T_h = sum_z x and integer target K_h: the repair keeps
    # T_h where it is integral and moves it to floor(T_h) or ceil(T_h)
    # otherwise, so, K_h being an integer, |T'_h - K_h| <= ceil(|T_h - K_h|).
    # Summed over the rows: the deviation after the repair is at most the
    # deviation of the LP plus the number of rows whose LP total is
    # fractional (rows with a fractional conservation slack), strictly less
    # when there is any; rows the LP conserves stay conserved.
    tolerance = controlled_lp.FRACTIONAL_TOLERANCE
    fractional_cases = 0
    for seed in range(30):
        problem, inc, pinc, _, _, weights = _random_case(seed)
        lp = controlled_lp.assemble(problem, inc, pinc, 0)
        result = controlled_lp.solve_lp(lp, SOLVER, 60)
        assert result.status_text in STATUS_SUCCESS
        x_lp = result.values[lp.blocks["x"]]
        fractional = (x_lp > tolerance) & (x_lp < 1.0 - tolerance)
        if not fractional.any():
            continue
        fractional_cases += 1

        x, stats = controlled_lp.repair_fractional(lp, x_lp, problem, SOLVER, 60)
        assert stats["repair_failed"] == 0
        assert stats["repaired_x"] == int(fractional.sum())
        assert stats["fractional_after_repair"] == 0
        assert np.isin(x, [0.0, 1.0]).all()
        np.testing.assert_array_equal(x[~fractional], np.round(x_lp[~fractional]))
        grid = np.zeros(problem.x_allowed.shape)
        grid[problem.x_allowed] = x
        np.testing.assert_array_equal(grid.sum(axis=1), problem.hh_count)

        # smart_round rounds up the hh_count[z] largest values of a zone:
        # exactly the repaired ones, so it changes nothing
        a = weights.astype(int)
        for z in range(grid.shape[0]):
            rounded = smart_round(a[z], grid[z], a[z].sum() + problem.hh_count[z])
            np.testing.assert_array_equal(rounded, a[z] + grid[z])

        lp_grid = np.zeros(problem.x_allowed.shape)
        lp_grid[problem.x_allowed] = x_lp
        dev_lp = np.abs(lp_grid.sum(axis=0) - problem.conserve_target)
        dev_repaired = np.abs(grid.sum(axis=0) - problem.conserve_target)
        assert (dev_repaired <= np.ceil(dev_lp - 1e-6) + 1e-9).all()
        fractional_rows = int((np.abs(dev_lp - np.round(dev_lp)) > 1e-6).sum())
        assert stats["conserve_deviation_lp"] == pytest.approx(dev_lp.sum(), abs=1e-9)
        assert stats["conserve_deviation_repaired"] == pytest.approx(
            dev_repaired.sum(), abs=1e-9
        )
        assert (
            stats["conserve_deviation_repaired"]
            <= stats["conserve_deviation_lp"] + fractional_rows + 1e-9
        )
    assert fractional_cases >= 10


def _failed(status_text):
    return controlled_lp.LpResult(status_text, None, float("nan"), 0.25)


@pytest.mark.parametrize("failure", ["status", "not integral"])
def test_repair_failure_keeps_the_lp_values_and_is_counted(
    failure, monkeypatch, caplog, fork_logging
):
    def failing(arrays, solver, timeout_seconds, **kwargs):
        if failure == "status":
            return _failed("NOT_SOLVED")
        values = np.full(arrays.objective.size, 0.5)
        return controlled_lp.LpResult(STATUS_OPTIMAL, values, 0.0, 0.0)

    problem, lp = _two_by_two(KEYS_DIAGONAL)
    values = np.array([0.6, 0.4, 0.4, 0.6])
    monkeypatch.setattr(controlled_lp, "solve_lp", failing)
    with caplog.at_level(logging.WARNING, logger=controlled_lp.logger.name):
        x, stats = controlled_lp.repair_fractional(lp, values, problem, SOLVER, 60)
    np.testing.assert_array_equal(x, values)
    assert stats["repair_failed"] == 1
    assert stats["repaired_x"] == 4
    assert stats["fractional_after_repair"] == 4
    assert stats["conserve_deviation_repaired"] == stats["conserve_deviation_lp"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "repair" in message
    assert ("NOT_SOLVED" if failure == "status" else "not integral") in message


def test_repair_failure_in_solve_controlled_hands_on_the_lp_values(monkeypatch):
    lp_out = _conservation_case()[3][1]
    original = controlled_lp.solve_lp

    def failing_repair(lp, solver, timeout_seconds, **kwargs):
        # the repair LP is the only one that is not a LinearProgram
        if isinstance(lp, controlled_lp.LinearProgram):
            return original(lp, solver, timeout_seconds, **kwargs)
        return _failed("ABNORMAL")

    monkeypatch.setattr(controlled_lp, "solve_lp", failing_repair)
    *_, (problem, resid_out, status, stats, _) = _conservation_case(
        _controlled_repaired
    )
    assert status in STATUS_SUCCESS
    assert stats["repair_failed"] == 1
    assert stats["fractional_after_repair"] == stats["fractional_x"] > 0
    assert np.isnan(stats["objective_repaired"])
    np.testing.assert_array_equal(resid_out, lp_out)


def _patch_failures(monkeypatch, failures):
    """solve_lp fails for the solver names in ``failures`` (name -> status);
    returns the list of solver names called."""
    original = controlled_lp.solve_lp
    calls = []

    def solve(lp, solver, timeout_seconds, **kwargs):
        calls.append(solver)
        if solver in failures:
            return _failed(failures[solver])
        return original(lp, solver, timeout_seconds, **kwargs)

    monkeypatch.setattr(controlled_lp, "solve_lp", solve)
    return calls


# stats that are measured times
SECONDS = {"repair_seconds", "objective_seconds"}


def test_retry_uses_the_second_solver_and_records_it(monkeypatch, caplog, fork_logging):
    problem, inc, pinc, resid, _, _ = _random_case(0)
    direct_out, direct_status, direct_stats = controlled_lp.solve_controlled(
        problem, inc, pinc, resid, 0, "clp-dual", "none", True, 60
    )
    calls = _patch_failures(monkeypatch, {"highs-ds": "NOT_SOLVED"})
    timing = {}
    with caplog.at_level(logging.WARNING, logger=controlled_lp.logger.name):
        resid_out, status, stats = controlled_lp.solve_controlled(
            problem, inc, pinc, resid, 0, "highs-ds", "clp-dual", True, 60, timing
        )
    assert calls[:2] == ["highs-ds", "clp-dual"]
    assert "highs-ds" not in calls[2:]
    assert status == direct_status == STATUS_OPTIMAL
    assert stats["attempts"] == 2
    assert direct_stats["attempts"] == 1
    assert stats["solver"] == "clp-dual"
    np.testing.assert_array_equal(resid_out, direct_out)
    assert {k: v for k, v in stats.items() if k not in SECONDS | {"attempts"}} == {
        k: v for k, v in direct_stats.items() if k not in SECONDS | {"attempts"}
    }
    assert timing["solve_s"] >= 0.25  # the failed attempt counts
    message = " ".join(
        r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
    )
    for token in ("highs-ds", "NOT_SOLVED", "clp-dual", "OPTIMAL"):
        assert token in message


def test_retry_none_returns_the_first_failure(monkeypatch):
    problem, inc, pinc, resid, _, _ = _random_case(0)
    calls = _patch_failures(monkeypatch, {"highs-ds": "INFEASIBLE"})
    resid_out, status, stats = controlled_lp.solve_controlled(
        problem, inc, pinc, resid, 0, "highs-ds", "none", True, 60
    )
    assert calls == ["highs-ds"]
    assert status == "INFEASIBLE"
    assert resid_out is resid
    assert stats["attempts"] == 1
    assert stats["solver"] == "highs-ds"
    assert stats["repaired_x"] == 0
    assert stats["repair_failed"] == 0
    assert np.isnan(stats["objective_lp"])
    assert np.isnan(stats["objective_repaired"])
    assert stats["hh_outside_lp"] == problem.hh_outside_lp

    # a retry solver equal to the first is no second attempt
    _, _, stats = controlled_lp.solve_controlled(
        problem, inc, pinc, resid, 0, "highs-ds", "highs-ds", True, 60
    )
    assert stats["attempts"] == 1
    # an unknown retry solver is refused before any solve
    with pytest.raises(ValueError, match="INTEGERIZER_CONTROLLED_RETRY_SOLVER"):
        controlled_lp.solve_controlled(
            problem, inc, pinc, resid, 0, "highs-ds", "gurobi", True, 60
        )


def test_both_solvers_failing_returns_the_input_residuals(
    monkeypatch, caplog, fork_logging
):
    problem, inc, pinc, resid, _, _ = _random_case(0)
    calls = _patch_failures(
        monkeypatch, {"highs-ds": "INFEASIBLE", "clp-dual": "ABNORMAL"}
    )
    with caplog.at_level(logging.WARNING, logger=controlled_lp.logger.name):
        resid_out, status, stats = controlled_lp.solve_controlled(
            problem, inc, pinc, resid, 0, "highs-ds", "clp-dual", True, 60
        )
    assert calls == ["highs-ds", "clp-dual"]
    assert status == "ABNORMAL"
    assert resid_out is resid
    assert stats["attempts"] == 2
    assert stats["solver"] == "clp-dual"
    assert stats["fractional_x"] == 0
    assert stats["repaired_x"] == 0
    message = " ".join(
        r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
    )
    assert "INFEASIBLE" in message and "ABNORMAL" in message


@pytest.mark.parametrize("seed", [0, 1, 2, INTEGRAL_SEEDS[0]])
def test_integrality_loss_is_reported(seed):
    problem, inc, pinc, resid, keys, weights = _random_case(seed)
    resid_out, status, stats = controlled_lp.solve_controlled(
        problem, inc, pinc, resid, 0, SOLVER, "none", True, 60
    )
    assert status in STATUS_SUCCESS
    assert set(stats) == CONTROLLED_STAT_KEYS
    lp = controlled_lp.assemble(problem, inc, pinc, 0)
    relaxed = controlled_lp.solve_lp(lp, SOLVER, 60)
    assert stats["objective_lp"] == pytest.approx(relaxed.objective, rel=1e-9)

    # the repaired x, fixed through its bounds, with the slacks re-optimised
    x = resid_out[problem.x_allowed]
    assert np.isin(x, [0.0, 1.0]).all()
    lower = np.array(lp.lower)
    upper = np.array(lp.upper)
    lower[lp.blocks["x"]] = x
    upper[lp.blocks["x"]] = x
    fixed = controlled_lp.solve_lp(
        dataclasses.replace(lp, lower=lower, upper=upper), SOLVER, 60
    )
    assert fixed.status_text in STATUS_SUCCESS
    assert stats["objective_repaired"] == pytest.approx(
        fixed.objective, rel=1e-9, abs=1e-6
    )
    # and from the definitions, independent of the LP arrays
    expected = _cost_of(
        resid_out, problem, weights, inc, pinc, keys, problem.conserve_penalty
    )
    assert stats["objective_repaired"] == pytest.approx(expected, rel=1e-9, abs=1e-6)
    # the LP optimum is an upper bound of every integer solution
    slack = 1e-9 * max(1.0, abs(stats["objective_lp"]))
    assert stats["objective_repaired"] <= stats["objective_lp"] + slack
    if seed in INTEGRAL_SEEDS:
        assert stats["repaired_x"] == 0
        assert stats["objective_repaired"] == pytest.approx(
            stats["objective_lp"], rel=1e-9
        )
    else:
        assert stats["repaired_x"] > 0
    for name in ("hh_unreachable", "hh_outside_lp", "hh_excess"):
        assert stats[name] == getattr(problem, name)


# --- the simultaneous integerizer in controlled mode ------------------------


@pytest.fixture
def settings(monkeypatch):
    """Inject a settings dict (path record off)."""
    from populationsim.core import config
    from populationsim.integerizing import bs_options, path_record

    values = {}
    monkeypatch.setattr(
        config, "setting", lambda key, default=None: values.get(key, default)
    )
    bs_options.reset_for_tests()
    path_record.reset_for_tests()
    yield values
    bs_options.reset_for_tests()
    path_record.reset_for_tests()


# columns: household total, a rare group, kind_a (sub controls) and kind_b
# (parent control). Four zones of three households each; row 0 is the rare
# group with parent weight 1 spread evenly (0.25 per zone), rows 1 and 2 have
# parent weights 6 and 5.
_ZONE_IDS = [11, 12, 13, 14]
_WEIGHTS = np.array(
    [
        [0.25, 0.25, 0.25, 0.25],
        [1.4, 1.2, 1.6, 1.8],
        [1.35, 1.55, 1.15, 0.95],
    ]
)
_INCIDENCE = {
    "num_hh": [1, 1, 1],
    "rare": [1, 0, 0],
    "kind_a": [0, 1, 0],
    "kind_b": [0, 0, 1],
}


def _simul_integerizer(label="PUMA_1"):
    import pandas as pd

    from populationsim.integerizing.simul_integerizer import SimulIntegerizer

    index = pd.RangeIndex(3, name="hh_id")
    incidence = pd.DataFrame(_INCIDENCE, index=index)
    sub_weights = pd.DataFrame(_WEIGHTS, index=index, columns=_ZONE_IDS)
    float_totals = _WEIGHTS.T @ incidence.to_numpy(dtype=np.float64)
    sub_controls = pd.DataFrame(
        np.round(float_totals[:, :3]).astype(np.int64),
        index=_ZONE_IDS,
        columns=["num_hh", "rare", "kind_a"],
    )
    spec = pd.DataFrame(
        {
            "target": list(_INCIDENCE),
            "importance": [1e9, 1000.0, 1000.0, 1000.0],
        }
    )
    return SimulIntegerizer(incidence, sub_weights, sub_controls, spec, "num_hh", label)


def _integer_weights(settings, mode, **extra):
    settings["INTEGERIZER_ROUNDING"] = mode
    settings.update(extra)
    integerizer = _simul_integerizer()
    status = integerizer.integerize()
    assert status in STATUS_SUCCESS
    # (rows, zones)
    return integerizer.integerized_weights.to_numpy()


def test_round_mode_calls_upstream_lp_only(settings, monkeypatch):
    from populationsim.integerizing import controlled_search

    calls = []
    upstream = lp_ortools.np_simul_integerizer_ortools
    controlled = controlled_lp.solve_controlled
    search = controlled_search.solve_search

    def upstream_spy(*args, **kwargs):
        calls.append("upstream")
        return upstream(*args, **kwargs)

    def controlled_spy(*args, **kwargs):
        calls.append("controlled")
        return controlled(*args, **kwargs)

    def search_spy(*args, **kwargs):
        calls.append("search")
        return search(*args, **kwargs)

    monkeypatch.setattr(lp_ortools, "np_simul_integerizer_ortools", upstream_spy)
    monkeypatch.setattr(controlled_lp, "solve_controlled", controlled_spy)
    monkeypatch.setattr(controlled_search, "solve_search", search_spy)
    _integer_weights(settings, "round")
    assert calls == ["upstream"]
    calls.clear()
    # the default controlled method is the search
    _integer_weights(settings, "controlled")
    assert calls == ["search"]
    calls.clear()
    _integer_weights(settings, "controlled", INTEGERIZER_CONTROLLED_METHOD="lp")
    assert calls == ["controlled"]


# the methods of the controlled rounding; None = the default (search)
METHODS = [None, "search", "lp"]


def _method_settings(method):
    return {} if method is None else {"INTEGERIZER_CONTROLLED_METHOD": method}


@pytest.mark.parametrize("method", METHODS)
def test_controlled_integerizer_conserves_parent_weights(settings, method):
    controlled = _integer_weights(settings, "controlled", **_method_settings(method))
    incidence = np.array(list(_INCIDENCE.values()), dtype=np.int64).T
    np.testing.assert_array_equal(controlled.sum(axis=1), [1, 6, 5])
    np.testing.assert_array_equal(controlled.sum(axis=0), [3, 3, 3, 3])
    float_totals = (_WEIGHTS.T @ incidence).sum(axis=0)
    np.testing.assert_allclose(float_totals, [12.0, 1.0, 6.0, 5.0])
    # the integer totals over the zones equal the float totals (up to float noise)
    np.testing.assert_allclose(
        (controlled.T @ incidence).sum(axis=0), float_totals, rtol=0, atol=1e-9
    )

    # contrast: upstream rounds the rare group's cell targets (0.25) to 0 in
    # every zone and loses it
    upstream = _integer_weights(settings, "round")
    np.testing.assert_array_equal(upstream.sum(axis=0), [3, 3, 3, 3])
    assert (upstream.T @ incidence)[:, 1].sum() == 0


@pytest.mark.parametrize("method", METHODS)
def test_controlled_with_draw_uses_the_call_generator(settings, method):
    def draw(seed):
        return _integer_weights(
            settings,
            "controlled",
            INTEGERIZER_DRAW="pareto",
            INTEGERIZER_RANDOM_SEED=seed,
            **_method_settings(method),
        )

    first = draw(1)
    np.testing.assert_array_equal(first, draw(1))
    np.testing.assert_array_equal(first.sum(axis=1), [1, 6, 5])
    outcomes = {draw(seed).tobytes() for seed in range(1, 11)}
    assert len(outcomes) > 1


def _simul_integerizer_for(float_weights, controls, label="PUMA_2"):
    """SimulIntegerizer for ``float_weights`` (zones x rows), only the household
    total control; ``controls`` is the integer household control per zone."""
    import pandas as pd

    from populationsim.integerizing.simul_integerizer import SimulIntegerizer

    float_weights = np.asarray(float_weights, dtype=np.float64)
    zones, rows = float_weights.shape
    index = pd.RangeIndex(rows, name="hh_id")
    zone_ids = list(range(1, zones + 1))
    incidence = pd.DataFrame({"num_hh": np.ones(rows, dtype=np.int64)}, index=index)
    sub_weights = pd.DataFrame(float_weights.T, index=index, columns=zone_ids)
    sub_controls = pd.DataFrame(
        {"num_hh": np.asarray(controls, dtype=np.int64)}, index=zone_ids
    )
    spec = pd.DataFrame({"target": ["num_hh"], "importance": [1e9]})
    return SimulIntegerizer(incidence, sub_weights, sub_controls, spec, "num_hh", label)


@pytest.mark.parametrize("method", ["search", "lp"])
def test_controlled_round_ups_outside_the_method_follow_the_residuals(settings, method):
    """With the draw off the round-ups a zone still needs after the method go to
    rows WITH a residual (largest first), not to rows by array position: zone 2
    keeps its integer part on the row without residual and takes the missing
    household from a row with residual 0.5."""
    settings["INTEGERIZER_ROUNDING"] = "controlled"
    settings["INTEGERIZER_CONTROLLED_METHOD"] = method
    integerizer = _simul_integerizer_for([[0.5, 0.5, 0.0], [0.5, 0.5, 1.0]], [2, 2])
    status = integerizer.integerize()
    assert status in STATUS_SUCCESS
    # (zones, rows)
    result = integerizer.integerized_weights.to_numpy().T
    np.testing.assert_array_equal(result.sum(axis=1), [2, 2])
    np.testing.assert_array_equal(result[0], [1, 1, 0])
    assert result[1, 2] == 1
    assert result[1, :2].sum() == 1
