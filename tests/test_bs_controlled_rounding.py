# PopulationSim
# See full license in LICENSE.txt.
"""Controlled rounding: targets and costs (pure numpy) and the controlled LP."""

import numpy as np
import pytest

from populationsim.integerizing import controlled_rounding as cr
from populationsim.integerizing import lp_ortools
from populationsim.integerizing.bs_options import PenaltySettings
from populationsim.integerizing.constants import STATUS_OPTIMAL, STATUS_SUCCESS
from populationsim.integerizing.smart_round import smart_round

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


def test_household_count_counts_rows_not_household_total_incidence(caplog):
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
}


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
    resid_out, status, stats = lp_ortools.np_simul_integerizer_controlled_ortools(
        problem,
        inputs["sub_incidence"],
        inputs["parent_incidence"],
        inputs["sub_float_weights"] % 1.0,
        inputs["total_hh_sub_control_index"],
        timeout_in_seconds=60,
        timing=timing,
    )
    assert set(stats) == STAT_KEYS
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


def _conservation_case():
    """Four zones; columns household total, c1, c2, persons; one parent control.
    Rows 0 and 2 have parent weight 1, rows 1 and 3 parent weight 2, row 4 7."""
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
    solved = _controlled(
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


@pytest.mark.xfail(
    strict=True,
    reason="the LP optimum of this case is a fractional vertex (x = 0.5 for "
    "rows 1 and 4 in zones 0 and 1); smart_round rounds each zone on its own "
    "and moves one unit from row 1 to row 4, so the rounded weights lose the "
    "conservation the LP solution has",
)
def test_controlled_rounding_keeps_parent_weight_at_a_fractional_vertex():
    weights, all_incidence, controls, solved = _conservation_case()
    problem, resid_out, status, stats, _ = solved
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
    from ortools.linear_solver import pywraplp

    def no_solver(*args, **kwargs):
        raise AssertionError("the solver must not be built without variables")

    monkeypatch.setattr(pywraplp, "Solver", no_solver)
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
    }
    rounded = _rounded(weights, resid_out, controls)
    np.testing.assert_array_equal(rounded, [[2, 1, 3]])
