# PopulationSim
# See full license in LICENSE.txt.
"""Targets and costs of the controlled rounding (pure numpy, no solver)."""

import numpy as np
import pytest

from populationsim.integerizing import controlled_rounding as cr
from populationsim.integerizing.bs_options import PenaltySettings
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
    # column 0 = household total, column 1 = a control with F = [0.3, 0.4, 0.3];
    # row 1 carries only the household total and has integer parts
    weights = [[0.3, 2.0], [0.4, 1.25], [0.3, 0.75]]
    incidence = [[1, 1], [1, 0]]
    problem = _build(
        weights,
        incidence,
        household_controls=[2, 2, 1],
        parent_incidence=[[1], [0]],
        parent_importance=[500.0],
    )

    np.testing.assert_array_equal(problem.band_lo[:, 1], [0, 0, 0])
    np.testing.assert_array_equal(problem.band_hi[:, 1], [1, 1, 1])
    np.testing.assert_allclose(problem.frac[:, 1], [0.3, 0.4, 0.3], atol=1e-12)
    assert problem.parent_sub_target[1] == 1
    assert problem.parent_target.tolist() == [1]
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

    np.testing.assert_array_equal(problem.hh_count, [2, 0, 4])
    assert problem.hh_count.dtype.kind == "i"
    assert problem.hh_excess == 1
    assert problem.hh_unreachable == 2

    # the count is the number of rows smart_round rounds up
    int_weights = np.array(row_weights).astype(int)
    resid = np.array(row_weights) - int_weights
    for z, control in enumerate(controls):
        added = smart_round(int_weights, resid, control).sum() - int_weights.sum()
        assert added == problem.hh_count[z]


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
