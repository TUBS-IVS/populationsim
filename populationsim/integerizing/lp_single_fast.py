# PopulationSim
# See full license in LICENSE.txt.
"""Fast build of the single-integerizer OR-Tools model (setting ``FAST_PATHS``).

Builds the same model as ``lp_ortools.np_integerizer_ortools`` and solves it the
same way, but calls ``SetCoefficient`` only for the non-zero incidence values.
Upstream calls it for every household of every control (about
4 x controls x households calls), although almost all incidence values of the
seed level are zero.

Why the model is the same (checked by ``tests/test_bs_fast_paths.py`` on the
LP and MPS exports and on the serialised model proto):

- ``MPConstraint::SetCoefficient`` returns without storing anything when the
  coefficient is 0.0 and the variable has no coefficient yet in that constraint.
  Upstream never sets a non-zero coefficient back to zero, so its constraints
  hold exactly the non-zero entries; the zero calls are no-ops.
- Variables and constraints are created in upstream's order with the same
  names, bounds and objective (the objective lines are copied unchanged), so
  every index is the same.
- The non-zero coefficients of a constraint are inserted in upstream's order:
  the households in ascending order, and the relaxation variable directly after
  household 0 when that household is non-zero, otherwise first (upstream sets it
  after household 0 in the first loop pass and only overwrites it with the same
  value later). With the same keys inserted in the same order the coefficient
  map of each constraint is built by the same sequence of operations. Its
  iteration order, which is the order in which the solver interface reads a
  row, also depends on the variable addresses, which differ between any two
  builds, upstream's included.
"""

from __future__ import annotations

import time

import numpy as np

from populationsim.integerizing.constants import STATUS_SUCCESS
from populationsim.integerizing.lp_ortools import _checked_draw_keys


def _set_row(constraint, x, nonzero_hh, nonzero_values, relax_var, relax_coefficient):
    """Insert the non-zero coefficients of one constraint in upstream's order."""
    rest = 0
    if len(nonzero_hh) and nonzero_hh[0] == 0:
        constraint.SetCoefficient(x[0], nonzero_values[0])
        rest = 1
    constraint.SetCoefficient(relax_var, relax_coefficient)
    for hh, value in zip(nonzero_hh[rest:], nonzero_values[rest:]):
        constraint.SetCoefficient(x[hh], value)


def _build_model(
    incidence,
    resid_weights,
    log_resid_weights,
    control_importance_weights,
    total_hh_control_index,
    lp_right_hand_side,
    relax_ge_upper_bound,
    hh_constraint_ge_bound,
    timeout_in_seconds,
    draw_keys=None,
):
    """Build the model of ``np_integerizer_ortools`` and return ``(solver, x)``.

    ``draw_keys`` must already be checked. The solver is returned ready to
    solve (time limit and output set as upstream does before ``Solve``).
    """
    from ortools.linear_solver import pywraplp

    control_count, sample_count = incidence.shape

    # - Instantiate a mixed-integer solver
    solver = pywraplp.Solver(
        "IntegerizeCbc", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
    )

    x = [None] * sample_count
    relax_le = [None] * control_count
    relax_ge = [None] * control_count

    # - Create binary integer variables (as upstream)
    for hh in range(0, sample_count):
        max_x = 1.0 - (resid_weights[hh] == 0.0)
        x[hh] = solver.NumVar(0.0, max_x, "x_" + str(hh))

    # - Create positive continuous constraint relaxation variables (as upstream)
    for c in range(0, control_count):
        if c != total_hh_control_index:
            relax_le[c] = solver.NumVar(
                0.0, lp_right_hand_side[c], "relax_le_" + str(c)
            )
            relax_ge[c] = solver.NumVar(
                0.0, relax_ge_upper_bound[c], "relax_ge_" + str(c)
            )

    # - Objective (copied unchanged from upstream)
    if draw_keys is None:
        epsilon = 1e-6  # Tiny weight tweak to break symmetry in non-unique cases
        household_term = solver.Sum(
            x[hh] * (log_resid_weights[hh] + epsilon * hh)  # forces unique preference
            for hh in range(sample_count)
        )
    else:
        household_term = solver.Sum(x[hh] * draw_keys[hh] for hh in range(sample_count))
    z = (
        household_term
        - solver.Sum(
            relax_le[c] * control_importance_weights[c]
            for c in range(control_count)
            if c != total_hh_control_index
        )
        - solver.Sum(
            relax_ge[c] * control_importance_weights[c]
            for c in range(control_count)
            if c != total_hh_control_index
        )
    )
    solver.Maximize(z)

    # - inequality constraints, non-zero coefficients only. Upstream sets the
    # relaxation coefficient inside the household loop, so with no household it
    # never sets it; the same here.
    for c in range(0, control_count):
        if c == total_hh_control_index:
            continue
        row = incidence[c]
        nonzero_hh = np.flatnonzero(row).tolist()
        nonzero_values = row[nonzero_hh].tolist()

        hh_constraint_le = solver.Constraint(0, lp_right_hand_side[c])
        if sample_count:
            _set_row(hh_constraint_le, x, nonzero_hh, nonzero_values, relax_le[c], -1.0)

        hh_constraint_ge = solver.Constraint(
            lp_right_hand_side[c], hh_constraint_ge_bound[c]
        )
        if sample_count:
            _set_row(hh_constraint_ge, x, nonzero_hh, nonzero_values, relax_ge[c], 1.0)

    # - equality constraint for the total households control (as upstream)
    total_hh_constraint = lp_right_hand_side[total_hh_control_index]
    constraint_eq = solver.Constraint(total_hh_constraint, total_hh_constraint)
    for hh in range(0, sample_count):
        constraint_eq.SetCoefficient(x[hh], 1.0)

    solver.set_time_limit(timeout_in_seconds * 1000)
    solver.EnableOutput()
    return solver, x


def np_integerizer_ortools_fast(
    incidence,
    resid_weights,
    log_resid_weights,
    control_importance_weights,
    total_hh_control_index,
    lp_right_hand_side,
    relax_ge_upper_bound,
    hh_constraint_ge_bound,
    timeout_in_seconds,
    draw_keys=None,
    timing=None,
):
    """``lp_ortools.np_integerizer_ortools`` with the fast model build.

    Same parameters and return value as the upstream function (see there).
    """
    from ortools.linear_solver import pywraplp

    STATUS_TEXT = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }

    control_count, sample_count = incidence.shape

    if draw_keys is not None:
        draw_keys = _checked_draw_keys(draw_keys, (sample_count,))

    build_start = time.perf_counter()
    solver, x = _build_model(
        incidence,
        resid_weights,
        log_resid_weights,
        control_importance_weights,
        total_hh_control_index,
        lp_right_hand_side,
        relax_ge_upper_bound,
        hh_constraint_ge_bound,
        timeout_in_seconds,
        draw_keys=draw_keys,
    )

    solve_start = time.perf_counter()
    result_status = solver.Solve()
    solve_end = time.perf_counter()
    if timing is not None:
        timing["build_s"] = solve_start - build_start
        timing["solve_s"] = solve_end - solve_start

    status_text = STATUS_TEXT[result_status]

    if status_text in STATUS_SUCCESS:
        resid_weights_out = np.asanyarray([v.solution_value() for v in x]).astype(
            np.float64
        )
    else:
        resid_weights_out = resid_weights

    return resid_weights_out, status_text
