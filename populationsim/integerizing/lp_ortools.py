# PopulationSim
# See full license in LICENSE.txt.

import time

import numpy as np
from populationsim.integerizing.constants import (
    STATUS_OPTIMAL,
    STATUS_FEASIBLE,
    STATUS_SUCCESS,
)


def _checked_draw_keys(draw_keys, expected_shape):
    """Validate fork draw keys: expected shape, no NaN or infinite value."""
    keys = np.asarray(draw_keys, dtype=np.float64)
    if keys.shape != expected_shape:
        raise ValueError(
            "draw_keys shape %s does not match the LP variables %s"
            % (keys.shape, expected_shape)
        )
    if not np.isfinite(keys).all():
        raise ValueError(
            "draw_keys: %s non-finite keys out of %s"
            % ((~np.isfinite(keys)).sum(), keys.size)
        )
    return keys


def np_integerizer_ortools(
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
    """
    ortools single-integerizer function taking numpy data types and conforming to a
    standard function signature that allows it to be swapped interchangeably with alternate
    LP implementations.


    Parameters
    ----------
    incidence : numpy.ndarray(control_count, sample_count) float
    resid_weights : numpy.ndarray(sample_count,) float
    log_resid_weights : numpy.ndarray(sample_count,) float
    control_importance_weights : numpy.ndarray(control_count,) float
    total_hh_control_index : int
    lp_right_hand_side : numpy.ndarray(control_count,) float
    relax_ge_upper_bound : numpy.ndarray(control_count,) float
    hh_constraint_ge_bound : numpy.ndarray(control_count,) float
    timeout_in_seconds : int
    timing : dict or None
        fork option (INTEGERIZER_PATH_RECORD): when a dict is given it is filled
        with ``build_s`` (model construction) and ``solve_s`` (solver) seconds.
    draw_keys : numpy.ndarray(sample_count,) float or None
        fork option (INTEGERIZER_DRAW): random order-sampling keys from
        populationsim.integerizing.draw.draw_keys. When given, the household
        term of the objective is x * draw_keys instead of
        x * (log_resid_weights + epsilon * row index); the row-index tie-break
        is omitted because with random keys it would be a row-position bias.
        None (default) keeps the upstream objective.

    Returns
    -------
    resid_weights_out : numpy.ndarray(sample_count,)
    status_text : str
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

    # - Instantiate a mixed-integer solver
    solver = pywraplp.Solver(
        "IntegerizeCbc", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
    )

    # Safe variable construction
    x = [None] * sample_count
    relax_le = [None] * control_count
    relax_ge = [None] * control_count
    hh_constraint_le = [None] * control_count
    hh_constraint_ge = [None] * control_count

    # - Create binary integer variables
    for hh in range(0, sample_count):
        # max_x == 0.0 if float_weights is an int, otherwise 1.0
        max_x = 1.0 - (resid_weights[hh] == 0.0)
        x[hh] = solver.NumVar(0.0, max_x, "x_" + str(hh))

    # - Create positive continuous constraint relaxation variables
    for c in range(0, control_count):
        # no relaxation for total households control
        if c != total_hh_control_index:
            relax_le[c] = solver.NumVar(
                0.0, lp_right_hand_side[c], "relax_le_" + str(c)
            )
            relax_ge[c] = solver.NumVar(
                0.0, relax_ge_upper_bound[c], "relax_ge_" + str(c)
            )

    # - Set objective function coefficients
    # use negative for objective and positive for relaxation penalties since solver is minimizing
    # objective = solver.Objective()
    # for hh in range(sample_count):
    #     objective.SetCoefficient(x[hh], -1.0 * log_resid_weights[hh])
    # for c in range(control_count):
    #     if c != total_hh_control_index:
    #         objective.SetCoefficient(relax_le[c], control_importance_weights[c])
    #         objective.SetCoefficient(relax_ge[c], control_importance_weights[c])

    if draw_keys is None:
        epsilon = 1e-6  # Tiny weight tweak to break symmetry in non-unique cases
        household_term = solver.Sum(
            x[hh] * (log_resid_weights[hh] + epsilon * hh)  # forces unique preference
            for hh in range(sample_count)
        )
    else:
        # fork: seeded order-sampling keys; no row-index tie-break (see docstring)
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

    # - Set objective function
    objective = solver.Maximize(z)  # noqa: F841

    # - inequality constraints
    for c in range(0, control_count):
        # don't add inequality constraints for total households control
        if c == total_hh_control_index:
            continue
        # add the lower bound relaxation inequality constraint
        hh_constraint_le[c] = solver.Constraint(0, lp_right_hand_side[c])
        for hh in range(0, sample_count):
            hh_constraint_le[c].SetCoefficient(x[hh], incidence[c, hh])
            hh_constraint_le[c].SetCoefficient(relax_le[c], -1.0)

        # add the upper bound relaxation inequality constraint
        hh_constraint_ge[c] = solver.Constraint(
            lp_right_hand_side[c], hh_constraint_ge_bound[c]
        )
        for hh in range(0, sample_count):
            hh_constraint_ge[c].SetCoefficient(x[hh], incidence[c, hh])
            hh_constraint_ge[c].SetCoefficient(relax_ge[c], 1.0)

    # using Add and Sum is easier to read but a lot slower
    # for c in range(control_count):
    #     if c == total_hh_control_index:
    #         continue
    #     solver.Add(solver.Sum(x[hh]*incidence[c, hh] for hh in range(sample_count)) - relax_le[c]
    #                >= 0)
    #     solver.Add(solver.Sum(x[hh]*incidence[c, hh] for hh in range(sample_count)) - relax_le[c]
    #                <= lp_right_hand_side[c])
    #     solver.Add(solver.Sum(x[hh]*incidence[c, hh] for hh in range(sample_count)) + relax_ge[c]
    #                >= lp_right_hand_side[c])
    #     solver.Add(solver.Sum(x[hh]*incidence[c, hh] for hh in range(sample_count)) + relax_ge[c]
    #                <= hh_constraint_ge_bound[c])

    # - equality constraint for the total households control
    total_hh_constraint = lp_right_hand_side[total_hh_control_index]
    constraint_eq = solver.Constraint(total_hh_constraint, total_hh_constraint)
    for hh in range(0, sample_count):
        constraint_eq.SetCoefficient(x[hh], 1.0)

    solver.set_time_limit(timeout_in_seconds * 1000)

    solver.EnableOutput()

    solve_start = time.perf_counter()
    result_status = solver.Solve()
    solve_end = time.perf_counter()
    if timing is not None:
        timing["build_s"] = solve_start - build_start
        timing["solve_s"] = solve_end - solve_start

    status_text = STATUS_TEXT[result_status]

    if status_text in STATUS_SUCCESS:
        resid_weights_out = np.asanyarray([x.solution_value() for x in x]).astype(
            np.float64
        )
    else:
        resid_weights_out = resid_weights

    return resid_weights_out, status_text


def np_simul_integerizer_ortools(
    sub_int_weights,
    parent_countrol_importance,
    parent_relax_ge_upper_bound,
    sub_control_importance,
    sub_float_weights,
    sub_resid_weights,
    lp_right_hand_side,
    parent_hh_constraint_ge_bound,
    sub_incidence,
    parent_incidence,
    total_hh_right_hand_side,
    relax_ge_upper_bound,
    parent_lp_right_hand_side,
    hh_constraint_ge_bound,
    parent_resid_weights,
    total_hh_sub_control_index,
    total_hh_parent_control_index,
    timeout_in_seconds,
    draw_keys=None,
    timing=None,
):
    """
    ortools-based siuml-integerizer function taking numpy data types and conforming to a
    standard function signature that allows it to be swapped interchangeably with alternate
    LP implementations.

    Parameters
    ----------
    sub_int_weights : numpy.ndarray(sub_zone_count, sample_count) int
    parent_countrol_importance : numpy.ndarray(parent_control_count,) float
    parent_relax_ge_upper_bound : numpy.ndarray(parent_control_count,) float
    sub_control_importance : numpy.ndarray(sub_control_count,) float
    sub_float_weights : numpy.ndarray(sub_zone_count, sample_count) float
    sub_resid_weights : numpy.ndarray(sub_zone_count, sample_count) float
    lp_right_hand_side : numpy.ndarray(sub_zone_count, sub_control_count) float
    parent_hh_constraint_ge_bound : numpy.ndarray(parent_control_count,) float
    sub_incidence : numpy.ndarray(sample_count, sub_control_count) float
    parent_incidence : numpy.ndarray(sample_count, parent_control_count) float
    total_hh_right_hand_side : numpy.ndarray(sub_zone_count,) float
    relax_ge_upper_bound : numpy.ndarray(sub_zone_count, sub_control_count) float
    parent_lp_right_hand_side : numpy.ndarray(parent_control_count,) float
    hh_constraint_ge_bound : numpy.ndarray(sub_zone_count, sub_control_count) float
    parent_resid_weights : numpy.ndarray(sample_count,) float
    total_hh_sub_control_index : int
    total_hh_parent_control_index : int
    timeout_in_seconds : int
    timing : dict or None
        fork option (INTEGERIZER_PATH_RECORD): when a dict is given it is filled
        with ``build_s`` (model construction) and ``solve_s`` (solver) seconds.
    draw_keys : numpy.ndarray(sub_zone_count, sample_count) float or None
        fork option (INTEGERIZER_DRAW): random order-sampling keys from
        populationsim.integerizing.draw.draw_keys. When given, they replace
        the log sub-zone residual weights in the household term, which becomes
        x[z, hh] * (draw_keys[z, hh] + log_parent_resid_weights[hh]); the
        parent term is unchanged. None (default) keeps the upstream objective.

    Returns
    -------
    resid_weights_out : numpy.ndarray of float
        residual weights in range [0..1] as solved,
        or, in case of failure, sub_resid_weights unchanged
    status_text : string
        STATUS_OPTIMAL, STATUS_FEASIBLE in case of success, or a solver-specific failure status
    """

    from ortools.linear_solver import pywraplp

    STATUS_TEXT = {
        pywraplp.Solver.OPTIMAL: STATUS_OPTIMAL,
        pywraplp.Solver.FEASIBLE: STATUS_FEASIBLE,
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }

    sample_count, sub_control_count = sub_incidence.shape
    _, parent_control_count = parent_incidence.shape
    sub_zone_count, _ = sub_float_weights.shape

    if draw_keys is not None:
        draw_keys = _checked_draw_keys(draw_keys, (sub_zone_count, sample_count))

    # setting indexes to -1 prevents creation of hh_controls relaxation variables
    # setting hh_control importance to zero eliminates them from the objective function
    # the latter approach is used by the cvx version
    # total_hh_sub_control_index = -1
    sub_control_importance[total_hh_sub_control_index] = 0

    # FIXME total_hh_parent_control_index should not exist???
    if total_hh_parent_control_index > 0:
        parent_countrol_importance[total_hh_parent_control_index] = 0

    build_start = time.perf_counter()

    # - Instantiate a mixed-integer solver
    solver = pywraplp.Solver(
        "SimulIntegerizeCbc", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
    )
    solver.EnableOutput()
    solver.set_time_limit(timeout_in_seconds * 1000)

    # x_max is 1.0 unless resid_weights is zero, in which case constrain x to 0.0
    x_max = (~(sub_float_weights == sub_int_weights)).astype(float)

    # - Create resid weight variables
    x = {}
    for z in range(sub_zone_count):
        for hh in range(sample_count):
            x[z, hh] = solver.NumVar(0.0, x_max[z, hh], "x[%s,%s]" % (z, hh))

    # - Create positive continuous constraint relaxation variables
    relax_le = {}
    relax_ge = {}
    for z in range(sub_zone_count):
        for c in range(sub_control_count):
            # no relaxation for total households control
            if c == total_hh_sub_control_index:
                continue
            relax_le[z, c] = solver.NumVar(
                0.0, lp_right_hand_side[z, c], "relax_le[%s,%s]" % (z, c)
            )
            relax_ge[z, c] = solver.NumVar(
                0.0, relax_ge_upper_bound[z, c], "relax_ge[%s,%s]" % (z, c)
            )

    parent_relax_le = {}
    parent_relax_ge = {}
    for c in range(parent_control_count):
        parent_relax_le[c] = solver.NumVar(
            0.0, parent_lp_right_hand_side[c], "parent_relax_le[%s]" % c
        )
        parent_relax_ge[c] = solver.NumVar(
            0.0, parent_relax_ge_upper_bound[c], "parent_relax_ge[%s]" % c
        )

    LOG_OVERFLOW = -725
    log_resid_weights = np.log(np.maximum(sub_resid_weights, np.exp(LOG_OVERFLOW)))
    assert not np.isnan(log_resid_weights).any()

    log_parent_resid_weights = np.log(
        np.maximum(parent_resid_weights, np.exp(LOG_OVERFLOW))
    )
    assert not np.isnan(log_parent_resid_weights).any()

    # fork: with INTEGERIZER_DRAW on, seeded order-sampling keys replace the
    # log sub-zone residual weights; the parent term below stays as upstream
    sub_household_coefficients = log_resid_weights if draw_keys is None else draw_keys

    # - Set objective function coefficients
    z = (
        solver.Sum(
            x[z, hh] * sub_household_coefficients[z, hh]
            for z in range(sub_zone_count)
            for hh in range(sample_count)
        )
        + solver.Sum(
            x[z, hh] * log_parent_resid_weights[hh]
            for hh in range(sample_count)
            for z in range(sub_zone_count)
        )
        - solver.Sum(
            relax_le[z, c] * sub_control_importance[c]
            for z in range(sub_zone_count)
            for c in range(sub_control_count)
            if c != total_hh_sub_control_index
        )
        - solver.Sum(
            relax_ge[z, c] * sub_control_importance[c]
            for z in range(sub_zone_count)
            for c in range(sub_control_count)
            if c != total_hh_sub_control_index
        )
        - solver.Sum(
            parent_relax_le[c] * parent_countrol_importance[c]
            for c in range(parent_control_count)
        )
        - solver.Sum(
            parent_relax_ge[c] * parent_countrol_importance[c]
            for c in range(parent_control_count)
        )
    )

    # - Set objective function
    objective = solver.Maximize(z)  # noqa: F841

    # - sub inequality constraints
    sub_constraint_ge = {}
    sub_constraint_le = {}
    for z in range(sub_zone_count):
        for c in range(sub_control_count):

            # don't add inequality constraints for total households control
            if c == total_hh_sub_control_index:
                continue

            sub_constraint_le[z, c] = solver.Constraint(0, lp_right_hand_side[z, c])
            for hh in range(sample_count):
                sub_constraint_le[z, c].SetCoefficient(x[z, hh], sub_incidence[hh, c])
                sub_constraint_le[z, c].SetCoefficient(relax_le[z, c], -1.0)

            sub_constraint_ge[z, c] = solver.Constraint(
                lp_right_hand_side[z, c], hh_constraint_ge_bound[z, c]
            )
            for hh in range(sample_count):
                sub_constraint_ge[z, c].SetCoefficient(x[z, hh], sub_incidence[hh, c])
                sub_constraint_ge[z, c].SetCoefficient(relax_ge[z, c], 1.0)

    # - equality constraint for the total households control
    constraint_eq = {}
    for z in range(sub_zone_count):
        total_hh_constraint = total_hh_right_hand_side[z]

        constraint_eq[z] = solver.Constraint(total_hh_constraint, total_hh_constraint)
        for hh in range(sample_count):
            constraint_eq[z].SetCoefficient(x[z, hh], 1.0)

    # - sub inequality constraints
    parent_constraint_le = {}
    parent_constraint_ge = {}
    for c in range(parent_control_count):

        if c == total_hh_parent_control_index:
            continue

        parent_constraint_le[c] = solver.Constraint(0, parent_lp_right_hand_side[c])
        parent_constraint_ge[c] = solver.Constraint(
            parent_lp_right_hand_side[c], parent_hh_constraint_ge_bound[c]
        )

        for z in range(sub_zone_count):
            for hh in range(sample_count):
                parent_constraint_le[c].SetCoefficient(
                    x[z, hh], parent_incidence[hh, c]
                )
                parent_constraint_le[c].SetCoefficient(parent_relax_le[c], -1.0)

                parent_constraint_ge[c].SetCoefficient(
                    x[z, hh], parent_incidence[hh, c]
                )
                parent_constraint_ge[c].SetCoefficient(parent_relax_ge[c], 1.0)

    solve_start = time.perf_counter()
    result_status = solver.Solve()
    solve_end = time.perf_counter()
    if timing is not None:
        timing["build_s"] = solve_start - build_start
        timing["solve_s"] = solve_end - solve_start

    status_text = STATUS_TEXT[result_status]

    if status_text in STATUS_SUCCESS:
        resid_weights_out = np.zeros(sub_resid_weights.shape)

        for z in range(sub_zone_count):
            for hh in range(sample_count):
                resid_weights_out[z, hh] = x[z, hh].solution_value()

        resid_weights_out = resid_weights_out.astype(np.float64)
    else:
        resid_weights_out = sub_resid_weights

    return resid_weights_out, status_text


# fork (INTEGERIZER_ROUNDING: controlled): an LP value strictly between these
# bounds counts as fractional in the stats of the controlled LP
FRACTIONAL_TOLERANCE = 1e-6


def _controlled_stats(x_count, fractional_x=0, slacks=None):
    """Stats dict of np_simul_integerizer_controlled_ortools."""
    stats = {
        "x_count": int(x_count),
        "fractional_x": int(fractional_x),
        "slack_cell": 0.0,
        "slack_parent_sub": 0.0,
        "slack_parent": 0.0,
        "slack_conserve": 0.0,
    }
    if slacks is not None:
        for name, variables in slacks.items():
            # clipped at the variable's lower bound 0 (solver noise of ~1e-13)
            stats[name] = float(
                sum(max(var.solution_value(), 0.0) for var in variables)
            )
    return stats


def _checked_controlled_inputs(
    problem, sub_incidence, parent_incidence, sub_resid_weights, total_index
):
    """Shapes of the controlled-LP inputs against the problem; ValueError."""
    zone_count, sample_count = problem.x_allowed.shape
    control_count = problem.band_lo.shape[1]
    parent_control_count = problem.parent_target.shape[0]
    expected = {
        "sub_incidence": (sub_incidence, (sample_count, control_count)),
        "parent_incidence": (parent_incidence, (sample_count, parent_control_count)),
        "sub_resid_weights": (sub_resid_weights, (zone_count, sample_count)),
    }
    for name, (array, shape) in expected.items():
        if np.shape(array) != shape:
            raise ValueError(
                "%s has shape %s, the problem expects %s"
                % (name, np.shape(array), shape)
            )
    if not 0 <= total_index < control_count:
        raise ValueError(
            "total_hh_sub_control_index %s out of range [0, %s)"
            % (total_index, control_count)
        )
    n_avail = problem.x_allowed.sum(axis=1)
    if ((problem.hh_count < 0) | (problem.hh_count > n_avail)).any():
        raise ValueError(
            "hh_count %s outside [0, rows with a residual %s]"
            % (problem.hh_count.tolist(), n_avail.tolist())
        )


def np_simul_integerizer_controlled_ortools(
    problem,
    sub_incidence,
    parent_incidence,
    sub_resid_weights,
    total_hh_sub_control_index,
    timeout_in_seconds,
    timing=None,
):
    """Fork (INTEGERIZER_ROUNDING: controlled): the controlled-rounding LP of
    the simultaneous integerizer for one parent zone.

    The targets, bands and prices come from
    :func:`populationsim.integerizing.controlled_rounding.build_problem`; its
    module docstring states the priorities and why the cell distance is linear.
    In maximisation form, with ``x[z, h]`` in ``[0, 1]`` created only where
    ``problem.x_allowed[z, h]`` and every slack in ``[0, infinity)``:

    - hard, per sub-zone: ``sum_h x[z, h] == hh_count[z]``;
    - per sub-zone and sub control ``c`` (household total excluded): the
      ranged row ``band_lo <= sum_h x * inc + u - o <= band_hi``, objective
      ``- under_price * u - over_price * o``; omitted when no variable of the
      zone has incidence in ``c`` and the band contains 0;
    - per sub control ``c`` (household total excluded):
      ``sum_z sum_h x * inc + U - O == parent_sub_target[c]``, objective
      ``- parent_sub_penalty[c] * (U + O)``;
    - per parent control ``p``:
      ``sum_z sum_h x * pinc + U - O == parent_target[p]``, objective
      ``- parent_penalty[p] * (U + O)``;
    - per row with a variable, unless ``conserve_penalty == 0``:
      ``sum_z x[z, h] + uc - oc == conserve_target[h]``, objective
      ``- conserve_penalty * (uc + oc)``;
    - plus ``sum x * x_coef``.

    A parent-total row without any variable and with target 0 is omitted as
    well (its slacks are 0 in every solution). Coefficients are set only where
    the incidence is non-zero. Solver, time limit and status texts are those of
    :func:`np_simul_integerizer_ortools`. The integer parts are not needed:
    every band and target of ``problem`` is already stated after subtracting
    them.

    Parameters
    ----------
    problem : populationsim.integerizing.controlled_rounding.ControlledProblem
    sub_incidence : numpy.ndarray(sample_count, sub_control_count) float
    parent_incidence : numpy.ndarray(sample_count, parent_control_count) float
    sub_resid_weights : numpy.ndarray(sub_zone_count, sample_count) float
    total_hh_sub_control_index : int
    timeout_in_seconds : int
    timing : dict or None
        when a dict is given it is filled with ``build_s`` and ``solve_s``
        seconds (both 0.0 when there is no variable and no solver is built)

    Returns
    -------
    resid_weights_out : numpy.ndarray(sub_zone_count, sample_count) float
        the solved ``x`` (0 where no variable exists); on failure, and when
        there is no variable at all, ``sub_resid_weights`` unchanged
    status_text : str
        STATUS_OPTIMAL or STATUS_FEASIBLE on success (STATUS_OPTIMAL without
        any variable), otherwise the solver's failure status
    stats : dict
        ``x_count`` (variables), ``fractional_x`` (solved values strictly
        between ``FRACTIONAL_TOLERANCE`` and ``1 - FRACTIONAL_TOLERANCE``) and
        the slack sums ``slack_cell``, ``slack_parent_sub``, ``slack_parent``,
        ``slack_conserve``; all but ``x_count`` are 0 on failure

    Raises
    ------
    ValueError
        when an input shape disagrees with ``problem``
    """
    sub_incidence = np.asarray(sub_incidence, dtype=np.float64)
    parent_incidence = np.asarray(parent_incidence, dtype=np.float64)
    _checked_controlled_inputs(
        problem,
        sub_incidence,
        parent_incidence,
        sub_resid_weights,
        total_hh_sub_control_index,
    )
    x_allowed = problem.x_allowed
    sub_zone_count, sample_count = x_allowed.shape
    sub_control_count = sub_incidence.shape[1]
    parent_control_count = parent_incidence.shape[1]
    x_count = int(x_allowed.sum())

    if x_count == 0:
        # integer weights only: nothing to decide, no solver
        if timing is not None:
            timing["build_s"] = 0.0
            timing["solve_s"] = 0.0
        return sub_resid_weights, STATUS_OPTIMAL, _controlled_stats(0)

    from ortools.linear_solver import pywraplp

    STATUS_TEXT = {
        pywraplp.Solver.OPTIMAL: STATUS_OPTIMAL,
        pywraplp.Solver.FEASIBLE: STATUS_FEASIBLE,
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }

    build_start = time.perf_counter()

    solver = pywraplp.Solver(
        "SimulIntegerizeControlledCbc",
        pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING,
    )
    solver.EnableOutput()
    solver.set_time_limit(timeout_in_seconds * 1000)
    infinity = solver.infinity()
    objective = solver.Objective()

    def slack_pair(name, under_cost, over_cost):
        under = solver.NumVar(0.0, infinity, "u_" + name)
        over = solver.NumVar(0.0, infinity, "o_" + name)
        objective.SetCoefficient(under, -float(under_cost))
        objective.SetCoefficient(over, -float(over_cost))
        return under, over

    # - x variables where the residual is positive; the objective term
    #   x_coef carries the keys and the folded cell distance
    x = {}
    allowed_zones, allowed_rows = np.nonzero(x_allowed)
    for z, hh in zip(allowed_zones.tolist(), allowed_rows.tolist()):
        x[z, hh] = solver.NumVar(0.0, 1.0, "x[%s,%s]" % (z, hh))
        objective.SetCoefficient(x[z, hh], float(problem.x_coef[z, hh]))

    # - hard: rows rounded up per sub-zone (hh_count is 0 without variables)
    zone_rows = [np.nonzero(x_allowed[z])[0] for z in range(sub_zone_count)]
    for z in range(sub_zone_count):
        if zone_rows[z].size == 0:
            continue
        count = float(problem.hh_count[z])
        constraint = solver.Constraint(count, count)
        for hh in zone_rows[z].tolist():
            constraint.SetCoefficient(x[z, hh], 1.0)

    slacks = {
        "slack_cell": [],
        "slack_parent_sub": [],
        "slack_parent": [],
        "slack_conserve": [],
    }

    # - per sub control (household total excluded): ranged cell rows and the
    #   parent total; coefficients only where the incidence is non-zero
    for c in range(sub_control_count):
        if c == total_hh_sub_control_index:
            continue
        control_rows = np.nonzero(sub_incidence[:, c])[0]
        parent_sub_hits = []
        for z in range(sub_zone_count):
            hits = control_rows[x_allowed[z, control_rows]].tolist()
            parent_sub_hits.extend((z, hh) for hh in hits)
            band_lo = float(problem.band_lo[z, c])
            band_hi = float(problem.band_hi[z, c])
            if not hits and band_lo <= 0.0 <= band_hi:
                continue  # nothing to decide in this cell
            under, over = slack_pair(
                "cell[%s,%s]" % (z, c),
                problem.under_price[z, c],
                problem.over_price[z, c],
            )
            slacks["slack_cell"].extend((under, over))
            constraint = solver.Constraint(band_lo, band_hi)
            for hh in hits:
                constraint.SetCoefficient(x[z, hh], float(sub_incidence[hh, c]))
            constraint.SetCoefficient(under, 1.0)
            constraint.SetCoefficient(over, -1.0)

        target = float(problem.parent_sub_target[c])
        if not parent_sub_hits and target == 0.0:
            continue
        price = problem.parent_sub_penalty[c]
        under, over = slack_pair("parent_sub[%s]" % c, price, price)
        slacks["slack_parent_sub"].extend((under, over))
        constraint = solver.Constraint(target, target)
        for z, hh in parent_sub_hits:
            constraint.SetCoefficient(x[z, hh], float(sub_incidence[hh, c]))
        constraint.SetCoefficient(under, 1.0)
        constraint.SetCoefficient(over, -1.0)

    # - per parent control: the parent total
    for p in range(parent_control_count):
        control_rows = np.nonzero(parent_incidence[:, p])[0]
        hit_zones, hit_index = np.nonzero(x_allowed[:, control_rows])
        hit_rows = control_rows[hit_index]
        target = float(problem.parent_target[p])
        if hit_rows.size == 0 and target == 0.0:
            continue
        price = problem.parent_penalty[p]
        under, over = slack_pair("parent[%s]" % p, price, price)
        slacks["slack_parent"].extend((under, over))
        constraint = solver.Constraint(target, target)
        for z, hh in zip(hit_zones.tolist(), hit_rows.tolist()):
            constraint.SetCoefficient(x[z, hh], float(parent_incidence[hh, p]))
        constraint.SetCoefficient(under, 1.0)
        constraint.SetCoefficient(over, -1.0)

    # - per row with a variable: conservation of its summed residual
    if problem.conserve_penalty > 0:
        for hh in np.nonzero(x_allowed.any(axis=0))[0].tolist():
            under, over = slack_pair(
                "conserve[%s]" % hh, problem.conserve_penalty, problem.conserve_penalty
            )
            slacks["slack_conserve"].extend((under, over))
            target = float(problem.conserve_target[hh])
            constraint = solver.Constraint(target, target)
            for z in np.nonzero(x_allowed[:, hh])[0].tolist():
                constraint.SetCoefficient(x[z, hh], 1.0)
            constraint.SetCoefficient(under, 1.0)
            constraint.SetCoefficient(over, -1.0)

    objective.SetMaximization()

    solve_start = time.perf_counter()
    result_status = solver.Solve()
    solve_end = time.perf_counter()
    if timing is not None:
        timing["build_s"] = solve_start - build_start
        timing["solve_s"] = solve_end - solve_start

    status_text = STATUS_TEXT[result_status]

    if status_text not in STATUS_SUCCESS:
        return sub_resid_weights, status_text, _controlled_stats(x_count)

    resid_weights_out = np.zeros((sub_zone_count, sample_count), dtype=np.float64)
    for (z, hh), var in x.items():
        resid_weights_out[z, hh] = var.solution_value()
    solved = resid_weights_out[x_allowed]
    fractional_x = int(
        ((solved > FRACTIONAL_TOLERANCE) & (solved < 1.0 - FRACTIONAL_TOLERANCE)).sum()
    )
    return (
        resid_weights_out,
        status_text,
        _controlled_stats(x_count, fractional_x, slacks),
    )
