# PopulationSim
# See full license in LICENSE.txt.
"""Fork test helper: the dense controlled-rounding LP, an independent reference.

This is the first formulation of the controlled-rounding LP: every constraint
is written directly on the ``x`` variables (ranged cell rows, parent totals as
dense rows over every variable of the parent), built through the OR-Tools
``pywraplp`` API and solved with CBC. It is too slow for production and was
replaced by the sparse equality form of
:mod:`populationsim.integerizing.controlled_lp`; it stays here, unchanged in
substance, so that the tests can show that both formulations reach the same
optimal objective. Not a test module (pytest does not collect it).
"""

from __future__ import annotations

import numpy as np

from populationsim.integerizing.constants import STATUS_FEASIBLE, STATUS_OPTIMAL


def solve_dense_reference(
    problem,
    sub_incidence,
    parent_incidence,
    sub_resid_weights,
    total_hh_sub_control_index,
    timeout_seconds,
):
    """Solve the controlled-rounding LP of ``problem`` in its dense form.

    Maximise, with ``x[z, h]`` in ``[0, 1]`` created only where
    ``problem.x_allowed[z, h]`` and every slack in ``[0, infinity)``:

    - hard, per sub-zone: ``sum_h x[z, h] == hh_count[z]``;
    - per sub-zone and sub control ``c`` (household total excluded): the
      ranged row ``band_lo <= sum_h x * inc + u - o <= band_hi``, objective
      ``- under_price * u - over_price * o``; omitted when no variable of the
      zone has incidence in ``c`` and the band contains 0;
    - per sub control ``c`` (household total excluded):
      ``sum_z sum_h x * inc + U - O == parent_sub_target[c]``, objective
      ``- parent_sub_penalty[c] * (U + O)``; omitted without any variable of
      non-zero incidence and with target 0;
    - per parent control ``p``:
      ``sum_z sum_h x * pinc + U - O == parent_target[p]``, objective
      ``- parent_penalty[p] * (U + O)``; omitted as the sub-control rows;
    - per row with a variable, unless ``conserve_penalty == 0``:
      ``sum_z x[z, h] + uc - oc == conserve_target[h]``, objective
      ``- conserve_penalty * (uc + oc)``;
    - plus ``sum x * x_coef``.

    Returns
    -------
    x_values : numpy.ndarray (Z, H)
        the solved ``x`` (0 where no variable exists); on failure, and when
        there is no variable at all, ``sub_resid_weights`` unchanged
    status_text : str
        upstream's status vocabulary (``OPTIMAL`` without any variable)
    objective_value : float
        the maximised objective; 0.0 without any variable, NaN on failure
    """
    from ortools.linear_solver import pywraplp

    sub_incidence = np.asarray(sub_incidence, dtype=np.float64)
    parent_incidence = np.asarray(parent_incidence, dtype=np.float64)
    x_allowed = problem.x_allowed
    sub_zone_count, sample_count = x_allowed.shape
    sub_control_count = sub_incidence.shape[1]
    parent_control_count = parent_incidence.shape[1]
    if not x_allowed.any():
        return sub_resid_weights, STATUS_OPTIMAL, 0.0

    status_texts = {
        pywraplp.Solver.OPTIMAL: STATUS_OPTIMAL,
        pywraplp.Solver.FEASIBLE: STATUS_FEASIBLE,
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }
    solver = pywraplp.Solver(
        "DenseControlledReference", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
    )
    solver.SuppressOutput()
    solver.set_time_limit(int(timeout_seconds * 1000))
    infinity = solver.infinity()
    objective = solver.Objective()

    def slack_pair(name, under_cost, over_cost):
        under = solver.NumVar(0.0, infinity, "u_" + name)
        over = solver.NumVar(0.0, infinity, "o_" + name)
        objective.SetCoefficient(under, -float(under_cost))
        objective.SetCoefficient(over, -float(over_cost))
        return under, over

    def equality(target, terms, under, over):
        constraint = solver.Constraint(target, target)
        for var, coefficient in terms:
            constraint.SetCoefficient(var, coefficient)
        constraint.SetCoefficient(under, 1.0)
        constraint.SetCoefficient(over, -1.0)

    x = {}
    allowed_zones, allowed_rows = np.nonzero(x_allowed)
    for z, hh in zip(allowed_zones.tolist(), allowed_rows.tolist()):
        x[z, hh] = solver.NumVar(0.0, 1.0, "x[%s,%s]" % (z, hh))
        objective.SetCoefficient(x[z, hh], float(problem.x_coef[z, hh]))

    # hard: rows rounded up per sub-zone
    for z in range(sub_zone_count):
        zone_rows = np.nonzero(x_allowed[z])[0].tolist()
        if not zone_rows:
            continue
        count = float(problem.hh_count[z])
        constraint = solver.Constraint(count, count)
        for hh in zone_rows:
            constraint.SetCoefficient(x[z, hh], 1.0)

    # per sub control (household total excluded): ranged cells, parent total
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
                continue
            under, over = slack_pair(
                "cell[%s,%s]" % (z, c),
                problem.under_price[z, c],
                problem.over_price[z, c],
            )
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
        terms = [(x[z, hh], float(sub_incidence[hh, c])) for z, hh in parent_sub_hits]
        equality(target, terms, under, over)

    # per parent control: the parent total
    for p in range(parent_control_count):
        control_rows = np.nonzero(parent_incidence[:, p])[0]
        hit_zones, hit_index = np.nonzero(x_allowed[:, control_rows])
        hit_rows = control_rows[hit_index]
        target = float(problem.parent_target[p])
        if hit_rows.size == 0 and target == 0.0:
            continue
        price = problem.parent_penalty[p]
        under, over = slack_pair("parent[%s]" % p, price, price)
        terms = [
            (x[z, hh], float(parent_incidence[hh, p]))
            for z, hh in zip(hit_zones.tolist(), hit_rows.tolist())
        ]
        equality(target, terms, under, over)

    # per row with a variable: conservation of its summed residual
    if problem.conserve_penalty > 0:
        for hh in np.nonzero(x_allowed.any(axis=0))[0].tolist():
            price = problem.conserve_penalty
            under, over = slack_pair("conserve[%s]" % hh, price, price)
            terms = [(x[z, hh], 1.0) for z in np.nonzero(x_allowed[:, hh])[0].tolist()]
            equality(float(problem.conserve_target[hh]), terms, under, over)

    objective.SetMaximization()
    status_text = status_texts[solver.Solve()]
    if status_text not in (STATUS_OPTIMAL, STATUS_FEASIBLE):
        return sub_resid_weights, status_text, float("nan")

    x_values = np.zeros((sub_zone_count, sample_count), dtype=np.float64)
    for (z, hh), var in x.items():
        x_values[z, hh] = var.solution_value()
    return x_values, status_text, float(objective.Value())
