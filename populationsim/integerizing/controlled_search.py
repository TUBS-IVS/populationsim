# PopulationSim
# See full license in LICENSE.txt.
"""Controlled rounding by a network start and a swap search (method ``search``).

What it decides
---------------
The controlled rounding of one parent zone decides, for every allowed cell
``x[z, h]`` of :class:`~populationsim.integerizing.controlled_rounding.ControlledProblem`
(sub-zone ``z``, household row ``h`` with a positive residual), whether the
row is rounded up in that sub-zone: ``x`` is 0 or 1. The module
``controlled_rounding`` states the priorities; this method meets the first two
as hard rules and then lowers the cell distance:

1. every sub-zone gets ``hh_count[z]`` rows rounded up, and every row is
   rounded up in ``conserve_target[h]`` sub-zones (its parent weight is kept),
   as far as both can hold at once;
2. among such tables it lowers the importance-weighted distance of the
   integer cell totals to the float cell totals::

       sum over z and every sub control c but the household total of
           cell_penalty[c] * | sum_h x[z, h] inc[h, c] - goal[z, c] |

   with ``goal = band_lo + frac``, the float cell total in the units of ``x``
   (after subtracting the integer parts).

The parent totals of the LP method are not priced here. Where every row
meets its ``conserve_target``, the integer parent total of every control is
``sum_h (sum_z a[z, h] + conserve_target[h]) * inc[h, c]``; it equals the
float parent total when every row's weight summed over the sub-zones is an
integer (as the balancer leaves it), and can differ from it otherwise.

How
---
Start (:func:`network_start`). A min-cost flow gives an integral table with
the margins of rule 1: source -> row (capacity ``conserve_target[h]``), row
-> zone for every allowed cell (capacity 1), zone -> sink (capacity
``hh_count[z]``). The flow is maximal, so as many round-ups as possible are
placed, and among the maximal flows it has the largest sum of the household
keys (cost ``-key``, keys clipped to ``KEY_CLIP``): the same keys that order
upstream's objective (the seeded draw keys or ``ln`` of the residual, plus
``ln`` of the row's summed residual), so the start follows the draw. A
network flow with integer capacities has an integral optimum, so the start is
0/1 without any repair. Where the margins disagree (more round-ups asked of
the zones than the rows hold, or the reverse) the shortfalls are counted, not
hidden: ``smart_round`` later adds the zones' missing round-ups among the rows
the search left at 0, outside this method's choice. With the draw on it orders
those rows by the draw keys; with the draw off by the input residuals (largest
first, stable order), so a missing round-up goes to a row with a residual
while one is left (``SimulIntegerizer._integerize_controlled``).

Search (:func:`solve_search`). Household ``u`` leaves sub-zone ``a`` for
``b`` and household ``v`` leaves ``b`` for ``a``: such a swap changes neither
a zone's count nor a row's sum, so it keeps rule 1 exactly. The search visits
the zone pairs ``a < b`` in index order. For one pair it ranks the members of
each zone that may move (target cell allowed and still 0) by the change of
the distance if they alone moved (a shortlist of the ``shortlist`` best per
zone), prices every pair of the two shortlists exactly, and applies the best
swap if it lowers the distance by more than ``IMPROVEMENT_TOLERANCE``; it
repeats this for the pair until no swap helps (at most
``MAX_SWAPS_PER_PAIR`` times per pair and sweep, so every loop ends). One
pass over all pairs is a sweep. The search stops after a sweep without any
swap (converged: no shortlisted swap of any pair lowers the distance) or
after ``max_sweeps`` sweeps (not converged, reported).

The result is a local optimum with respect to the swaps the shortlists
contain, not a proven optimum, hence the status ``FEASIBLE``. With a shortlist
at least as large as the largest zone every legal swap is priced and the
result is a local optimum over all swaps. There is no randomness inside the
search and every order is fixed, so equal inputs give equal outputs.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np
from numba import njit

from populationsim.integerizing.constants import STATUS_FEASIBLE, STATUS_OPTIMAL
from populationsim.integerizing.controlled_lp import STATUS_ABNORMAL
from populationsim.integerizing.controlled_rounding import KEY_CLIP

logger = logging.getLogger(__name__)

METHOD = "search"

# The flow costs are integers: -key * COST_SCALE, rounded. Keys within
# 1 / COST_SCALE of each other count as equal.
COST_SCALE = 1_000_000

# A swap is applied only if it lowers the distance by more than this.
IMPROVEMENT_TOLERANCE = 1e-7

# Swaps per zone pair and sweep at most; a pair stopped by the cap is
# visited again in the next sweep (the sweep had swaps, so it is not the
# last).
MAX_SWAPS_PER_PAIR = 200


def _min_cost_flow(tails, heads, capacities, costs, source, sink, supply):
    """Solve a max-flow with min cost with OR-Tools ``SimpleMinCostFlow``.

    ``tails``, ``heads``: node of every arc (int32); ``capacities``, ``costs``:
    int64 per arc; ``supply`` units offered at ``source`` and asked at
    ``sink``. Uses only API present in OR-Tools 9.7 and later (the array
    methods ``add_arcs_with_capacity_and_unit_cost`` and ``flows``); a fresh
    ``SimpleMinCostFlow`` numbers its arcs 0, 1, ... in the order added.

    Returns ``(status_name, arc_flows)``: ``"OPTIMAL"`` and the int64 flow of
    every arc in the order given, or another status name and None.
    """
    try:
        from ortools.graph.python import min_cost_flow
    except ImportError as error:
        raise ValueError(
            "INTEGERIZER_CONTROLLED_METHOD: search needs "
            "ortools.graph.python.min_cost_flow, which the installed OR-Tools "
            "does not provide (%s); install a newer OR-Tools or set "
            "INTEGERIZER_CONTROLLED_METHOD to 'lp'" % error
        ) from error

    flow = min_cost_flow.SimpleMinCostFlow()
    flow.add_arcs_with_capacity_and_unit_cost(tails, heads, capacities, costs)
    flow.set_node_supply(int(source), int(supply))
    flow.set_node_supply(int(sink), -int(supply))
    status = flow.solve_max_flow_with_min_cost()
    if status != flow.OPTIMAL:
        return _status_name(flow, status), None
    arcs = np.arange(len(tails), dtype=np.int32)
    return "OPTIMAL", np.asarray(flow.flows(arcs), dtype=np.int64)


def _status_name(flow, status):
    for name in (
        "NOT_SOLVED",
        "FEASIBLE",
        "INFEASIBLE",
        "UNBALANCED",
        "BAD_RESULT",
        "BAD_COST_RANGE",
    ):
        if status == getattr(flow, name, None):
            return name
    return str(status)


def network_start(problem, keys):
    """The start table of the search: the margins of rule 1, the largest key sum.

    Parameters
    ----------
    problem : populationsim.integerizing.controlled_rounding.ControlledProblem
        reads ``x_allowed``, ``hh_count`` and ``conserve_target``
    keys : numpy.ndarray (Z, H)
        household keys, already combined by the caller (draw key or ``ln r``,
        plus the log of the row's summed residual); clipped here to
        ``[-KEY_CLIP, KEY_CLIP]``; only the allowed cells are read

    Returns
    -------
    x : numpy.ndarray (Z, H) int8
        0 or 1, 1 only where ``x_allowed``; all 0 when the flow fails
    flow_stats : dict
        ``flow_status`` (``"OPTIMAL"`` or the OR-Tools status name),
        ``flow_wanted`` (``min(sum conserve_target, sum hh_count)``),
        ``flow_achieved`` (round-ups placed), ``zone_shortfall``
        (``sum hh_count - flow_achieved``: round-ups the zones still need)
        and ``row_shortfall`` (``sum conserve_target - flow_achieved``:
        parent weight that could not be placed); after a failed flow
        ``flow_achieved`` and both shortfalls are 0
    """
    allowed = np.asarray(problem.x_allowed, dtype=bool)
    zone_count, sample_count = allowed.shape
    targets = np.rint(np.asarray(problem.conserve_target, dtype=np.float64)).astype(
        np.int64
    )
    counts = np.asarray(problem.hh_count).astype(np.int64)
    wanted = int(min(targets.sum(), counts.sum()))
    x = np.zeros((zone_count, sample_count), dtype=np.int8)
    zones, rows = np.nonzero(allowed)

    status = "OPTIMAL"
    if zones.size > 0 and wanted > 0:
        # nodes: source 0, rows 1..H, zones H+1..H+Z, sink H+Z+1
        source = 0
        sink = sample_count + zone_count + 1
        row_nodes = np.arange(1, sample_count + 1, dtype=np.int32)
        zone_nodes = np.arange(sample_count + 1, sink, dtype=np.int32)
        clipped = np.clip(
            np.asarray(keys, dtype=np.float64)[zones, rows], -KEY_CLIP, KEY_CLIP
        )
        cell_costs = np.rint(-clipped * COST_SCALE).astype(np.int64)
        tails = np.concatenate(
            [np.full(sample_count, source, dtype=np.int32), row_nodes[rows], zone_nodes]
        )
        heads = np.concatenate(
            [row_nodes, zone_nodes[zones], np.full(zone_count, sink, dtype=np.int32)]
        )
        capacities = np.concatenate(
            [targets, np.ones(zones.size, dtype=np.int64), counts]
        )
        costs = np.concatenate(
            [
                np.zeros(sample_count, dtype=np.int64),
                cell_costs,
                np.zeros(zone_count, dtype=np.int64),
            ]
        )
        status, arc_flows = _min_cost_flow(
            tails, heads, capacities, costs, source, sink, wanted
        )
        if status == "OPTIMAL":
            x[zones, rows] = arc_flows[sample_count : sample_count + zones.size]

    achieved = int(x.sum(dtype=np.int64))
    failed = status != "OPTIMAL"
    return x, {
        "flow_status": status,
        "flow_wanted": wanted,
        "flow_achieved": achieved,
        "zone_shortfall": 0 if failed else int(counts.sum()) - achieved,
        "row_shortfall": 0 if failed else int(targets.sum()) - achieved,
    }


@njit(cache=True)
def _shortlist(
    zone_from,
    zone_to,
    members,
    n_members,
    x,
    allowed,
    incidence,
    totals,
    goal,
    price,
    picks,
    scores,
):
    """Fill ``picks`` with the member positions of ``zone_from`` whose single
    move to ``zone_to`` lowers the distance most; returns the number of picks.

    The score of a member is the change of the distance if it alone left
    ``zone_from`` for ``zone_to``; it ignores the partner of the swap and only
    ranks the candidates. Ties keep the member order.
    """
    size = picks.shape[0]
    n_controls = incidence.shape[1]
    count = 0
    for i in range(n_members[zone_from]):
        u = members[zone_from, i]
        if x[zone_to, u] or not allowed[zone_to, u]:
            continue
        score = 0.0
        for c in range(n_controls):
            value = incidence[u, c]
            if value != 0.0 and price[c] != 0.0:
                y_from = totals[zone_from, c]
                y_to = totals[zone_to, c]
                score += price[c] * (
                    abs(y_from - value - goal[zone_from, c])
                    - abs(y_from - goal[zone_from, c])
                    + abs(y_to + value - goal[zone_to, c])
                    - abs(y_to - goal[zone_to, c])
                )
        if count < size:
            position = count
            count += 1
        elif score < scores[size - 1]:
            position = size - 1
        else:
            continue
        while position > 0 and scores[position - 1] > score:
            scores[position] = scores[position - 1]
            picks[position] = picks[position - 1]
            position -= 1
        scores[position] = score
        picks[position] = i
    return count


@njit(cache=True)
def _swap_search(
    x, allowed, incidence, goal, price, max_sweeps, shortlist, max_repeat, tolerance
):
    """Lower the distance of ``x`` in place by swaps between zone pairs.

    ``x`` (Z, H) int8 0/1, ``allowed`` (Z, H) bool, ``incidence`` (H, C),
    ``goal`` (Z, C), ``price`` (C,) with 0 for the household total. Returns
    ``(swaps, sweeps, converged)``; ``converged`` is 1 when the last sweep
    applied no swap.
    """
    n_zones, n_rows = x.shape
    n_controls = incidence.shape[1]
    totals = np.zeros((n_zones, n_controls))
    members = np.full((n_zones, n_rows), -1, dtype=np.int64)
    n_members = np.zeros(n_zones, dtype=np.int64)
    for z in range(n_zones):
        for h in range(n_rows):
            if x[z, h]:
                members[z, n_members[z]] = h
                n_members[z] += 1
                for c in range(n_controls):
                    totals[z, c] += incidence[h, c]
    picks_a = np.zeros(shortlist, dtype=np.int64)
    picks_b = np.zeros(shortlist, dtype=np.int64)
    scores_a = np.zeros(shortlist)
    scores_b = np.zeros(shortlist)
    swaps = 0
    sweeps = 0
    improved = 0
    for _sweep in range(max_sweeps):
        sweeps += 1
        improved = 0
        for a in range(n_zones):
            for b in range(a + 1, n_zones):
                for _ in range(max_repeat):
                    count_a = _shortlist(
                        a,
                        b,
                        members,
                        n_members,
                        x,
                        allowed,
                        incidence,
                        totals,
                        goal,
                        price,
                        picks_a,
                        scores_a,
                    )
                    count_b = _shortlist(
                        b,
                        a,
                        members,
                        n_members,
                        x,
                        allowed,
                        incidence,
                        totals,
                        goal,
                        price,
                        picks_b,
                        scores_b,
                    )
                    best = -tolerance
                    best_i = -1
                    best_j = -1
                    for ia in range(count_a):
                        u = members[a, picks_a[ia]]
                        for jb in range(count_b):
                            v = members[b, picks_b[jb]]
                            delta = 0.0
                            for c in range(n_controls):
                                d = incidence[v, c] - incidence[u, c]
                                if d != 0.0 and price[c] != 0.0:
                                    ya = totals[a, c]
                                    yb = totals[b, c]
                                    delta += price[c] * (
                                        abs(ya + d - goal[a, c])
                                        - abs(ya - goal[a, c])
                                        + abs(yb - d - goal[b, c])
                                        - abs(yb - goal[b, c])
                                    )
                            if delta < best:
                                best = delta
                                best_i = picks_a[ia]
                                best_j = picks_b[jb]
                    if best_i < 0:
                        break
                    u = members[a, best_i]
                    v = members[b, best_j]
                    x[a, u] = 0
                    x[b, u] = 1
                    x[b, v] = 0
                    x[a, v] = 1
                    members[a, best_i] = v
                    members[b, best_j] = u
                    for c in range(n_controls):
                        d = incidence[v, c] - incidence[u, c]
                        totals[a, c] += d
                        totals[b, c] -= d
                    improved += 1
        swaps += improved
        if improved == 0:
            break
    converged = 1 if improved == 0 else 0
    return swaps, sweeps, converged


def _distance(x, incidence, goal, price):
    """The search objective of the 0/1 table ``x``."""
    totals = np.asarray(x, dtype=np.float64) @ incidence
    return float((np.abs(totals - goal) * price[np.newaxis, :]).sum())


def _is_count(value, minimum):
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and value >= minimum
    )


def _check_inputs(
    problem,
    sub_incidence,
    keys,
    sub_resid_weights,
    total_index,
    max_sweeps,
    shortlist,
):
    """Shapes and settings against ``problem``; ValueError naming them."""
    zone_count, sample_count = problem.x_allowed.shape
    control_count = problem.band_lo.shape[1]
    expected = {
        "sub_incidence": (sub_incidence, (sample_count, control_count)),
        "keys": (keys, (zone_count, sample_count)),
        "sub_resid_weights": (sub_resid_weights, (zone_count, sample_count)),
    }
    for name, (array, shape) in expected.items():
        if np.shape(array) != shape:
            raise ValueError(
                f"{name} has shape {np.shape(array)}, the problem expects {shape}"
            )
    if not _is_count(total_index, 0) or total_index >= control_count:
        raise ValueError(
            f"total_hh_sub_control_index must be an integer in "
            f"[0, {control_count}), got {total_index!r}"
        )
    if not _is_count(max_sweeps, 1):
        raise ValueError(f"max_sweeps must be a positive integer, got {max_sweeps!r}")
    if not _is_count(shortlist, 1):
        raise ValueError(f"shortlist must be an integer >= 1, got {shortlist!r}")


def solve_search(
    problem,
    sub_incidence,
    keys,
    sub_resid_weights,
    total_hh_sub_control_index,
    max_sweeps,
    shortlist,
    timing=None,
):
    """The controlled rounding of one parent zone by the network start and the
    swap search (module docstring).

    Parameters
    ----------
    problem : populationsim.integerizing.controlled_rounding.ControlledProblem
        reads ``x_allowed``, ``hh_count``, ``conserve_target``, ``band_lo``,
        ``frac``, ``cell_penalty`` and the household counts
    sub_incidence : numpy.ndarray (H, C) float
        the incidence ``problem`` was built from (household total included)
    keys : numpy.ndarray (Z, H) float
        household keys of the start (:func:`network_start`)
    sub_resid_weights : numpy.ndarray (Z, H) float
        returned unchanged when the start fails
    total_hh_sub_control_index : int
        column of the household total, left out of the distance
    max_sweeps : int
        sweeps at most (``INTEGERIZER_SEARCH_MAX_SWEEPS``), >= 1
    shortlist : int
        candidates per zone and pair (``INTEGERIZER_SEARCH_SHORTLIST``), >= 1
    timing : dict or None
        when a dict is given it is filled with ``build_s`` (the network
        start) and ``solve_s`` (the search) seconds

    Returns
    -------
    resid_weights_out : numpy.ndarray (Z, H) float
        the 0/1 table (0 where no cell is allowed); on failure
        ``sub_resid_weights`` unchanged
    status_text : str
        ``FEASIBLE`` for a searched table (a local optimum, not a proven
        optimum), ``OPTIMAL`` when no cell is allowed (nothing to decide),
        ``ABNORMAL`` when the flow of the start ends without an optimum (the
        caller's fallback runs, as after a failed LP)
    stats : dict
        ``method`` (``"search"``), ``x_count`` (allowed cells), ``swaps``,
        ``sweeps``, ``converged`` (1 when the last sweep applied no swap, 0
        when ``max_sweeps`` stopped the search), ``distance_start`` and
        ``distance_final`` (the objective of the start and of the result),
        ``flow_wanted``, ``flow_achieved``, ``zone_shortfall``,
        ``row_shortfall`` (:func:`network_start`), and ``hh_unreachable``,
        ``hh_outside_lp``, ``hh_excess`` of ``problem``. On failure the
        counts of the search and ``flow_achieved`` are 0 and both distances
        NaN.

    Raises
    ------
    ValueError
        naming the argument, for shapes that disagree with ``problem``, a
        household-total column out of range, or ``max_sweeps`` or
        ``shortlist`` not an integer >= 1
    """
    _check_inputs(
        problem,
        sub_incidence,
        keys,
        sub_resid_weights,
        total_hh_sub_control_index,
        max_sweeps,
        shortlist,
    )
    zone_count, sample_count = problem.x_allowed.shape
    x_count = int(np.count_nonzero(problem.x_allowed))
    stats = {
        "method": METHOD,
        "x_count": x_count,
        "swaps": 0,
        "sweeps": 0,
        "converged": 0,
        "distance_start": math.nan,
        "distance_final": math.nan,
        "flow_wanted": 0,
        "flow_achieved": 0,
        "zone_shortfall": 0,
        "row_shortfall": 0,
        "hh_unreachable": problem.hh_unreachable,
        "hh_outside_lp": problem.hh_outside_lp,
        "hh_excess": problem.hh_excess,
    }

    start = time.perf_counter()
    x, flow_stats = network_start(problem, keys)
    build_seconds = time.perf_counter() - start
    flow_status = flow_stats.pop("flow_status")
    stats.update(flow_stats)
    if timing is not None:
        timing["build_s"] = build_seconds
        timing["solve_s"] = 0.0
    if flow_status != "OPTIMAL":
        logger.warning(
            "controlled rounding (search): the min-cost flow of the start ended "
            "%s; the input residuals are handed on",
            flow_status,
        )
        return sub_resid_weights, STATUS_ABNORMAL, stats

    incidence = np.ascontiguousarray(sub_incidence, dtype=np.float64)
    goal = np.asarray(problem.band_lo, dtype=np.float64) + np.asarray(
        problem.frac, dtype=np.float64
    )
    price = np.array(problem.cell_penalty, dtype=np.float64)
    price[total_hh_sub_control_index] = 0.0
    stats["distance_start"] = _distance(x, incidence, goal, price)

    if x_count == 0:
        stats["converged"] = 1
        stats["distance_final"] = stats["distance_start"]
        return np.zeros((zone_count, sample_count)), STATUS_OPTIMAL, stats

    start = time.perf_counter()
    swaps, sweeps, converged = _swap_search(
        x,
        np.array(problem.x_allowed, dtype=np.bool_),
        incidence,
        np.ascontiguousarray(goal),
        price,
        int(max_sweeps),
        int(shortlist),
        MAX_SWAPS_PER_PAIR,
        IMPROVEMENT_TOLERANCE,
    )
    if timing is not None:
        timing["solve_s"] = time.perf_counter() - start
    stats["swaps"] = int(swaps)
    stats["sweeps"] = int(sweeps)
    stats["converged"] = int(converged)
    stats["distance_final"] = _distance(x, incidence, goal, price)
    return x.astype(np.float64), STATUS_FEASIBLE, stats
