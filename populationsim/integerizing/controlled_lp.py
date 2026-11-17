# PopulationSim
# See full license in LICENSE.txt.
"""The controlled-rounding LP as sparse arrays, and the solvers that solve it.

:mod:`populationsim.integerizing.controlled_rounding` computes the targets,
bands and prices of the controlled-rounding LP of one parent zone
(:class:`~populationsim.integerizing.controlled_rounding.ControlledProblem`;
its docstring states the priorities). This module writes that LP as plain
arrays (:func:`assemble`), solves the arrays with one of several LP solvers
(:func:`solve_lp`, names in :data:`SOLVERS`) and maps the solution back to the
``(sub-zone, row)`` layout of the simultaneous integerizer
(:func:`solve_relaxation`).

The equality form
-----------------
Every solver gets the same arrays: maximise ``objective . v`` subject to
``matrix v = rhs`` and ``lower <= v <= upper``. Every constraint is an
equality; a range or a soft target becomes an equality with extra variables.
In the notation of ``controlled_rounding`` (``x[z, h]`` in ``[0, 1]``
exists where ``x_allowed``; ``inc`` the sub incidence, ``pinc`` the parent
incidence, ``t`` the household-total column; every slack in
``[0, infinity)``):

cell rows, one per sub-zone ``z`` and sub control ``c != t``
    ``sum_h x[z, h] inc[h, c] + u - o - m = 0``. The band variable ``m``
    has the bounds ``[band_lo, band_hi]``, so the row says "the rounded-up
    incidence plus ``u`` minus ``o`` lies in the band"; ``u`` and ``o``
    cost ``under_price`` and ``over_price``, ``m`` costs nothing. A cell
    without any variable of non-zero incidence whose band contains 0 is
    left out: nothing decides it and it costs nothing.
zone-count rows, one per sub-zone with a variable
    ``sum_h x[z, h] = hh_count[z]`` (hard). A zone without any variable has
    ``hh_count`` 0 (checked) and no row.
conservation rows, one per household row ``h`` with a variable
    ``sum_z x[z, h] + uc_h - oc_h = K_h`` with ``K = conserve_target``;
    both slacks cost ``conserve_penalty``. The rows are there even when that
    price is 0, because the parent totals are written on their slacks.
parent totals, one row per sub control ``c != t`` and per parent control
    The parent total of ``c`` asks ``sum_z sum_h x[z, h] inc[h, c] + U_c -
    O_c = parent_sub_target[c]``. Written on ``x`` this row would hold every
    variable of the parent. The conservation row gives ``sum_z x[z, h] =
    K_h - uc_h + oc_h``; substituted, the same constraint reads::

        sum_h inc[h, c] (oc_h - uc_h) + U_c - O_c
            = parent_sub_target[c] - sum_h inc[h, c] K_h

    with one non-zero per household row instead of one per variable. The sum
    runs over the rows that have a conservation row; a row without any
    variable adds nothing on either side (``x`` cannot move it, and its
    ``K_h`` is 0, checked). The same holds for a parent control ``p`` with
    ``pinc`` and ``parent_target``. ``U`` and ``O`` cost
    ``parent_sub_penalty[c]`` (``parent_penalty[p]``). A parent row with no
    non-zero on a conservation slack and a right-hand side of 0 is left out
    (its slacks would be 0 in every optimum).

The objective puts ``x_coef`` on ``x`` (the household keys and the folded
cell distance) and minus the prices on the slacks. The optimum is that of the
formulation in ``controlled_rounding``: only the parent rows are rewritten,
by an exact substitution.

Variables are stored in blocks (:data:`VARIABLE_BLOCKS`), rows in blocks
(:data:`ROW_BLOCKS`), in that order; :class:`LinearProgram` records the
slices and the index arrays that map every row and slack back to its zone,
household row or control.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np
import scipy.sparse as sp

from populationsim.integerizing.constants import (
    STATUS_FEASIBLE,
    STATUS_OPTIMAL,
    STATUS_SUCCESS,
)

logger = logging.getLogger(__name__)

# the remaining status texts of upstream's vocabulary (lp_ortools STATUS_TEXT)
STATUS_INFEASIBLE = "INFEASIBLE"
STATUS_UNBOUNDED = "UNBOUNDED"
STATUS_ABNORMAL = "ABNORMAL"
STATUS_NOT_SOLVED = "NOT_SOLVED"

# the fork setting that names the solver; quoted in every refusal
SOLVER_SETTING = "INTEGERIZER_CONTROLLED_SOLVER"

# solver names: scipy.optimize.linprog with HiGHS (dual simplex, interior
# point), and OR-Tools pywraplp with Clp, Glop and CBC
SOLVERS = (
    "highs-ds",
    "highs-ipm",
    "clp-dual",
    "clp-primal",
    "clp-barrier",
    "glop-dual",
    "glop-primal",
    "cbc",
)

# pywraplp backend and LP algorithm (MPSolverParameters) per solver name;
# None keeps the backend's default algorithm
_PYWRAPLP = {
    "clp-dual": ("CLP", "DUAL"),
    "clp-primal": ("CLP", "PRIMAL"),
    "clp-barrier": ("CLP", "BARRIER"),
    "glop-dual": ("GLOP", "DUAL"),
    "glop-primal": ("GLOP", "PRIMAL"),
    "cbc": ("CBC", None),
}

# scipy.optimize.linprog status codes (HiGHS methods)
_LINPROG_STATUS = {
    0: STATUS_OPTIMAL,
    1: STATUS_NOT_SOLVED,  # iteration or time limit
    2: STATUS_INFEASIBLE,
    3: STATUS_UNBOUNDED,
}

# an x value strictly between these bounds counts as fractional in the stats
FRACTIONAL_TOLERANCE = 1e-6

VARIABLE_BLOCKS = (
    "x",
    "cell_band",
    "cell_under",
    "cell_over",
    "conserve_under",
    "conserve_over",
    "parent_sub_under",
    "parent_sub_over",
    "parent_under",
    "parent_over",
)

ROW_BLOCKS = ("cell", "zone_count", "conserve", "parent_sub", "parent")


@dataclass(frozen=True, eq=False)
class LinearProgram:
    """One controlled-rounding LP in equality form (module docstring).

    Maximise ``objective . v`` subject to ``matrix v = rhs`` and
    ``lower <= v <= upper``. ``n`` variables, ``m`` rows; the arrays are
    read-only. (``eq=False``: the generated ``__eq__`` cannot compare
    arrays.)

    Attributes
    ----------
    objective : (n,) float
        objective coefficients, maximised
    matrix : scipy.sparse.csr_matrix (m, n)
        equality rows, sorted indices, no explicit zeros
    rhs : (m,) float
    lower, upper : (n,) float
        variable bounds; ``numpy.inf`` for an unbounded slack
    x_index : (Z, H) int64
        variable index of ``x[z, h]``, -1 where no variable exists; the ``x``
        block runs over the allowed cells in row-major ``(z, h)`` order, so
        ``values[blocks["x"]]`` fills ``x_allowed`` in boolean-mask order
    blocks : dict of str to slice
        variable slice per name of :data:`VARIABLE_BLOCKS`, in that order
    row_blocks : dict of str to slice
        row slice per name of :data:`ROW_BLOCKS`, in that order
    cell_zone, cell_control : (k,) int64
        sub-zone and sub control of the k-th cell row; the k-th variable of
        each of the blocks ``cell_band``, ``cell_under`` and ``cell_over``
        belongs to it
    count_zone : (Z',) int64
        sub-zone of each zone-count row
    conserve_row : (R,) int64
        household row of each conservation row, and of the matching variable
        of ``conserve_under`` and ``conserve_over``
    parent_sub_control : (C',) int64
        sub control of each parent-sub row and of its ``parent_sub_under``
        and ``parent_sub_over`` variable
    parent_control : (P',) int64
        parent control of each parent row and of its ``parent_under`` and
        ``parent_over`` variable
    """

    objective: np.ndarray
    matrix: sp.csr_matrix
    rhs: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    x_index: np.ndarray
    blocks: dict
    row_blocks: dict
    cell_zone: np.ndarray
    cell_control: np.ndarray
    count_zone: np.ndarray
    conserve_row: np.ndarray
    parent_sub_control: np.ndarray
    parent_control: np.ndarray


class LpResult(NamedTuple):
    """Outcome of :func:`solve_lp`.

    ``status_text`` in upstream's vocabulary: ``OPTIMAL`` on success,
    otherwise ``INFEASIBLE``, ``UNBOUNDED``, ``ABNORMAL`` or ``NOT_SOLVED``;
    a stop at the time limit is ``NOT_SOLVED``, also where an OR-Tools
    backend reports ``FEASIBLE`` (a feasible point short of the optimum),
    so ``FEASIBLE`` is never returned; ``values`` the ``(n,)`` solution,
    ``None`` unless the
    status is a success; ``objective`` the maximised objective as the solver
    reports it, NaN unless the status is a success; ``solve_seconds`` the
    wall time of the backend call, loading the arrays into the solver's own
    model included (0.0 without variables).
    """

    status_text: str
    values: Optional[np.ndarray]
    objective: float
    solve_seconds: float


def _read_only(array):
    array.setflags(write=False)
    return array


def _check_inputs(problem, sub_incidence, parent_incidence, total_index):
    """Shapes of the LP inputs against ``problem``; ValueError naming them."""
    zone_count, sample_count = problem.x_allowed.shape
    control_count = problem.band_lo.shape[1]
    parent_control_count = problem.parent_target.shape[0]
    expected = {
        "sub_incidence": (sub_incidence, (sample_count, control_count)),
        "parent_incidence": (parent_incidence, (sample_count, parent_control_count)),
    }
    for name, (array, shape) in expected.items():
        if array.shape != shape:
            raise ValueError(
                f"{name} has shape {array.shape}, the problem expects {shape}"
            )
    if (
        isinstance(total_index, (bool, np.bool_))
        or not isinstance(total_index, (int, np.integer))
        or not 0 <= total_index < control_count
    ):
        raise ValueError(
            f"total_hh_sub_control_index must be an integer in "
            f"[0, {control_count}), got {total_index!r}"
        )
    n_avail = problem.x_allowed.sum(axis=1)
    if ((problem.hh_count < 0) | (problem.hh_count > n_avail)).any():
        raise ValueError(
            f"hh_count {problem.hh_count.tolist()} outside [0, rows with a "
            f"residual {n_avail.tolist()}]"
        )


def assemble(
    problem, sub_incidence, parent_incidence, total_hh_sub_control_index
) -> LinearProgram:
    """The controlled-rounding LP of ``problem`` in equality form.

    Parameters
    ----------
    problem : populationsim.integerizing.controlled_rounding.ControlledProblem
    sub_incidence : numpy.ndarray (H, C)
        the incidence ``problem`` was built from (household total included)
    parent_incidence : numpy.ndarray (H, P)
    total_hh_sub_control_index : int
        column of the household total in ``sub_incidence``

    Returns
    -------
    LinearProgram

    Raises
    ------
    ValueError
        when a shape disagrees with ``problem``, ``hh_count`` lies outside
        ``[0, rows with a residual]``, or a row without any variable has a
        non-zero ``conserve_target``
    """
    inc = np.asarray(sub_incidence, dtype=np.float64)
    pinc = np.asarray(parent_incidence, dtype=np.float64)
    t = total_hh_sub_control_index
    _check_inputs(problem, inc, pinc, t)
    x_allowed = np.asarray(problem.x_allowed, dtype=bool)
    zone_count, sample_count = x_allowed.shape
    control_count = inc.shape[1]

    # x variables, row-major over (zone, row)
    x_zone, x_row = np.nonzero(x_allowed)
    x_count = x_zone.size
    x_index = np.full((zone_count, sample_count), -1, dtype=np.int64)
    x_index[x_zone, x_row] = np.arange(x_count)

    # cells: a variable of non-zero incidence, or a band without 0
    nonzero = inc != 0
    hits = (x_allowed.astype(np.int64) @ nonzero.astype(np.int64)) > 0
    keep = hits | (problem.band_lo > 0) | (problem.band_hi < 0)
    keep[:, t] = False
    cell_zone, cell_control = np.nonzero(keep)
    cell_count = cell_zone.size
    cell_of = np.full((zone_count, control_count), -1, dtype=np.int64)
    cell_of[cell_zone, cell_control] = np.arange(cell_count)

    # zone-count rows (zones with a variable) and conservation rows (rows
    # with a variable)
    count_zone = np.flatnonzero(x_allowed.any(axis=1))
    conserve_row = np.flatnonzero(x_allowed.any(axis=0))
    outside = np.ones(sample_count, dtype=bool)
    outside[conserve_row] = False
    stray = outside & (problem.conserve_target != 0)
    if stray.any():
        raise ValueError(
            f"conserve_target is non-zero for {int(stray.sum())} rows without "
            f"any variable (rows {np.flatnonzero(stray)[:10].tolist()}); "
            f"their parent totals cannot be moved to the conservation slacks"
        )
    conserve_target = problem.conserve_target[conserve_row]

    # parent totals on the conservation slacks
    sub_controls = np.delete(np.arange(control_count), t)
    sub_on_rows = inc[np.ix_(conserve_row, sub_controls)]
    sub_rhs = problem.parent_sub_target[sub_controls] - conserve_target @ sub_on_rows
    sub_keep = (sub_on_rows != 0).any(axis=0) | (sub_rhs != 0)
    parent_sub_control = sub_controls[sub_keep]
    sub_on_rows = sub_on_rows[:, sub_keep]
    sub_rhs = sub_rhs[sub_keep]
    parent_on_rows = pinc[conserve_row]
    parent_rhs = problem.parent_target - conserve_target @ parent_on_rows
    parent_keep = (parent_on_rows != 0).any(axis=0) | (parent_rhs != 0)
    parent_control = np.flatnonzero(parent_keep)
    parent_on_rows = parent_on_rows[:, parent_keep]
    parent_rhs = parent_rhs[parent_keep]

    # variable and row layout
    conserve_count = conserve_row.size
    sub_count = parent_sub_control.size
    parent_count = parent_control.size
    blocks = _layout(
        VARIABLE_BLOCKS,
        (
            x_count,
            cell_count,
            cell_count,
            cell_count,
            conserve_count,
            conserve_count,
            sub_count,
            sub_count,
            parent_count,
            parent_count,
        ),
    )
    row_blocks = _layout(
        ROW_BLOCKS,
        (cell_count, count_zone.size, conserve_count, sub_count, parent_count),
    )
    variable_count = blocks[VARIABLE_BLOCKS[-1]].stop
    row_count = row_blocks[ROW_BLOCKS[-1]].stop

    def first(name):
        return blocks[name].start

    triplets = _Triplets()
    # cell rows: x of non-zero incidence (a loop over controls only)
    cell_start = row_blocks["cell"].start
    for c in np.unique(cell_control).tolist():
        control_rows = np.flatnonzero(nonzero[:, c])
        zones = cell_zone[cell_control == c]
        sub_index = x_index[np.ix_(zones, control_rows)]
        zone_pos, row_pos = np.nonzero(sub_index >= 0)
        triplets.add(
            cell_start + cell_of[zones[zone_pos], c],
            sub_index[zone_pos, row_pos],
            inc[control_rows[row_pos], c],
        )
    cell_rows = cell_start + np.arange(cell_count)
    triplets.unit(cell_rows, first("cell_band"), -1.0)
    triplets.unit(cell_rows, first("cell_under"), 1.0)
    triplets.unit(cell_rows, first("cell_over"), -1.0)

    # zone-count rows
    zone_of = np.full(zone_count, -1, dtype=np.int64)
    zone_of[count_zone] = np.arange(count_zone.size)
    x_columns = np.arange(x_count)
    triplets.add(row_blocks["zone_count"].start + zone_of[x_zone], x_columns, 1.0)

    # conservation rows
    conserve_of = np.full(sample_count, -1, dtype=np.int64)
    conserve_of[conserve_row] = np.arange(conserve_count)
    conserve_start = row_blocks["conserve"].start
    triplets.add(conserve_start + conserve_of[x_row], x_columns, 1.0)
    conserve_rows = conserve_start + np.arange(conserve_count)
    triplets.unit(conserve_rows, first("conserve_under"), 1.0)
    triplets.unit(conserve_rows, first("conserve_over"), -1.0)

    # parent rows: incidence * (over - under) of the conservation slacks
    for name, on_rows, count in (
        ("parent_sub", sub_on_rows, sub_count),
        ("parent", parent_on_rows, parent_count),
    ):
        start = row_blocks[name].start
        slack_pos, control_pos = np.nonzero(on_rows)
        weights = on_rows[slack_pos, control_pos]
        triplets.add(start + control_pos, first("conserve_over") + slack_pos, weights)
        triplets.add(start + control_pos, first("conserve_under") + slack_pos, -weights)
        rows = start + np.arange(count)
        triplets.unit(rows, first(name + "_under"), 1.0)
        triplets.unit(rows, first(name + "_over"), -1.0)

    matrix = triplets.csr((row_count, variable_count))

    rhs = np.concatenate(
        [
            np.zeros(cell_count),
            problem.hh_count[count_zone].astype(np.float64),
            conserve_target.astype(np.float64),
            sub_rhs,
            parent_rhs,
        ]
    )

    objective = np.zeros(variable_count)
    lower = np.zeros(variable_count)
    upper = np.full(variable_count, np.inf)
    objective[blocks["x"]] = problem.x_coef[x_zone, x_row]
    upper[blocks["x"]] = 1.0
    lower[blocks["cell_band"]] = problem.band_lo[cell_zone, cell_control]
    upper[blocks["cell_band"]] = problem.band_hi[cell_zone, cell_control]
    objective[blocks["cell_under"]] = -problem.under_price[cell_zone, cell_control]
    objective[blocks["cell_over"]] = -problem.over_price[cell_zone, cell_control]
    objective[blocks["conserve_under"]] = -problem.conserve_penalty
    objective[blocks["conserve_over"]] = -problem.conserve_penalty
    for side in ("_under", "_over"):
        objective[blocks["parent_sub" + side]] = -problem.parent_sub_penalty[
            parent_sub_control
        ]
        objective[blocks["parent" + side]] = -problem.parent_penalty[parent_control]

    return LinearProgram(
        objective=_read_only(objective),
        matrix=matrix,
        rhs=_read_only(rhs),
        lower=_read_only(lower),
        upper=_read_only(upper),
        x_index=_read_only(x_index),
        blocks=blocks,
        row_blocks=row_blocks,
        cell_zone=_read_only(cell_zone.astype(np.int64)),
        cell_control=_read_only(cell_control.astype(np.int64)),
        count_zone=_read_only(count_zone.astype(np.int64)),
        conserve_row=_read_only(conserve_row.astype(np.int64)),
        parent_sub_control=_read_only(parent_sub_control.astype(np.int64)),
        parent_control=_read_only(parent_control.astype(np.int64)),
    )


def _layout(names, sizes):
    """Consecutive slices of the given sizes, keyed by name, in order."""
    slices = {}
    start = 0
    for name, size in zip(names, sizes):
        slices[name] = slice(start, start + int(size))
        start += int(size)
    return slices


class _Triplets:
    """COO triplets of the constraint matrix, collected as arrays."""

    def __init__(self):
        self.rows, self.cols, self.vals = [], [], []

    def add(self, rows, cols, vals):
        rows = np.asarray(rows, dtype=np.int64)
        self.rows.append(rows)
        self.cols.append(np.asarray(cols, dtype=np.int64))
        self.vals.append(
            np.broadcast_to(np.asarray(vals, dtype=np.float64), rows.shape)
        )

    def unit(self, rows, first_column, value):
        """``value`` at (rows[i], first_column + i): one slack per row."""
        self.add(rows, first_column + np.arange(rows.size), value)

    def csr(self, shape):
        def joined(parts, dtype):
            return np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)

        matrix = sp.coo_matrix(
            (
                joined(self.vals, np.float64),
                (joined(self.rows, np.int64), joined(self.cols, np.int64)),
            ),
            shape=shape,
        ).tocsr()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        return matrix


def solve_lp(lp: LinearProgram, solver: str, timeout_seconds) -> LpResult:
    """Solve ``lp`` with the solver named ``solver`` (one of :data:`SOLVERS`).

    ``highs-ds`` and ``highs-ipm`` call ``scipy.optimize.linprog`` (HiGHS
    dual simplex, interior point with crossover; presolve on); the others
    load the arrays into an OR-Tools ``pywraplp`` model (Clp with the dual
    simplex, primal simplex or barrier; Glop with the dual or primal simplex;
    CBC with its default LP algorithm). Every backend runs with the time limit
    ``timeout_seconds`` and without output on stdout. A problem without any
    variable is ``OPTIMAL`` with an empty solution; no solver is called.

    Raises
    ------
    ValueError
        naming ``INTEGERIZER_CONTROLLED_SOLVER``, for an unknown name or a
        backend this environment cannot provide; never a silent substitute
    """
    if solver not in SOLVERS:
        raise ValueError(
            f"{SOLVER_SETTING}: unknown solver {solver!r}, expected one of "
            f"{', '.join(SOLVERS)}"
        )
    if lp.objective.size == 0:
        return LpResult(STATUS_OPTIMAL, np.zeros(0), 0.0, 0.0)
    if solver in _PYWRAPLP:
        return _solve_pywraplp(lp, solver, timeout_seconds)
    return _solve_linprog(lp, solver, timeout_seconds)


def _solve_linprog(lp, solver, timeout_seconds):
    try:
        from scipy.optimize import linprog
    except ImportError as error:
        raise ValueError(
            f"{SOLVER_SETTING}: solver {solver!r} needs scipy.optimize.linprog, "
            f"which this environment cannot import ({error})"
        ) from error
    start = time.perf_counter()
    try:
        result = linprog(
            -lp.objective,  # linprog minimises
            A_eq=lp.matrix,
            b_eq=lp.rhs,
            bounds=np.column_stack([lp.lower, lp.upper]),
            method=solver,
            options={"time_limit": float(timeout_seconds), "presolve": True},
        )
    except ValueError as error:
        if "unknown solver" in str(error).lower():
            raise ValueError(
                f"{SOLVER_SETTING}: solver {solver!r} is not offered by this "
                f"scipy.optimize.linprog ({error})"
            ) from error
        raise
    seconds = time.perf_counter() - start
    status_text = _LINPROG_STATUS.get(result.status, STATUS_ABNORMAL)
    if status_text != STATUS_OPTIMAL or result.x is None:
        return LpResult(status_text, None, math.nan, seconds)
    return LpResult(status_text, np.asarray(result.x), -float(result.fun), seconds)


def _solve_pywraplp(lp, solver, timeout_seconds):
    try:
        from ortools.linear_solver import pywraplp
    except ImportError as error:
        raise ValueError(
            f"{SOLVER_SETTING}: solver {solver!r} needs OR-Tools pywraplp, "
            f"which this environment cannot import ({error})"
        ) from error
    backend, algorithm = _PYWRAPLP[solver]
    start = time.perf_counter()
    model = pywraplp.Solver.CreateSolver(backend)
    if model is None:
        raise ValueError(
            f"{SOLVER_SETTING}: solver {solver!r} needs the OR-Tools backend "
            f"{backend}, which this OR-Tools build does not offer"
        )
    model.SuppressOutput()
    model.set_time_limit(int(math.ceil(timeout_seconds * 1000)))
    infinity = model.infinity()
    variables = [
        model.NumVar(
            -infinity if math.isinf(low) else low,
            infinity if math.isinf(high) else high,
            "",
        )
        for low, high in zip(lp.lower.tolist(), lp.upper.tolist())
    ]
    objective = model.Objective()
    for variable, coefficient in zip(variables, lp.objective.tolist()):
        if coefficient != 0.0:
            objective.SetCoefficient(variable, coefficient)
    objective.SetMaximization()
    indptr = lp.matrix.indptr.tolist()
    indices = lp.matrix.indices.tolist()
    data = lp.matrix.data.tolist()
    for row, value in enumerate(lp.rhs.tolist()):
        constraint = model.Constraint(value, value)
        for position in range(indptr[row], indptr[row + 1]):
            constraint.SetCoefficient(variables[indices[position]], data[position])

    parameters = pywraplp.MPSolverParameters()
    if algorithm is not None:
        parameters.SetIntegerParam(
            pywraplp.MPSolverParameters.LP_ALGORITHM,
            getattr(pywraplp.MPSolverParameters, algorithm),
        )
    status = model.Solve(parameters)
    seconds = time.perf_counter() - start

    status_texts = {
        pywraplp.Solver.OPTIMAL: STATUS_OPTIMAL,
        pywraplp.Solver.FEASIBLE: STATUS_FEASIBLE,
        pywraplp.Solver.INFEASIBLE: STATUS_INFEASIBLE,
        pywraplp.Solver.UNBOUNDED: STATUS_UNBOUNDED,
        pywraplp.Solver.ABNORMAL: STATUS_ABNORMAL,
        pywraplp.Solver.NOT_SOLVED: STATUS_NOT_SOLVED,
    }
    status_text = status_texts.get(status, STATUS_ABNORMAL)
    if status_text == STATUS_FEASIBLE:
        # an LP backend reports FEASIBLE only when it stops at a limit (time
        # or iterations) with a feasible point short of the optimum: a stop
        # at the limit is NOT_SOLVED, never a sub-optimal success
        status_text = STATUS_NOT_SOLVED
    if status_text not in STATUS_SUCCESS:
        return LpResult(status_text, None, math.nan, seconds)
    values = np.array([variable.solution_value() for variable in variables])
    return LpResult(status_text, values, float(objective.Value()), seconds)


def _net_deviation(values, lp, name):
    """Sum over the slack pairs of ``name`` of ``|over - under|``."""
    return float(
        np.abs(
            values[lp.blocks[name + "_over"]] - values[lp.blocks[name + "_under"]]
        ).sum()
    )


def solve_relaxation(
    problem,
    sub_incidence,
    parent_incidence,
    sub_resid_weights,
    total_hh_sub_control_index,
    solver,
    timeout_seconds,
    timing=None,
):
    """The controlled-rounding LP of one parent zone: assemble, solve, map back.

    Parameters
    ----------
    problem : populationsim.integerizing.controlled_rounding.ControlledProblem
    sub_incidence : numpy.ndarray (H, C) float
    parent_incidence : numpy.ndarray (H, P) float
    sub_resid_weights : numpy.ndarray (Z, H) float
        returned unchanged when the solve fails
    total_hh_sub_control_index : int
    solver : str
        one of :data:`SOLVERS`
    timeout_seconds : float
    timing : dict or None
        when a dict is given it is filled with ``build_s`` (assembly) and
        ``solve_s`` (:attr:`LpResult.solve_seconds`) seconds

    Returns
    -------
    resid_weights_out : numpy.ndarray (Z, H) float
        the solved ``x`` (0 where no variable exists); on failure
        ``sub_resid_weights`` unchanged
    status_text : str
        ``OPTIMAL`` on success (also without any variable), otherwise the
        failure status of :class:`LpResult`
    stats : dict
        ``x_count`` (variables ``x``), ``fractional_x`` (solved ``x`` strictly
        between ``FRACTIONAL_TOLERANCE`` and ``1 - FRACTIONAL_TOLERANCE``),
        ``slack_cell``, ``slack_parent_sub``, ``slack_parent``,
        ``slack_conserve`` (per slack pair ``|over - under|``, summed: the
        distance of the cell total to its band, of a parent total to its
        target, of a row's rounded-up count to ``conserve_target``) and
        ``solver``; all but ``x_count`` and ``solver`` are 0 on failure

    Raises
    ------
    ValueError
        for inputs that disagree with ``problem`` (:func:`assemble`) and
        for a solver :func:`solve_lp` refuses
    """
    zone_count, sample_count = problem.x_allowed.shape
    if np.shape(sub_resid_weights) != (zone_count, sample_count):
        raise ValueError(
            f"sub_resid_weights has shape {np.shape(sub_resid_weights)}, the "
            f"problem expects {(zone_count, sample_count)}"
        )
    build_start = time.perf_counter()
    lp = assemble(problem, sub_incidence, parent_incidence, total_hh_sub_control_index)
    build_seconds = time.perf_counter() - build_start
    result = solve_lp(lp, solver, timeout_seconds)
    if timing is not None:
        timing["build_s"] = build_seconds
        timing["solve_s"] = result.solve_seconds

    x_block = lp.blocks["x"]
    stats = {
        "x_count": x_block.stop - x_block.start,
        "fractional_x": 0,
        "slack_cell": 0.0,
        "slack_parent_sub": 0.0,
        "slack_parent": 0.0,
        "slack_conserve": 0.0,
        "solver": solver,
    }
    if result.status_text not in STATUS_SUCCESS:
        return sub_resid_weights, result.status_text, stats

    values = result.values
    x_values = values[x_block]
    resid_weights_out = np.zeros((zone_count, sample_count), dtype=np.float64)
    resid_weights_out[problem.x_allowed] = x_values
    stats["fractional_x"] = int(
        (
            (x_values > FRACTIONAL_TOLERANCE) & (x_values < 1.0 - FRACTIONAL_TOLERANCE)
        ).sum()
    )
    stats["slack_cell"] = _net_deviation(values, lp, "cell")
    stats["slack_parent_sub"] = _net_deviation(values, lp, "parent_sub")
    stats["slack_parent"] = _net_deviation(values, lp, "parent")
    stats["slack_conserve"] = _net_deviation(values, lp, "conserve")
    return resid_weights_out, result.status_text, stats
