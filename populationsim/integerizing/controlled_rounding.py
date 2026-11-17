# PopulationSim
# See full license in LICENSE.txt.
"""Targets and costs of the controlled rounding of the simultaneous integerizer.

What the controlled rounding does
---------------------------------
The simultaneous integerizer turns the float weights ``w[z, h]`` of the rows
``h`` (households or household groups) in the sub-zones ``z`` of one parent
zone into integers. Every weight keeps its integer part ``a = floor(w)``; for
each positive residual ``r = w - a`` a variable ``x[z, h]`` in ``[0, 1]``
decides whether the row is rounded up in that sub-zone. The integer cell total
of control ``c`` is then ``y[z, c] = A[z, c] + sum_h x[z, h] * inc[h, c]``
with ``A = a @ inc``; the float cell total is ``F = w @ inc``.

Upstream's LP asks every cell total to reach ``round(F)`` with one-sided,
bounded slacks; where that is impossible the LP is infeasible and the
integerizer falls back to sequential integerizing. The controlled rounding
(``INTEGERIZER_ROUNDING: controlled``) solves a different LP that is feasible
by construction, with this order of priorities:

1. hard: the number of rows rounded up in each sub-zone is the number the
   final ``smart_round`` enforces anyway, ``k[z] = clip(N[z] - sum_h a[z, h],
   0, n_avail[z])`` (``N`` the integer household control, ``n_avail`` the
   rows with a positive residual; ``smart_round`` counts rows, so the integer
   parts are summed as rows, not weighted by the household-total incidence);
2. conservation: each row keeps its weight summed over the sub-zones,
   ``sum_z x[z, h] = round(sum_z r[z, h])``;
3. parent totals: of every sub control (``round(sum_z F)``) and of every
   parent control (``round(sum_h (sum_z w) * pinc)``);
4. cell distance: the importance-weighted distance ``|y - F|`` of each integer
   cell total to its float total (household total excluded: rule 1 fixes it);
5. the household keys (draw key or ``ln r``, plus the log of the row's
   residual summed over the sub-zones, as upstream) only break the remaining
   ties.

Goals 2 to 4 are soft: penalised slacks without bounds, so no target can make
the LP infeasible. Their order comes from the penalty tiers of
:class:`populationsim.integerizing.bs_options.PenaltySettings`: the cell
penalty is the control importance clipped to ``[floor, cap]``, a parent total
costs ``parent_total_factor`` times that (at most ``cap``), conservation costs
``conserve_penalty``. The keys are clipped to ``[-KEY_CLIP, KEY_CLIP]`` so
that a penalty floor above that range always outranks them.

This module computes the targets, bands and prices of that LP without a
solver (:func:`build_problem`); the LP itself is built from
:class:`ControlledProblem` and solved in ``controlled_lp``.

The cell distance as a linear cost
----------------------------------
For one cell with float total ``F`` (snapped, see below), ``lo = floor(F)``,
``hi = ceil(F)``, ``f = F - lo`` and penalty ``q``, the ranged row
``lo <= y + u - o <= hi`` with unbounded slacks ``u, o >= 0`` and the cost::

    q * (1 - 2f) * (y - lo) + 2q(1 - f) * u + 2q f * o

equals ``q * |y - F| - q * f`` for every integer ``y`` (with the smallest
slacks, which a minimising LP picks). The constant ``q * f`` does not change
the optimum. Because ``y`` is linear in ``x``, the first term folds into the
objective coefficient of ``x`` (:attr:`ControlledProblem.x_coef`); the slack
prices are :attr:`ControlledProblem.under_price` and ``over_price``. The
equality holds for integer ``y``, i.e. for integer incidence and integer
``x``; at a fractional LP vertex the cost is that of the LP relaxation.

Units: the LP variables are the rounded-up residuals ``x``, so every band and
target here is stated after subtracting the integer part (``A`` for the
cells, ``sum_z a`` for the parent and conservation rows).

Float noise: the balancer rescales every row's sub-zone weights to the
parent's integer weight, so many float totals are integers up to rounding
noise. A total within ``SNAP_TOLERANCE`` of an integer is snapped to it, so
that ``2.9999999997`` gives the one-point band ``[3, 3]`` and not ``[2, 3]``;
the parent and conservation targets are rounded to integers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from populationsim.integerizing.bs_options import PenaltySettings

logger = logging.getLogger(__name__)

# A float total within this distance of an integer is treated as that integer.
SNAP_TOLERANCE = 1e-6

# Household keys enter the objective clipped to [-KEY_CLIP, KEY_CLIP]; the
# penalty floor is meant to lie above this range, so that no key difference
# outranks a control.
KEY_CLIP = 30.0


@dataclass(frozen=True, eq=False)
class ControlledProblem:
    """Targets, bands and prices of one controlled-rounding LP.

    Shapes: ``Z`` sub-zones, ``H`` rows, ``C`` sub controls (including the
    household total), ``P`` parent controls (may be 0). Bands and targets are
    in units of ``x``, i.e. after subtracting the integer part. The arrays are
    read-only. (``eq=False``: the generated ``__eq__`` cannot compare arrays.)

    Attributes
    ----------
    x_allowed : (Z, H) bool
        a variable ``x[z, h]`` exists where the residual ``r[z, h] > 0``
    hh_count : (Z,) int
        rows rounded up per sub-zone (hard), ``clip(N - sum_h a, 0, n_avail)``
        (``smart_round`` counts rows, not household-total incidence)
    band_lo, band_hi : (Z, C) float
        ``floor(F') - A`` and ``ceil(F') - A``, clipped at 0
    frac : (Z, C) float
        ``F' - floor(F')`` in ``[0, 1)``
    under_price, over_price : (Z, C) float
        slack prices of the ranged cell row, ``2 q (1 - frac)`` and
        ``2 q frac``
    cell_penalty : (C,) float
        ``q = clip(importance, floor, cap)``
    parent_sub_target : (C,) float
        ``round(sum_z F) - sum_z A``
    parent_sub_penalty : (C,) float
        ``min(parent_total_factor * q, cap)``
    parent_target : (P,) float
        ``round(sum_h (sum_z w) * pinc) - sum_h (sum_z a) * pinc``
    parent_penalty : (P,) float
        ``min(parent_total_factor * clip(importance, floor, cap), cap)``
    conserve_target : (H,) float
        ``round(sum_z r)``, an integer
    conserve_penalty : float
        price of one unit of conservation slack
    x_coef : (Z, H) float
        objective coefficient of ``x`` in maximisation form: the clipped key
        minus ``sum over c != t of q_c (1 - 2 frac) inc[h, c]``; only the
        entries where ``x_allowed`` enter the LP
    hh_unreachable : int
        households the integer controls ask for that ``smart_round`` never
        adds, ``sum max(N - sum_h a - H, 0)``: its shortfall is clipped at
        the ``H`` rows of the sub-zone
    hh_outside_lp : int
        households ``smart_round`` adds among rows without a residual, outside
        the LP's choice, ``sum (min(max(N - sum_h a, 0), H) - hh_count)``
    hh_excess : int
        households the integer parts already hold beyond the integer
        controls, ``sum max(sum_h a - N, 0)``; no rounding can remove them

    The household-total column ``t`` is present in every ``(Z, C)`` and
    ``(C,)`` array but unused by the LP: rule 1 fixes the count per sub-zone.
    """

    x_allowed: np.ndarray
    hh_count: np.ndarray
    band_lo: np.ndarray
    band_hi: np.ndarray
    frac: np.ndarray
    under_price: np.ndarray
    over_price: np.ndarray
    cell_penalty: np.ndarray
    parent_sub_target: np.ndarray
    parent_sub_penalty: np.ndarray
    parent_target: np.ndarray
    parent_penalty: np.ndarray
    conserve_target: np.ndarray
    conserve_penalty: float
    x_coef: np.ndarray
    hh_unreachable: int
    hh_outside_lp: int
    hh_excess: int


def _snap(values):
    """Values within ``SNAP_TOLERANCE`` of an integer, set to that integer."""
    values = np.asarray(values, dtype=np.float64)
    nearest = np.round(values)
    return np.where(np.abs(values - nearest) <= SNAP_TOLERANCE, nearest, values)


def _checked(name, value, shape, allow_inf=False):
    """``value`` as a float64 array of ``shape`` (``None`` = any length)."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape)
    ):
        wanted = tuple("any" if expected is None else expected for expected in shape)
        raise ValueError(f"{name} has shape {array.shape}, expected {wanted}")
    bad = np.isnan(array) if allow_inf else ~np.isfinite(array)
    if bad.any():
        kind = "NaN" if allow_inf else "non-finite"
        raise ValueError(f"{name}: {int(bad.sum())} {kind} values out of {array.size}")
    return array


def _read_only(array):
    array.setflags(write=False)
    return array


def build_problem(
    sub_float_weights,
    sub_incidence,
    parent_incidence,
    sub_importance,
    parent_importance,
    household_controls,
    total_hh_sub_control_index,
    keys,
    penalties: PenaltySettings,
) -> ControlledProblem:
    """Targets, bands and prices of the controlled-rounding LP of one parent.

    Parameters
    ----------
    sub_float_weights : numpy.ndarray (Z, H)
        balanced float weights, one row per sub-zone, non-negative
    sub_incidence : numpy.ndarray (H, C)
        incidence of the sub controls (household total included)
    parent_incidence : numpy.ndarray (H, P)
        incidence of the parent controls not among the sub controls; P may be 0
    sub_importance : numpy.ndarray (C,)
    parent_importance : numpy.ndarray (P,)
    household_controls : numpy.ndarray (Z,)
        integer household control per sub-zone (what ``smart_round`` enforces)
    total_hh_sub_control_index : int
        column of the household total in ``sub_incidence``
    keys : numpy.ndarray (Z, H)
        household keys, already combined by the caller (draw key or ``ln r``,
        plus the log of the row's summed residual); clipped here
    penalties : PenaltySettings

    Returns
    -------
    ControlledProblem

    Raises
    ------
    ValueError
        naming the argument, for disagreeing shapes, NaN (or, for weights,
        incidences and controls, infinite) values, negative weights, a
        non-integral household control or a household-total column out of
        range
    """
    w = _checked("sub_float_weights", sub_float_weights, (None, None))
    zones, rows = w.shape
    if (w < 0).any():
        raise ValueError(
            f"sub_float_weights: {int((w < 0).sum())} negative weights "
            f"out of {w.size}"
        )
    inc = _checked("sub_incidence", sub_incidence, (rows, None))
    controls = inc.shape[1]
    pinc = _checked("parent_incidence", parent_incidence, (rows, None))
    parent_controls = pinc.shape[1]
    sub_imp = _checked("sub_importance", sub_importance, (controls,), allow_inf=True)
    parent_imp = _checked(
        "parent_importance", parent_importance, (parent_controls,), allow_inf=True
    )
    n_control = _checked("household_controls", household_controls, (zones,))
    if (n_control != np.round(n_control)).any():
        raise ValueError(
            f"household_controls must be integers, got {n_control.tolist()!r}"
        )
    key_array = _checked("keys", keys, (zones, rows), allow_inf=True)
    t = total_hh_sub_control_index
    if (
        isinstance(t, (bool, np.bool_))
        or not isinstance(t, (int, np.integer))
        or not 0 <= t < controls
    ):
        raise ValueError(
            f"total_hh_sub_control_index must be an integer in [0, {controls}), "
            f"got {t!r}"
        )

    # integer parts and residuals as upstream (astype(int) floors w >= 0)
    a = w.astype(int).astype(np.float64)
    r = w - a
    x_allowed = r > 0

    # cell totals, float and integer part, and the band around the float total
    float_totals = w @ inc
    int_totals = a @ inc
    snapped = _snap(float_totals)
    lo = np.floor(snapped)
    hi = np.ceil(snapped)
    frac = snapped - lo
    band_lo = lo - int_totals
    band_hi = hi - int_totals
    # A <= F holds for non-negative incidence, and A is an integer for integer
    # incidence, so both bands are >= 0; only fractional incidence can push
    # them below 0. Clipped at 0 (y >= A anyway) and reported.
    below = (band_lo < 0) | (band_hi < 0)
    if below.any():
        logger.warning(
            "controlled rounding: %s of %s cell bands below the integer part "
            "(fractional incidence?), clipped at 0",
            int(below.sum()),
            below.size,
        )
    band_lo = np.maximum(band_lo, 0.0)
    band_hi = np.maximum(band_hi, 0.0)

    # penalty tiers and slack prices of the ranged cell rows
    cell_penalty = np.clip(sub_imp, penalties.floor, penalties.cap)
    parent_sub_penalty = np.minimum(
        penalties.parent_total_factor * cell_penalty, penalties.cap
    )
    parent_penalty = np.minimum(
        penalties.parent_total_factor
        * np.clip(parent_imp, penalties.floor, penalties.cap),
        penalties.cap,
    )
    under_price = 2.0 * cell_penalty * (1.0 - frac)
    over_price = 2.0 * cell_penalty * frac

    # parent and conservation targets, in units of x
    parent_sub_target = np.round(float_totals.sum(axis=0)) - int_totals.sum(axis=0)
    parent_target = np.round(w.sum(axis=0) @ pinc) - a.sum(axis=0) @ pinc
    conserve_target = np.round(r.sum(axis=0))

    # household count per sub-zone: what smart_round adds to the integer parts.
    # smart_round counts rows: its shortfall is N - sum_h a, clipped to
    # [0, H]; it rounds up the rows with the largest residual, so the LP picks
    # min(shortfall, n_avail) of them and the rest fall on rows without one.
    held = w.sum(axis=0) > 0
    off_one = int((inc[held, t] != 1).sum())
    if off_one:
        # the household count follows smart_round's row count; the
        # household-total cell (A[:, t] + x @ inc[:, t]) then differs from it
        logger.warning(
            "controlled rounding: household-total incidence is not 1 for %s of "
            "%s rows with weight; the household-total cell differs from the "
            "row count smart_round enforces",
            off_one,
            int(held.sum()),
        )
    n_avail = x_allowed.sum(axis=1)
    need = n_control - a.sum(axis=1)
    added = np.clip(need, 0, rows)  # rows smart_round adds per sub-zone
    hh_count = np.rint(np.clip(need, 0, n_avail)).astype(np.int64)
    hh_outside_lp = int(np.rint((added - hh_count).sum()))
    hh_unreachable = int(np.rint(np.maximum(need - rows, 0.0).sum()))
    hh_excess = int(np.rint(np.maximum(-need, 0.0).sum()))

    # objective coefficient of x (maximisation): clipped key minus the folded
    # cell-distance term q_c (1 - 2 f) inc of every control but the total
    cell_slope = cell_penalty * (1.0 - 2.0 * frac)
    cell_slope[:, t] = 0.0
    x_coef = np.clip(key_array, -KEY_CLIP, KEY_CLIP) - cell_slope @ inc.T

    return ControlledProblem(
        x_allowed=_read_only(x_allowed),
        hh_count=_read_only(hh_count),
        band_lo=_read_only(band_lo),
        band_hi=_read_only(band_hi),
        frac=_read_only(frac),
        under_price=_read_only(under_price),
        over_price=_read_only(over_price),
        cell_penalty=_read_only(cell_penalty),
        parent_sub_target=_read_only(parent_sub_target),
        parent_sub_penalty=_read_only(parent_sub_penalty),
        parent_target=_read_only(parent_target),
        parent_penalty=_read_only(parent_penalty),
        conserve_target=_read_only(conserve_target),
        conserve_penalty=float(penalties.conserve_penalty),
        x_coef=_read_only(x_coef),
        hh_unreachable=hh_unreachable,
        hh_outside_lp=hh_outside_lp,
        hh_excess=hh_excess,
    )


def cell_cost(y, float_total, penalty) -> float:
    """Reference cell cost ``penalty * |y - F'|`` (``F'`` the snapped total)."""
    return float(penalty * abs(float(y) - float(_snap(float_total))))
