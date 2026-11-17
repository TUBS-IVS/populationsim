# PopulationSim
# See full license in LICENSE.txt.
"""Random keys for the seeded weight-proportional draw of the integerizers.

Both OR-Tools integerizer LPs maximise ``sum x * ln(r)`` over the residual
weights ``r`` subject to the controls. With residuals far below 1 this is a
deterministic "largest residual first" choice, identical in every batch. With
``INTEGERIZER_DRAW`` on, ``ln(r)`` is replaced by a random key per household:
taking the k largest keys (the LP still enforces every control) is *order
sampling*, so the draw becomes weight-proportional and seed-dependent.

Two key types (setting ``INTEGERIZER_DRAW``):

``pareto``
    ``logit(r) + Logistic(0, 1)`` (Rosen's Pareto order sampling): the
    inclusion probability of a household is close to its residual ``r``.
``gumbel``
    ``ln(r) + Gumbel(0, 1)`` (successive sampling without replacement,
    "Gumbel top-k"): inclusion is below ``r`` for large ``r`` and above it for
    small ``r``.

These inclusion statements hold for the single integerizer, where the LP
takes the k largest keys (exactly so with only the household-total control;
binding controls reshape the choice). The simultaneous integerizer keeps
upstream's parent term: the coefficient of household ``hh`` in sub-zone ``z``
is ``key[z, hh] + ln(sum over sub-zones of the residuals of hh)``, so every
row is tilted by the log of its residual summed over the sub-zones, and the
inclusion statements above do not carry over unchanged.

Rounding after the LP (``smart_round`` with ``tie_keys``): rows are ordered by
the LP value first and by the same keys second, so the rows added to reach
the household target, beyond those the LP rounded up, follow the keys rather
than the row position. If the LP fails (status not OPTIMAL or FEASIBLE) it
returns the input residuals; with the draw on, the integerizers then round
with the value ``r > 0`` and the keys, i.e. they round up the largest keys
among the households with a positive residual, not the largest residuals.
``integerizing/wrappers.py`` has no rounding step of its own: its
"smart-rounded original weights" (failed single integerizer) and the
"rounded" sub-zones handed on by sequential integerizing are this failure
branch of ``Integerizer.integerize`` and therefore use the keys. A failed
``SimulIntegerizer`` result is discarded by ``do_simul_integerizing`` (which
then falls back to sequential integerizing); its keyed rounding has no effect
on the output.

The noise comes from the call generator of
:func:`populationsim.integerizing.bs_options.call_rng`.
"""

from __future__ import annotations

import numpy as np

from populationsim.integerizing import bs_options
from populationsim.integerizing.constants import STATUS_SUCCESS

DRAW_KEY_METHODS = ("pareto", "gumbel")

# Residuals are clipped before the logarithms: the lower bound is upstream's
# LOG_OVERFLOW floor (ln -> -725, so residual 0 gives a finite key); the upper
# bound keeps logit(r) finite for r == 1.
LOG_OVERFLOW = -725
RESID_UPPER = 1.0 - 1e-12

CVXPY_MESSAGE = "INTEGERIZER_DRAW requires the OR-Tools integerizer (USE_CVXPY false)"


def draw_keys(
    resid_weights: np.ndarray, method: str, rng: np.random.RandomState
) -> np.ndarray:
    """Random order-sampling keys, one per entry of ``resid_weights``.

    Parameters
    ----------
    resid_weights : numpy.ndarray
        residual weights in [0, 1], any shape (1-D single, 2-D simul)
    method : str
        ``"pareto"`` or ``"gumbel"``
    rng : numpy.random.RandomState
        generator of this integerizer call

    Returns
    -------
    numpy.ndarray of float64, same shape as ``resid_weights``
    """
    if method not in DRAW_KEY_METHODS:
        raise ValueError(
            f"draw key method must be one of {DRAW_KEY_METHODS}, got {method!r}"
        )
    resid = np.asarray(resid_weights, dtype=np.float64)
    if np.isnan(resid).any():
        raise ValueError(
            f"draw keys: {int(np.isnan(resid).sum())} NaN residual weights "
            f"out of {resid.size}"
        )
    r = np.clip(resid, np.exp(LOG_OVERFLOW), RESID_UPPER)
    if method == "gumbel":
        return np.log(r) + rng.gumbel(size=r.shape)
    return np.log(r) - np.log1p(-r) + rng.logistic(size=r.shape)


def rounding_values(lp_resid_weights, input_resid_weights, status):
    """Primary sort values for ``smart_round`` when the draw is on.

    LP success: the LP solution itself. LP failure (the LP then returns the
    input residuals): ``1.0`` where the input residual is positive, else
    ``0.0``, so that the keys (``tie_keys``) decide among those households.
    """
    if status in STATUS_SUCCESS:
        return lp_resid_weights
    return (np.asarray(input_resid_weights) > 0).astype(np.float64)


def resolve_draw_method(use_cvxpy) -> str:
    """``INTEGERIZER_DRAW`` for an integerizer, refused for the CVXPY solver."""
    method = bs_options.draw_method()
    if method != "off" and use_cvxpy:
        raise ValueError(CVXPY_MESSAGE)
    return method
