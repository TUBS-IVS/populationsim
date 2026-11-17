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

The noise comes from the call generator of
:func:`populationsim.integerizing.bs_options.call_rng`.
"""

from __future__ import annotations

import numpy as np

from populationsim.integerizing import bs_options

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


def resolve_draw_method(use_cvxpy) -> str:
    """``INTEGERIZER_DRAW`` for an integerizer, refused for the CVXPY solver."""
    method = bs_options.draw_method()
    if method != "off" and use_cvxpy:
        raise ValueError(CVXPY_MESSAGE)
    return method
