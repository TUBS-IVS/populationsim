# PopulationSim
# See full license in LICENSE.txt.
"""Fork-specific settings of the integerizers, read and validated in one place.

Every setting defaults to the behaviour of upstream v0.10.0.
"""

from __future__ import annotations

import logging
import math
from typing import NamedTuple

import numpy as np

from populationsim.core import config
from populationsim.core.random import hash32
from populationsim.integerizing.constants import CONTROLLED_SOLVERS

logger = logging.getLogger(__name__)

TIMEOUT_NAME = "INTEGERIZER_TIMEOUT"
TIMEOUT_ALIAS = "INTEGIZER_TIMEOUT"  # upstream's historical misspelling
DEFAULT_TIMEOUT_SECONDS = 60
DRAW_METHODS = ("off", "pareto", "gumbel")
CALL_KINDS = ("single", "simul")
ROUNDING_MODES = ("round", "controlled")
CVXPY_ROUNDING_MESSAGE = (
    "INTEGERIZER_ROUNDING=controlled requires the OR-Tools integerizer "
    "(USE_CVXPY false)"
)
CONTROLLED_SOLVER_NAME = "INTEGERIZER_CONTROLLED_SOLVER"
CONTROLLED_RETRY_SOLVER_NAME = "INTEGERIZER_CONTROLLED_RETRY_SOLVER"
CONTROLLED_REPAIR_NAME = "INTEGERIZER_CONTROLLED_REPAIR"
DEFAULT_CONTROLLED_SOLVER = "highs-ipm"
DEFAULT_CONTROLLED_RETRY_SOLVER = "clp-dual"
NO_RETRY = "none"
CONTROLLED_METHOD_NAME = "INTEGERIZER_CONTROLLED_METHOD"
# search: network start and swap search (controlled_search); lp: the LP with
# repair (controlled_lp)
CONTROLLED_METHODS = ("search", "lp")
DEFAULT_CONTROLLED_METHOD = "search"
SEARCH_MAX_SWEEPS_NAME = "INTEGERIZER_SEARCH_MAX_SWEEPS"
SEARCH_SHORTLIST_NAME = "INTEGERIZER_SEARCH_SHORTLIST"
DEFAULT_SEARCH_MAX_SWEEPS = 200
DEFAULT_SEARCH_SHORTLIST = 24

_logged = False


def _is_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def integerizer_timeout_seconds():
    """Solver time limit in seconds (``INTEGERIZER_TIMEOUT`` or alias)."""
    new = config.setting(TIMEOUT_NAME, None)
    old = config.setting(TIMEOUT_ALIAS, None)
    if new is not None and old is not None and new != old:
        raise ValueError(
            f"{TIMEOUT_NAME}={new!r} and {TIMEOUT_ALIAS}={old!r} are both set "
            f"with different values; set only one of them"
        )
    name, value = (TIMEOUT_NAME, new) if new is not None else (TIMEOUT_ALIAS, old)
    if value is None:
        return DEFAULT_TIMEOUT_SECONDS
    if not _is_int(value) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def draw_method():
    """``INTEGERIZER_DRAW``: one of off, pareto, gumbel (default off)."""
    value = config.setting("INTEGERIZER_DRAW", "off")
    if not isinstance(value, str) or value not in DRAW_METHODS:
        raise ValueError(
            f"INTEGERIZER_DRAW must be one of {DRAW_METHODS}, got {value!r}"
        )
    return value


def random_seed():
    """``INTEGERIZER_RANDOM_SEED``: integer in [0, 2**32), default 0."""
    value = config.setting("INTEGERIZER_RANDOM_SEED", 0)
    if not _is_int(value) or not 0 <= value < 2**32:
        raise ValueError(
            f"INTEGERIZER_RANDOM_SEED must be an integer in [0, 2**32), got {value!r}"
        )
    return int(value)


def rounding_mode():
    """``INTEGERIZER_ROUNDING``: ``round`` (default, upstream) or ``controlled``."""
    value = config.setting("INTEGERIZER_ROUNDING", "round")
    if not isinstance(value, str) or value not in ROUNDING_MODES:
        raise ValueError(
            f"INTEGERIZER_ROUNDING must be one of {ROUNDING_MODES}, got {value!r}"
        )
    return value


def resolve_rounding_mode(use_cvxpy):
    """``INTEGERIZER_ROUNDING`` for an integerizer, refused for the CVXPY solver."""
    mode = rounding_mode()
    if mode == "controlled" and use_cvxpy:
        raise ValueError(CVXPY_ROUNDING_MESSAGE)
    return mode


class PenaltySettings(NamedTuple):
    """Penalty tiers of the controlled rounding (see the fair-rounding design)."""

    floor: float
    cap: float
    parent_total_factor: float
    conserve_penalty: float


def _number(key, default):
    value = config.setting(key, default)
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{key} must be a finite number, got {value!r}")
    return float(value)


def penalty_settings():
    """Penalty tiers: floor > 0, cap >= floor, factor >= 1, conserve >= 0."""
    floor = _number("INTEGERIZER_PENALTY_FLOOR", 100.0)
    cap = _number("INTEGERIZER_PENALTY_CAP", 100000.0)
    factor = _number("INTEGERIZER_PARENT_TOTAL_FACTOR", 10.0)
    conserve = _number("INTEGERIZER_CONSERVE_PENALTY", 100000.0)
    if floor <= 0:
        raise ValueError(f"INTEGERIZER_PENALTY_FLOOR must be > 0, got {floor!r}")
    if cap < floor:
        raise ValueError(
            f"INTEGERIZER_PENALTY_CAP must be >= INTEGERIZER_PENALTY_FLOOR "
            f"({floor!r}), got {cap!r}"
        )
    if factor < 1:
        raise ValueError(
            f"INTEGERIZER_PARENT_TOTAL_FACTOR must be >= 1, got {factor!r}"
        )
    if conserve < 0:
        raise ValueError(f"INTEGERIZER_CONSERVE_PENALTY must be >= 0, got {conserve!r}")
    return PenaltySettings(floor, cap, factor, conserve)


def _flag(key, default=False):
    value = config.setting(key, default)
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{key} must be true or false, got {value!r}")
    return bool(value)


def _solver_name(key, default, allowed):
    value = config.setting(key, default)
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{key} must be one of {allowed}, got {value!r}")
    return value


def controlled_solver():
    """``INTEGERIZER_CONTROLLED_SOLVER``: LP solver of the controlled rounding.

    One of :data:`populationsim.integerizing.constants.CONTROLLED_SOLVERS`
    (default ``highs-ipm``); whether the environment provides it is checked
    when it is first used (``controlled_lp.solve_lp``).
    """
    return _solver_name(
        CONTROLLED_SOLVER_NAME, DEFAULT_CONTROLLED_SOLVER, CONTROLLED_SOLVERS
    )


def controlled_retry_solver():
    """``INTEGERIZER_CONTROLLED_RETRY_SOLVER``: solver of the second attempt.

    Solves the same LP once more when the first solver ends without success;
    a solver name (default ``clp-dual``) or ``none`` (no second attempt). It
    must differ from ``INTEGERIZER_CONTROLLED_SOLVER``.
    """
    value = _solver_name(
        CONTROLLED_RETRY_SOLVER_NAME,
        DEFAULT_CONTROLLED_RETRY_SOLVER,
        CONTROLLED_SOLVERS + (NO_RETRY,),
    )
    first = controlled_solver()
    if value == first:
        raise ValueError(
            f"{CONTROLLED_RETRY_SOLVER_NAME} must differ from "
            f"{CONTROLLED_SOLVER_NAME} (both {value!r}); set it to another "
            f"solver or to {NO_RETRY!r}"
        )
    return value


def controlled_repair():
    """``INTEGERIZER_CONTROLLED_REPAIR`` (default True): round the fractional
    values of the controlled LP jointly (``controlled_lp.repair_fractional``)
    before ``smart_round``."""
    return _flag(CONTROLLED_REPAIR_NAME, True)


def controlled_method():
    """``INTEGERIZER_CONTROLLED_METHOD``: how the controlled rounding decides.

    ``search`` (default): a network start and a swap search
    (``controlled_search.solve_search``); ``lp``: the LP with retry and repair
    (``controlled_lp.solve_controlled``, which reads the solver settings).
    """
    return _solver_name(
        CONTROLLED_METHOD_NAME, DEFAULT_CONTROLLED_METHOD, CONTROLLED_METHODS
    )


class SearchSettings(NamedTuple):
    """Limits of the swap search of the controlled rounding (``search``)."""

    max_sweeps: int
    shortlist: int


def _count(key, default):
    value = config.setting(key, default)
    if not _is_int(value) or value < 1:
        raise ValueError(f"{key} must be an integer >= 1, got {value!r}")
    return int(value)


def search_settings():
    """``INTEGERIZER_SEARCH_MAX_SWEEPS`` (default 200): sweeps over all zone
    pairs at most; ``INTEGERIZER_SEARCH_SHORTLIST`` (default 24): candidates
    per zone priced for each zone pair. Both integers >= 1."""
    return SearchSettings(
        _count(SEARCH_MAX_SWEEPS_NAME, DEFAULT_SEARCH_MAX_SWEEPS),
        _count(SEARCH_SHORTLIST_NAME, DEFAULT_SEARCH_SHORTLIST),
    )


def keep_fractional_sample_weights():
    """``KEEP_FRACTIONAL_SAMPLE_WEIGHTS`` (default False)."""
    return _flag("KEEP_FRACTIONAL_SAMPLE_WEIGHTS")


def path_record_enabled():
    """``INTEGERIZER_PATH_RECORD`` (default False)."""
    return _flag("INTEGERIZER_PATH_RECORD")


def call_rng(kind, trace_label):
    """Noise generator of one integerizer call, stable across runs and numpy.

    Legacy ``RandomState`` (frozen stream) seeded with the run seed and a
    32 bit hash of the call identity; never Python ``hash()``.
    """
    if kind not in CALL_KINDS:
        raise ValueError(f"call kind must be one of {CALL_KINDS}, got {kind!r}")
    return np.random.RandomState([random_seed(), hash32(f"{kind}|{trace_label}")])


def log_options_once():
    """Log the resolved fork options once per process (one INFO line)."""
    global _logged
    if _logged:
        return
    _logged = True
    mode = rounding_mode()
    message = (
        "integerizer options: timeout=%ss draw=%s seed=%s "
        "keep_fractional_sample_weights=%s path_record=%s rounding=%s"
    )
    args = [
        integerizer_timeout_seconds(),
        draw_method(),
        random_seed(),
        keep_fractional_sample_weights(),
        path_record_enabled(),
        mode,
    ]
    if mode == "controlled":
        message += (
            " penalty_floor=%s penalty_cap=%s parent_total_factor=%s "
            "conserve_penalty=%s controlled_method=%s search_max_sweeps=%s "
            "search_shortlist=%s controlled_solver=%s "
            "controlled_retry_solver=%s controlled_repair=%s"
        )
        args.extend(penalty_settings())
        args.append(controlled_method())
        args.extend(search_settings())
        args.extend(
            [controlled_solver(), controlled_retry_solver(), controlled_repair()]
        )
    logger.info(message, *args)


def reset_for_tests():
    """Forget that the options were logged (test helper)."""
    global _logged
    _logged = False
