# PopulationSim
# See full license in LICENSE.txt.
"""Fork-specific settings of the integerizers, read and validated in one place.

Every setting defaults to the behaviour of upstream v0.10.0.
"""

import logging

import numpy as np

from populationsim.core import config
from populationsim.core.random import hash32

logger = logging.getLogger(__name__)

TIMEOUT_NAME = "INTEGERIZER_TIMEOUT"
TIMEOUT_ALIAS = "INTEGIZER_TIMEOUT"  # upstream's historical misspelling
DEFAULT_TIMEOUT_SECONDS = 60
DRAW_METHODS = ("off", "pareto", "gumbel")
CALL_KINDS = ("single", "simul")

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
    name, value = (
        (TIMEOUT_NAME, new) if new is not None else (TIMEOUT_ALIAS, old)
    )
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


def _flag(key):
    value = config.setting(key, False)
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{key} must be true or false, got {value!r}")
    return bool(value)


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
    logger.info(
        "integerizer options: timeout=%ss draw=%s seed=%s "
        "keep_fractional_sample_weights=%s path_record=%s",
        integerizer_timeout_seconds(),
        draw_method(),
        random_seed(),
        keep_fractional_sample_weights(),
        path_record_enabled(),
    )


def reset_for_tests():
    """Forget that the options were logged (test helper)."""
    global _logged
    _logged = False
