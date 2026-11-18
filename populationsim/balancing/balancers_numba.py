import atexit
import logging
import numpy as np
from numba import njit
from populationsim.balancing.constants import (
    DEFAULT_MAX_ITERATIONS,
    MAX_DELTA32,
    MAX_GAMMA,
    MIN_GAMMA,
    IMPORTANCE_ADJUST,
    IMPORTANCE_ADJUST_COUNT,
    MIN_IMPORTANCE,
    MAX_RELAXATION_FACTOR,
    MIN_CONTROL_VALUE,
    ALT_MAX_DELTA,
)

logger = logging.getLogger(__name__)


@njit(fastmath=True, cache=True)
def np_balancer_numba(
    sample_count: int,
    control_count: int,
    master_control_index: int,
    incidence: np.ndarray,
    weights_initial: np.ndarray,
    weights_lower_bound: np.ndarray,
    weights_upper_bound: np.ndarray,
    controls_constraint: np.ndarray,
    controls_importance: np.ndarray,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> tuple[np.ndarray, np.ndarray, tuple[bool, int, float, float]]:
    # Upcast key scalars to float64 for stability
    weights_final = weights_initial.copy()
    relaxation_factors = np.empty(control_count, dtype=np.float64)

    for i in range(control_count):
        relaxation_factors[i] = 1.0

    # Precompute incidence squared
    incidence2 = incidence * incidence
    importance_adjustment = 1.0

    # Manual control reordering
    control_indexes = np.empty(control_count, dtype=np.int32)
    k = 0
    for i in range(control_count):
        if i != master_control_index:
            control_indexes[k] = i
            k += 1
    if master_control_index >= 0:
        control_indexes[k] = master_control_index

    for iter in range(max_iterations):
        delta = 0.0  # float64
        gamma = np.ones(control_count, dtype=np.float64)

        if iter > 0 and iter % IMPORTANCE_ADJUST_COUNT == 0:
            importance_adjustment /= IMPORTANCE_ADJUST

        for i in range(control_count):
            c = control_indexes[i]
            xx = 0.0
            yy = 0.0
            for j in range(sample_count):
                w = float(weights_final[j])
                inc = float(incidence[c, j])
                xx += w * inc
                yy += w * float(incidence2[c, j])

            imp = (
                float(controls_importance[c])
                if c == master_control_index
                else max(
                    float(controls_importance[c]) * importance_adjustment,
                    MIN_IMPORTANCE,
                )
            )

            if xx > 0.0:
                relaxed = float(controls_constraint[c]) * relaxation_factors[c]
                if relaxed < MIN_CONTROL_VALUE:
                    relaxed = MIN_CONTROL_VALUE

                gamma_val = 1.0 - (xx - relaxed) / (yy + relaxed / imp)
                gamma_val = max(gamma_val, MIN_GAMMA)
                gamma[c] = gamma_val
                log_gamma = np.log(gamma_val)

                for j in range(sample_count):
                    w_old = float(weights_final[j])
                    inc = float(incidence[c, j])
                    new_w = w_old * np.exp(log_gamma * inc)

                    lb = float(weights_lower_bound[j])
                    ub = float(weights_upper_bound[j])
                    new_w = min(max(new_w, lb), ub)

                    delta += abs(new_w - w_old)
                    weights_final[j] = new_w

                relax_factor = relaxation_factors[c] * (1.0 / gamma_val) ** (1.0 / imp)
                if relax_factor > MAX_RELAXATION_FACTOR:
                    relax_factor = MAX_RELAXATION_FACTOR
                relaxation_factors[c] = relax_factor

        delta /= sample_count
        max_gamma_dif = 0.0
        for i in range(control_count):
            g_dif = abs(gamma[i] - 1.0)
            if g_dif > max_gamma_dif:
                max_gamma_dif = g_dif

        converged = delta < MAX_DELTA32 and max_gamma_dif < MAX_GAMMA
        no_progress = delta < ALT_MAX_DELTA

        if converged or no_progress:
            return weights_final, relaxation_factors, (True, iter, delta, max_gamma_dif)

    return (
        weights_final,
        relaxation_factors,
        (False, max_iterations, delta, max_gamma_dif),
    )


@njit(fastmath=True, cache=True)
def np_simul_balancer_numba(
    sample_count: int,
    control_count: int,
    zone_count: int,
    master_control_index: int,
    incidence: np.ndarray,
    parent_weights: np.ndarray,
    weights_lower_bound: np.ndarray,
    weights_upper_bound: np.ndarray,
    sub_weights: np.ndarray,
    parent_controls: np.ndarray,
    controls_importance: np.ndarray,
    sub_controls: np.ndarray,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> tuple[np.ndarray, np.ndarray, tuple[bool, int, float, float]]:
    if parent_controls is not None:
        # Normalize sub_controls to match parent_controls if provided
        totals = sub_controls.sum(axis=0)
        scale_factors = np.ones_like(totals, dtype=np.float64)
        valid = totals > 0
        scale_factors[valid] = parent_controls[valid] / totals[valid]
        sub_controls *= scale_factors

    relaxation_factors = np.ones((zone_count, control_count), dtype=np.float64)
    incidence2 = incidence * incidence
    importance_adjustment = 1.0

    control_indexes = np.empty(control_count, dtype=np.int32)
    k = 0
    for i in range(control_count):
        if i != master_control_index:
            control_indexes[k] = i
            k += 1
    if master_control_index >= 0:
        control_indexes[k] = master_control_index

    for iter in range(max_iterations):
        weights_previous = sub_weights.copy()
        gamma = np.ones((zone_count, control_count), dtype=np.float64)

        if iter > 0 and iter % IMPORTANCE_ADJUST_COUNT == 0:
            importance_adjustment /= IMPORTANCE_ADJUST

        for i in range(control_count):
            c = control_indexes[i]

            if c == master_control_index:
                importance = float(controls_importance[c])
            else:
                importance = max(
                    float(controls_importance[c]) * importance_adjustment,
                    MIN_IMPORTANCE,
                )

            for z in range(zone_count):
                xx = 0.0
                yy = 0.0
                for j in range(sample_count):
                    w = float(sub_weights[z, j])
                    inc = float(incidence[c, j])
                    xx += w * inc
                    yy += w * float(incidence2[c, j])

                if xx > 0.0:
                    relaxed = float(sub_controls[z, c]) * relaxation_factors[z, c]
                    if relaxed < MIN_CONTROL_VALUE:
                        relaxed = MIN_CONTROL_VALUE

                    gamma_val = 1.0 - (xx - relaxed) / (yy + relaxed / importance)
                    gamma_val = max(gamma_val, MIN_GAMMA)
                    gamma[z, c] = gamma_val
                    log_gamma = np.log(gamma_val)

                    for j in range(sample_count):
                        w_old = float(sub_weights[z, j])
                        inc = float(incidence[c, j])
                        new_w = w_old * np.exp(log_gamma * inc)

                        lb = float(weights_lower_bound[j])
                        ub = float(weights_upper_bound[j])
                        new_w = min(max(new_w, lb), ub)
                        sub_weights[z, j] = new_w

                    relax_factor = relaxation_factors[z, c] * (1.0 / gamma_val) ** (
                        1.0 / importance
                    )
                    relaxation_factors[z, c] = min(relax_factor, MAX_RELAXATION_FACTOR)

        # Rescale
        zone_sums = np.sum(sub_weights, axis=0)
        for j in range(sample_count):
            scale = parent_weights[j] / zone_sums[j] if zone_sums[j] > 0 else 1.0
            for z in range(zone_count):
                sub_weights[z, j] *= scale

        max_gamma_dif = np.abs(gamma - 1).max()
        delta = np.abs(sub_weights - weights_previous).sum() / float(sample_count)

        converged = delta < MAX_DELTA32 and max_gamma_dif < MAX_GAMMA
        no_progress = delta < ALT_MAX_DELTA

        if converged or no_progress:
            return (
                sub_weights,
                relaxation_factors,
                (converged, iter, delta, max_gamma_dif),
            )

    return (
        sub_weights,
        relaxation_factors,
        (False, max_iterations, delta, max_gamma_dif),
    )


# Fork fast path (setting FAST_PATHS, see populationsim/core/bs_run_options.py).
#
# np_balancer_numba_fast and np_simul_balancer_numba_fast have the signatures
# and the results of the two kernels above. They differ in one place only: the
# weight update w_old * exp(log_gamma * inc). The factor exp(log_gamma * inc)
# depends only on the incidence VALUE, and one control has few distinct values
# (0, 1, 2, ... persons). So the incidence is encoded once per call
# (_encode_incidence: the sorted distinct values of every control and, per
# cell, the index of its value), and once log_gamma of a control (single) or of
# a zone and control (simultaneous) is known, the kernel computes the factor
# once per distinct value and the update loop looks it up.
#
# Why the weights are identical bit for bit: every factor is the same
# floating-point number as in the original (the same product of the same two
# float64 numbers, passed to the same exp function); the multiplication, the
# clip and the store are per element; no weight depends on the order of a sum
# inside the update loop. The xx / yy reductions, the relaxation factors, the
# rescale and the loop order are copied unchanged.
#
# What can differ is the association of the delta sum: LLVM may vectorise the
# new update loop differently. delta only decides convergence (delta below
# MAX_DELTA32 together with the gamma test, delta below ALT_MAX_DELTA), and a
# sum of n non-negative terms differs between two summation orders by about
# n * 2.2e-16 relative at most. The fast kernels therefore report
# near_threshold when, in any iteration, delta lies within a relative band of
# FAST_PATH_DELTA_BAND around either threshold; the caller then discards the
# fast result and runs the original kernel on the original inputs. Outside the
# band both kernels take the same decision in every iteration.
#
# Fallbacks to the original kernel, all counted (fast_path_counts) and logged:
# a control with more than FAST_PATH_MAX_VALUES distinct values, a non-finite
# incidence value, numba's SVML vector maths being active (the original update
# loop would then use the SVML exp for the vectorised elements and the scalar
# exp for the remainder, which a lookup table cannot reproduce), and the delta
# band above.

_LOGGED_KERNEL_CHOICES = set()

# Most distinct incidence values of one control the lookup table takes; the
# codes are uint8, so this cannot exceed _CODE_DTYPE_CAPACITY (checked at every
# fast call, not by an assert that python -O would strip).
FAST_PATH_MAX_VALUES = 256
_CODE_DTYPE_CAPACITY = np.iinfo(np.uint8).max + 1
# Relative band around MAX_DELTA32 and ALT_MAX_DELTA in which the result of a
# fast kernel is discarded (read at call time, so a test can widen it).
FAST_PATH_DELTA_BAND = 1.0e-6
# The two delta values the band is centred on (read at call time and handed to
# the kernel, because numba freezes module constants when it compiles; the
# convergence tests inside the kernels keep using MAX_DELTA32 and ALT_MAX_DELTA).
FAST_PATH_BAND_CENTER_1 = MAX_DELTA32
FAST_PATH_BAND_CENTER_2 = ALT_MAX_DELTA
# Fallback warnings logged in full; after that one summary every
# _FAST_PATH_WARNING_EVERY events.
_FAST_PATH_WARNING_LIMIT = 5
_FAST_PATH_WARNING_EVERY = 100

_FAST_PATH_OUTCOMES = (
    "fast",
    "fallback_value_cap",
    "fallback_nonfinite",
    "fallback_svml",
    "rerun_delta_band",
)
_FAST_PATH_COUNTS = {
    kind: dict.fromkeys(_FAST_PATH_OUTCOMES, 0) for kind in ("single", "simul")
}
_FAST_PATH_STATE = {
    "fallback_events": 0,
    "first_call_logged": set(),
    "atexit_registered": False,
}

_ENCODE_OK = 0
_ENCODE_VALUE_CAP = 1
_ENCODE_NONFINITE = 2


def fast_path_counts():
    """Per-process counts of the fast balancer calls by outcome.

    Returns ``{"single": {...}, "simul": {...}}``, each with the keys ``fast``
    (the fast kernel's result was used), ``fallback_value_cap``,
    ``fallback_nonfinite``, ``fallback_svml`` (the original kernel ran instead)
    and ``rerun_delta_band`` (the fast result was discarded and the original
    kernel re-run).
    """
    return {kind: dict(counts) for kind, counts in _FAST_PATH_COUNTS.items()}


def reset_fast_path_counts():
    """Reset the counts, the first-call log and the warning rate limit (tests)."""
    for counts in _FAST_PATH_COUNTS.values():
        for outcome in counts:
            counts[outcome] = 0
    _FAST_PATH_STATE["fallback_events"] = 0
    _FAST_PATH_STATE["first_call_logged"] = set()


def log_fast_path_rate(logger=None):
    """Log ONE summary line of the fast-path outcomes of this process.

    INFO when every call took the fast kernel, WARNING when any call fell back
    to the original kernel or was re-run after the delta guard. Registered with
    ``atexit`` on the first fast call, so every run ends with this line.
    """
    logger = logger or globals()["logger"]
    totals = dict.fromkeys(_FAST_PATH_OUTCOMES, 0)
    for counts in _FAST_PATH_COUNTS.values():
        for outcome, n in counts.items():
            totals[outcome] += n
    calls = sum(totals.values())
    percent = 100.0 * totals["fast"] / calls if calls else 0.0
    level = logging.INFO if calls == totals["fast"] else logging.WARNING
    logger.log(
        level,
        "balancer fast path: fast %d of %d calls (%.1f %%), fallback value cap %d, "
        "fallback SVML %d, fallback non-finite %d, re-run after the delta guard %d",
        totals["fast"],
        calls,
        percent,
        totals["fallback_value_cap"],
        totals["fallback_svml"],
        totals["fallback_nonfinite"],
        totals["rerun_delta_band"],
    )


def _register_exit_summary():
    if _FAST_PATH_STATE["atexit_registered"]:
        return
    _FAST_PATH_STATE["atexit_registered"] = True
    atexit.register(log_fast_path_rate)


def _check_value_cap():
    if not 1 <= FAST_PATH_MAX_VALUES <= _CODE_DTYPE_CAPACITY:
        raise ValueError(
            "FAST_PATH_MAX_VALUES=%d does not fit the uint8 incidence codes "
            "(allowed: 1 to %d)" % (FAST_PATH_MAX_VALUES, _CODE_DTYPE_CAPACITY)
        )


def log_kernel_choice_once(use_numba):
    """Log once per process which balancer kernels FAST_PATHS selects.

    Called by the balancer classes only when FAST_PATHS is on.
    """
    if use_numba:
        message = (
            "FAST_PATHS: the numba balancers use np_balancer_numba_fast and "
            "np_simul_balancer_numba_fast (exp(log_gamma * incidence) from a "
            "per-control lookup table; weights identical bit for bit, with "
            "counted fallbacks to the original kernels)"
        )
    else:
        message = (
            "FAST_PATHS: USE_NUMBA is off, so the balancers run the Python "
            "implementation, which has no fast path"
        )
    if message not in _LOGGED_KERNEL_CHOICES:
        _LOGGED_KERNEL_CHOICES.add(message)
        logger.info(message)


def _svml_active():
    import numba

    return bool(getattr(numba.config, "USING_SVML", False))


def _log_first_call(kind, kernel_name):
    if kind in _FAST_PATH_STATE["first_call_logged"]:
        return
    _FAST_PATH_STATE["first_call_logged"].add(kind)
    if _svml_active():
        logger.info(
            "FAST_PATHS: %s balancer: numba SVML is active, so every call runs "
            "the original kernel (counted as fallback_svml)",
            kind,
        )
    else:
        logger.info(
            "FAST_PATHS: %s balancer: kernel %s (lookup table, at most %d "
            "distinct incidence values per control, delta band %.1e)",
            kind,
            kernel_name,
            FAST_PATH_MAX_VALUES,
            FAST_PATH_DELTA_BAND,
        )


_OUTCOME_TEXT = {
    "fallback_value_cap": (
        "a control has more than %d distinct incidence values; the original "
        "kernel ran" % FAST_PATH_MAX_VALUES
    ),
    "fallback_nonfinite": "a non-finite incidence value; the original kernel ran",
    "fallback_svml": "numba SVML is active; the original kernel ran",
    "rerun_delta_band": (
        "delta came within the relative band around a convergence threshold; "
        "the fast result was discarded and the original kernel re-run"
    ),
}


def _record(kind, outcome):
    counts = _FAST_PATH_COUNTS[kind]
    counts[outcome] += 1
    if outcome == "fast":
        return
    _FAST_PATH_STATE["fallback_events"] += 1
    events = _FAST_PATH_STATE["fallback_events"]
    if events <= _FAST_PATH_WARNING_LIMIT or events % _FAST_PATH_WARNING_EVERY == 0:
        logger.warning(
            "FAST_PATHS: %s balancer: %s. Counts so far: %s%s",
            kind,
            _OUTCOME_TEXT[outcome],
            fast_path_counts(),
            (
                " (further fallback warnings only every %d events; see "
                "fast_path_counts())" % _FAST_PATH_WARNING_EVERY
                if events == _FAST_PATH_WARNING_LIMIT
                else ""
            ),
        )


# No fastmath here: the finiteness test must not be optimised away.
@njit(cache=True)
def _encode_incidence(incidence, max_values):
    """Sorted distinct values per control and the code of every cell.

    Returns ``(values, n_values, codes, status)``: ``values[c, :n_values[c]]``
    are the sorted distinct values of ``float(incidence[c, j])`` (the
    conversion the kernels apply), ``codes[c, j]`` the index of the cell's
    value in them; ``status`` is _ENCODE_OK, _ENCODE_VALUE_CAP (more than
    ``max_values`` distinct values in a control) or _ENCODE_NONFINITE.
    """
    control_count = incidence.shape[0]
    sample_count = incidence.shape[1]
    values = np.zeros((control_count, max_values), dtype=np.float64)
    n_values = np.zeros(control_count, dtype=np.int64)
    codes = np.zeros((control_count, sample_count), dtype=np.uint8)
    rank = np.empty(max_values, dtype=np.uint8)
    for c in range(control_count):
        n = 0
        last_value = 0.0
        last_code = -1
        for j in range(sample_count):
            x = float(incidence[c, j])
            if not np.isfinite(x):
                return values, n_values, codes, _ENCODE_NONFINITE
            if last_code >= 0 and x == last_value:
                codes[c, j] = last_code
                continue
            code = -1
            for k in range(n):
                if values[c, k] == x:
                    code = k
                    break
            if code < 0:
                if n == max_values:
                    return values, n_values, codes, _ENCODE_VALUE_CAP
                values[c, n] = x
                code = n
                n += 1
            codes[c, j] = code
            last_value = x
            last_code = code
        n_values[c] = n
        # sort the distinct values and remap the codes to the sorted order
        order = np.argsort(values[c, :n])
        sorted_values = values[c, :n][order]
        for k in range(n):
            rank[order[k]] = k
        for k in range(n):
            values[c, k] = sorted_values[k]
        for j in range(sample_count):
            codes[c, j] = rank[codes[c, j]]
    return values, n_values, codes, _ENCODE_OK


def _encoded_or_fallback(kind, incidence):
    """The encoded incidence, or None after counting the fallback reason."""
    _check_value_cap()
    if _svml_active():
        _record(kind, "fallback_svml")
        return None
    values, n_values, codes, status = _encode_incidence(incidence, FAST_PATH_MAX_VALUES)
    if status == _ENCODE_VALUE_CAP:
        _record(kind, "fallback_value_cap")
        return None
    if status == _ENCODE_NONFINITE:
        _record(kind, "fallback_nonfinite")
        return None
    return values, n_values, codes


def np_balancer_numba_fast(
    sample_count,
    control_count,
    master_control_index,
    incidence,
    weights_initial,
    weights_lower_bound,
    weights_upper_bound,
    controls_constraint,
    controls_importance,
    max_iterations=DEFAULT_MAX_ITERATIONS,
):
    """``np_balancer_numba`` with the factor exp(log_gamma * inc) from a table.

    Same arguments and result as ``np_balancer_numba`` (weights, relaxation
    factors, converged flag and iteration count identical bit for bit; delta
    may differ in its last bits). Falls back to, or re-runs, the original
    kernel as described above; every such call is counted and logged.
    """
    _register_exit_summary()
    _log_first_call("single", "np_balancer_numba_fast")
    original_args = (
        sample_count,
        control_count,
        master_control_index,
        incidence,
        weights_initial,
        weights_lower_bound,
        weights_upper_bound,
        controls_constraint,
        controls_importance,
        max_iterations,
    )
    encoded = _encoded_or_fallback("single", incidence)
    if encoded is None:
        return np_balancer_numba(*original_args)
    values, n_values, codes = encoded
    # The kernel copies weights_initial and mutates none of its inputs.
    weights, relaxation_factors, status, near_threshold = _np_balancer_lookup(
        *original_args,
        values,
        n_values,
        codes,
        float(FAST_PATH_DELTA_BAND),
        float(FAST_PATH_BAND_CENTER_1),
        float(FAST_PATH_BAND_CENTER_2),
    )
    if near_threshold:
        _record("single", "rerun_delta_band")
        return np_balancer_numba(*original_args)
    _record("single", "fast")
    return weights, relaxation_factors, status


def np_simul_balancer_numba_fast(
    sample_count,
    control_count,
    zone_count,
    master_control_index,
    incidence,
    parent_weights,
    weights_lower_bound,
    weights_upper_bound,
    sub_weights,
    parent_controls,
    controls_importance,
    sub_controls,
    max_iterations=DEFAULT_MAX_ITERATIONS,
):
    """``np_simul_balancer_numba`` with the factor exp(log_gamma * inc) from a table.

    Same arguments, result and side effects as ``np_simul_balancer_numba``
    (which balances ``sub_weights`` and scales ``sub_controls`` in place and
    returns ``sub_weights``). The lookup kernel works on copies, so that a
    re-run of the original kernel starts from the untouched inputs; the
    accepted result is copied back into the caller's arrays.
    """
    _register_exit_summary()
    _log_first_call("simul", "np_simul_balancer_numba_fast")

    def original():
        return np_simul_balancer_numba(
            sample_count,
            control_count,
            zone_count,
            master_control_index,
            incidence,
            parent_weights,
            weights_lower_bound,
            weights_upper_bound,
            sub_weights,
            parent_controls,
            controls_importance,
            sub_controls,
            max_iterations,
        )

    encoded = _encoded_or_fallback("simul", incidence)
    if encoded is None:
        return original()
    values, n_values, codes = encoded
    # order="K" keeps the caller's memory layout, so numba compiles the same
    # array layout as for the original kernel.
    fast_sub_weights = sub_weights.copy(order="K")
    fast_sub_controls = sub_controls.copy(order="K")
    weights, relaxation_factors, status, near_threshold = _np_simul_balancer_lookup(
        sample_count,
        control_count,
        zone_count,
        master_control_index,
        incidence,
        parent_weights,
        weights_lower_bound,
        weights_upper_bound,
        fast_sub_weights,
        parent_controls,
        controls_importance,
        fast_sub_controls,
        max_iterations,
        values,
        n_values,
        codes,
        float(FAST_PATH_DELTA_BAND),
        float(FAST_PATH_BAND_CENTER_1),
        float(FAST_PATH_BAND_CENTER_2),
    )
    if near_threshold:
        _record("simul", "rerun_delta_band")
        return original()
    _record("simul", "fast")
    sub_weights[...] = weights
    sub_controls[...] = fast_sub_controls
    return sub_weights, relaxation_factors, status


@njit(fastmath=True, cache=True)
def _np_balancer_lookup(
    sample_count,
    control_count,
    master_control_index,
    incidence,
    weights_initial,
    weights_lower_bound,
    weights_upper_bound,
    controls_constraint,
    controls_importance,
    max_iterations,
    values,
    n_values,
    codes,
    delta_band,
    band_center_1=MAX_DELTA32,
    band_center_2=ALT_MAX_DELTA,
):
    """Copy of ``np_balancer_numba`` with a per-control lookup table.

    Also returns ``near_threshold``: True (and an early return) when delta of
    an iteration lies within the relative ``delta_band`` around MAX_DELTA32
    or ALT_MAX_DELTA; the caller then discards the result.
    """
    # Upcast key scalars to float64 for stability
    weights_final = weights_initial.copy()
    relaxation_factors = np.empty(control_count, dtype=np.float64)

    for i in range(control_count):
        relaxation_factors[i] = 1.0

    # Precompute incidence squared
    incidence2 = incidence * incidence
    importance_adjustment = 1.0
    factor = np.empty(values.shape[1], dtype=np.float64)

    # Manual control reordering
    control_indexes = np.empty(control_count, dtype=np.int32)
    k = 0
    for i in range(control_count):
        if i != master_control_index:
            control_indexes[k] = i
            k += 1
    if master_control_index >= 0:
        control_indexes[k] = master_control_index

    for iter in range(max_iterations):
        delta = 0.0  # float64
        gamma = np.ones(control_count, dtype=np.float64)

        if iter > 0 and iter % IMPORTANCE_ADJUST_COUNT == 0:
            importance_adjustment /= IMPORTANCE_ADJUST

        for i in range(control_count):
            c = control_indexes[i]
            xx = 0.0
            yy = 0.0
            for j in range(sample_count):
                w = float(weights_final[j])
                inc = float(incidence[c, j])
                xx += w * inc
                yy += w * float(incidence2[c, j])

            imp = (
                float(controls_importance[c])
                if c == master_control_index
                else max(
                    float(controls_importance[c]) * importance_adjustment,
                    MIN_IMPORTANCE,
                )
            )

            if xx > 0.0:
                relaxed = float(controls_constraint[c]) * relaxation_factors[c]
                if relaxed < MIN_CONTROL_VALUE:
                    relaxed = MIN_CONTROL_VALUE

                gamma_val = 1.0 - (xx - relaxed) / (yy + relaxed / imp)
                gamma_val = max(gamma_val, MIN_GAMMA)
                gamma[c] = gamma_val
                log_gamma = np.log(gamma_val)

                # exp(log_gamma * inc) once per distinct value of the control
                for v in range(n_values[c]):
                    factor[v] = np.exp(log_gamma * values[c, v])

                for j in range(sample_count):
                    w_old = float(weights_final[j])
                    new_w = w_old * factor[codes[c, j]]

                    lb = float(weights_lower_bound[j])
                    ub = float(weights_upper_bound[j])
                    new_w = min(max(new_w, lb), ub)

                    delta += abs(new_w - w_old)
                    weights_final[j] = new_w

                relax_factor = relaxation_factors[c] * (1.0 / gamma_val) ** (1.0 / imp)
                if relax_factor > MAX_RELAXATION_FACTOR:
                    relax_factor = MAX_RELAXATION_FACTOR
                relaxation_factors[c] = relax_factor

        delta /= sample_count
        max_gamma_dif = 0.0
        for i in range(control_count):
            g_dif = abs(gamma[i] - 1.0)
            if g_dif > max_gamma_dif:
                max_gamma_dif = g_dif

        if (
            abs(delta - band_center_1) <= delta_band * band_center_1
            or abs(delta - band_center_2) <= delta_band * band_center_2
        ):
            return (
                weights_final,
                relaxation_factors,
                (False, iter, delta, max_gamma_dif),
                True,
            )

        converged = delta < MAX_DELTA32 and max_gamma_dif < MAX_GAMMA
        no_progress = delta < ALT_MAX_DELTA

        if converged or no_progress:
            return (
                weights_final,
                relaxation_factors,
                (True, iter, delta, max_gamma_dif),
                False,
            )

    return (
        weights_final,
        relaxation_factors,
        (False, max_iterations, delta, max_gamma_dif),
        False,
    )


@njit(fastmath=True, cache=True)
def _np_simul_balancer_lookup(
    sample_count,
    control_count,
    zone_count,
    master_control_index,
    incidence,
    parent_weights,
    weights_lower_bound,
    weights_upper_bound,
    sub_weights,
    parent_controls,
    controls_importance,
    sub_controls,
    max_iterations,
    values,
    n_values,
    codes,
    delta_band,
    band_center_1=MAX_DELTA32,
    band_center_2=ALT_MAX_DELTA,
):
    """Copy of ``np_simul_balancer_numba`` with a per-control lookup table.

    Also returns ``near_threshold`` as ``_np_balancer_lookup`` does.
    """
    if parent_controls is not None:
        # Normalize sub_controls to match parent_controls if provided
        totals = sub_controls.sum(axis=0)
        scale_factors = np.ones_like(totals, dtype=np.float64)
        valid = totals > 0
        scale_factors[valid] = parent_controls[valid] / totals[valid]
        sub_controls *= scale_factors

    relaxation_factors = np.ones((zone_count, control_count), dtype=np.float64)
    incidence2 = incidence * incidence
    importance_adjustment = 1.0
    factor = np.empty(values.shape[1], dtype=np.float64)

    control_indexes = np.empty(control_count, dtype=np.int32)
    k = 0
    for i in range(control_count):
        if i != master_control_index:
            control_indexes[k] = i
            k += 1
    if master_control_index >= 0:
        control_indexes[k] = master_control_index

    for iter in range(max_iterations):
        weights_previous = sub_weights.copy()
        gamma = np.ones((zone_count, control_count), dtype=np.float64)

        if iter > 0 and iter % IMPORTANCE_ADJUST_COUNT == 0:
            importance_adjustment /= IMPORTANCE_ADJUST

        for i in range(control_count):
            c = control_indexes[i]

            if c == master_control_index:
                importance = float(controls_importance[c])
            else:
                importance = max(
                    float(controls_importance[c]) * importance_adjustment,
                    MIN_IMPORTANCE,
                )

            for z in range(zone_count):
                xx = 0.0
                yy = 0.0
                for j in range(sample_count):
                    w = float(sub_weights[z, j])
                    inc = float(incidence[c, j])
                    xx += w * inc
                    yy += w * float(incidence2[c, j])

                if xx > 0.0:
                    relaxed = float(sub_controls[z, c]) * relaxation_factors[z, c]
                    if relaxed < MIN_CONTROL_VALUE:
                        relaxed = MIN_CONTROL_VALUE

                    gamma_val = 1.0 - (xx - relaxed) / (yy + relaxed / importance)
                    gamma_val = max(gamma_val, MIN_GAMMA)
                    gamma[z, c] = gamma_val
                    log_gamma = np.log(gamma_val)

                    # exp(log_gamma * inc) once per distinct value of the control
                    for v in range(n_values[c]):
                        factor[v] = np.exp(log_gamma * values[c, v])

                    for j in range(sample_count):
                        w_old = float(sub_weights[z, j])
                        new_w = w_old * factor[codes[c, j]]

                        lb = float(weights_lower_bound[j])
                        ub = float(weights_upper_bound[j])
                        new_w = min(max(new_w, lb), ub)
                        sub_weights[z, j] = new_w

                    relax_factor = relaxation_factors[z, c] * (1.0 / gamma_val) ** (
                        1.0 / importance
                    )
                    relaxation_factors[z, c] = min(relax_factor, MAX_RELAXATION_FACTOR)

        # Rescale
        zone_sums = np.sum(sub_weights, axis=0)
        for j in range(sample_count):
            scale = parent_weights[j] / zone_sums[j] if zone_sums[j] > 0 else 1.0
            for z in range(zone_count):
                sub_weights[z, j] *= scale

        max_gamma_dif = np.abs(gamma - 1).max()
        delta = np.abs(sub_weights - weights_previous).sum() / float(sample_count)

        if (
            abs(delta - band_center_1) <= delta_band * band_center_1
            or abs(delta - band_center_2) <= delta_band * band_center_2
        ):
            return (
                sub_weights,
                relaxation_factors,
                (False, iter, delta, max_gamma_dif),
                True,
            )

        converged = delta < MAX_DELTA32 and max_gamma_dif < MAX_GAMMA
        no_progress = delta < ALT_MAX_DELTA

        if converged or no_progress:
            return (
                sub_weights,
                relaxation_factors,
                (converged, iter, delta, max_gamma_dif),
                False,
            )

    return (
        sub_weights,
        relaxation_factors,
        (False, max_iterations, delta, max_gamma_dif),
        False,
    )
