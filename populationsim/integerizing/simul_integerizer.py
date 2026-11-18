# PopulationSim
# See full license in LICENSE.txt.

import logging

import numpy as np
import pandas as pd

from populationsim.core import config
from populationsim.integerizing.smart_round import smart_round
from populationsim.integerizing import bs_options, draw, lp_ortools, lp_cvx, path_record
from populationsim.integerizing import (
    controlled_lp,
    controlled_rounding,
    controlled_search,
)
from populationsim.integerizing.constants import STATUS_SUCCESS

logger = logging.getLogger(__name__)


class SimulIntegerizer:

    def __init__(
        self,
        incidence_df,
        sub_weights,
        sub_controls_df,
        control_spec,
        total_hh_control_col,
        trace_label="",
    ):

        sample_count = len(sub_weights.index)
        sub_zone_count = len(sub_weights.columns)

        assert len(sub_weights.index) == sample_count
        assert len(incidence_df.index) == sample_count
        assert len(sub_controls_df.index) == sub_zone_count
        assert len(sub_weights.columns) == sub_zone_count
        assert total_hh_control_col in sub_controls_df.columns
        assert total_hh_control_col in incidence_df.columns
        assert (incidence_df.columns == control_spec.target).all()

        self.incidence_df = incidence_df
        self.sub_weights = sub_weights
        self.total_hh_control_col = total_hh_control_col

        # control spec rows and control_df columns should be same and in same order
        sub_countrol_cols = list(sub_controls_df.columns)
        sub_control_spec = control_spec[control_spec.target.isin(sub_countrol_cols)]
        self.sub_controls_df = sub_controls_df[sub_control_spec.target.values]
        self.sub_control_importance = sub_control_spec.importance

        # only care about parent control columns NOT in sub_controls
        # control spec rows and control_df columns should be same and in same order
        parent_control_spec = control_spec[
            ~control_spec.target.isin(self.sub_controls_df.columns)
        ]
        self.parent_countrol_cols = parent_control_spec.target.values
        self.parent_countrol_importance = parent_control_spec.importance

        assert total_hh_control_col not in self.parent_countrol_cols

        self.trace_label = trace_label

        # Choose the integerizer function based on configuration
        use_cvxpy = config.setting("USE_CVXPY", False)
        if use_cvxpy:
            self.integerizer_func = lp_cvx.np_simul_integerizer_cvx
        else:
            self.integerizer_func = lp_ortools.np_simul_integerizer_ortools

        self.timeout_in_seconds = bs_options.integerizer_timeout_seconds()
        # fork: INTEGERIZER_DRAW (off = upstream objective); OR-Tools only
        self.draw_method = draw.resolve_draw_method(use_cvxpy)
        self.use_cvxpy = bool(use_cvxpy)
        # fork: INTEGERIZER_ROUNDING (round = upstream LP); OR-Tools only
        self.rounding_mode = bs_options.resolve_rounding_mode(use_cvxpy)
        path_record.check_usable()
        bs_options.log_options_once()

    def integerize(self):

        # fork: controlled rounding (INTEGERIZER_ROUNDING: controlled); with
        # round the method below is upstream's, unchanged
        if self.rounding_mode == "controlled":
            return self._integerize_controlled()

        # - subzone

        total_hh_sub_control_index = self.sub_controls_df.columns.get_loc(
            self.total_hh_control_col
        )

        # FIXME - shouldn't need this?
        total_hh_parent_control_index = -1

        sub_incidence = self.incidence_df[self.sub_controls_df.columns]
        sub_incidence = sub_incidence.values.astype(np.float64)

        sub_float_weights = self.sub_weights.values.transpose().astype(np.float64)
        sub_int_weights = sub_float_weights.astype(int)
        sub_resid_weights = sub_float_weights % 1.0

        # print "sub_float_weights\n", sub_float_weights
        # print "sub_int_weights\n", sub_int_weights
        # print "sub_resid_weights\n", sub_resid_weights

        sub_control_totals = np.asanyarray(self.sub_controls_df).astype(np.int64)
        sub_control_importance = np.asanyarray(self.sub_control_importance).astype(
            np.float64
        )

        relaxed_sub_control_totals = np.dot(sub_float_weights, sub_incidence)

        # lp_right_hand_side
        lp_right_hand_side = np.round(relaxed_sub_control_totals) - np.dot(
            sub_int_weights, sub_incidence
        )
        lp_right_hand_side = np.maximum(lp_right_hand_side, 0.0)

        # inequality constraint upper bounds
        sub_num_households = relaxed_sub_control_totals[
            :, (total_hh_sub_control_index,)
        ]
        sub_max_control_values = np.amax(sub_incidence, axis=0) * sub_num_households
        relax_ge_upper_bound = np.maximum(
            sub_max_control_values - lp_right_hand_side, 0
        )
        hh_constraint_ge_bound = np.maximum(sub_max_control_values, lp_right_hand_side)

        # equality constraint for the total households control
        total_hh_right_hand_side = lp_right_hand_side[:, total_hh_sub_control_index]

        # - parent
        parent_incidence = self.incidence_df[self.parent_countrol_cols]
        parent_incidence = parent_incidence.values.astype(np.float64)

        # note:
        # sum(sub_int_weights) might be different from parent_float_weights.astype(int)
        # parent_resid_weights might be > 1.0, OK as we are using in objective, not for rounding
        parent_float_weights = np.sum(sub_float_weights, axis=0)
        parent_int_weights = np.sum(sub_int_weights, axis=0)
        parent_resid_weights = np.sum(sub_resid_weights, axis=0)

        # print "parent_float_weights\n", parent_float_weights
        # print "parent_int_weights\n", parent_int_weights
        # print "parent_resid_weights\n", parent_resid_weights

        # - parent control totals based on sub_zone balanced weights
        relaxed_parent_control_totals = np.dot(parent_float_weights, parent_incidence)

        parent_countrol_importance = np.asanyarray(
            self.parent_countrol_importance
        ).astype(np.float64)

        parent_lp_right_hand_side = np.round(relaxed_parent_control_totals) - np.dot(
            parent_int_weights, parent_incidence
        )
        parent_lp_right_hand_side = np.maximum(parent_lp_right_hand_side, 0.0)

        # - create the inequality constraint upper bounds
        parent_num_households = np.sum(sub_num_households)

        parent_max_possible_control_values = (
            np.amax(parent_incidence, axis=0) * parent_num_households
        )
        parent_relax_ge_upper_bound = np.maximum(
            parent_max_possible_control_values - parent_lp_right_hand_side, 0
        )
        parent_hh_constraint_ge_bound = np.maximum(
            parent_max_possible_control_values, parent_lp_right_hand_side
        )

        # how could this not be the case?
        if not (
            parent_hh_constraint_ge_bound == parent_max_possible_control_values
        ).all():
            print("\nSimulIntegerizer integerizing", self.trace_label)
            logger.warning(
                "parent_hh_constraint_ge_bound != parent_max_possible_control_values"
            )
            logger.warning(
                "parent_hh_constraint_ge_bound:      %s" % parent_hh_constraint_ge_bound
            )
            logger.warning(
                "parent_max_possible_control_values: %s"
                % parent_max_possible_control_values
            )
            print("\n")
            # assert (parent_hh_constraint_ge_bound == parent_max_possible_control_values).all()

        # fork: seeded order-sampling keys replace ln(sub resid) in the LP
        # objective; with the draw off the call is exactly upstream's
        draw_keys = None
        draw_kwargs = {}
        if self.draw_method != "off":
            draw_keys = draw.draw_keys(
                sub_resid_weights,
                self.draw_method,
                bs_options.call_rng("simul", self.trace_label),
            )
            draw_kwargs["draw_keys"] = draw_keys

        # fork: LP timing for the path record (None = upstream call)
        timing = None
        if bs_options.path_record_enabled() and not self.use_cvxpy:
            timing = {}
            draw_kwargs["timing"] = timing

        resid_weights_out, status_text = self.integerizer_func(
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
            timeout_in_seconds=self.timeout_in_seconds,
            **draw_kwargs,
        )

        path_record.record_call(
            "simul",
            self.trace_label,
            status_text,
            zones=len(self.sub_weights.columns),
            rows=len(self.incidence_df.index),
            timing=timing,
        )

        # smart round resid_weights_out for each sub_zone
        total_household_controls = sub_control_totals[
            :, total_hh_sub_control_index
        ].flatten()
        integerized_weights = np.empty_like(sub_int_weights)

        sub_zone_count = len(self.sub_weights.columns)
        if draw_keys is not None:
            # fork: ties (and, after a failed LP, the positive residuals) are
            # ordered by the same draw keys, row z per sub-zone
            round_values = draw.rounding_values(
                resid_weights_out, sub_resid_weights, status_text
            )
        for i in range(sub_zone_count):
            if draw_keys is None:
                integerized_weights[i] = smart_round(
                    sub_int_weights[i],
                    resid_weights_out[i],
                    total_household_controls[i],
                )
            else:
                integerized_weights[i] = smart_round(
                    sub_int_weights[i],
                    round_values[i],
                    total_household_controls[i],
                    tie_keys=draw_keys[i],
                )

        # integerized_weights df: one column of integerized weights per sub_zone
        self.integerized_weights = pd.DataFrame(
            data=integerized_weights.T,
            columns=self.sub_weights.columns,
            index=self.incidence_df.index,
        )

        return status_text

    def _integerize_controlled(self):
        """Fork: integerize with the controlled rounding (``INTEGERIZER_ROUNDING:
        controlled``).

        The arrays are derived as in :meth:`integerize`; the problem is that
        of :mod:`populationsim.integerizing.controlled_rounding` (targets,
        bands, prices). ``INTEGERIZER_CONTROLLED_METHOD`` chooses how it is
        decided: ``search`` (default) by the network start and swap search
        of :func:`populationsim.integerizing.controlled_search.solve_search`,
        ``lp`` by the LP of
        :func:`populationsim.integerizing.controlled_lp.solve_controlled`
        (solver, retry, repair). The result is then rounded per sub-zone by
        ``smart_round`` exactly as upstream's LP result is. A failed call
        returns the input residuals and the failure status, so the wrapper's
        sequential path runs as after an upstream failure.

        The household keys are those of upstream's objective: the draw keys
        (``INTEGERIZER_DRAW``) or else ``ln`` of the sub-zone residual (with
        upstream's floor ``LOG_OVERFLOW``), plus ``ln`` of the row's summed
        residual (upstream's parent term).
        """
        total_hh_sub_control_index = self.sub_controls_df.columns.get_loc(
            self.total_hh_control_col
        )

        sub_incidence = self.incidence_df[self.sub_controls_df.columns]
        sub_incidence = sub_incidence.values.astype(np.float64)

        # (sub-zones, rows), as in integerize
        sub_float_weights = self.sub_weights.values.transpose().astype(np.float64)
        sub_int_weights = sub_float_weights.astype(int)
        sub_resid_weights = sub_float_weights % 1.0

        sub_control_totals = np.asanyarray(self.sub_controls_df).astype(np.int64)
        sub_control_importance = np.asanyarray(self.sub_control_importance).astype(
            np.float64
        )
        # the integer household control of each sub-zone (what smart_round meets)
        total_household_controls = sub_control_totals[
            :, total_hh_sub_control_index
        ].flatten()

        parent_incidence = self.incidence_df[self.parent_countrol_cols]
        parent_incidence = parent_incidence.values.astype(np.float64)
        parent_countrol_importance = np.asanyarray(
            self.parent_countrol_importance
        ).astype(np.float64)
        parent_resid_weights = np.sum(sub_resid_weights, axis=0)

        floor = np.exp(draw.LOG_OVERFLOW)
        draw_keys = None
        if self.draw_method != "off":
            draw_keys = draw.draw_keys(
                sub_resid_weights,
                self.draw_method,
                bs_options.call_rng("simul", self.trace_label),
            )
            sub_keys = draw_keys
        else:
            sub_keys = np.log(np.maximum(sub_resid_weights, floor))
        keys = sub_keys + np.log(np.maximum(parent_resid_weights, floor))[None, :]

        problem = controlled_rounding.build_problem(
            sub_float_weights,
            sub_incidence,
            parent_incidence,
            sub_control_importance,
            parent_countrol_importance,
            total_household_controls,
            total_hh_sub_control_index,
            keys,
            bs_options.penalty_settings(),
        )
        timing = {}
        if bs_options.controlled_method() == "lp":
            resid_weights_out, status_text, stats = controlled_lp.solve_controlled(
                problem,
                sub_incidence,
                parent_incidence,
                sub_resid_weights,
                total_hh_sub_control_index,
                bs_options.controlled_solver(),
                bs_options.controlled_retry_solver(),
                bs_options.controlled_repair(),
                self.timeout_in_seconds,
                timing=timing,
            )
            stats = dict(stats, method="lp")
            # solve_s of the record: the LP attempts plus the repair and its
            # pricing solve
            if "solve_s" in timing:
                timing["solve_s"] += stats.get("repair_seconds", 0.0) + stats.get(
                    "objective_seconds", 0.0
                )
            _log_controlled_call(self.trace_label, status_text, stats)
        else:
            search = bs_options.search_settings()
            resid_weights_out, status_text, stats = controlled_search.solve_search(
                problem,
                sub_incidence,
                keys,
                sub_resid_weights,
                total_hh_sub_control_index,
                search.max_sweeps,
                search.shortlist,
                timing=timing,
            )
            _log_search_call(self.trace_label, status_text, stats)

        path_record.record_call(
            "simul",
            self.trace_label,
            status_text,
            zones=len(self.sub_weights.columns),
            rows=len(self.incidence_df.index),
            timing=timing,
            extra=stats,
        )

        # smart round resid_weights_out for each sub_zone, as in integerize
        integerized_weights = np.empty_like(sub_int_weights)
        sub_zone_count = len(self.sub_weights.columns)
        if draw_keys is not None:
            round_values = draw.rounding_values(
                resid_weights_out, sub_resid_weights, status_text
            )
        for i in range(sub_zone_count):
            if draw_keys is None:
                integerized_weights[i] = smart_round(
                    sub_int_weights[i],
                    resid_weights_out[i],
                    total_household_controls[i],
                )
            else:
                integerized_weights[i] = smart_round(
                    sub_int_weights[i],
                    round_values[i],
                    total_household_controls[i],
                    tie_keys=draw_keys[i],
                )

        self.integerized_weights = pd.DataFrame(
            data=integerized_weights.T,
            columns=self.sub_weights.columns,
            index=self.incidence_df.index,
        )

        return status_text


def _share(count, total):
    return 100.0 * count / total if total else 0.0


def _log_controlled_call(trace_label, status_text, stats):
    """One INFO line per controlled call, and one WARNING when the call needs
    attention (failed status, a retry, a failed repair, fractional values left
    after the repair, households the household controls cannot reach)."""
    x_count = stats["x_count"]
    logger.info(
        "[simul controlled] %s: status %s, solver %s, attempts %d, x %d, "
        "fractional %d (%.2f%%) before and %d (%.2f%%) after the repair, "
        "slack cell %.6g, parent_sub %.6g, parent %.6g, conserve %.6g, "
        "objective lp %.9g, repaired %.9g",
        trace_label,
        status_text,
        stats["solver"],
        stats["attempts"],
        x_count,
        stats["fractional_x"],
        _share(stats["fractional_x"], x_count),
        stats["fractional_after_repair"],
        _share(stats["fractional_after_repair"], x_count),
        stats["slack_cell"],
        stats["slack_parent_sub"],
        stats["slack_parent"],
        stats["slack_conserve"],
        stats["objective_lp"],
        stats["objective_repaired"],
    )
    reasons = []
    if status_text not in STATUS_SUCCESS:
        reasons.append("status %s (the wrapper's fallback runs)" % status_text)
    if stats["attempts"] > 1:
        reasons.append("%d solver attempts" % stats["attempts"])
    if stats["repair_failed"]:
        reasons.append("repair failed")
    if stats["fractional_after_repair"] > 0:
        reasons.append(
            "%d fractional values after the repair" % stats["fractional_after_repair"]
        )
    if stats["hh_unreachable"] > 0:
        reasons.append("%d households unreachable" % stats["hh_unreachable"])
    if reasons:
        logger.warning("[simul controlled] %s: %s", trace_label, "; ".join(reasons))


def _log_search_call(trace_label, status_text, stats):
    """One INFO line per search call, and one WARNING when the call needs
    attention (failed status, search stopped by ``max_sweeps``, a zone or row
    shortfall of the start, households the household controls cannot
    reach)."""
    logger.info(
        "[simul controlled] %s: method search, status %s, x %d, swaps %d, "
        "sweeps %d, converged %d, distance %.9g -> %.9g, flow %d of %d, "
        "zone shortfall %d, row shortfall %d",
        trace_label,
        status_text,
        stats["x_count"],
        stats["swaps"],
        stats["sweeps"],
        stats["converged"],
        stats["distance_start"],
        stats["distance_final"],
        stats["flow_achieved"],
        stats["flow_wanted"],
        stats["zone_shortfall"],
        stats["row_shortfall"],
    )
    reasons = []
    if status_text not in STATUS_SUCCESS:
        reasons.append("status %s (the wrapper's fallback runs)" % status_text)
    elif not stats["converged"]:
        reasons.append(
            "not converged after %d sweeps (INTEGERIZER_SEARCH_MAX_SWEEPS)"
            % stats["sweeps"]
        )
    if stats["zone_shortfall"] > 0:
        reasons.append(
            "zone shortfall %d (round-ups smart_round adds outside the search)"
            % stats["zone_shortfall"]
        )
    if stats["row_shortfall"] > 0:
        reasons.append(
            "row shortfall %d (parent weight not placed)" % stats["row_shortfall"]
        )
    if stats["hh_unreachable"] > 0:
        reasons.append("%d households unreachable" % stats["hh_unreachable"])
    if reasons:
        logger.warning("[simul controlled] %s: %s", trace_label, "; ".join(reasons))
