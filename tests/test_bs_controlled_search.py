# PopulationSim
# See full license in LICENSE.txt.
"""Controlled rounding by a network start and a swap search (controlled_search)."""

import logging

import numpy as np
import pytest

from populationsim.integerizing import controlled_search as cs
from populationsim.integerizing import controlled_rounding as cr
from populationsim.integerizing.constants import STATUS_FEASIBLE, STATUS_OPTIMAL

IMPROVEMENT_TOLERANCE = 1e-7

SEARCH_STAT_KEYS = {
    "method",
    "x_count",
    "swaps",
    "sweeps",
    "converged",
    "distance_start",
    "distance_final",
    "flow_wanted",
    "flow_achieved",
    "zone_shortfall",
    "row_shortfall",
    "hh_unreachable",
    "hh_outside_lp",
    "hh_excess",
}


def _problem(allowed, counts, targets, goal=None, price=None):
    """A ControlledProblem with the fields the search reads set directly.

    ``goal`` (Z, C) is the float cell total in units of x, split into
    ``band_lo`` and ``frac``; ``price`` (C,) the cell penalty. Column 0 is the
    household total. The LP-only fields are zeros.
    """
    allowed = np.asarray(allowed, dtype=bool)
    zones, rows = allowed.shape
    if goal is None:
        goal = np.zeros((zones, 1))
    goal = np.asarray(goal, dtype=np.float64)
    controls = goal.shape[1]
    if price is None:
        price = np.full(controls, 100.0)
    band_lo = np.floor(goal)
    frac = goal - band_lo
    zeros = np.zeros((zones, controls))
    return cr.ControlledProblem(
        x_allowed=allowed,
        hh_count=np.asarray(counts, dtype=np.int64),
        band_lo=band_lo,
        band_hi=np.ceil(goal),
        frac=frac,
        under_price=zeros,
        over_price=zeros,
        cell_penalty=np.asarray(price, dtype=np.float64),
        parent_sub_target=np.zeros(controls),
        parent_sub_penalty=np.zeros(controls),
        parent_target=np.zeros(0),
        parent_penalty=np.zeros(0),
        conserve_target=np.asarray(targets, dtype=np.float64),
        conserve_penalty=100000.0,
        x_coef=np.zeros((zones, rows)),
        hh_unreachable=0,
        hh_outside_lp=0,
        hh_excess=0,
    )


def _objective(x, incidence, goal, price, total_index=0):
    """The search objective, computed independently of the kernel."""
    price = np.array(price, dtype=np.float64)
    price[total_index] = 0.0
    totals = np.asarray(x, dtype=np.float64) @ incidence
    return float((np.abs(totals - goal) * price[np.newaxis, :]).sum())


def _random_problem(seed):
    """A random consistent problem: (problem, incidence, keys, residuals).

    2 to 12 zones, 5 to 150 rows, 2 to 8 controls (column 0 the household
    total, the others with integer incidence 0..3), some cells not allowed,
    some rows with target 2. The counts are the zone sums of a random table
    that meets the targets, so a table with these margins exists.
    """
    rng = np.random.RandomState(seed)
    zones = rng.randint(2, 13)
    rows = rng.randint(5, 151)
    controls = rng.randint(2, 9)
    allowed = rng.uniform(size=(zones, rows)) < rng.uniform(0.4, 0.9)
    table = np.zeros((zones, rows), dtype=np.int64)
    targets = np.zeros(rows)
    for h in range(rows):
        candidates = np.flatnonzero(allowed[:, h])
        if candidates.size == 0:
            continue
        want = 2 if (candidates.size >= 2 and rng.uniform() < 0.2) else 1
        chosen = rng.choice(candidates, size=want, replace=False)
        table[chosen, h] = 1
        targets[h] = want
    counts = table.sum(axis=1)
    incidence = np.zeros((rows, controls))
    incidence[:, 0] = 1.0
    incidence[:, 1:] = rng.randint(0, 4, size=(rows, controls - 1))
    incidence[:, 1:] *= rng.uniform(size=(rows, controls - 1)) < 0.5
    goal = np.zeros((zones, controls))
    goal[:, 0] = counts
    expected = table @ incidence
    goal[:, 1:] = np.maximum(
        expected[:, 1:] + rng.uniform(-2.0, 2.0, size=(zones, controls - 1)), 0.0
    )
    price = rng.choice([100.0, 300.0, 1000.0, 5000.0], size=controls)
    problem = _problem(allowed, counts, targets, goal, price)
    keys = np.where(allowed, rng.gumbel(size=(zones, rows)), -np.inf)
    residuals = np.where(allowed, rng.uniform(0.05, 0.95, size=(zones, rows)), 0.0)
    return problem, incidence, keys, residuals


def _improving_swaps(x, allowed, incidence, goal, price, total_index=0):
    """Brute force over ALL legal swaps: (count of swaps lowering the
    objective by more than the tolerance, largest such improvement)."""
    price = np.array(price, dtype=np.float64)
    price[total_index] = 0.0
    totals = np.asarray(x, dtype=np.float64) @ incidence
    zones = x.shape[0]
    count = 0
    largest = 0.0
    for a in range(zones):
        for b in range(a + 1, zones):
            movers_a = np.flatnonzero((x[a] == 1) & (x[b] == 0) & allowed[b])
            movers_b = np.flatnonzero((x[b] == 1) & (x[a] == 0) & allowed[a])
            if movers_a.size == 0 or movers_b.size == 0:
                continue
            # d[i, j, c]: change of zone a's total when u = movers_a[i] leaves
            # and v = movers_b[j] enters
            d = incidence[movers_b][np.newaxis, :, :] - incidence[movers_a][:, None, :]
            ya, ga = totals[a], goal[a]
            yb, gb = totals[b], goal[b]
            delta = (
                price
                * (
                    np.abs(ya + d - ga)
                    - np.abs(ya - ga)
                    + np.abs(yb - d - gb)
                    - np.abs(yb - gb)
                )
            ).sum(axis=2)
            improving = delta < -IMPROVEMENT_TOLERANCE
            count += int(improving.sum())
            if improving.any():
                largest = max(largest, float(-delta.min()))
    return count, largest


def _search(problem, incidence, keys, residuals, max_sweeps=10000, shortlist=24):
    return cs.solve_search(
        problem, incidence, keys, residuals, 0, max_sweeps, shortlist
    )


# --- the network start --------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_start_meets_zone_counts_and_row_targets(seed):
    problem, incidence, keys, _ = _random_problem(seed)
    x, flow_stats = cs.network_start(problem, keys)
    assert x.shape == problem.x_allowed.shape
    assert set(np.unique(x)) <= {0, 1}
    assert not (x.astype(bool) & ~problem.x_allowed).any()
    np.testing.assert_array_equal(x.sum(axis=1), problem.hh_count)
    np.testing.assert_array_equal(x.sum(axis=0), problem.conserve_target)
    assert flow_stats["flow_wanted"] == int(problem.hh_count.sum())
    assert flow_stats["flow_achieved"] == flow_stats["flow_wanted"]
    assert flow_stats["zone_shortfall"] == 0
    assert flow_stats["row_shortfall"] == 0


@pytest.mark.parametrize(
    "keys, expected",
    [
        ([[5.0, 0.0], [0.0, 1.0]], [[1, 0], [0, 1]]),
        ([[0.0, 2.0], [4.0, 0.0]], [[0, 1], [1, 0]]),
        # unclipped the diagonal wins (80 against 45); clipped to KEY_CLIP the
        # 100 counts as 30 and the other diagonal wins (10 against 45)
        ([[100.0, 25.0], [20.0, -20.0]], [[0, 1], [1, 0]]),
    ],
)
def test_start_takes_the_table_with_the_larger_key_sum(keys, expected):
    assert cr.KEY_CLIP == 30.0
    problem = _problem(np.ones((2, 2), dtype=bool), [1, 1], [1, 1])
    x, _ = cs.network_start(problem, np.asarray(keys))
    np.testing.assert_array_equal(x, expected)


def test_start_counts_a_zone_count_above_the_available_rows():
    problem = _problem(np.ones((2, 2), dtype=bool), [3, 1], [1, 1])
    x, flow_stats = cs.network_start(problem, np.zeros((2, 2)))
    assert flow_stats["flow_wanted"] == 2
    assert flow_stats["flow_achieved"] == 2
    assert flow_stats["zone_shortfall"] == 2
    assert flow_stats["row_shortfall"] == 0
    np.testing.assert_array_equal(x.sum(axis=0), [1, 1])


def test_start_counts_a_row_target_above_its_allowed_zones():
    allowed = [[True, True], [False, True]]
    problem = _problem(allowed, [1, 1], [2, 0])
    x, flow_stats = cs.network_start(problem, np.zeros((2, 2)))
    assert flow_stats["flow_wanted"] == 2
    assert flow_stats["flow_achieved"] == 1
    assert flow_stats["zone_shortfall"] == 1
    assert flow_stats["row_shortfall"] == 1
    np.testing.assert_array_equal(x, [[1, 0], [0, 0]])


def test_empty_problem_returns_zeros_and_optimal():
    problem = _problem(np.zeros((2, 3), dtype=bool), [0, 0], [0, 0, 0])
    x, flow_stats = cs.network_start(problem, np.zeros((2, 3)))
    np.testing.assert_array_equal(x, np.zeros((2, 3)))
    assert flow_stats["flow_achieved"] == 0
    residuals = np.zeros((2, 3))
    out, status, stats = _search(problem, np.ones((3, 1)), np.zeros((2, 3)), residuals)
    assert status == STATUS_OPTIMAL
    np.testing.assert_array_equal(out, np.zeros((2, 3)))
    assert set(stats) == SEARCH_STAT_KEYS
    assert stats["x_count"] == 0 and stats["swaps"] == 0


def test_start_is_deterministic():
    problem, _, keys, _ = _random_problem(7)
    first, first_stats = cs.network_start(problem, keys)
    second, second_stats = cs.network_start(problem, keys)
    np.testing.assert_array_equal(first, second)
    assert first_stats == second_stats


def test_failed_flow_returns_the_residuals(monkeypatch):
    problem, incidence, keys, residuals = _random_problem(3)

    def failing(*args, **kwargs):
        return "BAD_COST_RANGE", None

    monkeypatch.setattr(cs, "_min_cost_flow", failing)
    out, status, stats = _search(problem, incidence, keys, residuals)
    assert status == "ABNORMAL"
    assert out is residuals
    assert set(stats) == SEARCH_STAT_KEYS
    assert stats["swaps"] == 0 and stats["flow_achieved"] == 0


# --- the swap search ----------------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_search_on_random_problems(seed):
    problem, incidence, keys, residuals = _random_problem(seed)
    goal = problem.band_lo + problem.frac
    price = problem.cell_penalty
    start, _ = cs.network_start(problem, keys)
    distance_start = _objective(start, incidence, goal, price)
    largest_zone = int(problem.hh_count.max())

    out, status, stats = _search(
        problem, incidence, keys, residuals, shortlist=max(largest_zone, 1)
    )
    assert status == STATUS_FEASIBLE
    assert set(stats) == SEARCH_STAT_KEYS
    assert stats["method"] == "search"
    assert stats["converged"] == 1
    assert out.dtype == np.float64
    assert set(np.unique(out)) <= {0.0, 1.0}
    assert not (out.astype(bool) & ~problem.x_allowed).any()
    np.testing.assert_array_equal(out.sum(axis=1), problem.hh_count)
    np.testing.assert_array_equal(out.sum(axis=0), problem.conserve_target)
    assert stats["distance_start"] == pytest.approx(distance_start, abs=1e-6)
    distance_final = _objective(out, incidence, goal, price)
    assert stats["distance_final"] == pytest.approx(distance_final, abs=1e-6)
    assert distance_final <= distance_start + 1e-9
    assert stats["x_count"] == int(problem.x_allowed.sum())
    # local optimum over ALL legal swaps (the shortlist holds every member)
    count, largest = _improving_swaps(
        out.astype(np.int64), problem.x_allowed, incidence, goal, price
    )
    assert count == 0, f"{count} improving swaps left, largest {largest}"

    again, again_status, again_stats = _search(
        problem, incidence, keys, residuals, shortlist=max(largest_zone, 1)
    )
    np.testing.assert_array_equal(out, again)
    assert again_stats == stats


@pytest.mark.parametrize("seed", range(5))
def test_default_shortlist_keeps_margins_and_never_worsens(seed):
    problem, incidence, keys, residuals = _random_problem(seed)
    out, status, stats = _search(problem, incidence, keys, residuals)
    assert status == STATUS_FEASIBLE
    np.testing.assert_array_equal(out.sum(axis=1), problem.hh_count)
    np.testing.assert_array_equal(out.sum(axis=0), problem.conserve_target)
    assert stats["distance_final"] <= stats["distance_start"] + 1e-9


def _two_sweep_problem():
    """A problem the search needs more than one sweep for (seed searched)."""
    for seed in range(200):
        case = _random_problem(seed)
        _, _, stats = _search(*case)
        if stats["sweeps"] > 2 and stats["converged"] == 1:
            return case
    raise AssertionError("no random problem needs more than two sweeps")


def test_one_sweep_on_a_problem_that_needs_more_is_not_converged():
    problem, incidence, keys, residuals = _two_sweep_problem()
    out, status, stats = _search(problem, incidence, keys, residuals, max_sweeps=1)
    assert status == STATUS_FEASIBLE
    assert stats["sweeps"] == 1
    assert stats["converged"] == 0
    assert stats["swaps"] > 0
    np.testing.assert_array_equal(out.sum(axis=1), problem.hh_count)
    np.testing.assert_array_equal(out.sum(axis=0), problem.conserve_target)


def test_one_swap_reaches_distance_zero():
    # rows: 0 carries c1, 1 does not. The keys put row 0 into zone 1 and row 1
    # into zone 0; the goal wants c1 once in zone 0 and never in zone 1.
    incidence = np.array([[1.0, 1.0], [1.0, 0.0]])
    goal = np.array([[1.0, 1.0], [1.0, 0.0]])
    problem = _problem(np.ones((2, 2), dtype=bool), [1, 1], [1, 1], goal, [1e9, 500.0])
    keys = np.array([[0.0, 10.0], [10.0, 0.0]])
    start, _ = cs.network_start(problem, keys)
    np.testing.assert_array_equal(start, [[0, 1], [1, 0]])
    out, status, stats = _search(problem, incidence, keys, np.full((2, 2), 0.5))
    assert status == STATUS_FEASIBLE
    np.testing.assert_array_equal(out, [[1.0, 0.0], [0.0, 1.0]])
    assert stats["swaps"] == 1
    assert stats["distance_start"] == pytest.approx(1000.0)
    assert stats["distance_final"] == 0.0
    assert stats["converged"] == 1


def test_household_total_column_is_not_priced():
    # column 1 is the household total here: its goal is far off, yet no swap
    # can change it and it must not enter the distance
    incidence = np.array([[0.0, 1.0], [2.0, 1.0]])
    goal = np.array([[0.0, 50.0], [2.0, 50.0]])
    problem = _problem(np.ones((2, 2), dtype=bool), [1, 1], [1, 1], goal, [500.0, 1e9])
    out, status, stats = cs.solve_search(
        problem,
        incidence,
        np.array([[0.0, 10.0], [10.0, 0.0]]),
        np.ones((2, 2)) * 0.5,
        1,
        50,
        24,
    )
    np.testing.assert_array_equal(out, [[1.0, 0.0], [0.0, 1.0]])
    assert stats["distance_start"] == pytest.approx(2000.0)
    assert stats["distance_final"] == 0.0


@pytest.mark.parametrize(
    "change, name",
    [
        (dict(max_sweeps=0), "max_sweeps"),
        (dict(max_sweeps=1.5), "max_sweeps"),
        (dict(shortlist=0), "shortlist"),
        (dict(shortlist=True), "shortlist"),
        (dict(total=5), "total_hh_sub_control_index"),
        (dict(incidence=np.ones((3, 2))), "sub_incidence"),
        (dict(keys=np.zeros((3, 3))), "keys"),
        (dict(residuals=np.zeros((1, 2))), "sub_resid_weights"),
    ],
)
def test_solve_search_rejects_invalid_arguments(change, name):
    problem = _problem(np.ones((2, 2), dtype=bool), [1, 1], [1, 1], np.zeros((2, 2)))
    arguments = dict(
        incidence=np.ones((2, 2)),
        keys=np.zeros((2, 2)),
        residuals=np.full((2, 2), 0.5),
        total=0,
        max_sweeps=10,
        shortlist=4,
    )
    arguments.update(change)
    with pytest.raises(ValueError, match=name):
        cs.solve_search(
            problem,
            arguments["incidence"],
            arguments["keys"],
            arguments["residuals"],
            arguments["total"],
            arguments["max_sweeps"],
            arguments["shortlist"],
        )


# --- the log lines of a search call (simul_integerizer) -----------------------


def _stats(**changes):
    stats = {
        "method": "search",
        "x_count": 10,
        "swaps": 3,
        "sweeps": 2,
        "converged": 1,
        "distance_start": 500.0,
        "distance_final": 200.0,
        "flow_wanted": 4,
        "flow_achieved": 4,
        "zone_shortfall": 0,
        "row_shortfall": 0,
        "hh_unreachable": 0,
        "hh_outside_lp": 0,
        "hh_excess": 0,
    }
    stats.update(changes)
    return stats


@pytest.mark.parametrize(
    "status, changes, reason",
    [
        (STATUS_FEASIBLE, {}, None),
        (STATUS_FEASIBLE, dict(converged=0), "not converged"),
        (STATUS_FEASIBLE, dict(zone_shortfall=2), "zone shortfall 2"),
        (STATUS_FEASIBLE, dict(row_shortfall=1), "row shortfall 1"),
        (STATUS_FEASIBLE, dict(hh_unreachable=3), "3 households unreachable"),
        (STATUS_FEASIBLE, dict(hh_outside_lp=2), "2 households placed by smart_round"),
        (STATUS_FEASIBLE, dict(hh_excess=4), "4 households: the integer parts"),
        ("ABNORMAL", {}, "status ABNORMAL"),
    ],
)
def test_search_call_log_lines(status, changes, reason, caplog, fork_logging):
    from populationsim.integerizing import simul_integerizer

    with caplog.at_level(logging.INFO, logger=simul_integerizer.logger.name):
        simul_integerizer._log_search_call("TAZ_7", status, _stats(**changes))
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(infos) == 1
    assert "[simul controlled] TAZ_7" in infos[0] and "method search" in infos[0]
    assert "swaps 3" in infos[0] and "sweeps 2" in infos[0]
    expected_outside = changes.get("hh_outside_lp", 0)
    expected_excess = changes.get("hh_excess", 0)
    assert "hh_outside_lp %d, hh_excess %d" % (expected_outside, expected_excess) in (
        infos[0]
    )
    if reason is None:
        assert warnings == []
    else:
        assert len(warnings) == 1 and reason in warnings[0]
        assert "TAZ_7" in warnings[0]


def _lp_stats(**changes):
    stats = {
        "x_count": 10,
        "solver": "CBC",
        "attempts": 1,
        "fractional_x": 0,
        "fractional_after_repair": 0,
        "slack_cell": 0.0,
        "slack_parent_sub": 0.0,
        "slack_parent": 0.0,
        "slack_conserve": 0.0,
        "objective_lp": 5.0,
        "objective_repaired": 5.0,
        "repair_failed": False,
        "hh_unreachable": 0,
        "hh_outside_lp": 0,
        "hh_excess": 0,
    }
    stats.update(changes)
    return stats


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({}, None),
        (dict(hh_outside_lp=2), "2 households placed by smart_round"),
        (dict(hh_excess=4), "4 households: the integer parts"),
    ],
)
def test_lp_call_log_lines_carry_the_household_count_fields(
    changes, reason, caplog, fork_logging
):
    from populationsim.integerizing import simul_integerizer

    with caplog.at_level(logging.INFO, logger=simul_integerizer.logger.name):
        simul_integerizer._log_controlled_call(
            "TAZ_7", STATUS_OPTIMAL, _lp_stats(**changes)
        )
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(infos) == 1
    assert (
        "hh_outside_lp %d, hh_excess %d"
        % (
            changes.get("hh_outside_lp", 0),
            changes.get("hh_excess", 0),
        )
        in infos[0]
    )
    if reason is None:
        assert warnings == []
    else:
        assert len(warnings) == 1 and reason in warnings[0]


# --- the truncating branch of the shortlist -------------------------------------


def _large_zone_problem(seed):
    """Four zones of 15 members each over 60 rows, every cell allowed, so that
    every zone has far more swap candidates than a shortlist of 2."""
    rng = np.random.RandomState(1000 + seed)
    zones, rows, controls = 4, 60, 4
    allowed = np.ones((zones, rows), dtype=bool)
    targets = np.ones(rows)
    counts = np.full(zones, rows // zones)
    incidence = np.zeros((rows, controls))
    incidence[:, 0] = 1.0
    incidence[:, 1:] = rng.randint(0, 4, size=(rows, controls - 1))
    goal = np.zeros((zones, controls))
    goal[:, 0] = counts
    goal[:, 1:] = rng.uniform(5.0, 30.0, size=(zones, controls - 1))
    price = rng.choice([100.0, 300.0, 1000.0], size=controls)
    problem = _problem(allowed, counts, targets, goal, price)
    keys = rng.gumbel(size=(zones, rows))
    residuals = rng.uniform(0.05, 0.95, size=(zones, rows))
    return problem, incidence, keys, residuals


@pytest.mark.parametrize("seed", range(5))
def test_truncated_shortlist_keeps_margins_and_never_worsens(seed):
    problem, incidence, keys, residuals = _large_zone_problem(seed)
    assert problem.hh_count.min() >= 8
    goal = problem.band_lo + problem.frac
    price = problem.cell_penalty
    start, _ = cs.network_start(problem, keys)
    # a shortlist of 2 against 15 members and 45 targets per zone truncates
    out, status, stats = _search(problem, incidence, keys, residuals, shortlist=2)
    assert status == STATUS_FEASIBLE
    assert stats["swaps"] > 0, "the case must exercise the search"
    np.testing.assert_array_equal(out.sum(axis=1), problem.hh_count)
    np.testing.assert_array_equal(out.sum(axis=0), problem.conserve_target)
    assert set(np.unique(out)) <= {0.0, 1.0}
    assert not (out.astype(bool) & ~problem.x_allowed).any()
    distance_start = _objective(start, incidence, goal, price)
    distance_final = _objective(out, incidence, goal, price)
    assert stats["distance_start"] == pytest.approx(distance_start, abs=1e-6)
    assert stats["distance_final"] == pytest.approx(distance_final, abs=1e-6)
    assert distance_final <= distance_start + 1e-9
    again, _, again_stats = _search(problem, incidence, keys, residuals, shortlist=2)
    np.testing.assert_array_equal(out, again)
    assert again_stats == stats


def test_missing_min_cost_flow_module_names_the_setting(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name.startswith("ortools.graph.python"):
            raise ImportError("No module named 'ortools.graph.python.min_cost_flow'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    with pytest.raises(ValueError) as exc:
        cs._min_cost_flow(
            np.array([0], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([1], dtype=np.int64),
            np.array([0], dtype=np.int64),
            0,
            1,
            1,
        )
    message = str(exc.value)
    assert "INTEGERIZER_CONTROLLED_METHOD: search" in message
    assert "ortools.graph.python.min_cost_flow" in message
