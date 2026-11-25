PopulationSim
=============

[![Build Status](https://travis-ci.org/activitysim/populationsim.svg?branch=master)](https://travis-ci.org/ActivitySim/populationsim) [![Coverage Status](https://coveralls.io/repos/ActivitySim/populationsim/badge.png?branch=master)](https://coveralls.io/r/ActivitySim/populationsim?branch=master)<a href="https://medium.com/zephyrfoundation/populationsim-the-synthetic-commons-670e17383048"><img src="https://github.com/ZephyrTransport/zephyr-website/blob/gh-pages/img/badging/project_pages/populationsim/PopulationSim.png" width="72.6" height="19.8"></a>


## TUBS-IVS fork

This is a fork of upstream PopulationSim, based on tag `v0.10.0` (branch `bs/main`, package
version `0.10.0+bs.2`). It is **not an upstream release**. It adds opt-in settings:

- `KEEP_FRACTIONAL_SAMPLE_WEIGHTS` (bool, default `false`): keep the summed survey weight of a seed group fractional instead of the int64 cast.
- `INTEGERIZER_DRAW` (`off` default, `pareto` or `gumbel`): seeded order sampling with random keys that replace `ln(residual)` in the objective of both OR-Tools integerizers and in the rounding after them; `off` keeps the upstream objective; not available with `USE_CVXPY: true`. The inclusion statements in `populationsim/integerizing/draw.py` hold for the single (seed-level) integerizer. The simultaneous integerizer keeps upstream's parent term (the log of the row's residual summed over the sub-zones), so they do not carry over unchanged there. With controlled rounding the draw decides the start table of the search (see below).
- `INTEGERIZER_RANDOM_SEED` (int in [0, 2**32), default `0`): the noise of each call derives from the seed, the call kind and the trace label.
- `INTEGERIZER_TIMEOUT` (positive int seconds, default `60`): the 0.10.0 name `INTEGIZER_TIMEOUT` is accepted as alias (both set and different is an error); non-integer values now raise.
- `INTEGERIZER_PATH_RECORD` (bool, default `false`): write `output/integerizer_paths.csv`, one row per LP call and per parent outcome, appended and without output prefix; start each run with an empty output directory; not with `multiprocess`. Column contract: `populationsim/integerizing/path_record.py`.
- `INTEGERIZER_ROUNDING` (`round` default or `controlled`): how the simultaneous integerizer rounds; see "Controlled rounding" below.
- `STORE_POSITIVE_WEIGHT_ROWS_ONLY` and `FAST_PATHS` (bool, default `false`): run time and memory, same synthetic population; see "Run time and memory" below.

**Neutral mode:** with all defaults the output equals v0.10.0. `tests/test_bs_neutral_equivalence.py`
runs the example with the working tree and with the exported tag `v0.10.0` and compares the results
(CI therefore checks out the full history). `tests/expected/` and `examples/` are unchanged from the tag.

### Controlled rounding (`INTEGERIZER_ROUNDING`)

**What upstream's `round` does.** The simultaneous integerizer keeps the integer part of every balanced
weight and decides with a linear programme which households are rounded up. For every control cell of every
sub-zone the programme gets the target `round(float total)`, and a slack that is bounded by that target. A
group of households with a small weight (a rare group) has a float cell total below 0.5, so its target is 0
and the programme has no reason to place it. Where the targets and the bounded slacks cannot be met together, the programme is
infeasible and the integerizer falls back to sequential integerizing.

**What `controlled` does.** The rounding of one parent zone always has a solution and follows this order:

1. every sub-zone gets the household count that the final `smart_round` enforces in both modes: its integer
   parts plus `clip(household control - integer parts, 0, rows of the parent)`;
2. as far as rule 1 allows, every household row of the parent is placed exactly as often as its parent weight
   says (the weight summed over the sub-zones is kept);
3. among such tables, the importance-weighted distance of the integer cell totals to the float cell totals is
   lowered (the control's importance, clipped to the penalty floor and cap, is the weight): `lp` solves a
   linear relaxation to its optimum and then rounds fractional values jointly, `search` ends in a local
   optimum that depends on its start table;
4. the household keys (the draw keys, or `ln` of the residual) decide the start table of `search` and break
   the ties in `lp`.

Rule 1 wins where rules 1 and 2 cannot both hold (the household controls of the sub-zones do not add up to the
parent's households, or a control asks for more than the rows with a residual can supply). What the rows cannot
place is counted in the path record (`row_shortfall`, `zone_shortfall`, `hh_unreachable`, `hh_outside_lp`,
`hh_excess`), not hidden, and `smart_round` adds the round-ups a zone still needs.

**Two methods** (`INTEGERIZER_CONTROLLED_METHOD`):

- `search` (default): a min-cost flow places as many round-ups as the margins allow, and among the maximal
  flows the one with the largest sum of draw keys (so the start follows the draw); then deterministic pairwise
  swaps between sub-zones lower the cell distance, which keeps every zone count and every row sum. The result
  is a local optimum, status `FEASIBLE`.
- `lp`: the linear programme of the same priorities, with a second solver if the first one fails and a joint
  repair of fractional values. Kept for comparison; slower on large parents.

| Setting | Default | Meaning |
|---|---|---|
| `INTEGERIZER_ROUNDING` | `round` | `round` (upstream) or `controlled`; `controlled` needs the OR-Tools integerizer (`USE_CVXPY: false`) and the simultaneous integerizer (`USE_SIMUL_INTEGERIZER: true`, `NO_INTEGERIZATION_EVER: false`); a run that cannot use it is refused instead of rounding as upstream |
| `INTEGERIZER_CONTROLLED_METHOD` | `search` | `search` or `lp` |
| `INTEGERIZER_SEARCH_MAX_SWEEPS` | `200` | `search`: sweeps over all zone pairs at most (integer, 1 to 1,000,000) |
| `INTEGERIZER_SEARCH_SHORTLIST` | `24` | `search`: swap candidates priced per zone and zone pair (integer, 1 to 100,000) |
| `INTEGERIZER_PENALTY_FLOOR` | `100.0` | lowest price of one unit of cell distance (> 0) |
| `INTEGERIZER_PENALTY_CAP` | `100000.0` | highest price (>= floor) |
| `INTEGERIZER_PARENT_TOTAL_FACTOR` | `10.0` | `lp` only: price of a parent total relative to a cell (>= 1) |
| `INTEGERIZER_CONSERVE_PENALTY` | `100000.0` | `lp` only: price of one unit of parent-weight deviation (>= 0) |
| `INTEGERIZER_CONTROLLED_SOLVER` | `highs-ipm` | `lp` only: `highs-ds`, `highs-ipm`, `clp-dual`, `clp-primal`, `clp-barrier`, `glop-dual`, `glop-primal` or `cbc` |
| `INTEGERIZER_CONTROLLED_RETRY_SOLVER` | `clp-dual` | `lp` only: solver of the second attempt, one of the above or `none`; must differ from the first |
| `INTEGERIZER_CONTROLLED_REPAIR` | `true` | `lp` only: round fractional values jointly |

**Limits.**

- `search` ends in a local optimum, not a proven one. The shortlist ranks single moves, so a swap that only
  pays as a pair may be missed; a shortlist at least as large as the largest zone prices every legal swap.
  `converged` in the path record tells whether the search stopped by itself (1) or by the sweep limit (0).
- The household total of a run can differ from a `round` run of the same inputs. On the simultaneous path both
  modes give a sub-zone the count of rule 1; where upstream's programme is infeasible, `round` takes the
  sequential path, which only sees the rows with a weight in that zone. In `examples/example_test` some
  sub-zones have a household control that the balanced weights cannot reach, so the household total of a
  `controlled` run differs from that of a `round` run;
  `test_controlled_zone_household_counts_follow_smart_round` in `tests/test_bs_release_equivalence.py` asserts
  the household count rule for every sub-zone of that example (it does not assert the totals of a `round` run).
- The result is deterministic for a given seed and OR-Tools version: ties between equal keys are resolved inside
  the min-cost-flow solver. The `lp` method has a time limit per attempt, so its status can depend on the
  machine load (counted in the record). A residual of float noise (for example 4e-16) opens a cell like any
  other, as in upstream's programme.
- Conserving every row's parent weight needs integer parent weights, i.e.
  `SUB_BALANCE_WITH_FLOAT_SEED_WEIGHTS: false`. With float seed weights the integer parent totals can differ
  from the float ones.
- Only the OR-Tools integerizer is supported, not CVXPY. The single (seed-level) integerizer keeps upstream's
  rounding.

**Path record in controlled mode.** With `INTEGERIZER_PATH_RECORD` on, the file gains these columns after the
`round`-mode ones (a column that the method of a call does not produce stays empty):

- `rounding`: `controlled`, or `round` on rows of the single integerizer.
- `method`: `search` or `lp`.
- `solver`: the `lp` solver that produced the result; `attempts`: 1 or 2.
- `x_count`: cells that may be rounded up (positive residual).
- `swaps`, `sweeps`: swaps applied and sweeps made by the search; `converged`: 1 if the last sweep had no
  swap, 0 if the sweep limit stopped the search.
- `distance_start`, `distance_final`: cell distance of the start table and of the result (`search`).
- `flow_wanted`, `flow_achieved`: round-ups the margins ask for and the start placed; `zone_shortfall`,
  `row_shortfall`: round-ups the zones still need and parent weight that could not be placed.
- `fractional_x`: fractional values of the LP; `repaired_x`: values the repair changed;
  `fractional_after_repair`: fractional values returned; `repair_failed`: 1 if the repair failed.
- `slack_cell`, `slack_parent_sub`, `slack_parent`, `slack_conserve`: sum of the absolute net slack of the
  cell, sub-control parent total, parent-control total and conservation rows of the LP.
- `hh_unreachable`: households the zone counts ask for that `smart_round` can never add; `hh_outside_lp`:
  households `smart_round` adds among rows without a residual, outside the choice of the method; `hh_excess`:
  households the integer parts already hold beyond the integer controls.
- `objective_lp`, `objective_repaired`: the LP optimum and the optimum with the repaired values fixed (`lp`).

In controlled mode `build_s` and `solve_s` are the network start and the swap search (`search`), or the LP
assembly and the LP solves with the repair (`lp`).

### Run time and memory (`STORE_POSITIVE_WEIGHT_ROWS_ONLY`, `FAST_PATHS`)

Both default to `false` and neither changes the synthetic population: `expanded_household_ids` is equal row for
row to that of a run with the option off. The tests that show it: `test_example_on_equals_off` in
`tests/test_bs_positive_weight_rows.py` and in `tests/test_bs_fast_paths.py`, and
`test_run_options_do_not_change_round_mode` and `test_example_controlled_with_every_option_on` in
`tests/test_bs_release_equivalence.py`. The example configs leave `USE_NUMBA` off, so the numba kernels are
compared by `test_example_with_numba_on_equals_off` and
`test_example_with_numba_and_both_run_time_options_equals_both_off` in `tests/test_bs_fast_paths.py`, and by the
numba variants of the two release tests.

- `STORE_POSITIVE_WEIGHT_ROWS_ONLY`: the weight tables of the sub-geographies keep only the rows whose balanced
  or integer weight is positive (upstream stores one row per seed household group for every zone, almost all
  with weight 0); the next geography rebuilds the full parent weight vector. A line `[positive weight rows]`
  per table says how many rows were kept. The seed-geography table is never filtered.
- `FAST_PATHS`: faster implementations of the upstream result: the numba balancers look up
  `exp(log_gamma * incidence)` per distinct incidence value, the single integerizer builds its OR-Tools model by
  setting only the non-zero coefficients (OR-Tools only: with `USE_CVXPY: true` it does not apply, which is
  logged once), and the household expansion takes all draws of the step in one batch, from the same random numbers in the
  same order as upstream's per-row choice, building the household lists only for the groups it uses (the used
  group ids are cast to the dtype of the group column first; a cast that changes an id is refused; invalid
  sample weights raise a `ValueError` that names the group, where upstream fails inside `RandomState.choice`),
  and the integerizers' per-zone result frames are built in one pass over whole arrays instead of one small
  `DataFrame` per zone (same frame, index, dtypes and positive-row tally and errors).

What differs from a run with the options off: the stored weight tables (`<geography>_weights`) and the
`trace_<geography>_weights` tables hold only the positive rows, and the `delta` of a fast balancer call may
differ in its last bits (it only decides convergence). `expanded_household_ids` and the `_sparse` tables are
equal; the example tests compare them exactly. The summary tables can differ in the last bits of float sums
(pandas' grouped sum is compensated, so dropping the zero rows can change the last bit); in the example they
are equal.

**The fast kernels and bit-for-bit equality.** The numba balancer kernels are compiled with `fastmath`. That
the fast kernels reproduce the original ones bit for bit is therefore a property that the tests verify for an
environment (numba and LLVM version, CPU), not one that holds by construction. Run
`tests/test_bs_balancer_fast_path.py` in the environment and on the machine type that you use. The options are
not tested with `multiprocess: true` for `FAST_PATHS` (the `STORE_POSITIVE_WEIGHT_ROWS_ONLY` option has a
multiprocess example test).

**Resuming a run.** A run must be resumed with the value of `STORE_POSITIVE_WEIGHT_ROWS_ONLY` it was started
with. A weight table stored with the option on and read with it off is refused by `sub_balancing` with a
`ValueError` that names the setting when a parent table holds fewer rows than the seed table; `summarize` does
not check the setting and can leave NaN for a table it reads in the other mode.

**Counted fallbacks of the fast balancer kernels.** A fast call falls back to the original kernel for a control
with more than 256 distinct incidence values, for a non-finite incidence value, and when numba's SVML vector
maths is active; it is re-run with the original kernel when the convergence measure lies within a relative band
of 1e-6 around a threshold. Each case is counted and warned about (the first few in full, then a summary), and
one line is logged at process exit: `balancer fast path: fast N of M calls (P %), fallback value cap ...,
fallback SVML ..., fallback non-finite ..., re-run after the delta guard ...` (INFO when every call was fast,
WARNING otherwise).

PopulationSim is an open platform for population synthesis.  It emerged
from Oregon DOT's desire to build a shared, open, platform that could be
easily adapted for statewide, regional, and urban transportation planning
needs.  PopulationSim is implemented in the
[ActivitySim](https://github.com/activitysim/activitysim) framework.

## Command-Line Interface

PopulationSim can be run directly from the command line:

```bash
populationsim -c /path/to/configs -d /path/to/data -o /path/to/output
```

See the [examples directory](examples/) for more information on using the command-line interface.

## Documentation

https://activitysim.github.io/populationsim/
