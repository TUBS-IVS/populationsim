# Changes of the TUBS-IVS fork

Base: upstream tag `v0.10.0`. Every new setting defaults to upstream behaviour.

## 0.10.0+bs.3 (unreleased)

- `FAST_PATHS`: the household expansion (`expand_households`, `GROUP_BY_INCIDENCE_SIGNATURE`) now takes all draws in one batch (`draw_household_ids`) from the same random numbers in the same order as upstream's per-row `choice`, so the table and the generator state are unchanged; two failure paths differ: invalid weights (not finite, negative, sum 0) give a `ValueError` naming the group (upstream fails inside `RandomState.choice`), and with float32 weights on a large group upstream raises "probabilities do not sum to 1" while the fast draw returns a population (on and off differ there); it replaces the lists-for-used-groups build of `0.10.0+bs.2`, which is removed.
- `FAST_PATHS`: the simultaneous integerizer's result frame (`reshape_result`) is built in one pass over whole arrays instead of one `DataFrame` per sub-zone (`_reshape_result_fast`); frame, index, dtypes, positive-row tally and `ValueError`s are those of the upstream loop, which stays as the off path.
- Correct: `_diff` columns of the zone summaries were subtracted by position; now by zone id.
- Correct: a zone of the control table that has no weight rows now gets a summary row with result 0 and diff = -control, and a warning with the share of such zones per geography. Before, the step stopped with a `ValueError`; with exactly one zone holding weight rows it silently took every zone's diff from that zone's result and left the other results empty (NaN); with no weight rows at all it left every result empty and added an extra zone `0` with empty controls, which turned every id and control into a float. That extra zone is gone, and the zero results keep the dtype of a summary with weight rows.
- Correct: with float weights (`balanced_weight`, as with `NO_INTEGERIZATION_EVER: true`) and a control table that is not sorted, the zone ids of the summaries were floats (`zone_20.0`, id `20.0`); they are now integers (`zone_20`, `20`), as with a sorted control table.

## 0.10.0+bs.2

- Add `INTEGERIZER_ROUNDING` (`round` default, `controlled`): the controlled rounding of the simultaneous integerizer places every household row as often as its parent weight says, gives every zone its household count (as far as the household counts and the row sums allow, shortfalls are counted) and lowers the importance-weighted distance of the integer to the float cell totals (`lp` solves a linear relaxation to its optimum and then rounds fractional values jointly, `search` ends in a local optimum). `round` is upstream's rounding and equals `0.10.0+bs.1`.
- Add `INTEGERIZER_CONTROLLED_METHOD`: `search` (default; min-cost-flow start that follows the draw keys, then deterministic pairwise swaps, a local optimum with status `FEASIBLE`) or `lp` (linear programme with retry solver and repair of fractional values, kept for comparison).
- Add the settings of the search `INTEGERIZER_SEARCH_MAX_SWEEPS` (200) and `INTEGERIZER_SEARCH_SHORTLIST` (24), the penalties `INTEGERIZER_PENALTY_FLOOR` (100), `INTEGERIZER_PENALTY_CAP` (100000), and for `lp` `INTEGERIZER_PARENT_TOTAL_FACTOR` (10), `INTEGERIZER_CONSERVE_PENALTY` (100000), `INTEGERIZER_CONTROLLED_SOLVER` (`highs-ipm`), `INTEGERIZER_CONTROLLED_RETRY_SOLVER` (`clp-dual`) and `INTEGERIZER_CONTROLLED_REPAIR` (true). Controlled rounding needs the OR-Tools integerizer and the simultaneous integerizer; a run with `USE_SIMUL_INTEGERIZER: false` or `NO_INTEGERIZATION_EVER: true` is refused. The two search settings are bounded (1,000,000 sweeps, shortlist 100,000).
- The path record gains the columns `rounding`, `method` and the stats of each method in controlled mode; in `round` mode its header is that of `0.10.0+bs.1`.
- Add `STORE_POSITIVE_WEIGHT_ROWS_ONLY`: the weight tables of the sub-geographies keep only the positive rows (less memory); the synthetic population is unchanged.
- Add `FAST_PATHS`: lookup-table kernels for the numba balancers, a sparse build of the single-integerizer OR-Tools model and a faster household expansion; the synthetic population is unchanged and every fallback of the balancer kernels is counted and logged (one line at process exit).
- Correct the description of `INTEGERIZER_DRAW`: the inclusion statements hold for the single integerizer; the simultaneous integerizer keeps upstream's parent term.
- Controlled rounding with the draw off: the round-ups a zone still needs after the method are placed by the input residuals (largest first), not by array position, so none lands on a row without a residual while a row with one is left at 0; the log lines and warnings carry `hh_outside_lp` and `hh_excess`.
- Run-time options: the household expansion casts the used group ids to the dtype of the group column before `np.isin` (an object array made it compare element by element, which took most of the step's time) and refuses a cast that changes an id; a numba without `config.USING_SVML` counts as SVML active (the original kernel runs, counted and logged once); `rebuild_parent_weights` raises a `ValueError` instead of an assertion; `sub_balancing` refuses a weight table stored with `STORE_POSITIVE_WEIGHT_ROWS_ONLY` and read without it; `USE_CVXPY` with `FAST_PATHS` logs that the fast model build does not apply.
- Correct the claims about the run-time options: the summary tables can differ in the last bits of float sums with `STORE_POSITIVE_WEIGHT_ROWS_ONLY`, and the bit-for-bit equality of the fast balancer kernels is a tested property of an environment (the kernels use `fastmath`), not a construction.
- Add numba variants of the example comparisons (`USE_NUMBA: true`; the example configs leave it off).
- Add tests: `tests/test_bs_release_equivalence.py` (round mode equals `v0.10.0+bs.1`, the example with every option on), the log and settings isolation helpers `tests/log_isolation.py` and `tests/settings_isolation.py`.
- Declare SciPy as a dependency. Only the `lp` method calls it, but the integerizer imports the module at load, so SciPy must be installed for every run. Set the package version to `0.10.0+bs.2`.

## 0.10.0+bs.1

- Add `KEEP_FRACTIONAL_SAMPLE_WEIGHTS`: keep fractional summed seed-group weights instead of the int64 cast.
- Add `INTEGERIZER_DRAW` (`off`, `pareto`, `gumbel`): seeded weight-proportional order sampling in both OR-Tools integerizers and the rounding after them.
- Add `INTEGERIZER_RANDOM_SEED`: per-call noise from the seed, the call kind and the trace label.
- Add `INTEGERIZER_TIMEOUT` with the alias `INTEGIZER_TIMEOUT`; conflicting values and non-integer values raise.
- Add `INTEGERIZER_PATH_RECORD`: `integerizer_paths.csv` with one row per LP call and per parent outcome.
- Reject the draw with `USE_CVXPY: true` and the path record with `multiprocess: true`.
- Add `tests/test_bs_neutral_equivalence.py`, a differential run against tag `v0.10.0`, and `tests/test_bs_all_on.py`.
- CI also runs on `bs/main` and fetches the full history (the tag is needed by the tests).
- Set the package version to `0.10.0+bs.1`.
