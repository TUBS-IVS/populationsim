# Changes of the TUBS-IVS fork

Base: upstream tag `v0.10.0`. Every new setting defaults to upstream behaviour.

## 0.10.0+bs.2

- Add `INTEGERIZER_ROUNDING` (`round` default, `controlled`): the controlled rounding of the simultaneous integerizer places every household row as often as its parent weight says, gives every zone its household count and minimises the importance-weighted distance of the integer to the float cell totals. `round` is upstream's rounding and equals `0.10.0+bs.1`.
- Add `INTEGERIZER_CONTROLLED_METHOD`: `search` (default; min-cost-flow start that follows the draw keys, then deterministic pairwise swaps, a local optimum with status `FEASIBLE`) or `lp` (linear programme with retry solver and repair of fractional values, kept for comparison).
- Add the settings of the search `INTEGERIZER_SEARCH_MAX_SWEEPS` (200) and `INTEGERIZER_SEARCH_SHORTLIST` (24), the penalties `INTEGERIZER_PENALTY_FLOOR` (100), `INTEGERIZER_PENALTY_CAP` (100000), and for `lp` `INTEGERIZER_PARENT_TOTAL_FACTOR` (10), `INTEGERIZER_CONSERVE_PENALTY` (100000), `INTEGERIZER_CONTROLLED_SOLVER` (`highs-ipm`), `INTEGERIZER_CONTROLLED_RETRY_SOLVER` (`clp-dual`) and `INTEGERIZER_CONTROLLED_REPAIR` (true). Controlled rounding needs the OR-Tools integerizer.
- The path record gains the columns `rounding`, `method` and the stats of each method in controlled mode; in `round` mode its header is that of `0.10.0+bs.1`.
- Add `STORE_POSITIVE_WEIGHT_ROWS_ONLY`: the weight tables of the sub-geographies keep only the positive rows (less memory); the synthetic population is unchanged.
- Add `FAST_PATHS`: lookup-table kernels for the numba balancers, a sparse build of the single-integerizer OR-Tools model and a faster household expansion; the synthetic population is unchanged and every fallback of the balancer kernels is counted and logged (one line at process exit).
- Correct the description of `INTEGERIZER_DRAW`: the inclusion statements hold for the single integerizer; the simultaneous integerizer keeps upstream's parent term.
- Add tests: `tests/test_bs_release_equivalence.py` (round mode equals `v0.10.0+bs.1`, the example with every option on), the log and settings isolation helpers `tests/log_isolation.py` and `tests/settings_isolation.py`.
- Declare SciPy as a dependency (the `lp` method imports it) and set the package version to `0.10.0+bs.2`.

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
