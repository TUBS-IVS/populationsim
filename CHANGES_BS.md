# Changes of the TUBS-IVS fork

Base: upstream tag `v0.10.0`. Every new setting defaults to upstream behaviour.

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
