PopulationSim
=============

[![Build Status](https://travis-ci.org/activitysim/populationsim.svg?branch=master)](https://travis-ci.org/ActivitySim/populationsim) [![Coverage Status](https://coveralls.io/repos/ActivitySim/populationsim/badge.png?branch=master)](https://coveralls.io/r/ActivitySim/populationsim?branch=master)<a href="https://medium.com/zephyrfoundation/populationsim-the-synthetic-commons-670e17383048"><img src="https://github.com/ZephyrTransport/zephyr-website/blob/gh-pages/img/badging/project_pages/populationsim/PopulationSim.png" width="72.6" height="19.8"></a>


## TUBS-IVS fork

This is a fork of upstream PopulationSim, based on tag `v0.10.0` (branch `bs/main`, package
version `0.10.0+bs.1`). It is **not an upstream release**. It adds five opt-in settings:

- `KEEP_FRACTIONAL_SAMPLE_WEIGHTS` (bool, default `false`): keep the summed survey weight of a seed group fractional instead of the int64 cast.
- `INTEGERIZER_DRAW` (`off` default, `pareto` or `gumbel`): seeded, weight-proportional order sampling in both OR-Tools integerizers and in the rounding after them; `off` keeps the upstream objective; not available with `USE_CVXPY: true`.
- `INTEGERIZER_RANDOM_SEED` (int in [0, 2**32), default `0`): the noise of each call derives from the seed, the call kind and the trace label.
- `INTEGERIZER_TIMEOUT` (positive int seconds, default `60`): the 0.10.0 name `INTEGIZER_TIMEOUT` is accepted as alias (both set and different is an error); non-integer values now raise.
- `INTEGERIZER_PATH_RECORD` (bool, default `false`): write `output/integerizer_paths.csv`, one row per LP call and per parent outcome, appended and without output prefix; start each run with an empty output directory; not with `multiprocess`. Column contract: `populationsim/integerizing/path_record.py`.

**Neutral mode:** with all defaults the output equals v0.10.0. `tests/test_bs_neutral_equivalence.py`
runs the example with the working tree and with the exported tag `v0.10.0` and compares the results
(CI therefore checks out the full history). `tests/expected/` and `examples/` are unchanged from the tag.

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
